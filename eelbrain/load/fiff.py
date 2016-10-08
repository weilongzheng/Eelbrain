# -*- coding: utf-8 -*-
'''
Tools for loading data from mne's fiff files.

.. autosummary::
   :toctree: generated

   events
   add_epochs
   add_mne_epochs
   epochs
   mne_epochs

Converting mne objects to :class:`NDVar`:

.. autosummary::
   :toctree: generated

   epochs_ndvar
   evoked_ndvar
   stc_ndvar


.. currentmodule:: eelbrain

Managing events with a :class:`Dataset`
---------------------------------------

To load events as :class:`Dataset`::

    >>> ds = load.fiff.events(raw_file_path)

By default, the :class:`Dataset` contains a variable called ``"trigger"``
with trigger values, and a variable called ``"i_start"`` with the indices of
the events::

    >>> print ds[:10]
    trigger   i_start
    -----------------
    2         27977
    3         28345
    1         28771
    4         29219
    2         29652
    3         30025
    1         30450
    4         30839
    2         31240
    3         31665

These events can be modified in ``ds`` (discarding events, modifying
``i_start``) before being used to load data epochs.

Epochs can be loaded as :class:`NDVar` with :func:`load.fiff.epochs`. Epochs
will be loaded based only on the ``"i_start"`` variable, so any modification
to this variable will affect the epochs that are loaded.::

    >>> ds['epochs'] = load.fiff.epochs(ds)

:class:`mne.Epochs` can be loaded with::

    >>> mne_epochs = load.fiff.mne_epochs(ds)

Note that the returned ``mne_epochs`` event does not contain meaningful event
ids, and ``mne_epochs.event_id`` is None.


Using Threshold Rejection
-------------------------

In case threshold rejection is used, the number of the epochs returned by
``load.fiff.epochs(ds, reject=reject_options)`` might not be the same as the
number of events in ``ds`` (whenever epochs are rejected). For those cases,
:func:`load.fiff.add_epochs`` will automatically resize the :class:`Dataset`::

    >>> epoch_ds = load.fiff.add_epochs(ds, -0.1, 0.6, reject=reject_options)

The returned ``epoch_ds`` will contain the epochs as NDVar as ``ds['meg']``.
If no epochs got rejected during loading, the length of ``epoch_ds`` is
identical with the input ``ds``. If epochs were rejected, ``epoch_ds`` is a
shorter copy of the original ``ds``.

:class:`mne.Epochs` can be added to ``ds`` in the same fashion with::

    >>> ds = load.fiff.add_mne_epochs(ds, -0.1, 0.6, reject=reject_options)


Separate events files
---------------------

If events are stored separately form the raw files, they can be loaded in
:func:`load.fiff.events` by supplying the path to the events file as
``events`` parameter::

    >>> ds = load.fiff.events(raw_file_path, events=events_file_path)

'''
from __future__ import division

import fnmatch
from itertools import izip_longest
from logging import getLogger
import os

import numpy as np

import mne
from mne.source_estimate import _BaseSourceEstimate
from mne.io.constants import FIFF
from mne.io import Raw as _mne_Raw
from mne.io import read_raw_kit as _mne_read_raw_kit
try:
    from mne.io.kit.constants import KIT_LAYOUT
except ImportError:  # mne < 0.13
    KIT_LAYOUT = {}

from .. import _colorspaces as _cs
from .._info import BAD_CHANNELS
from .._utils import ui
from .._data_obj import Var, NDVar, Dataset, Sensor, SourceSpace, UTS, \
    _matrix_graph


def mne_raw(path=None, proj=False, **kwargs):
    """
    Returns a mne Raw object with added projections if appropriate.

    Parameters
    ----------
    path : None | str
        path to a raw fiff or sqd file. If no path is supplied, a file can be
        chosen from a file dialog.
    proj : bool | str
        Add projections from a separate file to the Raw object.
        **``False``**: No proj file will be added.
        **``True``**: ``'{raw}*proj.fif'`` will be used.
        ``'{raw}'`` will be replaced with the raw file's path minus '_raw.fif',
        and '*' will be expanded using fnmatch. If multiple files match the
        pattern, a ValueError will be raised.
        **``str``**: A custom path template can be provided, ``'{raw}'`` and
        ``'*'`` will be treated as with ``True``.
    kwargs
        Additional keyword arguments are forwarded to mne Raw initialization.

    """
    if path is None:
        path = ui.ask_file("Pick a raw data file", "Pick a raw data file",
                           [('Functional image file (*.fif)', '*.fif'),
                            ('KIT Raw File (*.sqd,*.con', '*.sqd;*.con')])
        if not path:
            return

    if not os.path.isfile(path):
        raise IOError("%r is not a file" % path)

    if isinstance(path, basestring):
        _, ext = os.path.splitext(path)
        if ext.startswith('.fif'):
            raw = _mne_Raw(path, **kwargs)
        elif ext in ('.sqd', '.con'):
            raw = _mne_read_raw_kit(path, **kwargs)
        else:
            raise ValueError("Unknown extension: %r" % ext)
    else:
        raw = _mne_Raw(path, **kwargs)

    if proj:
        if proj is True:
            proj = '{raw}*proj.fif'

        if '{raw}' in proj:
            raw_file = raw.info['filename']
            raw_root, _ = os.path.splitext(raw_file)
            raw_root = raw_root.rstrip('raw')
            proj = proj.format(raw=raw_root)

        if '*' in proj:
            head, tail = os.path.split(proj)
            names = fnmatch.filter(os.listdir(head), tail)
            if len(names) == 1:
                proj = os.path.join(head, names[0])
            else:
                if len(names) == 0:
                    err = "No file matching %r"
                else:
                    err = "Multiple files matching %r"
                raise ValueError(err % proj)

        # add the projections to the raw file
        proj = mne.read_proj(proj)
        raw.add_proj(proj, remove_existing=True)

    return raw


def events(raw=None, merge=-1, proj=False, name=None, bads=None,
           stim_channel=None, events=None, **kwargs):
    """
    Load events from a raw fiff file.

    Parameters
    ----------
    raw : str(path) | None | mne Raw
        The raw fiff file from which to extract events (if raw and events are
        both ``None``, a file dialog will be displayed to select a raw file).
    merge : int
        Merge steps occurring in neighboring samples. The integer value
        indicates over how many samples events should be merged, and the sign
        indicates in which direction they should be merged (negative means
        towards the earlier event, positive towards the later event).
    proj : bool | str
        Path to the projections file that will be loaded with the raw file.
        ``'{raw}'`` will be expanded to the raw file's path minus extension.
        With ``proj=True``, ``'{raw}_*proj.fif'`` will be used,
        looking for any projection file starting with the raw file's name.
        If multiple files match the pattern, a ValueError will be raised.
    name : str | None
        A name for the Dataset. If ``None``, the raw filename will be used.
    bads : None | list
        Specify additional bad channels in the raw data file (these are added
        to the ones that are already defined in the raw file).
    stim_channel : None | string | list of string
        Name of the stim channel or all the stim channels
        affected by the trigger. If None, the config variables
        'MNE_STIM_CHANNEL', 'MNE_STIM_CHANNEL_1', 'MNE_STIM_CHANNEL_2',
        etc. are read. If these are not found, it will default to
        'STI 014'.
    events : None | str
        If events are stored in a fiff file separate from the Raw object, the
        path to the events file can be supplied here. The events in the Dataset
        will reflect the event sin the events file rather than the raw file.
    others :
        Keyword arguments for loading the raw file.

    Returns
    -------
    events : Dataset
        A Dataset with the following variables:
         - *i_start*: the index of the event in the raw file.
         - *trigger*: the event value.
        The Dataset's info dictionary contains the following values:
         - *raw*: the mne Raw object.

    """
    if (raw is None and events is None) or isinstance(raw, basestring):
        raw = mne_raw(raw, proj=proj, **kwargs)

    if bads is not None and raw is not None :
        raw.info['bads'].extend(bads)

    if name is None and raw is not None:
        raw_path = raw.info['filename']
        if isinstance(raw_path, basestring):
            name = os.path.basename(raw_path)
        else:
            name = None

    if events is None:
        evts = mne.find_stim_steps(raw, merge=merge, stim_channel=stim_channel)
        evts = evts[np.flatnonzero(evts[:, 2])]
    else:
        evts = mne.read_events(events)

    i_start = Var(evts[:, 0], name='i_start')
    trigger = Var(evts[:, 2], name='trigger')
    info = {'raw': raw}
    return Dataset((trigger, i_start), name, info=info)


def _guess_ndvar_data_type(info):
    """Guess which type of data to extract from an mne object.

    Checks for the presence of channels in that order: "mag", "eeg", "grad".
    If none are found, a ValueError is raised.

    Parameters
    ----------
    info : dict
        MNE info dictionary.

    Returns
    -------
    data : str
        Kind of data to extract
    """
    for ch in info['chs']:
        kind = ch['kind']
        if kind == FIFF.FIFFV_MEG_CH:
            if ch['unit'] == FIFF.FIFF_UNIT_T_M:
                return 'grad'
            elif ch['unit'] == FIFF.FIFF_UNIT_T:
                return 'mag'
        elif kind == FIFF.FIFFV_EEG_CH:
            return 'eeg'
    raise ValueError("No MEG or EEG channel found in info.")


def _picks(info, data, exclude):
    if data == 'eeg':
        meg = False
        eeg = True
        eog = False
    elif data == 'eeg&eog':
        meg = False
        eeg = True
        eog = True
    elif data in ['grad', 'mag']:
        meg = data
        eeg = False
        eog = False
    else:
        err = "data=%r (needs to be 'eeg', 'grad' or 'mag')" % data
        raise ValueError(err)
    picks = mne.pick_types(info, meg, eeg, False, eog, ref_meg=False,
                           exclude=exclude)
    return picks


def _ndvar_epochs_reject(data, reject):
    if reject:
        if not np.isscalar(reject):
            err = ("Reject must be scalar (rejection threshold); got %s." %
                   repr(reject))
            raise ValueError(err)
        reject = {data: reject}
    else:
        reject = None
    return reject


def epochs(ds, tmin=-0.1, tmax=0.6, baseline=None, decim=1, mult=1, proj=False,
           data='mag', reject=None, exclude='bads', info=None, name=None,
           raw=None, sensors=None, i_start='i_start'):
    """
    Load epochs as :class:`NDVar`.

    Parameters
    ----------
    ds : Dataset
        Dataset containing a variable which defines epoch cues (i_start).
    tmin, tmax : scalar
        First and last sample to include in the epochs in seconds.
    baseline : tuple(tmin, tmax) | ``None``
        Time interval for baseline correction. Tmin/tmax in seconds, or None to
        use all the data (e.g., ``(None, 0)`` uses all the data from the
        beginning of the epoch up to t=0). ``baseline=None`` for no baseline
        correction (default).
    decim : int
        Downsample the data by this factor when importing. ``1`` means no
        downsampling. Note that this function does not low-pass filter
        the data. The data is downsampled by picking out every
        n-th sample (see `Wikipedia <http://en.wikipedia.org/wiki/Downsampling>`_).
    mult : scalar
        multiply all data by a constant.
    proj : bool
        mne.Epochs kwarg (subtract projections when loading data)
    data : 'eeg' | 'mag' | 'grad'
        The kind of data to load.
    reject : None | scalar
        Threshold for rejecting epochs (peak to peak). Requires a for of
        mne-python which implements the Epochs.model['index'] variable.
    exclude : list of string | str
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    info : None | dict
        Entries for the ndvar's info dict.
    name : str
        name for the new NDVar.
    raw : None | mne Raw
        Raw file providing the data; if ``None``, ``ds.info['raw']`` is used.
    sensors : None | Sensor
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor instance can be supplied through this kwarg.
    i_start : str
        name of the variable containing the index of the events.

    Returns
    -------
    epochs : NDVar
        The epochs as NDVar object.
    """
    if raw is None:
        raw = ds.info['raw']

    picks = _picks(raw.info, data, exclude)
    reject = _ndvar_epochs_reject(data, reject)

    epochs_ = mne_epochs(ds, tmin, tmax, baseline, i_start, raw, decim=decim,
                         picks=picks, reject=reject, proj=proj)
    ndvar = epochs_ndvar(epochs_, name, data, mult=mult, info=info,
                         sensors=sensors)

    if len(epochs_) == 0:
        raise RuntimeError("No events left in %r" % raw.info['filename'])
    return ndvar


def add_epochs(ds, tmin=-0.1, tmax=0.6, baseline=None, decim=1, mult=1,
               proj=False, data='mag', reject=None, exclude='bads', info=None,
               name="meg", raw=None, sensors=None, i_start='i_start',
               sysname=None):
    """
    Load epochs and add them to a dataset as :class:`NDVar`.

    Unless the ``reject`` argument is specified, ``ds``
    is modified in place. With ``reject``, a subset of ``ds`` is returned
    containing only those events for which data was loaded.

    Parameters
    ----------
    ds : Dataset
        Dataset containing a variable which defines epoch cues (i_start) and to
        which the epochs are added.
    tmin, tmax : scalar
        First and last sample to include in the epochs in seconds.
    baseline : tuple(tmin, tmax) | ``None``
        Time interval for baseline correction. Tmin/tmax in seconds, or None to
        use all the data (e.g., ``(None, 0)`` uses all the data from the
        beginning of the epoch up to t=0). ``baseline=None`` for no baseline
        correction (default).
    decim : int
        Downsample the data by this factor when importing. ``1`` means no
        downsampling. Note that this function does not low-pass filter
        the data. The data is downsampled by picking out every
        n-th sample (see `Wikipedia <http://en.wikipedia.org/wiki/Downsampling>`_).
    mult : scalar
        multiply all data by a constant.
    proj : bool
        mne.Epochs kwarg (subtract projections when loading data)
    data : 'eeg' | 'mag' | 'grad'
        The kind of data to load.
    reject : None | scalar
        Threshold for rejecting epochs (peak to peak). Requires a for of
        mne-python which implements the Epochs.model['index'] variable.
    exclude : list of string | str
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    info : None | dict
        Entries for the ndvar's info dict.
    name : str
        name for the new NDVar.
    raw : None | mne Raw
        Raw file providing the data; if ``None``, ``ds.info['raw']`` is used.
    sensors : None | Sensor
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor instance can be supplied through this kwarg.
    i_start : str
        name of the variable containing the index of the events.
    sysname : str
        Name of the sensor system (used to load sensor connectivity).

    Returns
    -------
    ds : Dataset
        Dataset containing the epochs. If no events are rejected, ``ds`` is the
        same object as the input ``ds`` argument, otherwise a copy of it.
    """
    if raw is None:
        raw = ds.info['raw']

    picks = _picks(raw.info, data, exclude)
    reject = _ndvar_epochs_reject(data, reject)

    epochs_ = mne_epochs(ds, tmin, tmax, baseline, i_start, raw, decim=decim,
                         picks=picks, reject=reject, proj=proj, preload=True)
    ds = _trim_ds(ds, epochs_)
    ds[name] = epochs_ndvar(epochs_, name, data, mult=mult, info=info,
                            sensors=sensors, sysname=sysname)
    return ds


def add_mne_epochs(ds, tmin=-0.1, tmax=0.6, baseline=None, target='epochs',
                   **kwargs):
    """
    Load epochs and add them to a dataset as :class:`mne.Epochs`.

    If, after loading, the Epochs contain fewer cases than the Dataset, a copy
    of the Dataset is made containing only those events also contained in the
    Epochs. Note that the Epochs are always loaded with ``preload==True``.

    If the Dataset's info dictionary contains a 'bad_channels' entry, those bad
    channels are added to the epochs.


    Parameters
    ----------
    ds : Dataset
        Dataset with events from a raw fiff file (i.e., created by
        load.fiff.events).
    tmin, tmax : scalar
        First and last sample to include in the epochs in seconds.
    baseline : tuple(tmin, tmax) | ``None``
        Time interval for baseline correction. Tmin/tmax in seconds, or None to
        use all the data (e.g., ``(None, 0)`` uses all the data from the
        beginning of the epoch up to t=0). ``baseline=None`` for no baseline
        correction (default).
    target : str
        Name for the Epochs object in the Dataset.
    *others* :
        Any additional keyword arguments are forwarded to the mne Epochs
        object initialization.
    """
    kwargs['preload'] = True
    epochs_ = mne_epochs(ds, tmin, tmax, baseline, **kwargs)
    ds = _trim_ds(ds, epochs_)
    ds[target] = epochs_
    return ds


def _mne_events(ds=None, i_start='i_start', trigger='trigger'):
    """
    Convert events from a Dataset into mne events.
    """
    if isinstance(i_start, basestring):
        i_start = ds[i_start]

    N = len(i_start)

    if isinstance(trigger, basestring):
        trigger = ds[trigger]
    elif trigger is None:
        trigger = np.ones(N)

    events = np.empty((N, 3), dtype=np.int32)
    events[:, 0] = i_start.x
    events[:, 1] = 0
    events[:, 2] = trigger
    return events


def mne_epochs(ds, tmin=-0.1, tmax=0.6, baseline=None, i_start='i_start',
               raw=None, drop_bad_chs=True, **kwargs):
    """
    Load epochs as :class:`mne.Epochs`.

    Parameters
    ----------
    ds : Dataset
        Dataset containing a variable which defines epoch cues (i_start).
    tmin, tmax : scalar
        First and last sample to include in the epochs in seconds.
    baseline : tuple(tmin, tmax) | ``None``
        Time interval for baseline correction. Tmin/tmax in seconds, or None to
        use all the data (e.g., ``(None, 0)`` uses all the data from the
        beginning of the epoch up to t=0). ``baseline=None`` for no baseline
        correction (default).
    i_start : str
        name of the variable containing the index of the events.
    raw : None | mne Raw
        If None, ds.info['raw'] is used.
    drop_bad_chs : bool
        Drop all channels in raw.info['bads'] form the Epochs. This argument is
        ignored if the picks argument is specified.
    kwargs
        :class:`mne.Epochs` parameters.
    """
    if baseline is False:
        baseline = None

    if raw is None:
        raw = ds.info['raw']

    if drop_bad_chs and ('picks' not in kwargs) and raw.info['bads']:
        kwargs['picks'] = mne.pick_types(raw.info, eeg=True, eog=True, ref_meg=False)

    events = _mne_events(ds=ds, i_start=i_start)
    epochs = mne.Epochs(raw, events, None, tmin, tmax, baseline, **kwargs)
    if kwargs.get('reject', None) is None and len(epochs) != len(events):
        logger = getLogger(__name__)
        logger.warn("%s: MNE generated only %i Epochs for %i events. The raw "
                    "file might end before the end of the last epoch."
                    % (raw.info['filename'], len(epochs), len(events)))

    #  add bad channels from ds
    if BAD_CHANNELS in ds.info:
        invalid = []
        for ch_name in ds.info[BAD_CHANNELS]:
            if ch_name in raw.info['bads']:
                pass
            elif ch_name not in epochs.ch_names:
                invalid.append(ch_name)
            elif ch_name not in epochs.info['bads']:
                epochs.info['bads'].append(ch_name)
        if invalid:
            suffix = 's' * bool(invalid)
            raise ValueError("Invalid channel%s in ds.info[%r]: %s"
                             % (suffix, BAD_CHANNELS, ', '.join(invalid)))

    return epochs


def sensor_dim(fiff, picks=None, sysname=None):
    """
    Create a Sensor dimension object based on the info in a fiff object.

    Parameters
    ----------
    fiff : mne-python object
        Object that has a .info attribute that contains measurement info.
    picks : array of int
        Channel picks (as used in mne-python). By default all MEG and EEG
        channels are included.
    sysname : str
        Name of the sensor system (used to load sensor connectivity).

    Returns
    -------
    sensor_dim : Sensor
        Sensor dimension object.
    """
    info = fiff.info
    if picks is None:
        picks = mne.pick_types(info, eeg=True, ref_meg=False, exclude=())
    else:
        picks = np.asarray(picks, int)

    chs = [info['chs'][i] for i in picks]
    ch_locs = []
    ch_names = []
    for ch in chs:
        x, y, z = ch['loc'][:3]
        ch_name = ch['ch_name']
        ch_locs.append((x, y, z))
        ch_names.append(ch_name)

    # use KIT system ID if available
    sysname = KIT_LAYOUT.get(info.get('kit_system_id'), sysname)

    if sysname is not None:
        c_matrix, names = mne.channels.read_ch_connectivity(sysname)

        # fix channel names
        if sysname.startswith('neuromag'):
            names = [n[:3] + ' ' + n[3:] for n in names]

        # fix channel order
        if names != ch_names:
            index = np.array([names.index(name) for name in ch_names])
            c_matrix = c_matrix[index][:, index]

        conn = _matrix_graph(c_matrix)
    else:
        conn = None

    return Sensor(ch_locs, ch_names, sysname=sysname, connectivity=conn)


def epochs_ndvar(epochs, name=None, data=None, exclude='bads', mult=1,
                 info=None, sensors=None, vmax=None, sysname=None):
    """
    Convert an :class:`mne.Epochs` object to an :class:`NDVar`.

    Parameters
    ----------
    epochs : mne.Epochs | str
        The epochs object or path to an epochs FIFF file.
    name : None | str
        Name for the NDVar.
    data : 'eeg' | 'mag' | 'grad' | None
        The kind of data to include. If None (default) based on ``epochs.info``.
    exclude : list of string | str
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    mult : scalar
        multiply all data by a constant.
    info : None | dict
        Additional contents for the info dictionary of the NDVar.
    sensors : None | Sensor
        The default (``None``) reads the sensor locations from the fiff file.
        If the fiff file contains incorrect sensor locations, a different
        Sensor can be supplied through this kwarg.
    vmax : None | scalar
        Set a default range for plotting.
    sysname : str
        Name of the sensor system (used to load sensor connectivity).
    """
    if isinstance(epochs, basestring):
        epochs = mne.read_epochs(epochs)

    if data is None:
        data = _guess_ndvar_data_type(epochs.info)

    if data == 'eeg' or data == 'eeg&eog':
        info_ = _cs.eeg_info(vmax, mult)
        summary_vmax = 0.1 * vmax if vmax else None
        summary_info = _cs.eeg_info(summary_vmax, mult)
    elif data == 'mag':
        info_ = _cs.meg_info(vmax, mult)
        summary_vmax = 0.1 * vmax if vmax else None
        summary_info = _cs.meg_info(summary_vmax, mult)
    elif data == 'grad':
        info_ = _cs.meg_info(vmax, mult, 'T/cm', u'∆U')
        summary_vmax = 0.1 * vmax if vmax else None
        summary_info = _cs.meg_info(summary_vmax, mult, 'T/cm', u'∆U')
    else:
        raise ValueError("data=%r" % data)
    info_.update(proj='z root', samplingrate=epochs.info['sfreq'],
                 summary_info=summary_info)
    if info:
        info_.update(info)

    x = epochs.get_data()
    picks = _picks(epochs.info, data, exclude)
    if len(picks) < x.shape[1]:
        x = x[:, picks]

    if mult != 1:
        x *= mult

    sensor = sensors or sensor_dim(epochs, picks, sysname)
    time = UTS(epochs.times[0], 1. / epochs.info['sfreq'], len(epochs.times))
    return NDVar(x, ('case', sensor, time), info=info_, name=name)


def evoked_ndvar(evoked, name=None, data=None, exclude='bads', vmax=None,
                 sysname=None):
    """
    Convert one or more mne :class:`Evoked` objects to an :class:`NDVar`.

    Parameters
    ----------
    evoked : str | Evoked | list of Evoked
        The Evoked to convert to NDVar. Can be a string designating a file
        path to a evoked fiff file containing only one evoked.
    name : str
        Name of the NDVar.
    data : 'eeg' | 'mag' | 'grad' | None
        The kind of data to include. If None (default) based on ``epochs.info``.
    exclude : list of string | string
        Channels to exclude (:func:`mne.pick_types` kwarg).
        If 'bads' (default), exclude channels in info['bads'].
        If empty do not exclude any.
    vmax : None | scalar
        Set a default range for plotting.
    sysname : str
        Name of the sensor system (used to load sensor connectivity).

    Notes
    -----
    If evoked objects have different channels, the intersection is used (i.e.,
    only the channels present in all objects are retained).
    """
    if isinstance(evoked, basestring):
        evoked = mne.read_evokeds(evoked)

    if isinstance(evoked, mne.Evoked):
        case_out = False
        evoked = (evoked,)
    if isinstance(evoked, (tuple, list)):
        case_out = True
    else:
        raise TypeError("evoked=%s" % repr(evoked))

    # data type to load
    if data is None:
        data_set = {_guess_ndvar_data_type(e.info) for e in evoked}
        if len(data_set) > 1:
            raise ValueError("Different Evoked objects contain different "
                             "data types: %s" % ', '.join(data_set))
        data = data_set.pop()

    # MEG system
    kit_sys_ids = {e.info.get('kit_system_id') for e in evoked}
    kit_sys_ids -= {None}
    if len(kit_sys_ids) > 1:
        raise ValueError("Evoked objects from different KIT systems can not be "
                         "combined because they have different sensor layouts")
    elif kit_sys_ids:
        sysname = KIT_LAYOUT.get(kit_sys_ids.pop(), sysname)

    if data == 'mag':
        info = _cs.meg_info(vmax)
    elif data == 'eeg':
        info = _cs.eeg_info(vmax)
    elif data == 'grad':
        info = _cs.meg_info(vmax, unit='T/cm')
    else:
        raise ValueError("data=%s" % repr(data))

    e0 = evoked[0]
    if len(evoked) == 1:
        picks = _picks(e0.info, data, exclude)
        x = e0.data[picks]
        if case_out:
            x = x[None, :]
        first, last, sfreq = e0.first, e0.last, round(e0.info['sfreq'], 2)
    else:
        # timing:  round sfreq because precision is lost by FIFF format
        timing_set = {(e.first, e.last, round(e.info['sfreq'], 2)) for e in
                      evoked}
        if len(timing_set) == 1:
            first, last, sfreq = timing_set.pop()
        else:
            raise ValueError("Evoked objects have different timing "
                             "information (first, last, sfreq): " +
                             ', '.join(map(str, timing_set)))

        # find excluded channels
        ch_sets = [set(e.info['ch_names']) for e in evoked]
        all_chs = set.union(*ch_sets)
        common = set.intersection(*ch_sets)
        exclude = set.union(*map(set, (e.info['bads'] for e in evoked)))
        exclude.update(all_chs.difference(common))
        exclude = list(exclude)

        # get data
        x = []
        for e in evoked:
            picks = _picks(e.info, data, exclude)
            x.append(e.data[picks])

    sensor = sensor_dim(e0, picks, sysname)
    time = UTS.from_int(first, last, sfreq)
    if case_out:
        dims = ('case', sensor, time)
    else:
        dims = (sensor, time)
    return NDVar(x, dims, info=info, name=name)


def stc_ndvar(stc, subject, src, subjects_dir=None, method=None, fixed=None,
              name=None, check=True, parc='aparc', connectivity=None):
    """
    Convert one or more :class:`mne.SourceEstimate` objects to an :class:`NDVar`.

    Parameters
    ----------
    stc : SourceEstimate | list of SourceEstimates | str
        The source estimate object(s) or a path to an stc file.
    subject : str
        MRI subject (used for loading MRI in PySurfer plotting)
    src : str
        The kind of source space used (e.g., 'ico-4').
    subjects_dir : None | str
        The path to the subjects_dir (needed to locate the source space
        file).
    method : 'MNE' | 'dSPM' | 'sLORETA'
        Source estimation method (optional, used for generating info).
    fixed : bool
        Source estimation orientation constraint (optional, used for generating
        info).
    name : str | None
        Ndvar name.
    check : bool
        If multiple stcs are provided, check if all stcs have the same times
        and vertices.
    parc : None | str
        Parcellation to add to the source space.
    connectivity : 'link-midline'
        Modify source space connectivity to link medial sources of the two
        hemispheres across the midline.
    """
    subjects_dir = mne.utils.get_subjects_dir(subjects_dir)

    if isinstance(stc, basestring):
        stc = mne.read_source_estimate(stc)

    # construct data array
    if isinstance(stc, _BaseSourceEstimate):
        case = False
        x = stc.data
    else:
        case = True
        stcs = stc
        stc = stcs[0]
        if check:
            times = stc.times
            vertices = stc.vertices
            for stc_ in stcs[1:]:
                assert np.array_equal(stc_.times, times)
                for v1, v0 in izip_longest(stc_.vertices, vertices):
                    assert np.array_equal(v1, v0)
        x = np.array([s.data for s in stcs])

    # Construct NDVar Dimensions
    time = UTS(stc.tmin, stc.tstep, stc.shape[1])
    if isinstance(stc, mne.VolSourceEstimate):
        ss = SourceSpace([stc.vertices], subject, src, subjects_dir, parc)
    else:
        ss = SourceSpace(stc.vertices, subject, src, subjects_dir, parc)
    # Apply connectivity modification
    if isinstance(connectivity, str):
        if connectivity == 'link-midline':
            ss._link_midline()
        elif connectivity != '':
            raise ValueError("connectivity=%s" % repr(connectivity))
    elif connectivity is not None:
        raise TypeError("connectivity=%s" % repr(connectivity))
    # assemble dims
    if case:
        dims = ('case', ss, time)
    else:
        dims = (ss, time)

    # find the right measurement info
    info = {}
    if fixed is False:
        info['meas'] = 'Activation'
        if method == 'MNE' or method == 'dSPM' or method == 'sLORETA':
            info['unit'] = method
        elif method is not None:
            raise ValueError("method=%s" % repr(method))
    elif fixed is True:
        info['meas'] = 'Current Estimate'
        if method == 'MNE':
            info['unit'] = 'Am'
        elif method == 'dSPM' or method == 'sLORETA':
            info['unit'] = '%s(Am)' % method
        elif method is not None:
            raise ValueError("method=%s" % repr(method))
    elif fixed is not None:
        raise ValueError("fixed=%s" % repr(fixed))

    return NDVar(x, dims, info, name)


def _trim_ds(ds, epochs):
    """
    Trim a Dataset to account for rejected epochs. If no epochs were rejected,
    the original ds is rturned.

    Parameters
    ----------
    ds : Dataset
        Dataset that was used to construct epochs.
    epochs : Epochs
        Epochs loaded with mne_epochs()
    """
    if len(epochs) < ds.n_cases:
        ds = ds.sub(epochs.selection)

    return ds
