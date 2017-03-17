"""NDVar operations"""
from collections import defaultdict
from itertools import izip
from math import floor

import mne
import numpy as np
from scipy import linalg, signal

from . import mne_fixes
from . import _colorspaces as cs
from ._data_obj import NDVar, UTS, Ordered


def concatenate(ndvars, dim='time', name=None, tmin=0):
    """Concatenate multiple NDVars

    Parameters
    ----------
    ndvars : NDVar | sequence of NDVar
        NDVars to be concatenated. Can also be a single NDVar with ``case``
        dimension to concatenate the different cases.
    dim : str
        Dimension along which to concatenate (only 'time' and 'case' are
        implemented).
    name : str (optional)
        Name the NDVar holding the result.
    tmin : scalar
        Time axis start, only applies when ``dim == 'time'``; default is 0.

    Returns
    -------
    ndvar : NDVar
        NDVar with concatenated data.
    """
    try:
        ndvar = ndvars[0]
    except TypeError:
        ndvars = tuple(ndvars)
        ndvar = ndvars[0]
    axis = ndvar.get_axis(dim)
    dim_names = ndvar.get_dimnames((None,) * axis + (dim,) +
                                   (None,) * (ndvar.ndim - axis - 1))
    x = np.concatenate([v.get_data(dim_names) for v in ndvars], axis)
    if dim == 'time':
        out_dim = UTS(tmin, ndvar.time.tstep, x.shape[axis])
    elif dim == 'case':
        out_dim = 'case'
    else:
        raise NotImplementedError("dim=%s is not implemented; only 'time' and "
                                  "'case' are implemented" % repr(dim))
    dims = ndvar.dims[:axis] + (out_dim,) + ndvar.dims[axis + 1:]
    return NDVar(x, dims, {}, name or ndvar.name)


def convolve(h, x):
    """Convolve ``h`` and ``x`` along the time dimension

    Parameters
    ----------
    h : NDVar | sequence of NDVar
        Kernel.
    x : NDVar | sequence of NDVar
        Data to convolve, corresponding to ``h``.

    Returns
    -------
    y : NDVar
        Convolution, with same time dimension as ``x``.
    """
    if isinstance(h, NDVar):
        if not isinstance(x, NDVar):
            raise TypeError("If h is an NDVar, x also needs to be an NDVar "
                            "(got x=%r)" % (x,))
        elif h.ndim != x.ndim:
            raise ValueError("x and h do not have same number of dimensions")

        if h.ndim == 1:
            ht = h.get_dim('time')
            xt = x.get_dim('time')
        elif h.ndim == 2:
            hdim, ht = h.get_dims((None, 'time'))
            xdim, xt = x.get_dims((None, 'time'))
            if hdim != xdim:
                raise ValueError("h %s dimension and x %s dimension do not "
                                 "match" % (hdim.name, xdim.name))
        else:
            raise ValueError("h and x must be 1 or 2 dimensional, got "
                             "ndim=%i" % h.ndim)

        if ht.tstep != xt.tstep:
            raise ValueError(
                "h and x need to have same time-step (got h.time.tstep=%s, "
                "x.time.tstep=%s)" % (ht.tstep, xt.tstep))

        if h.ndim == 1:
            data = np.convolve(h.x, x.x)
        else:
            data = None
            for h_i, x_i in izip(h.get_data((hdim.name, 'time')),
                                 x.get_data((xdim.name, 'time'))):
                if data is None:
                    data = np.convolve(h_i, x_i)
                else:
                    data += np.convolve(h_i, x_i)

        i_start = -int(round(ht.tmin / ht.tstep))
        i_stop = i_start + xt.nsamples
        if i_start < 0:
            data = np.concatenate((np.zeros(-i_start), data[:i_stop]))
        else:
            data = data[i_start: i_stop]

        return NDVar(data, (xt,), x.info.copy(), x.name)
    else:
        out = None
        for h_, x_ in izip(h, x):
            if out is None:
                out = convolve(h_, x_)
            else:
                out += convolve(h_, x_)
        return out


def cwt_morlet(y, freqs, use_fft=True, n_cycles=3.0, zero_mean=False,
               out='magnitude'):
    """Time frequency decomposition with Morlet wavelets (mne-python)

    Parameters
    ----------
    y : NDVar with time dimension
        Signal.
    freqs : scalar | array
        Frequency/ies of interest. For a scalar, the output will not contain a
        frequency dimension.
    use_fft : bool
        Compute convolution with FFT or temporal convolution.
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    zero_mean : bool
        Make sure the wavelets are zero mean.
    out : 'complex' | 'magnitude' | 'phase'
        Format of the data in the returned NDVar.

    Returns
    -------
    tfr : NDVar
        Time frequency decompositions.
    """
    from mne.time_frequency.tfr import cwt_morlet

    if not y.get_axis('time') == y.ndim - 1:
        raise NotImplementedError
    x = y.x
    x = x.reshape((np.prod(x.shape[:-1]), x.shape[-1]))
    Fs = 1. / y.time.tstep
    if np.isscalar(freqs):
        freqs = [freqs]
        fdim = None
    else:
        fdim = Ordered("frequency", freqs, 'Hz')
        freqs = fdim.values
    x = cwt_morlet(x, Fs, freqs, use_fft, n_cycles, zero_mean)
    if out == 'magnitude':
        x = np.abs(x)
    elif out == 'complex':
        pass
    else:
        raise ValueError("out = %r" % out)

    new_shape = y.x.shape[:-1]
    dims = y.dims[:-1]
    if fdim is not None:
        new_shape += (len(freqs),)
        dims += (fdim,)
    new_shape += y.x.shape[-1:]
    dims += y.dims[-1:]

    x = x.reshape(new_shape)
    info = cs.set_info_cs(y.info, cs.default_info('A'))
    return NDVar(x, dims, info, y.name)


def dss(ndvar):
    """DSS decomposition

    Parameters
    ----------
    ndvar : NDVar (case, dim, time)
        Data to decompose. DSS is performed over 'case' and 'time' dimensions.

    Returns
    -------
    to_dss : NDVar (dss, dim)
        Transform data to DSS.
    from_dss : NDVar (dim, dss)
        Reconstruct data form DSS.

    Notes
    -----
    Uses DSS form mne-sandbox
    """
    dim_names = ndvar.get_dimnames(('case', None, 'time'))
    x = ndvar.get_data(dim_names)
    data_dim = ndvar.get_dim(dim_names[1])
    dss_mat = mne_fixes.dss(x, return_data=False)

    n_comp = len(dss_mat)
    dss_dim = Ordered('dss', np.arange(n_comp))
    to_dss = NDVar(dss_mat, (dss_dim, data_dim), {}, 'to dss')
    from_dss = NDVar(linalg.inv(dss_mat), (data_dim, dss_dim), {}, 'from dss')
    return to_dss, from_dss


def filter_data(ndvar, l_freq, h_freq, filter_length='auto',
                l_trans_bandwidth='auto', h_trans_bandwidth='auto',
                method='fir', iir_params=None, phase='zero',
                fir_window='hamming'):
    """Apply :func:`mne.filter.filter_data` to an NDVar

    Returns
    -------
    filtered_ndvar : NDVar
        NDVar with same dimensions as ``ndvar`` and filtered data.
    """
    axis = ndvar.get_axis('time')
    if axis == ndvar.ndim:
        axis = None
        data = ndvar.x
    else:
        data = ndvar.x.swapaxes(axis, -1)
    sfreq = 1. / ndvar.time.tstep

    x = mne.filter.filter_data(
        data, sfreq, l_freq, h_freq, None, filter_length, l_trans_bandwidth,
        h_trans_bandwidth, 1, method, iir_params, True, phase, fir_window
    )

    if axis is not None:
        x = x.swapaxes(axis, -1)
    return NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)


def neighbor_correlation(x, dim='sensor', obs='time', name=None):
    """Calculate Neighbor correlation

    Parameters
    ----------
    x : NDVar
        The data.
    dim : str
        Dimension over which to correlate neighbors (default 'sensor').
    obs : str
        Dimension which provides observations over which to compute the
        correlation (default 'time').
    name : str
        Name for the new NDVar.

    Returns
    -------
    correlation : NDVar
        NDVar that contains for each element in ``dim`` the with average
        correlation coefficient with its neighbors.
    """
    low_var = x.std(obs).x < 1e-25
    dim_obj = x.get_dim(dim)
    if np.any(low_var):
        raise ValueError("Low variance at %s = %s" %
                         (dim, dim_obj._index_repr(low_var)))


    # find neighbors
    neighbors = defaultdict(list)
    for a, b in dim_obj.connectivity():
        neighbors[a].append(b)
        neighbors[b].append(a)

    # for each point, find the average correlation with its neighbors
    data = x.get_data((dim, obs))
    cc = np.corrcoef(data)
    y = np.empty(len(dim_obj))
    for i in xrange(len(dim_obj)):
        y[i] = np.mean(cc[i, neighbors[i]])

    info = cs.set_info_cs(x.info, cs.stat_info('r'))
    return NDVar(y, (dim_obj,), info, name or x.name)


def resample(ndvar, sfreq, npad=100, window='none'):
    """Resample an NDVar along the 'time' dimension with appropriate filter

    Parameters
    ----------
    ndvar : NDVar
        Ndvar which should be resampled.
    sfreq : scalar
        New sampling frequency.
    npad : int
        Number of samples to use at the beginning and end for padding.
    window : string | tuple
        See :func:`scipy.signal.resample` for description.

    Notes
    -----
    By default (``window='none'``) this function uses
    :func:`scipy.signal.resample` directly. If ``window`` is set to a different
    value, the more sophisticated but slower :func:`mne.filter.resample` is
    used.

    This function can be very slow when the number of time samples is uneven
    (see :func:`scipy.signal.resample`).
    """
    axis = ndvar.get_axis('time')
    if window == 'none':
        new_tstep = 1. / sfreq
        new_num = int(floor((ndvar.time.tstop - ndvar.time.tmin) / new_tstep))
        # crop input data
        new_duration = new_tstep * new_num
        old_num = int(round(new_duration / ndvar.time.tstep))
        if old_num == ndvar.time.nsamples:
            x = ndvar.x
        else:
            idx = (slice(None),) * axis + (slice(None, old_num),)
            x = ndvar.x[idx]
        # resamples
        x = signal.resample(x, new_num, axis=axis)
        dims = (ndvar.dims[:axis] +
                (UTS(ndvar.time.tmin, new_tstep, new_num),) +
                ndvar.dims[axis + 1:])
        return NDVar(x, dims, ndvar.info.copy(), ndvar.name)
    old_sfreq = 1.0 / ndvar.time.tstep
    x = mne.filter.resample(ndvar.x, sfreq, old_sfreq, npad, axis, window)
    new_tstep = 1.0 / sfreq
    time = UTS(ndvar.time.tmin, new_tstep, x.shape[axis])
    dims = ndvar.dims[:axis] + (time,) + ndvar.dims[axis + 1:]
    return NDVar(x, dims, ndvar.info, ndvar.name)


class Filter(object):
    "Filter and downsample"
    def __init__(self, sfreq=None):
        self.sfreq = sfreq

    def __repr__(self):
        args = self._repr_args()
        if self.sfreq:
            args += ', sfreq=%i' % self.sfreq
        return '%s(%s)' % (self.__class__.__name__, args)

    def _repr_args(self):
        raise NotImplementedError

    def _get_b_a(self, tstep):
        raise NotImplementedError

    def __eq__(self, other):
        return isinstance(other, self.__class__) and self.sfreq == other.sfreq

    def filter(self, ndvar):
        """Filter an NDVar"""
        b, a = self._get_b_a(ndvar.time.tstep)
        if not np.all(np.abs(np.roots(a)) < 1):
            raise ValueError("Filter unstable")
        x = signal.lfilter(b, a, ndvar.x, ndvar.get_axis('time'))
        out = NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)
        if self.sfreq:
            return resample(out, self.sfreq)
        else:
            return out

    def filtfilt(self, ndvar):
        """Forward-backward filter an NDVar"""
        b, a = self._get_b_a(ndvar.time.tstep)
        if not np.all(np.abs(np.roots(a)) < 1):
            raise ValueError("Filter unstable")
        x = signal.filtfilt(b, a, ndvar.x, ndvar.get_axis('time'))
        out = NDVar(x, ndvar.dims, ndvar.info.copy(), ndvar.name)
        if self.sfreq:
            return resample(out, self.sfreq)
        else:
            return out


class Butterworth(Filter):
    """Butterworth filter

    Parameters
    ----------
    low : scalar
        Low cutoff frequency.
    high : scalar | None
        High cutoff frequency.
    order : int
        Filter order.
    sfreq : scalar
        Downsample filtered signal to this sampling frequency.

    Notes
    -----
    Uses :func:`scipy.signal.butter`.
    """
    def __init__(self, low, high, order, sfreq=None):
        if not low and not high:
            raise ValueError("Neither low nor high set")
        self.low = low
        self.high = high
        self.order = int(order)
        Filter.__init__(self, sfreq)

    def _repr_args(self):
        return '%s, %s, %i' % (self.low, self.high, self.order)

    def __eq__(self, other):
        return (Filter.__eq__(self, other) and self.low == other.low and
                self.high == other.high and self.order == other.order)

    def _get_b_a(self, tstep):
        nyq = 1. / tstep / 2.
        if self.low and self.high:
            return signal.butter(self.order, (self.low / nyq, self.high / nyq),
                                 'bandpass')
        elif self.low:
            return signal.butter(self.order, self.low / nyq, 'highpass')
        elif self.high:
            return signal.butter(self.order, self.high / nyq, 'lowpass')
        else:
            raise ValueError("Neither low nor high set")


class WindowFilter(Filter):
    _a = np.array([1])

    def __init__(self, length, window='blackman', sfreq=None):
        self.length = length
        self.window = window
        Filter.__init__(self, sfreq)

    def _repr_args(self):
        return "%r, %r" % (self.length, self.window)

    def __eq__(self, other):
        return (Filter.__eq__(self, other) and self.length == other.length and
                self.window == other.window)

    def _get_b_a(self, tstep):
        n = int(round(self.length / tstep))
        window = signal.get_window(self.window, n)
        window /= window.sum()
        return window, self._a


def segment(continuous, times, tstart, tstop, decim=1):
    """Segment a continuous NDVar

    Parameters
    ----------
    continuous : NDVar
        NDVar with a continuous time axis.
    times : sequence of scalar
        Times for which to extract segments.
    tstart : scalar
        Start time for segments.
    tstop : scalar
        Stop time for segments.
    decim : int
        Decimate data after segmenting by factor ``decim`` (the default is
        ``1``, i.e. no decimation).

    Returns
    -------
    segmented_data : NDVar
        NDVar with all data segments corresponding to ``times``, stacked along
        the ``case`` axis.
    """
    if continuous.has_case:
        raise ValueError("Continuous data can't have case dimension")
    axis = continuous.get_axis('time')
    tstep = None if decim == 1 else continuous.time.tstep * decim
    segments = [continuous.sub(time=(t + tstart, t + tstop, tstep)) for
                t in times]

    s0_time = segments[0].time
    dims = (('case',) +
            continuous.dims[:axis] +
            (UTS(tstart, s0_time.tstep, s0_time.nsamples),) +
            continuous.dims[axis + 1:])
    return NDVar(np.array(tuple(s.x for s in segments)), dims,
                 continuous.info.copy(), continuous.name)
