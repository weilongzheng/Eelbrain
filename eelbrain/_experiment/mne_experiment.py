# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""MneExperiment class to manage data from a experiment

For testing purposed, set up an experiment class without checking for data:

MneExperiment.path_version = 1
MneExperiment.auto_delete_cache = 'disable'
MneExperiment.sessions = ('session',)
e = MneExperiment('.', find_subjects=False)

"""
from __future__ import print_function

from collections import Counter, defaultdict, Sequence
from datetime import datetime
from glob import glob
import inspect
from itertools import chain, izip, product
import logging
import os
from os.path import exists, getmtime, isdir, join, relpath
import re
import shutil
import time

import numpy as np

import mne
from mne.baseline import rescale
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)
from tqdm import tqdm

from .. import _report
from .. import gui
from .. import load
from .. import plot
from .. import save
from .. import table
from .. import testnd
from .._data_obj import (
    Datalist, Dataset, Factor, Var,
    align, align1, all_equal, as_legal_dataset_key,
    asfactor, assert_is_legal_dataset_key, combine)
from .._exceptions import DefinitionError, DimensionMismatchError, OldVersionError
from .._info import BAD_CHANNELS
from .._io.fiff import KIT_NEIGHBORS
from .._io.pickle import update_subjects_dir
from .._names import INTERPOLATE_CHANNELS
from .._meeg import new_rejection_ds
from .._mne import (
    dissolve_label, labels_from_mni_coords, rename_label, combination_label,
    morph_source_space, shift_mne_epoch_trigger)
from ..mne_fixes import (
    write_labels_to_annot, _interpolate_bads_eeg, _interpolate_bads_meg)
from ..mne_fixes._trans import hsp_equal, mrk_equal
from .._ndvar import cwt_morlet
from ..fmtxt import List, Report, Image, read_meta
from .._report import named_list, enumeration, plural
from .._resources import predefined_connectivity
from .._stats.stats import ttest_t
from .._stats.testnd import _MergedTemporalClusterDist
from .._utils import WrappedFormater, ask, subp, keydefaultdict, log_level
from .._utils.mne_utils import fix_annot_names, is_fake_mri
from .definitions import (
    assert_dict_has_args, find_dependent_epochs,
    find_epochs_vars, find_test_vars, log_dict_change, log_list_change)
from .epochs import PrimaryEpoch, SecondaryEpoch, SuperEpoch
from .experiment import FileTree
from .parc import (
    FS_PARC, FSA_PARC, PARC_CLASSES, SEEDED_PARC_RE,
    Parcellation, CombinationParcellation, EelbrainParcellation,
    FreeSurferParcellation, FSAverageParcellation, SeededParcellation)
from .preprocessing import (
    assemble_pipeline, RawICA, pipeline_dict, compare_pipelines,
    ask_to_delete_ica_files)
from .test_def import EvokedTest, TwoStageTest, assemble_tests


# current cache state version
CACHE_STATE_VERSION = 7

# Allowable parameters
ICA_REJ_PARAMS = {'kind', 'source', 'epoch', 'interpolation', 'n_components',
                  'random_state', 'method'}
COV_PARAMS = {'epoch', 'session', 'method', 'reg', 'keep_sample_mean',
              'reg_eval_win_pad'}


inv_re = re.compile("(free|fixed|loose\.\d+)-"  # orientation constraint
                    "(\d*\.?\d+)-"  # SNR
                    "(MNE|dSPM|sLORETA)"  # method
                    "(?:-((?:0\.)?\d+))?"  # depth weighting
                    "(?:-(pick_normal))?"
                    "$")  # pick normal


def as_vardef_var(v):
    "Coerce ds.eval() output for use as variable"
    if isinstance(v, np.ndarray):
        if v.dtype.kind == 'b':
            return Var(v.astype(int))
        return Var(v)
    return v


# Eelbrain 0.24 raw/preprocessing pipeline
LEGACY_RAW = {
    'raw': {},
    '0-40': {
        'source': 'raw', 'type': 'filter', 'args': (None, 40),
        'kwargs': {'method': 'iir'}},
    '0.1-40': {
        'source': 'raw', 'type': 'filter', 'args': (0.1, 40),
        'kwargs': {'l_trans_bandwidth': 0.08, 'filter_length': '60s'}},
    '0.2-40': {
        'source': 'raw', 'type': 'filter', 'args': (0.2, 40),
        'kwargs': {'l_trans_bandwidth': 0.08, 'filter_length': '60s'}},
    '1-40': {
        'source': 'raw', 'type': 'filter', 'args': (1, 40),
        'kwargs': {'method': 'iir'}},
}


################################################################################
# Exceptions

class FileMissing(Exception):
    "An input file is missing"


################################################################################

def _mask_ndvar(ds, name):
    y = ds[name]
    mask = y.source.parc.startswith('unknown')
    if mask.any():
        ds[name] = y.sub(source=np.invert(mask))


def _time_str(t):
    "String for representing a time value"
    if t is None:
        return ''
    else:
        return '%i' % round(t * 1000)


def _time_window_str(window, delim='-'):
    "String for representing a time window"
    return delim.join(map(_time_str, window))


class DictSet(object):
    """Helper class for list of dicts without duplicates"""
    def __init__(self):
        self._list = []

    def __repr__(self):
        return "DictSet(%s)" % self._list

    def __iter__(self):
        return self._list.__iter__()

    def add(self, item):
        if item not in self._list:
            self._list.append(item)


class CacheDict(dict):

    def __init__(self, func, key_vars, *args):
        super(CacheDict, self).__init__()
        self._func = func
        self._key_vars = key_vars
        self._args = args

    def __getitem__(self, key):
        if key in self:
            return dict.__getitem__(self, key)

        if isinstance(key, basestring):
            out = self._func(*self._args, **{self._key_vars: key})
        else:
            out = self._func(*self._args, **dict(zip(self._key_vars, key)))

        self[key] = out
        return out


temp = {
    'eelbrain-log-file': join('{root}', 'eelbrain {class-name}.log'),
    'old-eelbrain-log-file': join('{root}', '.eelbrain.log'),

    # MEG
    'modality': ('', 'eeg', 'meeg'),
    'reference': ('', 'mastoids'),  # EEG reference
    'equalize_evoked_count': ('', 'eq'),
    # locations
    'meg-sdir': join('{root}', 'meg'),
    'meg-dir': join('{meg-sdir}', '{subject}'),
    'raw-dir': '{meg-dir}',

    # raw input files
    'raw-file': join('{raw-dir}', '{subject}_{session}-raw.fif'),
    'raw-file-bkp': join('{raw-dir}', '{subject}_{session}-raw*.fif'),
    'trans-file': join('{raw-dir}', '{mrisubject}-trans.fif'),
    # log-files (eye-tracker etc.)
    'log-dir': join('{meg-dir}', 'logs'),
    'log-rnd': '{log-dir}/rand_seq.mat',
    'log-data-file': '{log-dir}/data.txt',
    'log-file': '{log-dir}/log.txt',
    'edf-file': join('{log-dir}', '*.edf'),

    # created input files
    'bads-file': join('{raw-dir}', '{subject}_{session}-bad_channels.txt'),
    'raw-ica-file': join('{raw-dir}', '{subject} {raw}-ica.fif'),
    'rej-dir': join('{meg-dir}', 'epoch selection'),
    'rej-file': join('{rej-dir}', '{session}_{sns_kind}_{epoch}-{rej}.pickled'),
    'ica-file': join('{rej-dir}', '{session} {sns_kind} {rej}-ica.fif'),

    # cache
    'cache-dir': join('{root}', 'eelbrain-cache'),
    'input-state-file': join('{cache-dir}', 'input-state.pickle'),
    'cache-state-file': join('{cache-dir}', 'cache-state.pickle'),
    # raw
    'raw-cache-dir': join('{cache-dir}', 'raw', '{subject}'),
    'raw-cache-base': join('{raw-cache-dir}', '{session} {raw}'),
    'cached-raw-file': '{raw-cache-base}-raw.fif',
    'event-file': '{raw-cache-base}-evts.pickled',
    'interp-file': '{raw-cache-base}-interp.pickled',

    # forward modeling:
    # Two raw files with
    #  - different head shapes: raw files require different head-MRI trans-files
    #    (not implemented).
    #  - Same head shape, but different markers:  same trans-file, but different
    #    forward solutions.
    #  - Same head shape and markers:  raw files could potentially share forward
    #    solution (not implemented)
    'fwd-file': join('{raw-cache-dir}', '{session}-{mrisubject}-{src}-fwd.fif'),
    # sensor covariance
    'cov-dir': join('{cache-dir}', 'cov'),
    'cov-base': join('{cov-dir}', '{subject}', '{sns_kind} {cov}-{rej}'),
    'cov-file': '{cov-base}-cov.fif',
    'cov-info-file': '{cov-base}-info.txt',
    # evoked
    'evoked-dir': join('{cache-dir}', 'evoked'),
    'evoked-base': join('{evoked-dir}', '{subject}',
                        '{session} {sns_kind} {epoch} {model} {evoked_kind}'),
    'evoked-file': join('{evoked-base}-ave.fif'),
    'evoked-old-file': join('{evoked-base}.pickled'),  # removed for 0.25
    # test files
    'test-dir': join('{cache-dir}', 'test'),
    'data_parc': 'unmasked',  # for some tests, parc and mask parameter can be saved in same file
    'test-file': join('{test-dir}', '{analysis} {group}',
                      '{epoch} {test} {test_options} {data_parc}.pickled'),

    # MRIs
    'common_brain': 'fsaverage',
    # MRI base files
    'mri-sdir': join('{root}', 'mri'),
    'mri-dir': join('{mri-sdir}', '{mrisubject}'),
    'bem-dir': join('{mri-dir}', 'bem'),
    'mri-cfg-file': join('{mri-dir}', 'MRI scaling parameters.cfg'),
    'mri-file': join('{mri-dir}', 'mri', 'orig.mgz'),
    'bem-file': join('{bem-dir}', '{mrisubject}-inner_skull-bem.fif'),
    'bem-sol-file': join('{bem-dir}', '{mrisubject}-*-bem-sol.fif'),  # removed for 0.24
    'head-bem-file': join('{bem-dir}', '{mrisubject}-head.fif'),
    'src-file': join('{bem-dir}', '{mrisubject}-{src}-src.fif'),
    'fiducials-file': join('{bem-dir}', '{mrisubject}-fiducials.fif'),
    # Labels
    'hemi': ('lh', 'rh'),
    'label-dir': join('{mri-dir}', 'label'),
    'annot-file': join('{label-dir}', '{hemi}.{parc}.annot'),

    # (method) plots
    'plot-dir': join('{root}', 'plots'),
    'plot-file': join('{plot-dir}', '{analysis}', '{name}.{ext}'),

    # result output files
    # data processing parameters
    #    > group
    #        > kind of test
    #    > single-subject
    #        > kind of test
    #            > subject
    'res-dir': join('{root}', 'results'),
    'res-file': join('{res-dir}', '{analysis}', '{resname}.{ext}'),
    'res-deep-file': join('{res-dir}', '{analysis}', '{folder}', '{resname}.{ext}'),
    'report-file': join('{res-dir}', '{analysis} {group}', '{folder}',
                        '{epoch} {test} {test_options}.html'),
    'group-mov-file': join('{res-dir}', '{analysis} {group}',
                           '{epoch} {test_options} {resname}.mov'),
    'subject-res-dir': join('{res-dir}', '{analysis} subjects'),
    'subject-spm-report': join('{subject-res-dir}', '{test} {epoch} {test_options}',
                               '{subject}.html'),
    'subject-mov-file': join('{subject-res-dir}',
                             '{epoch} {test_options} {resname}', '{subject}.mov'),

    # plots
    # plot corresponding to a report (and using same folder structure)
    'res-plot-root': join('{root}', 'result plots'),
    'res-plot-dir': join('{res-plot-root}', '{analysis} {group}', '{folder}',
                         '{epoch} {test} {test_options}'),

    # besa
    'besa-root': join('{root}', 'besa'),
    'besa-trig': join('{besa-root}', '{subject}',
                      '{subject}_{session}_{epoch}_triggers.txt'),
    'besa-evt': join('{besa-root}', '{subject}',
                     '{subject}_{session}_{epoch}[{rej}].evt'),

    # MRAT
    'mrat_condition': '',
    'mrat-root': join('{root}', 'mrat'),
    'mrat-sns-root': join('{mrat-root}', '{sns_kind}',
                          '{epoch} {model} {evoked_kind}'),
    'mrat-src-root': join('{mrat-root}', '{src_kind}',
                          '{epoch} {model} {evoked_kind}'),
    'mrat-sns-file': join('{mrat-sns-root}', '{mrat_condition}',
                          '{mrat_condition}_{subject}-ave.fif'),
    'mrat_info-file': join('{mrat-root}', '{subject} info.txt'),
    'mrat-src-file': join('{mrat-src-root}', '{mrat_condition}',
                          '{mrat_condition}_{subject}'),
}


class MneExperiment(FileTree):
    """Analyze an MEG experiment (gradiometer only) with MNE

    Parameters
    ----------
    root : str | None
        the root directory for the experiment (usually the directory
        containing the 'meg' and 'mri' directories)
    find_subjects : bool
        Automatically look for subjects in the MEG-directory (default
        True). Set ``find_subjects=False`` to initialize the experiment
        without any files.
    ...
        Initial state parameters.

    Notes
    -----
    .. seealso::
        Guide on using :ref:`experiment-class-guide`.
    """
    path_version = None
    screen_log_level = logging.INFO
    auto_delete_results = False
    auto_delete_cache = True
    # what to do when the experiment class definition changed:
    #   True: delete outdated files
    #   False: raise an error
    #   'disable': ignore it
    #   'debug': prompt with debug options

    # tuple (if the experiment has multiple sessions)
    sessions = None

    # Raw preprocessing pipeline
    _raw = LEGACY_RAW
    raw = {}

    # add this value to all trigger times
    trigger_shift = 0

    # variables for automatic labeling {name: {trigger: label, triggers: label}}
    variables = {}

    # Default values for epoch definitions
    epoch_default = {}

    # named epochs
    epochs = {'epoch': dict(sel="stim=='target'"),
              'cov': dict(base='epoch', tmin=-0.1, tmax=0)}

    # Rejection
    # =========
    # eog_sns: The sensors to plot separately in the rejection GUI. The default
    # is the two MEG sensors closest to the eyes.
    _eog_sns = {None: (),
                'KIT-157': ('MEG 143', 'MEG 151'),
                'KIT-208': ('MEG 087', 'MEG 130'),
                'KIT-UMD-1': ('MEG 042', 'MEG 025'),
                'KIT-UMD-2': ('MEG 042', 'MEG 025'),
                'KIT-UMD-3': ('MEG 042', 'MEG 025'),
                'KIT-BRAINVISION': ('HEOGL', 'HEOGR', 'VEOGb'),
                'neuromag306mag': ('MEG 0121', 'MEG 1411')}
    #
    # artifact_rejection dict:
    #
    # kind : 'manual' | 'make' | 'ica'
    #     How the rejection is derived:
    #     'manual': manually create a rejection file (use the selection GUI
    #     through .make_rej())
    #     'make' a rejection file is created by the user
    # interpolation : bool
    #     enable by-epoch channel interpolation
    #
    # For manual rejection
    # ^^^^^^^^^^^^^^^^^^^^
    # decim : int
    #     Decim factor for the rejection GUI (default is to use epoch setting).
    _artifact_rejection = {'': {'kind': None},
                           'man': {'kind': 'manual',
                                   'interpolation': True,
                                   }}
    artifact_rejection = {}

    exclude = {}  # field_values to exclude (e.g. subjects)

    # groups can be defined as subject lists: {'group': ('member1', 'member2', ...)}
    # or by exclusion: {'group': {'base': 'all', 'exclude': ('member1', 'member2')}}
    groups = {}

    # whether to look for and load eye tracker data when loading raw files
    has_edf = defaultdict(lambda: False)

    # Pattern for subject names. The first group is used to determine what
    # MEG-system the data was recorded from
    _subject_re = '(R|A|Y|AD|QP)(\d{3,})$'
    # MEG-system. If None, the subject pattern is used to guess the system.
    meg_system = None
    _meg_systems = {'R': 'KIT-NY',
                    'A': 'KIT-AD', 'Y': 'KIT-AD', 'AD': 'KIT-AD', 'QP': 'KIT-AD'}

    # kwargs for regularization of the covariance matrix (see .make_cov())
    _covs = {'auto': {'epoch': 'cov', 'method': 'auto'},
             'bestreg': {'epoch': 'cov', 'reg': 'best'},
             'reg': {'epoch': 'cov', 'reg': True},
             'noreg': {'epoch': 'cov', 'reg': None},
             'emptyroom': {'session': 'emptyroom', 'reg': None}}

    # MRI subject names: {subject: mrisubject} mappings
    # selected with e.set(mri=dict_name)
    # default is identity (mrisubject = subject)
    _mri_subjects = {'': keydefaultdict(lambda s: s)}

    # state variables that are always shown in self.__repr__():
    _repr_kwargs = ('subject', 'rej')

    # Where to search for subjects (defined as a template name). If the
    # experiment searches for subjects automatically, it scans this directory
    # for subfolders matching _subject_re.
    _subject_loc = 'meg-sdir'

    # Parcellations
    __parcs = {
        'aparc.a2005s': FS_PARC,
        'aparc.a2009s': FS_PARC,
        'aparc': FS_PARC,
        'aparc.DKTatlas': FS_PARC,
        'PALS_B12_Brodmann': FSA_PARC,
        'PALS_B12_Lobes': FSA_PARC,
        'PALS_B12_OrbitoFrontal': FSA_PARC,
        'PALS_B12_Visuotopic': FSA_PARC,
        'lobes': EelbrainParcellation('lobes', morph_from_fsaverage=True,
                                      views=('lateral', 'medial')),
        'lobes-op': CombinationParcellation(
            'lobes-op', 'lobes', {'occipitoparietal': "occipital + parietal"},
            views=('lateral', 'medial')),
        'lobes-ot': CombinationParcellation(
            'lobes-ot', 'lobes', {'occipitotemporal': "occipital + temporal"},
            views=('lateral', 'medial')),
    }
    parcs = {}

    # Frequencies:  lowbound, highbound, step
    _freqs = {'gamma': {'frequencies': np.arange(25, 50, 2),
                        'n_cycles': 5}}
    freqs = {}

    # basic templates to use. Can be a string referring to a templates
    # dictionary in the module level _temp dictionary, or a templates
    # dictionary
    _templates = temp
    # specify additional templates
    _values = {}
    # specify defaults for specific fields (e.g. specify the initial subject
    # name)
    defaults = {}

    # model order: list of factors in the order in which models should be built
    # (default for factors not in this list is alphabetic)
    _model_order = []

    # Backup
    # ------
    # basic state for a backup
    _backup_state = {'subject': '*', 'mrisubject': '*', 'session': '*',
                     'raw': 'raw', 'modality': '*'}
    # files to back up, together with state modifications on the basic state
    _backup_files = (('raw-file-bkp', {}),
                     ('bads-file', {}),
                     ('rej-file', {'raw': '*', 'epoch': '*', 'rej': '*'}),
                     ('ica-file', {'raw': '*', 'epoch': '*', 'rej': '*'}),
                     ('trans-file', {}),
                     ('mri-cfg-file', {}),
                     ('log-dir', {}),)

    # Tests
    # -----
    # specify tests as (test_type, model, test_parameter) tuple. For example,
    # ("anova", "condition", "condition*subject")
    # ("t_contrast_rel", "ref%loc", "+min(ref|left>nref|*, ref|right>nref|*)")
    # Make sure dictionary keys (test names) are appropriate for filenames.
    # tests imply a model which is set automatically
    tests = {}
    _cluster_criteria = {
        '': {'time': 0.025, 'sensor': 4, 'source': 10},
        'all': {},
        '10ms': {'time': 0.01, 'sensor': 4, 'source': 10},
        'large': {'time': 0.025, 'sensor': 8, 'source': 20},
    }

    # plotting
    # --------
    _brain_plot_defaults = {'surf': 'inflated'}
    brain_plot_defaults = {}

    def __init__(self, root=None, find_subjects=True, **state):
        # checks
        if hasattr(self, 'cluster_criteria'):
            raise AttributeError("MneExperiment subclasses can not have a "
                                 ".cluster_criteria attribute. Please remove "
                                 "the attribute, delete the eelbrain-cache "
                                 "folder and use the select_clusters analysis "
                                 "parameter.")

        # create attributes (overwrite class attributes)
        self._subject_re = re.compile(self._subject_re)
        self._mri_subjects = self._mri_subjects.copy()
        self._templates = self._templates.copy()
        self._templates['class-name'] = self.__class__.__name__
        # templates version
        if self.path_version is None:
            raise ValueError("%s.path_version is not set. This parameter needs "
                             "to be specified explicitlty. See <https://"
                             "pythonhosted.org/eelbrain/experiment.html#"
                             "eelbrain.MneExperiment.path_version>" %
                             self.__class__.__name__)
        elif self.path_version == 0:
            self._templates['raw-dir'] = join('{meg-dir}', 'raw')
            self._templates['raw-file'] = join(
                '{raw-dir}', '{subject}_{session}_clm-raw.fif')
            self._templates['raw-file-bkp'] = join(
                '{raw-dir}', '{subject}_{session}_{sns_kind}-raw*.fif')
        elif self.path_version != 1:
            raise ValueError("MneExperiment.path_version needs to be 0 or 1")
        # update templates with _values
        for cls in reversed(inspect.getmro(self.__class__)):
            if hasattr(cls, '_values'):
                self._templates.update(cls._values)

        FileTree.__init__(self)
        self._log = log = logging.Logger(self.__class__.__name__, logging.DEBUG)

        ########################################################################
        # sessions
        if self.sessions is None:
            raise TypeError("The MneExperiment.sessions parameter needs to be "
                            "specified. The session name is contained in your "
                            "raw data files. For example if your file is named "
                            "`R0026_mysession-raw.fif` your session name is "
                            "'mysession' and you should set "
                            "MneExperiment.sessions='mysession'.")
        elif isinstance(self.sessions, basestring):
            self._sessions = (self.sessions,)
        elif isinstance(self.sessions, Sequence):
            self._sessions = tuple(self.sessions)
        else:
            raise TypeError("MneExperiment.sessions needs to be a string or a "
                            "tuple, got %s" % repr(self.sessions))

        ########################################################################
        # subjects
        if root is None:
            find_subjects = False
        else:
            root = self.get('root', root=root)

        if find_subjects:
            sub_dir = self.get(self._subject_loc)
            if not exists(sub_dir):
                raise IOError("Subjects directory not found: %s. Initialize "
                              "with root=None or find_subjects=False" % sub_dir)
            subjects = sorted(s for s in os.listdir(sub_dir) if
                              self._subject_re.match(s) and
                              isdir(join(sub_dir, s)))

            if len(subjects) == 0:
                print("%s: No subjects found in %r"
                      % (self.__class__.__name__, sub_dir))
        else:
            subjects = ()

        ########################################################################
        # groups
        groups = {'all': tuple(subjects)}
        group_definitions = self.groups.copy()
        while group_definitions:
            n_def = len(group_definitions)
            for name, group_def in group_definitions.items():
                if name == '*':
                    raise ValueError("'*' is not a valid group name")
                elif isinstance(group_def, dict):
                    base = (group_def.get('base', 'all'))
                    if base not in groups:
                        continue
                    exclude = group_def['exclude']
                    if isinstance(exclude, basestring):
                        exclude = (exclude,)
                    elif not isinstance(exclude, (tuple, list, set)):
                        raise TypeError("Exclusion must be defined as str | "
                                        "tuple | list | set; got "
                                        "%s" % repr(exclude))
                    group_members = (s for s in groups[base] if s not in exclude)
                elif isinstance(group_def, (list, tuple)):
                    missing = tuple(s for s in group_def if s not in subjects)
                    if missing:
                        raise DefinitionError(
                            "Group %s contains non-existing subjects: %s" %
                            (name, ', '.join(missing)))
                    group_members = sorted(group_def)
                    if len(set(group_members)) < len(group_members):
                        count = Counter(group_members)
                        duplicates = (s for s, n in count.iteritems() if n > 1)
                        raise DefinitionError(
                            "Group %r: the following subjects appear more than "
                            "once: %s" % (name, ', '.join(duplicates)))
                else:
                    raise TypeError("group %s=%r" % (name, group_def))
                groups[name] = tuple(group_members)
                group_definitions.pop(name)
            if len(group_definitions) == n_def:
                raise ValueError("Groups contain unresolvable definition")
        self._groups = groups

        ########################################################################
        # Preprocessing
        skip = {'subject', 'session'}
        raw_path = self._partial('raw-file', skip)
        bads_path = self._partial('bads-file', skip)
        skip.add('raw')
        cache_path = self._partial('cached-raw-file', skip)
        ica_path = self._partial('raw-ica-file', skip)
        raw_dict = self._raw.copy()
        raw_dict.update(self.raw)
        self._raw = assemble_pipeline(raw_dict, raw_path, bads_path, cache_path,
                                      ica_path, log)

        ########################################################################
        # variables
        self.variables = self.variables.copy()
        for k, v in self.variables.iteritems():
            assert_is_legal_dataset_key(k)
            triggers = []
            for trigger, name in v.iteritems():
                if not isinstance(name, basestring):
                    raise TypeError("Invalid cell name in variable "
                                    "definition: %s" % repr(name))

                if isinstance(trigger, tuple):
                    triggers.extend(trigger)
                else:
                    triggers.append(trigger)

            invalid = [t for t in triggers
                       if not (isinstance(t, int) or t == 'default')]
            if invalid:
                raise TypeError("Invalid trigger %s in variable definition %r: "
                                "%r" % (named_list(invalid, 'code'), k, v))

        ########################################################################
        # epochs
        epoch_default = {'session': self._sessions[0]}
        epoch_default.update(self.epoch_default)
        epochs = {}
        secondary_epochs = []
        super_epochs = []
        for name, parameters in self.epochs.iteritems():
            # filter out secondary epochs
            if 'sub_epochs' in parameters:
                super_epochs.append((name, parameters.copy()))
            elif 'base' in parameters:
                secondary_epochs.append((name, parameters.copy()))
            else:
                kwargs = epoch_default.copy()
                kwargs.update(parameters)
                epochs[name] = PrimaryEpoch(name, **kwargs)

        # integrate secondary epochs (epochs with base parameter)
        while secondary_epochs:
            n_secondary_epochs = len(secondary_epochs)
            for i in xrange(n_secondary_epochs - 1, -1, -1):
                name, parameters = secondary_epochs[i]
                if parameters['base'] in epochs:
                    parameters['base'] = epochs[parameters['base']]
                    epochs[name] = SecondaryEpoch(name, **parameters)
                    del secondary_epochs[i]
            if len(secondary_epochs) == n_secondary_epochs:
                raise ValueError("Invalid epoch definition: " +
                                 '; '.join('Epoch %s has non-existing base '
                                           '%r.' % p for p in secondary_epochs))
        # integrate super-epochs
        epochs_ = {}
        for name, parameters in super_epochs:
            try:
                sub_epochs = [epochs[n] for n in parameters.pop('sub_epochs')]
            except KeyError as err:
                msg = 'no epoch named %r' % err.args
                if err.args[0] in super_epochs:
                    msg += '. SuperEpochs can not be defined recursively'
                raise KeyError(msg)
            epochs_[name] = SuperEpoch(name, sub_epochs, parameters)
        epochs.update(epochs_)

        self._epochs = epochs

        ########################################################################
        # store epoch rejection settings
        artifact_rejection = {}
        for name, params in chain(self._artifact_rejection.iteritems(),
                                  self.artifact_rejection.iteritems()):
            if params['kind'] not in ('manual', 'make', 'ica', None):
                raise ValueError("Invalid value in %r rejection setting: "
                                 "kind=%r" % (name, params['kind']))
            params = params.copy()
            if params['kind'] == 'ica':
                if set(params) != ICA_REJ_PARAMS:
                    missing = ICA_REJ_PARAMS.difference(params)
                    unused = set(params).difference(ICA_REJ_PARAMS)
                    msg = "artifact_rejection definition %s" % name
                    if missing:
                        msg += " is missing parameters: (%s)" % ', '.join(missing)
                    if unused:
                        msg += " has unused parameters: (%s)" % ', '.join(unused)
                    raise ValueError(msg)
            artifact_rejection[name] = params

        self._artifact_rejection = artifact_rejection

        ########################################################################
        # Cov
        #####
        for k, params in self._covs.iteritems():
            params = set(params)
            n_datasource = ('epoch' in params) + ('session' in params)
            if n_datasource != 1:
                if n_datasource == 0:
                    raise ValueError("Cov %s has neither epoch nor session "
                                     "entry" % k)
                raise ValueError("Cov %s has both epoch and session entry" % k)
            if params.difference(COV_PARAMS):
                raise ValueError("Cov %s has unused entries: %s" %
                                 ', '.join(params.difference(COV_PARAMS)))

        ########################################################################
        # parcellations
        ###############
        # make : can be made if non-existent
        # morph_from_fraverage : can be morphed from fsaverage to other subjects
        if isinstance(self.parcs, dict):
            user_parcs = self.parcs
        elif self.parcs is None:
            user_parcs = {}
        elif isinstance(self.parcs, tuple):
            user_parcs = {name: FSA_PARC for name in self.parcs}
        else:
            raise TypeError("The MneExperiment.parcs attribute should be a "
                            "dict, got %s" % repr(self.parcs))

        illegal = set(user_parcs).intersection(self.__parcs)
        if illegal:
            raise KeyError("The following parc names are already used by "
                           "builtin parcellations: %s" % ', '.join(illegal))

        parcs = {}
        for name, p in chain(self.__parcs.iteritems(), user_parcs.iteritems()):
            if p == FS_PARC:
                parcs[name] = FreeSurferParcellation(name, ('lateral', 'medial'))
            elif p == FSA_PARC:
                parcs[name] = FSAverageParcellation(name, ('lateral', 'medial'))
            elif isinstance(p, Parcellation):
                parcs[name] = p
            elif isinstance(p, dict):
                p = p.copy()
                kind = p.pop('kind', None)
                if kind is None:
                    raise KeyError("Parcellation %s does not contain the "
                                   "required 'kind' entry" % name)
                elif kind not in PARC_CLASSES:
                    raise ValueError("Parcellation %s contains an invalid "
                                     "'kind' entry: %r" % (name, kind))
                cls = PARC_CLASSES[kind]
                assert_dict_has_args(p, cls, 'parc', name, 1)
                parcs[name] = cls(name, **p)
            else:
                raise ValueError("Parcellations need to be defined as %r, %r or "
                                 "dict, got %s: %r" % (FS_PARC, FSA_PARC, name, p))
        self._parcs = parcs
        parc_values = parcs.keys()
        parc_values += ['']

        ########################################################################
        # frequency
        freqs = {}
        for name, f in chain(self._freqs.iteritems(), self.freqs.iteritems()):
            if name in freqs:
                raise ValueError("Frequency %s defined twice" % name)
            elif 'frequencies' not in f:
                raise KeyError("Frequency values missing for %s" % name)
            elif 'n_cycles' not in f:
                raise KeyError("Number of cycles not defined for %s" % name)
            freqs[name] = f

        self._freqs = freqs

        ########################################################################
        # tests
        self._tests = assemble_tests(self.tests)

        ########################################################################
        # Experiment class setup
        ########################
        self._register_field('mri', sorted(self._mri_subjects))
        self._register_field('subject', subjects or None)
        self._register_field('group', self._groups.keys(), 'all',
                             post_set_handler=self._post_set_group)

        self._register_field('raw', sorted(self._raw))
        self._register_field('rej', self._artifact_rejection.keys(), 'man')

        # epoch
        epoch_keys = sorted(self._epochs)
        for default_epoch in epoch_keys:
            if isinstance(self._epochs[default_epoch], PrimaryEpoch):
                break
        else:
            raise RuntimeError("No primary epoch")
        self._register_field('epoch', epoch_keys, default_epoch)
        self._register_field('session', self._sessions, depends_on=('epoch',),
                             slave_handler=self._update_session)
        # cov
        if 'bestreg' in self._covs:
            default_cov = 'bestreg'
        else:
            default_cov = None
        self._register_field('cov', sorted(self._covs), default_cov)
        self._register_field('inv', default='free-3-dSPM',
                             eval_handler=self._eval_inv,
                             post_set_handler=self._post_set_inv)
        self._register_field('model', eval_handler=self._eval_model)
        self._register_field('test', sorted(self._tests) or None,
                             post_set_handler=self._post_set_test)
        self._register_field('parc', parc_values, 'aparc',
                             eval_handler=self._eval_parc)
        self._register_field('freq', self._freqs.keys())
        self._register_field('src', ('ico-2', 'ico-3', 'ico-4', 'ico-5',
                                     'vol-10', 'vol-7', 'vol-5'), 'ico-4')
        self._register_field('connectivity', ('', 'link-midline'))
        self._register_field('select_clusters', self._cluster_criteria.keys())

        # slave fields
        self._register_field('mrisubject', depends_on=('mri', 'subject'),
                             slave_handler=self._update_mrisubject)
        self._register_field('src-name', depends_on=('src',),
                             slave_handler=self._update_src_name)

        # fields used internally
        self._register_field('analysis', internal=True)
        self._register_field('test_options', internal=True)
        self._register_field('name', internal=True)
        self._register_field('folder', internal=True)
        self._register_field('resname', internal=True)
        self._register_field('ext', internal=True)

        # compounds
        self._register_compound('sns_kind', ('modality', 'raw'))
        self._register_compound('src_kind', ('sns_kind', 'cov', 'mri',
                                             'src-name', 'inv'))
        self._register_compound('evoked_kind', ('rej', 'equalize_evoked_count'))
        self._register_compound('eeg_kind', ('sns_kind', 'reference'))

        # Define make handlers
        self._bind_cache('cov-file', self.make_cov)
        self._bind_cache('src-file', self.make_src)
        self._bind_cache('fwd-file', self.make_fwd)

        # register experimental features
        self._subclass_init()

        # Check that the template model is complete
        self._find_missing_fields()

        # set initial values
        self.set(**state)
        self._store_state()

        ########################################################################
        # logger
        ########
        # log-file
        if root:
            if exists(self.get('old-eelbrain-log-file')):
                os.rename(self.get('old-eelbrain-log-file'),
                          self.get('eelbrain-log-file'))
            handler = logging.FileHandler(self.get('eelbrain-log-file'))
            formatter = logging.Formatter("%(levelname)-8s %(asctime)s %(message)s",
                                          "%m-%d %H:%M")  # %(name)-12s
            handler.setFormatter(formatter)
            handler.setLevel(logging.DEBUG)
            log.addHandler(handler)
        # Terminal log
        handler = logging.StreamHandler()
        # formatter = WrappedFormater("%(levelname)-8s %(name)s:  %(message)s",
        #                             width=100, indent=9)
        formatter = logging.Formatter("%(levelname)-8s:  %(message)s")
        handler.setFormatter(formatter)
        self._screen_log_level = log_level(self.screen_log_level)
        handler.setLevel(self._screen_log_level)
        log.addHandler(handler)
        self._screen_log_handler = handler

        # log package versions
        from .. import __version__
        log.info("*** %s initialized with root %s on %s ***", self.__class__.__name__, 
                 root, datetime.now().strftime('%Y-%m-%d %H:%M:%S'))
        msg = "Using eelbrain %s, mne %s." % (__version__, mne.__version__)
        if any('dev' in v for v in (__version__, mne.__version__)):
            log.warn(msg + " Development versions are more likely to contain "
                     "errors.")
        else:
            log.info(msg)

        if self.auto_delete_cache == 'disable':
            log.warn("Cache-management disabled")
            return

        ########################################################################
        # Cache
        #######
        if self.exclude:
            raise ValueError("MneExperiment.exclude must be unspecified for "
                             "cache management to work")
        elif not root:
            return

        # loading events will create cache-dir
        cache_dir = self.get('cache-dir')
        cache_dir_existed = exists(cache_dir)

        # collect input file information
        # ==============================
        raw_missing = []  # [(subject, session), ...]
        subjects_with_dig_changes = set()  # {subject, ...}
        events = {}  # {(subject, session): event_dataset}

        # saved mtimes
        input_state_file = self.get('input-state-file')
        if len(self._sessions) == 1:
            input_state = None  # currently no check is necessary
        elif exists(input_state_file):
            input_state = load.unpickle(input_state_file)
            if input_state['version'] > CACHE_STATE_VERSION:
                raise RuntimeError(
                    "You are trying to initialize an experiment with an older "
                    "version of Eelbrain than that which wrote the cache. If "
                    "you really need this, delete the eelbrain-cache folder "
                    "before proceeding."
                )
        else:
            input_state = {'version': CACHE_STATE_VERSION}

        # collect current events and mtime
        with self._temporary_state:
            for key in self.iter(('subject', 'session'), group='all', raw='raw'):
                raw_file = self.get('raw-file')
                if not exists(raw_file):
                    raw_missing.append(key)
                    continue
                # events
                events[key] = self.load_events(add_bads=False, data_raw=False)
                # mtime
                if input_state is not None:
                    mtime = getmtime(raw_file)
                    if key not in input_state or mtime != input_state[key]['raw-mtime']:
                        subjects_with_dig_changes.add(key[0])
                        input_state[key] = {'raw-mtime': mtime}
            # save input-state
            if input_state is not None:
                save.pickle(input_state, input_state_file)

        # check for digitizer data differences
        # ====================================
        #  - raw files with different head shapes require different head-mri
        #    trans files, which is currently not implemented
        #  - SuperEpochs currently need to have a single forward solution,
        #    hence marker positions need to be the same between sub-epochs
            if subjects_with_dig_changes:
                log.info("Raw input files changed, checking digitizer data")
                super_epochs = tuple(epoch for epoch in self._epochs.values() if
                                     isinstance(epoch, SuperEpoch))
            for subject in subjects_with_dig_changes:
                self.set(subject)
                digs = {}  # {session: dig}
                dig_ids = {}  # {session: id}
                for session in self.iter('session'):
                    key = subject, session
                    if key in raw_missing:
                        continue
                    raw = self.load_raw(False)
                    dig = raw.info['dig']
                    if dig is None:
                        continue  # assume that it is not used or the same

                    for session_, dig_ in digs.iteritems():
                        if not hsp_equal(dig, dig_):
                            raise NotImplementedError(
                                "Subject %s has different head shape data "
                                "for sessions %s and %s. This would require "
                                "different trans-files for the different "
                                "sessions, which is currently not implemented "
                                "in the MneExperiment class." %
                                (subject, session, session_)
                            )
                        elif mrk_equal(dig, dig_):
                            dig_ids[session] = dig_ids[session_]
                            break
                    else:
                        dig_ids[session] = len(digs)
                        digs[session] = dig

                # check super-epochs
                for epoch in super_epochs:
                    if len(set(dig_ids[s] for s in epoch.sessions if s in
                               dig_ids)) > 1:
                        raise NotImplementedError(
                            "SuperEpoch %s has sessions with incompatible "
                            "marker positions: %s; SuperEpochs with different "
                            "forward solutions are not implemented." %
                            (epoch.name, dig_ids)
                        )

        # Check the cache, delete invalid files
        # =====================================
        cache_state_path = self.get('cache-state-file')
        raw_state = pipeline_dict(self._raw)
        epoch_state = {k: v.as_dict() for k, v in self._epochs.iteritems()}
        parcs_state = {k: v.as_dict() for k, v in self._parcs.iteritems()}
        tests_state = {k: v.as_dict() for k, v in self._tests.iteritems()}
        if exists(cache_state_path):
            # check time stamp
            if getmtime(cache_state_path) > time.time():
                tc = time.ctime(getmtime(cache_state_path))
                tsys = time.asctime()
                raise RuntimeError("The cache's time stamp is in the future "
                                   "(%s). If the system time (%s) is wrong, "
                                   "adjust the system clock; if not, delete "
                                   "the eelbrain-cache folder." % (tc, tsys))
            cache_state = load.unpickle(cache_state_path)
            cache_state_v = cache_state.get('version', 0)
            if cache_state_v < CACHE_STATE_VERSION:
                log.debug("Updating cache-state %i -> %i", cache_state_v,
                          CACHE_STATE_VERSION)
            elif cache_state_v > CACHE_STATE_VERSION:
                raise RuntimeError("The %s cache is from a newer version of "
                                   "Eelbrain than you are currently using. "
                                   "Either upgrade Eelbrain or delete the cache "
                                   "folder.")

            # Backwards compatibility
            # =======================
            # Epochs represented as dict up to Eelbrain 0.24
            if cache_state_v >= 3:
                epoch_state_v = epoch_state
            else:
                epoch_state_v = {k: v.as_dict_24() for k, v in self._epochs.iteritems()}
                for e in cache_state['epochs'].values():
                    e.pop('base', None)
                    if 'sel_epoch' in e:
                        e.pop('n_cases', None)

            # events did not include session
            if cache_state_v < 4:
                if not events:
                    raise DefinitionError(
                        "No raw files ot events found. Did you set the MneExperiment.session "
                        "parameter correctly?")
                session = self._sessions[0]
                cache_events = {(subject, session): v for subject, v in
                                cache_state['events'].iteritems()}
            else:
                cache_events = cache_state['events']

            # raw pipeline
            if cache_state_v < 5:
                cache_raw = pipeline_dict(
                    assemble_pipeline(LEGACY_RAW, '', '', '', '', log))
            else:
                cache_raw = cache_state['raw']

            # parcellations represented as dicts
            cache_parcs = cache_state['parcs']
            if cache_state_v < 6:
                for params in cache_parcs.itervalues():
                    for key in ('morph_from_fsaverage', 'make'):
                        if key in params:
                            del params[key]

            # tests represented as dicts
            cache_tests = cache_state['tests']
            if cache_state_v < 7:
                for params in cache_tests.values():
                    if 'desc' in params:
                        del params['desc']
                cache_tests = {k: v.as_dict() for k, v in
                               assemble_tests(cache_tests).iteritems()}

            # Find modified definitions
            # =========================
            invalid_cache = defaultdict(set)
            # events (subject, session):  overall change in events
            # variables:  event change restricted to certain variables
            # raw: preprocessing definition changed
            # groups:  change in group members
            # epochs:  change in epoch parameters
            # parcs: parc def change
            # tests: test def change

            # check events
            # 'events' -> number or timing of triggers (includes trigger_shift)
            # 'variables' -> only variable change
            for key, old_events in cache_events.iteritems():
                new_events = events.get(key)
                if new_events is None:
                    invalid_cache['events'].add(key)
                    log.warn("  raw file removed: %s", '/'.join(key))
                elif new_events.n_cases != old_events.n_cases:
                    invalid_cache['events'].add(key)
                    log.warn("  event length: %s %i->%i", '/'.join(key),
                             old_events.n_cases, new_events.n_cases)
                elif not np.all(new_events['i_start'] == old_events['i_start']):
                    invalid_cache['events'].add(key)
                    log.warn("  trigger timing changed: %s", '/'.join(key))
                else:
                    for var in old_events:
                        if var == 'i_start':
                            continue
                        elif var not in new_events:
                            invalid_cache['variables'].add(var)
                            log.debug("  var removed: %s (%s)", var, '/'.join(key))
                            continue
                        old = old_events[var]
                        new = new_events[var]
                        if old.name != new.name:
                            invalid_cache['variables'].add(var)
                            log.warn("  var name changed: %s (%s) %s->%s", var,
                                     '/'.join(key), old.name, new.name)
                        elif new.__class__ is not old.__class__:
                            invalid_cache['variables'].add(var)
                            log.warn("  var type changed: %s (%s) %s->%s", var,
                                     '/'.join(key), old.__class__, new.__class)
                        elif not all_equal(old, new, True):
                            invalid_cache['variables'].add(var)
                            log.warn("  var changed: %s (%s) %i values", var,
                                     '/'.join(key), np.sum(new != old))

            # groups
            for group, members in cache_state['groups'].iteritems():
                if group not in self._groups:
                    invalid_cache['groups'].add(group)
                    log.warn("  Group removed: %s", group)
                elif set(members) != set(self._groups[group]):
                    invalid_cache['groups'].add(group)
                    log_list_change(log, "Group", group, members, self._groups[group])

            # raw
            changed, changed_ica = compare_pipelines(cache_raw, raw_state, log)
            if changed:
                invalid_cache['raw'].update(changed)
            for raw, status in changed_ica.iteritems():
                filenames = self.glob('raw-ica-file', raw=raw, subject='*')
                if filenames:
                    print("Outdated ICA files:\n" + '\n'.join(
                          relpath(path, root) for path in filenames))
                    ask_to_delete_ica_files(raw, status, filenames)

            # epochs
            for epoch, old_params in cache_state['epochs'].iteritems():
                new_params = epoch_state_v.get(epoch, None)
                if old_params != new_params:
                    invalid_cache['epochs'].add(epoch)
                    if new_params is None:
                        log.warn("  Epoch removed: %s", epoch)
                    else:
                        log_dict_change(log, 'Epoch', epoch, old_params, new_params)

            # parcs
            for parc, params in cache_parcs.iteritems():
                if parc not in parcs_state:
                    invalid_cache['parcs'].add(parc)
                    log.warn("  Parc %s removed", parc)
                elif params != parcs_state[parc]:
                    # FS_PARC:  Parcellations that are provided by the user
                    # should not be automatically removed.
                    # FSA_PARC:  for other mrisubjects, the parcellation
                    # should automatically update if the user changes the
                    # fsaverage file.
                    if not isinstance(self._parcs[parc],
                                      (FreeSurferParcellation, FSAverageParcellation)):
                        invalid_cache['parcs'].add(parc)
                        log_dict_change(log, "Parc", parc, params, parcs_state[parc])

            # tests
            for test, params in cache_tests.iteritems():
                if test not in tests_state or params != tests_state[test]:
                    invalid_cache['tests'].add(test)
                    if test in tests_state:
                        log_dict_change(log, "Test", test, params, tests_state[test])
                    else:
                        log.warn("  Test %s removed", test)

            # create message here, before secondary invalidations are added
            msg = []
            if cache_state_v < 2:
                msg.append("Check for invalid ANOVA tests (cache version %i)." %
                           cache_state_v)
            if invalid_cache:
                msg.append("Experiment definition changed:")
                for kind, values in invalid_cache.iteritems():
                    msg.append("  %s: %s" % (kind, ', '.join(map(str, values))))

            # Secondary  invalidations
            # ========================
            # changed events -> group result involving those subjects is also bad
            if 'events' in invalid_cache:
                subjects = {subject for subject, _ in invalid_cache['events']}
                for group, members in cache_state['groups'].iteritems():
                    if subjects.intersection(members):
                        invalid_cache['groups'].add(group)

            # tests/epochs based on variables
            if 'variables' in invalid_cache:
                bad_vars = invalid_cache['variables']
                # tests using bad variable
                for test, params in cache_tests.iteritems():
                    if test not in invalid_cache['tests']:
                        bad = bad_vars.intersection(find_test_vars(params))
                        if bad:
                            invalid_cache['tests'].add(test)
                            log.debug("  Test %s depends on changed variables %s",
                                      test, ', '.join(bad))
                # epochs using bad variable
                epochs_vars = find_epochs_vars(cache_state['epochs'])
                for epoch, evars in epochs_vars.iteritems():
                    bad = bad_vars.intersection(evars)
                    if bad:
                        invalid_cache['epochs'].add(epoch)
                        log.debug("  Epoch %s depends on changed variables %s",
                                  epoch, ', '.join(bad))

            # secondary epochs
            if 'epochs' in invalid_cache:
                for e in tuple(invalid_cache['epochs']):
                    invalid_cache['epochs'].update(find_dependent_epochs(e, cache_state['epochs']))

            # Collect invalid files
            # =====================
            if invalid_cache or cache_state_v < 2:
                rm = defaultdict(DictSet)

                # version
                if cache_state_v < 2:
                    bad_parcs = []
                    for parc, params in self._parcs.iteritems():
                        if params['kind'] == 'seeded':
                            bad_parcs.append(parc + '-?')
                            bad_parcs.append(parc + '-??')
                            bad_parcs.append(parc + '-???')
                        else:
                            bad_parcs.append(parc)
                    bad_tests = []
                    for test, params in tests_state.iteritems():
                        if params['kind'] == 'anova' and params['x'].count('*') > 1:
                            bad_tests.append(test)
                    if bad_tests and bad_parcs:
                        log.warning("  Invalid ANOVA tests: %s for %s",
                                    bad_tests, bad_parcs)
                    for test, parc in product(bad_tests, bad_parcs):
                        rm['test-file'].add({'test': test, 'data_parc': parc})
                        rm['report-file'].add({'test': test, 'folder': parc})

                # evoked files are based on old events
                for subject, session in invalid_cache['events']:
                    rm['evoked-file'].add({'subject': subject, 'session': session})

                # variables
                for var in invalid_cache['variables']:
                    rm['evoked-file'].add({'model': '*%s*' % var})

                # groups
                for group in invalid_cache['groups']:
                    rm['test-file'].add({'group': group})
                    rm['group-mov-file'].add({'group': group})
                    rm['report-file'].add({'group': group})

                # raw
                for raw in invalid_cache['raw']:
                    rm['cached-raw-file'].add({'raw': raw})
                    rm['evoked-file'].add({'raw': raw})
                    analysis = {'analysis': '* %s *' % raw}
                    rm['test-file'].add(analysis)
                    rm['report-file'].add(analysis)
                    rm['group-mov-file'].add(analysis)
                    rm['subject-mov-file'].add(analysis)

                # epochs
                for epoch in invalid_cache['epochs']:
                    rm['evoked-file'].add({'epoch': epoch})
                    for cov, cov_params in self._covs.iteritems():
                        if cov_params.get('epoch') != epoch:
                            continue
                        analysis = '* %s *' % cov
                        rm['test-file'].add({'analysis': analysis})
                        rm['report-file'].add({'analysis': analysis})
                        rm['group-mov-file'].add({'analysis': analysis})
                        rm['subject-mov-file'].add({'analysis': analysis})
                    rm['test-file'].add({'epoch': epoch})
                    rm['report-file'].add({'epoch': epoch})
                    rm['group-mov-file'].add({'epoch': epoch})
                    rm['subject-mov-file'].add({'epoch': epoch})

                # parcs
                bad_parcs = []
                for parc in invalid_cache['parcs']:
                    if cache_state['parcs'][parc]['kind'] == 'seeded':
                        bad_parcs.append(parc + '-?')
                        bad_parcs.append(parc + '-??')
                        bad_parcs.append(parc + '-???')
                    else:
                        bad_parcs.append(parc)
                for parc in bad_parcs:
                    rm['annot-file'].add({'parc': parc})
                    rm['test-file'].add({'data_parc': parc})
                    rm['report-file'].add({'folder': parc})
                    rm['report-file'].add({'folder': '%s *' % parc})
                    rm['report-file'].add({'folder': '%s *' % parc.capitalize()})  # pre 0.26
                    rm['res-file'].add({'analysis': 'Source Annot',
                                        'resname': parc + ' * *', 'ext': 'p*'})

                # tests
                for test in invalid_cache['tests']:
                    rm['test-file'].add({'test': test})
                    rm['report-file'].add({'test': test})

                # find actual files to delete
                log.debug("Outdated cache files:")
                files = set()
                result_files = []
                for temp, arg_dicts in rm.iteritems():
                    keys = self.find_keys(temp, False)
                    for args in arg_dicts:
                        kwargs = {k: args.get(k, '*') for k in keys}
                        pattern = self._glob_pattern(temp, vmatch=False, **kwargs)
                        filenames = glob(pattern)
                        files.update(filenames)
                        # log
                        rel_pattern = relpath(pattern, root)
                        rel_filenames = sorted('  ' + relpath(f, root) for f in filenames)
                        log.debug(' >%s', rel_pattern)
                        map(log.debug, rel_filenames)
                        # message to the screen unless log is already displayed
                        if rel_pattern.startswith('results'):
                            result_files.extend(rel_filenames)

                # handle invalid files
                n_result_files = len(result_files)
                if n_result_files and self.auto_delete_cache is True and not self.auto_delete_results:
                    if self._screen_log_level > logging.DEBUG:
                        msg = result_files[:]
                        msg.insert(0, "Outdated result files detected:")
                    else:
                        msg = []
                    msg.append("Delete %i outdated results?" % (n_result_files,))
                    options = [
                        ('delete', 'delete invalid result files'),
                        ('abort', 'raise an error'),
                    ]
                    command = ask('\n'.join(msg), options)
                    if command == 'abort':
                        raise RuntimeError("User aborted invalid result deletion")
                    elif command != 'delete':
                        raise RuntimeError("command=%r" % (command,))

                if files:
                    msg = []
                    if self.auto_delete_cache is False:
                        msg.append("Automatic cache management disabled. Either "
                                   "revert changes, or set e.auto_delete_cache=True")
                        raise RuntimeError('\n'.join(msg))
                    elif isinstance(self.auto_delete_cache, basestring):
                        if self.auto_delete_cache != 'debug':
                            raise ValueError("MneExperiment.auto_delete_cache=%r" %
                                             (self.auto_delete_cache,))
                        options = [
                            ('delete', 'delete invalid files'),
                            ('abort', 'raise an error'),
                            ('ignore', 'proceed without doing anything'),
                            ('revalidate', "don't delete any cache files but write a new cache-state file"),
                        ]
                        command = ask('\n'.join(msg), options)
                        if command == 'delete':
                            pass
                        elif command == 'abort':
                            raise RuntimeError("User aborted invalid cache deletion")
                        elif command == 'ignore':
                            log.warn("Ignoring invalid cache")
                            return
                        elif command == 'revalidate':
                            log.warn("Revalidating invalid cache")
                            files.clear()
                        else:
                            raise RuntimeError("command=%s" % repr(command))
                    elif self.auto_delete_cache is not True:
                        raise TypeError("MneExperiment.auto_delete_cache=%s" %
                                        repr(self.auto_delete_cache))

                    # delete invalid files
                    n_cache_files = len(files) - n_result_files
                    descs = []
                    if n_result_files:
                        descs.append("%i invalid result files" % n_result_files)
                    if n_cache_files:
                        descs.append("%i invalid cache files" % n_cache_files)
                    log.info("Deleting " + (' and '.join(descs)) + '...')
                    for path in files:
                        os.remove(path)
                else:
                    log.debug("No existing cache files affected.")
            else:
                log.debug("Cache up to date.")
        elif cache_dir_existed:  # cache-dir but no history
            if self.auto_delete_cache is True:
                log.info("Deleting cache-dir without history")
                shutil.rmtree(cache_dir)
                os.mkdir(cache_dir)
            elif self.auto_delete_cache == 'disable':
                log.warn("Ignoring cache-dir without history")
                pass
            elif self.auto_delete_cache == 'debug':
                command = ask("Cache directory without history",
                              (('validate', 'write a history file treating cache as valid'),
                               ('abort', 'raise an error')))
                if command == 'abort':
                    raise RuntimeError("User aborted")
                elif command == 'validate':
                    log.warn("Validating cache-dir without history")
                else:
                    raise RuntimeError("command=%r" % (command,))
            else:
                raise IOError("Cache directory without history, but "
                              "auto_delete_cache is not True")
        elif not exists(cache_dir):
            os.mkdir(cache_dir)

        new_state = {'version': CACHE_STATE_VERSION,
                     'raw': raw_state,
                     'groups': self._groups,
                     'epochs': epoch_state,
                     'tests': tests_state,
                     'parcs': parcs_state,
                     'events': events}
        save.pickle(new_state, cache_state_path)

    def _subclass_init(self):
        "Allow subclass to register experimental features"

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        for subject in self.iter():
            yield subject

    # mtime methods
    # -------------
    # _mtime() functions return the time at which any input files affecting the
    # given file changed, and None if inputs are missing. They don't check
    # whether the file actually exists (usually there is no need to recompute an
    # intermediate file if it is not needed).
    # _file_mtime() functions directly return the file's mtime, or None if it
    # does not exists or is outdated
    def _annot_file_mtime(self, make_for=None):
        """Return max mtime of annot files or None if they do not exist.

        Can be user input, so we need to check the actual file.
        """
        if make_for:
            with self._temporary_state:
                self.make_annot(mrisubject=make_for)
                return self._annot_file_mtime()

        mtime = 0
        for _ in self.iter('hemi'):
            fpath = self.get('annot-file')
            if exists(fpath):
                mtime = max(mtime, getmtime(fpath))
            else:
                return
        return mtime

    def _cov_mtime(self):
        params = self._covs[self.get('cov')]
        with self._temporary_state:
            if 'epoch' in params:
                self.set(epoch=params['epoch'])
                return self._epochs_mtime()
            else:
                self.set(session=params['session'])
                return self._raw_mtime()

    def _epochs_mtime(self):
        bads_path = self.get('bads-file')
        if exists(bads_path):
            raw_mtime = self._raw_mtime()
            bads_mtime = getmtime(bads_path)
            epoch = self._epochs[self.get('epoch')]
            rej_mtime = self._rej_mtime(epoch)
            if rej_mtime:
                return max(raw_mtime, bads_mtime, rej_mtime)

    def _epochs_stc_mtime(self):
        "Mtime affecting source estimates; does not check annot"
        epochs_mtime = self._epochs_mtime()
        if epochs_mtime:
            inv_mtime = self._inv_mtime()
            if inv_mtime:
                return max(epochs_mtime, inv_mtime)

    def _evoked_mtime(self):
        return self._epochs_mtime()

    def _evoked_stc_mtime(self):
        "Mtime if up-to-date, else None; do not check annot"
        evoked_mtime = self._evoked_mtime()
        if evoked_mtime:
            inv_mtime = self._inv_mtime()
            if inv_mtime:
                return max(evoked_mtime, inv_mtime)

    def _fwd_mtime(self):
        "The last time at which input files affecting fwd-file changed"
        trans = self.get('trans-file')
        if exists(trans):
            src = self.get('src-file')
            if exists(src):
                trans_mtime = getmtime(trans)
                src_mtime = getmtime(src)
                return max(self._raw_mtime('raw'), trans_mtime, src_mtime)

    def _ica_file_mtime(self, rej):
        "Mtime if the file exists, else None; do not check raw mtime"
        ica_path = self.get('ica-file')
        if exists(ica_path):
            ica_mtime = getmtime(ica_path)
            if rej['source'] == 'raw':
                return ica_mtime
            else:
                ica_epoch = self._epochs[rej['epoch']]
                rej_mtime = self._rej_mtime(ica_epoch, pre_ica=True)
                if rej_mtime and ica_mtime > rej_mtime:
                    return ica_mtime

    def _inv_mtime(self):
        fwd_mtime = self._fwd_mtime()
        if fwd_mtime:
            cov_mtime = self._cov_mtime()
            if cov_mtime:
                return max(cov_mtime, fwd_mtime)

    def _raw_mtime(self, raw=None):
        if raw is None:
            pipe = self._raw[self.get('raw')]
            return pipe.mtime(self.get('subject'), self.get('session'))
        elif raw == 'raw':
            return getmtime(self.get('raw-file'))
        else:
            raise RuntimeError("raw-mtime with raw=%s" % repr(raw))

    def _rej_mtime(self, epoch, pre_ica=False):
        """rej-file mtime for secondary epoch definition

        Parameters
        ----------
        epoch : dict
            Epoch definition.
        pre_ica : bool
            Only analyze mtime before ICA file estimation.
        """
        rej = self._artifact_rejection[self.get('rej')]
        if rej['kind'] is None:
            return 1  # no rejection
        with self._temporary_state:
            paths = [self.get('rej-file', epoch=e) for e in epoch.rej_file_epochs]
        if all(exists(path) for path in paths):
            mtime = max(getmtime(path) for path in paths)
            if pre_ica or rej['kind'] != 'ica' or rej['source'] == 'raw':
                return mtime
            # incorporate ICA-file
            ica_mtime = self._ica_file_mtime(rej)
            if ica_mtime > mtime:
                return ica_mtime

    def _result_file_mtime(self, dst, data, single_subject=False):
        """For several results (reports and movies)

        Parameters
        ----------
        dst : str
            Filename.
        data : 'source' | 'sensor'
            Data type.
        cached_test : bool
            Whether a corresponding test is being cached and needs to be
            checked (e.g. for reports that are based on a cached test).
        single_subject : bool
            Whether the corresponding test is performed for a single subject
            (as opposed to the current group).
        """
        if exists(dst):
            mtime = self._result_mtime(data, single_subject)
            if mtime:
                dst_mtime = getmtime(dst)
                if dst_mtime > mtime:
                    return dst_mtime

    def _result_mtime(self, data, single_subject=False):
        "See ._result_file_mtime() above"
        if data == 'source':
            if single_subject:
                mrisubject = 'mrisubject'
            else:
                mrisubject = 'common_brain'

            out = self._annot_file_mtime(self.get(mrisubject))
            if not out:
                return

            mtime_func = self._epochs_stc_mtime
        else:
            out = 1
            mtime_func = self._epochs_mtime

        if single_subject:
            mtime_iterator = (mtime_func(),)
        else:
            mtime_iterator = (mtime_func() for _ in self)

        for mtime in mtime_iterator:
            if not mtime:
                return
            out = max(out, mtime)
        return out

    def _process_subject_arg(self, subject, kwargs):
        """Process subject arg for methods that work on groups and subjects

        Returns
        -------
        subject : None | str
            Subject name if the value specifies a subject, None otherwise.
        group : None | str
            Group name if the value specifies a group, None otherwise.
        """
        if subject is None:
            group = None
            subject_ = self.get('subject', **kwargs)
        elif subject is True:
            group = self.get('group', **kwargs)
            subject_ = None
        elif subject in self.get_field_values('group'):
            group = subject
            subject_ = None
            self.set(group=group, **kwargs)
        else:
            group = None
            subject_ = subject
            self.set(subject=subject, **kwargs)

        return subject_, group

    def _cluster_criteria_kwargs(self, dims):
        criteria = self._cluster_criteria[self.get('select_clusters')]
        return {'min' + dim: criteria[dim] for dim in dims if dim in criteria}

    def _add_epochs(self, ds, epoch, baseline, ndvar, data_raw, pad, decim,
                    reject, apply_ica, trigger_shift, eog, tmin, tmax, tstop):
        modality = self.get('modality')
        if tmin is None:
            tmin = epoch.tmin
        if tmax is None and tstop is None:
            tmax = epoch.tmax
        if baseline is True:
            baseline = epoch.baseline
        if pad:
            tmin -= pad
            tmax += pad
        if decim is None:
            decim = epoch.decim

        # determine ICA
        if apply_ica and self._artifact_rejection[self.get('rej')]['kind'] == 'ica':
            ica = self.load_ica()
            baseline_ = None
        else:
            ica = None
            baseline_ = baseline

        ds = load.fiff.add_mne_epochs(ds, tmin, tmax, baseline_, decim=decim,
                                      drop_bad_chs=False, tstop=tstop)

        # post baseline-correction trigger shift
        if trigger_shift and epoch.post_baseline_trigger_shift:
            ds['epochs'] = shift_mne_epoch_trigger(ds['epochs'],
                                                   ds[epoch.post_baseline_trigger_shift],
                                                   epoch.post_baseline_trigger_shift_min,
                                                   epoch.post_baseline_trigger_shift_max)

        # interpolate channels
        if reject and ds.info[INTERPOLATE_CHANNELS]:
            if modality == '':
                interp_path = self.get('interp-file')
                if exists(interp_path):
                    interp_cache = load.unpickle(interp_path)
                else:
                    interp_cache = {}
                n_in_cache = len(interp_cache)
                _interpolate_bads_meg(ds['epochs'], ds[INTERPOLATE_CHANNELS],
                                      interp_cache)
                if len(interp_cache) > n_in_cache:
                    save.pickle(interp_cache, interp_path)
            else:
                _interpolate_bads_eeg(ds['epochs'], ds[INTERPOLATE_CHANNELS])

        # ICA
        if ica is not None:
            ica.apply(ds['epochs'])
            if baseline:
                ds['epochs'].apply_baseline(baseline)

        if ndvar:
            sysname = self._sysname(ds.info['raw'], ds.info['subject'], modality)
            name = self._ndvar_name_for_modality(modality)
            ds[name] = load.fiff.epochs_ndvar(ds['epochs'], sysname=sysname,
                                              data=self._data_arg(modality, eog))
            if ndvar != 'both':
                del ds['epochs']
            if modality == 'eeg':
                self._fix_eeg_ndvar(ds[ndvar], True)

        if data_raw is False:
            del ds.info['raw']

        return ds

    def _add_epochs_stc(self, ds, ndvar, baseline, morph, mask):
        """
        Transform epochs contained in ds into source space

        Data is added to ``ds`` as a list of :class:`mne.SourceEstimate`.

        Parameters
        ----------
        ds : Dataset
            The Dataset containing the mne Epochs for the desired trials.
        ndvar : bool
            Add the source estimates as NDVar named 'src' (default). Set to
            False to add a list of MNE SourceEstimate objects named 'stc'.
        baseline : None | True | tuple
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to apply no baseline
            correction (None).
        morph : bool
            Morph the source estimates to the common_brain (default False).
        mask : bool | str
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False).
        """
        subject = ds['subject']
        if len(subject.cells) != 1:
            err = "ds must have a subject variable with exactly one subject"
            raise ValueError(err)
        subject = subject.cells[0]
        self.set(subject=subject)
        if baseline is True:
            baseline = self._epochs[self.get('epoch')].baseline

        epochs = ds['epochs']
        inv = self.load_inv(epochs)
        stc = apply_inverse_epochs(epochs, inv, **self._params['apply_inv_kw'])

        if ndvar:
            parc = self.get('parc') or None
            if isinstance(mask, basestring) and parc != mask:
                parc = mask
                self.set(parc=mask)
            self.make_annot()
            subject = self.get('mrisubject')
            src = self.get('src')
            mri_sdir = self.get('mri-sdir')
            src = load.fiff.stc_ndvar(stc, subject, src, mri_sdir,
                                      self._params['apply_inv_kw']['method'],
                                      self._params['make_inv_kw'].get('fixed', False),
                                      parc=parc,
                                      connectivity=self.get('connectivity'))
            if baseline:
                src -= src.summary(time=baseline)

            if morph:
                common_brain = self.get('common_brain')
                with self._temporary_state:
                    self.make_annot(mrisubject=common_brain)
                ds['srcm'] = morph_source_space(src, common_brain)
                if mask:
                    _mask_ndvar(ds, 'srcm')
            else:
                ds['src'] = src
                if mask:
                    _mask_ndvar(ds, 'src')
        else:
            if baseline:
                raise NotImplementedError("Baseline for SourceEstimate")
            if morph:
                raise NotImplementedError("Morphing for SourceEstimate")
            ds['stc'] = stc

    def _add_evoked_stc(self, ds, ind_stc=False, ind_ndvar=False, morph_stc=False,
                        morph_ndvar=False, baseline=None, keep_evoked=False,
                        mask=False):
        """
        Add source estimates to a dataset with evoked data.

        Parameters
        ----------
        ds : Dataset
            The Dataset containing the Evoked objects.
        ind_stc : bool
            Add source estimates on individual brains as list of
            :class:`mne.SourceEstimate`.
        ind_ndvar : bool
            Add source estimates on individual brain as :class:`NDVar` (only
            possible for datasets containing data of a single subject).
        morph_stc : bool
            Add source estimates morphed to the common brain as list of
            :class:`mne.SourceEstimate`.
        morph_ndvar : bool
            Add source estimates morphed to the common brain as :class:`NDVar`.
        baseline : None | True | tuple
            Apply baseline correction (in source space) using this period. True
            to use the epoch's baseline specification. The default is to not
            apply baseline correction (None).
        keep_evoked : bool
            Keep the sensor space data in the Dataset that is returned (default
            False).
        mask : bool | str
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False).

        Notes
        -----
        Assumes that all Evoked of the same subject share the same inverse
        operator.
        """
        if not any((ind_stc, ind_ndvar, morph_stc, morph_ndvar)):
            raise ValueError("Nothing to load, set at least one of (ind_stc, "
                             "ind_ndvar, morph_stc, morph_ndvar) to True")

        if isinstance(baseline, str):
            raise NotImplementedError("Baseline form different epoch")
        elif baseline is True:
            baseline = self._epochs[self.get('epoch')].baseline

        # find from subjects
        common_brain = self.get('common_brain')
        meg_subjects = ds.eval('subject.cells')
        n_subjects = len(meg_subjects)
        if ind_ndvar and n_subjects > 1:
            err = ("Can't use ind_ndvar with data from more than one "
                   "subjects; an NDVar can only be created from stcs that are "
                   "estimated on the same brain. Use morph_ndvar=True "
                   "instead.")
            raise ValueError(err)
        from_subjects = {}  # from-subject for the purpose of morphing
        for subject in meg_subjects:
            if is_fake_mri(self.get('mri-dir', subject=subject)):
                subject_from = common_brain
            else:
                subject_from = self.get('mrisubject', subject=subject)
            from_subjects[subject] = subject_from

        collect_morphed_stcs = (morph_stc or morph_ndvar)
        collect_ind_stcs = ind_stc or ind_ndvar

        # make sure annot files are available (needed only for NDVar)
        all_are_common_brain = all(v == common_brain for v in from_subjects.values())
        if (ind_ndvar and all_are_common_brain) or morph_ndvar:
            self.make_annot(mrisubject=common_brain)
        if ind_ndvar and not all_are_common_brain:
            self.make_annot(mrisubject=from_subjects[meg_subjects[0]])

        # convert evoked objects
        stcs = []
        mstcs = []
        invs = {}
        mm_cache = CacheDict(self.load_morph_matrix, 'mrisubject')
        for subject, evoked in izip(ds['subject'], ds['evoked']):
            subject_from = from_subjects[subject]

            # get inv
            if subject in invs:
                inv = invs[subject]
            else:
                inv = invs[subject] = self.load_inv(evoked, subject=subject)

            # apply inv
            stc = apply_inverse(evoked, inv, **self._params['apply_inv_kw'])

            # baseline correction
            if baseline:
                rescale(stc._data, stc.times, baseline, 'mean', copy=False)

            if collect_ind_stcs:
                stcs.append(stc)

            if collect_morphed_stcs:
                if subject_from == common_brain:
                    if ind_stc:
                        stc = stc.copy()
                    stc.subject = common_brain
                else:
                    mm, v_to = mm_cache[subject_from]
                    stc = mne.morph_data_precomputed(subject_from, common_brain,
                                                     stc, v_to, mm)
                mstcs.append(stc)

        # add to Dataset
        src = self.get('src')
        parc = mask if isinstance(mask, basestring) else self.get('parc') or None
        mri_sdir = self.get('mri-sdir')
        # for name, key in izip(do, keys):
        if ind_stc:
            ds['stc'] = stcs
        if ind_ndvar:
            subject = from_subjects[meg_subjects[0]]
            ds['src'] = load.fiff.stc_ndvar(stcs, subject, src, mri_sdir,
                                            self._params['apply_inv_kw']['method'],
                                            self._params['make_inv_kw'].get('fixed', False),
                                            parc=parc,
                                            connectivity=self.get('connectivity'))
            if mask:
                _mask_ndvar(ds, 'src')
        if morph_stc or morph_ndvar:
            if morph_stc:
                ds['stcm'] = mstcs
            if morph_ndvar:
                ds['srcm'] = load.fiff.stc_ndvar(mstcs, common_brain, src, mri_sdir,
                                                 self._params['apply_inv_kw']['method'],
                                                 self._params['make_inv_kw'].get('fixed', False),
                                                 parc=parc,
                                                 connectivity=self.get('connectivity'))
                if mask:
                    _mask_ndvar(ds, 'srcm')

        if not keep_evoked:
            del ds['evoked']

    def _add_vars(self, ds, vardef):
        """Add vars to the dataset

        Parameters
        ----------
        ds : Dataset
            Event dataset.
        vardef : dict | tuple
            Variable definition.
        """
        if isinstance(vardef, str):
            try:
                vardef = self._tests[vardef].vars
            except KeyError:
                raise ValueError("vardef must be a valid test definition, got "
                                 "vardef=%r" % vardef)
        if vardef is None:
            return

        if isinstance(vardef, tuple):
            for item in vardef:
                name, vdef = item.split('=', 1)
                ds[name.strip()] = as_vardef_var(ds.eval(vdef))
        elif isinstance(vardef, dict):
            new = {}
            for name, definition in vardef.iteritems():
                if isinstance(definition, str):
                    new[name] = as_vardef_var(ds.eval(definition))
                else:
                    source, codes = definition
                    new[name] = asfactor(source, ds=ds).as_var(codes, 0, name)
            ds.update(new, True)
        else:
            raise TypeError("type(vardef)=%s; needs to be dict or tuple" %
                            type(vardef))

    def _backup(self, dst_root, v=False):
        """Backup all essential files to ``dst_root``.

        Parameters
        ----------
        dst_root : str
            Directory to use as root for the backup.
        v : bool
            Verbose mode:  list all files that will be copied and ask for
            confirmation.

        Notes
        -----
        For repeated backups ``dst_root`` can be the same. If a file has been
        previously backed up, it is only copied if the local copy has been
        modified more recently than the previous backup. If the backup has been
        modified more recently than the local copy, a warning is displayed.

        Currently, the following files are included in the backup::

         * Input raw file (raw='raw')
         * Bad channels file
         * All rejection files
         * The trans-file
         * All files in the ``meg/{subject}/logs`` directory
         * For scaled MRIs, the file specifying the scale parameters

        MRIs are currently not backed up.
        """
        self._log.debug("Initiating backup to %s" % dst_root)
        root = self.get('root')
        root_len = len(root) + 1

        dirs = []  # directories to create
        pairs = []  # (src, dst) pairs to copy
        for temp, state_mod in self._backup_files:
            # determine state
            if state_mod:
                state = self._backup_state.copy()
                state.update(state_mod)
            else:
                state = self._backup_state

            # find files to back up
            if temp.endswith('dir'):
                paths = []
                for dirpath in self.glob(temp, **state):
                    for root_, _, filenames in os.walk(dirpath):
                        paths.extend(join(root_, fn) for fn in filenames)
            else:
                paths = self.glob(temp, **state)

            # convert to (src, dst) pairs
            for src in paths:
                if not src.startswith(root):
                    raise ValueError("Can only backup files in root directory")
                tail = src[root_len:]
                dst = join(dst_root, tail)
                if exists(dst):
                    src_m = getmtime(src)
                    dst_m = getmtime(dst)
                    if dst_m == src_m:
                        continue
                    elif dst_m > src_m:
                        self._log.warn("Backup more recent than original: %s",
                                       tail)
                        continue
                else:
                    i = 0
                    while True:
                        i = tail.find(os.sep, i + 1)
                        if i == -1:
                            break
                        path = tail[:i]
                        if path not in dirs:
                            dirs.append(path)

                pairs.append((src, dst))

        if len(pairs) == 0:
            if v:
                print("All files backed up.")
            else:
                self._log.info("All files backed up.")
            return

        # verbose file list
        if v:
            paths = [relpath(src, root) for src, _ in pairs]
            print('\n'.join(paths))
            cmd = 'x'
            while cmd not in 'yn':
                cmd = raw_input("Proceed ([y]/n)? ")
            if cmd == 'n':
                print("Abort.")
                return
            else:
                print("Backing up %i files ..." % len(pairs))

        self._log.info("Backing up %i files ..." % len(pairs))
        # create directories
        for dirname in dirs:
            dirpath = join(dst_root, dirname)
            if not exists(dirpath):
                os.mkdir(dirpath)
        # copy files
        for src, dst in pairs:
            shutil.copy2(src, dst)

    def clear_cache(self, level=1):
        """Remove cached files.

        Parameters
        ----------
        level : int
            Level up to which to clear the cache (see notes below). The default
            is 1, which deletes all cached files.

        Notes
        -----
        Each lower level subsumes the higher levels:

        ``1``
            Delete all cached files.
        ``2``
            Epoched files - these need to be cleared when anything about the
            epoch definition changes (tmin, tmax, event inclusion, ...). Note
            that you might also have to manually update epoch rejection files
            with the :meth:`MneExperiment.make_rej` method.
        ``5``
            tests - these need to be cleared when the members of the relevant
            subject groups change.

        Examples
        --------
        To delete only test files, after adding raw data for a new subject to
        the experiment::

            >>> e.clear_cache(5)

        To delete cached data files after changing the selection criteria for
        a secondary epoch::

            >>> e.clear_cache(2)

        If criteria on a primary epoch are changed, the trial rejection has to
        be re-done in addition to clearing the cache.

        To delete all cached files and clear up hard drive space::

            >>> e.clear_cache(1)
        """
        if level <= 1:
            self.rm('cache-dir', confirm=True)
            print("All cached data cleared.")
        else:
            if level <= 2:
                self.rm('evoked-dir', confirm=True)
                self.rm('cov-dir', confirm=True)
                print("Cached epoch data cleared")
            if level <= 5:
                self.rm('test-dir', confirm=True)
                print("Cached tests cleared.")

    def _fix_eeg_ndvar(self, ndvar, apply_standard_montag):
        # connectivity
        ndvar.sensor.set_connectivity(predefined_connectivity('BrainCap32Ch'))
        # montage
        if apply_standard_montag:
            m = mne.channels.read_montage('easycap-M1')
            m.ch_names = [n.upper() for n in m.ch_names]
            m.ch_names[m.ch_names.index('TP9')] = 'A1'
            m.ch_names[m.ch_names.index('TP10')] = 'A2'
            m.ch_names[m.ch_names.index('FP2')] = 'VEOGt'
            ndvar.sensor.set_sensor_positions(m)
        # reference
        reference = self.get('reference')
        if reference:
            if reference == 'mastoids':
                ndvar -= ndvar.summary(sensor=['A1', 'A2'])
            else:
                raise ValueError("Unknown reference: reference=%r" % reference)

    def get_field_values(self, field, exclude=False):
        """Find values for a field taking into account exclusion

        Parameters
        ----------
        field : str
            Field for which to find values.
        exclude : bool | list of values
            Exclude values. If True, exclude values based on ``self.exclude``.
            For 'mrisubject', exclusions are done on 'subject'. For 'group',
            no exclusions are done.
        """
        if exclude is True:
            exclude = self.exclude.get(field, None)
        elif isinstance(exclude, basestring):
            exclude = (exclude,)

        if field == 'mrisubject':
            subjects = FileTree.get_field_values(self, 'subject')
            mri_subjects = self._mri_subjects[self.get('mri')]
            mrisubjects = sorted(mri_subjects[s] for s in subjects)
            if exclude:
                mrisubjects = [s for s in mrisubjects if s not in exclude]
            common_brain = self.get('common_brain')
            if common_brain and (not exclude or common_brain not in exclude):
                mrisubjects.insert(0, common_brain)
            return mrisubjects
        else:
            return FileTree.get_field_values(self, field, exclude)

    def iter(self, fields='subject', exclude=True, values={}, group=None,
             **kwargs):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : sequence | str
            Field(s) over which should be iterated.
        exclude : dict  {str: str, str: iterator over str, ...}
            Values to exclude from the iteration with {name: value} and/or
            {name: (sequence of values, )} entries.
        values : dict  {str: iterator over str}
            Fields with custom values to iterate over (instead of the
            corresponding field values) with {name: (sequence of values)}
            entries.
        group : None | str
            If iterating over subjects, use this group ('all' for all except
            excluded subjects, 'all!' for all including excluded subjects, or
            a name defined in experiment.groups).
        ...
            Fields with constant values throughout the iteration.
        """
        if group is not None:
            kwargs['group'] = group
        return FileTree.iter(self, fields, exclude, values, **kwargs)

    def iter_range(self, start=None, stop=None, field='subject'):
        """Iterate through a range on a field with ordered values.

        Parameters
        ----------
        start : None | str
            Start value (inclusive). With ``None``, begin at the first value.
        stop : None | str
            Stop value (inclusive). With ``None``, end with the last value.
        field : str
            Name of the field.

        Returns
        -------
        iterator over value : str
            Current field value.
        """
        values = self.get_field_values(field)
        if start is not None:
            start = values.index(start)
        if stop is not None:
            stop = values.index(stop) + 1
        idx = slice(start, stop)
        values = values[idx]

        with self._temporary_state:
            for value in values:
                self._restore_state(discard_tip=False)
                self.set(**{field: value})
                yield value

    def label_events(self, ds):
        """Add event labels to events loaded from raw files

        Parameters
        ----------
        ds : Dataset
            A Dataset containing events (with variables as returned by
            :func:`load.fiff.events`).

        Notes
        -----
        Override this method in MneExperiment subclasses to add event labels.
        The session the events are from can be determined with
        ``ds.info['session']``.
        Call the original (super-class) method to add T (time) and SOA (stimulus
        onset asynchrony) to the Dataset.
        """
        ds['T'] = ds['i_start'] / ds.info['sfreq']
        ds['SOA'] = ds['T'].diff(0)
        if len(self._sessions) > 1:
            ds[:, 'session'] = ds.info['session']

        for name, coding in self.variables.iteritems():
            ds[name] = ds['trigger'].as_factor(coding, name)

        # add subject label
        ds['subject'] = Factor([ds.info['subject']], repeat=ds.n_cases, random=True)
        return ds

    def label_subjects(self, ds):
        """Label the subjects in ds based on .groups

        Parameters
        ----------
        ds : Dataset
            A Dataset with 'subject' entry.
        """
        subject = ds['subject']
        for name, subjects in self._groups.iteritems():
            ds[name] = Var(subject.isin(subjects))

    def load_annot(self, **state):
        """Load a parcellation (from an annot file)

        Returns
        -------
        labels : list of Label
            Labels in the parcellation (output of
            :func:`mne.read_labels_from_annot`).
        ...
            State parameters.
        """
        self.make_annot(**state)
        return mne.read_labels_from_annot(self.get('mrisubject'),
                                          self.get('parc'), 'both',
                                          subjects_dir=self.get('mri-sdir'))

    def load_bad_channels(self, **kwargs):
        """Load bad channels
        
        Parameters
        ----------
        ...
            State parameters.

        Returns
        -------
        bad_chs : list of str
            Bad chnnels.
        """
        pipe = self._raw[self.get('raw', **kwargs)]
        return pipe.load_bad_channels(self.get('subject'), self.get('session'))

    def _load_bem(self):
        subject = self.get('mrisubject')
        if subject == 'fsaverage' or is_fake_mri(self.get('mri-dir')):
            return mne.read_bem_surfaces(self.get('bem-file'))
        else:
            bem_dir = self.get('bem-dir')
            surfs = ('inner_skull', 'outer_skull', 'outer_skin')
            paths = {s: join(bem_dir, s + '.surf') for s in surfs}
            missing = [s for s in surfs if not exists(paths[s])]
            if missing:
                bem_dir = self.get('bem-dir')
                temp = join(".*", "bem", "(.*)")
                for surf in missing[:]:
                    path = paths[surf]
                    if os.path.islink(path):
                        # try to fix broken symlinks
                        old_target = os.readlink(path)
                        m = re.match(temp, old_target)
                        if m:
                            new_target = m.group(1)
                            if exists(join(bem_dir, new_target)):
                                self._log.info("Fixing broken symlink for %s "
                                               "%s surface file", subject, surf)
                                os.unlink(path)
                                os.symlink(new_target, path)
                                missing.remove(surf)
                        #         continue
                        # self._log.info("Deleting broken symlink " + path)
                        # os.unlink(path)
                if missing:
                    self._log.info("%s %s missing for %s. Running "
                                   "mne.make_watershed_bem()...",
                                   enumeration(missing).capitalize(),
                                   plural('surface', len(missing)), subject)
                    # re-run watershed_bem
                    # mne-python expects the environment variable
                    os.environ['FREESURFER_HOME'] = subp.get_fs_home()
                    mne.bem.make_watershed_bem(subject, self.get('mri-sdir'),
                                               overwrite=True)

            return mne.make_bem_model(subject, conductivity=(0.3,),
                                      subjects_dir=self.get('mri-sdir'))

    def load_cov(self, **kwargs):
        """Load the covariance matrix

        Parameters
        ----------
        ...
            State parameters.
        """
        return mne.read_cov(self.get('cov-file', make=True, **kwargs))

    def load_edf(self, **kwargs):
        """Load the edf file ("edf-file" template)
        
        Parameters
        ----------
        ...
            State parameters.
        """
        path = self.get('edf-file', fmatch=False, **kwargs)
        return load.eyelink.Edf(path)

    @staticmethod
    def _ndvar_name_for_modality(modality):
        if modality == 'meeg':
            raise NotImplementedError("NDVar for sensor space data combining "
                                      "EEG and MEG data")
        elif modality == '':
            return 'meg'
        elif modality == 'eeg':
            return 'eeg'
        else:
            raise ValueError("modality=%r" % modality)

    def _sysname(self, fiff, subject, modality):
        if fiff.info.get('kit_system_id'):
            try:
                return KIT_NEIGHBORS[fiff.info['kit_system_id']]
            except KeyError:
                raise NotImplementedError("Unknown KIT system-ID: %r" %
                                          (fiff.info['kit_system_id'],))
        elif modality != '':
            return  # handled in self._fix_eeg_ndvar()
        if isinstance(self.meg_system, str):
            return self.meg_system
        subject_prefix = self._subject_re.match(subject).group(1)
        if isinstance(self.meg_system, dict):
            return self.meg_system.get(subject_prefix)
        elif self.meg_system is not None:
            raise TypeError("MneExperiment.meg_system needs to be a str or a "
                            "dict, not %s" % repr(self.meg_system))
        # go by nothing but subject name
        if subject_prefix.startswith('A'):
            return 'KIT-208'
        else:
            raise RuntimeError("Unknown MEG system encountered. Please set "
                               "MneExperiment.meg_system.")

    @staticmethod
    def _data_arg(modality, eog=False):
        "Data argument for FIFF-to-NDVar conversion"
        if modality == 'meeg':
            raise NotImplementedError("NDVar for sensor space data combining "
                                      "EEG and MEG data")
        elif modality == '':
            return 'mag'
        elif modality == 'eeg':
            if eog:
                return 'eeg&eog'
            else:
                return 'eeg'
        else:
            raise ValueError("modality=%r" % modality)

    def load_epochs(self, subject=None, baseline=False, ndvar=True,
                    add_bads=True, reject=True, cat=None,
                    decim=None, pad=0, data_raw=False, vardef=None,
                    eog=False, trigger_shift=True, apply_ica=True, tmin=None,
                    tmax=None, tstop=None, **kwargs):
        """
        Load a Dataset with epochs for a given epoch definition

        Parameters
        ----------
        subject : str
            Subject(s) for which to load epochs. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        baseline : bool | tuple
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to not apply baseline
            correction.
        ndvar : bool | 'both'
            Convert epochs to an NDVar (named 'meg' for MEG data and 'eeg' for
            EEG data). Use 'both' to include NDVar and MNE Epochs.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        reject : bool
            Whether to apply epoch rejection or not. The kind of rejection
            employed depends on the ``rej`` setting.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        decim : int
            Data decimation factor (the default is the factor specified in the
            epoch definition).
        pad : scalar
            Pad the epochs with this much time (in seconds; e.g. for spectral
            analysis).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        vardef : str
            Name of a 2-stage test defining additional variables.
        eog : bool
            When loading EEG data as NDVar, also add the EOG channels.
        trigger_shift : bool
            Apply post-baseline trigger-shift if it applies to the epoch
            (default True).
        apply_ica : bool
            If the current rej setting uses ICA, remove the excluded ICA
            components from the Epochs.
        tmin : scalar
            Override the epoch's ``tmin`` parameter.
        tmax : scalar
            Override the epoch's ``tmax`` parameter.
        tstop : scalar
            Override the epoch's ``tmax`` parameter as exclusive ``tstop``.
        ...
            State parameters.
        """
        if ndvar:
            if isinstance(ndvar, basestring):
                if ndvar != 'both':
                    raise ValueError("ndvar=%s" % repr(ndvar))
        subject, group = self._process_subject_arg(subject, kwargs)

        if group is not None:
            dss = []
            for _ in self.iter(group=group):
                ds = self.load_epochs(None, baseline, ndvar, add_bads, reject,
                                      cat, decim, pad, data_raw, vardef,
                                      tmin=tmin, tmax=tmax, tstop=tstop)
                dss.append(ds)

            return combine(dss)
        elif self.get('modality') == 'meeg':  # single subject, combine MEG and EEG
            # FIXME: combine MEG/EEG based on different pipes
            with self._temporary_state:
                ds_meg = self.load_epochs(subject, baseline, ndvar, add_bads,
                                          reject, cat, decim, pad, data_raw,
                                          vardef, tmin=tmin, tmax=tmax,
                                          tstop=tstop, modality='')
                ds_eeg = self.load_epochs(subject, baseline, ndvar, add_bads,
                                          reject, cat, decim, pad, data_raw,
                                          vardef, tmin=tmin, tmax=tmax,
                                          tstop=tstop, modality='eeg')
            ds, eeg_epochs = align(ds_meg, ds_eeg['epochs'], 'index',
                                   ds_eeg['index'])
            ds['epochs'] = mne.epochs.add_channels_epochs((ds['epochs'], eeg_epochs))
            return ds
        # single subject, single modality
        epoch = self._epochs[self.get('epoch')]
        with self._temporary_state:
            ds = self.load_selected_events(add_bads=add_bads, reject=reject,
                                           data_raw=data_raw or True,
                                           vardef=vardef, cat=cat)
            if ds.n_cases == 0:
                err = ("No events left for epoch=%r, subject=%r" %
                       (epoch.name, subject))
                if cat:
                    err += ", cat=%s" % repr(cat)
                raise RuntimeError(err)

            # load sensor space data
            ds = self._add_epochs(ds, epoch, baseline, ndvar, data_raw, pad,
                                  decim, reject, apply_ica, trigger_shift, eog,
                                  tmin, tmax, tstop)

        return ds

    def load_epochs_stc(self, subject=None, sns_baseline=True,
                        src_baseline=False, ndvar=True, cat=None,
                        keep_epochs=False, morph=False, mask=False,
                        data_raw=False, vardef=None, decim=None, **kwargs):
        """Load a Dataset with stcs for single epochs

        Parameters
        ----------
        subject : str
            Subject(s) for which to load epochs. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        ndvar : bool
            Add the source estimates as NDVar named "src" instead of a list of
            SourceEstimate objects named "stc" (default True).
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        keep_epochs : bool
            Keep the sensor space data in the Dataset that is returned (default
            False).
        morph : bool
            Morph the source estimates to the common_brain (default False).
        mask : bool | str
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        vardef : str
            Name of a 2-stage test defining additional variables.
        decim : None | int
            Set to an int in order to override the epoch decim factor.
        ...
            State parameters.

        Returns
        -------
        epochs_dataset : Dataset
            Dataset containing single trial data (epochs).
        """
        if not sns_baseline and src_baseline and \
                self._epochs[self.get('epoch')].post_baseline_trigger_shift:
            raise NotImplementedError("post_baseline_trigger_shift is not "
                                      "implemented for baseline correction in "
                                      "source space")

        subject, group = self._process_subject_arg(subject, kwargs)
        if group is not None:
            if data_raw is not False:
                raise ValueError("Can not keep data_raw when combining data "
                                 "from multiple subjects. Set data_raw=False "
                                 "(default).")
            elif keep_epochs:
                raise ValueError("Can not combine Epochs objects for different "
                                 "subjects. Set keep_epochs=False (default).")
            elif not morph:
                raise ValueError("Source estimates can only be combined after "
                                 "morphing data to common brain model. Set "
                                 "morph=True.")
            dss = []
            for _ in self.iter(group=group):
                ds = self.load_epochs_stc(None, sns_baseline, src_baseline,
                                          ndvar, cat, keep_epochs, morph, mask,
                                          False, vardef, decim)
                dss.append(ds)
            return combine(dss)
        else:
            ds = self.load_epochs(subject, sns_baseline, False, cat=cat,
                                  decim=decim, data_raw=data_raw, vardef=vardef)
            self._add_epochs_stc(ds, ndvar, src_baseline, morph, mask)
            if not keep_epochs:
                del ds['epochs']
            return ds

    def load_events(self, subject=None, add_bads=True, data_raw=True, **kwargs):
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject : str (state)
            Subject for which to load events.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default True).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        ...
            State parameters.
        """
        evt_file = self.get('event-file', mkdir=True, subject=subject, **kwargs)

        # search for and check cached version
        ds = None
        if exists(evt_file):
            if getmtime(evt_file) > self._raw_mtime():
                ds = load.unpickle(evt_file)
                #  <0.19 cache
                if 'sfreq' not in ds.info:
                    ds = None

        # refresh cache
        subject = self.get('subject')
        if ds is None:
            self._log.debug("Extracting events for %s %s", subject,
                            self.get('session'))
            if self.get('modality') == '':
                merge = -1
            else:
                merge = 0
            raw = self.load_raw(add_bads, subject=subject)
            ds = load.fiff.events(raw, merge)
            del ds.info['raw']
            ds.info['sfreq'] = raw.info['sfreq']

            # add edf
            if self.has_edf[subject]:
                edf = self.load_edf()
                edf.add_t_to(ds)
                ds.info['edf'] = edf

            save.pickle(ds, evt_file)
        elif data_raw is True:
            raw = self.load_raw(add_bads, subject=subject)

        # if data should come from different raw settings than events
        if isinstance(data_raw, str):
            with self._temporary_state:
                raw = self.load_raw(add_bads, subject=subject, raw=data_raw)
        elif not isinstance(data_raw, bool):
            raise TypeError("data_raw=%s; needs to be str or bool"
                            % repr(data_raw))

        ds.info['subject'] = subject
        ds.info['session'] = self.get('session')
        if data_raw is not False:
            ds.info['raw'] = raw

        if self.trigger_shift:
            if isinstance(self.trigger_shift, dict):
                trigger_shift = self.trigger_shift[subject]
            else:
                trigger_shift = self.trigger_shift

            if trigger_shift:
                ds['i_start'] += int(round(trigger_shift * ds.info['sfreq']))

        # label events
        ds = self.label_events(ds)
        if ds is None:
            raise RuntimeError("The MneExperiment.label_events() function must "
                               "return the events-Dataset")
        return ds

    def load_evoked(self, subject=None, baseline=False, ndvar=True, cat=None,
                    decim=None, data_raw=False, vardef=None, **kwargs):
        """
        Load a Dataset with the evoked responses for each subject.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        baseline : bool | tuple
            Apply baseline correction using this period. True to use the
            epoch's baseline specification. The default is to not apply baseline
            correction.
        ndvar : bool
            Convert the mne Evoked objects to an NDVar (the name in the
            Dataset is 'meg' or 'eeg').
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        decim : int
            Data decimation factor (the default is the factor specified in the
            epoch definition).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        vardef : str
            Name of a 2-stage test defining additional variables.
        model : str (state)
            Model according to which epochs are grouped into evoked responses.
        ...
            State parameters.
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        modality = self.get('modality')
        epoch = self._epochs[self.get('epoch')]
        if baseline is True:
            baseline = epoch.baseline

        if group is not None:
            dss = [self.load_evoked(None, baseline, False, cat, decim, data_raw,
                                    vardef)
                   for _ in self.iter(group=group)]
            if ndvar:
                sysnames = set(ds.info['sysname'] for ds in dss)
                if len(sysnames) != 1:
                    err = ("Can not combine different MEG systems in a single "
                           "NDVar (trying to load data with systems %s)" %
                           enumeration(sysnames))
                    raise NotImplementedError(err)
            ds = combine(dss, incomplete='drop')

            # check consistency in MNE objects' number of time points
            lens = [len(e.times) for e in ds['evoked']]
            ulens = set(lens)
            if len(ulens) > 1:
                err = ["Unequal time axis sampling (len):"]
                alens = np.array(lens)
                for l in ulens:
                    err.append('%i: %r' % (l, ds['subject', alens == l].cells))
                raise DimensionMismatchError('\n'.join(err))
        else:  # single subject
            ds = self._make_evoked(decim, data_raw)

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)
                if ds.n_cases == 0:
                    raise RuntimeError("Selection with cat=%s resulted in "
                                       "empty Dataset" % repr(cat))

            self._add_vars(ds, vardef)

            # baseline correction
            if isinstance(baseline, str):
                raise NotImplementedError
            elif baseline and not epoch.post_baseline_trigger_shift:
                for e in ds['evoked']:
                    rescale(e.data, e.times, baseline, 'mean', copy=False)

            # info
            ds.info['sysname'] = self._sysname(ds[0, 'evoked'], subject, modality)

        # convert to NDVar
        if ndvar:
            name = self._ndvar_name_for_modality(modality)
            ds[name] = load.fiff.evoked_ndvar(ds['evoked'],
                                              data=self._data_arg(modality),
                                              sysname=ds.info['sysname'])
            if modality == 'eeg':
                self._fix_eeg_ndvar(ds[name], group)

        return ds

    def load_epochs_stf(self, subject=None, sns_baseline=True, mask=True,
                        morph=False, keep_stc=False, **kwargs):
        """Load frequency space single trial data

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        mask : bool | str
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default True).
        morph : bool
            Morph the source estimates to the common_brain (default False).
        keep_stc : bool
            Keep the source timecourse data in the Dataset that is returned
            (default False).
        ...
            State parameters.
        """
        ds = self.load_epochs_stc(subject, sns_baseline, ndvar=True,
                                  morph=morph, mask=mask, data_raw='raw',
                                  **kwargs)
        name = 'srcm' if morph else 'src'

        # apply morlet transformation
        freq_params = self.freqs[self.get('freq')]
        freq_range = freq_params['frequencies']
        ds['stf'] = cwt_morlet(ds[name], freq_range, use_fft=True,
                               n_cycles=freq_params['n_cycles'],
                               zero_mean=False, out='magnitude')

        if not keep_stc:
            del ds[name]

        return ds

    def load_evoked_stf(self, subject=None, sns_baseline=True, mask=True,
                        morph=False, keep_stc=False, **kwargs):
        """Load frequency space evoked data

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        sns_baseline : None | True | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        mask : bool | str
            Whether to just load the sources from the parcellation that are not
            defined as "unknown". Default is True.
        morph : bool
            Morph the source estimates to the common_brain (default False).
        keep_stc : bool
            Keep the source timecourse data in the Dataset that is returned
            (default False).
        ...
            State parameters.
        """
        ds = self.load_evoked_stc(subject, sns_baseline, morph_ndvar=morph,
                                  ind_ndvar=not morph, mask=mask,
                                  data_raw='raw', **kwargs)
        name = 'srcm' if morph else 'src'

        # apply morlet transformation
        freq_params = self.freqs[self.get('freq')]
        freq_range = freq_params['frequencies']
        ds['stf'] = cwt_morlet(ds[name], freq_range, use_fft=True,
                               n_cycles=freq_params['n_cycles'],
                               zero_mean=False, out='magnitude')

        if not keep_stc:
            del ds[name]

        return ds

    def load_evoked_stc(self, subject=None, sns_baseline=True,
                        src_baseline=False, sns_ndvar=False, ind_stc=False,
                        ind_ndvar=False, morph_stc=False, morph_ndvar=False,
                        cat=None, keep_evoked=False, mask=False, data_raw=False,
                        vardef=None, **kwargs):
        """Load evoked source estimates.

        Parameters
        ----------
        subject : str
            Subject(s) for which to load evoked files. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification. The default is True.
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        ind_stc : bool
            Add source estimates on individual brains as list of
            :class:`mne.SourceEstimate`.
        ind_ndvar : bool
            Add source estimates on individual brains as :class:`NDVar`.
        morph_stc : bool
            Add source estimates morphed to the common brain as list of
            :class:`mne.SourceEstimate`.
        morph_ndvar : bool
            Add source estimates morphed to the common brain as :class:`NDVar`.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        keep_evoked : bool
            Keep the sensor space data in the Dataset that is returned (default
            False).
        mask : bool | str
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False). Can be set to a parcellation
            name or ``True`` to use the current parcellation.
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        vardef : str
            Name of a 2-stage test defining additional variables.
        ...
            State parameters.
        """
        if not any((ind_stc, ind_ndvar, morph_stc, morph_ndvar)):
            err = ("Nothing to load, set at least one of (ind_stc, ind_ndvar, "
                   "morph_stc, morph_ndvar) to True")
            raise ValueError(err)

        if kwargs:
            self.set(**kwargs)

        if not sns_baseline and src_baseline and \
                self._epochs[self.get('epoch')].post_baseline_trigger_shift:
            raise NotImplementedError("post_baseline_trigger_shift is not "
                                      "implemented for baseline correction in "
                                      "source space")

        ds = self.load_evoked(subject, sns_baseline, sns_ndvar, cat, None,
                              data_raw, vardef)
        self._add_evoked_stc(ds, ind_stc, ind_ndvar, morph_stc, morph_ndvar,
                             src_baseline, keep_evoked, mask)

        return ds

    def load_fwd(self, surf_ori=True, ndvar=False, mask=None):
        """Load the forward solution

        Parameters
        ----------
        surf_ori : bool
            Force surface orientation (default True; only applies if
            ``ndvar=False``, :class:`NDVar` forward operators are alsways
            surface based).
        ndvar : bool
            Return forward solution as :class:`NDVar` (default is
            :class:`mne.forward.Forward`).
        mask : str | bool
            Remove source labelled "unknown". Can be parcellation name or True,
            in which case the current parcellation is used.

        Returns
        -------
        forward_operator : mne.forward.Forward | NDVar
            Forward operator.
        """
        if mask and not ndvar:
            raise NotImplemented("mask is only implemented for ndvar=True")
        elif isinstance(mask, basestring):
            self.set(parc=mask)
            mask = True
        fwd_file = self.get('fwd-file', make=True)
        if ndvar:
            self.make_annot()
            fwd = load.fiff.forward_operator(
                fwd_file, self.get('src'), self.get('mri-sdir'), self.get('parc'))
            if mask:
                fwd = fwd.sub(source=np.invert(
                    fwd.source.parc.startswith('unknown')))
            return fwd
        else:
            return mne.read_forward_solution(fwd_file, surf_ori=surf_ori)

    def load_ica(self):
        """Load the ICA object for the current subject/rej setting

        Returns
        -------
        ica : mne.preprocessing.ICA
            ICA object for the current subject/rej setting.
        """
        pipe = self._raw[self.get('raw')]
        if isinstance(pipe, RawICA):
            return pipe.load_ica(self.get('subject'))
        path = self.get('ica-file')
        if not exists(path):
            raise RuntimeError("ICA file does not exist at %s. Run "
                               "e.make_ica_selection() to create it." %
                               relpath(path, self.get('root')))
        return mne.preprocessing.read_ica(path)

    def load_inv(self, fiff=None, ndvar=False, mask=None, **kwargs):
        """Load the inverse operator

        Parameters
        ----------
        fiff : Raw | Epochs | Evoked | ...
            Object which provides the mne info dictionary (default: load the
            raw file).
        ndvar : bool
            Return the inverse operator as NDVar (default is 
            :class:`mne.minimum_norm.InverseOperator`). The NDVar representation 
            does not take into account any direction selectivity (loose/free 
            orientation) or noise normalization properties.
        mask : str | bool
            Remove source labelled "unknown". Can be parcellation name or True,
            in which case the current parcellation is used.
        ...
            State parameters.
        """
        if self.get('modality', **kwargs) != '':
            raise NotImplementedError("Source reconstruction for EEG data")
        elif mask and not ndvar:
            raise NotImplemented("mask is only implemented for ndvar=True")
        elif isinstance(mask, basestring):
            self.set(parc=mask)
            mask = True

        if fiff is None:
            fiff = self.load_raw()

        fwd = self.load_fwd()
        cov = self.load_cov()
        kwargs = self._params['make_inv_kw']
        inv = make_inverse_operator(fiff.info, fwd, cov, **kwargs)

        if ndvar:
            inv = load.fiff.inverse_operator(
                inv, self.get('src'), self.get('mri-sdir'), self.get('parc'))
            if mask:
                inv = inv.sub(source=np.invert(
                    inv.source.parc.startswith('unknown')))
        return inv

    def load_label(self, label, **kwargs):
        """Retrieve a label as mne Label object

        Parameters
        ----------
        label : str
            Name of the label. If the label name does not end in '-bh' or '-rh'
            the combination of the labels ``label + '-lh'`` and
            ``label + '-rh'`` is returned.
        ...
            State parameters.
        """
        labels = self._load_labels(label, **kwargs)
        if label in labels:
            return labels[label]
        elif not label.endswith(('-lh', '-rh')):
            return labels[label + '-lh'] + labels[label + '-rh']
        else:
            raise ValueError("Label %r could not be found in parc %r."
                             % (label, self.get('parc')))

    def _load_labels(self, regexp=None, **kwargs):
        """Load labels from an annotation file."""
        self.make_annot(**kwargs)
        mri_sdir = self.get('mri-sdir')
        labels = mne.read_labels_from_annot(self.get('mrisubject'),
                                            self.get('parc'), regexp=regexp,
                                            subjects_dir=mri_sdir)
        return {l.name: l for l in labels}

    def load_morph_matrix(self, **state):
        """Load the morph matrix from mrisubject to common_brain

        Parameters
        ----------
        ...
            State parameters.

        Returns
        -------
        mm : sparse matrix
            Morph matrix.
        vertices_to : list of 2 array
            Vertices of the morphed data.
        """
        subjects_dir = self.get('mri-sdir', **state)
        subject_to = self.get('common_brain')
        subject_from = self.get('mrisubject')

        src_to = self.load_src(mrisubject=subject_to, match=False)
        src_from = self.load_src(mrisubject=subject_from, match=False)

        vertices_to = [src_to[0]['vertno'], src_to[1]['vertno']]
        vertices_from = [src_from[0]['vertno'], src_from[1]['vertno']]

        mm = mne.compute_morph_matrix(subject_from, subject_to, vertices_from,
                                      vertices_to, None, subjects_dir)
        return mm, vertices_to

    def load_raw(self, add_bads=True, preload=False, ndvar=False, decim=1, **kwargs):
        """
        Load a raw file as mne Raw object.

        Parameters
        ----------
        add_bads : bool
            Add bad channel information to the bad channels text file (default
            True).
        preload : bool
            Mne Raw parameter.
        ndvar : bool
            Load as NDVar instead of mne Raw object (default False).
        decim : int
            Decimate data (implies preload=True; default 1, i.e. no decimation)
        ...
            State parameters.

        Notes
        -----
        Bad channels defined in the raw file itself are ignored in favor of the
        bad channels in the bad channels file.
        """
        if not isinstance(add_bads, int):
            raise TypeError("add_bads must be boolean, got %s" % repr(add_bads))
        pipe = self._raw[self.get('raw', **kwargs)]
        raw = pipe.load(self.get('subject'), self.get('session'), add_bads,
                        preload)
        if decim > 1:
            sfreq = int(round(raw.info['sfreq'] / decim))
            raw.load_data()
            raw.resample(sfreq)

        if ndvar:
            raw = load.fiff.raw_ndvar(raw)

        return raw

    def _load_result_plotter(self, test, tstart, tstop, pmin, parc=None,
                             mask=None, samples=10000, data='source',
                             sns_baseline=True, src_baseline=None,
                             colors=None, labels=None, h=1.2, rc=None,
                             dst=None, vec_fmt='svg', pix_fmt='png', **kwargs):
        """Load cluster-based test result plotter

        Parameters
        ----------
        test : str
            Name of the test.
        tstart, tstop, pmin, parc, mask, samples, data, sns_baseline, src_baseline
            Test parameters.
        colors : dict
            Colors for data cells as ``{cell: matplotlib_color}`` dictionary.
        labels : dict
            Labels for data in a ``{cell: label}`` dictionary (the default is to
            use cell names).
        h : scalar
            Plot height in inches (default 1.1).
        rc : dict
            Matplotlib rc-parameters dictionary (the default is optimized for
            the default plot size ``h=1.1``).
        dst : str
            Directory in which to place results (default is the ``result plots``
            directory).
        vec_fmt : str
            Format for vector graphics (default 'pdf').
        pix_fmt : str
            Format for pixel graphics (default 'png').
        ...
            State parameters.
        """
        if not isinstance(self._tests[test], EvokedTest):
            raise NotImplementedError("Result-plots for %s" %
                                      self._tests[test].__class__.__name__)
        elif data != 'source':
            raise NotImplementedError("data=%s" % repr(data))
        elif not isinstance(pmin, float):
            raise NotImplementedError("Threshold-free tests")

        from .._result_plots import ClusterPlotter

        # calls _set_analysis_options():
        ds, res = self.load_test(test, tstart, tstop, pmin, parc, mask, samples,
                                 data, sns_baseline, src_baseline, True,
                                 **kwargs)
        if dst is None:
            dst = self.get('res-plot-dir', mkdir=True)

        return ClusterPlotter(ds, res, colors, dst, vec_fmt, pix_fmt, labels, h,
                              rc)

    def load_selected_events(self, subject=None, reject=True, add_bads=True,
                             index=True, data_raw=False, vardef=None, cat=None,
                             **kwargs):
        """
        Load events and return a subset based on epoch and rejection

        Parameters
        ----------
        subject : str
            Subject(s) for which to load events. Can be a single subject
            name or a group name such as 'all'. The default is the current
            subject in the experiment's state.
        reject : bool | 'keep'
            Reject bad trials. For True, bad trials are removed from the
            Dataset. For 'keep', the 'accept' variable is added to the Dataset
            and bad trials are kept.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be specified.
        index : bool | str
            Index the Dataset before rejection (provide index name as str).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        vardef : str
            Name of a 2-stage test defining additional variables.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        ...
            State parameters.

        Notes
        -----
        When trial rejection is set to automatic, not rejection is performed
        because no epochs are loaded.
        """
        # process arguments
        if reject not in (True, False, 'keep'):
            raise ValueError("reject=%s" % repr(reject))

        if index is True:
            index = 'index'
        elif index and not isinstance(index, str):
            raise TypeError("index=%s" % repr(index))

        # case of loading events for a group
        subject, group = self._process_subject_arg(subject, kwargs)
        if group is not None:
            if data_raw is not False:
                raise ValueError("data_var=%s: can't load data raw when "
                                 "combining different subjects" % repr(data_raw))
            dss = [self.load_selected_events(reject=reject, add_bads=add_bads,
                                             index=index, vardef=vardef)
                   for _ in self.iter(group=group)]
            ds = combine(dss)
            return ds

        epoch = self._epochs[self.get('epoch')]

        # rejection comes from somewhere else
        if isinstance(epoch, SuperEpoch):
            with self._temporary_state:
                dss = []
                raw = None
                # find bad channels
                if add_bads:
                    bad_channels = sorted(set.union(*(
                        set(self.load_bad_channels(session=session)) for
                        session in epoch.sessions)))
                else:
                    bad_channels = []
                # load events
                for session in epoch.sessions:
                    self.set(session=session)
                    # load events for this session
                    session_dss = []
                    for sub_epoch in epoch.sub_epochs:
                        if self._epochs[sub_epoch].session != session:
                            continue
                        ds = self.load_selected_events(
                            subject, reject, add_bads, index, data_raw or True,
                            epoch=sub_epoch)
                        ds[:, 'epoch'] = sub_epoch
                        session_dss.append(ds)
                    ds = combine(session_dss)
                    # combine raw
                    if raw is None:
                        raw = ds.info['raw']
                        raw.info['bads'] = bad_channels
                    else:
                        raw_ = ds.info['raw']
                        raw_.info['bads'] = bad_channels
                        ds['i_start'] += raw.last_samp + 1 - raw_.first_samp
                        raw.append(raw_)
                    del ds.info['raw']
                    dss.append(ds)

            # combine bad channels
            ds = combine(dss)
            if data_raw is not False:
                ds.info['raw'] = raw
            ds.info[BAD_CHANNELS] = bad_channels
        elif isinstance(epoch, SecondaryEpoch):
            with self._temporary_state:
                ds = self.load_selected_events(None, 'keep' if reject else False,
                                               add_bads, index, data_raw,
                                               epoch=epoch.sel_epoch)

            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if index:
                ds.index(index)

            if reject is True:
                if self._artifact_rejection[self.get('rej')]['kind'] is not None:
                    ds = ds.sub('accept')
        else:
            rej_params = self._artifact_rejection[self.get('rej')]
            # load files
            with self._temporary_state:
                ds = self.load_events(add_bads=add_bads, data_raw=data_raw,
                                      session=epoch.session)
                if reject and rej_params['kind'] is not None:
                    rej_file = self.get('rej-file')
                    if exists(rej_file):
                        ds_sel = load.unpickle(rej_file)
                    else:
                        raise FileMissing("The rejection file at %s does not "
                                          "exist. Run .make_rej() first." %
                                          self._get_rel('rej-file', 'root'))
                else:
                    ds_sel = None

            # primary event selection
            if epoch.sel:
                ds = ds.sub(epoch.sel)
            if index:
                ds.index(index)
            if epoch.n_cases is not None and ds.n_cases != epoch.n_cases:
                raise RuntimeError("Number of epochs %i, expected %i" %
                                   (ds.n_cases, epoch.n_cases))

            # rejection
            if ds_sel is not None:
                # check file
                if not np.all(ds['trigger'] == ds_sel['trigger']):
                    #  TODO:  this warning should be given in make_rej already
                    if np.all(ds[:-1, 'trigger'] == ds_sel['trigger']):
                        ds = ds[:-1]
                        self._log.warn(self.format("Last epoch for {subject} is missing"))
                    elif np.all(ds[1:, 'trigger'] == ds_sel['trigger']):
                        ds = ds[1:]
                        self._log.warn(self.format("First epoch for {subject} is missing"))
                    else:
                        raise RuntimeError(
                            "The epoch selection file contains different events (trigger IDs) "
                            "from the epoch data loaded from the raw file. If the "
                            "events included in the epoch were changed intentionally, "
                            "delete the corresponding trial rejection file and create a new "
                            "one:\n %s" % (rej_file,))

                if rej_params['interpolation']:
                    ds.info[INTERPOLATE_CHANNELS] = True
                    if INTERPOLATE_CHANNELS in ds_sel:
                        ds[INTERPOLATE_CHANNELS] = ds_sel[INTERPOLATE_CHANNELS]
                    else:
                        ds[INTERPOLATE_CHANNELS] = Datalist([[]] * ds.n_cases,
                                                            INTERPOLATE_CHANNELS,
                                                            'strlist')
                else:
                    ds.info[INTERPOLATE_CHANNELS] = False

                # subset events
                if reject == 'keep':
                    ds['accept'] = ds_sel['accept']
                elif reject is True:
                    ds = ds.sub(ds_sel['accept'])
                else:
                    raise RuntimeError("reject=%s" % repr(reject))

                # bad channels
                if add_bads:
                    if BAD_CHANNELS in ds_sel.info:
                        ds.info[BAD_CHANNELS] = ds_sel.info[BAD_CHANNELS]
                    else:
                        ds.info[BAD_CHANNELS] = []
            else:  # no artifact rejection
                ds.info[INTERPOLATE_CHANNELS] = False
                ds.info[BAD_CHANNELS] = []

        # apply trigger-shift
        if epoch.trigger_shift:
            shift = epoch.trigger_shift
            if isinstance(shift, basestring):
                shift = ds.eval(shift)
            if isinstance(shift, Var):
                shift = shift.x

            if np.isscalar(shift):
                ds['i_start'] += int(round(shift * ds.info['sfreq']))
            else:
                ds['i_start'] += np.round(shift * ds.info['sfreq']).astype(int)

        # Additional variables
        self._add_vars(ds, epoch.vars)
        self._add_vars(ds, vardef)

        # apply cat subset
        if cat:
            model = ds.eval(self.get('model'))
            idx = model.isin(cat)
            ds = ds.sub(idx)

        return ds

    def _load_spm(self, sns_baseline=True, src_baseline=False):
        "Load LM"
        subject = self.get('subject')
        test = self.get('test')
        test_obj = self._tests[test]
        if not isinstance(test_obj, TwoStageTest):
            raise NotImplementedError("Test kind %r" % test_obj.__class__.__name__)
        ds = self.load_epochs_stc(subject, sns_baseline, src_baseline, mask=True,
                                  vardef=test_obj.vars)
        return testnd.LM('src', test_obj.stage_1, ds, subject=subject)

    def load_src(self, add_geom=False, **state):
        """Load the current source space
        
        Parameters
        ----------
        add_geom : bool
            Parameter for :func:`mne.read_source_spaces`.
        ...
            State parameters.
        """
        fpath = self.get('src-file', make=True, **state)
        return mne.read_source_spaces(fpath, add_geom)

    def load_test(self, test, tstart, tstop, pmin, parc=None, mask=None,
                  samples=10000, data='source', sns_baseline=True,
                  src_baseline=None, return_data=False, make=False, **kwargs):
        """Create and load spatio-temporal cluster test results

        Parameters
        ----------
        test : None | str
            Test for which to create a report (entry in MneExperiment.tests;
            None to use the test that was specified most recently).
        tstart, tstop : None | scalar
            Time window for finding clusters.
        pmin : float | 'tfce' | None
            Kind of test.
        parc : None | str
            Parcellation for which to collect distribution.
        mask : None | str
            Mask whole brain.
        samples : int
            Number of random permutations of the data used to determine cluster
            p values (default 10'000).
        data : 'sensor' | 'source'
            Whether the analysis is in sensor or source space.
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        return_data : bool
            Return the data along with the test result (see below).
        make : bool
            If the target file does not exist, create it (could take a long
            time depending on the test; if False, raise an IOError).
        ...
            State parameters (Use the ``group`` state parameter to select the 
            subject group for which to perform the test).

        Returns
        -------
        ds : Dataset (if return_data==True)
            Data that forms the basis of the test.
        res : TestResult
            Test result for the specified test.
        """
        self.set(test=test, **kwargs)
        self._set_analysis_options(data, sns_baseline, src_baseline, pmin,
                                   tstart, tstop, parc, mask, (data, 'time'))
        return self._load_test(test, tstart, tstop, pmin, parc, mask, samples,
                               data, sns_baseline, src_baseline, return_data,
                               make)

    def _load_test(self, test, tstart, tstop, pmin, parc, mask, samples, data,
                   sns_baseline, src_baseline, return_data, make):
        "Load a cached test after _set_analysis_options() has been called"
        test_obj = self._tests[test]

        # find data to use
        modality = self.get('modality')
        if data == 'sensor':
            dims = ('time', 'sensor')
            if modality == '':
                y_name = 'meg'
            elif modality == 'eeg':
                y_name = 'eeg'
            else:
                raise ValueError("data=%r, modality=%r" % (data, modality))
        elif data == 'source':
            dims = ('time', 'source')
            y_name = 'srcm'
        else:
            raise ValueError("data=%s" % repr(data))

        #  parc/mask
        if parc:
            mask = True
            parc_dim = 'source'
        elif mask:
            if pmin is None:  # can as well collect dist for parc
                parc_dim = 'source'
            else:  # parc means disconnecting
                parc_dim = None
        else:
            parc_dim = None

        dst = self.get('test-file', mkdir=True)

        # try to load cached test
        res = None
        load_data = True
        desc = self._get_rel('test-file', 'test-dir')
        if self._result_file_mtime(dst, data):
            try:
                res = load.unpickle(dst)
                if data == 'source':
                    update_subjects_dir(res, self.get('mri-sdir'), 2)
            except OldVersionError:
                res = None
            else:
                if res.samples >= samples or res.samples == -1:
                    self._log.info("Load cached test: %s", desc)
                    load_data = return_data
                elif not make:
                    raise IOError("The requested test %s is cached with "
                                  "samples=%i, but you request samples=%i; Set "
                                  "make=True to perform the test." %
                                  (desc, res.samples, samples))
                else:
                    res = None
        elif not make and exists(dst):
            raise IOError("The requested test is outdated: %s. Set make=True "
                          "to perform the test." % desc)

        if res is None and not make:
            raise IOError("The requested test is not cached: %s. Set make=True "
                          "to perform the test." % desc)

        if res is None:
            test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop, dims,
                                            parc_dim)

        # two-stage tests
        if isinstance(test_obj, TwoStageTest):
            if data != 'source':
                raise NotImplementedError("Two-stage test with data != 'source'")

            if load_data:
                if test_obj.model is not None:
                    self.set(model=test_obj.model)

                # stage 1
                lms = []
                dss = []
                for subject in tqdm(self, "Loading stage 1 models",
                                    len(self.get_field_values('subject'))):
                    if test_obj.model is None:
                        ds = self.load_epochs_stc(subject, sns_baseline,
                                                  src_baseline, morph=True,
                                                  mask=mask, vardef=test_obj.vars)
                    else:
                        ds = self.load_evoked_stc(subject, sns_baseline,
                                                  src_baseline, morph_ndvar=True,
                                                  mask=mask, vardef=test_obj.vars)

                    if res is None:
                        lms.append(testnd.LM(y_name, test_obj.stage_1, ds,
                                             subject=subject))
                    if return_data:
                        dss.append(ds)

                if res is None:
                    res = testnd.LMGroup(lms)
                    res.compute_column_ttests(**test_kwargs)
                    # cache
                    save.pickle(res, dst)

            if return_data:
                return combine(dss), res
            else:
                return res

        # load data
        if load_data:
            if data == 'sensor':
                ds = self.load_evoked(True, sns_baseline, True, test_obj.cat)
            elif data == 'source':
                ds = self.load_evoked_stc(True, sns_baseline, src_baseline,
                                          morph_ndvar=True, cat=test_obj.cat,
                                          mask=mask)

        # perform the test if it was not cached
        if res is None:
            self._log.info("Make test: %s", desc)
            res = self._make_test(ds[y_name], ds, test, test_kwargs)
            # cache
            save.pickle(res, dst)

        if return_data:
            return ds, res
        else:
            return res

    def make_annot(self, redo=False, **state):
        """Make sure the annot files for both hemispheres exist

        Parameters
        ----------
        redo : bool
            Even if the file exists, recreate it (default False).
        ...
            State parameters.

        Returns
        -------
        mtime : float | None
            Modification time of the existing files, or None if they were newly
            created.
        """
        self.set(**state)

        # variables
        parc, p = self._get_parc()
        if p is None:
            return

        mrisubject = self.get('mrisubject')
        common_brain = self.get('common_brain')
        mtime = self._annot_file_mtime()
        if mrisubject != common_brain:
            is_fake = is_fake_mri(self.get('mri-dir'))
            if p.morph_from_fsaverage or is_fake:
                # make sure annot exists for common brain
                self.set(mrisubject=common_brain, match=False)
                common_brain_mtime = self.make_annot()
                self.set(mrisubject=mrisubject, match=False)
                if not redo and mtime > common_brain_mtime:
                    return mtime
                elif is_fake:
                    for _ in self.iter('hemi'):
                        self.make_copy('annot-file', 'mrisubject', common_brain,
                                       mrisubject)
                else:
                    self.get('label-dir', make=True)
                    subjects_dir = self.get('mri-sdir')
                    for hemi in ('lh', 'rh'):
                        cmd = ["mri_surf2surf", "--srcsubject", common_brain,
                               "--trgsubject", mrisubject, "--sval-annot", parc,
                               "--tval", parc, "--hemi", hemi]
                        subp.run_freesurfer_command(cmd, subjects_dir)
                    fix_annot_names(mrisubject, parc, common_brain,
                                    subjects_dir=subjects_dir)
                return

        if not redo and mtime:
            return mtime
        elif not p.make:
            if redo and mtime:
                msg = ("The %s parcellation can not be created automatically "
                       "for %s." % (parc, mrisubject))
            else:
                msg = ("The %s parcellation can not be created automatically "
                       "and is missing for %s." % (parc, mrisubject))
            raise RuntimeError(msg)

        # make parcs:  common_brain | non-morphed
        labels = self._make_annot(parc, p, mrisubject)
        write_labels_to_annot(labels, mrisubject, parc, True,
                              self.get('mri-sdir'))

    def _make_annot(self, parc, p, subject):
        """Return labels

        Notes
        -----
        Only called to make custom annotation files for the common_brain
        """
        subjects_dir = self.get('mri-sdir')
        if isinstance(p, CombinationParcellation):
            with self._temporary_state:
                base = {l.name: l for l in self.load_annot(parc=p.base)}
            labels = []
            for name, exp in p.labels.iteritems():
                labels += combination_label(name, exp, base, subjects_dir)
        elif isinstance(p, SeededParcellation):
            if p.mask:
                with self._temporary_state:
                    self.make_annot(parc=p.mask)
            name, extent = SEEDED_PARC_RE.match(parc).groups()
            labels = labels_from_mni_coords(p.seeds, float(extent), subject,
                                            p.surface, p.mask, subjects_dir,
                                            parc)
        elif isinstance(p, EelbrainParcellation) and p.name == 'lobes':
            if subject != 'fsaverage':
                raise RuntimeError("lobes parcellation can only be created for "
                                   "fsaverage, not for %s" % subject)

            # load source annot
            with self._temporary_state:
                labels = self.load_annot(parc='PALS_B12_Lobes')

            # sort labels
            labels = [l for l in labels if l.name[:-3] != 'MEDIAL.WALL']

            # rename good labels
            rename_label(labels, 'LOBE.FRONTAL', 'frontal')
            rename_label(labels, 'LOBE.OCCIPITAL', 'occipital')
            rename_label(labels, 'LOBE.PARIETAL', 'parietal')
            rename_label(labels, 'LOBE.TEMPORAL', 'temporal')

            # reassign unwanted labels
            targets = ('frontal', 'occipital', 'parietal', 'temporal')
            dissolve_label(labels, 'LOBE.LIMBIC', targets, subjects_dir)
            dissolve_label(labels, 'GYRUS', targets, subjects_dir, 'rh')
            dissolve_label(labels, '???', targets, subjects_dir)
            dissolve_label(labels, '????', targets, subjects_dir, 'rh')
            dissolve_label(labels, '???????', targets, subjects_dir, 'rh')
        else:
            raise NotImplementedError(
                "At least one of the annot files for the custom parcellation "
                "%r is missing for %r, and a make function is not "
                "implemented." % (parc, subject))
        return labels

    def make_bad_channels(self, bad_chs=(), redo=False, **kwargs):
        """Write the bad channel definition file for a raw file

        If the file already exists, new bad channels are added to the old ones.
        In order to replace the old file with only the new values, set
        ``redo=True``.

        Parameters
        ----------
        bad_chs : iterator of str
            Names of the channels to set as bad. Numerical entries are
            interpreted as "MEG XXX". If bad_chs contains entries not present
            in the raw data, a ValueError is raised.
        redo : bool
            If the file already exists, replace it (instead of adding).
        ...
            State parameters.

        See Also
        --------
        make_bad_channels_auto : find bad channels automatically
        load_bad_channels : load the current bad_channels file
        merge_bad_channels : merge bad channel definitions for all sessions
        """
        pipe = self._raw[self.get('raw', **kwargs)]
        pipe.make_bad_channels(self.get('subject'), self.get('session'),
                               bad_chs, redo)

    def make_bad_channels_auto(self, flat=1e-14):
        """Automatically detect bad channels

        Works on ``raw='raw'``

        Parameters
        ----------
        flat : scalar
            Threshold for detecting flat channels: channels with ``std < flat``
            are considered bad (default 1e-14).
        """
        pipe = self._raw['raw']
        pipe.make_bad_channels_auto(self.get('subject'), self.get('session'),
                                    flat)

    def make_besa_evt(self, redo=False, **state):
        """Make the trigger and event files needed for besa

        Parameters
        ----------
        redo : bool
            If besa files already exist, overwrite them.
        ...
            State parameters.

        Notes
        -----
        Ignores the *decim* epoch parameter.

        Target files are saved relative to the *besa-root* location.
        """
        self.set(**state)
        rej = self.get('rej')
        trig_dest = self.get('besa-trig', rej='', mkdir=True)
        evt_dest = self.get('besa-evt', rej=rej, mkdir=True)
        if not redo and exists(evt_dest) and exists(trig_dest):
            return

        # load events
        ds = self.load_selected_events(reject='keep')

        # save triggers
        if redo or not exists(trig_dest):
            save.meg160_triggers(ds, trig_dest, pad=1)
            if not redo and exists(evt_dest):
                return
        else:
            ds.index('besa_index', 1)

        # reject bad trials
        ds = ds.sub('accept')

        # save evt
        epoch = self._epochs[self.get('epoch')]
        save.besa_evt(ds, tstart=epoch.tmin, tstop=epoch.tmax, dest=evt_dest)

    def make_copy(self, temp, field, src, dst, redo=False):
        """Make a copy of a file

        Parameters
        ----------
        temp : str
            Template of the file which to copy.
        field : str
            Field in which the source and target of the link are distinguished.
        src : str
            Value for field on the source file.
        dst : str
            Value for field on the destination filename.
        redo : bool
            If the target file already exists, overwrite it.
        """
        dst_path = self.get(temp, mkdir=True, **{field: dst})
        if not redo and exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        if isdir(src_path):
            raise ValueError("Can only copy files, not directories.")
        shutil.copyfile(src_path, dst_path)

    def make_cov(self):
        "Make a noise covariance (cov) file"
        dest = self.get('cov-file', mkdir=True)
        if exists(dest):
            mtime = self._cov_mtime()
            if mtime and getmtime(dest) > mtime:
                return

        params = self._covs[self.get('cov')]
        method = params.get('method', 'empirical')
        keep_sample_mean = params.get('keep_sample_mean', True)
        reg = params.get('reg', None)

        if 'epoch' in params:
            with self._temporary_state:
                epochs = self.load_epochs(None, True, False, decim=1,
                                          epoch=params['epoch'])['epochs']
            cov = mne.compute_covariance(epochs, keep_sample_mean, method=method)
        else:
            with self._temporary_state:
                raw = self.load_raw(session=params['session'])
            cov = mne.compute_raw_covariance(raw, method=method)

        if reg is True:
            cov = mne.cov.regularize(cov, epochs.info)
        elif isinstance(reg, dict):
            cov = mne.cov.regularize(cov, epochs.info, **reg)
        elif reg == 'best':
            if mne.pick_types(epochs.info, meg='grad', eeg=True, ref_meg=False):
                raise NotImplementedError("EEG or gradiometer sensors")
            reg_vs = np.arange(0, 0.21, 0.01)
            covs = [mne.cov.regularize(cov, epochs.info, mag=v) for v in reg_vs]

            # compute whitened global field power
            evoked = epochs.average()
            picks = mne.pick_types(evoked.info, meg='mag', ref_meg=False)
            gfps = [mne.whiten_evoked(evoked, cov, picks).data.std(0)
                    for cov in covs]

            # apply padding
            t_pad = params.get('reg_eval_win_pad', 0)
            if t_pad:
                n_pad = int(t_pad * epochs.info['sfreq'])
                if len(gfps[0]) <= 2 * n_pad:
                    msg = "Covariance padding (%s) is bigger than epoch" % t_pad
                    raise ValueError(msg)
                padding = slice(n_pad, -n_pad)
                gfps = [gfp[padding] for gfp in gfps]

            vs = [gfp.mean() for gfp in gfps]
            i = np.argmin(np.abs(1 - np.array(vs)))
            cov = covs[i]

            # save cov value
            with open(self.get('cov-info-file', mkdir=True), 'w') as fid:
                fid.write('%s\n' % reg_vs[i])

        cov.save(dest)

    def make_empty_room_raw(self, source_file, redo=False):
        """Generate am empty room raw file with the subject's digitizer info

        Parameters
        ----------
        source_file : str
            Path to the empty room raw file.
        redo : bool
            If the file already exists, recreate and overwrite it (default
            False).
        """
        # figure out sessions
        if 'emptyroom' not in self._sessions:
            raise ValueError("The expleriment does not have a session called "
                             "'emptyroom'")
        for session in self._sessions:
            if session != 'emptyroom':
                break
        else:
            raise ValueError("The experiment does not have a session other than "
                             "'emptyroom'")
        # find destination path
        with self._temporary_state:
            dst = self.get('raw-file', raw='raw', session='emptyroom')
            if not redo and os.path.exists(dst):
                return
            dig_raw = self.load_raw(add_bads=False, raw='raw', session=session)
        # generate file
        raw = mne.io.read_raw_fif(source_file)
        for key in ('dev_head_t', 'dig'):
            raw.info[key] = dig_raw.info[key]
        raw.save(dst, overwrite=redo)

    def _make_evoked(self, decim, data_raw):
        """Make files with evoked sensor data.

        Parameters
        ----------
        decim : int
            Data decimation factor (the default is the factor specified in the
            epoch definition).
        """
        dst = self.get('evoked-file', mkdir=True)
        epoch = self._epochs[self.get('epoch')]
        use_cache = ((not decim or decim == epoch.decim) and
                     (isinstance(data_raw, int) or data_raw == self.get('raw')))
        model = self.get('model')
        equal_count = self.get('equalize_evoked_count') == 'eq'
        if use_cache and exists(dst):
            mtime = self._evoked_mtime()
            if mtime and getmtime(dst) > mtime:
                ds = self.load_selected_events(data_raw=data_raw)
                ds = ds.aggregate(model, drop_bad=True, equal_count=equal_count,
                                  drop=('i_start', 't_edf', 'T', 'index'))
                ds['evoked'] = mne.read_evokeds(dst, proj=False)
                return ds

        # load the epochs (post baseline-correction trigger shift requires
        # baseline corrected evoked
        if epoch.post_baseline_trigger_shift:
            ds = self.load_epochs(ndvar=False, baseline=True, decim=decim,
                                  data_raw=data_raw)
        else:
            ds = self.load_epochs(ndvar=False, decim=decim, data_raw=data_raw)

        # aggregate
        ds_agg = ds.aggregate(model, drop_bad=True, equal_count=equal_count,
                              drop=('i_start', 't_edf', 'T', 'index'),
                              never_drop=('epochs',))
        ds_agg.rename('epochs', 'evoked')

        # save
        if use_cache:
            mne.write_evokeds(dst, ds_agg['evoked'])

        return ds_agg

    def make_fwd(self):
        """Make the forward model"""
        dst = self.get('fwd-file')
        if exists(dst):
            fwd_mtime = getmtime(dst)
            if fwd_mtime > self._fwd_mtime():
                return
        elif self.get('modality') != '':
            raise NotImplementedError("Source reconstruction with EEG")

        trans = self.get('trans-file')
        src = self.get('src-file', make=True)
        raw = self.get('raw-file')
        bem = self._load_bem()
        src = mne.read_source_spaces(src)

        self._log.debug("make_fwd %s...", os.path.split(dst)[1])
        bemsol = mne.make_bem_solution(bem)
        fwd = mne.make_forward_solution(raw, trans, src, bemsol,
                                        ignore_ref=True)
        for s, s0 in izip(fwd['src'], src):
            if s['nuse'] != s0['nuse']:
                msg = ("The forward solution %s contains fewer sources than "
                       "the source space. This could be due to a corrupted "
                       "bem file with source outside the inner skull surface." %
                       (os.path.split(dst)[1]))
                raise RuntimeError(msg)
        mne.write_forward_solution(dst, fwd, True)

    def make_ica_selection(self, epoch=None, decim=None):
        """Select ICA components to remove through a GUI.

        Parameters
        ----------
        epoch : str
            Epoch to use for visualization in the GUI (default is current
            epoch; does not apply to ICA specified through artifact_rejection).
        decim : int (optional)
            Downsample epochs (for visualization only).

        Notes
        -----
        Computing ICA decomposition can take a while. In order to precompute
        the decomposition for all subjects before doing the selection use
        :meth:`.make_ica()` in a loop as in::

            >>> for _ in e:
            ...     e.make_ica()
            ...
        """
        path, ds = self.make_ica(epoch or True, decim)
        self._g = gui.select_components(
            path, ds, self._sysname(
                ds['epochs'], ds.info['subject'], self.get('modality')))

    def make_ica(self, return_data=False, decim=None):
        """Compute the ICA decomposition

        If a corresponding file exists, a basic check is done as to whether the
        bad channels have changed, and if so the ICA is recomputed.

        Parameters
        ----------
        return_data : bool | str
            Return epoch data for ICA component selection. Can be a string to
            load secific epoch.
        decim : int (optional)
            Downsample epochs (for visualization only).

        Returns
        -------
        path : str
            Path to the ICA file.
        [ds : Dataset]
            Dataset with the epoch data the ICA is based on (only if
            ``return_data`` is ``True``)

        Notes
        -----
        ICA decomposition can take some time. This function can be used to
        precompute ICA decompositions for all subjects after trial pre-rejection
        has been completed::

            >>> for s in e:
            ...     e.make_ica()

        """
        pipe = self._raw[self.get('raw')]
        if isinstance(pipe, RawICA):
            path = pipe.make_ica(self.get('subject'))
            if not return_data:
                return path
            epoch = self.get('epoch') if return_data is True else return_data
            with self._temporary_state:
                ds = self.load_epochs(ndvar=False, epoch=epoch, reject=False,
                                      raw=pipe.source.name, decim=decim)
            return path, ds

        # ICA as rej setting
        params = self._artifact_rejection[self.get('rej')]
        if params['kind'] != 'ica':
            raise RuntimeError("Current raw (%s) or rej (%s) do not involve "
                               "ICA" % (self.get('raw'), self.get('rej')))

        path = self.get('ica-file', mkdir=True)
        make = not exists(path)
        if params['source'] == 'raw':
            load_epochs = return_data
            inst = self.load_raw()
        elif params['source'] == 'epochs':
            if decim is not None:
                raise TypeError("decim can not be specified for ICA based on "
                                "epochs")
            load_epochs = True
            if not make and not self._rej_mtime(self._epochs[params['epoch']]):
                make = True
        else:
            raise ValueError("rej['source']=%r, needs to be 'raw' or 'epochs'" %
                             params['source'])

        if load_epochs:
            with self._temporary_state:
                if isinstance(params['epoch'], basestring):
                    epoch = params['epoch']
                elif isinstance(params['epoch'], dict):
                    epoch = params['epoch'][self.get('session')]
                else:
                    raise TypeError("ICA param epoch=%s" % repr(params['epoch']))
                ds = self.load_epochs(ndvar=False, apply_ica=False, epoch=epoch,
                                      reject=not params['source'] == 'raw',
                                      decim=decim)
            if params['source'] == 'epochs':
                inst = ds['epochs']

        if not make:  # check bad channels
            ica = self.load_ica()
            picks = mne.pick_types(inst.info, ref_meg=False)
            if ica.ch_names != [inst.ch_names[i] for i in picks]:
                self._log.info("%s: ICA outdated due to change in bad "
                               "channels", self.get('subject'))
                make = True

        if make:
            ica = mne.preprocessing.ICA(params['n_components'],
                                        random_state=params['random_state'],
                                        method=params['method'], max_iter=256)
            # reject presets from meeg-preprocessing
            ica.fit(inst, reject={'mag': 5e-12, 'grad': 5000e-13, 'eeg': 300e-6})
            ica.save(path)

        if return_data:
            return path, ds
        else:
            return path

    def make_link(self, temp, field, src, dst, redo=False):
        """Make a hard link

        Make a hard link at the file with the ``dst`` value on ``field``,
        linking to the file with the ``src`` value of ``field``.

        Parameters
        ----------
        temp : str
            Template of the file for which to make a link.
        field : str
            Field in which the source and target of the link are distinguished.
        src : str
            Value for field on the source file.
        dst : str
            Value for field on the destination filename.
        redo : bool
            If the target file already exists, overwrite it.
        """
        dst_path = self.get(temp, **{field: dst})
        if not redo and exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        os.link(src_path, dst_path)

    def make_mov_ga_dspm(self, subject=None, sns_baseline=True, src_baseline=False,
                         fmin=2, surf=None, views=None, hemi=None, time_dilation=4.,
                         foreground=None, background=None, smoothing_steps=None,
                         dst=None, redo=False, **kwargs):
        """Make a grand average movie from dSPM values (requires PySurfer 0.6)

        Parameters
        ----------
        subject : None | str
            Subject or group.
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        fmin : scalar
            Minimum dSPM value to draw (default 2). fmax is 3 * fmin.
        surf : str
            Surface on which to plot data.
        views : str | tuple of str
            View(s) of the brain to include in the movie.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot.
        time_dilation : scalar
            Factor by which to slow the passage of time. For example, with
            ``time_dilation=4`` (the default) a segment of data for 500 ms will
            last 2 s.
        foreground : mayavi color
            Figure foreground color (i.e., the text color).
        background : mayavi color
            Figure background color.
        smoothing_steps : None | int
            Number of smoothing steps if data is spatially undersampled (pysurfer
            ``Brain.add_data()`` argument).
        dst : str (optional)
            Path to save the movie. The default is a file in the results
            folder with a name determined based on the input data. Plotting
            parameters (``view`` and all subsequent parameters) are not
            included in the filename. "~" is expanded to the user's home
            folder.
        redo : bool
            Make the movie even if the target file exists already.
        ...
            State parameters.
        """
        kwargs['model'] = ''
        subject, group = self._process_subject_arg(subject, kwargs)
        brain_kwargs = self._surfer_plot_kwargs(surf, views, foreground, background,
                                                smoothing_steps, hemi)
        self._set_analysis_options('source', sns_baseline, src_baseline, None,
                                   None, None, None)
        self.set(equalize_evoked_count='',
                 resname="GA dSPM %s %s" % (brain_kwargs['surf'], fmin))

        if dst is None:
            if group is None:
                dst = self.get('subject-mov-file', mkdir=True)
            else:
                dst = self.get('group-mov-file', mkdir=True)
        else:
            dst = os.path.expanduser(dst)

        if not redo and self._result_file_mtime(dst, 'source', group is None):
            return

        plot._brain.assert_can_save_movies()
        if group is None:
            ds = self.load_evoked_stc(subject, sns_baseline, src_baseline,
                                      ind_ndvar=True)
            y = ds['src']
        else:
            ds = self.load_evoked_stc(group, sns_baseline, src_baseline,
                                      morph_ndvar=True)
            y = ds['srcm']

        brain = plot.brain.dspm(y, fmin, fmin * 3, colorbar=False, **brain_kwargs)
        brain.save_movie(dst, time_dilation)
        brain.close()

    def make_mov_ttest(self, subject, model='', c1=None, c0=None, p=0.05,
                       sns_baseline=True, src_baseline=False,
                       surf=None, views=None, hemi=None, time_dilation=4.,
                       foreground=None,  background=None, smoothing_steps=None,
                       dst=None, redo=False, **kwargs):
        """Make a t-test movie (requires PySurfer 0.6)

        Parameters
        ----------
        subject : str
            Group name for a between-subject t-test, or subject name for a
            within-subject t-test.
        model : None | str
            Model on which the conditions c1 and c0 are defined. The default
            (``''``) is the grand average.
        c1 : None | str | tuple
            Test condition (cell in model). If None, the grand average is
            used and c0 has to be a scalar.
        c0 : str | scalar
            Control condition (cell on model) or scalar against which to
            compare c1.
        p : 0.1 | 0.05 | 0.01 | .001
            Maximum p value to draw.
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        surf : str
            Surface on which to plot data.
        views : str | tuple of str
            View(s) of the brain to include in the movie.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot.
        time_dilation : scalar
            Factor by which to slow the passage of time. For example, with
            ``time_dilation=4`` (the default) a segment of data for 500 ms will
            last 2 s.
        foreground : mayavi color
            Figure foreground color (i.e., the text color).
        background : mayavi color
            Figure background color.
        smoothing_steps : None | int
            Number of smoothing steps if data is spatially undersampled (pysurfer
            ``Brain.add_data()`` argument).
        dst : str (optional)
            Path to save the movie. The default is a file in the results
            folder with a name determined based on the input data. Plotting
            parameters (``view`` and all subsequent parameters) are not
            included in the filename. "~" is expanded to the user's home
            folder.
        redo : bool
            Make the movie even if the target file exists already.
        ...
            State parameters.
        """
        if p == 0.1:
            pmid = 0.05
            pmin = 0.01
        elif p == 0.05:
            pmid = 0.01
            pmin = 0.001
        elif p == 0.01:
            pmid = 0.001
            pmin = 0.001
        elif p == 0.001:
            pmid = 0.0001
            pmin = 0.00001
        else:
            raise ValueError("p=%s" % p)

        brain_kwargs = self._surfer_plot_kwargs(surf, views, foreground, background,
                                                smoothing_steps, hemi)
        surf = brain_kwargs['surf']
        if model:
            if not c1:
                raise ValueError("If x is specified, c1 needs to be specified; "
                                 "got c1=%s" % repr(c1))
            elif c0:
                resname = "t-test %s-%s {test_options} %s" % (c1, c0, surf)
                cat = (c1, c0)
            else:
                resname = "t-test %s {test_options} %s" % (c1, surf)
                cat = (c1,)
        elif c1 or c0:
            raise ValueError("If x is not specified, c1 and c0 should not be "
                             "specified either; got c1=%s, c0=%s"
                             % (repr(c1), repr(c0)))
        else:
            resname = "t-test GA {test_options} %s" % surf
            cat = None

        kwargs.update(resname=resname, model=model)
        with self._temporary_state:
            subject, group = self._process_subject_arg(subject, kwargs)
            self._set_analysis_options('source', sns_baseline, src_baseline, p,
                                       None, None, None)

            if dst is None:
                if group is None:
                    dst = self.get('subject-mov-file', mkdir=True)
                else:
                    dst = self.get('group-mov-file', mkdir=True)
            else:
                dst = os.path.expanduser(dst)

            if not redo and self._result_file_mtime(dst, 'source', group is None):
                return

            plot._brain.assert_can_save_movies()
            if group is None:
                ds = self.load_epochs_stc(subject, sns_baseline, src_baseline, cat=cat)
                y = 'src'
            else:
                ds = self.load_evoked_stc(group, sns_baseline, src_baseline,
                                          morph_ndvar=True, cat=cat)
                y = 'srcm'

            # find/apply cluster criteria
            kwargs = self._cluster_criteria_kwargs(('source', 'time'))
            if kwargs:
                kwargs.update(samples=0, pmin=p)

        # compute t-maps
        if c0:
            if group:
                res = testnd.ttest_rel(y, model, c1, c0, match='subject', ds=ds, **kwargs)
            else:
                res = testnd.ttest_ind(y, model, c1, c0, ds=ds, **kwargs)
        else:
            res = testnd.ttest_1samp(y, ds=ds, **kwargs)

        # select cluster-corrected t-map
        if kwargs:
            tmap = res.masked_parameter_map(None)
        else:
            tmap = res.t

        # make movie
        brain = plot.brain.dspm(tmap, ttest_t(p, res.df), ttest_t(pmin, res.df),
                                ttest_t(pmid, res.df), surf=surf)
        brain.save_movie(dst, time_dilation)
        brain.close()

    def make_mrat_evoked(self, **kwargs):
        """Produce the sensor data fiff files needed for MRAT sensor analysis

        Parameters
        ----------
        ...
            State parameters.

        Examples
        --------
        To produce evoked files for all subjects in the experiment:

        >>> experiment.set(model='factor1%factor2')
        >>> for _ in experiment:
        >>>     experiment.make_mrat_evoked()
        ...
        """
        ds = self.load_evoked(ndvar=False, **kwargs)

        # create fiffs
        model = self.get('model')
        factors = [f.strip() for f in model.split('%')]
        for case in ds.itercases():
            condition = '_'.join(case[f] for f in factors)
            path = self.get('mrat-sns-file', mkdir=True,
                            mrat_condition=condition)
            evoked = case['evoked']
            evoked.save(path)

    def make_mrat_stcs(self, **kwargs):
        """Produce the STC files needed for the MRAT analysis tool

        Parameters
        ----------
        ...
            State parameters.

        Examples
        --------
        To produce stc files for all subjects in the experiment:

        >>> experiment.set_inv('free')
        >>> experiment.set(model='factor1%factor2')
        >>> for _ in experiment:
        >>>     experiment.make_mrat_stcs()
        ...
        """
        ds = self.load_evoked_stc(morph_stc=True, **kwargs)

        # save condition info
        info_file = self.get('mrat_info-file', mkdir=True)
        ds.save_txt(info_file)

        # create stcs
        model = self.get('model')
        factors = [f.strip() for f in model.split('%')]
        for case in ds.itercases():
            condition = '_'.join(case[f] for f in factors)
            path = self.get('mrat-src-file', mkdir=True,
                            mrat_condition=condition)
            stc = case['stcm']
            stc.save(path)

    def make_plot_annot(self, surf='inflated', redo=False, **state):
        """Create a figure for the contents of an annotation file

        Parameters
        ----------
        surf : str
            FreeSurfer surface on which to plot the annotation.
        redo : bool
            If the target file already exists, overwrite it.
        ...
            State parameters.
        """
        if is_fake_mri(self.get('mri-dir', **state)):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)

        dst = self.get('res-file', mkdir=True, ext='png',
                       analysis='Source Annot',
                       resname="{parc} {mrisubject} %s" % surf)
        if not redo and exists(dst):
            return

        brain = self.plot_annot(surf=surf, axw=600)
        brain.save_image(dst, 'rgba', True)
        legend = brain.plot_legend(show=False)
        legend.save(dst[:-3] + 'pdf', transparent=True)
        brain.close()
        legend.close()

    def make_plot_label(self, label, surf='inflated', redo=False, **state):
        if is_fake_mri(self.get('mri-dir', **state)):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)

        dst = self._make_plot_label_dst(surf, label)
        if not redo and exists(dst):
            return

        brain = self.plot_label(label, surf=surf)
        brain.save_image(dst, 'rgba', True)

    def make_plots_labels(self, surf='inflated', redo=False, **state):
        self.set(**state)
        with self._temporary_state:
            if is_fake_mri(self.get('mri-dir')):
                self.set(mrisubject=self.get('common_brain'), match=False)

            labels = self._load_labels().values()
            dsts = [self._make_plot_label_dst(surf, label.name)
                    for label in labels]
        if not redo and all(exists(dst) for dst in dsts):
            return

        brain = self.plot_brain(surf, None, 'split', ['lat', 'med'], w=1200)
        for label, dst in zip(labels, dsts):
            brain.add_label(label)
            brain.save_image(dst, 'rgba', True)
            brain.remove_labels(hemi='lh')

    def _make_plot_label_dst(self, surf, label):
        return self.get('res-deep-file', mkdir=True, analysis='Source Labels',
                        folder="{parc} {mrisubject} %s" % surf, resname=label,
                        ext='png')

    def make_raw(self, **kwargs):
        """Make a raw file
        
        Parameters
        ----------
        ...
            State parameters.

        Notes
        -----
        Due to the electronics of the KIT system sensors, signal lower than
        0.16 Hz is not recorded even when recording at DC.
        """
        if kwargs:
            self.set(**kwargs)
        pipe = self._raw[self.get('raw')]
        pipe.cache(self.get('subject'), self.get('session'))

    def make_rej(self, decim=None, auto=None, overwrite=False, **kwargs):
        """Open the SelectEpochs GUI for manual epoch selection

        The GUI is opened with the correct file name; if the corresponding
        file exists, it is loaded, and upon saving the correct path is
        the default.

        Parameters
        ----------
        decim : int
            Decimate epochs for the purpose of faster display. Decimation is
            applied relative to the raw data file (i.e., if the raw data is
            sampled at a 1000 Hz, ``decim=10`` results in a sampling rate of
            100 Hz for display purposes. The default is to use the decim
            parameter specified in the epoch definition.
        auto : scalar (optional)
            Perform automatic rejection instead of showing the GUI by supplying
            a an absolute threshold (for example, ``1e-12`` to reject any epoch
            in which the absolute of at least one channel exceeds 1 picotesla).
            If a rejection file already exists also set ``overwrite=True``.
        overwrite : bool
            If ``auto`` is specified and a rejection file already exists,
            overwrite has to be set to ``True`` to overwrite the old file.
        ... :
            Kwargs for SelectEpochs
        """
        rej_args = self._artifact_rejection[self.get('rej')]
        if rej_args['kind'] == 'manual':
            apply_ica = False
        elif rej_args['kind'] == 'ica':
            apply_ica = rej_args['source'] == 'raw'
        else:
            raise RuntimeError("Epoch rejection for rej=%r is not manual" %
                               self.get('rej'))

        epoch = self._epochs[self.get('epoch')]
        if not isinstance(epoch, PrimaryEpoch):
            if isinstance(epoch, SecondaryEpoch):
                raise ValueError(
                    "The current epoch {cur!r} inherits rejections from "
                    "{sel!r}. To access a rejection file for this epoch, call "
                    "`e.set(epoch={sel!r})` and then call `e.make_rej()` "
                    "again.".format(cur=epoch.name, sel=epoch.sel_epoch))
            elif isinstance(epoch, SuperEpoch):
                raise ValueError(
                    "The current epoch {cur!r} inherits rejections from these "
                    "other epochs: {sel!r}. To access trial rejection for "
                    "these epochs, call `e.set(epoch=epoch)` and then call "
                    "`e.make_rej()` "
                    "again.".format(cur=epoch.name, sel=epoch.sub_epochs))
            else:
                raise ValueError(
                    "The current epoch {cur!r} is not a primary epoch and "
                    "inherits rejections from other epochs. Generate trial "
                    "rejection for these epochs.".format(cur=epoch.name))

        path = self.get('rej-file', mkdir=True)
        modality = self.get('modality')

        if auto is not None and overwrite is not True and exists(path):
            msg = ("A rejection file already exists for {subject}, epoch "
                   "{epoch}, rej {rej}. Set overwrite=True if you are sure you "
                   "want to replace that file.")
            raise IOError(self.format(msg))

        if decim is None:
            decim = rej_args.get('decim', None)

        if modality == '':
            ds = self.load_epochs(reject=False, trigger_shift=False,
                                  apply_ica=apply_ica, eog=True, decim=decim)
            eog_sns = self._eog_sns.get(ds['meg'].sensor.sysname)
            data = 'meg'
            vlim = 2e-12
        elif modality == 'eeg':
            ds = self.load_epochs(reject=False, eog=True, baseline=True,
                                  decim=decim, trigger_shift=False)
            eog_sns = self._eog_sns['KIT-BRAINVISION']
            data = 'eeg'
            vlim = 1.5e-4
        else:
            raise ValueError("modality=%r" % modality)

        if auto is not None:
            # create rejection
            rej_ds = new_rejection_ds(ds)
            rej_ds[:, 'accept'] = ds[data].abs().max(('sensor', 'time')) <= auto
            # create description for info
            args = ["auto=%r" % auto]
            if overwrite is True:
                args.append("overwrite=True")
            if decim is not None:
                args.append("decim=%s" % repr(decim))
            rej_ds.info['desc'] = ("Created with %s.make_rej(%s)" %
                                   (self.__class__.__name__, ', '.join(args)))
            # save
            save.pickle(rej_ds, path)
            # print info
            n_rej = rej_ds.eval("sum(accept == False)")
            msg = ("%i of %i epochs rejected with threshold %s for {subject}, "
                   "epoch {epoch}" % (n_rej, rej_ds.n_cases, auto))
            print(self.format(msg))
            return

        # don't mark eog sns if it is bad
        bad_channels = self.load_bad_channels()
        eog_sns = [c for c in eog_sns if c not in bad_channels]

        gui.select_epochs(ds, data, path=path, vlim=vlim, mark=eog_sns, **kwargs)

    def _need_not_recompute_report(self, dst, samples, data, redo):
        "Check (and log) whether the report needs to be redone"
        desc = self._get_rel('report-file', 'res-dir')
        if not exists(dst):
            self._log.debug("New report: %s", desc)
        elif redo:
            self._log.debug("Redoing report: %s", desc)
        elif not self._result_file_mtime(dst, data):
            self._log.debug("Report outdated: %s", desc)
        else:
            meta = read_meta(dst)
            if 'samples' in meta:
                if int(meta['samples']) >= samples:
                    self._log.debug("Report up to date: %s", desc)
                    return True
                else:
                    self._log.debug("Report file used %s samples, recomputing "
                                    "with %i: %s", meta['samples'], samples,
                                    desc)
            else:
                self._log.debug("Report created prior to Eelbrain 0.25, can "
                                "not check number of samples. Delete manually "
                                "to recompute: %s", desc)
                return True

    def make_report(self, test, parc=None, mask=None, pmin=None, tstart=0.15,
                    tstop=None, samples=10000, sns_baseline=True,
                    src_baseline=None, include=0.2, redo=False, **state):
        """Create an HTML report on spatio-temporal clusters

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        parc : None | str
            Find clusters in each label of parc (as opposed to the whole
            brain).
        mask : None | str
            Parcellation to apply as mask. Can only be specified if parc==None.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 10,000).
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        include : 0 < scalar <= 1
            Create plots for all clusters with p-values smaller or equal this value.
        redo : bool
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        ...
            State parameters.
        """
        if samples < 1:
            raise ValueError("samples needs to be > 0")
        elif include <= 0 or include > 1:
            raise ValueError("include needs to be 0 < include <= 1, got %s"
                             % repr(include))

        self.set(**state)
        self._set_analysis_options('source', sns_baseline, src_baseline, pmin,
                                   tstart, tstop, parc, mask)
        dst = self.get('report-file', mkdir=True, test=test)
        if self._need_not_recompute_report(dst, samples, 'source', redo):
            return

        # start report
        title = self.format('{session} {epoch} {test} {test_options}')
        report = Report(title)

        if isinstance(self._tests[test], TwoStageTest):
            self._two_stage_report(report, test, sns_baseline, src_baseline,
                                   pmin, samples, tstart, tstop, parc, mask,
                                   include)
        else:
            self._evoked_report(report, test, sns_baseline, src_baseline, pmin,
                                samples, tstart, tstop, parc, mask, include)

        # report signature
        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst, meta={'samples': samples})

    def _evoked_report(self, report, test, sns_baseline, src_baseline, pmin,
                       samples, tstart, tstop, parc, mask, include):
        # load data
        ds, res = self._load_test(test, tstart, tstop, pmin, parc, mask, samples,
                                  'source', sns_baseline, src_baseline, True, True)

        # info
        surfer_kwargs = self._surfer_plot_kwargs()
        self._report_test_info(report.add_section("Test Info"), ds, test, res,
                               'source', include)
        if parc:
            section = report.add_section(parc)
            caption = "Labels in the %s parcellation." % parc
            self._report_parc_image(section, caption)
        elif mask:
            title = "Whole Brain Masked by %s" % mask
            section = report.add_section(title)
            caption = "Mask: %s" % mask.capitalize()
            self._report_parc_image(section, caption)

        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.source_time_results(res, ds, colors, include,
                                                  surfer_kwargs, parc=parc))

    def _two_stage_report(self, report, test, sns_baseline, src_baseline, pmin,
                          samples, tstart, tstop, parc, mask, include):
        model = self._tests[test].model
        rlm = self._load_test(test, tstart, tstop, pmin, parc, mask, samples,
                              'source', sns_baseline, src_baseline, bool(model),
                              True)
        if model:
            group_ds, rlm = rlm
        else:
            group_ds = None

        # start report
        surfer_kwargs = self._surfer_plot_kwargs()
        info_section = report.add_section("Test Info")
        if parc:
            section = report.add_section(parc)
            caption = "Labels in the %s parcellation." % parc
            self._report_parc_image(section, caption)
        elif mask:
            title = "Whole Brain Masked by %s" % mask
            section = report.add_section(title)
            caption = "Mask: %s" % mask.capitalize()
            self._report_parc_image(section, caption)

        # Design matrix
        section = report.add_section("Design Matrix")
        section.append(rlm.design())

        # add results to report
        for term in rlm.column_names:
            res = rlm.tests[term]
            ds = rlm.coefficients_dataset(term)
            report.append(_report.source_time_results(res, ds, None, include,
                                                      surfer_kwargs, term))

        self._report_test_info(info_section, group_ds or ds, test, res, 'source')

    def make_report_rois(self, test, parc=None, pmin=None, tstart=0.15, tstop=None,
                         samples=10000, sns_baseline=True, src_baseline=False,
                         redo=False, **state):
        """Create an HTML report on ROI time courses

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        parc : str
            Parcellation that defines ROIs.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        sns_baseline : bool | tuple
            Apply baseline correction using this period in sensor space.
            True to use the epoch's baseline specification (default).
        src_baseline : bool | tuple
            Apply baseline correction using this period in source space.
            True to use the epoch's baseline specification. The default is to
            not apply baseline correction.
        redo : bool
            If the target file already exists, delete and recreate it.
        ...
            State parameters.
        """
        test_obj = self._tests[test]
        if samples < 1:
            raise ValueError("Need samples > 0 to run permutation test.")
        elif isinstance(test_obj, TwoStageTest):
            raise NotImplementedError("ROI analysis not implemented for two-"
                                      "stage tests")

        if parc is not None:
            state['parc'] = parc
        parc = self.get('parc', **state)
        if not parc:
            raise ValueError("No parcellation specified")
        self._set_analysis_options('source', sns_baseline, src_baseline, pmin,
                                   tstart, tstop, parc, dims=('time',))
        dst = self.get('report-file', mkdir=True, fmatch=False, test=test,
                       folder="%s ROIs" % parc)
        if self._need_not_recompute_report(dst, samples, 'source', redo):
            return

        # load data
        dss = defaultdict(list)
        n_trials_dss = []
        n_subjects = len(self.get_field_values('subject'))
        for _ in tqdm(self, "Loading data", n_subjects, unit='subject'):
            ds = self.load_evoked_stc(None, sns_baseline, src_baseline, ind_ndvar=True)
            src = ds.pop('src')
            n_trials_dss.append(ds.copy())
            for label in src.source.parc.cells:
                if label.startswith('unknown-'):
                    continue
                label_ds = ds.copy()
                label_ds['label_tc'] = src.mean(source=label)
                dss[label].append(label_ds)
            del src
        n_trials_ds = combine(n_trials_dss, incomplete='drop')

        # n subjects per label
        n_per_label = {label: len(dss[label]) for label in dss}

        # sorted labels
        labels_lh = []
        labels_rh = []
        for label in dss.keys():
            if label.endswith('-lh'):
                labels_lh.append(label)
            elif label.endswith('-rh'):
                labels_rh.append(label)
            else:
                raise NotImplementedError("Label named %s" % repr(label.name))
        labels_lh.sort()
        labels_rh.sort()

        # compute results
        test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop, ('time',), None)
        do_mcc = (
            len(labels_lh) + len(labels_rh) > 1 and  # more than one test
            pmin not in (None, 'tfce') and  # not implemented
            len(set(n_per_label.values())) == 1  # equal n permutations
        )
        label_results = {}
        for label in labels_lh + labels_rh:
            ds = combine(dss[label], incomplete='drop')
            res = self._make_test(ds['label_tc'], ds, test, test_kwargs, do_mcc)
            label_results[label] = res

        if do_mcc:
            cdists = [r._cdist for r in label_results.values()]
            merged_dist = _MergedTemporalClusterDist(cdists)
        else:
            merged_dist = None

        # start report
        title = self.format('{session} {epoch} {test} {test_options}')
        report = Report(title)

        # method intro (compose it later when data is available)
        info_section = report.add_section("Test Info")
        self._report_test_info(info_section, n_trials_ds, test, res, 'source')

        # add parc image
        section = report.add_section(parc)
        caption = "ROIs in the %s parcellation." % parc
        self._report_parc_image(section, caption)

        # add content body
        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        for hemi, label_names in (('Left', labels_lh), ('Right', labels_rh)):
            section = report.add_section("%s Hemisphere" % hemi)
            for label in label_names:
                res = label_results[label]
                title = label[:-3].capitalize()
                caption = "Mean in label %s." % label
                n = n_per_label[label]
                if n < n_subjects:
                    title += ' (n=%i)' % n
                    caption += " Data from %i of %i subjects." % (n, n_subjects)
                section.append(_report.time_results(
                    res, ds, colors, title, caption, merged_dist=merged_dist))

        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst, meta={'samples': samples})

    def _make_report_eeg(self, test, pmin=None, tstart=0.15, tstop=None,
                         samples=10000, baseline=True, include=1, **state):
        # outdated (cache, load_test())
        """Create an HTML report on EEG sensor space spatio-temporal clusters

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        baseline : bool | tuple
            Apply baseline correction using this period. True to use the epoch's
            baseline specification (default).
        include : 0 < scalar <= 1
            Create plots for all clusters with p-values smaller or equal this
            value (the default is 1, i.e. to show all clusters).
        ...
            State parameters.
        """
        self._set_analysis_options('eeg', baseline, None, pmin, tstart, tstop,
                                   None, dims=('sensor', 'time'))
        dst = self.get('report-file', mkdir=True, fmatch=False, test=test,
                       folder="EEG Spatio-Temporal", modality='eeg',
                       **state)
        if self._need_not_recompute_report(dst, samples, 'sensor', False):
            return

        # load data
        ds, res = self.load_test(test, tstart, tstop, pmin, None, None, samples,
                                 'sensor', baseline, None, True, True)

        # start report
        title = self.format('{session} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        info_section = report.add_section("Test Info")
        self._report_test_info(info_section, ds, test, res, 'sensor', include)

        # add connectivity image
        p = plot.SensorMap(ds['eeg'], connectivity=True, show=False)
        image_conn = p.image("connectivity.png")
        info_section.add_figure("Sensor map with connectivity", image_conn)
        p.close()

        colors = plot.colors_for_categorial(ds.eval(res._plot_model()))
        report.append(_report.sensor_time_results(res, ds, colors, include))
        report.sign(('eelbrain', 'mne', 'scipy', 'numpy'))
        report.save_html(dst)

    def _make_report_eeg_sensors(self, test, sensors=('FZ', 'CZ', 'PZ', 'O1', 'O2'),
                                 pmin=None, tstart=0.15, tstop=None,
                                 samples=10000, baseline=True, redo=False,
                                 **state):
        # outdated (cache)
        """Create an HTML report on individual EEG sensors

        Parameters
        ----------
        test : str
            Test for which to create a report (entry in MneExperiment.tests).
        sensors : sequence of str
            Names of the sensors which to include.
        pmin : None | scalar, 1 > pmin > 0 | 'tfce'
            Equivalent p-value for cluster threshold, or 'tfce' for
            threshold-free cluster enhancement.
        tstart : None | scalar
            Beginning of the time window for finding clusters.
        tstop : None | scalar
            End of the time window for finding clusters.
        samples : int > 0
            Number of samples used to determine cluster p values for spatio-
            temporal clusters (default 1000).
        baseline : bool | tuple
            Apply baseline correction using this period. True to use the epoch's
            baseline specification (default).
        redo : bool
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        ...
            State parameters.
        """
        self._set_analysis_options('eeg', baseline, None, pmin, tstart, tstop,
                                   None, dims=('time',))
        dst = self.get('report-file', mkdir=True, fmatch=False, test=test,
                       folder="EEG Sensors", modality='eeg', **state)
        if self._need_not_recompute_report(dst, samples, 'sensor', redo):
            return

        # load data
        ds = self.load_evoked(self.get('group'), baseline, True)

        # test that sensors are in the data
        eeg = ds['eeg']
        missing = [s for s in sensors if s not in eeg.sensor.names]
        if missing:
            raise ValueError("The following sensors are not in the data: %s" % missing)

        # start report
        title = self.format('{session} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        info_section = report.add_section("Test Info")

        # add sensor map
        p = plot.SensorMap(ds['eeg'], show=False)
        p.mark_sensors(sensors)
        info_section.add_figure("Sensor map", p)
        p.close()

        # main body
        caption = "Signal at %s."
        test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop, ('time', 'sensor'), None)
        ress = [self._make_test(eeg.sub(sensor=sensor), ds, test, test_kwargs) for
                sensor in sensors]
        colors = plot.colors_for_categorial(ds.eval(ress[0]._plot_model()))
        for sensor, res in izip(sensors, ress):
            report.append(_report.time_results(res, ds, colors, sensor, caption % sensor))

        self._report_test_info(info_section, ds, test, res, 'sensor')
        report.sign(('eelbrain', 'mne', 'scipy', 'numpy'))
        report.save_html(dst)

    def _report_subject_info(self, ds, model):
        # add subject information to experiment
        if model:
            s_ds = table.repmeas('n', model, 'subject', ds=ds)
        else:
            s_ds = ds
        s_ds2 = self.show_subjects(asds=True)
        s_ds2_aligned = align1(s_ds2[('subject', 'mri')], s_ds['subject'], 'subject')
        s_ds.update(s_ds2_aligned)
        s_table = s_ds.as_table(midrule=True, count=True, caption="All "
                                "subjects included in the analysis with "
                                "trials per condition")
        return s_table

    def _analysis_info(self, data):
        info = List("Analysis:")
        epoch = self.format('epoch = {epoch} {evoked_kind}').strip()
        model = self.format('{model}').strip()
        if model:
            epoch += ' ~ ' + model
        info.add_item(epoch)
        if data == 'source':
            info.add_item(self.format("cov = {cov}"))
            info.add_item(self.format("inv = {inv}"))
        return info

    def _report_test_info(self, section, ds, test, res, data, include=None):
        info = self._analysis_info(data)
        test_obj = self._tests[test]
        info.add_item("test = %s  (%s)" % (test_obj.test_kind, test_obj.desc))
        if include is not None:
            info.add_item("Separate plots of all clusters with a p-value < %s"
                          % include)
        section.append(info)

        # Test info (for temporal tests, res is only representative)
        info = res.info_list(data is not None)
        section.append(info)

        section.append(self._report_subject_info(ds, test_obj.model))
        section.append(self.show_state(hide=('hemi', 'subject', 'mrisubject')))

    def _report_parc_image(self, section, caption):
        "Add picture of the current parcellation"
        with self._temporary_state:
            self.set(mrisubject=self.get('common_brain'))
            brain = self.plot_annot(axw=500)
            legend = brain.plot_legend(show=False)

        content = [brain.image('parc'), legend.image('parc-legend')]
        section.add_image_figure(content, caption)

        legend.close()

    def _make_report_lm(self, pmin=0.01, sns_baseline=True, src_baseline=False,
                        mask='lobes'):
        """Report for a first level (single subject) LM

        Parameters
        ----------
        pmin : scalar
            Threshold p-value for uncorrected SPMs.
        """
        if not isinstance(self._tests[self.get('test')], TwoStageTest):
            raise NotImplementedError("Only two-stage tests")

        with self._temporary_state:
            self._set_analysis_options('source', sns_baseline, src_baseline, pmin,
                                       None, None, None, mask)
            dst = self.get('subject-spm-report', mkdir=True)
            lm = self._load_spm(sns_baseline, src_baseline)

            title = self.format('{session} {epoch} {test} {test_options}')

        report = Report(title)
        report.append(_report.source_time_lm(lm, pmin))

        # report signature
        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst)

    def make_report_coreg(self, file_name):
        """Create HTML report with plots of the MEG/MRI coregistration

        Parameters
        ----------
        file_name : str
            Where to save the report.
        """
        from matplotlib import pyplot
        from mayavi import mlab

        mri = self.get('mri')
        title = 'Coregistration'
        if mri:
            title += ' ' + mri
        report = Report(title)

        for subject in self:
            mrisubject = self.get('mrisubject')
            fig = self.plot_coreg()
            fig.scene.camera.parallel_projection = True
            fig.scene.camera.parallel_scale = .175
            mlab.draw(fig)

            # front
            mlab.view(90, 90, 1, figure=fig)
            im_front = Image.from_array(mlab.screenshot(figure=fig), 'front')

            # left
            mlab.view(0, 270, 1, roll=90, figure=fig)
            im_left = Image.from_array(mlab.screenshot(figure=fig), 'left')

            mlab.close(fig)

            # MRI/BEM figure
            if is_fake_mri(self.get('mri-dir')):
                bem_fig = None
            else:
                bem_fig = mne.viz.plot_bem(mrisubject, self.get('mri-sdir'),
                                           brain_surfaces='white', show=False)

            # add to report
            if subject == mrisubject:
                title = subject
                caption = "Coregistration for subject %s." % subject
            else:
                title = "%s (%s)" % (subject, mrisubject)
                caption = ("Coregistration for subject %s (MRI-subject %s)." %
                           (subject, mrisubject))
            section = report.add_section(title)
            if bem_fig is None:
                section.add_figure(caption, (im_front, im_left))
            else:
                section.add_figure(caption, (im_front, im_left, bem_fig))
                pyplot.close(bem_fig)

        report.sign()
        report.save_html(file_name)

    def make_src(self, **kwargs):
        """Make the source space
        
        Parameters
        ----------
        ...
            State parameters.
        """
        dst = self.get('src-file', **kwargs)
        subject = self.get('mrisubject')
        common_brain = self.get('common_brain')

        is_scaled = (subject != common_brain) and is_fake_mri(self.get('mri-dir'))

        if is_scaled:
            # make sure the source space exists for the original
            with self._temporary_state:
                self.make_src(mrisubject=common_brain)
                orig = self.get('src-file')

            if exists(dst):
                if getmtime(dst) >= getmtime(orig):
                    return

            src = self.get('src')
            subjects_dir = self.get('mri-sdir')
            mne.scale_source_space(subject, src, subjects_dir=subjects_dir)
        elif exists(dst):
            return
        else:
            src = self.get('src')
            kind, param = src.split('-')
            if kind == 'vol':
                mri = self.get('mri-file')
                bem = self._load_bem()
                mne.setup_volume_source_space(subject, dst, pos=float(param),
                                              mri=mri, bem=bem, mindist=0.,
                                              exclude=0.,
                                              subjects_dir=self.get('mri-sdir'))
            else:
                spacing = kind + param
                sss = mne.setup_source_space(subject, fname=None,
                                             spacing=spacing, add_dist=True,
                                             subjects_dir=self.get('mri-sdir'))
                mne.write_source_spaces(dst, sss)

    def _test_kwargs(self, samples, pmin, tstart, tstop, dims, parc_dim):
        "Compile kwargs for testnd tests"
        kwargs = {'samples': samples, 'tstart': tstart, 'tstop': tstop,
                  'parc': parc_dim}
        if pmin == 'tfce':
            kwargs['tfce'] = True
        elif pmin is not None:
            kwargs['pmin'] = pmin
            kwargs.update(self._cluster_criteria_kwargs(dims))
        return kwargs

    def _make_test(self, y, ds, test, kwargs, force_permutation=False):
        """Compute test results

        Parameters
        ----------
        y : NDVar
            Dependent variable.
        ds : Dataset
            Other variables.
        test : str
            Name of the test to perform
        kwargs : dict
            Test parameters (from :meth:`._test_kwargs`).
        force_permutation : bool
            Conduct permutations regardless of whether there are any clusters.
        """
        test_obj = self._tests[test]
        if not isinstance(test_obj, EvokedTest):
            raise RuntimeError("Test kind=%s" % test_obj.test_kind)
        return test_obj.make(y, ds, force_permutation, kwargs)

    def merge_bad_channels(self):
        """Merge bad channel definitions for different sessions

        Load the bad channel definitions for all sessions of the current
        subject and save the union for all sessions.

        See Also
        --------
        make_bad_channels : set bad channels for a single session
        """
        n_chars = max(map(len, self._sessions))
        # collect bad channels
        bads = set()
        sessions = []
        with self._temporary_state:
            # ICARaw merges bad channels dynamically, so explicit merge needs to
            # be performed lower in the hierarchy
            self.set(raw='raw')
            for session in self.iter('session'):
                if exists(self.get('raw-file')):
                    bads.update(self.load_bad_channels())
                    sessions.append(session)
                else:
                    print("%%-%is: skipping, raw file missing" % n_chars % session)
            # update bad channel files
            for session in sessions:
                print(session.ljust(n_chars), end=': ')
                self.make_bad_channels(bads, session=session)

    def next(self, field='subject', group=None):
        """Change field to the next value

        Parameters
        ----------
        field : str
            The field for which the value should be changed (default 'subject').
        group : str
            If cycling through subjects, only use subjects form that group.
            Does not set change the experiment's group value.
        """
        current = self.get(field)
        values = self.get_field_values(field)

        # find the index of the next value
        if current in values:
            idx = values.index(current) + 1
            if idx == len(values):
                idx = -1
        else:
            for idx in xrange(len(values)):
                if values[idx] > current:
                    break
            else:
                idx = -1

        # set the next value
        if idx == -1:
            next_ = values[0]
            print("The last %s was reached; rewinding to "
                  "%r" % (field, next_))
        else:
            next_ = values[idx]
            print("%s: %r -> %r" % (field, current, next_))
        self.set(**{field: next_})

    def plot_annot(self, parc=None, surf=None, views=None, hemi=None,
                   borders=False, alpha=0.7, w=None, h=None, axw=None, axh=None,
                   foreground=None, background=None):
        """Plot the annot file on which the current parcellation is based

        Parameters
        ----------
        parc : None | str
            Parcellation to plot. If None (default), use parc from the current
            state.
        surf : 'inflated' | 'pial' | 'smoothwm' | 'sphere' | 'white'
            Freesurfer surface to use as brain geometry.
        views : str | iterator of str
            View or views to show in the figure.
        hemi : 'lh' | 'rh' | 'both' | 'split'
            Which hemispheres to plot (default includes hemisphere with more than one
            label in the annot file).
        borders : bool | int
            Show only label borders (PySurfer Brain.add_annotation() argument).
        alpha : scalar
            Alpha of the annotation (1=opaque, 0=transparent, default 0.7).
        axw : int
            Figure width per hemisphere.
        foreground : mayavi color
            Figure foreground color (i.e., the text color).
        background : mayavi color
            Figure background color.

        Returns
        -------
        brain : Brain
            PySurfer Brain with the parcellation plot.
        legend : ColorList
            ColorList figure with the legend.
        """
        if parc is None:
            parc = self.get('parc')
        else:
            parc = self.get('parc', parc=parc)

        self.make_annot()
        mri_sdir = self.get('mri-sdir')
        if is_fake_mri(self.get('mri-dir')):
            subject = self.get('common_brain')
        else:
            subject = self.get('mrisubject')

        kwa = self._surfer_plot_kwargs(surf, views, foreground, background,
                                       None, hemi)
        return plot.brain.annot(parc, subject, borders=borders, alpha=alpha,
                                w=w, h=h, axw=axw, axh=axh,
                                subjects_dir=mri_sdir, **kwa)

    def plot_brain(self, common_brain=True, **brain_kwargs):
        """Plot the brain model

        Parameters
        ----------
        common_brain : bool
            If the current mrisubject is a scaled MRI, use the common_brain
            instead.
        ... :
            :class:`~plot._brain_object.Brain` options as keyword arguments.
        """
        from ..plot._brain_object import Brain

        brain_args = self._surfer_plot_kwargs()
        brain_args.update(brain_kwargs)
        brain_args['subjects_dir'] = self.get('mri-sdir')

        # find subject
        if common_brain and is_fake_mri(self.get('mri-dir')):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)
        else:
            mrisubject = self.get('mrisubject')

        return Brain(mrisubject, **brain_args)

    def plot_coreg(self, ch_type=None, dig=True, **kwargs):
        """Plot the coregistration (Head shape and MEG helmet)

        Parameters
        ----------
        ch_type : 'meg' | 'eeg'
            Plot only MEG or only EEG sensors (default is both).
        dig : bool
            Plot the digitization points (default True).
        ...
            State parameters.

        Notes
        -----
        Uses :func:`mne.viz.plot_trans`
        """
        self.set(**kwargs)
        raw = self.load_raw()
        return mne.viz.plot_trans(raw.info, self.get('trans-file'),
                                  self.get('mrisubject'), self.get('mri-sdir'),
                                  ch_type, 'head', dig=dig)

    def plot_whitened_gfp(self, s_start=None, s_stop=None, run=None):
        """Plot the GFP of the whitened evoked to evaluate the the covariance matrix

        Parameters
        ----------
        s_start : str
            Subject at which to start (default is the first subject).
        s_stop: str
            Subject at which to stop (default is the last subject).
        run : bool
            Run the GUI after plotting (default depends on environment).
        """
        gfps = []
        subjects = []
        with self._temporary_state:
            self.set(model='')
            for subject in self.iter_range(s_start, s_stop):
                cov = self.load_cov()
                picks = range(len(cov.ch_names))
                ds = self.load_evoked(baseline=True)
                whitened_evoked = mne.whiten_evoked(ds[0, 'evoked'], cov, picks)
                gfp = whitened_evoked.data.std(0)

                gfps.append(gfp)
                subjects.append(subject)

        colors = plot.colors_for_oneway(subjects)
        title = "Whitened Global Field Power (%s)" % self.get('cov')
        fig = plot._base.Figure(1, title, h=7, run=run)
        ax = fig._axes[0]
        for subject, gfp in izip(subjects, gfps):
            ax.plot(whitened_evoked.times, gfp, label=subject,
                    color=colors[subject])
        ax.legend(loc='right')
        fig.show()
        return fig

    def plot_evoked(self, subject=None, separate=False, baseline=True, ylim='same',
                    run=None, **kwargs):
        """Plot evoked sensor data

        Parameters
        ----------
        subject : str
            Subject or group name (default is current subject).
        separate : bool
            When plotting a group, plot all subjects separately instead or the group
            average (default False).
        baseline : bool | tuple
            Apply baseline correction using this period. True to use the epoch's
            baseline specification (default).
        ylim : 'same' | 'different'
            Use the same or different y-axis limits for different subjects
            (default 'same').
        run : bool
            Run the GUI after plotting (default in accordance with plotting
            default).
        ...
            State parameters.
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        y = self._ndvar_name_for_modality(self.get('modality'))
        model = self.get('model') or None
        epoch = self.get('epoch')
        if model:
            model_name = " ~" + model
        else:
            model_name = None

        if subject:
            ds = self.load_evoked(baseline=baseline)
            title = subject + " " + epoch + (model_name or " Average")
            return plot.TopoButterfly(y, model, ds=ds, title=title, run=run)
        elif separate:
            plots = []
            vlim = []
            for subject in self.iter(group=group):
                ds = self.load_evoked(baseline=baseline)
                title = subject + " " + epoch + (model_name or " Average")
                p = plot.TopoButterfly(y, model, ds=ds, title=title, run=False)
                plots.append(p)
                vlim.append(p.get_vlim())

            if ylim.startswith('s'):
                vlim = np.array(vlim)
                vmax = np.abs(vlim, out=vlim).max()
                for p in plots:
                    p.set_vlim(vmax)
            elif not ylim.startswith('d'):
                raise ValueError("ylim=%s" % repr(ylim))

            if run or plot._base.do_autorun():
                gui.run()
        else:
            ds = self.load_evoked(group, baseline=baseline)
            title = group + " " + epoch + (model_name or " Grand Average")
            return plot.TopoButterfly(y, model, ds=ds, title=title, run=run)

    def plot_label(self, label, surf='inflated', w=600, clear=False):
        """Plot a label"""
        if isinstance(label, basestring):
            label = self.load_label(label)
        title = label.name

        brain = self.plot_brain(surf, title, 'split', ['lat', 'med'], w, clear)
        brain.add_label(label, alpha=0.75)
        return brain

    def reset(self):
        """Reset all field values to the state at initialization
        
        This function can be used in cases where the same MneExperiment instance 
        is used to perform multiple independent operations, where parameters set 
        during one operation should not affect the next operation.
        """
        self._restore_state(0, False)

    def run_mne_analyze(self, modal=False):
        """Run mne_analyze

        Parameters
        ----------
        modal : bool
            Causes the shell to block until mne_analyze is closed.

        Notes
        -----
        Sets the current directory to raw-dir, and sets the SUBJECT and
        SUBJECTS_DIR to current values
        """
        subp.run_mne_analyze(self.get('raw-dir'), self.get('mrisubject'),
                             self.get('mri-sdir'), modal)

    def run_mne_browse_raw(self, modal=False):
        """Run mne_analyze

        Parameters
        ----------
        modal : bool
            Causes the shell to block until mne_browse_raw is closed.

        Notes
        -----
        Sets the current directory to raw-dir, and sets the SUBJECT and
        SUBJECTS_DIR to current values
        """
        subp.run_mne_browse_raw(self.get('raw-dir'), self.get('mrisubject'),
                                self.get('mri-sdir'), modal)

    def set(self, subject=None, **state):
        """
        Set variable values.

        Parameters
        ----------
        subject : str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        ...
            State parameters.
        """
        if subject is not None:
            state['subject'] = subject
        FileTree.set(self, **state)

    def _post_set_group(self, _, group):
        if group == '*' or group not in self._groups:
            return
        group_members = self._groups[group]
        self._field_values['subject'] = group_members
        subject = self.get('subject')
        if subject != '*' and subject not in group_members and group_members:
            self.set(group_members[0])

    def set_inv(self, ori='free', snr=3, method='dSPM', depth=None,
                pick_normal=False):
        """Set the type of inverse solution used for source estimation

        Parameters
        ----------
        ori : 'free' | 'fixed' | float ]0, 1]
            Orientation constraint (default 'free'; use a float to specify a
            loose constraint).
        snr : scalar
            SNR estimate for regularization (default 3).
        method : 'MNE' | 'dSPM' | 'sLORETA'
            Inverse method.
        depth : None | float [0, 1]
            Depth weighting (default 0.8; 0 to disable depth weighting).
        pick_normal : bool
            Pick the normal component of the estimated current vector (default
            False).
        """
        self.set(inv=self._inv_str(ori, snr, method, depth, pick_normal))

    @staticmethod
    def _inv_str(ori, snr, method, depth, pick_normal):
        "Construct inv str from settings"
        if isinstance(ori, basestring):
            if ori not in ('free', 'fixed'):
                raise ValueError('ori=%r' % (ori,))
        elif not 0 <= ori <= 1:
            raise ValueError("ori=%r; must be in range [0, 1]" % (ori,))
        else:
            ori = 'loose%s' % str(ori)[1:]

        if snr <= 0:
            raise ValueError("snr=%r" % (snr,))

        if method not in ('MNE', 'dSPM', 'sLORETA'):
            raise ValueError("method=%r" % (method,))

        items = [ori, '%g' % snr, method]

        if depth is None or depth == 0.8:
            pass
        elif depth < 0 or depth > 1:
            raise ValueError("depth=%r; must be in range [0, 1]" % (depth,))
        else:
            items.append('%g' % depth)

        if pick_normal:
            items.append('pick_normal')

        return '-'.join(items)

    @staticmethod
    def _inv_params(inv):
        "(ori, snr, method, depth, pick_normal)"
        m = inv_re.match(inv)
        if m is None:
            raise ValueError("Invalid inverse specification: inv=%r" % inv)

        ori, snr, method, depth, pick_normal = m.groups()
        if ori.startswith('loose'):
            ori = float(ori[5:])
            if not 0 <= ori <= 1:
                raise ValueError('inv=%r (first value of inv (loose parameter) needs '
                                 'to be in [0, 1]')
        elif ori not in ('free', 'fixed'):
            raise ValueError('inv=%r (ori=%r)' % (inv, ori))

        snr = float(snr)
        if snr <= 0:
            raise ValueError('inv=%r (snr=%r)' % (inv, snr))

        if method not in ('MNE', 'dSPM', 'sLORETA'):
            raise ValueError("inv=%r (method=%r)" % (inv, method))

        if depth is not None:
            depth = float(depth)
            if not 0 <= depth <= 1:
                raise ValueError("inv=%r (depth=%r, needs to be in range "
                                 "[0, 1])" % (inv, depth))
            elif depth == 0.8:
                raise ValueError("inv=%r (depth=0.8 is the default, use "
                                 "depth=None instead)" % (inv,))

        return ori, snr, method, depth, bool(pick_normal)

    @classmethod
    def _eval_inv(cls, inv):
        cls._inv_params(inv)
        return inv

    def _post_set_inv(self, _, inv):
        if '*' in inv:
            self._params['make_inv_kw'] = None
            self._params['apply_inv_kw'] = None
            return

        ori, snr, method, depth, pick_normal = self._inv_params(inv)

        if ori == 'fixed':
            make_kw = {'fixed': True, 'loose': None}
        elif ori == 'free':
            make_kw = {'loose': 1}
        elif isinstance(ori, float):
            make_kw = {'loose': ori}
        else:
            raise RuntimeError("ori=%r (in inv=%r)" % (ori, inv))

        if depth is None:
            make_kw['depth'] = 0.8
        elif depth == 0:
            make_kw['depth'] = None
        else:
            make_kw['depth'] = depth

        apply_kw = {'method': method, 'lambda2': 1. / snr ** 2}
        if pick_normal:
            apply_kw['pick_normal'] = pick_normal

        self._params['make_inv_kw'] = make_kw
        self._params['apply_inv_kw'] = apply_kw

    def _eval_model(self, model):
        if model == '':
            return model
        elif len(model) > 1 and '*' in model:
            raise ValueError("Specify model with '%' instead of '*'")

        factors = [v.strip() for v in model.split('%')]

        # find order value for each factor
        ordered_factors = {}
        unordered_factors = []
        for factor in sorted(factors):
            assert_is_legal_dataset_key(factor)
            if factor in self._model_order:
                ordered_factors[self._model_order.index(factor)] = factor
            else:
                unordered_factors.append(factor)

        # recompose
        model = [ordered_factors[v] for v in sorted(ordered_factors)]
        if unordered_factors:
            model.extend(unordered_factors)
        return '%'.join(model)

    def _update_mrisubject(self, fields):
        subject = fields['subject']
        if subject == '*':
            return '*'
        mri = fields['mri']
        return self._mri_subjects[mri][subject]

    def _update_session(self, fields):
        epoch = self._epochs.get(fields['epoch'])
        if epoch is None:
            return '*'
        if isinstance(epoch, (PrimaryEpoch, SecondaryEpoch)):
            return epoch.session

    def _update_src_name(self, fields):
        "Becuase 'ico-4' is treated in filenames  as ''"
        return '' if fields['src'] == 'ico-4' else fields['src']

    def _eval_parc(self, parc):
        if parc in self._parcs:
            if isinstance(self._parcs[parc], SeededParcellation):
                raise ValueError("Seeded parc set without size, use e.g. "
                                 "parc='%s-25'" % parc)
            else:
                return parc
        m = SEEDED_PARC_RE.match(parc)
        if m:
            name = m.group(1)
            if isinstance(self._parcs.get(name), SeededParcellation):
                return parc
            else:
                raise ValueError("No seeded parc with name %r" % name)
        else:
            raise ValueError("parc=%r" % parc)

    def _get_parc(self):
        parc = self.get('parc')
        if parc == '':
            return '', None
        elif parc in self._parcs:
            return parc, self._parcs[parc]
        else:
            return parc, self._parcs[SEEDED_PARC_RE.match(parc).group(1)]

    def _post_set_test(self, _, test):
        if test != '*':
            test_obj = self._tests[test]
            if test_obj.model is not None:
                self.set(model=test_obj.model)

    def _set_analysis_options(self, data, sns_baseline, src_baseline, pmin,
                              tstart, tstop, parc, mask=None,
                              dims=('source', 'time'), decim=None,
                              test_options=(), folder_options=()):
        """Set templates for paths with test parameters

        analysis:  preprocessing up to source estimate epochs (not parcellation)
        folder: parcellation (human readable)
        data_parc: parcellation (as used for spatio-temporal cluster test
        test_options: baseline, permutation test method etc.

        also sets `parc`

        Parameters
        ----------
        data : 'sensor' | 'source'
            Whether the analysis is in sensor or source space.
        ...
        src_baseline :
            Should be False if data=='sensor'.
        ...
        dims : sequence of str
            Dimensions included in the analysis.
        decim : int
            Decimation factor (default is None, i.e. based on epochs).
        test_options : sequence of str
            Additional, test-specific tags.
        """
        # data kind (sensor or source space)
        if data == 'sensor':
            analysis = '{sns_kind} {evoked_kind}'
        elif data == 'source':
            analysis = '{src_kind} {evoked_kind}'
        elif data == 'eeg':
            analysis = '{eeg_kind} {evoked_kind}'
        else:
            raise ValueError("data=%r. Needs to be 'sensor', 'source' or 'eeg'" % data)

        # determine report folder
        # TODO: for sensor space data; {parc} needed for ROI but not sensor
        # tests with dims=('time'). Move into analysis-like compound?
        # Add a global {pipeline} compound that can span multiple folders?
        folder = "{parc}"
        if 'source' in dims:
            if parc is None:
                if mask:
                    folder = "%s masked" % mask
                else:
                    folder = "Whole Brain"
            elif mask:
                raise ValueError("Can't specify mask together with parc")
            elif pmin is None or pmin == 'tfce':
                msg = ("Threshold-free test (pmin=%r) is not implemented for "
                       "parcellation (parc parameter). Use a mask instead, or "
                       "do a cluster-based test." % pmin)
                raise NotImplementedError(msg)

        if folder_options:
            folder += ' ' + ' '.join(folder_options)

        # test properties
        items = []

        # baseline (default is baseline correcting in sensor space)
        epoch_baseline = self._epochs[self.get('epoch')].baseline
        if src_baseline:
            if sns_baseline is True or sns_baseline == epoch_baseline:
                items.append('snsbl')
            elif sns_baseline:
                items.append('snsbl=%s' % _time_window_str(sns_baseline))

            if src_baseline is True or src_baseline == epoch_baseline:
                items.append('srcbl')
            else:
                items.append('srcbl=%s' % _time_window_str(src_baseline))
        else:
            if not sns_baseline:
                items.append('nobl')
            elif sns_baseline is True or sns_baseline == epoch_baseline:
                pass
            else:
                items.append('bl=%s' % _time_window_str(sns_baseline))

        # pmin
        if pmin is not None:
            # source connectivity
            connectivity = self.get('connectivity')
            if connectivity and data != 'source':
                raise NotImplementedError("connectivity=%r is not implemented "
                                          "for data=%r" % (connectivity, data))
            elif connectivity:
                items.append(connectivity)

            items.append(str(pmin))

            # cluster criteria
            if pmin != 'tfce':
                select_clusters = self.get('select_clusters')
                if select_clusters:
                    items.append(select_clusters)

        # time window
        if tstart is not None or tstop is not None:
            items.append(_time_window_str((tstart, tstop)))
        if decim is not None:
            assert isinstance(decim, int)
            items.append(str(decim))

        # parc and data_parc args from parc/mask
        kwargs = {}
        if data == 'source':
            if parc:
                kwargs['parc'] = parc
                kwargs['data_parc'] = parc
            elif mask:
                kwargs['parc'] = mask
                if pmin is None:
                    # When not doing clustering, parc does not affect results,
                    # so we don't need to distinguish parc and mask
                    kwargs['data_parc'] = mask
                else:  # parc means disconnecting
                    kwargs['data_parc'] = '%s-mask' % mask
            else:
                # only compute unmasked test once (probably rare anyways)
                kwargs['parc'] = 'aparc'
                kwargs['data_parc'] = 'unmasked'
        elif parc:
            raise ValueError("Analysis with data=%r can't have parc" % (data,))

        items.extend(test_options)

        self.set(test_options=' '.join(items), analysis=analysis, folder=folder,
                 **kwargs)

    def show_bad_channels(self):
        """List bad channels for each subject/session combination

        Notes
        -----
        ICA Raw pipes merge bad channels from different sessions (by combining
        the bad channels from all sessions).
        """
        bad_channels = {k: self.load_bad_channels() for k in
                        self.iter(('subject', 'session'))}

        # whether they are equal between sessions
        bad_by_s = {}
        for (subject, session), bads in bad_channels.iteritems():
            if subject in bad_by_s:
                if bad_by_s[subject] != bads:
                    sessions_congruent = False
                    break
            else:
                bad_by_s[subject] = bads
        else:
            sessions_congruent = True

        # display
        if sessions_congruent:
            print("All sessions equal:")
            for subject in sorted(bad_by_s):
                print("%s: %s" % (subject, bad_by_s[subject]))
        else:
            subject_len = 1
            session_len = 1
            for subject, session in bad_channels:
                subject_len = max(subject_len, len(subject))
                session_len = max(session_len, len(session))

            template = '{:%i} {:%i}: {}' % (subject_len + 1, session_len + 1)
            for subject, session in sorted(bad_channels):
                print(template.format(subject, session,
                                      bad_channels[subject, session]))

    def show_file_status(self, temp, col=None, row='subject', *args, **kwargs):
        """Compile a table about the existence of files

        Parameters
        ----------
        temp : str
            The name of the path template for the files to examine.
        col : None | str
            Field over which to alternate columns (default is a single column).
        row : str
            Field over which to alternate rows (default 'subject').
        count : bool
            Add a column with a number for each line (default True).
        present : 'time' | 'date' | str
            String to display when a given file is present. 'time' to use last
            modification date and time (default); 'date' for date only.
        absent : str
            String to display when a given file is absent (default '-').
        ... :
            :meth:`MneExperiment.iter` parameters.

        Examples
        --------
        >>> e.show_file_status('rej-file')
            Subject   Rej-file
        -------------------------------
        0   A0005     07/22/15 13:03:08
        1   A0008     07/22/15 13:07:57
        2   A0028     07/22/15 13:22:04
        3   A0048     07/22/15 13:25:29
        >>> e.show_file_status('rej-file', 'raw')
            Subject   0-40   0.1-40              1-40   Clm
        ---------------------------------------------------
        0   A0005     -      07/22/15 13:03:08   -      -
        1   A0008     -      07/22/15 13:07:57   -      -
        2   A0028     -      07/22/15 13:22:04   -      -
        3   A0048     -      07/22/15 13:25:29   -      -
        """
        return FileTree.show_file_status(self, temp, row, col, *args, **kwargs)

    def show_raw_info(self):
        "Display the selected pipeline for raw processing"
        raw = self.get('raw')
        pipe = source_pipe = self._raw[raw]
        pipeline = [pipe]
        while source_pipe.name != 'raw':
            source_pipe = source_pipe.source
            pipeline.insert(0, source_pipe)
        print("Preprocessing pipeline: " +
              ' --> '.join(p.name for p in pipeline))

        # pipe-specific
        if isinstance(pipe, RawICA):
            subjects = []
            statuses = []
            for s in self:
                subjects.append(s)
                filename = self.get('raw-ica-file')
                if exists(filename):
                    ica = self.load_ica()
                    status = "%i components rejected" % len(ica.exclude)
                else:
                    status = "No ICA-file"
                statuses.append(status)
            ds = Dataset()
            ds['subject'] = Factor(subjects)
            ds['status'] = Factor(statuses)
            print()
            print(ds)

    def show_reg_params(self, asds=False, **kwargs):
        """Show the covariance matrix regularization parameters

        Parameters
        ----------
        asds : bool
            Return a dataset with the parameters (default False).
        ...
            State parameters.
        """
        if kwargs:
            self.set(**kwargs)
        subjects = []
        reg = []
        for subject in self:
            path = self.get('cov-info-file')
            if exists(path):
                with open(path, 'r') as fid:
                    text = fid.read()
                reg.append(float(text.strip()))
            else:
                reg.append(float('nan'))
            subjects.append(subject)
        ds = Dataset()
        ds['subject'] = Factor(subjects)
        ds['reg'] = Var(reg)
        if asds:
            return ds
        else:
            print(ds)

    def show_rej_info(self, flagp=None, asds=False):
        """Information about artifact rejection

        Parameters
        ----------
        flagp : scalar
            Flag entries whose percentage of good trials is lower than this
            number.
        asds : bool
            Return a Dataset with the information (default is to print it).
        """
        # TODO: include ICA raw preprocessing pipes
        raw_name = self.get('raw')
        epoch_name = self.get('epoch')
        rej_name = self.get('rej')
        rej = self._artifact_rejection[rej_name]
        has_ica = rej['kind'] == 'ica'
        has_interp = rej.get('interpolation')

        subjects = []
        n_events = []
        n_good = []
        bad_chs = []
        if has_interp:
            n_interp = []
        if has_ica:
            n_ics = []

        for subject in self:
            subjects.append(subject)
            bads_raw = self.load_bad_channels()
            try:
                ds = self.load_selected_events(reject='keep')
            except FileMissing:
                ds = self.load_selected_events(reject=False)
                n_good.append(float('nan'))
                bad_chs.append(str(len(bads_raw)))
                if has_interp:
                    n_interp.append(float('nan'))
            else:
                n_good.append(ds['accept'].sum())
                bads_rej = set(ds.info[BAD_CHANNELS]).difference(bads_raw)
                bad_chs.append("%i + %i" % (len(bads_raw), len(bads_rej)))
                if has_interp:
                    n_interp.append(np.mean([len(chi) for chi in ds[INTERPOLATE_CHANNELS]]))
            n_events.append(ds.n_cases)
            if has_ica:
                ica_path = self.get('ica-file')
                if exists(ica_path):
                    ica = mne.preprocessing.read_ica(ica_path)
                    n_ics.append(len(ica.exclude))
                else:
                    n_ics.append(np.nan)

        caption = ("Rejection info for raw=%s, epoch=%s, rej=%s. "
                   "Percent is rounded to one decimal. Bad channels: "
                   "defined in bad_channels file and in rej-file." %
                   (raw_name, epoch_name, rej_name))
        if has_interp:
            caption += (" ch_interp: average number of channels interpolated "
                        "per epoch, rounded to one decimal.")
        else:
            caption += " Channel interpolation disabled."
        out = Dataset(caption=caption)
        out['subject'] = Factor(subjects)
        out['n_events'] = Var(n_events)
        out['n_good'] = Var(n_good)
        out['percent'] = Var(np.round(100 * out['n_good'] / out['n_events'], 1))
        if flagp:
            out['flag'] = Factor(out['percent'] < flagp, labels={False: '', True: '*'})
        out['bad_channels'] = Factor(bad_chs)
        if has_interp:
            out['ch_interp'] = Var(np.round(n_interp, 1))
        if has_ica:
            out['ics_rejected'] = Var(n_ics)

        if asds:
            return out
        else:
            print(out)

    def show_subjects(self, mri=True, mrisubject=False, caption=True,
                      asds=False, **state):
        """Create a Dataset with subject information

        Parameters
        ----------
        mri : bool
            Add a column specifying whether the subject is using a scaled MRI
            or whether it has its own MRI.
        mrisubject : bool
            Add a column showing the MRI subject corresponding to each subject.
        caption : bool | str
            Caption for the table (For True, use the default "Subject in group
            {group}".
        asds : bool
            Return the table as Dataset instead of an FMTxt Table.
        ...
            State parameters.
        """
        if state:
            self.set(**state)

        # caption
        if caption is True:
            caption = self.format("Subjects in group {group}")

        subject_list = []
        mri_list = []
        mrisubject_list = []
        for subject in self.iter():
            subject_list.append(subject)
            mrisubject_ = self.get('mrisubject')
            mrisubject_list.append(mrisubject_)
            if mri:
                mri_dir = self.get('mri-dir')
                if not exists(mri_dir):
                    mri_list.append('*missing')
                elif is_fake_mri(mri_dir):
                    mri_sdir = self.get('mri-sdir')
                    info = mne.coreg.read_mri_cfg(mrisubject_, mri_sdir)
                    cell = "%s * %s" % (info['subject_from'],
                                        str(info['scale']))
                    mri_list.append(cell)
                else:
                    mri_list.append(mrisubject_)

        ds = Dataset(caption=caption)
        ds['subject'] = Factor(subject_list)
        if mri:
            ds['mri'] = Factor(mri_list)
        if mrisubject:
            ds['mrisubject'] = Factor(mrisubject_list)

        if asds:
            return ds
        else:
            return ds.as_table(midrule=True, count=True)

    def show_input_tree(self):
        """Print a tree of the files needed as input

        See Also
        --------
        show_tree: show complete tree (including secondary, optional and cache)
        """
        return self.show_tree(fields=['raw-file', 'trans-file', 'mri-dir'])

    def _surfer_plot_kwargs(self, surf=None, views=None, foreground=None,
                            background=None, smoothing_steps=None, hemi=None):
        out = self._brain_plot_defaults.copy()
        out.update(self.brain_plot_defaults)
        if views:
            out['views'] = views
        else:
            parc, p = self._get_parc()
            if p is not None and p.views:
                out['views'] = p.views

        if surf:
            out['surf'] = surf
        if foreground:
            out['foreground'] = foreground
        if background:
            out['background'] = background
        if smoothing_steps:
            out['smoothing_steps'] = smoothing_steps
        if hemi:
            out['hemi'] = hemi
        return out
