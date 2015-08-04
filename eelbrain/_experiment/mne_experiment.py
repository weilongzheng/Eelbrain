# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from __future__ import print_function

from collections import defaultdict
import inspect
from itertools import chain, izip, product
import logging
import os
import re
import shutil
import time
from warnings import warn

import numpy as np

import mne
from mne.baseline import rescale
from mne.minimum_norm import (make_inverse_operator, apply_inverse,
                              apply_inverse_epochs)

from .. import _report
from .. import gui
from .. import load
from .. import plot
from .. import save
from .. import table
from .. import testnd
from .. import Dataset, Factor, Var, NDVar, combine
from .._info import BAD_CHANNELS
from .._names import INTERPOLATE_CHANNELS
from .._mne import dissolve_label, \
    labels_from_mni_coords, rename_label, combination_label, \
    morph_source_space, shift_mne_epoch_trigger
from ..mne_fixes import write_labels_to_annot
from ..mne_fixes import _interpolate_bads_eeg, _interpolate_bads_meg
from .._data_obj import (isvar, asfactor, align, DimensionMismatchError,
                         as_legal_dataset_key, assert_is_legal_dataset_key,
                         cwt_morlet, OldVersionError)
from ..fmtxt import List, Report
from .._report import named_list
from .._resources import predefined_connectivity
from .._stats import spm
from .._stats.stats import ttest_t
from .._utils import subp, ui, keydefaultdict
from .._utils.mne_utils import fix_annot_names, is_fake_mri
from .experiment import FileTree
from .definitions import find_dependent_epochs, find_epochs_vars, find_test_vars


__all__ = ['MneExperiment']


# Allowable epoch parameters
EPOCH_PARAMS = {'sel_epoch', 'sel', 'tmin', 'tmax', 'decim', 'baseline', 'n_cases',
                'vars', 'trigger_shift', 'post_baseline_trigger_shift',
                'post_baseline_trigger_shift_max',
                'post_baseline_trigger_shift_min'}
SECONDARY_EPOCH_PARAMS = {'base', 'sel', 'tmin', 'tmax', 'decim', 'baseline',
                          'vars', 'trigger_shift', 'post_baseline_trigger_shift',
                          'post_baseline_trigger_shift_max',
                          'post_baseline_trigger_shift_min'}
SUPER_EPOCH_PARAMS = {'sub_epochs'}
SUPER_EPOCH_INHERITED_PARAMS = {'tmin', 'tmax', 'decim', 'baseline'}

FS_PARC = 'subject_parc'  # Parcellation that come with every MRI-subject
FSA_PARC = 'fsaverage_parc'  # Parcellation that comes with fsaverage
EELBRAIN_PARC = 'eelbrain_parc'
SEEDED_PARC_RE = re.compile('(\w+)-(\d+)$')

inv_re = re.compile("(free|fixed|loose\.\d+)-"  # orientation constraint
                    "(\d*\.?\d+)-"  # SNR
                    "(MNE|dSPM|sLORETA)"  # method
                    "(?:-(0?\.\d+))?"  # depth weighting
                    "(?:-(pick_normal|pick\+|pick-))?"  # pick component
                    "(?:-sm(\d+))?"  # smoothing (in even mm)
                    "$")


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


temp = {'eelbrain-log-file': os.path.join('{root}', 'eelbrain {experiment}.log'),
        'old-eelbrain-log-file': os.path.join('{root}', '.eelbrain.log'),

        # MEG
        'experiment': None,
        'modality': ('', 'eeg', 'meeg'),
        'reference': ('', 'mastoids'),  # EEG reference
        'equalize_evoked_count': ('', 'eq'),
        # locations
        'meg-sdir': os.path.join('{root}', 'meg'),
        'meg-dir': os.path.join('{meg-sdir}', '{subject}'),
        'raw-dir': '{meg-dir}',

        # raw input files
        'raw-file': os.path.join('{raw-dir}', '{subject}_{experiment}-raw.fif'),
        'trans-file': os.path.join('{raw-dir}', '{mrisubject}-trans.fif'),
        # log-files (eye-tracker etc.)
        'log-dir': os.path.join('{meg-dir}', 'logs'),
        'log-rnd': '{log-dir}/rand_seq.mat',
        'log-data-file': '{log-dir}/data.txt',
        'log-file': '{log-dir}/log.txt',
        'edf-file': os.path.join('{log-dir}', '*.edf'),

        # created input files
        'bads-file': os.path.join('{raw-dir}', '{subject}_{bads-compound}-bad_channels.txt'),
        'rej-dir': os.path.join('{meg-dir}', 'epoch selection'),
        'rej-file': os.path.join('{rej-dir}', '{experiment}_{sns_kind}_{epoch}-{rej}.pickled'),

        # cache
        'cache-dir': os.path.join('{root}', 'eelbrain-cache'),
        'cache-state-file': os.path.join('{cache-dir}', 'cache-state.pickle'),
        # raw
        'raw-cache-dir': os.path.join('{cache-dir}', 'raw'),
        'raw-cache-base': os.path.join('{raw-cache-dir}', '{subject}', '{experiment} {raw_kind}'),
        'cached-raw-file': '{raw-cache-base}-raw.fif',
        'event-file': '{raw-cache-base}-evts.pickled',
        # mne secondary/forward modeling
        'proj-file': '{raw-cache-base}_{proj}-proj.fif',
        'fwd-file': '{raw-cache-base}_{mrisubject}-{src}-fwd.fif',
        # sensor covariance
        'cov-dir': os.path.join('{cache-dir}', 'cov'),
        'cov-base': os.path.join('{cov-dir}', '{subject}', '{experiment} '
                                 '{raw_kind} {cov}-{cov-rej}-{proj}'),
        'cov-file': '{cov-base}-cov.fif',
        'cov-info-file': '{cov-base}-info.txt',
        # evoked
        'evoked-dir': os.path.join('{cache-dir}', 'evoked'),
        'evoked-base': os.path.join('{evoked-dir}', '{subject}', '{experiment} '
                                    '{sns_kind} {epoch} {model} {evoked_kind}'),
        'evoked-file': os.path.join('{evoked-base}.pickled'),
        'evoked-fiff-file': os.path.join('{evoked-base}-ave.fif'),
        # test files
        'test-dir': os.path.join('{cache-dir}', 'test'),
        'data_parc': 'unmasked',
        'test-file': os.path.join('{test-dir}', '{analysis} {group}',
                                  '{epoch} {test} {test_options} {data_parc}.pickled'),

        # MRIs
        'common_brain': 'fsaverage',
        # MRI base files
        'mri-sdir': os.path.join('{root}', 'mri'),
        'mri-dir': os.path.join('{mri-sdir}', '{mrisubject}'),
        'bem-dir': os.path.join('{mri-dir}', 'bem'),
        'mri-cfg-file': os.path.join('{mri-dir}', 'MRI scaling parameters.cfg'),
        'mri-file': os.path.join('{mri-dir}', 'mri', 'orig.mgz'),
        'bem-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem.fif'),
        'bem-sol-file': os.path.join('{bem-dir}', '{mrisubject}-*-bem-sol.fif'),
        'head-bem-file': os.path.join('{bem-dir}', '{mrisubject}-head.fif'),
        'src-file': os.path.join('{bem-dir}', '{mrisubject}-{src}-src.fif'),
        # Labels
        'hemi': ('lh', 'rh'),
        'label-dir': os.path.join('{mri-dir}', 'label'),
        'annot-file': os.path.join('{label-dir}', '{hemi}.{parc}.annot'),

        # (method) plots
        'plot-dir': os.path.join('{root}', 'plots'),
        'plot-file': os.path.join('{plot-dir}', '{analysis}', '{name}.{ext}'),

        # general analysis parameters
        'analysis': '',  # analysis parameters (sns_kind, src_kind, ...)
        'test_options': '',
        'name': '',

        # result output files
        # data processing parameters
        #    > group
        #        > kind of test
        #    > single-subject
        #        > kind of test
        #            > subject
        'folder': '',  # intermediate folder for deep files
        'resname': '',  # analysis name (GA-movie, max plot, ...)
        'ext': 'pickled',  # file extension
        'res-dir': os.path.join('{root}', 'results'),
        'res-file': os.path.join('{res-dir}', '{analysis}', '{resname}.{ext}'),
        'res-deep-file': os.path.join('{res-dir}', '{analysis}', '{folder}',
                                      '{resname}.{ext}'),
        'report-file': os.path.join('{res-dir}', '{analysis} {group}', '{folder}',
                                    '{epoch} {test} {test_options}.html'),
        'group-mov-file': os.path.join('{res-dir}', '{analysis} {group}',
                                       '{epoch} {test_options} {resname}.mov'),
        'subject-mov-file': os.path.join('{res-dir}', '{analysis} subjects',
                                         '{epoch} {test_options} {resname}',
                                         '{subject}.mov'),

        # besa
        'besa-root': os.path.join('{root}', 'besa'),
        'besa-trig': os.path.join('{besa-root}', '{subject}', '{subject}_'
                                  '{experiment}_{epoch}_triggers.txt'),
        'besa-evt': os.path.join('{besa-root}', '{subject}', '{subject}_'
                                 '{experiment}_{epoch}[{rej}].evt'),

        # MRAT
        'mrat_condition': '',
        'mrat-root': os.path.join('{root}', 'mrat'),
        'mrat-sns-root': os.path.join('{mrat-root}', '{sns_kind}',
                                      '{epoch} {model} {evoked_kind}'),
        'mrat-src-root': os.path.join('{mrat-root}', '{src_kind}',
                                      '{epoch} {model} {evoked_kind}'),
        'mrat-sns-file': os.path.join('{mrat-sns-root}', '{mrat_condition}',
                                      '{mrat_condition}_{subject}-ave.fif'),
        'mrat_info-file': os.path.join('{mrat-root}', '{subject} info.txt'),
        'mrat-src-file': os.path.join('{mrat-src-root}', '{mrat_condition}',
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

    Notes
    -----
    .. seealso::
        Guide on using :ref:`experiment-class-guide`.
    """
    path_version = None
    auto_delete_cache = True
    # what to do when the experiment class definition changed:
    #   True: delete outdated files
    #   False: raise an error
    #   'disable': ignore it
    #   'debug': prompt for what to do in the terminal

    # Experiment Constants
    # ====================

    # add this value to all trigger times
    trigger_shift = 0

    # variables for automatic labeling {name: {trigger: label, triggers: label}}
    variables = {}

    # Default values for epoch definitions
    _epoch_default = {'tmin':-0.1, 'tmax': 0.6, 'decim': 5, 'baseline': (None, 0)}
    epoch_default = {}

    # named epochs
    epochs = {'epoch': dict(sel="stim=='target'"),
              'cov': dict(sel_epoch='epoch', tmin=-0.1, tmax=0)}
    # Rejection
    # =========
    # eog_sns: The sensors to plot separately in the rejection GUI. The default
    # is the two MEG sensors closest to the eyes.
    _eog_sns = {'KIT-NY': ['MEG 143', 'MEG 151'],
                'KIT-AD': ['MEG 087', 'MEG 130'],
                'KIT-BRAINVISION': ['HEOGL', 'HEOGR', 'VEOGb']}
    #
    # epoch_rejection dict:
    #
    # kind : 'manual' | 'make' | None
    #     How the rejection is derived:
    #     'manual': manually create a rejection file (use the selection GUI
    #     through .make_rej())
    #     'make' a rejection file is created by the user
    # cov-rej : str
    #     rej setting to use for computing the covariance matrix. Default is
    #     same as rej.
    #
    # For manual rejection
    # ^^^^^^^^^^^^^^^^^^^^
    # decim : int
    #     Decim factor for the rejection GUI (default is to use epoch setting).
    _epoch_rejection = {'': {'kind': None},
                        'man': {'kind': 'manual', 'interpolation': True}}
    epoch_rejection = {}

    exclude = {}  # field_values to exclude (e.g. subjects)

    # groups can be defined as subject lists: {'group': ('member1', 'member2', ...)}
    # or by exclusion: {'group': {'base': 'all', 'exclude': ('member1', 'member2')}}
    groups = {}

    # whether to look for and load eye tracker data when loading raw files
    has_edf = defaultdict(lambda: False)

    # raw processing settings {name: (args, kwargs)}
    _raw = {'clm': None,
            '0-40': ((None, 40), {'method': 'iir'}),
            '0.1-40': ((0.1, 40), {'l_trans_bandwidth': 0.08,
                                   'filter_length': '60s'}),
            '1-40': ((1, 40), {'method': 'iir'})}

    # projection definition:
    # "base": 'raw' for raw file, or epoch name
    # "rej": rejection setting to use (only applies for epoch projs)
    # r.g. {'ironcross': {'base': 'adj', 'rej': 'man'}}
    projs = {}

    # Pattern for subject names. The first group is used to determine what
    # MEG-system the data was recorded from
    _subject_re = '(R|A|Y|AD|QP)(\d{3,})$'
    _meg_systems = {'R': 'KIT-NY',
                    'A': 'KIT-AD', 'Y': 'KIT-AD', 'AD': 'KIT-AD', 'QP': 'KIT-AD'}

    # kwargs for regularization of the covariance matrix (see .make_cov())
    _covs = {'auto': {'epoch': 'cov', 'method': 'auto'},
             'bestreg': {'epoch': 'cov', 'reg': 'best'},
             'reg': {'epoch': 'cov', 'reg': True},
             'noreg': {'epoch': 'cov', 'reg': None}}

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
    __parcs = {'aparc.a2005s': FS_PARC,
               'aparc.a2009s': FS_PARC,
               'aparc': FS_PARC,
               'aparc.DKTatlas': FS_PARC,
               'PALS_B12_Brodmann': FSA_PARC,
               'PALS_B12_Lobes': FSA_PARC,
               'PALS_B12_OrbitoFrontal': FSA_PARC,
               'PALS_B12_Visuotopic': FSA_PARC,
               'lobes': {'kind': EELBRAIN_PARC, 'make': True,
                         'morph_from_fsaverage': True},
               'lobes-op': {'kind': 'combination',
                            'base': 'lobes',
                            'labels': {'occipitoparietal': "occipital + parietal"}},
               'lobes-ot': {'kind': 'combination',
                            'base': 'lobes',
                            'labels': {'occipitotemporal': "occipital + temporal"}}}
    parcs = {}

    # Frequencies
    _freqs = {'gamma': {'frequencies': np.arange(25, 50, 2), # lowbound, highbound, step
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
    _backup_state = {'subject': '*', 'mrisubject': '*', 'experiment': '*',
                     'raw': 'clm', 'modality': '*'}
    # files to back up, together with state modifications on the basic state
    _backup_files = (('raw-file', {}),
                     ('bads-file', {}),
                     ('rej-file', {'raw': '*', 'epoch': '*', 'rej': '*'}),
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
    _cluster_criteria = {'': {'time': 0.025, 'sensor': 4, 'source': 10},
                         'all': {},
                         'large': {'time': 0.025, 'sensor': 8, 'source': 20}}

    # plotting
    # --------
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
        self.groups = self.groups.copy()
        self.projs = self.projs.copy()
        self._mri_subjects = self._mri_subjects.copy()
        self._templates = self._templates.copy()
        # templates version
        if self.path_version is None:
            warn("The default for MneExperiment.path_version will change to 1"
                 "after version 0.20. To keep using the old scheme, please "
                 "specify MneExperiment.path_version = 0 explicitly.",
                 DeprecationWarning)

        if self.path_version is None or self.path_version == 0:
            self._templates['raw-dir'] = os.path.join('{meg-dir}', 'raw')
            self._templates['raw-file'] = os.path.join('{raw-dir}', '{subject}_'
                                              '{experiment}_{raw_kind}-raw.fif')
        elif self.path_version != 1:
            raise ValueError("path_version needs to be 0 or 1")
        # update templates with _values
        for cls in reversed(inspect.getmro(self.__class__)):
            if hasattr(cls, '_values'):
                self._templates.update(cls._values)

        FileTree.__init__(self)

        ########################################################################
        # subjects
        if root is None:
            find_subjects = False
        else:
            self.set(root=root)

        if find_subjects:
            sub_dir = self.get(self._subject_loc)
            if not os.path.exists(sub_dir):
                raise IOError("Subjects directory not found: %s. Initialize "
                              "with root=None or find_subjects=False" % sub_dir)
            subjects = sorted(s for s in os.listdir(sub_dir) if
                              self._subject_re.match(s) and
                              os.path.isdir(os.path.join(sub_dir, s)))

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
                    group_members = (s for s in subjects if s in group_def)
                else:
                    raise TypeError("group %s=%r" % (name, group_def))
                groups[name] = tuple(group_members)
                group_definitions.pop(name)
            if len(group_definitions) == n_def:
                raise ValueError("Groups contain unresolvable definition")
        self._groups = groups

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
        epochs = {}
        secondary_epochs = []
        super_epochs = {}
        for name, parameters in self.epochs.iteritems():
            # filter out secondary epochs
            if 'sub_epochs' in parameters:
                if set(parameters).difference(SUPER_EPOCH_PARAMS):
                    msg = ("Super-epochs can only have one parameters called "
                           "'sub_epochs'; got %r" % parameters)
                    raise ValueError(msg)
                super_epochs[name] = parameters.copy()
            elif 'base' in parameters:
                if set(parameters).difference(SECONDARY_EPOCH_PARAMS):
                    raise ValueError("Invalid key in epoch definition for %s: %s"
                                     % (name, parameters))
                secondary_epochs.append((name, parameters))
            elif set(parameters).difference(EPOCH_PARAMS):
                raise ValueError("Invalid key in epoch definition for %s: %s"
                                 % (name, parameters))
            elif 'sel_epoch' in parameters and 'n_cases' in parameters:
                raise ValueError("Epoch %s can not have both sel_epochs and "
                                 "n_cases entries" % name)
            else:
                epochs[name] = epoch = self._epoch_default.copy()
                epoch.update(self.epoch_default)
                epoch.update(parameters)
        # integrate epochs with base
        while secondary_epochs:
            n_secondary_epochs = len(secondary_epochs)
            for i in xrange(n_secondary_epochs - 1, -1, -1):
                name, parameters = secondary_epochs[i]
                if parameters['base'] in epochs:
                    epoch = epochs[parameters['base']].copy()
                    epoch['sel_epoch'] = parameters['base']
                    for name_ in ('sel', 'vars', 'trigger_shift'):
                        if name_ in epoch:
                            del epoch[name_]
                    epoch.update(parameters)
                    epochs[name] = epoch
                    del secondary_epochs[i]
            if len(secondary_epochs) == n_secondary_epochs:
                txt = ' '.join('Epoch %s has non-existing base %r.' % p for p in secondary_epochs)
                raise ValueError("Invalid epoch definition: %s" % txt)
        # re-integrate super-epochs
        for name, parameters in super_epochs.iteritems():
            sub_epochs = parameters['sub_epochs']

            # make sure definition is not recursive
            if any('sub_epochs' in epochs[n] for n in sub_epochs):
                msg = ("Super-epochs can't be defined recursively (%r)" % name)
                raise ValueError(msg)

            # find params
            for param in SUPER_EPOCH_INHERITED_PARAMS:
                values = {epochs[n][param] for n in sub_epochs}
                if len(values) > 1:
                    param_repr = ', '.join(repr(v) for v in values)
                    msg = ("All sub_epochs of a super-epoch must have the "
                           "same setting for %r; super-epoch %r got {%s}."
                           % (param, name, param_repr))
                    raise ValueError(msg)
                parameters[param] = values.pop()
        epochs.update(super_epochs)
        # add name
        for name, epoch in epochs.iteritems():
            epoch['name'] = name
        # find rej-files needed for each epoch (for cache checking)
        def _rej_epochs(epoch):
            "Find which rej-files an epoch depends on"
            if 'sub_epochs' in epoch:
                names = epoch['sub_epochs']
                return sum((_rej_epochs(epochs[n]) for n in names), ())
            elif 'sel_epoch' in epoch:
                return _rej_epochs(epochs[epoch['sel_epoch']])
            else:
                return (epoch['name'],)
        for name, epoch in epochs.iteritems():
            if 'sub_epochs' in epoch or 'sel_epoch' in epoch:
                epoch['_rej_file_epochs'] = _rej_epochs(epoch)
        # check parameters
        for name, epoch in epochs.iteritems():
            if 'post_baseline_trigger_shift' in epoch:
                if not ('post_baseline_trigger_shift_min' in epoch and
                        'post_baseline_trigger_shift_max' in epoch):
                    raise ValueError("Epoch %s contains post_baseline_trigger_shift "
                                     "but is missing post_baseline_trigger_shift_min "
                                     "and/or post_baseline_trigger_shift_max" % name)
        self._epochs = epochs

        ########################################################################
        # store epoch rejection settings
        epoch_rejection = self._epoch_rejection.copy()
        for name, params in self.epoch_rejection.iteritems():
            if params['kind'] not in ('manual', 'make'):
                raise ValueError("Invalid value in %r rejection setting: "
                                 "kind=%r" % (name, params['kind']))
            epoch_rejection[name] = params.copy()

        self._epoch_rejection = epoch_rejection

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
            if name in parcs:
                raise ValueError("Parcellation %s defined twice" % name)
            elif p == FS_PARC or p == FSA_PARC:
                p = {'kind': p}
            elif isinstance(p, dict):
                if 'kind' not in p:
                    raise KeyError("Parcellation %s does not contain the "
                                   "required 'kind' entry" % name)
                p = p.copy()
            else:
                raise ValueError("Parcellations need to be defined as %r, %r or "
                                 "dict, got %s: %r" % (FS_PARC, FSA_PARC, name, p))

            kind = p['kind']
            if kind == EELBRAIN_PARC:
                pass
            elif kind == FS_PARC:
                if len(p) > 1:
                    raise ValueError("Unknown keys in parcellation %r: %r"
                                     % (name, p))
                p['make'] = False
                p['morph_from_fsaverage'] = False
            elif kind == FSA_PARC:
                if len(p) > 1:
                    raise ValueError("Unknown keys in parcellation %r: %r"
                                     % (name, p))
                p['make'] = False
                p['morph_from_fsaverage'] = True
            elif kind == 'combination':
                if set(p) != {'kind', 'base', 'labels'}:
                    raise ValueError("Incorrect keys in parcellation %r: %r"
                                     % (name, p))
                p['make'] = True
                p['morph_from_fsaverage'] = False
            elif kind == 'seeded':
                if 'seeds' not in p:
                    raise KeyError("Seeded parcellation %s is missing 'seeds' "
                                   "entry" % name)
                unused = set(p).difference({'kind', 'seeds', 'surface', 'mask'})
                if unused:
                    raise ValueError("Seeded parcellation %s has invalid keys "
                                     "%s" % (name, tuple(unused)))
                p['make'] = True
                p['morph_from_fsaverage'] = False
            else:
                raise ValueError("Parcellation %s with invalid  'kind': %r"
                                 % (name, kind))

            parcs[name] = p
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
        tests = {}
        for test, params in self.tests.iteritems():
            # backwards compatibility for old test specification
            if isinstance(params, (tuple, list)):
                warn("MneExperiment.tests should be defined as dictionaries, "
                     "test definitions with tuples/lists will not work after "
                     "version 0.20. Please change your MneExperiment subclass "
                     "definition.", DeprecationWarning)
                kind, model, test_parameter = params
                if kind == 'anova':
                    params = {'kind': kind, 'model': model, 'x': test_parameter}
                elif kind == 'ttest_rel':
                    m = re.match(r"\s*([\w|]+)\s*([<=>])\s*([\w|]+)$", test_parameter)
                    if m is None:
                        raise ValueError("The contrast definition %s for test "
                                         "%s could not be parsed." %
                                         (repr(test_parameter), test))
                    c1, tail, c0 = m.groups()
                    if '|' in c1:
                        c1 = tuple(c1.split('|'))
                        c0 = tuple(c0.split('|'))

                    if tail == '=':
                        tail = 0
                    elif tail == '>':
                        tail = 1
                    elif tail == '<':
                        tail = -1
                    else:
                        raise ValueError("%r in t-test contrast=%r"
                                         % (tail, test_parameter))
                    params = {'kind': kind, 'model': model, 'c1': c1, 'c0': c0,
                              'tail': tail}
                elif kind == 't_contrast_rel':
                    params = {'kind': kind, 'model': model,
                              'contrast': test_parameter}
                else:
                    raise ValueError("Unknown test: %s" % repr(kind))
            elif not isinstance(params, dict):
                raise TypeError("Tests need to be specified as dictionary, "
                                "got %s" % repr(params))
            else:
                params = params.copy()

            # test descriptions
            kind = params['kind']
            if kind == 'anova':
                desc = params['x']
            elif kind == 'ttest_rel':
                tail = params.get('tail', 0)
                if tail == 0:
                    link = ' = '
                elif tail > 0:
                    link = ' > '
                else:
                    link = ' < '
                desc = link.join((params['c1'], params['c0']))
            elif kind == 't_contrast_rel':
                desc = params['contrast']
            elif kind == 'two-stage':
                all_params = set(params)
                missing = {'kind', 'stage 1'}.difference(all_params)
                if missing:
                    raise KeyError("Test definition %s is missing %s"
                                   % (test, named_list(tuple(missing), 'entry')))
                unknown = all_params.difference({'kind', 'stage 1', 'vars'})
                if unknown:
                    raise KeyError("Test definition %s has unknown %s"
                                   % (test, named_list(tuple(unknown), 'entry')))

                desc = params['stage 1']
            elif kind == 'custom':
                desc = 'Custom test'
            else:
                raise NotImplementedError("Invalid test kind: %r" % kind)
            params['desc'] = desc

            tests[test] = params
        self._tests = tests

        ########################################################################
        # Experiment class setup
        ########################
        self._register_field('mri', sorted(self._mri_subjects))
        self._register_field('mrisubject')
        self._register_field('subject', subjects or None)
        self._register_field('group', self._groups.keys(), 'all',
                             post_set_handler=self._post_set_group)

        self._register_field('raw', sorted(self._raw))
        self._register_field('rej', self._epoch_rejection.keys(), 'man',
                             post_set_handler=self._post_set_rej)
        # epoch
        epoch_keys = sorted(self._epochs)
        for default_epoch in epoch_keys:
            if 'sel_epoch' not in self._epochs[default_epoch]:
                break
        else:
            default_epoch = None
        self._register_field('epoch', epoch_keys, default_epoch)
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
        self._register_field('proj', [''] + self.projs.keys())
        self._register_field('freq', self._freqs.keys())
        self._register_field('src', ('ico-4', 'vol-10', 'vol-7', 'vol-5'))
        self._register_field('select_clusters', self._cluster_criteria.keys())

        # compounds
        self._register_compound('bads-compound', ('experiment', 'modality'))
        self._register_compound('raw_kind', ('modality', 'raw'))
        self._register_compound('sns_kind', ('raw_kind', 'proj'))
        self._register_compound('src_kind', ('sns_kind', 'cov', 'mri', 'inv'))
        self._register_compound('evoked_kind', ('rej', 'equalize_evoked_count'))
        self._register_compound('eeg_kind', ('sns_kind', 'reference'))

        # Define make handlers
        self._bind_cache('cached-raw-file', self.make_raw)
        self._bind_cache('cov-file', self.make_cov)
        self._bind_cache('src-file', self.make_src)
        self._bind_cache('fwd-file', self.make_fwd)
        self._bind_make('bem-sol-file', self.make_bem_sol)

        # Check that the template model is complete
        self._find_missing_fields()

        # set initial values
        self.set(**state)
        self.store_state()
        self.brain = None

        ########################################################################
        # Cache
        #######
        if self.exclude:
            raise ValueError("MneExperiment.exclude must be unspecified for "
                             "cache management to work")
        elif not root:
            return

        # logger
        if os.path.exists(self.get('old-eelbrain-log-file')):
            os.rename(self.get('old-eelbrain-log-file'),
                      self.get('eelbrain-log-file'))
        self._log = logging.Logger(self.__class__.__name__, logging.DEBUG)
        handler = logging.FileHandler(self.get('eelbrain-log-file'))
        formatter = logging.Formatter("%(levelname)-8s %(asctime)s %(message)s",
                                      "%m-%d %H:%M")  # %(name)-12s
        handler.setFormatter(formatter)
        self._log.addHandler(handler)
        handler.setLevel(logging.DEBUG)

        # log package versions
        from .. import __version__
        msg = ("%s initialized with root=%r, eelbrain %s, mne %s"
               % (self.__class__.__name__, root, __version__, mne.__version__))
        if any('dev' in v for v in (__version__, mne.__version__)):
            self._log.warn(msg)
            for package, version in (('eelbrain', __version__),
                                     ('mne', mne.__version__)):
                if 'dev' in version:
                    warn("Using %s %s: Using development versions can have "
                         "unanticipated consequences" % (package, version))
        else:
            self._log.info(msg)

        if self.auto_delete_cache == 'disable':
            self._log.warn("Cache-management disabled")
            return

        # collect events for current setup
        with self._temporary_state:
            self.set(raw='clm')
            events = {s: self.load_events(data_raw=False) for s in
                      self.iter(group='all')}

        # check the cache, delete invalid files
        cache_dir = self.get('cache-dir')
        cache_state_path = self.get('cache-state-file')
        if os.path.exists(cache_state_path):
            # check time stamp
            if os.path.getmtime(cache_state_path) > time.time():
                tc = time.ctime(os.path.getmtime(cache_state_path))
                tsys = time.asctime()
                raise RuntimeError("The cache's time stamp is in the future "
                                   "(%s). If the system time (%s) is wrong, "
                                   "adjust the system clock; if not, delete "
                                   "the eelbrain-cache folder." % (tc, tsys))
            cache_state = load.unpickle(cache_state_path)
            cache_state_v = cache_state.get('version', 0)

            # Find modified definitions
            # =========================
            invalid_cache = defaultdict(set)
            # events (subject):  overall change in events
            # variables:  event change restricted to certain variables
            # groups:  change in group members
            # epochs:  change in epoch parameters
            # parcs: parc def change
            # tests: test def change

            # check events (includes trigger_shift)
            cache_events = cache_state['events']
            overlapping_subjects = [s for s in events if s in cache_events]
            for subject in overlapping_subjects:
                new_events = events[subject]
                cached_events = cache_state['events'][subject]
                if new_events.n_cases != cached_events.n_cases:
                    invalid_cache['events'].add(subject)
                    self._log.debug("  event length: %s %i->%i", subject,
                                    cached_events.n_cases, new_events.n_cases)
                elif not np.all(new_events['i_start'] == cached_events['i_start']):
                    invalid_cache['events'].add(subject)
                    self._log.debug("  trigger timing changed: %s", subject)
                else:
                    for var in cached_events:
                        if var == 'i_start':
                            continue
                        elif var not in new_events:
                            invalid_cache['variables'].add(var)
                            self._log.debug("  var removed: %s (%s)", var, subject)
                        elif not np.all(cached_events[var] == new_events[var]):
                            invalid_cache['variables'].add(var)
                            self._log.debug("  var changed: %s (%s)", var, subject)

            # groups
            for group, members in cache_state['groups'].iteritems():
                if group not in self._groups or members != self._groups[group]:
                    invalid_cache['groups'].add(group)
                    self._log.debug("  group: %s" % group)

            # epochs
            for epoch, params in cache_state['epochs'].iteritems():
                if epoch not in self._epochs or params != self._epochs[epoch]:
                    invalid_cache['epochs'].add(epoch)

            # parcs
            for parc, params in cache_state['parcs'].iteritems():
                if parc not in self._parcs:
                    invalid_cache['parcs'].add(parc)
                elif params != self._parcs[parc]:
                    # FS_PARC:  Parcellations that are provided by the user
                    # should not be automatically removed.
                    # FSA_PARC:  for other mrisubjects, the parcellation
                    # should automatically update if the user changes the
                    # fsaverage file.
                    if self._parcs[parc]['kind'] not in (FS_PARC, FSA_PARC):
                        invalid_cache['parcs'].add(parc)

            # tests
            for test, params in cache_state['tests'].iteritems():
                if test not in self._tests or params != self._tests[test]:
                    invalid_cache['tests'].add(test)
                    if test in self._tests:
                        self._log.debug("  test %s: \n   %s\n ->%s", test,
                                        params, self._tests[test])
                    else:
                        self._log.debug("  test %s removed", test)

            # create message here, before secondary invalidations are added
            msg = []
            if cache_state_v < 2:
                msg.append("Check for invalid ANOVA tests (cache version %i)." %
                           cache_state_v)
            if invalid_cache:
                msg.append("Experiment definition changed:")
                for kind, values in invalid_cache.iteritems():
                    msg.append("  %s: %s" % (kind, ', '.join(values)))

            # Secondary  invalidations
            # ========================
            # dependent epochs
            if 'epochs' in invalid_cache:
                for e in tuple(invalid_cache['epochs']):
                    invalid_cache['epochs'].update(find_dependent_epochs(e, cache_state['epochs']))

            # tests/epochs based on variables
            if 'variables' in invalid_cache:
                bad_vars = invalid_cache['variables']
                # tests using bad variable
                for test, params in cache_state['tests'].iteritems():
                    if test not in invalid_cache['tests']:
                        bad = bad_vars.intersection(find_test_vars(params))
                        if bad:
                            invalid_cache['tests'].add(test)
                            self._log.debug("  test %s depends on changed "
                                            "variables %s", test, bad)
                # epochs using bad variable
                epochs_vars = find_epochs_vars(cache_state['epochs'])
                for epoch, evars in epochs_vars.iteritems():
                    bad = bad_vars.intersection(evars)
                    if bad:
                        invalid_cache['epochs'].add(epoch)
                        self._log.debug("  epoch %s depends on changed "
                                        "variables %s", epochs, bad)

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
                    for test, params in self._tests.iteritems():
                        if params['kind'] == 'anova' and params['x'].count('*') > 1:
                            bad_tests.append(test)
                    if bad_tests and bad_parcs:
                        self._log.warning("  Invalid ANOVA tests: %s for %s",
                                          bad_tests, bad_parcs)
                    for test, parc in product(bad_tests, bad_parcs):
                        rm['test-file'].add({'test': test, 'data_parc': parc})
                        rm['report-file'].add({'test': test, 'folder': parc})

                # evoked files are based on old events
                for subject in invalid_cache['events']:
                    rm['evoked-file'].add({'subject': subject})
                # any group involving those subjects is also bad
                for group, members in cache_state['groups'].iteritems():
                    if invalid_cache['events'].intersection(members):
                        invalid_cache['groups'].add(group)

                # variables
                for var in invalid_cache['variables']:
                    rm['evoked-file'].add({'model': '*%s*' % var})

                # groups
                for group in invalid_cache['groups']:
                    rm['test-file'].add({'group': group})
                    rm['group-mov-file'].add({'group': group})
                    rm['report-file'].add({'group': group})

                # epochs
                for epoch in invalid_cache['epochs']:
                    rm['evoked-file'].add({'epoch': epoch})
                    for cov, cov_params in self._covs.iteritems():
                        if cov_params['epoch'] != epoch:
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
                    rm['res-file'].add({'analysis': 'Source Annot',
                                        'resname': parc + ' * *', 'ext': 'p*'})

                # tests
                for test in invalid_cache['tests']:
                    rm['test-file'].add({'test': test})
                    rm['report-file'].add({'test': test})

                # update for "companion" files
                if 'evoked-file' in rm:
                    for arg_dict in rm['evoked-file']:
                        rm['evoked-fiff-file'].add(arg_dict)

                # find actual files to delete
                files = set()
                for temp, arg_dicts in rm.iteritems():
                    keys = self.find_keys(temp, False)
                    for args in arg_dicts:
                        kwargs = {k: args.get(k, '*') for k in keys}
                        files.update(self.glob(temp, vmatch=False, **kwargs))

                # log
                if files:
                    msg.append("Files to be deleted:")
                    msg.extend(sorted('  ' + os.path.relpath(f, root) for f in files))
                else:
                    msg.append("No cache files affected.")
                self._log.debug(os.linesep.join(msg))

                # handle invalid files
                if files:
                    if self.auto_delete_cache is False:
                        msg.append("Automatic cache management disabled. Either "
                                   "revert changes, or set e.auto_delete_cache=True")
                        raise RuntimeError(os.linesep.join(msg))
                    elif self.auto_delete_cache == 'debug':
                        print(os.linesep.join(msg))
                        print("delete:  delete invalid cache files\n"
                              "abort:  raise an error\n"
                              "ignore:  proceed without doing anything\n"
                              "revalidate:  don't delete any cache files but "
                              "write a new cache-state file")
                        while True:
                            command = raw_input(" > ")
                            if command in ('delete', 'abort', 'ignore', 'revalidate'):
                                break
                            else:
                                print("invalid entry")

                        if command == 'delete':
                            pass
                        elif command == 'abort':
                            raise RuntimeError("User aborted invalid cache deletion")
                        elif command == 'ignore':
                            self._log.warn("Ignoring invalid cache")
                            return
                        elif command == 'revalidate':
                            self._log.warn("Revalidating invalid cache")
                            files.clear()
                        else:
                            raise RuntimeError("command=%s" % repr(command))
                    elif self.auto_delete_cache is not True:
                        raise ValueError("MneExperiment.auto_delete_cache=%s"
                                         % repr(self.auto_delete_cache))

                    # delete invalid files
                    self._log.info("Deleting %i invalid cache files", len(files))
                    for path in files:
                        os.remove(path)
            else:
                self._log.info("Cache up to date.")
        elif os.path.exists(cache_dir):
            if self.auto_delete_cache is True:
                self._log.info("Deleting cache-dir without history")
                shutil.rmtree(cache_dir)
                os.mkdir(cache_dir)
            elif self.auto_delete_cache == 'disable':
                self._log.warn("Ignoring cache-dir without history")
                pass
            elif self.auto_delete_cache == 'debug':
                print("Cache directory without history (validate|abort).")
                while True:
                    command = raw_input(" > ")
                    if command == 'abort':
                        raise RuntimeError("User aborted")
                    elif command == 'validate':
                        self._log.warn("Validating cache-dir without history")
                        break
                    else:
                        print("invalid entry")
            else:
                raise IOError("Cache directory without history, but "
                              "auto_delete_cache is not True")
        else:
            os.mkdir(cache_dir)

        new_state = {'version': 2,
                     'groups': self._groups,
                     'epochs': self._epochs,
                     'tests': self._tests,
                     'parcs': self._parcs,
                     'events': events}
        save.pickle(new_state, cache_state_path)

    def __iter__(self):
        "Iterate state through subjects and yield each subject name."
        for subject in self.iter():
            yield subject

    def _annot_mtime(self, make_for=None):
        "Return max mtime of annot files or None if they do not exist."
        if make_for:
            with self._temporary_state:
                self.make_annot(mrisubject=make_for)
                return self._annot_mtime()

        mtime = 0
        for _ in self.iter('hemi'):
            fpath = self.get('annot-file')
            if os.path.exists(fpath):
                mtime = max(mtime, os.path.getmtime(fpath))
            else:
                return
        return mtime

    def _epochs_mtime(self):
        bads_path = self.get('bads-file')
        if os.path.exists(bads_path):
            raw_mtime = os.path.getmtime(self._get_raw_path(make=True))
            bads_mtime = os.path.getmtime(bads_path)
            epoch = self._epochs[self.get('epoch')]
            rej_mtime = self._rej_mtime(epoch)
            if rej_mtime:
                return max(raw_mtime, bads_mtime, rej_mtime)

    def _epochs_stc_mtime(self):
        "mtime if up-to-date, else None; does not check annot"
        mtime = self._epochs_mtime()
        if mtime:
            return max(mtime, self._inv_mtime())

    def _evoked_mtime(self):
        "Return mtime if the evoked file is up-to-date, None otherwise"
        path = self.get('evoked-file')
        if os.path.exists(path):
            evoked_mtime = os.path.getmtime(path)
            if evoked_mtime > self._epochs_mtime():
                return evoked_mtime

    def _evoked_stc_mtime(self):
        "mtime if up-to-date, else None; does not check annot"
        mtime = self._evoked_mtime()
        if mtime:
            return max(mtime, self._inv_mtime())

    def _inv_mtime(self):
        return max(os.path.getmtime(self._get_raw_path(make=True)),
                   os.path.getmtime(self.get('fwd-file', make=True)),
                   os.path.getmtime(self.get('cov-file', make=True)))

    def _rej_mtime(self, epoch):
        "rej-file mtime for secondary epoch definition"
        epochs = epoch.get('_rej_file_epochs', None) or (epoch['name'],)
        with self._temporary_state:
            paths = [self.get('rej-file', epoch=e) for e in epochs]
        if all(os.path.exists(path) for path in paths):
            return max(os.path.getmtime(path) for path in paths)

    def _result_mtime(self, dst, data, cached_test=False, single_subject=False):
        """For several results (reports and movies)

        Parameters
        ----------
        dst : str
            Filename.
        data : 'src' | 'sns'
            Data type.
        cached_test : bool
            Whether a corresponding test is being cached and needs to be
            checked (e.g. for reports that are based on a cached test).
        single_subject : bool
            Whether the corresponding test is performed for a single subject
            (as opposed to the current group).
        """
        if os.path.exists(dst):
            dst_mtime = os.path.getmtime(dst)
            if cached_test:
                return self._result_mtime(self.get('test-file'), data)
            else:
                if data == 'src':
                    if single_subject:
                        mrisubject = 'mrisubject'
                    else:
                        mrisubject = 'common_brain'

                    if self._annot_mtime(self.get(mrisubject)) > dst_mtime:
                        return
                    mtime_func = self._epochs_stc_mtime
                else:
                    mtime_func = self._epochs_mtime

                if single_subject:
                    mtime_iterator = (mtime_func(),)
                else:
                    mtime_iterator = (mtime_func() for _ in self)

                for mtime in mtime_iterator:
                    if mtime is None or mtime > dst_mtime:
                        return
                return dst_mtime

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

    def _add_epochs_stc(self, ds, ndvar, baseline, morph, mask):
        """
        Transform epochs contained in ds into source space (adds a list of mne
        SourceEstimates to ds)

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
        mask : bool
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
            baseline = self._epochs[self.get('epoch')]['baseline']

        epochs = ds['epochs']
        inv = self.load_inv(epochs)
        stc = apply_inverse_epochs(epochs, inv, **self._params['apply_inv_kw'])

        if ndvar:
            self.make_annot()
            subject = self.get('mrisubject')
            src = self.get('src')
            mri_sdir = self.get('mri-sdir')
            parc = self.get('parc') or None
            src = load.fiff.stc_ndvar(stc, subject, src, mri_sdir,
                                      self._params['apply_inv_kw']['method'],
                                      self._params['make_inv_kw'].get('fixed', False),
                                      parc=parc)

            if 'pick' in self._params['post_inv_kw']:
                if self._params['post_inv_kw']['pick'] == '+':
                    np.clip(src.x, 0, np.inf, src.x)
                else:
                    np.clip(src.x, -np.inf, 0, src.x)

            if 'smooth' in self._params['post_inv_kw']:
                smooth = self._params['post_inv_kw']['smooth']
                src = src.smooth('source', smooth)

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
            elif self._params['post_inv_kw']:
                raise NotImplementedError("MNE SourceEstimate post-processing")
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
        mask : bool
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False).

        Notes
        -----
        Assumes that all Evoked of the same subject share the same inverse
        operator.
        """
        if not any((ind_stc, ind_ndvar, morph_stc, morph_ndvar)):
            err = ("Nothing to load, set at least one of (ind_stc, ind_ndvar, "
                   "morph_stc, morph_ndvar) to True")
            raise ValueError(err)
        elif self._params['post_inv_kw']:
            raise NotImplementedError("MNE SourceEstimate post-processing")

        if isinstance(baseline, str):
            raise NotImplementedError("Baseline form different epoch")
        elif baseline is True:
            baseline = self._epochs[self.get('epoch')]['baseline']

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
        parc = self.get('parc') or None
        mri_sdir = self.get('mri-sdir')
        # for name, key in izip(do, keys):
        if ind_stc:
            ds['stc'] = stcs
        if ind_ndvar:
            subject = from_subjects[meg_subjects[0]]
            ds['src'] = load.fiff.stc_ndvar(stcs, subject, src, mri_sdir,
                                            self._params['apply_inv_kw']['method'],
                                            self._params['make_inv_kw'].get('fixed', False),
                                            parc=parc)
            if mask:
                _mask_ndvar(ds, 'src')
        if morph_stc or morph_ndvar:
            if morph_stc:
                ds['stcm'] = mstcs
            if morph_ndvar:
                ds['srcm'] = load.fiff.stc_ndvar(mstcs, common_brain, src, mri_sdir,
                                                 self._params['apply_inv_kw']['method'],
                                                 self._params['make_inv_kw'].get('fixed', False),
                                                 parc=parc)
                if mask:
                    _mask_ndvar(ds, 'srcm')

        if not keep_evoked:
            del ds['evoked']

    @staticmethod
    def _add_vars(ds, vardef):
        """Add vars to the dataset

        Parameters
        ----------
        ds : Dataset
            Event dataset.
        vardef : dict
            Variable definition.
        """
        new = {}
        for name, definition in vardef.iteritems():
            if isinstance(definition, str):
                new[name] = ds.eval(definition)
            else:
                source, codes = definition
                new[name] = asfactor(source, ds=ds).as_var(codes, 0, name)
        ds.update(new, True)

    def _backup(self, dst_root):
        """Backup all essential files to ``dst_root``.

        Parameters
        ----------
        dst_root : str
            Directory to use as root for the backup.

        Notes
        -----
        For repeated backups ``dst_root`` can be the same. If a file has been
        previously backed up, it is only copied if the local copy has been
        modified more recently than the previous backup. If the backup has been
        modified more recently than the local copy, a warning is displayed.

        Currently, the following files are included in the backup::

         * Calmed raw file (raw='clm')
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
                        paths.extend(os.path.join(root_, fn) for fn in filenames)
            else:
                paths = self.glob(temp, **state)

            # convert to (src, dst) pairs
            for src in paths:
                if not src.startswith(root):
                    raise ValueError("Can only backup files in root directory")
                tail = src[root_len:]
                dst = os.path.join(dst_root, tail)
                if os.path.exists(dst):
                    src_m = os.path.getmtime(src)
                    dst_m = os.path.getmtime(dst)
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
            self._log.info("All files backed up.")
            return

        self._log.info("Backing up %i files ..." % len(pairs))
        # create directories
        for dirname in dirs:
            dirpath = os.path.join(dst_root, dirname)
            if not os.path.exists(dirpath):
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

    def _get_raw_path(self, make=False):
        if self.get('modality') == 'meeg':
            raise RuntimeError("No raw files for combined MEG & EEG")
        elif self._raw[self.get('raw')] is None:
            path = self.get('raw-file')
            if os.path.isfile(path):
                return path
            else:
                raise IOError("Raw file for %s is missing at %s"
                              % (self.get('subject'), path))
        else:
            return self.get('cached-raw-file', make=make)

    def iter(self, fields='subject', exclude=True, values={}, group=None, *args,
             **kwargs):
        """
        Cycle the experiment's state through all values on the given fields

        Parameters
        ----------
        fields : list | str
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
        mail : bool | str
            Send an email when iteration is finished. Can be True or an email
            address. If True, the notification is sent to :attr:`.owner`.
        prog : bool | str
            Show a progress dialog; str for dialog title.
        *others* :
            Fields with constant values throughout the iteration.
        """
        if group is not None:
            kwargs['group'] = group
        return FileTree.iter(self, fields, exclude, values, *args, **kwargs)

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
                self.restore_state(discard_tip=False)
                self.set(**{field: value})
                yield value

    def label_events(self, ds):
        """
        Adds T (time) and SOA (stimulus onset asynchrony) to the Dataset.

        Parameters
        ----------
        ds : Dataset
            A Dataset containing events (as returned by
            :func:`load.fiff.events`).

        Notes
        -----
        Subclass this method to specify events.
        """
        ds['T'] = ds['i_start'] / ds.info['sfreq']
        ds['SOA'] = Var(np.ediff1d(ds['T'].x, 0))

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
        for name, subjects in self.groups.iteritems():
            ds[name] = Var(subject.isin(subjects))

    def load_annot(self, **state):
        """Load a parcellation (from an annot file)

        Returns
        -------
        labels : list of Label
            Labels in the parcellation (output of
            :func:`mne.read_labels_from_annot`).
        """
        self.make_annot(**state)
        return mne.read_labels_from_annot(self.get('mrisubject'),
                                          self.get('parc'), 'both',
                                          subjects_dir=self.get('mri-sdir'))

    def load_bad_channels(self, **kwargs):
        """Load bad channels

        Returns
        -------
        bad_chs : list of str
            Bad chnnels.
        """
        path = self.get('bads-file', **kwargs)
        if os.path.exists(path):
            with open(path) as fid:
                names = [l for l in fid.read().splitlines() if l]
            return names
        else:
            print("No bad channel definition for: %s/%s, creating empty file" %
                  (self.get('subject'), self.get('experiment')))
            self.make_bad_channels(())
            return []

    def load_cov(self, **kwargs):
        """Load the covariance matrix

        Parameters
        ----------
        others :
            State parameters.
        """
        return mne.read_cov(self.get('cov-file', make=True, **kwargs))

    def load_edf(self, **kwargs):
        """Load the edf file ("edf-file" template)"""
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

    def _data_arg_for_modality(self, subject, modality, eog=False):
        "data argument for FIFF-to-NDVar conversion"
        if modality == 'meeg':
            raise NotImplementedError("NDVar for sensor space data combining "
                                      "EEG and MEG data")
        elif modality == '':
            subject_prefix = self._subject_re.match(subject).group(1)
            meg_system = self._meg_systems[subject_prefix]
            sysname = 'KIT-157' if meg_system == 'KIT-NY' else 'KIT-208'
            return 'mag', sysname
        elif modality == 'eeg':
            if eog:
                return 'eeg&eog', None
            else:
                return 'eeg', None
        else:
            raise ValueError("modality=%r" % modality)

    def load_epochs(self, subject=None, baseline=False, ndvar=True,
                    add_bads=True, reject=True, add_proj=True, cat=None,
                    decim=None, pad=0, data_raw=False, vardef=None,
                    eog=False, trigger_shift=True, **kwargs):
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
        add_proj : bool
            Add the projections to the Raw object.
        cat : sequence of cell-names
            Only load data for these cells (cells of model).
        decim : None | int
            Set to an int in order to override the epoch decim factor.
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
        """
        modality = self.get('modality')
        if ndvar:
            if isinstance(ndvar, basestring):
                if ndvar != 'both':
                    raise ValueError("ndvar=%s" % repr(ndvar))
        subject, group = self._process_subject_arg(subject, kwargs)

        if group is not None:
            dss = []
            for _ in self.iter(group=group):
                ds = self.load_epochs(None, baseline, ndvar, add_bads, reject,
                                      add_proj, cat, decim, pad, data_raw,
                                      vardef)
                dss.append(ds)

            ds = combine(dss)
        elif modality == 'meeg':  # single subject, combine MEG and EEG
            with self._temporary_state:
                ds_meg = self.load_epochs(subject, baseline, ndvar, add_bads,
                                          reject, add_proj, cat, decim, pad,
                                          data_raw, vardef, modality='')
                ds_eeg = self.load_epochs(subject, baseline, ndvar, add_bads,
                                          reject, add_proj, cat, decim, pad,
                                          data_raw, vardef, modality='eeg')
            ds, eeg_epochs = align(ds_meg, ds_eeg['epochs'], 'index',
                                   ds_eeg['index'])
            ds['epochs'] = mne.epochs.add_channels_epochs((ds['epochs'], eeg_epochs))
        else:  # single subject, single modality
            epoch = self._epochs[self.get('epoch')]
            if baseline is True:
                baseline = epoch['baseline']

            ds = self.load_selected_events(add_bads=add_bads, reject=reject,
                                           add_proj=add_proj,
                                           data_raw=data_raw or True,
                                           vardef=vardef)
            if ds.n_cases == 0:
                raise RuntimeError("No events left for epoch=%s, subject=%s"
                                   % (repr(self.get('epoch')), repr(subject)))

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)
                if ds.n_cases == 0:
                    raise RuntimeError("Selection with cat=%s resulted in "
                                       "empty Dataset" % repr(cat))

            # load sensor space data
            tmin = epoch['tmin']
            tmax = epoch['tmax']
            if pad:
                tmin -= pad
                tmax += pad
            if decim is None:
                decim = epoch['decim']
            ds = load.fiff.add_mne_epochs(ds, tmin, tmax, baseline, decim=decim)

            # post baseline-correction trigger shift
            if trigger_shift and 'post_baseline_trigger_shift' in epoch:
                ds['epochs'] = shift_mne_epoch_trigger(ds['epochs'],
                                                       ds[epoch['post_baseline_trigger_shift']],
                                                       epoch['post_baseline_trigger_shift_min'],
                                                       epoch['post_baseline_trigger_shift_max'])

            # interpolate channels
            if reject and ds.info[INTERPOLATE_CHANNELS]:
                if modality == '':
                    _interpolate_bads_meg(ds['epochs'], ds[INTERPOLATE_CHANNELS])
                else:
                    _interpolate_bads_eeg(ds['epochs'], ds[INTERPOLATE_CHANNELS])

            if not data_raw:
                del ds.info['raw']

            if ndvar:
                name = self._ndvar_name_for_modality(modality)
                data, sysname = self._data_arg_for_modality(subject, modality, eog)
                ds[name] = load.fiff.epochs_ndvar(ds['epochs'], data=data, sysname=sysname)
                if ndvar != 'both':
                    del ds['epochs']
                if modality == 'eeg':
                    self._fix_eeg_ndvar(ds[ndvar], group)

        return ds

    def load_epochs_stc(self, subject=None, sns_baseline=True,
                        src_baseline=False, ndvar=True, cat=None,
                        keep_epochs=False, morph=False, mask=False,
                        data_raw=False, **kwargs):
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
        mask : bool
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        """
        if not sns_baseline and src_baseline and \
                self._epochs[self.get('epoch')].get('post_baseline_trigger_shift', None):
            raise NotImplementedError("post_baseline_trigger_shift is not "
                                      "implemented for baseline correction in "
                                      "source space")
        ds = self.load_epochs(subject, sns_baseline, False, cat=cat,
                              data_raw=data_raw, **kwargs)
        self._add_epochs_stc(ds, ndvar, src_baseline, morph, mask)
        if not keep_epochs:
            del ds['epochs']
        return ds

    def load_events(self, subject=None, add_proj=True, add_bads=True,
                    data_raw=True, edf=True, **kwargs):
        """
        Load events from a raw file.

        Loads events from the corresponding raw file, adds the raw to the info
        dict.

        Parameters
        ----------
        subject : str (state)
            Subject for which to load events.
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default True).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        edf : bool
            Load the EDF file (if available) and add it as ``ds.info['edf']``.
        others :
            Update state.
        """
        evt_file = self.get('event-file', mkdir=True, subject=subject,
                            **kwargs)
        raw_file = self._get_raw_path(True)

        # search for and check cached version
        ds = None
        if os.path.exists(evt_file):
            event_mtime = os.path.getmtime(evt_file)
            raw_mtime = os.path.getmtime(raw_file)
            if event_mtime > raw_mtime:
                ds = load.unpickle(evt_file)
                #  <0.19 cache
                if 'sfreq' not in ds.info:
                    ds = None

        # refresh cache
        subject = self.get('subject')
        if ds is None:
            if self.get('modality') == '':
                merge = -1
            else:
                merge = 0
            raw = self.load_raw(add_proj, add_bads, subject=subject)
            ds = load.fiff.events(raw, merge)
            del ds.info['raw']
            ds.info['sfreq'] = raw.info['sfreq']

            if edf and self.has_edf[subject]:  # add edf
                edf = self.load_edf()
                edf.add_t_to(ds)
                ds.info['edf'] = edf

            if edf or not self.has_edf[subject]:
                save.pickle(ds, evt_file)
        elif data_raw is True:
            raw = self.load_raw(add_proj, add_bads, subject=subject)

        # if data should come from different raw settings than events
        if isinstance(data_raw, str):
            with self._temporary_state:
                raw = self.load_raw(add_proj, add_bads, subject=subject,
                                    raw=data_raw)
        elif not isinstance(data_raw, bool):
            raise TypeError("data_raw=%s; needs to be str or bool"
                            % repr(data_raw))

        ds.info['subject'] = subject
        ds.info['experiment'] = self.get('experiment')
        if data_raw:
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
                    decim=None, data_raw=False, **kwargs):
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
        decim : None | int
            Set to an int in order to override the epoch decim factor.
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        model : str (state)
            Model according to which epochs are grouped into evoked responses.
        *others* : str
            State parameters.
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        if baseline is True:
            baseline = self._epochs[self.get('epoch')]['baseline']

        if group is not None:
            dss = [self.load_evoked(None, baseline, False, cat, decim, data_raw)
                   for _ in self.iter(group=group)]
            ds = combine(dss, incomplete='drop')

            # check consistency in MNE objects' number of time points
            lens = [len(e.times) for e in ds['evoked']]
            ulens = set(lens)
            if len(ulens) > 1:
                err = ["Unequal time axis sampling (len):"]
                alens = np.array(lens)
                for l in ulens:
                    err.append('%i: %r' % (l, ds['subject', alens == l].cells))
                raise DimensionMismatchError(os.linesep.join(err))

        else:  # single subject
            ds = self._make_evoked(decim, data_raw)

            if cat:
                model = ds.eval(self.get('model'))
                idx = model.isin(cat)
                ds = ds.sub(idx)
                if ds.n_cases == 0:
                    raise RuntimeError("Selection with cat=%s resulted in "
                                       "empty Dataset" % repr(cat))

            # baseline correction
            if isinstance(baseline, str):
                raise NotImplementedError
            elif baseline and not self._epochs[self.get('epoch')]\
                    .get('post_baseline_trigger_shift', None):
                for e in ds['evoked']:
                    rescale(e.data, e.times, baseline, 'mean', copy=False)

        # convert to NDVar
        if ndvar:
            modality = self.get('modality')
            name = self._ndvar_name_for_modality(modality)
            data, sysname = self._data_arg_for_modality(subject, modality)
            ds[name] = load.fiff.evoked_ndvar(ds['evoked'], data=data,
                                                  sysname=sysname)
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
        mask : bool
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default True).
        morph : bool
            Morph the source estimates to the common_brain (default False).
        keep_stc : bool
            Keep the source timecourse data in the Dataset that is returned
            (default False).
        """
        ds = self.load_epochs_stc(subject, sns_baseline, ndvar=True,
                                  morph=morph, mask=mask, data_raw='clm',
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
        mask : bool
            Whether to just load the sources from the parcellation that are not
            defined as "unknown". Default is True.
        morph : bool
            Morph the source estimates to the common_brain (default False).
        keep_stc : bool
            Keep the source timecourse data in the Dataset that is returned
            (default False).
        """
        if self._params['post_inv_kw']:
            raise NotImplementedError("MNE SourceEstimate post-processing")

        ds = self.load_evoked_stc(subject, sns_baseline, morph_ndvar=morph,
                                  ind_ndvar=not morph, mask=mask,
                                  data_raw='clm', **kwargs)
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
                        **kwargs):
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
        mask : bool
            Discard data that is labelled 'unknown' by the parcellation (only
            applies to NDVars, default False).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        *others* : str
            State parameters.
        """
        if not any((ind_stc, ind_ndvar, morph_stc, morph_ndvar)):
            err = ("Nothing to load, set at least one of (ind_stc, ind_ndvar, "
                   "morph_stc, morph_ndvar) to True")
            raise ValueError(err)
        elif not sns_baseline and src_baseline and \
                self._epochs[self.get('epoch')].get('post_baseline_trigger_shift', None):
            raise NotImplementedError("post_baseline_trigger_shift is not "
                                      "implemented for baseline correction in "
                                      "source space")

        ds = self.load_evoked(subject, sns_baseline, sns_ndvar, cat, None,
                              data_raw, **kwargs)
        self._add_evoked_stc(ds, ind_stc, ind_ndvar, morph_stc, morph_ndvar,
                             src_baseline, keep_evoked, mask)

        return ds

    def load_inv(self, fiff=None, **kwargs):
        """Load the inverse operator

        Parameters
        ----------
        fiff : Raw | Epochs | Evoked | ...
            Object which provides the mne info dictionary (default: load the
            raw file).
        others :
            State parameters.
        """
        if self.get('modality', **kwargs) != '':
            raise NotImplementedError("Source reconstruction for EEG data")

        if fiff is None:
            fiff = self.load_raw()

        fwd_file = self.get('fwd-file', make=True)
        fwd = mne.read_forward_solution(fwd_file, surf_ori=True)
        cov = self.load_cov()
        inv = make_inverse_operator(fiff.info, fwd, cov,
                                    **self._params['make_inv_kw'])
        return inv

    def load_label(self, label, **kwargs):
        """Retrieve a label as mne Label object

        Parameters
        ----------
        label : str
            Name of the label. If the label name does not end in '-bh' or '-rh'
            the combination of the labels ``label + '-lh'`` and
            ``label + '-rh'`` is returned.
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

    def load_raw(self, add_proj=True, add_bads=True, preload=False, **kwargs):
        """
        Load a raw file as mne Raw object.

        Parameters
        ----------
        add_proj : bool
            Add the projections to the Raw object. This does *not* set the
            proj variable.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        preload : bool
            Mne Raw parameter.
        """
        if kwargs:
            self.set(**kwargs)

        if add_proj:
            proj = self.get('proj')
            if proj:
                proj = self.get('proj-file')
            else:
                proj = None
        else:
            proj = None

        raw = load.fiff.mne_raw(self._get_raw_path(True), proj, preload=preload)
        if add_bads:
            if add_bads is True:
                bad_chs = self.load_bad_channels()
            else:
                bad_chs = add_bads

            raw.info['bads'] = bad_chs

        return raw

    def load_selected_events(self, subject=None, reject=True, add_proj=True,
                             add_bads=True, index=True, data_raw=False,
                             vardef=None, **kwargs):
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
        add_proj : bool
            Add the projections to the Raw object.
        add_bads : False | True | list
            Add bad channel information to the Raw. If True, bad channel
            information is retrieved from the 'bads-file'. Alternatively,
            a list of bad channels can be sumbitted.
        index : bool | str
            Index the Dataset before rejection (provide index name as str).
        data_raw : bool | str
            Keep the mne.io.Raw instance in ds.info['raw'] (default False).
            Can be specified as raw name (str) to include a different raw object
            than the one from which events are loaded (used for frequency
            analysis).
        vardef : str
            Name of a 2-stage test defining additional variables.
        others :
            Update the experiment state.

        Notes
        -----
        When trial rejection is set to automatic, not rejection is performed
        because no epochs are loaded.
        """
        # process arguments
        if reject not in (True, False, 'keep'):
            raise ValueError("Invalid reject value: %r" % reject)

        if index is True:
            index = 'index'
        elif index and not isinstance(index, str):
            raise TypeError("index=%s" % repr(index))

        # case of loading events for a group
        subject, group = self._process_subject_arg(subject, kwargs)
        if group is not None:
            if data_raw:
                raise ValueError("data_var=%s: can't load data raw when "
                                 "combining different subjects" % repr(data_raw))
            dss = [self.load_selected_events(reject=reject, add_proj=add_proj,
                                             add_bads=add_bads, index=index,
                                             vardef=vardef)
                   for _ in self.iter(group=group)]
            ds = combine(dss)
            return ds

        # retrieve & check epoch parameters
        epoch = self._epochs[self.get('epoch')]
        sel = epoch.get('sel', None)
        sel_epoch = epoch.get('sel_epoch', None)
        sub_epochs = epoch.get('sub_epochs', None)

        # rejection comes from somewhere else
        if sub_epochs is not None:
            with self._temporary_state:
                dss = [self.load_selected_events(subject, reject, add_proj,
                                                 add_bads, index, data_raw,
                                                 epoch=sub_epoch)
                       for sub_epoch in sub_epochs]

                # combine bad channels
                bad_channels = set()
                for ds in dss:
                    if BAD_CHANNELS in ds.info:
                        bad_channels.update(ds.info[BAD_CHANNELS])

                ds = combine(dss)
                if data_raw:
                    ds.info['raw'] = dss[0].info['raw']
                if bad_channels:
                    ds.info[BAD_CHANNELS] = sorted(bad_channels)
        elif sel_epoch is not None:
            with self._temporary_state:
                ds = self.load_selected_events(None, 'keep', add_proj, add_bads,
                                               index, data_raw, epoch=sel_epoch)

            if sel is not None:
                ds = ds.sub(sel)
            if index:
                ds.index(index)

            if reject is True:
                ds = ds.sub('accept')
        else:
            ds = self.load_events(add_proj=add_proj, add_bads=add_bads,
                                  data_raw=data_raw)
            if sel is not None:
                ds = ds.sub(sel)
            if index:
                ds.index(index)

            n_cases = epoch.get('n_cases', None)
            if n_cases is not None and ds.n_cases != n_cases:
                err = "Number of epochs %i, expected %i" % (ds.n_cases, n_cases)
                raise RuntimeError(err)

            # rejection
            rej_params = self._epoch_rejection[self.get('rej')]
            if reject and rej_params['kind']:
                path = self.get('rej-file')
                if not os.path.exists(path):
                    if rej_params['kind'] == 'manual':
                        raise RuntimeError("The rejection file at %r does not "
                                           "exist. Run .make_rej() first." % path)
                    else:
                        raise RuntimeError("The rejection file at %r does not "
                                           "exist and has to be user-generated."
                                           % path)

                # load and check file
                ds_sel = load.unpickle(path)
                if not np.all(ds['trigger'] == ds_sel['trigger']):
                    if np.all(ds[:-1, 'trigger'] == ds_sel['trigger']):
                        ds = ds[:-1]
                        msg = self.format("Last epoch for {subject} is missing")
                        self._log.warn(msg)
                    else:
                        raise RuntimeError("The epoch selection file contains different "
                                           "events than the data. Something went wrong...")

                if rej_params.get('interpolation', True) and INTERPOLATE_CHANNELS in ds_sel:
                    ds[INTERPOLATE_CHANNELS] = ds_sel[INTERPOLATE_CHANNELS]
                    ds.info[INTERPOLATE_CHANNELS] = True
                else:
                    ds.info[INTERPOLATE_CHANNELS] = False

                # subset events
                if reject == 'keep':
                    ds['accept'] = ds_sel['accept']
                elif reject is True:
                    ds = ds.sub(ds_sel['accept'])
                else:
                    err = ("reject parameter must be bool or 'keep', not "
                           "%r" % reject)
                    raise ValueError(err)

                # bad channels
                if add_bads and BAD_CHANNELS in ds_sel.info:
                    ds.info[BAD_CHANNELS] = ds_sel.info[BAD_CHANNELS]

        # apply trigger-shift
        if 'trigger_shift' in epoch:
            shift = epoch['trigger_shift']
            if isinstance(shift, basestring):
                shift = ds.eval(shift)
            if isvar(shift):
                shift = shift.x

            if np.isscalar(shift):
                ds['i_start'] += int(round(shift * ds.info['sfreq']))
            else:
                ds['i_start'] += np.round(shift * ds.info['sfreq']).astype(int)

        # Additional variables
        if 'vars' in epoch:
            self._add_vars(ds, epoch['vars'])
        if isinstance(vardef, str):
            vardef = self._tests[vardef].get('vars', None)
        if vardef:
            self._add_vars(ds, vardef)
        return ds

    def load_src(self, add_geom=False, **state):
        "Load the current source space"
        fpath = self.get('src-file', make=True, **state)
        src = mne.read_source_spaces(fpath, add_geom)
        return src

    def load_test(self, test, tstart, tstop, pmin, parc=None, mask=None,
                  samples=10000, data='src', sns_baseline=True,
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
        data : 'sns' | 'src'
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
        group : str
            Group for which to perform the test.

        Returns
        -------
        ds : Dataset (if return_data==True)
            Data that forms the basis of the test.
        res : TestResult
            Test result for the specified test.
        """
        self.set(test=test, **kwargs)
        self._set_analysis_options(data, sns_baseline, src_baseline, pmin,
                                   tstart, tstop, parc, mask)
        return self._load_test(test, tstart, tstop, pmin, parc, mask, samples,
                               data, sns_baseline, src_baseline, return_data,
                               make)

    def _load_test(self, test, tstart, tstop, pmin, parc, mask, samples, data,
                   sns_baseline, src_baseline, return_data, make):
        "Requires that _set_analysis_options() has been called"
        test_params = self._tests[test]

        if test_params['kind'] == 'custom':
            raise RuntimeError("Don't know how to perform 'custom' test")

        # find data to use
        modality = self.get('modality')
        if data == 'sns':
            dims = ('time', 'sensor')
            if modality == '':
                y_name = 'meg'
            elif modality == 'eeg':
                y_name = 'eeg'
            else:
                raise ValueError("data=%r, modality=%r" % (data, modality))
        elif data == 'src':
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

        # two-stage tests (not cached)
        if test_params['kind'] == 'two-stage':
            if data != 'src':
                raise NotImplementedError("Two-stage test with data != 'src'")
            elif return_data:
                raise ValueError("return_data argument is invalid for two-"
                                 "stage tests")

            # find params
            model = test_params['stage 1']
            vardef = test_params['vars'] if 'vars' in test_params else None

            # stage 1
            n = len(self.get_field_values('subject'))
            prog_str = "\x1b[2KLoading 2-stage model [%%-%ss] %%s" % n
            lms = []
            for i, subject in enumerate(self):
                print(prog_str % ('#' * i, subject), end='\r')
                ds = self.load_epochs_stc(subject, sns_baseline, src_baseline,
                                          morph=True, mask=mask, vardef=vardef)
                lms.append(spm.LM(y_name, model, ds, subject=subject))
            print(prog_str % ('#' * n, 'done'))
            return spm.RandomLM(lms)

        # try to load cached test
        res = None
        load_data = True
        desc = self._get_rel('test-file', 'test-dir')
        if self._result_mtime(dst, data):
            try:
                res = load.unpickle(dst)
            except OldVersionError:
                res = None
            else:
                if res.samples >= samples or res.samples == -1:
                    self._log.info("Load cached test: %s", desc)
                    load_data = return_data
                else:
                    res = None

        if res is None and not make:
            raise IOError("The requested test is not cached. Set make=True to "
                          "perform the test.")

        # load data
        if load_data:
            # determine categories to load
            if test_params['kind'] == 'ttest_rel':
                cat = (test_params['c1'], test_params['c0'])
            else:
                cat = None

            # load data
            if data == 'sns':
                ds = self.load_evoked(True, sns_baseline, True, cat)
            elif data == 'src':
                ds = self.load_evoked_stc(True, sns_baseline, src_baseline,
                                          morph_ndvar=True, cat=cat, mask=mask)

        # perform the test if it was not cached
        if res is None:
            self._log.info("Make test: %s", desc)
            test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop, dims,
                                            parc_dim)
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

        Returns
        -------
        mtime : float | None
            Modification time of the existing files, or None if they were newly
            created.
        """
        self.set(**state)

        # variables
        parc = self.get('parc')
        if parc == '':
            return
        elif parc in self._parcs:
            p = self._parcs[parc]
        else:
            p = self._parcs[SEEDED_PARC_RE.match(parc).group(1)]

        mrisubject = self.get('mrisubject')
        common_brain = self.get('common_brain')
        mtime = self._annot_mtime()
        if mrisubject != common_brain:
            is_fake = is_fake_mri(self.get('mri-dir'))
            if p['morph_from_fsaverage'] or is_fake:
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
        elif not p['make']:
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
        """Returns labels

        Notes
        -----
        Only called to make custom annotation files for the common_brain
        """
        subjects_dir = self.get('mri-sdir')
        if p['kind'] == 'combination':
            with self._temporary_state:
                base = {l.name: l for l in self.load_annot(parc=p['base'])}
            labels = []
            for name, exp in p['labels'].iteritems():
                labels += combination_label(name, exp, base, subjects_dir)
        elif p['kind'] == 'seeded':
            mask = p.get('mask', None)
            if mask:
                with self._temporary_state:
                    self.make_annot(parc=mask)
            name, extent = SEEDED_PARC_RE.match(parc).groups()
            labels = labels_from_mni_coords(p['seeds'], float(extent), subject,
                                            p.get('surface', 'white'), mask,
                                            subjects_dir, parc)
        elif parc == 'lobes':
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
            msg = ("At least one of the annot files for the custom parcellation "
                   "%r is missing for %r, and a make function is not "
                   "implemented." % (parc, subject))
            raise NotImplementedError(msg)
        return labels

    def make_bad_channels(self, bad_chs, redo=False, **kwargs):
        """Write the bad channel definition file for a raw file

        Parameters
        ----------
        bad_chs : iterator of str
            Names of the channels to set as bad. Numerical entries are
            interpreted as "MEG XXX". If bad_chs contains entries not present
            in the raw data, a ValueError is raised.
        redo : bool
            If the file already exists, replace it.

        See Also
        --------
        load_bad_channels : load the current bad_channels file
        """
        dst = self.get('bads-file', **kwargs)
        if os.path.exists(dst):
            old_bads = self.load_bad_channels()
            if not redo:
                msg = ("Bads file already exists with %s. In order to replace "
                       "it, use `redo=True`." % old_bads)
                raise IOError(msg)
        else:
            old_bads = None

        raw = self.load_raw(add_bads=False)
        sensor = load.fiff.sensor_dim(raw)
        chs = sensor._normalize_sensor_names(bad_chs)
        if old_bads is None:
            print("-> %s" % chs)
        else:
            print("%s -> %s" % (old_bads, chs))
        text = os.linesep.join(chs)
        with open(dst, 'w') as fid:
            fid.write(text)

    def make_bem_sol(self):
        self._log.info(self.format("Creating bem-sol file for {mrisubject}"))
        bin_path = subp.get_bin('mne', 'mne_prepare_bem_model')
        bem_path = self.get('bem-file', fmatch=True)
        mne.utils.run_subprocess([bin_path, '--bem', bem_path])

    def make_besa_evt(self, redo=False, **state):
        """Make the trigger and event files needed for besa

        Parameters
        ----------
        redo : bool
            If besa files already exist, overwrite them.

        Notes
        -----
        Ignores the *decim* epoch parameter.

        Target files are saved relative to the *besa-root* location.
        """
        self.set(**state)
        rej = self.get('rej')
        trig_dest = self.get('besa-trig', rej='', mkdir=True)
        evt_dest = self.get('besa-evt', rej=rej, mkdir=True)
        if not redo and os.path.exists(evt_dest) and os.path.exists(trig_dest):
            return

        # load events
        ds = self.load_selected_events(reject='keep')

        # save triggers
        if redo or not os.path.exists(trig_dest):
            save.meg160_triggers(ds, trig_dest, pad=1)
            if not redo and os.path.exists(evt_dest):
                return
        else:
            ds.index('besa_index', 1)

        # reject bad trials
        ds = ds.sub('accept')

        # save evt
        epoch = self._epochs[self.get('epoch')]
        save.besa_evt(ds, tstart=epoch['tmin'], tstop=epoch['tmax'], dest=evt_dest)

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
        if not redo and os.path.exists(dst_path):
            return

        src_path = self.get(temp, **{field: src})
        if os.path.isdir(src_path):
            raise ValueError("Can only copy files, not directories.")
        shutil.copyfile(src_path, dst_path)

    def make_cov(self, redo=False):
        """Make a noise covariance (cov) file

        Parameters
        ----------
        redo : bool
            If the cov file already exists, overwrite it.
        """
        dest = self.get('cov-file', mkdir=True)
        params = self._covs[self.get('cov')]
        epoch = params['epoch']
        rej = self.get('cov-rej')
        if (not redo) and os.path.exists(dest):
            cov_mtime = os.path.getmtime(dest)
            raw_mtime = os.path.getmtime(self._get_raw_path())
            bads_mtime = os.path.getmtime(self.get('bads-file'))
            with self._temporary_state:
                self.set(rej=rej)
                rej_mtime = self._rej_mtime(self._epochs[epoch])
            if cov_mtime > max(raw_mtime, bads_mtime, rej_mtime):
                return

        method = params.get('method', 'empirical')
        keep_sample_mean = params.get('keep_sample_mean', True)
        reg = params.get('reg', None)

        with self._temporary_state:
            ds = self.load_epochs(None, True, False, decim=1, epoch=epoch, rej=rej)
        epochs = ds['epochs']
        cov = mne.compute_covariance(epochs, keep_sample_mean, method=method)

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

    def _make_evoked(self, decim, data_raw, **kwargs):
        """
        Creates datasets with evoked sensor data.

        Parameters
        ----------
        decim : None | int
            Set to an int in order to override the epoch decim factor.
        """
        dst = self.get('evoked-file', mkdir=True, **kwargs)
        fiff_dst = self.get('evoked-fiff-file')
        epoch = self._epochs[self.get('epoch')]
        use_cache = ((not decim or decim == epoch['decim']) and
                     (isinstance(data_raw, bool) or data_raw == self.get('raw')))
        if use_cache and self._evoked_mtime():
            ds = load.unpickle(dst)
            if 'evoked' in ds:
                if ds.info.get('mne_version', None) == mne.__version__:
                    return ds
            else:
                ds['evoked'] = mne.read_evokeds(fiff_dst, proj=False)
                return ds

        # load the epochs (post baseline-correction trigger shift requires
        # baseline corrected evoked
        post_baseline_trigger_shift = epoch.get('post_baseline_trigger_shift', None)
        if post_baseline_trigger_shift:
            ds = self.load_epochs(ndvar=False, baseline=True, decim=decim,
                                  data_raw=data_raw)
        else:
            ds = self.load_epochs(ndvar=False, decim=decim, data_raw=data_raw)

        # aggregate
        equal_count = self.get('equalize_evoked_count') == 'eq'
        ds_agg = ds.aggregate(self.get('model'), drop_bad=True,
                              drop=('i_start', 't_edf', 'T', 'index'),
                              equal_count=equal_count, never_drop=('epochs',))

        # save
        ds_agg.rename('epochs', 'evoked')
        if 'raw' in ds_agg.info:
            del ds_agg.info['raw']

        if use_cache:
            evoked = ds_agg.pop('evoked')
            save.pickle(ds_agg, dst)
            mne.write_evokeds(fiff_dst, evoked)
            ds_agg['evoked'] = evoked

        return ds_agg

    def make_fwd(self, redo=False):
        """Make the forward model"""
        dst = self.get('fwd-file')
        raw = self._get_raw_path(make=True)
        trans = self.get('trans-file')
        src = self.get('src-file', make=True)
        bem = self.get('bem-sol-file', make=True, fmatch=True)

        if not redo and os.path.exists(dst):
            fwd_mtime = os.path.getmtime(dst)
            raw_mtime = os.path.getmtime(raw)
            trans_mtime = os.path.getmtime(trans)
            src_mtime = os.path.getmtime(src)
            bem_mtime = os.path.getmtime(bem)
            if fwd_mtime > max(raw_mtime, trans_mtime, src_mtime, bem_mtime):
                return

        if self.get('modality') != '':
            raise NotImplementedError("Source reconstruction with EEG")
        src = mne.read_source_spaces(src)
        fwd = mne.make_forward_solution(raw, trans, src, bem, ignore_ref=True)
        for s, s0 in izip(fwd['src'], src):
            if s['nuse'] != s0['nuse']:
                msg = ("The forward solution contains fewer sources than the "
                       "source space. This could be due to a corrupted bem "
                       "file with source outside the inner skull surface.")
                raise RuntimeError(msg)
        mne.write_forward_solution(dst, fwd, True)

    def make_link(self, temp, field, src, dst, redo=False):
        """Make a hard link at the file with the dst value on field, linking to
        the file with the src value of field

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
        if not redo and os.path.exists(dst_path):
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
        *others* :
            Experiment state parameters.
        """
        kwargs['model'] = ''
        subject, group = self._process_subject_arg(subject, kwargs)
        brain_kwargs = self._surfer_plot_kwargs(surf, views, foreground, background,
                                                smoothing_steps, hemi)
        self._set_analysis_options('src', sns_baseline, src_baseline, None,
                                   None, None, None, None)
        self.set(equalize_evoked_count='',
                 resname="GA dSPM %s %s" % (brain_kwargs['surf'], fmin))

        if dst is None:
            if group is None:
                dst = self.get('subject-mov-file', mkdir=True)
            else:
                dst = self.get('group-mov-file', mkdir=True)
        else:
            dst = os.path.expanduser(dst)

        if not redo and self._result_mtime(dst, 'src', False, group is None):
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
        *others* :
            Experiment state parameters.
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
            self._set_analysis_options('src', sns_baseline, src_baseline, p,
                                       None, None, None, None)

            if dst is None:
                if group is None:
                    dst = self.get('subject-mov-file', mkdir=True)
                else:
                    dst = self.get('group-mov-file', mkdir=True)
            else:
                dst = os.path.expanduser(dst)

            if not redo and self._result_mtime(dst, 'src', False, group is None):
                return

            plot._brain.assert_can_save_movies()
            if group is None:
                ds = self.load_epochs_stc(subject, sns_baseline, src_baseline, cat=cat)
                y = 'src'
            else:
                ds = self.load_evoked_stc(group, sns_baseline, src_baseline, morph_ndvar=True, cat=cat)
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
        kwargs :
            State arguments

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
        kwargs :
            State arguments

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
        """
        if is_fake_mri(self.get('mri-dir', **state)):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)

        dst = self.get('res-file', mkdir=True, ext='png',
                       analysis='Source Annot',
                       resname="{parc} {mrisubject} %s" % surf)
        if not redo and os.path.exists(dst):
            return

        brain, legend = self.plot_annot(surf=surf, axw=600, show=False)
        brain.save_image(dst)
        legend.save(dst[:-3] + 'pdf', transparent=True)
        brain.close()
        legend.close()

    def make_plot_label(self, label, surf='inflated', redo=False, **state):
        if is_fake_mri(self.get('mri-dir', **state)):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)

        dst = self._make_plot_label_dst(surf, label)
        if not redo and os.path.exists(dst):
            return

        brain = self.plot_label(label, surf=surf)
        brain.save_image(dst)

    def make_plots_labels(self, surf='inflated', redo=False, **state):
        self.set(**state)
        with self._temporary_state:
            if is_fake_mri(self.get('mri-dir')):
                self.set(mrisubject=self.get('common_brain'), match=False)

            labels = self._load_labels().values()
            dsts = [self._make_plot_label_dst(surf, label.name)
                    for label in labels]
        if not redo and all(os.path.exists(dst) for dst in dsts):
            return

        brain = self.plot_brain(surf, None, 'split', ['lat', 'med'], w=1200)
        for label, dst in zip(labels, dsts):
            brain.add_label(label)
            brain.save_image(dst)
            brain.remove_labels(hemi='lh')

    def _make_plot_label_dst(self, surf, label):
        return self.get('res-deep-file', mkdir=True, analysis='Source Labels',
                        folder="{parc} {mrisubject} %s" % surf, resname=label,
                        ext='png')

    def make_proj(self, save=True, save_plot=True):
        """
        computes the first ``n_mag`` PCA components, plots them, and asks for
        user input (a tuple) on which ones to save.

        Parameters
        ----------
        epochs : mne.Epochs
            epochs which should be used for the PCA
        dest : str(path)
            path where to save the projections
        n_mag : int
            number of components to compute
        save : False | True | tuple
            False: don'r save proj fil; True: manuall pick componentws to
            include in the proj file; tuple: automatically include these
            components
        save_plot : False | str(path)
            target path to save the plot
        """
        proj = self.get('proj')
        opt = self.projs[proj]
        reject = {'mag': 1.5e-11}  # far lower than the data range
        if opt['base'] == 'raw':
            raw = self.load_raw(add_proj=False)

            # select time range of events
            events = load.fiff.events(raw)
            time = events['i_start'] / events.info['samplingrate']
            tstart = time.x.min() - 5
            tstop = time.x.max() + 5

            projs = mne.compute_proj_raw(raw, tstart, tstop, duration=5,
                                         n_grad=0, n_mag=9, n_eeg=0,
                                         reject=reject, n_jobs=1)
            fiff_obj = raw
        else:
            epoch = opt['base']
            rej = opt.get('rej', None)
            rej_ = '' if rej is None else rej
            self.set(epoch=epoch, rej=rej_)
            ds = self.load_epochs(ndvar=False, add_proj=False)
            epochs = ds['epochs']
            if rej is None:
                epochs.reject = reject.copy()
                epochs._reject_setup()
            projs = mne.compute_proj_epochs(epochs, n_grad=0, n_mag=9, n_eeg=0)
            fiff_obj = epochs

        # convert projs to NDVar
        picks = mne.epochs.pick_types(fiff_obj.info, exclude='bads')
        sensor = load.fiff.sensor_dim(fiff_obj, picks=picks)
        projs_ndvars = []
        for p in projs:
            d = p['data']['data'][0]
            name = p['desc'][-5:]
            v = NDVar(d, (sensor,), name=name)
            projs_ndvars.append(v)

        # plot PCA components
        proj_file = self.get('proj-file')
        p = plot.Topomap(projs_ndvars, title=proj_file, ncol=3, w=9)
        if save_plot:
            dest = self.get('plot-file', analysis='proj', ext='pdf',
                            name='{subject}_{experiment}_{raw_kind}')
            p.figure.savefig(dest)
        if save:
            rm = save
            title = "Select Projections"
            msg = ("which Projections do you want to select? (tuple / 'x' to "
                   "abort)")
            while not isinstance(rm, tuple):
                answer = ui.ask_str(msg, title, default='(0,)')
                rm = eval(answer)
                if rm == 'x': raise

            p.close()
            projs = [projs[i] for i in rm]
            mne.write_proj(proj_file, projs)

    def make_raw(self, redo=False, n_jobs=1, **kwargs):
        """Make a raw file

        Parameters
        ----------
        raw : str
            Name of the raw file to make.
        redo : bool
            If the file already exists, recreate it.
        n_jobs : int
            Number of processes for multiprocessing.

        Notes
        -----
        Due to the electronics of the KIT system sensors, signal lower than
        0.16 Hz is not recorded even when recording at DC.
        """
        raw_dst = self.get('raw', **kwargs)
        if self._raw[raw_dst] is None:
            raise RuntimeError("Can't make %r raw file because it is an input "
                               "file" % raw_dst)
        dst = self.get('cached-raw-file', mkdir=True)
        with self._temporary_state:
            if not redo and os.path.exists(dst):
                src = self.get('raw-file', raw='clm')
                src_mtime = os.path.getmtime(src)
                dst_mtime = os.path.getmtime(dst)
                if dst_mtime > src_mtime:
                    return

            apply_proj = False
            raw = self.load_raw(raw='clm', add_proj=apply_proj, add_bads=False,
                                preload=True)

        if apply_proj:
            raw.apply_projector()

        args, kwargs = self._raw[raw_dst]
        raw.filter(*args, n_jobs=n_jobs, **kwargs)
        raw.save(dst, overwrite=True)

    def make_rej(self, **kwargs):
        """Show the SelectEpochs GUI to do manual epoch selection for a given
        epoch

        The GUI is opened with the correct file name; if the corresponding
        file exists, it is loaded, and upon saving the correct path is
        the default.

        Parameters
        ----------
        kwargs :
            Kwargs for SelectEpochs
        """
        rej_args = self._epoch_rejection[self.get('rej')]
        if not rej_args['kind'] == 'manual':
            err = ("Epoch rejection kind for rej=%r is not manual."
                   % self.get('rej'))
            raise RuntimeError(err)

        epoch = self._epochs[self.get('epoch')]
        if 'sel_epoch' in epoch:
            msg = ("The current epoch {cur!r} inherits rejections from "
                   "{sel!r}. To access a rejection file for this epoch, call "
                   "`e.set(epoch={sel!r})` and then `e.make_rej()` "
                   "again.".format(cur=epoch['name'], sel=epoch['sel_epoch']))
            raise ValueError(msg)
        elif 'sub_epochs' in epoch:
            msg = ("The current epoch {cur!r} inherits rejections from these "
                   "other epochs: {sel!r}. To access trial rejection for these "
                   "epochs, call `e.set(epoch=epoch)` and then `e.make_rej()` "
                   "again.".format(cur=epoch['name'], sel=epoch['sub_epochs']))
            raise ValueError(msg)

        path = self.get('rej-file', mkdir=True)
        modality = self.get('modality')

        if modality == '':
            ds = self.load_epochs(reject=False, eog=True,
                                  decim=rej_args.get('decim', None),
                                  trigger_shift=False)
            subject = self.get('subject')
            subject_prefix = self._subject_re.match(subject).group(1)
            meg_system = self._meg_systems[subject_prefix]
            eog_sns = self._eog_sns[meg_system]
            data = 'meg'
            vlim = 2e-12
        elif modality == 'eeg':
            ds = self.load_epochs(reject=False, eog=True, baseline=True,
                                  decim=rej_args.get('decim', None),
                                  trigger_shift=False)
            eog_sns = self._eog_sns['KIT-BRAINVISION']
            data = 'eeg'
            vlim = 1.5e-4
        else:
            raise ValueError("modality=%r" % modality)

        # don't mark eog sns if it is bad
        bad_channels = self.load_bad_channels()
        eog_sns = [c for c in eog_sns if c not in bad_channels]

        gui.select_epochs(ds, data, path=path, vlim=vlim, mark=eog_sns, **kwargs)

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
        """
        if samples < 1:
            raise ValueError("samples needs to be > 0")

        if include <= 0 or include > 1:
            raise ValueError("include needs to be 0 < include <= 1, got %s"
                             % repr(include))

        # determine report file name
        if parc is None:
            if mask:
                folder = "%s Masked" % mask.capitalize()
            else:
                folder = "Whole Brain"
        elif mask:
            raise ValueError("Can't specify mask together with parc")
        elif pmin is None or pmin == 'tfce':
            raise NotImplementedError("Threshold-free test (pmin=%r) is not "
                                      "implemented for parcellation (parc "
                                      "parameter). Use a mask instead, or do a "
                                      "cluster-based test." % (pmin,))
        else:
            folder = "{parc}"
        self.set(**state)
        self._set_analysis_options('src', sns_baseline, src_baseline, pmin,
                                   tstart, tstop, parc, mask)
        dst = self.get('report-file', mkdir=True, fmatch=False, folder=folder,
                       test=test)
        is_twostage = self._tests[test]['kind'] == 'two-stage'
        desc = self._get_rel('report-file', 'res-dir')
        if redo:
            self._log.debug("Redoing report: %s", desc)
        elif self._result_mtime(dst, 'src', not is_twostage):
            self._log.debug("Report up to date: %s", desc)
            return
        elif os.path.exists(dst):
            self._log.debug("Report outdated: %s", desc)
        else:
            self._log.debug("New report: %s", desc)

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        if is_twostage:
            self._two_stage_report(report, test, sns_baseline, src_baseline,
                                   pmin, samples, tstart, tstop, parc, mask,
                                   include)
        else:
            self._evoked_report(report, test, sns_baseline, src_baseline, pmin,
                                samples, tstart, tstop, parc, mask, include)

        # report signature
        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst)

    def _evoked_report(self, report, test, sns_baseline, src_baseline, pmin,
                       samples, tstart, tstop, parc, mask, include):
        # load data
        ds, res = self._load_test(test, tstart, tstop, pmin, parc, mask, samples,
                                  'src', sns_baseline, src_baseline, True, True)

        # info
        surfer_kwargs = self._surfer_plot_kwargs()
        self._report_test_info(report.add_section("Test Info"), ds, test, res,
                               'src', include)
        if parc:
            section = report.add_section(parc)
            caption = "Labels in the %s parcellation." % parc
            self._report_parc_image(section, caption)
        elif mask:
            title = "Whole Brain Masked by %s" % mask
            section = report.add_section(title)
            caption = "Mask: %s" % mask.capitalize()
            self._report_parc_image(section, caption)

        model = self._tests[test]['model']
        colors = plot.colors_for_categorial(ds.eval(model))
        report.append(_report.source_time_results(res, ds, colors, include,
                                                  surfer_kwargs, parc=parc))

    def _two_stage_report(self, report, test, sns_baseline, src_baseline, pmin,
                          samples, tstart, tstop, parc, mask, include):
        rlm = self._load_test(test, tstart, tstop, pmin, parc, mask, samples,
                              'src', sns_baseline, src_baseline, False, True)

        # stage 2
        parc_dim = 'source' if parc else None
        test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop,
                                        ('time', 'source'), parc_dim)
        results = rlm._column_ttests(True, **test_kwargs)

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

        # add results to report
        for term in rlm.column_names:
            res, ds = results[term]
            report.append(_report.source_time_results(res, ds, None, include,
                                                      surfer_kwargs, term))

        self._report_test_info(info_section, ds, test, res, 'src')

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
        """
        parc = self.get('parc', parc=parc, **state)
        if not parc:
            raise ValueError("No parcellation specified")
        self._set_analysis_options('src', sns_baseline, src_baseline, pmin,
                                   tstart, tstop, parc, None)
        dst = self.get('report-file', mkdir=True, fmatch=False, test=test,
                       folder="%s ROIs" % parc.capitalize())
        if not redo and self._result_mtime(dst, 'src'):
            return

        # load data
        label_names = None
        dss = []
        for _ in self:
            ds = self.load_evoked_stc(None, sns_baseline, src_baseline, ind_ndvar=True)
            src = ds.pop('src')
            if label_names is None:
                label_keys = {k: as_legal_dataset_key(k) for k in src.source.parc.cells}
                label_names = {v: k for k, v in label_keys.iteritems()}
                if len(label_names) != len(label_keys):
                    raise RuntimeError("Label key conflict")
            elif set(label_keys) != set(src.source.parc.cells):
                raise RuntimeError("Not all subjects have the same labels")

            for name, key in label_keys.iteritems():
                ds[key] = src.summary(source=name)
            del src
            dss.append(ds)
        ds = combine(dss, incomplete='drop')
        # prune label_keys
        for name, key in label_keys.iteritems():
            if key not in ds:
                del label_keys[name]
        ds.info['label_keys'] = label_keys

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # method intro (compose it later when data is available)
        info_section = report.add_section("Test Info")

        # add parc image
        section = report.add_section(parc)
        caption = "ROIs in the %s parcellation." % parc
        self._report_parc_image(section, caption)

        # sort labels
        labels_lh = []
        labels_rh = []
        for label in label_keys:
            if label.startswith('unknown'):
                continue
            elif label.endswith('-lh'):
                labels_lh.append(label)
            elif label.endswith('-rh'):
                labels_rh.append(label)
            else:
                raise NotImplementedError("Label named %s" % repr(label.name))
        labels_lh.sort()
        labels_rh.sort()

        # add content body
        model = self._tests[test]['model']
        colors = plot.colors_for_categorial(ds.eval(model))
        test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop, ('time',), None)
        for hemi, label_names in (('Left', labels_lh), ('Right', labels_rh)):
            section = report.add_section("%s Hemisphere" % hemi)
            for label in label_names:
                res = self._make_test(ds[label_keys[label]], ds, test, test_kwargs)
                _report.roi_timecourse(section, ds, label, res, colors)

        # compose info
        self._report_test_info(info_section, ds, test, res, 'src')

        report.sign(('eelbrain', 'mne', 'surfer', 'scipy', 'numpy'))
        report.save_html(dst)

    def _make_report_eeg(self, test, pmin=None, tstart=0.15, tstop=None,
                         samples=10000, baseline=True, include=1,
                         redo=False, redo_test=False, **state):
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
        redo : bool
            If the target file already exists, delete and recreate it. This
            only applies to the HTML result file, not to the test.
        redo_test : bool
            Redo the test even if a cached file exists.
        """
        self._set_analysis_options('eeg', baseline, None, pmin, tstart, tstop,
                                   None, None)
        dst = self.get('report-file', mkdir=True, fmatch=False, test=test,
                       folder="EEG Spatio-Temporal", modality='eeg',
                       **state)
        if not redo and not redo_test and os.path.exists(dst):
            return

        # load data
        ds, res = self.load_test(None, tstart, tstop, pmin, None, None, samples,
                                 'sns', baseline, None, True, True, redo_test)

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        info_section = report.add_section("Test Info")
        self._report_test_info(info_section, ds, test, res, 'sns', include)

        # add connectivity image
        p = plot.SensorMap(ds['eeg'], connectivity=True, show=False)
        image_conn = p.image("connectivity.png")
        info_section.add_figure("Sensor map with connectivity", image_conn)
        p.close()

        model = self._tests[test]['model']
        colors = plot.colors_for_categorial(ds.eval(model))
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
        """
        self._set_analysis_options('eeg', baseline, None, pmin, tstart, tstop,
                                   None, None)
        dst = self.get('report-file', mkdir=True, fmatch=False, test=test,
                       folder="EEG Sensors", modality='eeg', **state)
        if not redo and os.path.exists(dst):
            return

        # load data
        ds = self.load_evoked(self.get('group'), baseline, True)

        # test that sensors are in the data
        eeg = ds['eeg']
        missing = [s for s in sensors if s not in eeg.sensor.names]
        if missing:
            raise ValueError("The following sensors are not in the data: %s" % missing)

        # start report
        title = self.format('{experiment} {epoch} {test} {test_options}')
        report = Report(title)

        # info
        info_section = report.add_section("Test Info")

        # add sensor map
        p = plot.SensorMap(ds['eeg'], show=False)
        p.mark_sensors(sensors)
        info_section.add_figure("Sensor map", p)
        p.close()

        # main body
        model = self._tests[test]['model']
        caption = "Signal at %s."
        colors = plot.colors_for_categorial(ds.eval(model))
        test_kwargs = self._test_kwargs(samples, pmin, tstart, tstop, ('time', 'sensor'), None)
        for sensor in sensors:
            y = eeg.sub(sensor=sensor)
            res = self._make_test(y, ds, test, test_kwargs)
            report.append(_report.time_results(res, ds, colors, sensor,
                                               caption % sensor))

        self._report_test_info(info_section, ds, test, res, 'sns')
        report.sign(('eelbrain', 'mne', 'scipy', 'numpy'))
        report.save_html(dst)

    def _report_subject_info(self, ds, model):
        # add subject information to experiment
        if model:
            s_ds = table.repmeas('n', model, 'subject', ds=ds)
        else:
            s_ds = ds
        s_ds2 = self.show_subjects(asds=True)
        s_ds.update(s_ds2[('subject', 'mri')])
        s_table = s_ds.as_table(midrule=True, count=True, caption="All "
                                "subjects included in the analysis with "
                                "trials per condition")
        return s_table

    def _report_test_info(self, section, ds, test, res, data, include=None):
        test_params = self._tests[test]

        # Analysis info
        info = List("Analysis:")
        info.add_item(self.format('epoch = {epoch} {evoked_kind} ~ {model}'))
        if data == 'src':
            info.add_item(self.format("cov = {cov}"))
            info.add_item(self.format("inv = {inv}"))
        info.add_item("test = %s  (%s)" % (test_params['kind'], test_params['desc']))
        if include is not None:
            info.add_item("Separate plots of all clusters with a p-value < %s"
                          % include)
        section.append(info)

        # Test info (for temporal tests, res is only representative)
        info = res.info_list(data is not None)
        section.append(info)

        section.append(self._report_subject_info(ds, test_params.get('model', None)))
        section.append(self.show_state(hide=('hemi', 'subject', 'mrisubject')))

    def _report_parc_image(self, section, caption):
        "Add picture of the current parcellation"
        with self._temporary_state:
            self.set(mrisubject=self.get('common_brain'))
            brain, legend = self.plot_annot(axw=500, show=False)

        content = [brain.image('parc'), legend.image('parc-legend')]
        section.add_image_figure(content, caption)

        legend.close()

    def make_src(self, redo=False, **kwargs):
        """Make the source space

        Parameters
        ----------
        redo : bool
            Recreate the source space even if the corresponding file already
            exists.
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

            if not redo and os.path.exists(dst):
                if os.path.getmtime(dst) >= os.path.getmtime(orig):
                    return

            src = self.get('src')
            subjects_dir = self.get('mri-sdir')
            mne.scale_source_space(subject, src, subjects_dir=subjects_dir)
        elif not redo and os.path.exists(dst):
            return
        else:
            src = self.get('src')
            kind, param = src.split('-')
            if kind == 'vol':
                mri = self.get('mri-file')
                bem = self.get('bem-file', fmatch=True)
                mne.setup_volume_source_space(subject, dst, pos=float(param),
                                              mri=mri, bem=bem, mindist=0.,
                                              exclude=0.,
                                              subjects_dir=self.get('mri-sdir'))
            else:
                spacing = kind + param
                mne.setup_source_space(subject, dst, spacing, overwrite=redo,
                                       subjects_dir=self.get('mri-sdir'),
                                       add_dist=True)

    def _test_kwargs(self, samples, pmin, tstart, tstop, dims, parc_dim):
        "testnd keyword arguments"
        kwargs = {'samples': samples, 'tstart': tstart, 'tstop': tstop,
                  'parc': parc_dim}
        if pmin == 'tfce':
            kwargs['tfce'] = True
        elif pmin is not None:
            kwargs['pmin'] = pmin
            kwargs.update(self._cluster_criteria_kwargs(dims))
        return kwargs

    def _make_test(self, y, ds, test, kwargs):
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
        """
        p = self._tests[test]
        kind = p['kind']
        if kind == 'ttest_rel':
            res = testnd.ttest_rel(y, p['model'], p['c1'], p['c0'], 'subject',
                                   ds=ds, tail=p.get('tail', 0), **kwargs)
        elif kind == 't_contrast_rel':
            res = testnd.t_contrast_rel(y, p['model'], p['contrast'], 'subject',
                                        ds=ds, tail=p.get('tail', 0), **kwargs)
        elif kind == 'anova':
            res = testnd.anova(y, p['x'], match='subject', ds=ds, **kwargs)
        else:
            raise RuntimeError("Test kind=%s" % repr(kind))

        return res

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
                   foreground=None, background=None, show=True):
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
        show : bool
            Show the plot (set to False to save the plot without displaying it;
            only works for the legend).

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
        brain = plot.brain.annot(parc, subject, kwa['surf'], borders, alpha,
                                 kwa['hemi'], kwa['views'], w, h, axw, axh,
                                 kwa['foreground'], kwa['background'],
                                 subjects_dir=mri_sdir)

        legend = plot.brain.annot_legend(self.get('annot-file', hemi='lh'),
                                         self.get('annot-file', hemi='rh'),
                                         show=show)

        return brain, legend

    def plot_brain(self, surf='inflated', title=None, hemi='lh', views=['lat'],
                   w=500, clear=True, common_brain=True):
        """Create a PuSyrfer Brain instance

        Parameters
        ----------
        w : int
            Total window width.
        clear : bool
            If self.brain exists, replace it with a new plot (if False,
            the existsing self.brain is returned).
        common_brain : bool
            If the current mrisubject is a scaled MRI, use the common_brain
            instead.
        """
        import surfer
        if clear:
            self.brain = None
        else:
            if self.brain is None:
                pass
            elif self.brain._figures == [[None, None], [None, None]]:
                self.brain = None
            else:
                return self.brain

        # find subject
        mri_sdir = self.get('mri-sdir')
        if common_brain and is_fake_mri(self.get('mri-dir')):
            mrisubject = self.get('common_brain')
            self.set(mrisubject=mrisubject, match=False)
        else:
            mrisubject = self.get('mrisubject')

        if title is not None:
            title = title.format(mrisubject=mrisubject)

        if hemi in ('lh', 'rh'):
            self.set(hemi=hemi)
            height = len(views) * w * 3 / 4.
        else:
            height = len(views) * w * 3 / 8.

        config_opts = dict(background=(1, 1, 1), foreground=(0, 0, 0),
                           width=w, height=height)
        brain = surfer.Brain(mrisubject, hemi, surf, True, title, config_opts,
                             None, mri_sdir, views)

        self.brain = brain
        return brain

    def plot_coreg(self, ch_type=None, **kwargs):
        """Plot the coregistration (Head shape and MEG helmet)

        Parameters
        ----------
        ch_type : 'meg' | 'eeg'
            Plot only MEG or only EEG sensors (default is both).
        """
        self.set(**kwargs)
        raw = self.load_raw()
        return mne.viz.plot_trans(raw.info, self.get('trans-file'),
                                  self.get('subject'), self.get('mri-sdir'),
                                  ch_type, 'head')

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

    def plot_evoked(self, subject=None, separate=False, baseline=True, run=None,
                    **kwargs):
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
        run : bool
            Run the GUI after plotting (default in accordance with plotting
            default).
        """
        subject, group = self._process_subject_arg(subject, kwargs)
        y = self._ndvar_name_for_modality(self.get('modality'))
        model = self.get('model') or None
        if subject:
            ds = self.load_evoked(baseline=baseline)
            return plot.TopoButterfly(y, model, ds=ds, title=subject, run=run)
        elif separate:
            plots = []
            vlim = []
            for subject in self.iter(group=group):
                ds = self.load_evoked(baseline=baseline)
                p = plot.TopoButterfly(y, model, ds=ds, title=subject, run=False)
                plots.append(p)
                vlim.append(p.get_vlim())

            # same vmax for all plots
            vlim = np.array(vlim)
            vmax = np.abs(vlim, out=vlim).max()
            for p in plots:
                p.set_vlim(vmax)

            if run or plot._base.do_autorun():
                gui.run()
        else:
            ds = self.load_evoked(group, baseline=baseline)
            return plot.TopoButterfly(y, model, ds=ds, title=subject, run=run)

    def plot_label(self, label, surf='inflated', w=600, clear=False):
        """Plot a label"""
        if isinstance(label, basestring):
            label = self.load_label(label)
        title = label.name

        brain = self.plot_brain(surf, title, 'split', ['lat', 'med'], w, clear)
        brain.add_label(label, alpha=0.75)
        return brain

    def run_mne_analyze(self, subject=None, modal=False):
        subjects_dir = self.get('mri-sdir')
        subject = subject or self.get('mrisubject')
        fif_dir = self.get('raw-dir', subject=subject)
        subp.run_mne_analyze(fif_dir, subject=subject,
                             subjects_dir=subjects_dir, modal=modal)

    def run_mne_browse_raw(self, subject=None, modal=False):
        fif_dir = self.get('raw-dir', subject=subject)
        subp.run_mne_browse_raw(fif_dir, modal)

    def set(self, subject=None, **state):
        """
        Set variable values.

        Parameters
        ----------
        subject : str
            Set the `subject` value. The corresponding `mrisubject` is
            automatically set to the corresponding mri subject.
        *other* : str
            All other keywords can be used to set templates.
        """
        if subject is not None:
            state['subject'] = subject
            if 'mrisubject' not in state:
                if 'mri' in state:
                    mri = state['mri']
                else:
                    mri = self.get('mri')
                state['mrisubject'] = self._mri_subjects[mri][subject]

        FileTree.set(self, **state)

    def _post_set_group(self, _, group):
        if group == '*' or group not in self._groups:
            return
        group_members = self._groups[group]
        self._field_values['subject'] = group_members
        if self.get('subject') not in group_members and group_members:
            self.set(group_members[0])

    def set_inv(self, ori='free', snr=3, method='dSPM', depth=None, pick=None,
                smooth=None):
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
        depth : None | float
            Depth weighting (default None).
        pick : 'normal' | '+' | '-'
            Pick a specific component of the estimates (``'normal'``: the
            normal component of the estimated current vector; ``'+'`` or
            ``'-'``: only positice or negative estimates).
        smooth : scalar
            Full-width half maximum (in m, will be rounded to the next
            millimeter).
        """
        if not isinstance(ori, basestring):
            ori = 'loose%s' % str(ori)[1:]
        items = [ori, str(snr), method]

        if depth:
            items.append(str(depth))

        if pick:
            if pick == 'normal':
                items.append('pick_normal')
            elif pick in '+-':
                items.append('pick%s' % pick)
            else:
                raise ValueError("pick=%s" % repr(pick))

        if smooth:
            items.append('sm%s' % int(round(smooth * 1e3)))

        inv = '-'.join(items)
        self.set(inv=inv)

    @staticmethod
    def _eval_inv(inv):
        m = inv_re.match(inv)
        if m is None:
            raise ValueError("Invalid inverse specification: inv=%r" % inv)

        ori, snr, method, depth, pick, smooth = m.groups()
        if ori.startswith('loose'):
            loose = float(ori[5:])
            if not 0 <= loose <= 1:
                raise ValueError('First value of inv (loose parameter) needs to be in [0, 1]')
            elif pick in ('pick+', 'pick-'):
                raise ValueError("ori='loose' and pick=%s" % repr(pick))
        elif ori == 'free' and pick in ('pick+', 'pick-'):
            raise ValueError("ori='free' and pick=%s" % repr(pick[-1]))

        return inv

    def _post_set_inv(self, _, inv):
        if '*' in inv:
            self._params['make_inv_kw'] = None
            self._params['apply_inv_kw'] = None
            self._params['post_inv_kw'] = None
            return

        m = inv_re.match(inv)
        ori, snr, method, depth, pick, smooth = m.groups()

        make_kw = {}
        apply_kw = {}
        post_kw = {}

        if ori == 'fixed':
            make_kw['fixed'] = True
            make_kw['loose'] = None
        elif ori == 'free':
            make_kw['loose'] = 1
        elif ori.startswith('loose'):
            make_kw['loose'] = float(ori[5:])

        if depth is not None:
            make_kw['depth'] = float(depth)

        apply_kw['method'] = method
        apply_kw['lambda2'] = 1. / float(snr) ** 2

        if pick:
            if pick == 'normal':
                apply_kw['pick_normal'] = True
            elif pick in ('pick+', 'pick-'):
                post_kw['pick'] = pick[-1]

        if smooth:
            post_kw['smooth'] = float(smooth) * 1e-3

        self._params['make_inv_kw'] = make_kw
        self._params['apply_inv_kw'] = apply_kw
        self._params['post_inv_kw'] = post_kw

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
                v = self._model_order.index(factor)
                ordered_factors[v] = factor
            else:
                unordered_factors.append(factor)

        # recompose
        model = [ordered_factors[v] for v in sorted(ordered_factors)]
        if unordered_factors:
            model.extend(unordered_factors)
        return '%'.join(model)

    def _eval_parc(self, parc):
        if parc in self._parcs:
            if self._parcs[parc]['kind'] == 'seeded':
                raise ValueError("Seeded parc set without size, use e.g. "
                                 "parc='%s-25'" % parc)
            else:
                return parc
        m = SEEDED_PARC_RE.match(parc)
        if m:
            name = m.group(1)
            if name in self._parcs and self._parcs[name]['kind'] == 'seeded':
                return parc
            else:
                raise ValueError("No seeded parc with name %r" % name)
        else:
            raise ValueError("parc=%r" % parc)

    def _post_set_rej(self, _, rej):
        if rej == '*':
            self._fields['cov-rej'] = '*'
        elif rej in self._epoch_rejection:
            self._fields['cov-rej'] = self._epoch_rejection[rej].get('cov-rej', rej)
        else:
            self._fields['cov-rej'] = '<invalid rej>'

    def _post_set_test(self, _, test):
        if test != '*' and test in self._tests and 'model' in self._tests[test]:
            self.set(model=self._tests[test]['model'])

    def _set_analysis_options(self, data, sns_baseline, src_baseline, pmin,
                              tstart, tstop, parc, mask):
        """Set templates for test paths with test parameters

        Can be set before or after the test template.

        Parameters
        ----------
        data : 'sns' | 'src'
            Whether the analysis is in sensor or source space.
        ...
        src_baseline :
            Should be None if data=='sns'.
        """
        # data kind (sensor or source space)
        if data == 'sns':
            analysis = '{sns_kind} {evoked_kind}'
        elif data == 'src':
            analysis = '{src_kind} {evoked_kind}'
        elif data == 'eeg':
            analysis = '{eeg_kind} {evoked_kind}'
        else:
            raise ValueError("data=%r. Needs to be 'sns', 'src' or 'eeg'" % data)

        # test properties
        items = []

        # baseline (default is baseline correcting in sensor space)
        epoch_baseline = self._epochs[self.get('epoch')]['baseline']
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
            items.append(str(pmin))
            # cluster criteria
            if pmin != 'tfce':
                select_clusters = self.get('select_clusters')
                if select_clusters:
                    items.append(select_clusters)

        # time window
        if tstart is not None or tstop is not None:
            items.append(_time_window_str((tstart, tstop)))

        # parc/mask
        if parc:
            if pmin == 'tfce':
                raise NotImplementedError("tfce analysis can't have parc")
            elif data == 'sns':
                raise NotImplementedError("sns analysis can't have parc")
            parc_ = parc
            data_parc = parc
        elif mask:
            if data == 'sns':
                raise NotImplementedError("sns analysis can't have mask")
            parc_ = mask
            if pmin is None:  # can as well collect dist for parc
                data_parc = mask
            else:  # parc means disconnecting
                data_parc = '%s-mask' % mask
        else:
            parc_ = 'aparc'
            data_parc = 'unmasked'

        self.set(test_options=' '.join(items), analysis=analysis, parc=parc_,
                 data_parc=data_parc)

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
        others :
            ``self.iter()`` kwargs.

        Examples
        --------
        >>> e.show_file_status('rej-file')
             Subject   Rej-file
        --------------------------------
         0   A0005     07/22/15 13:03:08
         1   A0008     07/22/15 13:07:57
         2   A0028     07/22/15 13:22:04
         3   A0048     07/22/15 13:25:29
        >>> e.show_file_status('rej-file', 'raw')
             Subject   0-40   0.1-40              1-40   Clm
        ----------------------------------------------------
         0   A0005     -      07/22/15 13:03:08   -      -
         1   A0008     -      07/22/15 13:07:57   -      -
         2   A0028     -      07/22/15 13:22:04   -      -
         3   A0048     -      07/22/15 13:25:29   -      -
         """
        return FileTree.show_file_status(self, temp, row, col, *args, **kwargs)

    def show_reg_params(self, asds=False, **kwargs):
        """Show the covariance matrix regularization parameters

        Parameters
        ----------
        asds : bool
            Return a dataset with the parameters (default False).
        """
        if kwargs:
            self.set(**kwargs)
        subjects = []
        reg = []
        for subject in self:
            path = self.get('cov-info-file')
            if os.path.exists(path):
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

    def show_subjects(self, mri=True, mrisubject=False, caption=True,
                      asds=False):
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
        """
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
                if not os.path.exists(mri_dir):
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
        return {'surf': surf or self.brain_plot_defaults.get('surf', 'inflated'),
                'views': views or self.brain_plot_defaults.get('views', ('lat', 'med')),
                'hemi': hemi,
                'foreground': foreground or self.brain_plot_defaults.get('foreground', None),
                'background': background or self.brain_plot_defaults.get('background', None),
                'smoothing_steps': smoothing_steps or self.brain_plot_defaults.get('smoothing_steps', None)}
