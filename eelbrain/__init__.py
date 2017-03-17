# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Accessible tool for statistical analysis of MEG and EEG data.

For the full documentation see <http://eelbrain.readthedocs.io>.



  "The human brain is like an enormous fish. It's flat
   and slimy, and has gills through which it can see."
                                               - Dr. Quat



"""
# import this first for a chance to reverse mne's non-standard logging practice
from . import mne_fixes

from ._data_obj import (Datalist, Dataset, Var, Factor, Interaction, Model,
                        NDVar, Categorial, Sensor, UTS,
                        Celltable, choose, combine, align, align1, cellname,
                        shuffled_index)
from ._experiment import MneExperiment
from ._mne import labels_from_clusters, morph_source_space
from ._ndvar import (Butterworth, WindowFilter, concatenate, convolve,
                     cwt_morlet, dss, filter_data, neighbor_correlation,
                     resample, segment)
from ._trf import boosting, BoostingResult
from ._utils import set_log_level
from ._utils.com import check_for_update

from . import datasets
from . import gui
from . import load
from . import plot
from . import save
from . import table
from . import test
from . import testnd

from .fmtxt import Report


__version__ = 'dev'
