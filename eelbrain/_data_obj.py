# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""
Data Representation
===================

Data is stored in three main vessels:

:class:`Factor`:
    stores categorical data
:class:`Var`:
    stores numeric data
:class:`NDVar`:
    stores numerical data where each cell contains an array if data (e.g., EEG
    or MEG data)


managed by

    * Dataset

"""

from __future__ import division

from collections import OrderedDict, defaultdict
from copy import deepcopy
from fnmatch import fnmatchcase
import itertools
from itertools import chain, izip
from keyword import iskeyword
from math import ceil, log10
import cPickle as pickle
import operator
import os
import re
import string
from warnings import warn

from matplotlib.ticker import FormatStrFormatter, FuncFormatter, IndexFormatter
import mne
from mne import Evoked as _mne_Evoked
from nibabel.freesurfer import read_annot
import numpy as np
from numpy import dot
import scipy.stats
from scipy.linalg import inv
from scipy.optimize import leastsq
from scipy.spatial import ConvexHull
from scipy.spatial.distance import cdist, pdist, squareform

from . import fmtxt
from . import _colorspaces as cs
from ._data_opt import gaussian_smoother
from ._utils import ui, LazyProperty, natsorted
from ._utils.numpy_utils import slice_to_arange, full_slice


preferences = dict(fullrepr=False,  # whether to display full arrays/dicts in __repr__ methods
                   repr_len=5,  # length of repr
                   dataset_str_n_cases=500,
                   var_repr_n_cases=100,
                   factor_repr_n_cases=100,
                   bool_fmt='%s',
                   float_fmt='%.6g',
                   int_fmt='%s',
                   factor_repr_use_labels=True,
                   short_repr=True,  # "A % B" vs "Interaction(A, B)"
                   )


UNNAMED = '<?>'
SEQUENCE_TYPES = (tuple, list)
_pickled_ds_wildcard = ("Pickled Dataset (*.pickled)", '*.pickled')
_tex_wildcard = ("TeX (*.tex)", '*.tex')
_tsv_wildcard = ("Plain Text Tab Separated Values (*.txt)", '*.txt')
_txt_wildcard = ("Plain Text (*.txt)", '*.txt')


class DimensionMismatchError(Exception):
    pass


class OldVersionError(Exception):
    "Trying to load a file from a version that is no longer supported"
    pass


def _effect_eye(n):
    """Effect coding for n categories. E.g.::

    Examples
    --------
    >>> _effect_eye(4)
    array([[ 1,  0,  0],
           [ 0,  1,  0],
           [ 0,  0,  1],
           [-1, -1, -1]])
    """
    x = np.empty((n, n - 1), dtype=np.int8)
    x[:n - 1] = np.eye(n - 1, dtype=np.int8)
    x[n - 1] = -1
    return x


def _effect_interaction(a, b):
    k = a.shape[1]
    out = [a[:, i, None] * b for i in range(k)]
    return np.hstack(out)


def cellname(cell, delim=' '):
    """
    Returns a consistent ``str`` representation for cells.

    * for Factor cells: the cell (str)
    * for Interaction cell: delim.join(cell).

    """
    if isinstance(cell, str):
        return cell
    elif isinstance(cell, (list, tuple)):
        return delim.join(cell)
    elif cell is None:
        return ''
    else:
        return unicode(cell)


def longname(x):
    if getattr(x, 'name', None) is not None:
        return x.name
    elif isnumeric(x) and 'longname' in x.info:
        return x.info['longname']
    elif np.isscalar(x):
        return repr(x)
    return '<unnamed>'


def rank(x, tol=1e-8):
    """
    Rank of a matrix, from
    http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html

    """
    s = np.linalg.svd(x, compute_uv=0)
    return np.sum(np.where(s > tol, 1, 0))


def check_length(objs, n=None):
    for obj in objs:
        if obj is None:
            pass
        elif n is None:
            n = len(obj)
        elif n != len(obj):
            err = ("%r has wrong length: %i (%i needed)." %
                   (obj.name, len(obj), n))
            raise ValueError(err)


def isbalanced(x):
    """Determine whether x is balanced

    Parameters
    ----------
    x : categorial
        Categorial Model, Factor or Interaction.
    """
    if ismodel(x):
        return all(isbalanced(e) for e in x.effects)
    else:
        return len({np.sum(x == c) for c in x.cells}) <= 1


def iscategorial(x):
    "factors as well as interactions are categorial"
    if isfactor(x) or isnested(x):
        return True
    elif isinteraction(x):
        return x.is_categorial
    elif ismodel(x):
        return all(iscategorial(e) for e in x.effects)
    else:
        return False


def isdataobject(x):
    return getattr(x, '_stype', None) in ("model", "var", "ndvar", "factor",
                                          "interaction", "nonbasic", "nested", "list")


def isdataset(x):
    return getattr(x, '_stype', None) == 'dataset'


def iseffect(x):
    return getattr(x, '_stype', None) in ("factor", "var", "interaction", "nonbasic", "nested")


def isdatalist(x, contains=None, test_all=True):
    """Test whether x is a Datalist instance

    Parameters
    ----------
    x : object
        Object to test.
    contains : None | class
        Test whether the content is instances of a specific class.
    test_all : bool
        If contains is provided, test all items' class (otherwise just test the
        first item).
    """
    is_dl = isinstance(x, Datalist)
    if is_dl and contains:
        if test_all:
            is_dl = all(isinstance(item, contains) for item in x)
        else:
            is_dl = isinstance(x[0], contains)
    return is_dl


def isfactor(x):
    return getattr(x, '_stype', None) == "factor"


def isinteraction(x):
    return getattr(x, '_stype', None) == "interaction"


def ismodel(x):
    return getattr(x, '_stype', None) == "model"


def isnested(x):
    "Determine whether x is nested"
    return getattr(x, '_stype', None) == "nested"


def isnestedin(item, item2):
    "Returns True if item is nested in item2, False otherwise"
    if hasattr(item, 'nestedin'):
        return item.nestedin and (item2 in find_factors(item.nestedin))
    else:
        return False


def isndvar(x):
    "Determine whether x is an NDVar"
    return getattr(x, '_stype', None) == "ndvar"


def isnumeric(x):
    "Determine wether x is numeric (a Var or an NDVar)"
    return getattr(x, '_stype', None) in ("ndvar", "var")


def isuv(x):
    "Determine whether x is univariate (a Var or a Factor)"
    return getattr(x, '_stype', None) in ("factor", "var")


def isvar(x):
    "Determine whether x is a Var"
    return getattr(x, '_stype', None) == "var"


def isboolvar(x):
    "Determine whether x is a Var whose data type is boolean"
    return isvar(x) and x.x.dtype.kind == 'b'


def isintvar(x):
    "Determine whether x is a Var whose data type is integer"
    return isvar(x) and x.x.dtype.kind in 'iu'


def is_higher_order_effect(e1, e0):
    """Determine whether e1 is a higher order term of e0

    Returns True if e1 is a higher order term of e0 (i.e., if all factors in
    e0 are contained in e1).

    Parameters
    ----------
    e1, e0 : effects
        The effects to compare.
    """
    f1s = find_factors(e1)
    return all(f in f1s for f in find_factors(e0))


def hasemptycells(x):
    "True iff a categorial has one or more empty cells"
    if isfactor(x):
        return False
    elif isinteraction(x):
        if x.is_categorial:
            for cell in x.cells:
                if not np.any(x == cell):
                    return True
        return False
    elif ismodel(x):
        for e in x.effects:
            if isinteraction(e) and e.is_categorial:
                for cell in e.cells:
                    if not np.any(e == cell):
                        return True
        return False
    raise TypeError("Need categorial (got %s)" % type(x))


def hasrandom(x):
    """True if x is or contains a random effect, False otherwise"""
    if isfactor(x) or isnested(x):
        return x.random
    elif isinteraction(x):
        for e in x.base:
            if isfactor(e) and e.random:
                return True
    elif ismodel(x):
        return any(hasrandom(e) for e in x.effects)
    return False


def as_case_identifier(x, sub=None, ds=None):
    "Coerce input to a variable that can identify each of its cases"
    if isinstance(x, basestring):
        if ds is None:
            err = ("Parameter was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if sub is not None:
        x = x[sub]

    if isvar(x):
        n = len(x.values)
    elif isfactor(x):
        n = len(x.n_cells)
    elif isinteraction(x):
        n = set(map(tuple, x.as_effects))
    else:
        raise TypeError("Need a Var, Factor or Interaction to identify cases, "
                        "got %s" % repr(x))

    if n < len(x):
        raise ValueError("%s can not serve as a case identifier because it has "
                         "at least one non-unique value" % x.name.capitalize())

    return x


def asarray(x, kind=None):
    "Coerce input to array"
    if isvar(x):
        x = x.x
    else:
        x = np.asarray(x)

    if kind is not None and x.dtype.kind not in kind:
        # boolean->int conversion
        if 'i' in kind and x.dtype.kind == 'b':
            x = x.astype(int)
        else:
            raise TypeError("Expected array of kind %r, got %r (%s)"
                            % (kind, x.dtype.kind, x.dtype))
    return x


def ascategorial(x, sub=None, ds=None, n=None):
    if isinstance(x, basestring):
        if ds is None:
            err = ("Parameter was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if iscategorial(x):
        pass
    elif isinteraction(x):
        x = Interaction([e if isfactor(e) else e.as_factor() for e in x.base])
    else:
        x = asfactor(x)

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asdataobject(x, sub=None, ds=None, n=None):
    "Convert to any data object or numpy array."
    if isinstance(x, basestring):
        if ds is None:
            err = ("Data object was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if isdataobject(x):
        pass
    elif isinstance(x, np.ndarray):
        pass
    else:
        x = Datalist(x)

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asepochs(x, sub=None, ds=None, n=None):
    "Convert to mne Epochs object"
    if isinstance(x, basestring):
        if ds is None:
            err = ("Epochs object was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if isinstance(x, mne.Epochs):
        pass
    else:
        raise TypeError("Need mne Epochs object, got %s" % repr(x))

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asfactor(x, sub=None, ds=None, n=None):
    if isinstance(x, basestring):
        if ds is None:
            err = ("Factor was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if isfactor(x):
        pass
    elif hasattr(x, 'as_factor'):
        x = x.as_factor(name=x.name)
    else:
        x = Factor(x)

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asmodel(x, sub=None, ds=None, n=None):
    if isinstance(x, basestring):
        if ds is None:
            err = ("Model was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if ismodel(x):
        pass
    else:
        x = Model(x)

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asndvar(x, sub=None, ds=None, n=None):
    if isinstance(x, basestring):
        if ds is None:
            err = ("Ndvar was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    # convert MNE objects
    if isinstance(x, (mne.Epochs, mne.EpochsArray)):
        from .load.fiff import epochs_ndvar
        x = epochs_ndvar(x)
    elif isinstance(x, _mne_Evoked):
        from .load.fiff import evoked_ndvar
        x = evoked_ndvar(x)
    elif isinstance(x, list):
        item_0 = x[0]
        if isinstance(item_0, _mne_Evoked):
            from .load.fiff import evoked_ndvar
            x = evoked_ndvar(x)

    if not isndvar(x):
        raise TypeError("NDVar required, got %s" % repr(x))

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asnumeric(x, sub=None, ds=None, n=None):
    "Var, NDVar"
    if isinstance(x, basestring):
        if ds is None:
            err = ("Numeric argument was specified as string, but no Dataset "
                   "was specified")
            raise TypeError(err)
        x = ds.eval(x)

    if not isnumeric(x):
        raise TypeError("Numeric argument required (Var or NDVar), got %s" % repr(x))

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def assub(sub, ds=None):
    "Interpret the sub argument."
    if isinstance(sub, basestring):
        if ds is None:
            err = ("the sub parameter was specified as string, but no Dataset "
                   "was specified")
            raise TypeError(err)
        sub = ds.eval(sub)
    return sub


def asuv(x, sub=None, ds=None, n=None):
    "As Var or Factor"
    if isinstance(x, basestring):
        if ds is None:
            err = ("Parameter was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        x = ds.eval(x)

    if isuv(x):
        pass
    elif all(isinstance(v, basestring) for v in x):
        x = Factor(x)
    else:
        x = Var(x)

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def asvar(x, sub=None, ds=None, n=None):
    if isinstance(x, basestring):
        if ds is None:
            err = "Var was specified as string, but no Dataset was specified"
            raise TypeError(err)
        x = ds.eval(x)

    if isvar(x):
        pass
    else:
        x = Var(x)

    if sub is not None:
        x = x[sub]

    if n is not None and len(x) != n:
        raise ValueError("Arguments have different length")

    return x


def index_ndim(index):
    """Determine the dimensionality of an index

    Parameters
    ----------
    index : numpy_index
        Any valid numpy index.

    Returns
    -------
    ndim : int
        Number of index dimensions: 0 for an index to a single element, 1 for
        an index to a sequence.
    """
    if np.iterable(index):
        return 1
    elif isinstance(index, slice):
        return 1
    elif isinstance(index, int):
        return 0
    else:
        raise TypeError("unknown index type: %s" % repr(index))


def _empty_like(obj, n=None, name=None):
    "Create an empty object of the same type as obj"
    n = n or len(obj)
    name = name or obj.name
    if isfactor(obj):
        return Factor([''], repeat=n, name=name)
    elif isvar(obj):
        return Var(np.empty(n) * np.NaN, name=name)
    elif isndvar(obj):
        shape = (n,) + obj.shape[1:]
        return NDVar(np.empty(shape) * np.NaN, dims=obj.dims, name=name)
    elif isdatalist(obj):
        return Datalist([None] * n, name, obj._fmt)
    else:
        err = "Type not supported: %s" % type(obj)
        raise TypeError(err)


# --- sorting ---

def align(d1, d2, i1='index', i2=None, out='data'):
    """
    Aligns two data-objects d1 and d2 based on two index variables, i1 and i2.

    Before aligning, d1 and d2 describe the same cases, but their order does
    not correspond. Align uses the indexes (i1 and i2) to match each case in
    d2 to a case in d1 (i.e., d1 is used as the basis for the case order).
    Cases that are not present in both d1 and d2 are dropped.


    Parameters
    ----------
    d1, d2 : data-object
        Two data objects which are to be aligned
    i1, i2 : str | Var | Factor | Interaction
        Indexes for cases in d1 and d2. If d1 and d2 are Datasets, i1 and i2
        can be keys for variables in d1 and d2 (if i2 is identical to i1 it can
        be omitted). Indexes have to supply a unique value for each case.
    out : 'data' | 'index'
        **'data'**: returns the two aligned data objects. **'index'**: returns two
        indices index1 and index2 which can be used to align the datasets with
        ``d1[index1]; d2[index2]``.


    Examples
    --------
    See `examples/datasets/align.py <https://github.com/christianbrodbeck/
    Eelbrain/blob/master/examples/datasets/align.py>`_.
    """
    if i2 is None and isinstance(i1, basestring):
        i2 = i1
    i1 = as_case_identifier(i1, ds=d1)
    i2 = as_case_identifier(i2, ds=d2)

    if not ((isvar(i1) and isvar(i2))
            or (isfactor(i1) and isfactor(i2))
            or (isinteraction(i1) and isinteraction(i2))):
        raise TypeError("i1 and i2 need to be of the same type, got: \n"
                        "i1=%s\ni2=%s." % (repr(i1), repr(i2)))

    idx1 = []
    idx2 = []
    for i, case_id in enumerate(i1):
        if case_id in i2:
            idx1.append(i)
            idx2.append(i2.index(case_id)[0])

    if out == 'data':
        if all(i == v for i, v in enumerate(idx1)):
            return d1, d2[idx2]
        else:
            return d1[idx1], d2[idx2]
    elif out == 'index':
        return idx1, idx2
    else:
        raise ValueError("Invalid value for out parameter: %r" % out)


def align1(d, idx, d_idx='index', out='data'):
    """
    Align a data object to an index

    Parameters
    ----------
    d : data object, n_cases = n1
        Data object with cases that should be aligned to idx.
    idx : Var | array_like, len = n2
        Index array to which d should be aligned.
    d_idx : str | index array, len = n1
        Indices of cases in d. If d is a Dataset, d_idx can be a name in d.
    out : 'data' | 'index'
        Return a restructured copy of d or an array of numerical indices into
        d.
    """
    idx = asuv(idx)
    if not isinstance(d_idx, basestring):
        # check d_idx length
        if isdataset(d):
            if len(d_idx) != d.n_cases:
                msg = ("d_idx does not have the same number of cases as d "
                       "(d_idx: %i, d: %i)" % (len(d_idx), d.n_cases))
                raise ValueError(msg)
        else:
            if len(d_idx) != len(d):
                msg = ("d_idx does not have the same number of cases as d "
                       "(d_idx: %i, d: %i)" % (len(d_idx), len(d)))
                raise ValueError(msg)
    d_idx = asuv(d_idx, ds=d)

    align_idx = np.empty(len(idx), int)
    for i, v in enumerate(idx):
        where = d_idx.index(v)
        if len(where) == 1:
            align_idx[i] = where[0]
        elif len(where) == 0:
            raise ValueError("%s does not occur in d_idx" % v)
        else:
            raise ValueError("%s occurs more than once in d_idx" % v)

    if out == 'data':
        return d[align_idx]
    elif out == 'index':
        return align_idx
    else:
        ValueError("Invalid value for out parameter: %r" % out)


def choose(choice, sources, name=None):
    """Combine data-objects picking from a different object for each case

    Parameters
    ----------
    choice : array of int
        Array specifying for each case from which of the sources the data should
        be taken.
    sources : list of data-objects
        Data that should be combined.
    name : str
        Name for the new data-object (optional).

    Notes
    -----
    Analogous to :func:`numpy.choose`. Only implemented for NDVars at this time.
    """
    choice = asarray(choice, 'i')
    if choice.min() < 0:
        raise ValueError("Choice can not be < 0")
    elif choice.max() > len(sources) - 1:
        raise ValueError("choice contains values exceeding the number of sources")

    s0 = sources[0]
    s1 = sources[1:]
    if isndvar(s0):
        if not all(isndvar(s) for s in s1):
            raise TypeError("Sources have different types")
        elif any(s.dims != s0.dims for s in s1):
            raise DimensionMismatchError("Sources have different dimensions")
        x = np.empty_like(s0.x)
        index_flat = np.empty(len(choice), bool)
        index = index_flat.reshape(index_flat.shape + (1,) * (x.ndim - 1))
        for i, s in enumerate(sources):
            np.equal(choice, i, index_flat)
            np.copyto(x, s.x, where=index)
        return NDVar(x, s0.dims, {}, name)
    else:
        raise NotImplementedError


class Celltable(object):
    """Divide Y into cells defined by X.

    Parameters
    ----------
    Y : data-object
        dependent measurement
    X : categorial
        Model (Factor or Interaction) for dividing Y.
    match : categorial
        Factor on which cases are matched (i.e. subject for a repeated
        measures comparisons). If several data points with the same
        case fall into one cell of X, they are combined using
        match_func. If match is not None, Celltable.groups contains the
        {Xcell -> [match values of data points], ...} mapping corres-
        ponding to self.data
    sub : bool array
        Bool array of length N specifying which cases to include
    match_func : callable
        see match
    cat : None | sequence of cells of X
        Only retain data for these cells. Data will be sorted in the order
        of cells occuring in cat.
    ds : Dataset
        If a Dataset is specified, input items (Y / X / match / sub) can
        be str instead of data-objects, in which case they will be
        retrieved from the Dataset.
    coercion : callable
        Function to convert the Y parameter to to the dependent varaible
        (default: asdataobject).


    Examples
    --------
    Split a repeated-measure variable Y into cells defined by the
    interaction of A and B::

        >>> c = Celltable(Y, A % B, match=subject)


    Attributes
    ----------
    .Y, .X,
        Y and X after sub was applied.
    .sub, .match:
        Input arguments.
    .cells : list of (str | tuple)
        List of all cells in X.
    .data : dict(cell -> data)
        Data (``Y[index]``) in each cell.
    .data_indexes : dict(cell -> index-array)
        For each cell, a boolean-array specifying the index for that cell in
        ``X``.

    **If ``match`` is specified**:

    .within : dict(cell1, cell2 -> bool)
        Dictionary that specifies for each cell pair whether the corresponding
        comparison is a repeated-measures or an independent measures
        comparison (only available when the input argument ``match`` is
        specified.
    .all_within : bool
        Whether all comparison are repeated-measures comparisons or not.
    .groups : dict(cell -> group)
        A slice of the match argument describing the group members for each
        cell.

    """
    def __init__(self, Y, X=None, match=None, sub=None, match_func=np.mean,
                 cat=None, ds=None, coercion=asdataobject):
        self.sub = sub
        sub = assub(sub, ds)
        if X is None:
            Y = coercion(Y, sub, ds)
        else:
            X = ascategorial(X, sub, ds)
            if cat is not None:
                # determine cat
                is_none = list(c is None for c in cat)
                if any(is_none):
                    if len(cat) == len(X.cells):
                        if all(is_none):
                            cat = X.cells
                        else:
                            cells = [c for c in X.cells if c not in cat]
                            cat = tuple(cells.pop(0) if c is None else c
                                        for c in cat)
                    else:
                        err = ("Categories can only be specified as None if X "
                               "contains exactly as many cells as categories are "
                               "required (%i)." % len(cat))
                        raise ValueError(err)

                if not isinteraction(X):
                    cat = tuple(str(c) for c in cat)

                # make sure all categories are in data
                missing = [c for c in cat if c not in X.cells]
                if missing:
                    raise ValueError("Categories not in data: %s" % ', '.join(missing))

                # apply cat
                sort_idx = X.sort_idx(order=cat)
                X = X[sort_idx]
                if sub is None:
                    sub = sort_idx
                else:
                    imax = max(len(sub), np.max(sub))
                    sub = np.arange(imax)[sub][sort_idx]
            Y = coercion(Y, sub, ds, len(X))

        if match is not None:
            match = ascategorial(match, sub, ds, len(Y))
            cell_model = match if X is None else X % match
            sort_idx = None
            if len(cell_model) > len(cell_model.cells):
                # need to aggregate
                Y = Y.aggregate(cell_model)
                match = match.aggregate(cell_model)
                if X is not None:
                    X = X.aggregate(cell_model)
                    if cat is not None:
                        sort_idx = X.sort_idx(order=cat)
            else:
                sort_idx = cell_model.sort_idx()
                if X is not None and cat is not None:
                    X_ = X[sort_idx]
                    sort_X_idx = X_.sort_idx(order=cat)
                    sort_idx = sort_idx[sort_X_idx]

            if (sort_idx is not None) and (not np.all(np.diff(sort_idx) == 1)):
                Y = Y[sort_idx]
                match = match[sort_idx]
                if X is not None:
                    X = X[sort_idx]

        # save args
        self.Y = Y
        self.X = X
        self.cat = cat
        self.match = match
        self.coercion = coercion.__name__
        self.n_cases = len(Y)

        # extract cell data
        self.data = {}
        self.data_indexes = {}
        if X is None:
            self.data[None] = Y
            self.data_indexes[None] = full_slice
            self.cells = [None]
            self.n_cells = 1
            self.all_within = match is not None
            return
        self.cells = X.cells
        self.n_cells = len(self.cells)
        self.groups = {}
        for cell in X.cells:
            idx = X.index_opt(cell)
            self.data_indexes[cell] = idx
            self.data[cell] = Y[idx]
            if match:
                self.groups[cell] = match[idx]

        # determine which comparisons are within subject comparisons
        if match:
            self.within = {}
            for cell1, cell2 in itertools.combinations(X.cells, 2):
                group1 = self.groups[cell1]
                if len(group1) == 0:
                    continue
                group2 = self.groups[cell2]
                if len(group2) == 0:
                    continue
                within = np.all(group1 == group2)
                self.within[cell1, cell2] = within
                self.within[cell2, cell1] = within
            self.any_within = any(self.within.values())
            self.all_within = all(self.within.values())
        else:
            self.any_within = False
            self.all_within = False

    def __repr__(self):
        args = [self.Y.name, self.X.name]
        rpr = "Celltable(%s)"
        if self.match is not None:
            args.append("match=%s" % self.match.name)
        if self.sub is not None:
            if isvar(self.sub):
                args.append('sub=%s' % self.sub.name)
            else:
                indexes = ' '.join(str(i) for i in self.sub[:4])
                args.append("sub=[%s...]" % indexes)
        if self.coercion != 'asdataobject':
            args.append("coercion=%s" % self.coercion)
        return rpr % (', '.join(args))

    def __len__(self):
        return self.n_cells

    def cellname(self, cell, delim=' '):
        """Produce a str label for a cell.

        Parameters
        ----------
        cell : tuple | str
            Cell.
        delim : str
            Interaction cells (represented as tuple of strings) are joined by
            ``delim``.
        """
        return cellname(cell, delim=delim)

    def cellnames(self, delim=' '):
        """Returns a list of all cell names as strings.

        See Also
        --------
        .cellname : Produce a str label for a single cell.
        """
        return [cellname(cell, delim) for cell in self.cells]

    def data_for_cell(self, cell):
        """Retrieve data for a cell, allowing advanced cell combinations

        Parameters
        ----------
        cell : str | tuple of str
            Name fo the cell. See notes for special cell names. After a special
            cell is retrieved for the first time it is also add to
            ``self.data``.

        Notes
        -----
        Special cell names can be used to retrieve averages between different
        primary cells. The names should be composed so that a case sensitive
        version of fnmatch will find the source cells. For examples, if all
        cells are ``[('a', '1'), ('a', '2'), ('b', '1'), ('b', '2')]``,
        ``('a', '*')`` will retrieve the average of ``('a', '1')`` and
        ``('a', '2')``.
        """
        if cell in self.data:
            return self.data[cell]

        # find cells matched by `cell`
        if isinstance(cell, basestring):
            cells = [c for c in self.cells if fnmatchcase(c, cell)]
            name = cell
        else:
            cells = [c for c in self.cells if all(fnmatchcase(c_, cp)
                                                  for c_, cp in izip(c, cell))]
            name = '|'.join(cell)

        # check that all are repeated measures
        for cell1, cell2 in itertools.combinations(cells, 2):
            if not self.within[(cell1, cell2)]:
                err = ("Combinatory cells can only be formed from repeated "
                       "measures cells, %r and %r are not." % (cell1, cell2))
                raise ValueError(err)

        # combine data
        cell0 = cells[0]
        x = np.empty_like(self.data[cell0].x)
        for cell_ in cells:
            x += self.data[cell_].x
        x /= len(cells)
        out = NDVar(x, cell0.dims, {}, name)
        self.data[cell] = out
        return out


    def get_data(self, out=list):
        if out is dict:
            return self.data
        elif out is list:
            return [self.data[cell] for cell in self.cells]

    def get_statistic(self, func=np.mean, a=1, **kwargs):
        """
        Returns a list with a * func(data) for each data cell.

        Parameters
        ----------

        func : callable | str
            statistics function that is applied to the data. Can be string,
            such as '[X]sem', '[X]std', or '[X]ci', e.g. '2sem'.
        a : scalar
            Multiplier (if not provided in ``function`` string).
        kwargs :
            Are submitted to the statistic function.


        Notes
        ----

        :py:meth:`get_statistic_dict`


        See also
        --------

        Celltable.get_statistic_dict : return statistics in a dict

        """
        if isinstance(func, basestring):
            if func.endswith('ci'):
                if len(func) > 2:
                    a = float(func[:-2])
                elif a == 1:
                    a = .95
                from ._stats.stats import confidence_interval
                func = confidence_interval
            elif func.endswith('sem'):
                if len(func) > 3:
                    a = float(func[:-3])
                func = scipy.stats.sem
            elif func.endswith('std'):
                if len(func) > 3:
                    a = float(func[:-3])
                func = np.std
                if 'ddof' not in kwargs:
                    kwargs['ddof'] = 1
            else:
                raise ValueError('unrecognized statistic: %r' % func)

        Y = [a * func(self.data[cell].x, **kwargs) for cell in self.cells]
        return Y

    def get_statistic_dict(self, func=np.mean, a=1, **kwargs):
        """
        Same as :py:meth:`~Celltable.get_statistic`, except that he result is returned in
        a {cell: value} dictionary.

        """
        return zip(self.cells, self.get_statistic(func=func, a=a, **kwargs))


def combine(items, name=None, check_dims=True, incomplete='raise',
            fill_in_missing=None):
    """Combine a list of items of the same type into one item.

    Parameters
    ----------
    items : collection
        Collection (:py:class:`list`, :py:class:`tuple`, ...) of data objects
        of a single type (Dataset, Var, Factor, NDVar or Datalist).
    name : None | str
        Name for the resulting data-object. If None, the name of the combined
        item is the common prefix of all items.
    check_dims : bool
        For NDVars, check dimensions for consistency between items (e.g.,
        channel locations in a Sensor dimension). Default is ``True``. Set to
        ``False`` to ignore non-fatal mismatches.
    incomplete : "raise" | "drop" | "fill in"
        Only applies when combining Datasets: how to handle variables that are
        missing from some of the input Datasets. With ``"raise"`` (default), a
        KeyError to be raised. With ``"drop"``, partially missing variables are
        dropped. With ``"fill in"``, they are retained and missing values are
        filled in with empty values (``""`` for factors, ``NaN`` for variables).

    Notes
    -----
    The info dict inherits only entries that are equal (``x is y or
    np.array_equal(x, y)``) for all items.
    """
    if fill_in_missing is not None:
        warn("The fill_in_missing argument to combine() is deprecated and will "
             "be removed after version 0.20. Use the new incomplete argument "
             "instead.", DeprecationWarning)
        incomplete = 'fill in' if fill_in_missing else 'raise'
    elif not isinstance(incomplete, basestring):
        warn("The fill_in_missing argument to combine() has ben renamed to "
             "`incomplete` and should be a string (got %s). After version 0.20 "
             "this will raise an error" % repr(incomplete), DeprecationWarning)
        incomplete = 'fill in' if incomplete else 'raise'
    elif incomplete not in ('raise', 'drop', 'fill in'):
        raise ValueError("incomplete=%s" % repr(incomplete))

    # check input
    if len(items) == 0:
        raise ValueError("combine() called with empty sequence %s" % repr(items))

    # find type
    stypes = set(getattr(item, '_stype', None) for item in items)
    if None in stypes:
        raise TypeError("Can only combine data-objects, got at least one other "
                        "item.")
    elif len(stypes) > 1:
        raise TypeError("All items to be combined need to have the same type, "
                        "got %s." % ', '.join(tuple(stypes)))
    stype = stypes.pop()

    # find name
    if name is None:
        names = filter(None, (item.name for item in items))
        name = os.path.commonprefix(names) or None

    # combine objects
    if stype == 'dataset':
        out = Dataset(name=name, info=_merge_info(items))
        item0 = items[0]
        if incomplete == 'fill in':
            # find all keys and data types
            keys = item0.keys()
            sample = dict(item0)
            for item in items:
                for key in item.keys():
                    if key not in keys:
                        keys.append(key)
                        sample[key] = item[key]
            # create output
            for key in keys:
                pieces = [ds[key] if key in ds else
                          _empty_like(sample[key], ds.n_cases) for ds in items]
                out[key] = combine(pieces, check_dims=check_dims)
        else:
            keys = set(item0)
            if incomplete == 'raise':
                if any(set(item) != keys for item in items[1:]):
                    raise KeyError("Datasets have unequal keys. Combine with "
                                   "fill_in_missing=True to combine anyways.")
                out_keys = item0
            else:
                keys.intersection_update(*items[1:])
                out_keys = (k for k in item0 if k in keys)

            for key in out_keys:
                out[key] = combine([ds[key] for ds in items])
        return out
    elif stype == 'var':
        x = np.hstack(i.x for i in items)
        return Var(x, name, info=_merge_info(items))
    elif stype == 'factor':
        random = set(f.random for f in items)
        if len(random) > 1:
            raise ValueError("Factors have different values for random parameter")
        random = random.pop()
        item0 = items[0]
        labels = item0._labels
        if all(f._labels == labels for f in items[1:]):
            x = np.hstack(f.x for f in items)
            return Factor(x, name, random, labels=labels)
        else:
            x = sum((i.as_labels() for i in items), [])
            return Factor(x, name, random)
    elif stype == 'ndvar':
        v_have_case = [v.has_case for v in items]
        if all(v_have_case):
            has_case = True
            all_dims = (item.dims[1:] for item in items)
        elif any(v_have_case):
            raise DimensionMismatchError("Some items have a 'case' dimension, "
                                         "others do not")
        else:
            has_case = False
            all_dims = (item.dims for item in items)

        dims = reduce(lambda x, y: intersect_dims(x, y, check_dims), all_dims)
        idx = {d.name: d for d in dims}
        items = [item.sub(**idx) for item in items]
        if has_case:
            x = np.concatenate([v.x for v in items], axis=0)
        else:
            x = np.array([v.x for v in items])
        dims = ('case',) + dims
        return NDVar(x, dims, _merge_info(items), name)
    elif stype == 'list':
        return Datalist(sum(items, []), name, items[0]._fmt)
    else:
        raise RuntimeError("combine with stype = %r" % stype)


def _is_equal(a, b):
    "Test equality, taking into account array values"
    if a is b:
        return True
    elif type(a) is not type(b):
        return False
    a_iterable = np.iterable(a)
    b_iterable = np.iterable(b)
    if a_iterable != b_iterable:
        return False
    elif not a_iterable:
        return a == b
    elif len(a) != len(b):
        return False
    elif isinstance(a, np.ndarray):
        if a.shape == b.shape:
            return (a == b).all()
        else:
            return False
    elif isinstance(a, SEQUENCE_TYPES):
        return all(_is_equal(a_, b_) for a_, b_ in izip(a, b))
    elif isinstance(a, dict):
        if a.viewkeys() == b.viewkeys():
            return all(_is_equal(a[k], b[k]) for k in a)
        else:
            return False
    else:
        return a == b


def _merge_info(items):
    "Merge info dicts from several objects"
    info0 = items[0].info
    other_infos = [i.info for i in items[1:]]
    # find shared keys
    info_keys = set(info0.keys())
    for info in other_infos:
        info_keys.intersection_update(info.keys())
    # find shared values
    out = {}
    for key in info_keys:
        v0 = info0[key]
        if all(_is_equal(info[key], v0) for info in other_infos):
            out[key] = v0
    return out


def find_factors(obj):
    "returns a list of all factors contained in obj"
    if isinstance(obj, EffectList):
        f = set()
        for e in obj:
            f.update(find_factors(e))
        return EffectList(f)
    elif isuv(obj):
        return EffectList([obj])
    elif ismodel(obj):
        f = set()
        for e in obj.effects:
            f.update(find_factors(e))
        return EffectList(f)
    elif isnested(obj):
        return find_factors(obj.effect)
    elif isinteraction(obj):
        return obj.base
    else:  # NonbasicEffect
        try:
            return EffectList(obj.factors)
        except:
            raise TypeError("%r has no factors" % obj)


class EffectList(list):
    def __repr__(self):
        return 'EffectList((%s))' % ', '.join(self.names())

    def __contains__(self, item):
        for f in self:
            if ((f.name == item.name) and (f._stype == item._stype)
                and (len(f) == len(item)) and np.all(item == f)):
                return True
        return False

    def index(self, item):
        for i, f in enumerate(self):
            if (len(f) == len(item)) and np.all(item == f):
                return i
        raise ValueError("Factor %r not in EffectList" % item.name)

    def names(self):
        names = [e.name if isuv(e) else repr(e) for e in self]
        return [UNNAMED if n is None else n for n in names]


class Var(object):
    """Container for scalar data.

    Parameters
    ----------
    x : array_like
        Data; is converted with ``np.asarray(x)``. Multidimensional arrays
        are flattened as long as only 1 dimension is longer than 1.
    name : str | None
        Name of the variable
    repeat : int | array of int
        repeat each element in ``x``, either a constant or a different number
        for each element.
    tile : int
        Repeat ``x`` as a whole ``tile`` many times.
    info : dict
        Info dictionary. The "longname" entry is used for display purposes.

    Attributes
    ----------
    x : numpy.ndarray
        The data stored in the Var.
    name : None | str
        The Var's name.

    Notes
    -----
    While :py:class:`Var` objects support a few basic operations in a
    :py:mod:`numpy`-like fashion (``+``, ``-``, ``*``, ``/``, ``//``), their
    :py:attr:`Var.x` attribute provides access to the corresponding
    :py:class:`numpy.array` which can be used for anything more complicated.
    :py:attr:`Var.x` can be read and modified, but should not be replaced.
    """
    _stype = "var"
    ndim = 1

    def __init__(self, x, name=None, repeat=1, tile=1, info=None):
        if isinstance(x, basestring):
            raise TypeError("Var can't be initialized with a string")

        x = np.asarray(x)
        if x.ndim > 1:
            if np.count_nonzero(i > 1 for i in x.shape) <= 1:
                x = np.ravel(x)
            else:
                err = ("X needs to be one-dimensional. Use NDVar class for "
                       "data with more than one dimension.")
                raise ValueError(err)

        if not (isinstance(repeat, int) and repeat == 1):
            x = np.repeat(x, repeat)

        if tile > 1:
            x = np.tile(x, tile)

        if info is None:
            info = {}
        elif not isinstance(info, dict):
            raise TypeError("type(info)=%s; need dict" % type(info))

        self.__setstate__((x, name, info))

    def __setstate__(self, state):
        if len(state) == 3:
            x, name, info = state
        else:
            x, name = state
            info = {}
        # raw
        self.name = name
        self.x = x
        self.info = info
        # constants
        self._n_cases = len(x)
        self.df = 1
        self.random = False

    def __getstate__(self):
        return (self.x, self.name, self.info)

    def __repr__(self, full=False):
        n_cases = preferences['var_repr_n_cases']

        if isintvar(self):
            fmt = preferences['int_fmt']
        elif isboolvar(self):
            fmt = preferences['bool_fmt']
        else:
            fmt = preferences['float_fmt']

        if full or len(self.x) <= n_cases:
            x = [fmt % v for v in self.x]
        else:
            x = [fmt % v for v in self.x[:n_cases]]
            x.append('... (N=%s)' % len(self.x))

        args = ['[%s]' % ', '.join(x)]
        if self.name is not None:
            args.append('name=%r' % self.name)

        if self.info:
            args.append('info=%r' % self.info)

        return "Var(%s)" % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

    @property
    def __array_interface__(self):
        return self.x.__array_interface__

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, index):
        "if Factor: return new variable with mean values per Factor category"
        if isfactor(index):
            f = index
            x = []
            for v in np.unique(f.x):
                x.append(np.mean(self.x[f == v]))
            return Var(x, self.name, info=self.info.copy())
        elif isvar(index):
            index = index.x

        x = self.x[index]
        if np.iterable(x):
            return Var(x, self.name, info=self.info.copy())
        else:
            return x

    def __setitem__(self, index, value):
        self.x[index] = value

    def __contains__(self, value):
        return value in self.x

    # numeric ---
    def __neg__(self):
        x = -self.x
        info = self.info.copy()
        info['longname'] = '-' + longname(self)
        return Var(x, info=info)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.abs()

    def __add__(self, other):
        if isdataobject(other):
            # ??? should Var + Var return sum or Model?
            return Model((self, other))

        x = self.x + other
        info = self.info.copy()
        info['longname'] = longname(self) + ' + ' + longname(other)
        return Var(x, info=info)

    def __iadd__(self, other):
        self.x += other
        return self

    def __radd__(self, other):
        if np.isscalar(other):
            x = other + self.x
        elif len(other) != len(self):
            raise ValueError("Objects have different length (%i vs "
                             "%i)" % (len(other), len(self)))
        else:
            x = other + self.x

        info = self.info.copy()
        info['longname'] = longname(other) + ' + ' + longname(self)
        return Var(x, info=info)

    def __sub__(self, other):
        "subtract: values are assumed to be ordered. Otherwise use .sub method."
        if np.isscalar(other):
            x = self.x - other
        elif len(other) != len(self):
            err = ("Objects have different length (%i vs "
                   "%i)" % (len(self), len(other)))
            raise ValueError(err)
        else:
            x = self.x - other.x

        info = self.info.copy()
        info['longname'] = longname(self) + ' - ' + longname(other)
        return Var(x, info=info)

    def __isub__(self, other):
        self.x -= other
        return self

    def __rsub__(self, other):
        if np.isscalar(other):
            x = other - self.x
        elif len(other) != len(self):
            raise ValueError("Objects have different length (%i vs "
                             "%i)" % (len(other), len(self)))
        else:
            x = other - self.x

        info = self.info.copy()
        info['longname'] = longname(other) + ' - ' + longname(self)
        return Var(x, info=info)

    def __mul__(self, other):
        if iscategorial(other):
            return Model((self, other, self % other))
        elif isvar(other):
            x = self.x * other.x
        else:
            x = self.x * other

        info = self.info.copy()
        info['longname'] = longname(self) + ' * ' + longname(other)
        return Var(x, info=info)

    def __imul__(self, other):
        self.x *= other
        return self

    def __rmul__(self, other):
        if np.isscalar(other):
            x = other * self.x
        elif len(other) != len(self):
            raise ValueError("Objects have different length (%i vs "
                             "%i)" % (len(other), len(self)))
        else:
            x = other * self.x

        info = self.info.copy()
        info['longname'] = longname(other) + ' * ' + longname(self)
        return Var(x, info=info)

    def __floordiv__(self, other):
        if isvar(other):
            x = self.x // other.x
        else:
            x = self.x // other

        info = self.info.copy()
        info['longname'] = longname(self) + ' // ' + longname(other)
        return Var(x, info=info)

    def __ifloordiv__(self, other):
        self.x //= other
        return self

    def __mod__(self, other):
        if ismodel(other):
            return Model(self) % other
        elif isvar(other):
            x = self.x % other.x
        elif isdataobject(other):
            return Interaction((self, other))
        else:
            x = self.x % other

        info = self.info.copy()
        info['longname'] = longname(self) + ' % ' + longname(other)
        return Var(x, info=info)

    def __imod__(self, other):
        self.x %= other
        return self

    def __lt__(self, y):
        return self.x < y

    def __le__(self, y):
        return self.x <= y

    def __eq__(self, y):
        return self.x == y

    def __ne__(self, y):
        return self.x != y

    def __gt__(self, y):
        return self.x > y

    def __ge__(self, y):
        return self.x >= y

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __div__(self, other):
        """
        type of other:
        scalar:
            returns var divided by other
        Factor:
            returns a separate slope for each level of the Factor; needed for
            ANCOVA

        """
        if isvar(other):
            x = self.x / other.x
        elif iscategorial(other):
            dummy_factor = other.as_dummy_complete
            codes = dummy_factor * self.as_effects
            # center
            means = codes.sum(0) / dummy_factor.sum(0)
            codes -= dummy_factor * means
            # create effect
            name = '%s per %s' % (self.name, other.name)
            return NonbasicEffect(codes, [self, other], name,
                                  beta_labels=other.dummy_complete_labels)
        else:
            x = self.x / other

        info = self.info.copy()
        info['longname'] = longname(self) + ' / ' + longname(other)
        return Var(x, info=info)

    def __idiv__(self, other):
        self.x /= other
        return self

    def __pow__(self, other):
        if isvar(other):
            x = self.x / other.x
        else:
            x = self.x / other
        info = self.info.copy()
        info['longname'] = longname(self) + ' ** ' + longname(other)
        return Var(x, info=info)

    def _coefficient_names(self, method):
        return longname(self),

    def abs(self, name=None):
        "Return a Var with the absolute value."
        info = self.info.copy()
        info['longname'] = 'abs(' + longname(self) + ')'
        return Var(np.abs(self.x), name, info=info)

    def argmax(self):
        """:func:`numpy.argmax`"""
        return np.argmax(self.x)

    def argmin(self):
        """:func:`numpy.argmin`"""
        return np.argmin(self.x)

    def argsort(self, kind='quicksort'):
        """:func:`numpy.argsort`

        Parameters
        ----------
        kind : 'quicksort' | 'mergesort' | 'heapsort'
            Sorting algorithm (default 'quicksort').

        Returns
        -------
        index_array : array of int
            Array of indices that sort `a` along the specified axis.
            In other words, ``a[index_array]`` yields a sorted `a`.
        """
        return np.argsort(self.x, kind=kind)

    @property
    def as_dummy(self):
        "for effect initialization"
        return self.x[:, None]

    @property
    def as_effects(self):
        "for effect initialization"
        return self.centered()[:, None]

    def as_factor(self, labels='%r', name=True, random=False):
        """Convert the Var into a Factor

        Parameters
        ----------
        labels : str | dict
            Either a format string for converting values into labels (default:
            ``'%r'``) or a dictionary mapping values to labels (see examples).
            In a dictionary, multiple values can be assigned the same label by
            providing multiple keys in a tuple. A special key 'default' can be
            used to assign values that are not otherwise specified in the
            dictionary (by default this is the empty string ``''``).
        name : None | True | str
            Name of the output Factor, ``True`` to keep the current name
            (default ``True``).
        random : bool
            Whether the Factor is a random Factor (default ``False``).

        Examples
        --------
        >>> v = Var([0, 1, 2, 3])
        >>> v.as_factor()
        Factor(['0', '1', '2', '3'])
        >>> v.as_factor({0: 'a', 1: 'b'})
        Factor(['a', 'b', '', ''])
        >>> v.as_factor({(0, 1): 'a', (2, 3): 'b'})
        Factor(['a', 'a', 'b', 'b'])
        >>> v.as_factor({0: 'a', 1: 'b', 'default': 'c'})
        Factor(['a', 'b', 'c', 'c'])
        """
        labels_ = {}
        if isinstance(labels, dict):
            # flatten
            for key, v in labels.iteritems():
                if isinstance(key, SEQUENCE_TYPES):
                    for k in key:
                        labels_[k] = v
                else:
                    labels_[key] = v

            default = labels_.pop('default', '')
            if default is not None:
                for key in np.unique(self.x):
                    if key not in labels_:
                        labels_[key] = default
        else:
            for value in np.unique(self.x):
                labels_[value] = labels % value

        if name is True:
            name = self.name

        return Factor(self.x, name, random, labels=labels_)

    def centered(self):
        return self.x - self.x.mean()

    def copy(self, name=True):
        "returns a deep copy of itself"
        x = self.x.copy()
        if name is True:
            name = self.name
        return Var(x, name, info=deepcopy(self.info))

    def count(self):
        """Count the number of occurrence of each value

        Notes
        -----
        Counting starts with zero (see examples). This is to facilitate
        integration with indexing.

        Examples
        --------
        >>> v = Var([1, 2, 3, 1, 1, 1, 3])
        >>> v.count()
        Var([0, 0, 0, 1, 2, 3, 1])
        """
        x = np.empty(len(self.x), int)
        index = np.empty(len(self.x), bool)
        for v in np.unique(self.x):
            np.equal(self.x, v, index)
            x[index] = np.arange(index.sum())
        return Var(x)

    def aggregate(self, X, func=np.mean, name=True):
        """Summarize cases within cells of X

        Parameters
        ----------
        X : categorial
            Model defining cells in which to aggregate.
        func : callable
            Function that converts arrays into scalars, used to summarize data
            within each cell of X.
        name : None | True | str
            Name of the output Var, ``True`` to keep the current name (default
            ``True``).

        Returns
        -------
        aggregated_var : Var
            A Var instance with a single value for each cell in X.
        """
        if len(X) != len(self):
            err = "Length mismatch: %i (Var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            x_cell = self.x[X == cell]
            if len(x_cell) > 0:
                x.append(func(x_cell))

        if name is True:
            name = self.name

        x = np.array(x)
        return Var(x, name, info=self.info.copy())

    @property
    def beta_labels(self):
        return [self.name]

    def diff(self, X, v1, v2, match):
        """
        Subtract X==v2 from X==v1; sorts values according to match (ascending)

        Parameters
        ----------
        X : categorial
            Model to define cells.
        v1, v2 : str | tuple
            Cells on X for subtraction.
        match : categorial
            Model that defines how to mach cells in v1 to cells in v2.
        """
        raise NotImplementedError
        # FIXME: use celltable
        assert isfactor(X)
        I1 = (X == v1);         I2 = (X == v2)
        Y1 = self[I1];          Y2 = self[I2]
        m1 = match[I1];         m2 = match[I2]
        s1 = np.argsort(m1);    s2 = np.argsort(m2)
        y = Y1[s1] - Y2[s2]
        name = "{n}({x1}-{x2})".format(n=self.name,
                                       x1=X.cells[v1],
                                       x2=X.cells[v2])
        return Var(y, name, info=self.info.copy())

    @classmethod
    def from_dict(cls, base, values, name=None, default=0, info=None):
        """
        Construct a Var object by mapping ``base`` to ``values``.

        Parameters
        ----------
        base : sequence
            Sequence to be mapped to the new Var.
        values : dict
            Mapping from values in base to values in the new Var.
        name : None | str
            Name for the new Var.
        default : scalar
            Default value to supply for entries in ``base`` that are not in
            ``values``.

        Examples
        --------
        >>> base = Factor('aabbcde')
        >>> Var.from_dict(base, {'a': 5, 'e': 8}, default=0)
        Var([5, 5, 0, 0, 0, 0, 8])
        """
        return cls([values.get(b, default) for b in base], name, info=info)

    @classmethod
    def from_apply(cls, base, func, name=None, info=None):
        """
        Construct a Var instance by applying a function to each value in a base

        Parameters
        ----------
        base : sequence, len = n
            Base for the new Var. Can be an NDVar, if ``func`` is a
            dimensionality reducing function such as :func:`numpy.mean`.
        func : callable
            A function that when applied to each element in ``base`` returns
            the desired value for the resulting Var.
        """
        if isvar(base) or isndvar(base):
            base = base.x

        if isinstance(func, np.ufunc):
            x = func(base)
        elif getattr(base, 'ndim', 1) > 1:
            x = func(base.reshape((len(base), -1)), axis=1)
        else:
            x = np.array([func(val) for val in base])

        return cls(x, name, info=info)

    def index(self, value):
        "``v.index(value)`` returns an array of indices where v equals value"
        return np.flatnonzero(self == value)

    def isany(self, *values):
        "Boolean index, True where the Var is equal to one of the values"
        return np.in1d(self.x, values)

    def isin(self, values):
        "Boolean index, True where the Var value is in values"
        return np.in1d(self.x, values)

    def isnot(self, *values):
        "Boolean index, True where the Var is not equal to one of the values"
        return np.in1d(self.x, values, invert=True)

    def isnotin(self, values):
        "Boolean index, True where the Var value is not in values"
        return np.in1d(self.x, values, invert=True)

    def log(self, name=None):
        "Natural logarithm"
        info = self.info.copy()
        info['longname'] = 'log(' + longname(self) + ')'
        return Var(np.log(self.x), name, info=info)

    def max(self):
        "Returns the highest value"
        return self.x.max()

    def mean(self):
        "Returns the mean"
        return self.x.mean()

    def min(self):
        "Returns the smallest value"
        return self.x.min()

    def repeat(self, repeats, name=True):
        """
        Repeat each element ``repeats`` times

        Parameters
        ----------
        repeats : int | array of int
            Number of repeats, either a constant or a different number for each
            element.
        name : None | True | str
            Name of the output Var, ``True`` to keep the current name (default
            ``True``).
        """
        if name is True:
            name = self.name
        return Var(self.x.repeat(repeats), name, info=self.info.copy())

    def split(self, n=2, name=None):
        """
        A Factor splitting Y in ``n`` categories with equal number of cases

        Parameters
        ----------
        n : int
            number of categories
        name : str
            Name of the output Factor.

        Examples
        --------
        Use n = 2 for a median split::

            >>> y = Var([1,2,3,4])
            >>> y.split(2)
            Factor(['0', '0', '1', '1'])

            >>> z = Var([7, 6, 5, 4, 3, 2])
            >>> z.split(3)
            Factor(['2', '2', '1', '1', '0', '0'])

        """
        y = self.x

        d = 100. / n
        percentile = np.arange(d, 100., d)
        values = [scipy.stats.scoreatpercentile(y, p) for p in percentile]
        x = np.zeros(len(y), dtype=int)
        for v in values:
            x += y > v
        return Factor(x, name)

    def std(self):
        "Returns the standard deviation"
        return self.x.std()

    def sort_idx(self, descending=False):
        """Create an index that could be used to sort the Var.

        Parameters
        ----------
        descending : bool
            Sort in descending instead of an ascending order.
        """
        idx = np.argsort(self.x, kind='mergesort')
        if descending:
            idx = idx[::-1]
        return idx

    @property
    def values(self):
        return np.unique(self.x)


class _Effect(object):
    # numeric ---
    def __add__(self, other):
        return Model(self) + other

    def __mul__(self, other):
        if isinstance(other, Model):
            return Model((self, other, self % other))
        return Model((self, other, self % other))

    def __mod__(self, other):
        if isinstance(other, Model):
            return Model((self % e for e in other.effects))
        return Interaction((self, other))

    def count(self, value, start=-1):
        """Cumulative count of the occurrences of ``value``

        Parameters
        ----------
        value : str | tuple  (value in .cells)
            Cell value which is to be counted.
        start : int
            Value at which to start counting (with the default of -1, the first
            occurrence will be 0).

        Returns
        -------
        count : Var of int,  len = len(self)
            Cumulative count of value in self.

        Examples
        --------
        >>> a = Factor('abc', tile=3)
        >>> a
        Factor(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
        >>> a.count('a')
        array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        """
        return Var(np.cumsum(self == value) + start)

    def enumerate_cells(self, name=None):
        """Enumerate the occurrence of each cell value throughout the data

        Parameters
        ----------
        name : None | str
            Name for the returned Var.

        Returns
        -------
        enum : Var
            Result.

        Examples
        --------
        >>> f = Factor('aabbccabc')
        >>> f.enumerate_cells()
        Var([0, 1, 0, 1, 0, 1, 2, 2, 2])
        """
        counts = {cell: 0 for cell in self.cells}
        enum = np.empty(len(self), int)
        for i, value in enumerate(self):
            enum[i] = counts[value]
            counts[value] += 1
        return Var(enum, name)

    def index(self, cell):
        """``e.index(cell)`` returns an array of indices where e equals cell

        Examples
        --------
        >>> f = Factor('abcabcabc')
        >>> f
        Factor(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
        >>> f.index('b')
        array([1, 4, 7])
        >>> f[f.index('b')] = 'new_b'
        >>> f
        Factor(['a', 'new_b', 'c', 'a', 'new_b', 'c', 'a', 'new_b', 'c'])
        """
        return np.flatnonzero(self == cell)

    def index_opt(self, cell):
        """Find an optimized index for a given cell.

        Returns
        -------
        index : slice | array
            If possible, a ``slice`` object is returned. Otherwise, an array
            of indices (as with ``e.index(cell)``).
        """
        index = np.flatnonzero(self == cell)
        d_values = np.unique(np.diff(index))
        if len(d_values) == 1:
            start = index.min() or None
            step = d_values[0]
            stop = index.max() + 1
            if stop > len(self) - step:
                stop = None
            if step == 1:
                step = None
            index = slice(start, stop, step)
        return index

    def sort_idx(self, descending=False, order=None):
        """Create an index that could be used to sort this data_object.

        Parameters
        ----------
        descending : bool
            Sort in descending instead of the default ascending order.
        order : None | sequence
            Sequence of cells to define a custom order. Any cells that are not
            present in ``order`` will be omitted in the sort_index, i.e. the
            sort_index will be shorter than its source.

        Returns
        -------
        sort_index : array of int
            Array which can be used to sort a data_object in the desired order.
        """
        idx = np.empty(len(self), dtype=np.uint32)
        if order is None:
            cells = self.cells
        else:
            cells = order
            idx[:] = -1

        for i, cell in enumerate(cells):
            idx[self == cell] = i

        sort_idx = np.argsort(idx, kind='mergesort')
        if order is not None:
            i_cut = -np.count_nonzero(idx == np.uint32(-1))
            if i_cut:
                sort_idx = sort_idx[:i_cut]

        if descending:
            sort_idx = sort_idx[::-1]

        return sort_idx


class Factor(_Effect):
    """Container for categorial data.

    Parameters
    ----------
    x : iterator
        Sequence of Factor values (see also the ``labels`` kwarg).
    name : str
        Name of the Factor.
    random : bool
        Treat Factor as random factor (for ANOVA; default is False).
    repeat : int | array of int
        repeat each element in ``x``, either a constant or a different number
        for each element.
    tile : int
        Repeat ``x`` as a whole ``tile`` many times.
    labels : dict | OrderedDict | tuple
        An optional dictionary mapping values as they occur in ``x`` to the
        Factor's cell labels. Since :class`dict`s are unordered, labels are
        sorted alphabetically by default. In order to define cells in a
        different order, use a :class:`collections.OrderedDict` object or
        define labels as ``((key, value), ...)`` tuple.

    Attributes
    ----------
    .name : None | str
        The Factor's name.
    .cells : tuple of str
        Sorted names of all cells.
    .random : bool
        Whether the Factor is defined as random factor (for ANOVA).

    Examples
    --------
    The most obvious way to initialize a Factor is a list of strings::

        >>> Factor(['in', 'in', 'in', 'out', 'out', 'out'])
        Factor(['in', 'in', 'in', 'out', 'out', 'out'])

    The same can be achieved with a list of integers plus a labels dict::

        >>> Factor([1, 1, 1, 0, 0, 0], labels={1: 'in', 0: 'out'})
        Factor(['in', 'in', 'in', 'out', 'out', 'out'])

    Or more parsimoniously:

        >>> Factor([1, 0], labels={1: 'in', 0: 'out'}, repeat=3)
        Factor(['in', 'in', 'in', 'out', 'out', 'out'])

    Since the Factor initialization simply iterates over the ``x``
    argument, a Factor with one-character codes can also be initialized
    with a single string::

        >>> Factor('iiiooo')
        Factor(['i', 'i', 'i', 'o', 'o', 'o'])
    """
    _stype = "factor"

    def __init__(self, x, name=None, random=False, repeat=1, tile=1, labels={}):
        if not (np.any(repeat) or np.any(tile)):
            self.__setstate__({'x': np.empty((0,), np.uint32), 'labels': {},
                               'name': name, 'random': random})
            return

        try:
            n_cases = len(x)
        except TypeError:  # for generators:
            x = tuple(x)
            n_cases = len(x)

        # find mapping and ordered values
        if isinstance(labels, dict):
            labels_dict = labels
            values = labels.values()
            if not isinstance(labels, OrderedDict):
                values = natsorted(values)
        else:
            labels_dict = dict(labels)
            values = [pair[1] for pair in labels]

        # convert x to codes
        highest_code = -1
        codes = {}  # {label -> code}
        x_ = np.empty(n_cases, dtype=np.uint32)
        for i, value in enumerate(x):
            if value in labels_dict:
                label = labels_dict[value]
            elif isinstance(value, unicode):
                label = value
            else:
                label = str(value)

            if label in codes:
                x_[i] = codes[label]
            else:  # new code
                x_[i] = codes[label] = highest_code = highest_code + 1

        if highest_code >= 2**32:
            raise RuntimeError("Too many categories in this Factor")

        # collect ordered_labels
        ordered_labels = OrderedDict()
        for label in values:
            if label in codes:
                ordered_labels[codes[label]] = label
        for label in natsorted(codes):
            if label not in values:
                ordered_labels[codes[label]] = label

        if not (isinstance(repeat, int) and repeat == 1):
            x_ = x_.repeat(repeat)

        if tile > 1:
            x_ = np.tile(x_, tile)

        self.__setstate__({'x': x_, 'ordered_labels': ordered_labels,
                           'name': name, 'random': random})

    def __setstate__(self, state):
        self.x = x = state['x']
        self.name = state['name']
        self.random = state['random']
        if 'ordered_labels' in state:
            # 0.13:  ordered_labels replaced labels
            self._labels = state['ordered_labels']
            self._codes = {lbl: code for code, lbl in self._labels.iteritems()}
        else:
            labels = state['labels']
            cells = natsorted(labels.values())
            self._codes = codes = {lbl: code for code, lbl in labels.iteritems()}
            self._labels = OrderedDict([(codes[label], label) for label in cells])

        self._n_cases = len(x)

    def __getstate__(self):
        state = {'x': self.x,
                 'name': self.name,
                 'random': self.random,
                 'ordered_labels': self._labels}
        return state

    def __repr__(self, full=False):
        use_labels = preferences['factor_repr_use_labels']
        n_cases = preferences['factor_repr_n_cases']

        if use_labels:
            values = self.as_labels()
        else:
            values = self.x.tolist()

        if full or len(self.x) <= n_cases:
            x = repr(values)
        else:
            x = [repr(v) for v in values[:n_cases]]
            x.append('<... N=%s>' % len(self.x))
            x = '[' + ', '.join(x) + ']'

        args = [x]

        if self.name is not None:
            args.append('name=%r' % self.name)

        if self.random:
            args.append('random=True')

        if not use_labels:
            args.append('labels=%s' % self._labels)

        return 'Factor(%s)' % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, index):
        """
        sub needs to be int or an array of bools of shape(self.x)
        this method is valid for factors and nonbasic effects

        """
        if isvar(index):
            index = index.x

        x = self.x[index]
        if np.iterable(x):
            return Factor(x, self.name, self.random, labels=self._labels)
        else:
            return self._labels[x]

    def __setitem__(self, index, x):
        # convert x to code
        if isinstance(x, basestring):
            code = self._get_code(x)
        elif np.iterable(x):
            code = np.empty(len(x), dtype=np.uint16)
            for i, v in enumerate(x):
                code[i] = self._get_code(v)

        # assign
        self.x[index] = code

        # obliterate redundant labels
        codes_in_use = set(np.unique(self.x))
        rm = set(self._labels) - codes_in_use
        for code in rm:
            label = self._labels.pop(code)
            del self._codes[label]

    def _get_code(self, label):
        "add the label if it does not exists and return its code"
        try:
            return self._codes[label]
        except KeyError:
            code = 0
            while code in self._labels:
                code += 1

            if code >= 2**32:
                raise ValueError("Too many categories in this Factor.")

            self._labels[code] = label
            self._codes[label] = code
            return code

    def __iter__(self):
        return (self._labels[i] for i in self.x)

    def __contains__(self, value):
        try:
            code = self._codes[value]
        except KeyError:
            return False
        return code in self.x

    # numeric ---
    def __eq__(self, other):
        return self.x == self._encode(other)

    def __ne__(self, other):
        return self.x != self._encode(other)

    def _encode(self, x):
        if isinstance(x, basestring):
            return self._encode_1(x)
        else:
            return self._encode_seq(x)

    def _encode_1(self, value):
        return self._codes.get(value, -1)

    def _encode_seq(self, values):
        return np.array([self._codes.get(value, -1) for value in values])

    def __call__(self, other):
        """
        Create a nested effect. A factor A is nested in another factor B if
        each level of A only occurs together with one level of B.

        """
        return NestedEffect(self, other)

    def _interpret_y(self, Y, create=False):
        """
        Parameters
        ----------
        Y : str | list of str
            String(s) to be converted to code values.

        Returns
        -------
        codes : int | list of int
            List of values (codes) corresponding to the categories.

        """
        if isinstance(Y, basestring):
            if Y in self._codes:
                return self._codes[Y]
            elif create:
                code = 0
                while code in self._labels:
                    code += 1
                if code >= 65535:
                    raise ValueError("Too many categories in this Factor.")
                self._labels[code] = Y
                self._codes[Y] = code
                return code
            else:
                return 65535  # code for values not present in the Factor
        elif np.iterable(Y):
            out = np.empty(len(Y), dtype=np.uint16)
            for i, y in enumerate(Y):
                out[i] = self._interpret_y(y, create=create)
            return out
        elif Y in self._labels:
            return Y
        else:
            raise ValueError("unknown cell: %r" % Y)

    @property
    def as_dummy(self):  # x_dummy_coded
        shape = (self._n_cases, self.df)
        codes = np.empty(shape, dtype=np.int8)
        for i, cell in enumerate(self.cells[:-1]):
            codes[:, i] = (self == cell)

        return codes

    @property
    def as_dummy_complete(self):
        x = self.x[:, None]
        categories = np.unique(x)
        codes = np.hstack([x == cat for cat in categories])
        return codes.astype(np.int8)

    @property
    def as_effects(self):  # x_deviation_coded
        shape = (self._n_cases, self.df)
        codes = np.empty(shape, dtype=np.int8)
        for i, cell in enumerate(self.cells[:-1]):
            codes[:, i] = (self == cell)

        contrast = (self == self.cells[-1])
        codes -= contrast[:, None]
        return codes

    def _coefficient_names(self, method):
        if method == 'dummy':
            return ["%s:%s" % (self.name, cell) for cell in self.cells[:-1]]
        contrast_cell = self.cells[-1]
        return ["%s:%s-%s" % (self.name, cell, contrast_cell)
                for cell in self.cells[:-1]]

    def as_labels(self):
        "Convert the Factor to a list of str"
        return [self._labels[v] for v in self.x]

    def as_var(self, labels, default=None, name=None):
        """Convert the Factor into a Var

        Parameters
        ----------
        labels : dict
            A ``{factor_value: var_value}`` mapping.
        default : None | scalar
            Default value for factor values not mentioned in ``labels``. If not
            specified, factor values missing from ``labels`` will raise a
            ``KeyError``.
        name : None | True | str
            Name of the output Var, ``True`` to keep the current name (default
            ``None``).
        """
        if default is None:
            x = [labels[v] for v in self]
        else:
            x = [labels.get(v, default) for v in self]

        if name is True:
            name = self.name

        return Var(x, name)

    @property
    def beta_labels(self):
        cells = self.cells
        txt = '{0}=={1}'
        return [txt.format(cells[i], cells[-1]) for i in range(len(cells) - 1)]

    @property
    def cells(self):
        return tuple(self._labels.values())

    def _cellsize(self):
        "-1 if cell size is not equal"
        codes = self._labels.keys()
        buf = self.x == codes[0]
        n = buf.sum()
        for code in codes[1:]:
            n_ = np.equal(self.x, code, buf).sum()
            if n_ != n:
                return -1
        return n

    def aggregate(self, X, name=True):
        """
        Summarize the Factor by collapsing within cells in `X`.

        Raises an error if there are cells that contain more than one value.

        Parameters
        ----------
        X : categorial
            A categorial model defining cells to collapse.
        name : None | True | str
            Name of the output Factor, ``True`` to keep the current name
            (default ``True``).

        Returns
        -------
        f : Factor
            A copy of self with only one value for each cell in X
        """
        if len(X) != len(self):
            err = "Length mismatch: %i (Var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            idx = (X == cell)
            if np.sum(idx):
                x_i = np.unique(self.x[idx])
                if len(x_i) > 1:
                    labels = tuple(self._labels[code] for code in x_i)
                    err = ("Can not determine aggregated value for Factor %r "
                           "in cell %r because the cell contains multiple "
                           "values %r. Set drop_bad=True in order to ignore "
                           "this inconsistency and drop the Factor."
                           % (self.name, cell, labels))
                    raise ValueError(err)
                else:
                    x.append(x_i[0])

        if name is True:
            name = self.name

        out = Factor(x, name, self.random, labels=self._labels)
        return out

    def copy(self, name=True, repeat=1, tile=1):
        "returns a deep copy of itself"
        if name is True:
            name = self.name
        return Factor(self.x.copy(), name, self.random, repeat, tile, self._labels)

    @property
    def df(self):
        return max(0, len(self._labels) - 1)

    def endswith(self, substr):
        """Create an index that is true for all cases whose name ends with
        ``substr``

        Parameters
        ----------
        substr : str
            String for selecting cells that end with substr.

        Returns
        -------
        idx : boolean array,  len = len(self)
            Index that is true wherever the value ends with ``substr``.

        Examples
        --------
        >>> a = Factor(['a1', 'a2', 'b1', 'b2'])
        >>> a.endswith('1')
        array([True, False,  True,  False], dtype=bool)
        """
        values = [v for v in self.cells if v.endswith(substr)]
        return self.isin(values)

    def floodfill(self, regions, empty=''):
        """Fill in empty regions in a Factor from the nearest non-empty value

        Parameters
        ----------
        regions : array_like
            Object with same length as factor that indicates regions to fill.
        empty : str
            Value that is to be treated as empty (default is '').

        Examples
        --------
        >>> f = Factor(['', '', 'a', '', '', '', 'b', ''])
        >>> f.floodfill([1, 1, 1, 1, 2, 2, 2, 2])
        Factor(['a', 'a', 'a', 'a', 'b', 'b', 'b', 'b'])
        >>> f.floodfill([1, 1, 1, 2, 2, 2, 2, 2])
        Factor(['a', 'a', 'a', 'b', 'b', 'b', 'b', 'b'])
        >>> f.floodfill([1, 1, 1, 1, 1, 1, 1, 1])
        Factor(['a', 'a', 'a', 'a', 'a', 'a', 'b', 'b'])
        """
        assert(len(regions) == self._n_cases)
        out = self.copy(None)
        if empty not in self._codes:
            return out
        x = out.x

        empty = out._codes[empty]
        i_region_start = 0
        region_v = -1 if regions[0] is None else None
        fill_with = empty
        for i in xrange(self._n_cases):
            if regions[i] == region_v:
                if x[i] == empty:
                    if fill_with != empty:
                        x[i] = fill_with
                else:
                    if fill_with == empty:
                        x[i_region_start:i] = x[i]
                    fill_with = x[i]
            else:  # region change
                region_v = regions[i]
                fill_with = x[i]
                if fill_with == empty:
                    i_region_start = i

        # remove redundant label
        if empty not in x:
            del out._codes[out._labels.pop(empty)]

        return out

    def get_index_to_match(self, other):
        """
        Assuming that ``other`` is a shuffled version of self, this method
        returns ``index`` to transform from the order of self to the order of
        ``other``. To guarantee exact matching, each value can only occur once
        in self.

        Example::

            >>> index = factor1.get_index_to_match(factor2)
            >>> all(factor1[index] == factor2)
            True

        """
        assert self._labels == other._labels
        index = []
        for v in other.x:
            where = np.where(self.x == v)[0]
            if len(where) == 1:
                index.append(where[0])
            else:
                msg = "%r contains several cases of %r" % (self, v)
                raise ValueError(msg)
        return np.array(index)

    def isany(self, *values):
        """Find the index of entries matching one of the ``*values``

        Returns
        -------
        index : array of bool
            For each case True if the value is in values, else False.

        Examples
        --------
        >>> a = Factor('aabbcc')
        >>> b.isany('b', 'c')
        array([False, False,  True,  True,  True,  True], dtype=bool)
        """
        return self.isin(values)

    def isin(self, values):
        """Find the index of entries matching one of the ``values``

        Returns
        -------
        index : array of bool
            For each case True if the value is in values, else False.

        Examples
        --------
        >>> f = Factor('aabbcc')
        >>> f.isin(('b', 'c'))
        array([False, False,  True,  True,  True,  True], dtype=bool)
        """
        return np.in1d(self.x, self._encode_seq(values))

    def isnot(self, *values):
        """Find the index of entries not in ``values``

        Returns
        -------
        index : array of bool
            For each case False if the value is in values, else True.
        """
        return self.isnotin(values)

    def isnotin(self, values):
        """Find the index of entries not in ``values``

        Returns
        -------
        index : array of bool
            For each case False if the value is in values, else True.
        """
        return np.in1d(self.x, self._encode_seq(values), invert=True)

    def label_length(self, name=None):
        """Create Var with the length of each label string

        Parameters
        ----------
        name : str
            Name of the output Var (default ``None``).

        Examples
        --------
        >>> f = Factor(['a', 'ab', 'long_label'])
        >>> f.label_length()
        Var([1, 2, 10])
        """
        label_lengths = {code: len(label) for code, label in self._labels.iteritems()}
        x = np.empty(len(self))
        for i, code in enumerate(self.x):
            x[i] = label_lengths[code]

        if name:
            longname = name
        elif self.name:
            longname = self.name + '.label_length()'
        else:
            longname = 'label_length'

        return Var(x, name, info={"longname": longname})

    @property
    def n_cells(self):
        return len(self._labels)

    def update_labels(self, labels):
        """Change one or more labels in place

        Parameters
        ----------
        labels : dict
            Mapping from old labels to new labels. Existing labels that are not
            in ``labels`` are kept.

        Examples
        --------
        >>> f = Factor('aaabbbccc')
        >>> f
        Factor(['a', 'a', 'a', 'b', 'b', 'b', 'c', 'c', 'c'])
        >>> f.update_labels({'a': 'v1', 'b': 'v2'})
        >>> f
        Factor(['v1', 'v1', 'v1', 'v2', 'v2', 'v2', 'c', 'c', 'c'])

        In order to create a copy of the Factor with different labels just
        use the labels argument when initializing a new Factor:

        >>> Factor(f, labels={'c': 'v3'})
        Factor(['v1', 'v1', 'v1', 'v2', 'v2', 'v2', 'v3', 'v3', 'v3'])

        Notes
        -----
        If ``labels`` contains a key that is not a label of the Factor, a
        ``KeyError`` is raised.
        """
        missing = tuple(old for old in labels if old not in self._codes)
        if missing:
            if len(missing) == 1:
                msg = ("Factor does not contain label %r" % missing[0])
            else:
                msg = ("Factor does not contain labels %s"
                       % ', '.join(repr(m) for m in missing))
            raise KeyError(msg)

        # check for merged labels
        new_labels = {c: labels.get(l, l) for c, l in self._labels.iteritems()}
        codes_ = sorted(new_labels)
        labels_ = tuple(new_labels[c] for c in codes_)
        for i, label in enumerate(labels_):
            if label in labels_[:i]:
                old_code = codes_[i]
                new_code = codes_[labels_.index(label)]
                self.x[self.x == old_code] = new_code
                del new_labels[old_code]

        self._labels = new_labels
        self._codes = {l: c for c, l in new_labels.iteritems()}

    def startswith(self, substr):
        """Create an index that is true for all cases whose name starts with
        ``substr``

        Parameters
        ----------
        substr : str
            String for selecting cells that start with substr.

        Returns
        -------
        idx : boolean array,  len = len(self)
            Index that is true wherever the value starts with ``substr``.

        Examples
        --------
        >>> a = Factor(['a1', 'a2', 'b1', 'b2'])
        >>> a.startswith('b')
        array([False, False,  True,  True], dtype=bool)
        """
        values = [v for v in self.cells if v.startswith(substr)]
        return self.isin(values)

    def table_categories(self):
        "returns a table containing information about categories"
        table = fmtxt.Table('rll')
        table.title(self.name)
        for title in ['i', 'Label', 'n']:
            table.cell(title)
        table.midrule()
        for code, label in self._labels.iteritems():
            table.cell(code)
            table.cell(label)
            table.cell(np.sum(self.x == code))
        return table

    def repeat(self, repeats, name=True):
        """
        Repeat each element ``repeats`` times

        Parameters
        ----------
        repeats : int | array of int
            Number of repeats, either a constant or a different number for each
            element.
        name : None | True | str
            Name of the output Var, ``True`` to keep the current name (default
            ``True``).
        """
        if name is True:
            name = self.name
        return Factor(self.x, name, self.random, repeats, labels=self._labels)



class NDVar(object):
    """Container for n-dimensional data.
    
    Parameters
    ----------
    x : array_like
        The data.
    dims : tuple
        The dimensions characterizing the axes of the data. If present,
        ``'case'`` should be provided as a :py:class:`str`, and should
        always occupy the first position.
    info : dict
        A dictionary with data properties (can contain arbitrary
        information that will be accessible in the info attribute).
    name : None | str
        Name for the NDVar.


    Notes
    -----
    ``x`` and ``dims`` are stored without copying. A shallow
    copy of ``info`` is stored. Make sure the relevant objects
    are not modified externally later.


    Examples
    --------
    Importing 600 epochs of data for 80 time points:

    >>> data.shape
    (600, 80)
    >>> time = UTS(-.2, .01, 80)
    >>> dims = ('case', time)
    >>> Y = NDVar(data, dims=dims)

    """
    _stype = "ndvar"
    
    def __init__(self, x, dims=('case',), info={}, name=None):
        # check data shape
        dims = tuple(dims)
        ndim = len(dims)
        x = np.asarray(x)
        if ndim != x.ndim:
            err = ("Unequal number of dimensions (data: %i, dims: %i)" %
                   (x.ndim, ndim))
            raise DimensionMismatchError(err)

        # check dimensions
        d0 = dims[0]
        if isinstance(d0, basestring):
            if d0 == 'case':
                has_case = True
            else:
                err = ("The only dimension that can be specified as a string"
                       "is 'case' (got %r)" % d0)
                raise ValueError(err)
        else:
            has_case = False

        for dim, n in zip(dims, x.shape)[has_case:]:
            if isinstance(dim, basestring):
                err = ("Invalid dimension: %r in %r. First dimension can be "
                       "'case', other dimensions need to be Dimension "
                       "subclasses." % (dim, dims))
                raise TypeError(err)
            n_dim = len(dim)
            if n_dim != n:
                err = ("Dimension %r length mismatch: %i in data, "
                       "%i in dimension %r" % (dim.name, n, n_dim, dim.name))
                raise DimensionMismatchError(err)

        state = {'x': x, 'dims': dims, 'info': dict(info),
                 'name': name}
        self.__setstate__(state)

    def __setstate__(self, state):
        self.dims = dims = state['dims']
        self.has_case = (dims[0] == 'case')
        self._truedims = truedims = dims[self.has_case:]

        # dimnames
        self.dimnames = tuple(dim.name for dim in truedims)
        if self.has_case:
            self.dimnames = ('case',) + self.dimnames

        self.x = x = state['x']
        self.name = state['name']
        if 'info' in state:
            self.info = state['info']
        else:
            self.info = state['properties']
        # derived
        self.ndim = len(dims)
        self.shape = x.shape
        self._len = len(x)
        self._dim_2_ax = dict(zip(self.dimnames, xrange(self.ndim)))
        # attr
        for dim in truedims:
            if hasattr(self, dim.name):
                err = ("invalid dimension name: %r (already present as NDVar"
                       " attr)" % dim.name)
                raise ValueError(err)
            else:
                setattr(self, dim.name, dim)

    def __getstate__(self):
        state = {'dims': self.dims,
                 'x': self.x,
                 'name': self.name,
                 'info': self.info}
        return state

    @property
    def __array_interface__(self):
        return self.x.__array_interface__

    # numeric ---
    def __neg__(self):
        x = -self.x
        info = self.info.copy()
        return NDVar(x, self.dims, info)

    def __pos__(self):
        return self

    def __abs__(self):
        return self.abs()

    def __lt__(self, other):
        y = self._ialign(other)
        x = self.x < y
        return NDVar(x, self.dims, self.info.copy())

    def __le__(self, other):
        y = self._ialign(other)
        x = self.x <= y
        return NDVar(x, self.dims, self.info.copy())

    def __eq__(self, other):
        y = self._ialign(other)
        x = self.x == y
        return NDVar(x, self.dims, self.info.copy())

    def __ne__(self, other):
        y = self._ialign(other)
        x = self.x != y
        return NDVar(x, self.dims, self.info.copy())

    def __gt__(self, other):
        y = self._ialign(other)
        x = self.x > y
        return NDVar(x, self.dims, self.info.copy())

    def __ge__(self, other):
        y = self._ialign(other)
        x = self.x >= y
        return NDVar(x, self.dims, self.info.copy())

    def _align(self, other):
        """Align data from 2 NDVars.

        Notes
        -----
        For unequal but overlapping dimensions, the intersection is used.
        """
        if isvar(other):
            return self.dims, self.x, self._ialign(other)
        elif isndvar(other):
            dimnames = list(self.dimnames)
            i_add = 0
            for dimname in other.dimnames:
                if dimname not in dimnames:
                    dimnames.append(dimname)
                    i_add += 1

            # find data axes
            self_axes = self.dimnames
            if i_add:
                self_axes += (None,) * i_add
            other_axes = tuple(name if name in other.dimnames else None
                               for name in dimnames)

            # find dims
            dims = []
            crop = False
            crop_self = []
            crop_other = []
            for name, other_name in izip(self_axes, other_axes):
                if name is None:
                    dim = other.get_dim(other_name)
                    cs = co = full_slice
                elif other_name is None:
                    dim = self.get_dim(name)
                    cs = co = full_slice
                else:
                    self_dim = self.get_dim(name)
                    other_dim = other.get_dim(other_name)
                    if self_dim == other_dim:
                        dim = self_dim
                        cs = co = full_slice
                    else:
                        dim = self_dim.intersect(other_dim)
                        crop = True
                        cs = self_dim.dimindex(dim)
                        os = other_dim.dimindex(dim)
                dims.append(dim)
                crop_self.append(cs)
                crop_other.append(co)

            x_self = self.get_data(self_axes)
            x_other = other.get_data(other_axes)
            if crop:
                x_self = x_self[tuple(crop_self)]
                x_other = x_other[tuple(crop_other)]
            return dims, x_self, x_other
        elif np.isscalar(other):
            return self.dims, self.x, other
        else:
            raise TypeError("Need Var or NDVar")

    def _ialign(self, other):
        "align for self-modifying operations (+=, ...)"
        if np.isscalar(other):
            return other
        elif isvar(other):
            assert self.has_case
            n = len(other)
            shape = (n,) + (1,) * (self.x.ndim - 1)
            return other.x.reshape(shape)
        elif isndvar(other):
            assert all(dim in self.dimnames for dim in other.dimnames)
            i_other = []
            for dim in self.dimnames:
                if dim in other.dimnames:
                    i_other.append(dim)
                else:
                    i_other.append(None)
            return other.get_data(i_other)
        else:
            raise TypeError

    def __add__(self, other):
        dims, x_self, x_other = self._align(other)
        x = x_self + x_other
        info = self.info.copy()
        return NDVar(x, dims, info)

    def __iadd__(self, other):
        self.x += self._ialign(other)
        return self

    def __div__(self, other):
        dims, x_self, x_other = self._align(other)
        x = x_self / x_other
        info = self.info.copy()
        return NDVar(x, dims, info)

    def __idiv__(self, other):
        self.x /= self._ialign(other)
        return self

    def __truediv__(self, other):
        return self.__div__(other)

    def __itruediv__(self, other):
        return self.__idiv__(other)

    def __mul__(self, other):
        dims, x_self, x_other = self._align(other)
        x = x_self * x_other
        info = self.info.copy()
        return NDVar(x, dims, info)

    def __imul__(self, other):
        self.x *= self._ialign(other)
        return self

    def __pow__(self, other):
        dims, x_self, x_other = self._align(other)
        x = np.power(x_self, x_other)
        info = self.info.copy()
        return NDVar(x, dims, info)

    def __ipow__(self, other):
        self.x **= self._ialign(other)
        return self

    def __sub__(self, other):
        dims, x_self, x_other = self._align(other)
        x = x_self - x_other
        info = self.info.copy()
        return NDVar(x, dims, info)

    def __isub__(self, other):
        self.x -= self._ialign(other)
        return self

    def __rsub__(self, other):
        x = other - self.x
        info = self.info.copy()
        return NDVar(x, self.dims, info, self.name)

    # container ---
    def __getitem__(self, index):
        '''Options for NDVar indexing:

         - First element only: numpy-like case index (int, array).
         - All elements: 1d boolean NDVar.
        '''
        if isinstance(index, tuple):
            return self.sub(*index)
        else:
            return self.sub(index)

    def __len__(self):
        return self._len

    def __repr__(self):
        rep = '<NDVar %(name)r: %(dims)s>'
        if self.has_case:
            dims = [(self._len, 'case')]
        else:
            dims = []
        dims.extend([(len(dim), dim.name) for dim in self._truedims])

        dims = ' X '.join('%i (%s)' % fmt for fmt in dims)
        args = dict(dims=dims, name=self.name or '')
        return rep % args

    def abs(self, name=None):
        """Compute the absolute values"""
        x = np.abs(self.x)
        dims = self.dims
        info = self.info.copy()
        return NDVar(x, dims, info, name)

    def any(self, dims=(), **regions):
        """Compute presence of any value other than zero over given dimensions

        Parameters
        ----------
        dims : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute whether there are any nonzero values at all.
            An boolean NDVar with the same dimensions as the data can be used
            to find nonzero values in specific elements (if the NDVar has cases
            on a per case basis).
        *regions*
            Regions over which to aggregate. For example, to check for nonzero
            values between time=0.1 and time=0.2, use
            ``ndvar.any(time=(0.1, 0.2))``.

        Returns
        -------
        any : NDVar | Var | float
            Boolean data indicating presence of nonzero value over specified
            dimensions. Returns a Var if only the case dimension remains, and a
            float if the function collapses over all data.
        """
        return self._aggregate_over_dims(dims, regions, np.any)

    def assert_dims(self, dims):
        if self.dimnames != dims:
            err = "Dimensions of %r do not match %r" % (self, dims)
            raise DimensionMismatchError(err)

    def aggregate(self, X, func=np.mean, name=True):
        """
        Summarize data in each cell of ``X``.

        Parameters
        ----------
        X : categorial
            Categorial whose cells define which cases to aggregate.
        func : function with axis argument
            Function that is used to create a summary of the cases falling
            into each cell of X. The function needs to accept the data as
            first argument and ``axis`` as keyword-argument. Default is
            ``numpy.mean``.
        name : None | True | str
            Name of the output NDVar, ``True`` to keep the current name
            (default ``True``).

        Returns
        -------
        aggregated_ndvar : NDVar
            Returns an
        """
        if not self.has_case:
            raise DimensionMismatchError("%r has no case dimension" % self)
        if len(X) != len(self):
            err = "Length mismatch: %i (Var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            idx = (X == cell)
            if np.sum(idx):
                x_cell = self.x[idx]
                x.append(func(x_cell, axis=0))

        # update info for summary
        info = self.info.copy()
        if 'summary_info' in info:
            info.update(info.pop('summary_info'))

        if name is True:
            name = self.name

        x = np.array(x)
        out = NDVar(x, self.dims, info, name)
        return out

    def _aggregate_over_dims(self, axis, regions, func):
        if regions:
            data = self.sub(**regions)
            additional_axis = [dim for dim in regions if data.has_dim(dim)]
            if additional_axis:
                if isndvar(axis):
                    data = data.sub(axis)
                    axis = additional_axis
                elif not axis:
                    axis = additional_axis
                elif isinstance(axis, basestring):
                    axis = [axis] + additional_axis
                else:
                    axis = list(axis) + additional_axis
            return data._aggregate_over_dims(axis, None, func)
        elif not axis:
            return func(self.x)
        elif isndvar(axis):
            if axis.ndim == 1:
                dim = axis.dims[0]
                dim_axis = self.get_axis(dim.name)
                if self.get_dim(dim.name) != dim:
                    msg = "Index dimension does not match data dimension"
                    raise DimensionMismatchError(msg)
                index = (full_slice,) * dim_axis + (axis.x,)
                x = func(self.x[index], dim_axis)
                dims = (dim_ for dim_ in self.dims if not dim_ == dim)
            else:
                # if the index does not contain all dimensions, numpy indexing
                # is weird
                if self.ndim - self.has_case != axis.ndim - axis.has_case:
                    msg = ("If the index is not one dimensional, it needs to "
                           "have the same dimensions as the data.")
                    raise NotImplementedError(msg)
                dims, self_x, index = self._align(axis)
                if self.has_case:
                    if axis.has_case:
                        x = np.array([func(x_[i]) for x_, i in izip(self_x, index)])
                    else:
                        index = index[0]
                        x = np.array([func(x_[index]) for x_ in self_x])
                    return Var(x, self.name, info=self.info.copy())
                elif axis.has_case:
                    msg = ("Index with case dimension can not be applied to "
                           "data without case dimension")
                    raise IndexError(msg)
                else:
                    return func(self_x[index])
        elif isinstance(axis, basestring):
            axis = self._dim_2_ax[axis]
            x = func(self.x, axis=axis)
            dims = (self.dims[i] for i in xrange(self.ndim) if i != axis)
        else:
            axes = tuple(self._dim_2_ax[dim_name] for dim_name in axis)
            x = func(self.x, axes)
            dims = (self.dims[i] for i in xrange(self.ndim) if i not in axes)

        dims = tuple(dims)
        name = self.name
        if len(dims) == 0:
            return x
        elif dims == ('case',):
            return Var(x, name, info=self.info.copy())
        else:
            return NDVar(x, dims, self.info.copy(), name)

    def bin(self, tstep, tstart=None, tstop=None, func=None):
        """Bin the data along the time axis

        Parameters
        ----------
        tstep : scalar
            Time step between bins.
        tstart : None | scalar
            Earliest time point (default is from the beginning).
        tstop : None | scalar
            End of the data to use (default is to the end).
        func : callable
            Function to summarize data, needs axis argument (default is the
            mean)

        Returns
        -------
        binned_ndvar : NDVar
            NDVar with data binned along the time axis (i.e., each time point
            reflects one time bin).
        """
        time = self.get_dim('time')
        time_axis = self.get_axis('time')

        # summary-func
        if func is None:
            meas = self.info.get('meas', '').lower()
            if meas == 'p':
                func = np.min
            elif meas == 'f':
                func = np.max
            elif meas in ('t', 'r'):
                func = extrema
            else:
                func = np.mean

        # find time bin boundaries
        if tstart is None:
            tstart = time.tmin

        if tstop is None:
            tstop = time.tmax  # -> avoid adding 1 sample bins

        times = [tstart]
        t = tstart + tstep
        while t < tstop:
            times.append(t)
            t += tstep
        times.append(min(t, time.tstop))

        n_bins = len(times) - 1
        out_shape = list(self.shape)
        out_shape[time_axis] = n_bins
        x = np.empty(out_shape)
        bins = []
        idx_prefix = (full_slice,) * time_axis
        for i in xrange(n_bins):
            t0 = times[i]
            t1 = times[i + 1]
            src_idx = idx_prefix + (time.dimindex((t0, t1)),)
            dst_idx = idx_prefix + (i,)
            x[dst_idx] = func(self.x[src_idx], axis=time_axis)
            if t1 is None:
                t1 = time.tmax + time.tstep
            bins.append((t0, t1))

        out_time = UTS(tstart + tstep / 2, tstep, n_bins)
        dims = list(self.dims)
        dims[time_axis] = out_time
        info = self.info.copy()
        info['bins'] = bins
        return NDVar(x, dims, info)

    def copy(self, name=True):
        """returns an NDVar with a deep copy of its data

        Parameters
        ----------
        name : None | True | str
            Name of the output NDVar, ``True`` to keep the current name
            (default ``True``).

        Returns
        -------
        ndvar_copy : NDVar
            An copy of the ndvar with a deep copy of the data.

        Notes
        -----
        The info dictionary is still a shallow copy.
        """
        x = self.x.copy()
        info = self.info.copy()
        if name is True:
            name = self.name
        return NDVar(x, self.dims, info, name)

    def diminfo(self, str_out=False):
        """Information about the dimensions

        Parameters
        ----------
        str_out : bool
            Return a string with the information (as opposed to the default
            which is to print the information).

        Returns
        -------
        info : None | str
            If str_out is True, the dimension description as str.
        """
        ns = []
        dim_info = ["<NDVar %r" % self.name]
        if self.has_case:
            ns.append(len(self))
            dim_info.append("cases")

        for dim in self._truedims:
            ns.append(len(dim))
            dim_info.append(dim._diminfo())
        dim_info[-1] += '>'

        n_digits = int(max(ceil(log10(n)) for n in ns))

        info = '\n '.join('{:{}d} {:s}'.format(n, n_digits, desc) for n, desc
                          in izip(ns, dim_info))
        if str_out:
            return info
        else:
            print info

    def get_axis(self, name):
        if self.has_dim(name):
            i = self._dim_2_ax[name]
        else:
            msg = "%r has no dimension named %r" % (self, name)
            raise DimensionMismatchError(msg)
        return i

    def get_data(self, dims):
        """Retrieve the NDVar's data with a specific axes order.

        Parameters
        ----------
        dims : str | sequence of str
            Sequence of dimension names (or single dimension name). The array
            that is returned will have axes in this order. To insert a new
            axis with size 1 use ``numpy.newaxis``/``None``.

        Notes
        -----
        A shallow copy of the data is returned. To retrieve the data with the
        stored axes order use the .x attribute.
        """
        if isinstance(dims, str):
            dims = (dims,)

        dims_ = tuple(d for d in dims if d is not np.newaxis)
        if set(dims_) != set(self.dimnames) or len(dims_) != len(self.dimnames):
            err = "Requested dimensions %r from %r" % (dims, self)
            raise DimensionMismatchError(err)

        # transpose
        axes = tuple(self.dimnames.index(d) for d in dims_)
        x = self.x.transpose(axes)

        # insert axes
        if len(dims) > len(dims_):
            for ax, dim in enumerate(dims):
                if dim is np.newaxis:
                    x = np.expand_dims(x, ax)

        return x

    def get_dim(self, name):
        "Returns the Dimension object named ``name``"
        i = self.get_axis(name)
        dim = self.dims[i]
        return dim

    def get_dims(self, names):
        "Returns a tuple with the requested Dimension objects"
        return tuple(self.get_dim(name) for name in names)

    def has_dim(self, name):
        return name in self._dim_2_ax

    def max(self, dims=(), **regions):
        """Compute the maximum over given dimensions

        Parameters
        ----------
        dims : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the maximum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the maximum in specific elements (if the data has a case
            dimension, the maximum is computed for each case).
        *regions*
            Regions over which to aggregate. For example, to get the maximum
            between time=0.1 and time=0.2, use ``ndvar.max(time=(0.1, 0.2))``.

        Returns
        -------
        max : NDVar | Var | float
            The maximum over specified dimensions. Returns a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.
        """
        return self._aggregate_over_dims(dims, regions, np.max)

    def mean(self, dims=(), **regions):
        """Compute the mean over given dimensions

        Parameters
        ----------
        dims : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the mean over all dimensions.
            A boolean NDVar with the same dimensions as the data can be used
            to compute the mean in specific elements (if the data has a case
            dimension, the mean is computed for each case).
        *regions*
            Regions over which to aggregate. For example, to get the mean
            between time=0.1 and time=0.2, use ``ndvar.mean(time=(0.1, 0.2))``.

        Returns
        -------
        mean : NDVar | Var | float
            The mean over specified dimensions. Returns a Var if only the case
            dimension remains, and a float if the function collapses over all
            data.
        """
        return self._aggregate_over_dims(dims, regions, np.mean)

    def min(self, dims=(), **regions):
        """Compute the minimum over given dimensions

        Parameters
        ----------
        dims : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the minimum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the minimum in specific elements (if the data has a case
            dimension, the minimum is computed for each case).
        *regions*
            Regions over which to aggregate. For example, to get the minimum
            between time=0.1 and time=0.2, use ``ndvar.min(time=(0.1, 0.2))``.

        Returns
        -------
        min : NDVar | Var | float
            The minimum over specified dimensions. Returns a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.
        """
        return self._aggregate_over_dims(dims, regions, np.min)

    def ols(self, x, name=None):
        """Sample-wise ordinary least squares regressions

        Parameters
        ----------
        x : Model
            Predictor or predictors. Can also be supplied as argument that can
            be converted to a Model, for example ``Var`` or list of ``Var``.
        name : str
            Name for the output NDVar.

        Returns
        -------
        beta : NDVar
            Per sample beta weights. The case dimension reflects the predictor
            variables in the same order as the Model's effects.

        Notes
        -----
        The intercept is generated internally, and betas for the intercept are
        not returned.

        See Also
        --------
        .ols_t : T-values for regression coefficients
        """
        from ._stats import stats

        if not self.has_case:
            msg = ("Can only apply regression to NDVar with case dimension")
            raise DimensionMismatchError(msg)

        x = asmodel(x)
        if len(x) != len(self):
            msg = ("Predictors do not have same number of cases (%i) as the "
                   "dependent variable (%i)" % (len(x), len(self)))
            raise DimensionMismatchError(msg)

        betas = stats.betas(self.x, x)[1:]  # drop intercept
        info = self.info.copy()
        info.update(meas='beta', unit=None)
        if 'summary_info' in info:
            del info['summary_info']
        return NDVar(betas, self.dims, info, name)

    def ols_t(self, x, name=None):
        """T-values for sample-wise ordinary least squares regressions

        Parameters
        ----------
        x : Model
            Predictor or predictors. Can also be supplied as argument that can
            be converted to a Model, for example ``Var`` or list of ``Var``.
        name : str
            Name for the output NDVar.

        Returns
        -------
        t : NDVar
            Per sample t-values. The case dimension reflects the predictor
            variables in the same order as the Model's effects.

        Notes
        -----
        Betas for the intercept are not returned.

        See Also
        --------
        .ols : Regression coefficients
        """
        from ._stats import stats

        if not self.has_case:
            msg = ("Can only apply regression to NDVar with case dimension")
            raise DimensionMismatchError(msg)

        x = asmodel(x)
        if len(x) != len(self):
            msg = ("Predictors do not have same number of cases (%i) as the "
                   "dependent variable (%i)" % (len(x), len(self)))
            raise DimensionMismatchError(msg)

        t = stats.lm_t(self.x, x)[1:]  # drop intercept
        info = self.info.copy()
        return NDVar(t, self.dims, info, name)

    def repeat(self, repeats, dim='case', name=True):
        """
        Analogous to :py:func:`numpy.repeat`

        Parameters
        ----------
        repeats : int | array of ints
            The number of repetitions for each element. `repeats` is
            broadcasted to fit the shape of the given dimension.
        dim : str
            The dimension along which to repeat values (default 'case').
        name : None | True | str
            Name of the output NDVar, ``True`` to keep the current name
            (default ``True``).
        """
        ax = self.get_axis(dim)
        x = self.x.repeat(repeats, axis=ax)

        repdim = self.dims[ax]
        if not isinstance(repdim, str):
            repdim = repdim.repeat(repeats)

        dims = self.dims[:ax] + (repdim,) + self.dims[ax + 1:]
        info = self.info.copy()
        if name is True:
            name = self.name
        return NDVar(x, dims, info, name)

    def residuals(self, x, name=None):
        """
        Residuals of sample-wise ordinary least squares regressions

        Parameters
        ----------
        x : Model
            Predictor or predictors. Can also be supplied as argument that can
            be converted to a Model, for example ``Var`` or list of ``Var``.
        name : str
            Name for the output NDVar.

        Returns
        -------
        residuals : NDVar
            Residual for each case and sample (same dimensions as data).
        """
        if not self.has_case:
            msg = ("Can only apply regression to NDVar with case dimension")
            raise DimensionMismatchError(msg)

        x = asmodel(x)
        if len(x) != len(self):
            msg = ("Predictors do not have same number of cases (%i) as the "
                   "dependent variable (%i)" % (len(x), len(self)))
            raise DimensionMismatchError(msg)

        from ._stats import stats
        res = stats.residuals(self.x, x)
        info = self.info.copy()
        return NDVar(res, self.dims, info, name)

    def rms(self, axis=(), **regions):
        """Compute the root mean square over given dimensions

        Parameters
        ----------
        axis : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the standard deviation over all values.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the RMS in specific elements (if the data has a case
            dimension, the RMS is computed for each case).
        *regions*
            Regions over which to aggregate. For example, to get the RMS
            between time=0.1 and time=0.2, use ``ndvar.rms(time=(0.1, 0.2))``.

        Returns
        -------
        rms : NDVar | Var | float
            The root mean square over specified dimensions. Returns a Var if
            only the case dimension remains, and a float if the function
            collapses over all data.
        """
        from ._stats.stats import rms
        return self._aggregate_over_dims(axis, regions, rms)

    def smooth(self, dim, fwhm):
        """Smooth the ndvar over the given dimension

        Parameters
        ----------
        dim : str
            Name of the dimension over which to smooth.
        fwhm : scalar
            The full width at half maximum.
        """
        dim_ = self.get_dim(dim)
        m = gaussian_smoother(dim_.distances(), fwhm)
        ax = self.get_axis(dim)
        x = np.tensordot(m, self.x, (1, ax))
        if ax:
            x = x.swapaxes(0, ax)
        return NDVar(x, self.dims, self.info.copy(), None)

    def std(self, dims=(), **regions):
        """Compute the standard deviation over given dimensions

        Parameters
        ----------
        dims : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the standard deviation over all values.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the standard deviation in specific elements (if the data
            has a case dimension, the standard deviation is computed for each
            case).
        *regions*
            Regions over which to aggregate. For example, to get the STD
            between time=0.1 and time=0.2, use ``ndvar.std(time=(0.1, 0.2))``.

        Returns
        -------
        std : NDVar | Var | float
            The standard deviation over specified dimensions. Returns a Var if
            only the case dimension remains, and a float if the function
            collapses over all data.
        """
        return self._aggregate_over_dims(dims, regions, np.std)

    def summary(self, *dims, **regions):
        r"""
        Returns a new NDVar with specified dimensions collapsed.

        .. warning::
            Data is collapsed over the different dimensions in turn using the
            provided function with an axis argument. For certain functions
            this is not equivalent to collapsing over several axes concurrently
            (e.g., np.var).

        dimension:
            A whole dimension is specified as string argument. This
            dimension is collapsed over the whole range.
        range:
            A range within a dimension is specified through a keyword-argument.
            Only the data in the specified range is included. Use like the
            :py:meth:`.sub` method.


        **additional kwargs:**

        func : callable
            Function used to collapse the data. Needs to accept an "axis"
            kwarg (default: np.mean)
        name : str
            Name for the new NDVar.


        Examples
        --------

        Assuming ``data`` is a normal time series. Get the average in a time
        window::

            >>> Y = data.summary(time=(.1, .2))

        Get the peak in a time window::

            >>> Y = data.summary(time=(.1, .2), func=np.max)

        Assuming ``meg`` is an NDVar with dimensions time and sensor. Get the
        average across sensors 5, 6, and 8 in a time window::

            >>> roi = [5, 6, 8]
            >>> Y = meg.summary(sensor=roi, time=(.1, .2))

        Get the peak in the same data:

            >>> roi = [5, 6, 8]
            >>> peak = meg.summary(sensor=roi, time=(.1, .2), func=np.max)

        Get the RMS over all sensors

            >>> meg_rms = meg.summary('sensor', func=rms)

        """
        if 'func' in regions:
            func = regions.pop('func')
        elif 'summary_func' in self.info:
            func = self.info['summary_func']
        else:
            func = np.mean
        name = regions.pop('name', None)
        if len(dims) + len(regions) == 0:
            dims = ('case',)

        if regions:
            dims = list(dims)
            data = self.sub(**regions)
            dims.extend(dim for dim in regions if data.has_dim(dim))
            return data.summary(*dims, func=func, name=name)
        else:
            x = self.x
            axes = [self._dim_2_ax[dim] for dim in dims]
            dims = list(self.dims)
            for axis in sorted(axes, reverse=True):
                x = func(x, axis=axis)
                dims.pop(axis)

            # update info for summary
            info = self.info.copy()
            if 'summary_info' in info:
                info.update(info.pop('summary_info'))

            if len(dims) == 0:
                return x
            elif dims == ['case']:
                return Var(x, name, info=info)
            else:
                return NDVar(x, dims, info, name)

    def sub(self, *args, **kwargs):
        """Retrieve a slice through the NDVar.

        Returns an NDVar object with a slice of the current NDVar's data.
        The slice is specified using arguments and keyword arguments. Indexes
        for dimensions can ether be specified in order, or with dimension names
        as keywords, e.g.::

            >>> Y.sub(time = 1)

        returns a slice for time point 1 (second). If time is the first
        dimension, this is equivalent::

            >>> Y.sub(1)

        For dimensions whose values change monotonically, a tuple can be used
        to specify a window::

            >>> Y.sub(time = (.2, .6))

        returns a slice containing all values for times .2 seconds to .6
        seconds.

        The name of the new NDVar can be set with a ``name`` parameter. The
        default is the name of the current NDVar.
        """
        var_name = kwargs.pop('name', self.name)
        info = self.info.copy()
        dims = list(self.dims)
        n_axes = len(dims)
        index = [full_slice] * n_axes
        index_args = [None] * n_axes

        # sequence args
        for i, arg in enumerate(args):
            if isndvar(arg):
                if arg.has_case:
                    raise ValueError("NDVar with case dimension can not serve"
                                     "as NDVar index")
                dimax = self.get_axis(arg.dims[0].name)
                if index_args[dimax] is None:
                    index_args[dimax] = arg
                else:
                    raise IndexError("Index for %s dimension specified twice."
                                     % arg.dims[0].name)
            else:
                index_args[i] = arg

        # sequence kwargs
        for dimname, arg in kwargs.iteritems():
            dimax = self.get_axis(dimname)
            if index_args[dimax] is None:
                index_args[dimax] = arg
            else:
                raise RuntimeError("Index for %s dimension specified twice." % dimname)

        # process indexes
        for dimax, idx in enumerate(index_args):
            if idx is None:
                continue
            dim = self.dims[dimax]

            # find index
            if dimax >= self.has_case:
                idx = dim.dimindex(idx)
            else:
                idx = dimindex_case(idx)
            index[dimax] = idx

            # find corresponding dim
            if np.isscalar(idx):
                dims[dimax] = None
            elif dimax >= self.has_case:
                dims[dimax] = dim[idx]
            else:
                dims[dimax] = dim

        # adjust index dimension
        if sum(isinstance(idx, np.ndarray) for idx in index) > 1:
            ndim_increment = 0
            for i in xrange(n_axes - 1, -1, -1):
                idx = index[i]
                if ndim_increment and isinstance(idx, (slice, np.ndarray)):
                    if isinstance(idx, slice):
                        idx = slice_to_arange(idx, len(dims[i]))
                    elif idx.dtype.kind == 'b':
                        idx = np.flatnonzero(idx)
                    index[i] = idx[(full_slice,) + (None,) * ndim_increment]

                if isinstance(idx, np.ndarray):
                    ndim_increment += 1

        # create NDVar
        dims = tuple(dim for dim in dims if dim is not None)
        if dims == ('case',):
            return Var(self.x[tuple(index)], var_name, info=info)
        elif dims:
            return NDVar(self.x[tuple(index)], dims, info, var_name)
        else:
            return self.x[tuple(index)]

    def sum(self, dims=(), **regions):
        """Compute the sum over given dimensions

        Parameters
        ----------
        dims : str | tuple of str | boolean NDVar
            Dimensions over which to operate. A str is used to specify a single
            dimension, a tuple of str to specify several dimensions, None to
            compute the sum over all dimensions.
            An boolean NDVar with the same dimensions as the data can be used
            to compute the sum in specific elements (if the data has a case
            dimension, the sum is computed for each case).
        *regions*
            Regions over which to aggregate. For example, to get the sum
            between time=0.1 and time=0.2, use ``ndvar.sum(time=(0.1, 0.2))``.

        Returns
        -------
        sum : NDVar | Var | float
            The sum over specified dimensions. Returns a Var if only the
            case dimension remains, and a float if the function collapses over
            all data.
        """
        return self._aggregate_over_dims(dims, regions, np.sum)


def extrema(x, axis=0):
    "Extract the extreme values in x"
    max = np.max(x, axis)
    min = np.min(x, axis)
    return np.where(np.abs(max) > np.abs(min), max, min)


class Datalist(list):
    """:py:class:`list` subclass for including lists in in a Dataset.

    Parameters
    ----------
    items : sequence
        Content for the Datalist.
    name : str
        Name of the Datalist.
    fmt : 'repr' | 'str' | 'strlist'
        How to format items when converting Datasets to tables (default 'repr'
        uses the normal object representation).


    Notes
    -----
    Modifications:

     - adds certain methods that makes indexing behavior more similar to numpy
       and other data objects
     - blocks methods for in place modifications that would change the lists's
       length


    Examples
    --------
    Concise string representation:

    >>> l = [['a', 'b'], [], ['a']]
    >>> print Datalist(l)
    [['a', 'b'], [], ['a']]
    >>> print Datalist(l, fmt='strlist')
    [[a, b], [], [a]]
    """
    _stype = 'list'
    _fmt = 'repr'  # for backwards compatibility with old pickles

    def __init__(self, items=None, name=None, fmt='repr'):
        if fmt not in ('repr', 'str', 'strlist'):
            raise ValueError("fmt=%s" % repr(fmt))

        self.name = name
        self._fmt = fmt
        if items:
            super(Datalist, self).__init__(items)
        else:
            super(Datalist, self).__init__()

    def __repr__(self):
        args = super(Datalist, self).__repr__()
        if self.name is not None:
            args += ', %s' % repr(self.name)
        if self._fmt != 'repr':
            args += ', fmt=%s' % repr(self._fmt)
        return "Datalist(%s)" % args

    def __str__(self):
        return "[%s]" % ', '.join(self._item_repr(i) for i in self)

    def _item_repr(self, item):
        if self._fmt == 'str':
            return str(item)
        elif self._fmt == 'repr':
            out = repr(item)
            if len(out) > 15:
                return out[:12] + '...'
            else:
                return out
        elif self._fmt == 'strlist':
            return "[%s]" % ', '.join(item)
        else:
            raise RuntimeError("Datalist._fmt=%s" % repr(self._fmt))

    def __eq__(self, other):
        if len(self) != len(other):
            raise ValueError("Unequal length")
        return np.array([s == o for s, o in izip(self, other)])

    def __ne__(self, other):
        if len(self) != len(other):
            raise ValueError("Unequal length")
        return np.array([s != o for s, o in izip(self, other)])

    def __getitem__(self, index):
        if isinstance(index, int):
            return list.__getitem__(self, index)
        elif isinstance(index, slice):
            return Datalist(list.__getitem__(self, index), fmt=self._fmt)

        index = np.asarray(index)
        if index.dtype.kind == 'b':
            if len(index) != len(self):
                raise ValueError("Boolean index needs to have same length as "
                                 "Datalist")
            return Datalist((self[i] for i in np.flatnonzero(index)), fmt=self._fmt)
        elif index.dtype.kind == 'i':
            return Datalist((self[i] for i in index), fmt=self._fmt)
        else:
            err = ("Unsupported type of index for Datalist: %r" % index)
            raise TypeError(err)

    def __getslice__(self, i, j):
        return Datalist(list.__getslice__(self, i, j), fmt=self._fmt)

    def __add__(self, other):
        return Datalist(super(Datalist, self).__add__(other), fmt=self._fmt)

    def aggregate(self, X, merge='mean'):
        """
        Summarize cases for each cell in X

        Parameters
        ----------
        X : categorial
            Cells which to aggregate.
        merge : str
            How to merge entries.
            ``'mean'``: sum elements and dividie by cell length
        """
        if len(X) != len(self):
            err = "Length mismatch: %i (Var) != %i (X)" % (len(self), len(X))
            raise ValueError(err)

        x = []
        for cell in X.cells:
            x_cell = self[X == cell]
            n = len(x_cell)
            if n == 1:
                x.append(x_cell)
            elif n > 1:
                if merge == 'mean':
                    xc = reduce(lambda x, y: x + y, x_cell)
                    xc /= n
                else:
                    raise ValueError("Invalid value for merge: %r" % merge)
                x.append(xc)

        return Datalist(x, fmt=self._fmt)

    def __iadd__(self, other):
        return self + other

    def append(self, p_object):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def extend(self, iterable):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def insert(self, index, p_object):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def pop(self, index=None):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def remove(self, value):
        raise TypeError("Datalist has fixed length to conform to Dataset")

    def _update_listlist(self, other):
        "update list elements from another list of lists"
        if len(self) != len(other):
            raise ValueError("Unequal length")
        for e1, e2 in izip(self, other):
            for item in e2:
                if item not in e1:
                    e1.append(item)


legal_dataset_key_re = re.compile("[_A-Za-z][_a-zA-Z0-9]*$")
def assert_is_legal_dataset_key(key):
    if iskeyword(key):
        msg = ("%r is a reserved keyword and can not be used as variable name "
               "in a Dataset" % key)
        raise ValueError(msg)
    elif not legal_dataset_key_re.match(key):
        msg = ("%r is not a valid keyword and can not be used as variable name "
               "in a Dataset" % key)
        raise ValueError(msg)


def as_legal_dataset_key(key):
    "Convert str to a legal dataset key"
    if iskeyword(key):
        return "%s_" % key
    elif legal_dataset_key_re.match(key):
        return key
    else:
        if ' ' in key:
            key = key.replace(' ', '_')
        for c in string.punctuation:
            if c in key:
                key = key.replace(c, '_')

        if key == '':
            key = '_'
        elif key[0].isdigit():
            key = "_%s" % key

        if legal_dataset_key_re.match(key):
            return key
        else:
            raise RuntimeError("Could not convert %r to legal dataset key")


def cases_arg(cases, n_cases):
    "coerce cases argument to iterator"
    if isinstance(cases, int):
        if cases < 1:
            cases = n_cases + cases
            if cases < 0:
                raise ValueError("Can't get table for fewer than 0 cases")
        else:
            cases = min(cases, n_cases)
        return xrange(cases)
    else:
        return cases


class Dataset(OrderedDict):
    """
    Stores multiple variables pertaining to a common set of measurement cases

    Superclass: :class:`collections.OrderedDict`

    Parameters
    ----------
    items : iterator
        Items contained in the Dataset. Items can be either named
        data-objects or ``(name, data_object)`` tuples. The Dataset stores
        the input items themselves, without making a copy().
    name : str
        Name for the Dataset.
    caption : str
        Caption for the table.
    info : dict
        Info dictionary, can contain arbitrary entries and can be accessed
        as ``.info`` attribute after initialization. The Dataset makes a
        shallow copy.
    n_cases : int
        Specify the number of cases in the Dataset if no items are added
        upon initialization (by default the number is inferred when the
        fist item is added).


    Attributes
    ----------
    n_cases : None | int
        The number of cases in the Dataset (corresponding to the number of
        rows in the table representation). None if no variables have been
        added.
    n_items : int
        The number of items (variables) in the Dataset (corresponding to the
        number of columns in the table representation).


    Notes
    -----
    A Dataset represents a data table as a ``{variable_name: value_list}``
    dictionary. Each variable corresponds to a column, and each index in the
    value list corresponds to a row, or case.

    The Dataset class inherits most of its behavior from its superclass
    :py:class:`collections.OrderedDict`.
    Dictionary keys are enforced to be :py:class:`str` objects and should
    correspond to the variable names.
    As for a dictionary, The Dataset's length (``len(ds)``) reflects the number
    of variables in the Dataset (i.e., the number of rows).


    **Accessing Data**

    Standard indexing with :class:`str` is used to access the contained Var
    and Factor objects:

    - ``ds['var1']`` --> ``var1``.
    - ``ds['var1',]`` --> ``Dataset([var1])``.
    - ``ds['var1', 'var2']`` --> ``Dataset([var1, var2])``

    When indexing numerically, the first index defines cases (rows):

    - ``ds[1]`` --> row 1
    - ``ds[1:5]`` or ``ds[1,2,3,4]`` --> rows 1 through 4
    - ``ds[1, 5, 6, 9]`` or ``ds[[1, 5, 6, 9]]`` --> rows 1, 5, 6 and 9

    The second index accesses columns, so case indexing can be combined with
    column indexing:

     - ``ds[:4, :2]`` --> first 4 rows of first 2 columns

    Index a single case retrieves an individual case as ``{name: value}``
    dictionaries:

    - ``ds[1]`` --> ``{'var': 1, 'factor': 'value', ...}``

    The :meth:`.itercases` method can be used to iterate over cases as
    :class:`dict`.


    **Naming**

    While Var and Factor objects themselves need not be named, they need
    to be named when added to a Dataset. This can be done by a) adding a
    name when initializing the Dataset::

        >>> ds = Dataset((('v1', var1), ('v2', var2)))

    or b) by adding the Var or Factor with a key::

        >>> ds['v3'] = var3

    If a Var/Factor that is added to a Dataset does not have a name, the new
    key is automatically assigned to the Var/Factor's ``.name`` attribute.


    Examples
    --------
    Datasets can be initialize with data-objects, or with
    ('name', data-object) tuples::

        >>> ds = Dataset((var1, var2))
        >>> ds = Dataset((('v1', var1), ('v2', var2)))

    Alternatively, variables can be added after initialization::

        >>> ds = Dataset(n_cases=3)
        >>> ds['var', :] = 0
        >>> ds['factor', :] = 'a'
        >>> print ds
        var    factor
        -------------
        0      a
        0      a
        0      a

    """
    _stype = "dataset"

    @staticmethod
    def _args(items=(), name=None, caption=None, info={}, n_cases=None):
        return items, name, caption, info, n_cases

    def __init__(self, *args, **kwargs):
        # backwards compatibility
        if args:
            fmt_1 = isdataobject(args[0])
            fmt_2 = isinstance(args[0], tuple) and isinstance(args[0][0], str)
            if fmt_1:
                warn("Initializing Datasets with multiple data-objects is "
                     "deprecated and will not be possible after version 0.20. "
                     "Provide a list of data-objects instead.",
                     DeprecationWarning)
            if fmt_1 or fmt_2:
                items, name, caption, info, n_cases = self._args(args, **kwargs)
            else:
                items, name, caption, info, n_cases = self._args(*args, **kwargs)
        else:
            items, name, caption, info, n_cases = self._args(**kwargs)

        # collect initial items
        args = []
        for item in items:
            if isdataobject(item):
                if item.name:
                    args.append((item.name, item))
                else:
                    err = ("items need to be named in a Dataset; use "
                            "Dataset(('name', item), ...), or ds = Dataset(); "
                            "ds['name'] = item")
                    raise ValueError(err)
            else:
                name, v = item
                if not v.name:
                    v.name = name
                args.append(item)

        if n_cases is not None:
            assert isinstance(n_cases, int)

        self.n_cases = n_cases
        super(Dataset, self).__init__(args)

        # set state
        self.name = name
        self.info = info.copy()
        self._caption = caption

    def __setstate__(self, state):
        # for backwards compatibility
        self.name = state['name']
        self.info = state['info']
        self._caption = state.get('caption', None)

    def __reduce__(self):
        return self.__class__, (self.items(), self.name, self._caption,
                                self.info, self.n_cases)

    def __delitem__(self, key):
        if isinstance(key, basestring):
            super(Dataset, self).__delitem__(key)
        elif isinstance(key, tuple):
            m = super(Dataset, self).__delitem__
            for k in key:
                m(k)
        else:
            raise KeyError("Invalid Dataset key: %s" % repr(key))

    def __getitem__(self, index):
        """
        possible::

            >>> ds[9]        (int) -> dictionary for one case
            >>> ds[9:12]     (slice) -> subset with those cases
            >>> ds[[9, 10, 11]]     (list) -> subset with those cases
            >>> ds['MEG1']  (strings) -> Var
            >>> ds['MEG1', 'MEG2']  (list of strings) -> list of vars; can be nested!

        """
        if isinstance(index, int):
            return self.get_case(index)
        elif isinstance(index, slice):
            return self.sub(index)

        if isinstance(index, basestring):
            return super(Dataset, self).__getitem__(index)

        if not np.iterable(index):
            raise KeyError("Invalid index for Dataset: %r" % index)

        if all(isinstance(item, basestring) for item in index):
            return Dataset(((item, self[item]) for item in index))

        if isinstance(index, tuple):
            if len(index) != 2:
                raise KeyError("Invalid index for Dataset: %s" % repr(index))

            i0, i1 = index
            if isinstance(i0, basestring):
                return self[i1, i0]
            elif isinstance(i1, basestring):
                return self[i1][i0]
            elif np.iterable(i0) and isinstance(i0[0], basestring):
                return self[i1, i0]
            elif np.iterable(i1) and all(isinstance(item, basestring) for item
                                         in i1):
                keys = i1
            else:
                keys = Datalist(self.keys())[i1]
                if isinstance(keys, basestring):
                    return self[i1][i0]

            subds = Dataset(((k, self[k][i0]) for k in keys))
            return subds

        return self.sub(index)

    def __repr__(self):
        class_name = self.__class__.__name__
        if self.n_cases is None:
            items = []
            if self.name:
                items.append('name=%r' % self.name)
            if self.info:
                info = repr(self.info)
                if len(info) > 60:
                    info = '<...>'
                items.append('info=%s' % info)
            return '%s(%s)' % (class_name, ', '.join(items))

        rep_tmp = "<%(class_name)s %(name)s%(N)s{%(items)s}>"
        fmt = {'class_name': class_name}
        fmt['name'] = '%r ' % self.name if self.name else ''
        fmt['N'] = 'n_cases=%i ' % self.n_cases
        items = []
        for key in self:
            v = self[key]
            if isinstance(v, Var):
                lbl = 'V'
            elif isinstance(v, Factor):
                lbl = 'F'
            elif isinstance(v, NDVar):
                lbl = 'Vnd'
            else:
                lbl = type(v).__name__

            if getattr(v, 'name', key) == key:
                item = '%r:%s' % (key, lbl)
            else:
                item = '%r:<%s %r>' % (key, lbl, v.name)

            items.append(item)

        fmt['items'] = ', '.join(items)
        return rep_tmp % fmt

    def _repr_pretty_(self, p, cycle):
        if cycle:
            raise NotImplementedError
        p.text(self.__repr__())

    def __setitem__(self, index, item, overwrite=True):
        if isinstance(index, basestring):
            # test if name already exists
            if (not overwrite) and (index in self):
                raise KeyError("Dataset already contains variable of name %r" % index)
            assert_is_legal_dataset_key(index)

            # coerce item to data-object
            if isdataobject(item) or isinstance(object, Datalist):
                if not item.name:
                    item.name = index
            elif isinstance(item, (list, tuple)):
                item = Datalist(item, name=index)
            else:
                pass

            # make sure the item has the right length
            if isndvar(item) and not item.has_case:
                N = 0
            else:
                N = len(item)

            if self.n_cases is None:
                self.n_cases = N
            elif self.n_cases != N:
                msg = ("Can not assign item to Dataset. The item`s length "
                       "(%i) is different from the number of cases in the "
                       "Dataset (%i)." % (N, self.n_cases))
                raise ValueError(msg)

            super(Dataset, self).__setitem__(index, item)
        elif isinstance(index, tuple):
            if len(index) != 2:
                err = ("Dataset indexes can have only two components; direct "
                       "access to NDVars is not implemented")
                raise NotImplementedError(err)
            key, idx = index
            if isinstance(idx, basestring):
                key, idx = idx, key
            elif not isinstance(key, basestring):
                TypeError("Dataset indexes need variable specified as string")

            if key in self:
                self[key][idx] = item
            elif isinstance(idx, slice):
                if idx.start is None and idx.stop is None:
                    if isinstance(item, basestring):
                        self[key] = Factor([item], repeat=self.n_cases)
                    else:
                        self[key] = Var([item] * self.n_cases)
                else:
                    err = ("Can only add Factor with general value for all "
                           "cases (ds['name',:] = ...")
                    raise NotImplementedError(err)
            else:
                raise NotImplementedError("Advanced Dataset indexing")
        else:
            raise NotImplementedError("Advanced Dataset indexing")

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        if sum(isuv(i) or isdatalist(i) for i in self.values()) == 0:
            return self.__repr__()

        maxn = preferences['dataset_str_n_cases']
        txt = unicode(self.as_table(maxn, '%.5g', midrule=True, lfmt=True))
        if self.n_cases > maxn:
            note = "... (use .as_table() method to see the whole Dataset)"
            txt = os.linesep.join((txt, note))
        return txt

    def _check_n_cases(self, X, empty_ok=True):
        """Check that an input argument has the appropriate length.

        Also raise an error if empty_ok is False and the Dataset is empty.
        """
        if self.n_cases is None:
            if empty_ok == True:
                return
            else:
                err = ("Dataset is empty.")
                raise RuntimeError(err)

        n = len(X)
        if self.n_cases != n:
            name = getattr(X, 'name', "the argument")
            err = ("The Dataset has a different length (%i) than %s "
                   "(%i)" % (self.n_cases, name, n))
            raise ValueError(err)

    def add(self, item, replace=False):
        """``ds.add(item)`` -> ``ds[item.name] = item``

        unless the Dataset already contains a variable named item.name, in
        which case a KeyError is raised. In order to replace existing
        variables, set ``replace`` to True::

            >>> ds.add(item, True)

        """
        if not isdataobject(item):
            raise ValueError("Not a valid data-object: %r" % item)
        elif item.name is None:
            raise ValueError("Dataset.add(obj) can only take named objects "
                             "(obj.name can not be None)")
        elif (item.name in self) and not replace:
            raise KeyError("Dataset already contains variable named %r" % item.name)
        else:
            self[item.name] = item

    def add_empty_var(self, name, dtype=np.float64):
        """Create an empty variable in the dataset

        Parameters
        ----------
        name : str
            Name for the new variable.
        dtype : numpy dtype
            Data type of the new variable (default is float64).

        Returns
        -------
        var : Var
            The new variable.
        """
        if self.n_cases is None:
            err = "Can't add variable to a Dataset without length"
            raise RuntimeError(err)
        x = np.empty(self.n_cases, dtype=dtype)
        v = Var(x)
        self[name] = v
        return v

    def as_table(self, cases=0, fmt='%.6g', sfmt='%s', sort=False, header=True,
                 midrule=False, count=False, title=None, caption=None,
                 ifmt='%s', bfmt='%s', lfmt=False):
        r"""
        Create an fmtxt.Table containing all Vars and Factors in the Dataset.
        Can be used for exporting in different formats such as csv.

        Parameters
        ----------
        cases : int | iterator of int
            Cases to include (int includes that many cases from the beginning,
            0 includes all; negative number works like negative indexing).
        fmt : str
            Format string for float variables (default ``'%.6g'``).
        sfmt : str | None
            Formatting for strings (None -> code; default ``'%s'``).
        sort : bool
            Sort the columns alphabetically.
        header : bool
            Include the varibale names as a header row.
        midrule : bool
            print a midrule after table header.
        count : bool
            Add an initial column containing the case number.
        title : None | str
            Title for the table.
        caption : None | str
            Caption for the table (default is the Dataset's caption).
        ifmt : str
            Formatting for integers (default ``'%s'``).
        bfmt : str
            Formatting for booleans (default ``'%s'``).
        lfmt : bool
            Include Datalists.
        """
        cases = cases_arg(cases, self.n_cases)
        keys = [k for k, v in self.iteritems() if isuv(v) or (lfmt and isdatalist(v))]
        if sort:
            keys = sorted(keys)

        if caption is None:
            caption = self._caption

        values = [self[key] for key in keys]
        fmts = []
        for v in values:
            if isfactor(v):
                fmts.append(sfmt)
            elif isintvar(v):
                fmts.append(ifmt)
            elif isboolvar(v):
                fmts.append(bfmt)
            elif isdatalist(v):
                fmts.append('dl')
            else:
                fmts.append(fmt)

        columns = 'l' * (len(keys) + count)
        table = fmtxt.Table(columns, True, title, caption)

        if header:
            if count:
                table.cell('#')
            for name in keys:
                table.cell(name)

            if midrule:
                table.midrule()

        for i in cases:
            if count:
                table.cell(i)

            for v, fmt_ in izip(values, fmts):
                if fmt_ is None:
                    table.cell(v.x[i])
                elif fmt_ == 'dl':
                    table.cell(v._item_repr(v[i]))
                elif fmt_.endswith(('r', 's')):
                    table.cell(fmt_ % v[i])
                else:
                    table.cell(fmtxt.Number(v[i], fmt=fmt_))

        return table

    def _asfmtext(self):
        return self.as_table()

    def eval(self, expression):
        """
        Evaluate an expression involving items stored in the Dataset.

        Parameters
        ----------
        expression : str
            Python expression to evaluate, with scipy constituting the global
            namespace and the current Dataset constituting the local namespace.

        Notes
        -----
        ``ds.eval(expression)`` is equivalent to
        ``eval(expression, scipy, ds)``.

        Examples
        --------
        In a Dataset containing factors 'A' and 'B'::

            >>> ds.eval('A % B')
            A % B

        """
        if not isinstance(expression, basestring):
            raise TypeError("Eval needs expression of type unicode or str. Got "
                            "%s" % repr(expression))
        return eval(expression, vars(scipy), self)

    @classmethod
    def from_caselist(cls, names, cases):
        """Create a Dataset from a list of cases

        Parameters
        ----------
        names : sequence of str
            Names for the variables.
        cases : sequence
            A sequence of cases, whereby each case is itself represented as a
            sequence of values (str or scalar). Variable type (Factor or Var)
            is inferred from whether values are str or not.
        """
        ds = cls()
        for i, name in enumerate(names):
            values = [case[i] for case in cases]
            if any(isinstance(v, basestring) for v in values):
                ds[name] = Factor(values)
            else:
                ds[name] = Var(values)
        return ds

    @classmethod
    def from_r(cls, name):
        """Create a Dataset from an R data frame through ``rpy2``

        Parameters
        ----------
        name : str
            Name of the dataframe in R.

        Examples
        --------
        Getting an example dataset from R:

        >>> from rpy2.robjects import r
        >>> r('data(sleep)')
        >>> ds = Dataset.from_r('sleep')
        >>> print ds
        extra   group   ID
        ------------------
        0.7     1       1
        -1.6    1       2
        -0.2    1       3
        -1.2    1       4
        -0.1    1       5
        3.4     1       6
        3.7     1       7
        0.8     1       8
        0       1       9
        2       1       10
        1.9     2       1
        0.8     2       2
        1.1     2       3
        0.1     2       4
        -0.1    2       5
        4.4     2       6
        5.5     2       7
        1.6     2       8
        4.6     2       9
        3.4     2       10
        """
        from rpy2 import robjects as ro
        df = ro.r[name]
        if not isinstance(df, ro.DataFrame):
            raise ValueError("R object %r is not a DataFrame")
        ds = cls(name=name)
        for item_name, item in df.items():
            if isinstance(item, ro.FactorVector):
                x = np.array(item)
                labels = {i:l for i, l in enumerate(item.levels, 1)}
                ds[item_name] = Factor(x, labels=labels)
            elif isinstance(item, (ro.FloatVector, ro.IntVector)):
                x = np.array(item)
                ds[item_name] = Var(x)
            else:
                raise NotImplementedError(str(type(item)))
        return ds

    def get_case(self, i):
        "returns the i'th case as a dictionary"
        return dict((k, v[i]) for k, v in self.iteritems())

    def get_subsets_by(self, X, exclude=[], name='{name}[{cell}]'):
        """
        splits the Dataset by the cells of a Factor and
        returns as dictionary of subsets.

        """
        if isinstance(X, basestring):
            X = self[X]

        out = {}
        for cell in X.cells:
            if cell not in exclude:
                setname = name.format(name=self.name, cell=cell)
                index = (X == cell)
                out[cell] = self.sub(index, setname)
        return out

    def aggregate(self, x=None, drop_empty=True, name='{name}', count='n',
                  drop_bad=False, drop=(), equal_count=False, never_drop=()):
        """
        Return a Dataset with one case for each cell in X.

        Parameters
        ----------
        x : None | str | categorial
            Model defining cells to which to reduce cases. By default (``None``)
            the Dataset is reduced to a single case.
        drop_empty : bool
            Drops empty cells in X from the Dataset. This is currently the only
            option.
        name : str
            Name of the new Dataset.
        count : None | str
            Add a variable with this name to the new Dataset, containing the
            number of cases in each cell in X.
        drop_bad : bool
            Drop bad items: silently drop any items for which compression
            raises an error. This concerns primarily factors with non-unique
            values for cells in X (if drop_bad is False, an error is raised
            when such a Factor is encountered)
        drop : sequence of str
            Additional data-objects to drop.
        equal_count : bool
            Make sure the same number of rows go into each average. First, the
            cell with the smallest number of rows is determined. Then, for each
            cell, rows beyond that number are dropped.
        never_drop : sequence of str
            If the drop_bad=True setting would lead to dropping a variable
            whose name is in never_drop, raise an error instead.

        Notes
        -----
        Handle mne Epoch objects by creating a list with an mne Evoked object
        for each cell.
        """
        if not drop_empty:
            raise NotImplementedError('drop_empty = False')

        if x:
            if equal_count:
                self = self.equalize_counts(x)
            x = ascategorial(x, ds=self)
        else:
            x = Factor('a' * self.n_cases)

        ds = Dataset(name=name.format(name=self.name), info=self.info)

        if count:
            n_cases = filter(None, (np.sum(x == cell) for cell in x.cells))
            ds[count] = Var(n_cases)

        for k, v in self.iteritems():
            if k in drop:
                continue
            try:
                if hasattr(v, 'aggregate'):
                    ds[k] = v.aggregate(x)
                elif isinstance(v, (mne.Epochs, mne.EpochsArray)):
                    evokeds = []
                    for cell in x.cells:
                        idx = (x == cell)
                        if idx.sum():
                            evokeds.append(v[idx].average())
                    ds[k] = evokeds
                else:
                    err = ("Unsupported value type: %s" % type(v))
                    raise TypeError(err)
            except:
                if drop_bad and k not in never_drop:
                    pass
                else:
                    raise

        return ds

    def copy(self, name=True):
        "ds.copy() returns an shallow copy of ds"
        if name is True:
            name = self.name
        return Dataset(self.items(), name, self._caption, self.info,
                       self.n_cases)

    def equalize_counts(self, X):
        """Create a copy of the Dataset with equal counts in each cell of X

        Parameters
        ----------
        X : categorial
            Model which defines the cells in which to equalize the counts.

        Returns
        -------
        equalized_ds : Dataset
            Dataset with equal number of cases in each cell of X.

        Notes
        -----
        First, the cell with the smallest number of rows is determined (empty
        cells are ignored). Then, for each cell, rows beyond that number are
        dropped.
        """
        X = ascategorial(X, ds=self)
        self._check_n_cases(X, empty_ok=False)
        indexes = np.array([X == cell for cell in X.cells])
        n_by_cell = indexes.sum(1)
        n = np.setdiff1d(n_by_cell, [0]).min()
        for index in indexes:
            np.logical_and(index, index.cumsum() <= n, index)
        index = indexes.any(0)
        return self[index]

    def head(self, n=10):
        "Table with the first n cases in the Dataset"
        return self.as_table(n, '%.5g', midrule=True, lfmt=True)

    def index(self, name='index', start=0):
        """
        Add an index to the Dataset (i.e., `range(n_cases)`), e.g. for later
        alignment.

        Parameters
        ----------
        name : str
            Name of the new index variable.
        start : int
            Number at which to start the index.
        """
        self[name] = Var(np.arange(start, self.n_cases + start))

    def itercases(self, start=None, stop=None):
        "iterate through cases (each case represented as a dict)"
        if start is None:
            start = 0

        if stop is None:
            stop = self.n_cases
        elif stop < 0:
            stop = self.n_cases - stop

        for i in xrange(start, stop):
            yield self.get_case(i)

    @property
    def n_items(self):
        return super(Dataset, self).__len__()

    def rename(self, old, new):
        """Shortcut to rename a data-object in the Dataset.

        Parameters
        ----------
        old : str
            Current name of the data-object.
        new : str
            New name for the data-object.
        """
        if old not in self:
            raise KeyError("No item named %r" % old)
        if new in self:
            raise ValueError("Dataset already has variable named %r" % new)
        assert_is_legal_dataset_key(new)

        # update map
        node = self._OrderedDict__map.pop(old)
        node[2] = new
        self._OrderedDict__map[new] = node

        # update dict entry
        obj = self[old]
        dict.__delitem__(self, old)
        dict.__setitem__(self, new, obj)

        # update object name
        if hasattr(obj, 'name'):
            obj.name = new
        self[new] = obj

    def repeat(self, repeats, name='{name}'):
        """
        Returns a new Dataset with each row repeated ``n`` times.

        Parameters
        ----------
        repeats : int | array of int
            Number of repeats, either a constant or a different number for each
            element.
        name : str
            Name for the new Dataset.
        """
        if self.n_cases is None:
            raise RuntimeError("Can't repeat Dataset with unspecified n_cases")

        if isinstance(repeats, int):
            n_cases = self.n_cases * repeats
        else:
            n_cases = sum(repeats)

        return Dataset(((k, v.repeat(repeats)) for k, v in self.iteritems()),
                       name.format(name=self.name), self._caption, self.info,
                       n_cases)

    @property
    def shape(self):
        return (self.n_cases, self.n_items)

    def sort(self, order, descending=False):
        """Sort the Dataset in place.

        Parameters
        ----------
        order : str | data-object
            Data object (Var, Factor or interactions) according to whose values
            to sort the Dataset, or its name in the Dataset.
        descending : bool
            Sort in descending instead of an ascending order.

        See Also
        --------
        .sort_idx : Create an index that could be used to sort the Dataset
        .sorted : Create a sorted copy of the Dataset
        """
        idx = self.sort_idx(order, descending)
        for k in self:
            self[k] = self[k][idx]

    def sort_idx(self, order, descending=False):
        """Create an index that could be used to sort the Dataset.

        Parameters
        ----------
        order : str | data-object
            Data object (Var, Factor or interactions) according to whose values
            to sort the Dataset, or its name in the Dataset.
        descending : bool
            Sort in descending instead of an ascending order.

        See Also
        --------
        .sort : sort the Dataset in place
        .sorted : Create a sorted copy of the Dataset
        """
        if isinstance(order, basestring):
            order = self.eval(order)

        if not len(order) == self.n_cases:
            err = ("Order must be of same length as Dataset; got length "
                   "%i." % len(order))
            raise ValueError(err)

        idx = order.sort_idx(descending=descending)
        return idx

    def save(self):
        """Shortcut to save the Dataset, will display a system file dialog

        Notes
        -----
        Use specific save methods for more options.

        See Also
        --------
        .save_pickled : Pickle the Dataset
        .save_txt : Save as text file
        .save_tex : Save as teX table
        .as_table : Create a table with more control over formatting
        """
        title = "Save Dataset"
        if self.name:
            title += ' %s' % self.name
        msg = ""
        filetypes = [_pickled_ds_wildcard, _tsv_wildcard, _tex_wildcard]
        path = ui.ask_saveas(title, msg, filetypes, defaultFile=self.name)
        _, ext = os.path.splitext(path)
        if ext == '.pickled':
            self.save_pickled(path)
        elif ext == '.txt':
            self.save_txt(path)
        elif ext == '.tex':
            self.save_tex(path)
        else:
            err = ("Unrecognized extension: %r. Needs to be .pickled, .txt or "
                   ".tex." % ext)
            raise ValueError(err)

    def save_rtf(self, path=None, fmt='%.3g'):
        """Save the Dataset as TeX table.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.tex' is appended.
        fmt : format string
            Formatting for scalar values.
        """
        table = self.as_table(fmt=fmt)
        table.save_rtf(path)

    def save_tex(self, path=None, fmt='%.3g', header=True, midrule=True):
        """Save the Dataset as TeX table.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.tex' is appended.
        fmt : format string
            Formatting for scalar values.
        header : bool
            Include the varibale names as a header row.
        midrule : bool
            print a midrule after table header.
        """
        if not isinstance(path, basestring):
            title = "Save Dataset"
            if self.name:
                title += ' %s' % self.name
            title += " as TeX Table"
            msg = ""
            path = ui.ask_saveas(title, msg, [_tex_wildcard],
                                 defaultFile=self.name)

        _, ext = os.path.splitext(path)
        if not ext:
            path += '.tex'

        table = self.as_table(fmt=fmt, header=header, midrule=midrule)
        table.save_tex(path)

    def save_txt(self, path=None, fmt='%s', delim='\t', header=True):
        """Save the Dataset as text file.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.txt' is appended.
        fmt : format string
            Formatting for scalar values.
        delim : str
            Column delimiter (default is tab).
        header : bool
            write the variables' names in the first line
        """
        if not isinstance(path, basestring):
            title = "Save Dataset"
            if self.name:
                title += ' %s' % self.name
            title += " as Text"
            msg = ""
            path = ui.ask_saveas(title, msg, [_tsv_wildcard],
                                 defaultFile=self.name)

        _, ext = os.path.splitext(path)
        if not ext:
            path += '.txt'

        table = self.as_table(fmt=fmt, header=header)
        table.save_tsv(path, fmt=fmt, delimiter=delim)

    def save_pickled(self, path=None):
        """Pickle the Dataset.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.pickled' is appended.
        """
        if not isinstance(path, basestring):
            title = "Pickle Dataset"
            if self.name:
                title += ' %s' % self.name
            msg = ""
            path = ui.ask_saveas(title, msg, [_pickled_ds_wildcard],
                                 defaultFile=self.name)

        _, ext = os.path.splitext(path)
        if not ext:
            path += '.pickled'

        with open(path, 'wb') as fid:
            pickle.dump(self, fid, pickle.HIGHEST_PROTOCOL)

    def sorted(self, order, descending=False):
        """Create an sorted copy of the Dataset.

        Parameters
        ----------
        order : str | data-object
            Data object (Var, Factor or interactions) according to whose values
            to sort the Dataset, or its name in the Dataset.
        descending : bool
            Sort in descending instead of an ascending order.

        See Also
        --------
        .sort : sort the Dataset in place
        .sort_idx : Create an index that could be used to sort the Dataset
        """
        idx = self.sort_idx(order, descending)
        ds = self[idx]
        return ds

    def sub(self, index, name='{name}'):
        """
        Returns a Dataset containing only the cases selected by `index`.

        Parameters
        ----------
        index : int | array | str
            Index for selecting a subset of cases. Can be an valid numpy index
            or a string (the name of a variable in Dataset, or an expression
            to be evaluated in the Dataset's namespace).
        name : str
            name for the new Dataset.

        Notes
        -----
        Keep in mind that index is passed on to numpy objects, which means
        that advanced indexing always returns a copy of the data, whereas
        basic slicing (using slices) returns a view.
        """
        if isinstance(index, int):
            if index == -1:
                index = slice(-1, None)
            else:
                index = slice(index, index + 1)
        elif isinstance(index, str):
            index = self.eval(index)

        if isvar(index):
            index = index.x

        return Dataset(((k, v[index]) for k, v in self.iteritems()),
                       name.format(name=self.name), self._caption, self.info)

    def tail(self, n=10):
        "Table with the last n cases in the Dataset"
        return self.as_table(xrange(-n, 0), '%.5g', midrule=True, lfmt=True)

    def to_r(self, name=None):
        """Place the Dataset into R as dataframe using rpy2

        Parameters
        ----------
        name : str
            Name for the R dataframe (default is self.name).

        Examples
        --------
        >>> from rpy2.robjects import r
        >>> ds = datasets.get_uv()
        >>> print ds[:6]
        A    B    rm     intvar   fltvar     fltvar2    index
        -----------------------------------------------------
        a1   b1   s000   13       0.25614    0.7428     True
        a1   b1   s001   8        -1.5174    -0.75498   True
        a1   b1   s002   11       -0.5071    -0.13828   True
        a1   b1   s003   11       2.1491     -2.1249    True
        a1   b1   s004   15       -0.19358   -1.03      True
        a1   b1   s005   17       2.141      -0.51745   True
        >>> ds.to_r('df')
        >>> print r("head(df)")
           A  B   rm intvar     fltvar    fltvar2 index
        1 a1 b1 s000     13  0.2561439  0.7427957  TRUE
        2 a1 b1 s001      8 -1.5174371 -0.7549815  TRUE
        3 a1 b1 s002     11 -0.5070960 -0.1382827  TRUE
        4 a1 b1 s003     11  2.1490761 -2.1249203  TRUE
        5 a1 b1 s004     15 -0.1935783 -1.0300188  TRUE
        6 a1 b1 s005     17  2.1410424 -0.5174519  TRUE

        """
        import rpy2.robjects as ro

        if name is None:
            name = self.name
            if name is None:
                raise TypeError('Need a valid name for the R data frame')

        items = OrderedDict()
        for k, v in self.iteritems():
            if isvar(v):
                if v.x.dtype.kind == 'b':
                    item = ro.BoolVector(v.x)
                elif v.x.dtype.kind == 'i':
                    item = ro.IntVector(v.x)
                else:
                    item = ro.FloatVector(v.x)
            elif isfactor(v):
                x = ro.IntVector(v.x)
                codes = sorted(v._labels)
                levels = ro.IntVector(codes)
                labels = ro.StrVector(tuple(v._labels[c] for c in codes))
                item = ro.FactorVector(x, levels, labels)
            else:
                continue
            items[k] = item

        df = ro.DataFrame(items)
        ro.globalenv[name] = df

    def update(self, ds, replace=False, info=True):
        """Update the Dataset with all variables in ``ds``.

        Parameters
        ----------
        ds : dict-like
            A dictionary like object whose keys are strings and whose values
            are data-objects.
        replace : bool
            If a variable in ds is already present, replace it. If False,
            duplicates raise a ValueError (unless they are equivalent).
        info : bool
            Also update the info dictionary.

        Notes
        -----
        By default, if a key is present in both Datasets, and the corresponding
        variables are not equal on all cases, a ValueError is raised. If all
        values are equal, the variable in ds is copied into the Dataset that is
        being updated (the expected behavior of .update()).
        """
        if not replace:
            unequal = []
            for key in set(self).intersection(ds):
                if not np.all(self[key] == ds[key]):
                    unequal.append(key)
            if unequal:
                err = ("The following variables are present twice but are not "
                       "equal: %s" % unequal)
                raise ValueError(err)

        super(Dataset, self).update(ds)

        if info and isdataset(ds):
            self.info.update(ds.info)



class Interaction(_Effect):
    """Represents an Interaction effect.

    Usually not initialized directly but through operations on Factors/Vars.

    Parameters
    ----------
    base : list
        List of data-objects that form the basis of the interaction.

    Attributes
    ----------
    factors :
        List of all factors (i.e. nonbasic effects are broken up into
        factors).
    base :
        All effects.
    """
    _stype = "interaction"

    def __init__(self, base):
        # FIXME: Interaction does not update when component factors update
        self.base = EffectList()
        self.is_categorial = True
        self.nestedin = EffectList()

        for b in base:
            if isuv(b):
                self.base.append(b.copy()),
                if isvar(b):
                    if self.is_categorial:
                        self.is_categorial = False
                    else:
                        raise TypeError("No Interaction between two Var objects")
            elif isinteraction(b):
                if (not b.is_categorial) and (not self.is_categorial):
                    raise TypeError("No Interaction between two Var objects")
                else:
                    self.base.extend(b.base)
                    self.is_categorial = (self.is_categorial and b.is_categorial)
            elif b._stype == "nested":  # TODO: nested effects
                self.base.append(b)
                if b.nestedin not in self.nestedin:
                    self.nestedin.append(b.nestedin)
            else:
                raise TypeError("Invalid type for Interaction: %r" % type(b))

        if len(self.base) < 2:
            raise ValueError("Interaction needs a base of at least two Factors "
                             "(got %s)" % repr(base))
        self._n_cases = N = len(self.base[0])
        if not all([len(f) == N for f in self.base[1:]]):
            err = ("Interactions only between effects with the same number of "
                   "cases")
            raise ValueError(err)

        self.base_names = [str(f.name) for f in self.base]
        self.name = ' x '.join(self.base_names)
        self.random = False
        self.df = reduce(operator.mul, [f.df for f in self.base])

        # determine cells:
        factors = EffectList(filter(isfactor, self.base))
        self.cells = tuple(itertools.product(*(f.cells for f in factors)))
        self.cell_header = tuple(f.name for f in factors)

        self.beta_labels = ['?'] * self.df  # TODO:

    def __repr__(self):
        names = [UNNAMED if f.name is None else f.name for f in self.base]
        if preferences['short_repr']:
            return ' % '.join(names)
        else:
            return "Interaction({n})".format(n=', '.join(names))

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, index):
        if isvar(index):
            index = index.x

        out = tuple(f[index] for f in self.base)

        if index_ndim(index) == 1:
            return Interaction(out)
        else:
            return out

    def __contains__(self, item):
        if isinstance(item, tuple):
            return item in self._value_set
        return self.base.__contains__(item)

    def __iter__(self):
        for i in xrange(len(self)):
            yield tuple(b[i] for b in self.base)

    # numeric ---
    def __eq__(self, other):
        if isinteraction(other) and len(other.base) == len(self.base):
            x = np.vstack((b == bo for b, bo in izip(self.base, other.base)))
            return np.all(x, 0)
        elif isinstance(other, tuple) and len(other) == len(self.base):
            x = np.vstack(factor == level for factor, level in izip(self.base, other))
            return np.all(x, 0)
        else:
            return np.zeros(len(self), bool)

    def __ne__(self, other):
        if isinteraction(other) and len(other.base) == len(self.base):
            x = np.vstack((b != bo for b, bo in izip(self.base, other.base)))
            return np.any(x, 0)
        elif isinstance(other, tuple) and len(other) == len(self.base):
            x = np.vstack(factor != level for factor, level in izip(self.base, other))
            return np.any(x, 0)
        return np.ones(len(self), bool)

    def as_factor(self, delim=' ', name=None):
        """Convert the Interaction to a factor

        Parameters
        ----------
        delim : str
            Delimiter to join factor cell values (default ``" "``).
        name : str
            Name for the Factor (default is None).

        Examples
        --------
        >>> print ds[::20, 'A']
        Factor(['a1', 'a1', 'a2', 'a2'], name='A')
        >>> print ds[::20, 'B']
        Factor(['b1', 'b2', 'b1', 'b2'], name='B')
        >>> i = ds.eval("A % B")
        >>> print i.as_factor()[::20]
        Factor(['a1 b1', 'a1 b2', 'a2 b1', 'a2 b2'], name='AxB')
        >>> print i.as_factor("_")[::20]
        Factor(['a1_b1', 'a1_b2', 'a2_b1', 'a2_b2'], name='AxB')
        """
        return Factor(self.as_labels(delim), name)

    def as_cells(self):
        """All values as a list of tuples."""
        return [case for case in self]

    @LazyProperty
    def as_dummy(self):
        codelist = [f.as_dummy for f in self.base]
        return reduce(_effect_interaction, codelist)

    @LazyProperty
    def as_effects(self):
        "effect coding"
        codelist = [f.as_effects for f in self.base]
        return reduce(_effect_interaction, codelist)

    def _coefficient_names(self, method):
        if self.df == 1:
            return [self.name]
        return ["%s %i" % (self.name, i) for i in xrange(self.df)]

    def as_labels(self, delim=' '):
        """All values as a list of strings.

        Parameters
        ----------
        delim : str
            Delimiter with which to join the elements of cells.
        """
        return [delim.join(filter(None, map(str, case))) for case in self]

    def aggregate(self, X):
        return Interaction(f.aggregate(X) for f in self.base)

    def isin(self, cells):
        """An index that is true where the Interaction equals any of the cells.

        Parameters
        ----------
        cells : sequence of tuples
            Cells for which the index will be true. Cells described as tuples
            of strings.
        """
        is_v = [self == cell for cell in cells]
        return np.any(is_v, 0)

    @LazyProperty
    def _value_set(self):
        return set(self)


class diff(object):
    """
    helper to create difference values for correlation.

    """
    def __init__(self, X, c1, c2, match, sub=None):
        """
        X: Factor providing categories
        c1: category 1
        c2: category 2
        match: Factor matching values between categories

        """
        raise NotImplementedError
        # FIXME: use celltable
        sub = X.isany(c1, c2)
#        ct = celltable
#        ...
        i1 = X.code_for_label(c1)
        i2 = X.code_for_label(c2)
        self.I1 = X == i1
        self.I2 = X == i2

        if sub is not None:
            self.I1 = self.I1 * sub
            self.I2 = self.I2 * sub

        m1 = match.x[self.I1]
        m2 = match.x[self.I2]
        self.s1 = np.argsort(m1)
        self.s2 = np.argsort(m2)
        assert np.all(np.unique(m1) == np.unique(m2))
        self.name = "{n}({x1}-{x2})".format(n='{0}', x1=X.cells[i1], x2=X.cells[i2])

    def subtract(self, Y):
        ""
        assert type(Y) is Var
#        if self.sub is not None:
#            Y = Y[self.sub]
        Y1 = Y[self.I1]
        Y2 = Y[self.I2]
        y = Y1[self.s1] - Y2[self.s2]
        name = self.name.format(Y.name)
        # name = Y.name + '_DIFF'
        return Var(y, name)

    def extract(self, Y):
        ""
        y1 = Y[self.I1].x[self.s1]
        y2 = Y[self.I2].x[self.s2]
        assert np.all(y1 == y2), Y.name
        if type(Y) is Factor:
            return Factor(y1, Y.name, random=Y.random, labels=Y.cells)
        else:
            return Var(y1, Y.name)

    @property
    def N(self):
        return np.sum(self.I1)


def box_cox_transform(X, p, name=None):
    """
    :returns: a variable with the Box-Cox transform applied to X. With p==0,
        this is the log of X; otherwise (X**p - 1) / p

    :arg Var X: Source variable
    :arg float p: Parameter for Box-Cox transform

    """
    if isvar(X):
        X = X.x

    if p == 0:
        y = np.log(X)
    else:
        y = (X ** p - 1) / p

    return Var(y, name)


class NestedEffect(object):

    _stype = "nested"

    def __init__(self, effect, nestedin):
        if not iscategorial(nestedin):
            raise TypeError("Effects can only be nested in categorial base")

        self.effect = effect
        self.nestedin = nestedin
        self.random = effect.random
        self.cells = effect.cells
        self._n_cases = len(effect)

        if isfactor(self.effect):
            e_name = self.effect.name
        else:
            e_name = '(%s)' % self.effect
        self.name = "%s(%s)" % (e_name, nestedin.name)

        if len(nestedin) != self._n_cases:
            err = ("Unequal lengths: effect %r len=%i, nestedin %r len=%i" %
                   (e_name, len(effect), nestedin.name, len(nestedin)))
            raise ValueError(err)

    def __repr__(self):
        return self.name

    def __iter__(self):
        return self.effect.__iter__()

    def __len__(self):
        return self._n_cases

    @property
    def df(self):
        return len(self.effect.cells) - len(self.nestedin.cells)

    @property
    def as_effects(self):
        "create effect codes"
        codes = np.zeros((self._n_cases, self.df))
        ix = 0
        for outer_cell in self.nestedin.cells:
            outer_idx = (self.nestedin == outer_cell)
            inner_model = self.effect[outer_idx]
            n = len(inner_model.cells)
            inner_codes = _effect_eye(n)
            for i, cell in enumerate(inner_model.cells):
                codes[self.effect == cell, ix:ix + n - 1] = inner_codes[i]
            ix += n - 1

        return codes

    def _coefficient_names(self, method):
        return ["%s %i" % (self.name, i) for i in xrange(self.df)]


class NonbasicEffect(object):

    _stype = "nonbasic"

    def __init__(self, effect_codes, factors, name, nestedin=[],
                 beta_labels=None):
        if beta_labels is not None and len(beta_labels) != effect_codes.shape[1]:
            raise ValueError("beta_labels need one entry per model column "
                             "(%s); got %s"
                             % (effect_codes.shape[1], repr(beta_labels)))
        self.nestedin = nestedin
        self.name = name
        self.random = False
        self.as_effects = effect_codes
        self._n_cases, self.df = effect_codes.shape
        self.factors = factors
        self.beta_labels = beta_labels

    def __repr__(self):
        txt = "<NonbasicEffect: {n}>"
        return txt.format(n=self.name)

    # container ---
    def __len__(self):
        return self._n_cases

    def _coefficient_names(self, method):
        if self.beta_labels is None:
            return ["%s %i" % (self.name, i) for i in xrange(self.df)]
        else:
            return self.beta_labels


class Model(object):
    """A list of effects.

    Parameters
    ----------
    x : effect | iterator of effects
        Effects to be included in the model (Var, Factor, Interaction ,
        ...). Can also contain models, in which case all the model's
        effects will be added.

    Notes
    -----
    a Model's data is exhausted by its :attr:`.effects` list; all the rest are
    @properties.

    Accessing effects:
     - as list in Model.effects
     - with name as Model[name]
    """
    _stype = "model"

    def __init__(self, x):
        effects = EffectList()

        # find effects in input
        if iseffect(x):
            effects.append(x)
            n_cases = len(x)
        elif ismodel(x):
            effects += x.effects
            n_cases = len(x)
        else:
            n_cases = None
            for e in x:
                # check n_cases
                if n_cases is None:
                    n_cases = len(e)
                elif len(e) != n_cases:
                    e0 = effects[0]
                    err = ("All effects contained in a Model need to describe"
                           " the same number of cases. %r has %i cases, %r has"
                           " %i cases." % (e0.name, len(e0), e.name, len(e)))
                    raise ValueError(err)

                # find effects
                if iseffect(e):
                    effects.append(e)
                elif ismodel(e):
                    effects += e.effects
                else:
                    err = ("Model needs to be initialized with effect (Var, "
                           "Factor, Interaction, ...) and/or Model objects "
                           "(got %s)" % type(e))
                    raise TypeError(err)

        self.effects = effects
        self._n_cases = n_cases

        # beta indices
        self.beta_index = beta_index = {}
        i = 1
        for e in effects:
            if isfactor(e) and len(e.cells) == 1:
                raise ValueError("The Factor %s has only one level (%s). The "
                                 "intercept is implicit in each model and "
                                 "should not be specified explicitly."
                                 % (e.name, e.cells[0]))
            k = i + e.df
            beta_index[e] = slice(i, k)
            i = k

        # dfs
        self.df_total = df_total = n_cases
        self.df = df = sum(e.df for e in effects) + 1  # intercept
        self.df_error = df_error = df_total - df

        if df_error < 0:
            raise ValueError("Model overspecified")

        # names
        self.name = ' + '.join([str(e.name) for e in self.effects])

    def __repr__(self):
        names = self.effects.names()
        if preferences['short_repr']:
            return ' + '.join(names)
        else:
            x = ', '.join(names)
            return "Model((%s))" % x

    def __str__(self):
        return str(self.as_table())

    # container ---
    def __len__(self):
        return self._n_cases

    def __getitem__(self, sub):
        if isinstance(sub, str):
            for e in self.effects:
                if e.name == sub:
                    return e
            raise ValueError("No effect named %r" % sub)
        else:
            return Model((x[sub] for x in self.effects))

    def __contains__(self, effect):
        return id(effect) in map(id, self.effects)

    def sorted(self):
        """
        returns sorted Model, interactions last

        """
        out = []
        i = 1
        while len(out) < len(self.effects):
            for e in self.effects:
                if len(e.factors) == i:
                    out.append(e)
            i += 1
        return Model(out)

    # numeric ---
    def __add__(self, other):
        return Model((self, other))

    def __mul__(self, other):
        return Model((self, other, self % other))

    def __mod__(self, other):
        out = []
        for e_self in self.effects:
            for e_other in Model(other).effects:
                out.append(e_self % e_other)
        return Model(out)

    def __eq__(self, other):
        if not isinstance(other, Model):
            return False
        elif not len(self) == len(other):
            return False
        elif not len(self.effects) == len(other.effects):
            return False

        for e, eo in izip(self.effects, other.effects):
            if not np.all(e == eo):
                return False
        return True

    def __ne__(self, other):
        return not self.__eq__(other)

    # repr ---
    @property
    def model_eq(self):
        return self.name

    # coding ---
    @LazyProperty
    def _effect_to_beta(self):
        """An array indicating for each effect which beta weights it occupies

        Returns
        -------
        effects_to_beta : np.ndarray (n_effects, 2)
            For each effect, indicating the first index in betas and df
        """
        out = np.empty((len(self.effects), 2), np.int16)
        beta_start = 1
        for i, e in enumerate(self.effects):
            out[i, 0] = beta_start
            out[i, 1] = e.df
            beta_start += e.df
        return out

    @LazyProperty
    def as_effects(self):
        return np.hstack((e.as_effects for e in self.effects))

    def as_table(self, method='dummy', cases=0, group_terms=True):
        """Return a table with the model codes

        Parameters
        ----------
        method : 'effect' | 'dummy'
            Coding scheme: effect coding or dummy coding.
        cases : int | iterator of int
            Cases to include (int includes that many cases from the beginning,
            0 includes all; negative number works like negative indexing).
        group_terms : bool
            Group model columns that represent the same effect under one
            heading.

        Returns
        --------
        table : FMText Table
            The full model as a table.
        """
        cases = cases_arg(cases, self._n_cases)
        p = self._parametrize(method)
        table = fmtxt.Table('l' * len(p.column_names))

        # Header
        if group_terms:
            for term in p.effect_names:
                index = p.terms[term]
                ncolumns = index.stop - index.start
                table.cell(term, width=ncolumns)
        else:
            for term in p.column_names:
                table.cell(term)
        table.midrule()

        # data
        for case in cases:
            for i in p.x[case]:
                table.cell('%g' % i)

        return table

    def fit(self, Y):
        """
        Find the beta weights by fitting the model to data

        Parameters
        ----------
        Y : Var | array, shape = (n_cases,)
            Data to fit the model to.

        Returns
        -------
        beta : array, shape = (n_regressors, )
            The beta weights.
        """
        Y = asvar(Y)
        beta = dot(self.xsinv, Y.x)
        return beta

    @LazyProperty
    def full(self):
        "returns the full model including an intercept"
        out = np.empty((self._n_cases, self.df))

        # intercept
        out[:, 0] = 1
        self.full_index = {'I': slice(0, 1)}

        # effects
        i = 1
        for e in self.effects:
            j = i + e.df
            out[:, i:j] = e.as_effects
            self.full_index[e] = slice(i, j)
            i = j
        return out

    def head(self, n=10):
        "Table with the first n cases in the Model"
        return self.as_table(cases=n)

    # checking model properties
    def check(self, v=True):
        "shortcut to check linear independence and orthogonality"
        return self.lin_indep(v) + self.orthogonal(v)

    def lin_indep(self, v=True):
        "Checks the Model for linear independence of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
        for i in range(ne):
            for j in range(i + 1, ne):
                e1 = self.effects[i]
                e2 = self.effects[j]
                X = np.hstack((codes[i], codes[j]))
                if rank(X) < X.shape[1]:
                    if v:
                        errtxt = "Linear Dependence Warning: {0} and {1}"
                        msg.append(errtxt.format(e1.name, e2.name))
        return msg

    def orthogonal(self, v=True):
        "Checks the Model for orthogonality of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
#        allok = True
        for i in range(ne):
            for j in range(i + 1, ne):
                ok = True
                e1 = self.effects[i]
                e2 = self.effects[j]
                e1e = codes[i]
                e2e = codes[j]
                for i1 in range(e1.df):
                    for i2 in range(e2.df):
                        dotp = np.dot(e1e[:, i1], e2e[:, i2])
                        if dotp != 0:
                            ok = False
#                            allok = False
                if v and (not ok):
                    errtxt = "Not orthogonal: {0} and {1}"
                    msg.append(errtxt.format(e1.name, e2.name))
        return msg

    def _parametrize(self, method='effect'):
        "Create a design matrix"
        return Parametrization(self, method)

    def repeat(self, n):
        "Analogous to numpy repeat method"
        effects = [e.repeat(n) for e in self.effects]
        return Model(effects)

    def tail(self, n=10):
        "Table with the last n cases in the Model"
        return self.as_table(cases=xrange(-n, 0))

    @LazyProperty
    def xsinv(self):
        x = self.full
        x_t = x.T
        return dot(inv(dot(x_t, x)), x_t)


class Parametrization(object):
    """Parametrization of a statistical model

    Parameters
    ----------
    model : Model
        Model to be parametrized.
    method : 'effect' | 'dummy'
        Coding scheme: effect coding or dummy coding.

    Attributes
    ----------
    model : Model
        The model that is parametrized.
    x : array (n_cases, n_coeffs)
        Design matrix.
    terms : {str: slice}
        Location of each term in x.
    column_names : list of str
        Name of each column.

    Notes
    -----
    A :class:`Model` is a list of effects. A :class:`Parametrization` contains
    a realization of those effects in a model matrix with named columns.
    """
    def __init__(self, model, method):
        model = asmodel(model)
        x = np.empty((model._n_cases, model.df))
        x[:, 0] = 1
        column_names = ['intercept']
        effect_names = ['intercept']
        higher_level_effects = {}
        terms = {'intercept': slice(0, 1)}
        i = 1
        for e in model.effects:
            j = i + e.df
            if method == 'effect':
                x[:, i:j] = e.as_effects
            elif method == 'dummy':
                x[:, i:j] = e.as_dummy
            else:
                raise ValueError("method=%s" % repr(method))
            name = longname(e)
            if name in terms:
                raise KeyError("Duplicate term name: %s" % repr(name))
            terms[name] = slice(i, j)
            effect_names.append(name)
            col_names = e._coefficient_names(method)
            column_names.extend(col_names)
            for col, col_name in enumerate(col_names, i):
                terms[col_name] = slice(col, col + 1)
            i = j

            # find comparison models
            higher_level_effects[name] = [e_ for e_ in model.effects
                                          if e_ is not e
                                          and is_higher_order_effect(e_, e)]

        # model basics
        self.model = model
        self.x = x
        self.terms = terms
        self.column_names = column_names
        self.effect_names = effect_names
        self._higher_level_effects = higher_level_effects

        # projector
        x_t = x.T
        self.projector = inv(x_t.dot(x)).dot(x_t)

    def reduced_model_index(self, term):
        "Boolean index into model columns for model comparison"
        out = np.ones(self.x.shape[1], bool)
        out[self.terms[term]] = False
        for e in self._higher_level_effects[term]:
            out[self.terms[e.name]] = False
        return out


# ---NDVar dimensions---

DIMINDEX_RAW_TYPES = (int, slice, list)

def dimindex_case(arg):
    if isinstance(arg, DIMINDEX_RAW_TYPES):
        return arg
    elif isvar(arg):
        return arg.x
    elif isinstance(arg, np.ndarray) and arg.dtype.kind in 'bi':
        return arg
    else:
        raise TypeError("Unknown index type for case dimension: %s"
                        % repr(arg))


def find_time_point(times, time, rnd='closest'):
    """
    Returns (index, time) for the closest point to ``time`` in ``times``

    Parameters
    ----------
    times : array, 1d
        Monotonically increasing time values.
    time : scalar
        Time point for which to find a match.
    rnd : 'down' | 'closest' | 'up'
        Rounding: how to handle time values that do not have an exact match in
        times. Round 'up', 'down', or to the 'closest' value.
    """
    if time in times:
        i = np.where(times == time)[0][0]
    else:
        gr = (times > time)
        if np.all(gr):
            if times[1] - times[0] > times[0] - time:
                return 0, times[0]
            else:
                name = repr(times.name) if hasattr(times, 'name') else ''
                raise ValueError("time=%s lies outside array %r" % (time, name))
        elif np.any(gr):
            pass
        elif times[-1] - times[-2] >= time - times[-1]:
            return len(times) - 1, times[-1]
        else:
            name = repr(times.name) if hasattr(times, 'name') else ''
            raise ValueError("time=%s lies outside array %r" % (time, name))

        i_next = np.where(gr)[0][0]
        t_next = times[i_next]

        if rnd == 'up':
            return i_next, t_next

        sm = times < time
        i_prev = np.where(sm)[0][-1]
        t_prev = times[i_prev]

        if rnd == 'down':
            return i_prev, t_prev
        elif rnd != 'closest':
            raise ValueError("Invalid argument rnd=%r" % rnd)

        if (t_next - time) < (time - t_prev):
            i = i_next
            time = t_next
        else:
            i = i_prev
            time = t_prev
    return i, time


def _subgraph_edges(connectivity, int_index):
    "Extract connectivity for a subset of a graph"
    if connectivity is None:
        return None

    idx = np.logical_and(np.in1d(connectivity[:, 0], int_index),
                         np.in1d(connectivity[:, 1], int_index))
    if np.any(idx):
        new_c = connectivity[idx]

        # remap to new vertex indices
        if np.any(np.diff(int_index) < 1):  # non-monotonic index
            argsort = np.argsort(int_index)
            flat_conn_ = np.digitize(new_c.ravel(), int_index[argsort], True)
            flat_conn = argsort[flat_conn_]
        else:
            flat_conn = np.digitize(new_c.ravel(), int_index, True)

        return flat_conn.reshape(new_c.shape).astype(np.uint32)
    else:
        return np.empty((0, 2), dtype=np.uint32)


class Dimension(object):
    """Base class for dimensions.

    Attributes
    ----------
    x : array_like
        Numerical values (e.g. for locating categories on an axis).
    values : sequence
        Meaningful point descriptions (e.g. time points, sensor names, ...).
    """
    name = 'Dimension'
    adjacent = True
    _axis_unit = None

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __eq__(self, other):
        if isinstance(other, basestring):
            return False
        return self.name == other.name

    def __ne__(self, other):
        return not self == other

    def __getitem__(self, index):
        """
         - int -> label or value for that location
         - [int] -> Dimension object with 1 location
         - [int, ...] -> Dimension object
        """
        raise NotImplementedError

    def _diminfo(self):
        "Return a str describing the dimension in on line (79 chars)"
        return str(self.name)

    def _axis_im_extent(self):
        "extent for im plots"
        return -0.5, len(self) - 0.5

    def _axis_formatter(self):
        "Return a matplotlib axis formatter"
        raise NotImplementedError

    def dimindex(self, arg):
        """Process index parameter

        Notes
        -----
        Boolean and int arrays are always considered indexing self.values.
        """
        if isndvar(arg):
            return self._dimindex_for_ndvar(arg)
        elif isvar(arg):
            return arg.x
        elif isinstance(arg, np.ndarray) and arg.dtype.kind in 'bi':
            return arg
        elif isinstance(arg, (slice, int)):
            return arg
        elif isinstance(arg, SEQUENCE_TYPES):
            if len(arg) == 0:
                return np.empty(0, np.int8)
            return np.array([self.dimindex(a) for a in arg])
        else:
            raise TypeError("Unknown index type for %s dimension: %s" 
                            % (self.name, repr(arg)))

    def _dimindex_for_ndvar(self, arg):
        "Dimindex for NDVar index"
        if arg.x.dtype.kind != 'b':
            raise IndexError("Only NDVars with boolean data can serve "
                             "as indexes. Got %s." % repr(arg))
        elif arg.ndim != 1:
            raise IndexError("Only NDVars with ndim 1 can serve as "
                             "indexes. Got %s." % repr(arg))
        elif arg.dims[0] != self:
            raise IndexError("Index dimension %s is different from data "
                             "dimension" % arg.dims[0].name)
        else:
            return arg.x

    def distances(self):
        raise NotImplementedError("Distances for %s dimension" % self.name)

    def intersect(self, dim, check_dims=True):
        """Create a Dimension that is the intersection with dim

        Parameters
        ----------
        dim : Dimension
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple).

        Returns
        -------
        intersection : Dimension
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        raise NotImplementedError

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : None | Dataset
            A dataset with variables describing cluster properties.
        """
        return None


class Categorial(Dimension):
    """Simple categorial dimension

    Parameters
    ----------
    name : str
        Dimension name.
    values : list of str
        Names of the entries.
    """
    def __init__(self, name, values):
        if len(set(values)) < len(values):
            raise ValueError("Dimension can not have duplicate values")
        values = np.asarray(values)
        if values.dtype.kind not in 'SU':
            raise ValueError("All Categorial values must be strings")
        self.name = name
        self.values = values

    def __getstate__(self):
        state = {'name': self.name,
                 'values': self.values}
        return state

    def __setstate__(self, state):
        name = state['name']
        values = state['values']
        self.__init__(name, values)

    def __repr__(self):
        args = (repr(self.name), str(self.values))
        return "%s(%s)" % (self.__class__.__name__, ', '.join(args))

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        is_equal = (Dimension.__eq__(self, other)
                    and np.all(self.values == other.values))
        return is_equal

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.values[index]

        values = self.values[index]
        return Categorial(self.name, values)

    def _axis_formatter(self):
        return IndexFormatter(self.values)

    def dimindex(self, arg):
        if isinstance(arg, self.__class__):
            s_idx, a_idx = np.nonzero(self.values[:, None] == arg.values)
            return s_idx[np.argsort(a_idx)]
        elif isinstance(arg, basestring):
            return np.flatnonzero(self.values == arg)[0]
        else:
            return super(Categorial, self).dimindex(arg)

    def _diminfo(self):
        return "%s" % self.name.capitalize()

    def intersect(self, dim, check_dims=False):
        """Create a dimension object that is the intersection with dim

        Parameters
        ----------
        dim : type(self)
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : type(self)
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        if np.array_equal(self.values, dim.values):
            return self
        values = np.intersect1d(self.values, dim.values)
        if np.array_equal(self.values, values):
            return self
        elif np.array_equal(dim.values, values):
            return dim

        return self.__class__(self.name, values)


class Scalar(Dimension):
    "Simple scalar dimension"
    def __init__(self, name, values, unit=None):
        self.x = self.values = values = np.asarray(values)
        if len(np.unique(values)) < len(values):
            raise ValueError("Dimension can not have duplicate values")
        self.name = name
        self.unit = unit

    def __getstate__(self):
        state = {'name': self.name,
                 'values': self.values,
                 'unit': self.unit}
        return state

    def __setstate__(self, state):
        name = state['name']
        values = state['values']
        unit = state.get('unit', None)
        self.__init__(name, values, unit)

    def __repr__(self):
        args = [repr(self.name), str(self.values)]
        if self.unit is not None:
            args.append(repr(self.unit))
        return "%s(%s)" % (self.__class__.__name__, ', '.join(args))

    def __len__(self):
        return len(self.values)

    def __eq__(self, other):
        is_equal = (Dimension.__eq__(self, other)
                    and np.array_equal(self.values, other.values))
        return is_equal

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.values[index]

        values = self.values[index]
        return Scalar(self.name, values, self.unit)

    def _axis_formatter(self):
        return IndexFormatter(self.values)

    def dimindex(self, arg):
        if isinstance(arg, self.__class__):
            s_idx, a_idx = np.nonzero(self.values[:, None] == arg.values)
            return s_idx[np.argsort(a_idx)]
        elif np.isscalar(arg):
            return np.argmin(np.abs(self.values - arg))
        else:
            return super(Scalar, self).dimindex(arg)

    def _diminfo(self):
        return "%s" % self.name.capitalize()

    def intersect(self, dim, check_dims=False):
        """Create a dimension object that is the intersection with dim

        Parameters
        ----------
        dim : type(self)
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : type(self)
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        if np.all(self.values == dim.values):
            return self
        values = np.intersect1d(self.values, dim.values)
        if np.all(self.values == values):
            return self
        elif np.all(dim.values == values):
            return dim

        return self.__class__(self.name, values)


class Ordered(Scalar):
    """Scalar with guarantee that values are ordered"""
    def __init__(self, name, values, unit=None):
        values = np.sort(values)
        Scalar.__init__(self, name, values, unit=unit)

    def _axis_formatter(self):
        return IndexFormatter(self.values)

    def dimindex(self, arg):
        if isinstance(arg, tuple):
            if len(arg) != 2:
                raise ValueError("Tuple indexes for the %s dimension signify "
                                 "intervals and need to be exactly of length "
                                 "2 (got %s)" % (self.name, repr(arg)))
            start, stop = arg
            if start is not None:
                start = super(Ordered, self).dimindex(start)
            if stop is not None:
                stop = super(Ordered, self).dimindex(stop)
            return slice(start, stop)
        else:
            return super(Ordered, self).dimindex(arg)

    def _diminfo(self):
        name = self.name.capitalize(),
        vmin = self.x.min()
        vmax = self.x.max()
        return "%s [%s, %s]" % (name, vmin, vmax)

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : None | Dataset
            A dataset with variables describing cluster properties.
        """
        ds = Dataset()
        where = [np.flatnonzero(cluster) for cluster in x]
        ds['%s_min' % self.name] = Var([self.values[w[0]] for w in where])
        ds['%s_max' % self.name] = Var([self.values[w[-1]] for w in where])
        return ds


class Sensor(Dimension):
    """Dimension class for representing sensor information

    Parameters
    ----------
    locs : array-like
        list of (x, y, z) coordinates;
        ``x``: anterior - posterior,
        ``y``: left - right,
        ``z``: top - bottom
    names : list of str | None
        sensor names, same order as locs (optional)
    groups : None | dict
        Named sensor groups.
    sysname : None | str
        Name of the sensor system (only used for information purposes).
    proj2d:
        default 2d projection. For options, see the class documentation.
    connectivity : array (n_edges, 2)
        Sensor connectivity (optional).

    Attributes
    ----------
    channel_idx : dict
        Dictionary mapping channel names to indexes.
    locs : array, shape = (n_sensors, 3)
        Spatial position of all sensors.
    names : list of str
        Ordered list of sensor names.
    x, y, z : array, len = n_sensors
        X, y and z positions of the sensors.

    Notes
    -----
    The following are possible 2d-projections:

    ``None``:
        Just use horizontal coordinates
    ``'z root'``:
        the radius of each sensor is set to equal the root of the vertical
        distance from the top of the net.
    ``'cone'``:
        derive x/y coordinate from height based on a cone transformation
    ``'lower cone'``:
        only use cone for sensors with z < 0

    Examples
    --------
    >>> sensors = [(0,  0,   0),
                   (0, -.25, -.45)]
    >>> sensor_dim = Sensor(sensors, names=["Cz", "Pz"])
    """
    name = 'sensor'
    adjacent = False
    _proj_aliases = {'left': 'x-', 'right': 'x+', 'back': 'y-', 'front': 'y+',
                     'top': 'z+', 'bottom': 'z-'}

    def __init__(self, locs, names=None, groups=None, sysname=None,
                 proj2d='z root', connectivity=None):
        self.sysname = sysname
        self.default_proj2d = self._interpret_proj(proj2d)
        self._connectivity = connectivity

        # 'z root' transformation fails with 32-bit floats
        self.locs = locs = np.asarray(locs, dtype=np.float64)
        self.x = locs[:, 0]
        self.y = locs[:, 1]
        self.z = locs[:, 2]

        self.n = len(locs)

        if names is None:
            self.names_dist = names = [str(i) for i in xrange(self.n)]
        self.names = Datalist(names)
        self.channel_idx = {name: i for i, name in enumerate(self.names)}
        pf = os.path.commonprefix(self.names)
        if pf:
            n_pf = len(pf)
            short_names = {name[n_pf:]: i for i, name in enumerate(self.names)}
            self.channel_idx.update(short_names)

        # cache for transformed locations
        self._transformed = {}

        # groups
        self.groups = groups

    def __getstate__(self):
        state = {'proj2d': self.default_proj2d,
                 'groups': self.groups,
                 'locs': self.locs,
                 'names': self.names,
                 'sysname': self.sysname,
                 'connectivity': self._connectivity}
        return state

    def __setstate__(self, state):
        locs = state['locs']
        names = state['names']
        groups = state['groups']
        sysname = state['sysname']
        proj2d = state['proj2d']
        connectivity = state.get('connectivity', None)

        self.__init__(locs, names, groups, sysname, proj2d, connectivity)

    def __repr__(self):
        return "<Sensor n=%i, name=%r>" % (self.n, self.sysname)

    def __len__(self):
        return self.n

    def __eq__(self, other):
        "Based on having same sensor names"
        is_equal = (Dimension.__eq__(self, other)
                    and len(self) == len(other)
                    and all(n == no for n, no in zip(self.names, other.names)))
        return is_equal

    def __getitem__(self, index):
        index = self.dimindex(index)
        if np.isscalar(index):
            return self.names[index]
        else:
            int_index = np.arange(len(self))[index]
            if len(int_index) == 0:
                return None
            locs = self.locs[index]
            names = self.names[index]
            # TODO: groups
            return Sensor(locs, names, None, self.sysname, self.default_proj2d,
                          _subgraph_edges(self._connectivity, int_index))

    def _axis_formatter(self):
        return IndexFormatter(self.names)

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : None | Dataset
            A dataset with variables describing cluster properties.
        """
        return Dataset(('n_sensors', Var(x.sum(1))))

    def dimindex(self, arg):
        "Convert dimension indexes into numpy indexes"
        if isinstance(arg, basestring):
            return self.channel_idx[arg]
        elif isinstance(arg, Sensor):
            return np.array([self.names.index(name) for name in arg.names])
        else:
            return super(Sensor, self).dimindex(arg)

    def connectivity(self):
        """Retrieve the sensor connectivity

        Returns
        -------
        connetivity : array of int, (n_pairs, 2)
            array of sorted [src, dst] pairs, with all src < dts.

        See Also
        --------
        .set_connectivity() : define the connectivity
        .neighbors() : Neighboring sensors for each sensor in a dictionary.
        """
        if self._connectivity is None:
            raise RuntimeError("Sensor connectivity is not defined. Use "
                               "Sensor.set_connectivity().")
        else:
            return self._connectivity

    @classmethod
    def from_xyz(cls, path=None, **kwargs):
        """Create a Sensor instance from a text file with xyz coordinates
        """
        locs = []
        names = []
        with open(path) as f:
            l1 = f.readline()
            n = int(l1.split()[0])
            for line in f:
                elements = line.split()
                if len(elements) == 4:
                    x, y, z, name = elements
                    x = float(x)
                    y = float(y)
                    z = float(z)
                    locs.append((x, y, z))
                    names.append(name)
        assert len(names) == n
        return cls(locs, names, **kwargs)

    @classmethod
    def from_sfp(cls, path=None, **kwargs):
        """Create a Sensor instance from an sfp file
        """
        locs = []
        names = []
        for line in open(path):
            elements = line.split()
            if len(elements) == 4:
                name, x, y, z = elements
                x = float(x)
                y = float(y)
                z = float(z)
                locs.append((x, y, z))
                names.append(name)
        return cls(locs, names, **kwargs)

    @classmethod
    def from_lout(cls, path=None, transform_2d=None, **kwargs):
        """Create a Sensor instance from a *.lout file
        """
        kwargs['transform_2d'] = transform_2d
        locs = []
        names = []
        with open(path) as fileobj:
            fileobj.readline()
            for line in fileobj:
                w, x, y, t, f, name = line.split('\t')
                x = float(x)
                y = float(y)
                locs.append((x, y, 0))
                names.append(name)
        return cls(locs, names, **kwargs)

    def _interpret_proj(self, proj):
        if proj == 'default':
            return self.default_proj2d
        elif proj in self._proj_aliases:
            return self._proj_aliases[proj]
        elif proj is None:
            return 'z+'
        else:
            return proj

    def get_locs_2d(self, proj='default', extent=1, frame=0, invisible=True):
        """
        returns a sensor X location array, the first column reflecting the x,
        and the second column containing the y coordinate of each sensor.

        Parameters
        ----------
        proj : str
            How to transform 3d coordinates into a 2d map; see class
            documentation for options.
        extent : int
            coordinates will be scaled with minimum value 0 and maximum value
            defined by the value of ``extent``.
        frame : scalar
            Distance of the outermost points from 0 and ``extent`` (default 0).
        invisible : bool
            Return invisible sensors (sensors that would be hidden behind the
            head; default True).
        """
        proj = self._interpret_proj(proj)

        index = (proj, extent, frame)
        if index in self._transformed:
            locs2d = self._transformed[index]
        else:
            locs2d = self._make_locs_2d(proj, extent, frame)
            self._transformed[index] = locs2d

        if not invisible:
            visible = self._visible_sensors(proj)
            if visible is not None:
                return locs2d[visible]
        return locs2d

    @LazyProperty
    def _sphere_fit(self):
        """Fit the 3d sensor locations to a sphere

        Returns
        -------
        params : tuple
            Radius and center (r, cx, cy, cz).
        """
        locs = self.locs

        # error function
        def err(params):
            # params: [r, cx, cy, cz]
            out = np.sum((locs - params[1:]) ** 2, 1)
            out -= params[0] ** 2
            return out

        # initial guess of sphere parameters (radius and center)
        center_0 = np.mean(locs, 0)
        r_0 = np.mean(np.sqrt(np.sum((locs - center_0) ** 2, axis=1)))
        start_params = np.hstack((r_0, center_0))
        # do fit
        estimate, _ = leastsq(err, start_params)
        return tuple(estimate)

    def _make_locs_2d(self, proj, extent, frame):
        if proj in ('cone', 'lower cone', 'z root'):
            r, cx, cy, cz = self._sphere_fit

            # center the sensor locations based on the sphere and scale to
            # radius 1
            sphere_center = np.array((cx, cy, cz))
            locs3d = self.locs - sphere_center
            locs3d /= r

            # implement projection
            locs2d = np.copy(locs3d[:, :2])

            if proj == 'cone':
                locs2d[:, [0, 1]] *= (1 - locs3d[:, [2]])
            elif proj == 'lower cone':
                lower_half = locs3d[:, 2] < 0
                if any(lower_half):
                    locs2d[lower_half] *= (1 - locs3d[lower_half][:, [2]])
            elif proj == 'z root':
                z = locs3d[:, 2]
                z_dist = (z.max() + 0.01) - z  # distance form top, buffer so that top points don't stick together
                r = np.sqrt(z_dist)  # desired 2d radius
                r_xy = np.sqrt(np.sum(locs3d[:, :2] ** 2, 1))  # current radius in xy
                idx = (r_xy != 0)  # avoid zero division
                F = r[idx] / r_xy[idx]  # stretching Factor accounting for current r
                locs2d[idx, :] *= F[:, None]
        else:
            match = re.match('([xyz])([+-])', proj)
            if match:
                ax, sign = match.groups()
                if ax == 'x':
                    locs2d = np.copy(self.locs[:, 1:])
                    if sign == '-':
                        locs2d[:, 0] *= -1
                elif ax == 'y':
                    locs2d = np.copy(self.locs[:, [0, 2]])
                    if sign == '+':
                        locs2d[:, 0] *= -1
                elif ax == 'z':
                    locs2d = np.copy(self.locs[:, :2])
                    if sign == '-':
                        locs2d[:, 1] *= -1
            else:
                raise ValueError("invalid proj kwarg: %r" % proj)

        # correct extent
        if extent:
            locs2d -= np.min(locs2d, axis=0)  # move to bottom left
            locs2d /= (np.max(locs2d) / extent)  # scale to extent
            locs2d += (extent - np.max(locs2d, axis=0)) / 2  # center
            if frame:
                locs2d *= (1 - 2 * frame)
                locs2d += frame

        return locs2d

    def _topomap_outlines(self, proj):
        "outline argument for mne-python topomaps"
        proj = self._interpret_proj(proj)
        if proj in ('cone', 'lower cone', 'z root', 'z+'):
            return 'top'
        else:
            return None

    def _visible_sensors(self, proj):
        "Create an index for sensors that are visible under a given proj"
        proj = self._interpret_proj(proj)
        match = re.match('([xyz])([+-])', proj)
        if match:
            # logger.debug("Computing sensors visibility for %s" % proj)
            ax, sign = match.groups()

            # depth:  + = closer
            depth = self.locs[:, 'xyz'.index(ax)]
            if sign == '-':
                depth = -depth

            locs2d = self.get_locs_2d(proj)

            n_vertices = len(locs2d)
            all_vertices = np.arange(n_vertices)
            out = np.ones(n_vertices, bool)

            # find duplicate points
            # TODO OPT:  use pairwise distance
            x, y = np.where(cdist(locs2d, locs2d) == 0)
            duplicate_vertices = ((v1, v2) for v1, v2 in izip(x, y) if v1 < v2)
            for v1, v2 in duplicate_vertices:
                if depth[v1] > depth[v2]:
                    out[v2] = False
                    # logger.debug("%s is hidden behind %s" % (self.names[v2], self.names[v1]))
                else:
                    out[v1] = False
                    # logger.debug("%s is hidden behind %s" % (self.names[v1], self.names[v2]))
            use_vertices = all_vertices[out]  # use for hull check

            hull = ConvexHull(locs2d[use_vertices])
            hull_vertices = use_vertices[hull.vertices]

            # for each point:
            # find the closest point on the hull
            # determine whether it's in front or behind
            non_hull_vertices = np.setdiff1d(use_vertices, hull_vertices, True)

            hull_locs = locs2d[hull_vertices]
            non_hull_locs = locs2d[non_hull_vertices]
            dists = cdist(non_hull_locs, hull_locs)

            closest = np.argmin(dists, 1)
            hide_non_hull_vertices = depth[non_hull_vertices] < depth[hull_vertices][closest]
            hide_vertices = non_hull_vertices[hide_non_hull_vertices]
            # logger.debug("%s are hidden behind convex hull" % ' '.join(self.names[hide_vertices]))
            out[hide_vertices] = False
            return out
        else:
            return None

    def get_ROIs(self, base):
        """
        returns list if list of sensors, grouped according to closest
        spatial proximity to elements of base (=list of sensor ids)"

        """
        locs3d = self.locs
        # print loc3d
        base_locs = locs3d[base]
        ROI_dic = dict((i, [Id]) for i, Id in enumerate(base))
        for i, loc in enumerate(locs3d):
            if i not in base:
                dist = np.sqrt(np.sum((base_locs - loc) ** 2, 1))
                min_i = np.argmin(dist)
                ROI_dic[min_i].append(i)
        out = ROI_dic.values()
        return out

    def get_subnet_ROIs(self, ROIs, loc='first'):
        """
        returns new Sensor instance, combining groups of sensors in the old
        instance into single sensors in the new instance. All sensors for
        each element in ROIs are the basis for one new sensor.

        ! Only implemented for numeric indexes, not for boolean indexes !

        **parameters:**

        ROIs : list of lists of sensor ids
            each ROI defines one sensor in the new net
        loc : str
            'first': use the location of the first sensor of each ROI (default);
            'mean': use the mean location

        """
        names = []
        locs = np.empty((len(ROIs, 3)))
        for i, ROI in enumerate(ROIs):
            i = ROI[0]
            names.append(self.names[i])

            if loc == 'first':
                ROI_loc = self.locs[i]
            elif loc == 'mean':
                ROI_loc = self.locs[ROI].mean(0)
            else:
                raise ValueError("invalid value for loc (%s)" % loc)
            locs[i] = ROI_loc

        return Sensor(locs, names, sysname=self.sysname)

    def index(self, exclude=None, names=False):
        """Construct an index for specified sensors

        Parameters
        ----------
        exclude : None | list of str, int
            Sensors to exclude (by name or index).

        Returns
        -------
        index : numpy index
            Numpy index indexing good channels.
        """
        if exclude is None:
            return full_slice

        index = np.ones(len(self), dtype=bool)
        for idx in exclude:
            if isinstance(idx, str):
                idx = self.channel_idx[idx]
            else:
                idx = int(idx)

            index[idx] = False

        if names:
            index = self.names[index]

        return index

    def _normalize_sensor_names(self, names):
        "Process a user-input list of sensor names"
        valid_chs = set()
        missing_chs = set()
        for name in names:
            if isinstance(name, int):
                name = '%03i' % name

            if name.isdigit():
                if name in self.names:
                    valid_chs.add(name)
                    continue
                else:
                    name = 'MEG %s' % name

            if name in self.names:
                valid_chs.add(name)
            else:
                missing_chs.add(name)

        if missing_chs:
            msg = ("The following channels are not in the raw data: "
                   "%s" % ', '.join(sorted(missing_chs)))
            raise ValueError(msg)

        return sorted(valid_chs)

    def intersect(self, dim, check_dims=True):
        """Create a Sensor dimension that is the intersection with dim

        Parameters
        ----------
        dim : Sensor
            Sensor dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (e.g., channel locations). Default
            is ``True``. Set to ``False`` to intersect channels based on names
            only and ignore mismatch between locations for channels with the
            same name.

        Returns
        -------
        sensor : Sensor
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.name != dim.name:
            raise DimensionMismatchError("Dimensions don't match")

        n_self = len(self)
        names = set(self.names)
        names.intersection_update(dim.names)
        n_intersection = len(names)
        if n_intersection == n_self:
            return self
        elif n_intersection == len(dim.names):
            return dim

        names = sorted(names)
        idx = map(self.names.index, names)
        locs = self.locs[idx]
        if check_dims:
            idxd = map(dim.names.index, names)
            if not np.all(locs == dim.locs[idxd]):
                err = "Sensor locations don't match between dimension objects"
                raise ValueError(err)

        new = Sensor(locs, names, sysname=self.sysname,
                     proj2d=self.default_proj2d)
        return new

    def neighbors(self, connect_dist):
        """Find neighboring sensors.

        Parameters
        ----------
        connect_dist : scalar
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor.

        Returns
        -------
        neighbors : dict
            Dictionaries whose keys are sensor indices, and whose values are
            lists of neighbors represented as sensor indices.
        """
        nb = {}
        pd = pdist(self.locs)
        pd = squareform(pd)
        n = len(self)
        for i in xrange(n):
            d = pd[i, np.arange(n)]
            d[i] = d.max()
            idx = np.nonzero(d < d.min() * connect_dist)[0]
            nb[i] = idx

        return nb

    def set_connectivity(self, neighbors=None, connect_dist=None):
        """Define the sensor connectivity through neighbors or distance

        Parameters
        ----------
        neighbors : sequence of (str, str)
            A list of connections, all assumed to be bidirectional.
        connect_dist : None | scalar
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor.
            e.g., 1.75 or 1.6
        """
        pairs = set()
        if neighbors is not None and connect_dist is not None:
            raise TypeError("Can only specify either neighbors or connect_dist")
        elif connect_dist is None:
            for src, dst in neighbors:
                a = self.names.index(src)
                b = self.names.index(dst)
                if a < b:
                    pairs.add((a, b))
                else:
                    pairs.add((b, a))
        else:
            nb = self.neighbors(connect_dist)
            for k, vals in nb.iteritems():
                for v in vals:
                    if k < v:
                        pairs.add((k, v))
                    else:
                        pairs.add((v, k))

        self._connectivity = np.array(sorted(pairs), np.uint32)

    def set_sensor_positions(self, pos, names=None):
        """Set the sensor positions

        Parameters
        ----------
        pos : array (n_locations, 3) | MNE Montage
            Array with 3 columns describing sensor locations (x, y, and z), or
            an MNE Montage object describing the sensor layout.
        names : None | list of str
            If locations is an array, names should specify a name
            corresponding to each entry.
        """
        # MNE Montage
        if hasattr(pos, 'pos') and hasattr(pos, 'ch_names'):
            if names is not None:
                raise TypeError("Can't specify names parameter with Montage")
            names = pos.ch_names
            pos = pos.pos
        elif names is not None and len(names) != len(pos):
            raise ValueError("Mismatch between number of locations (%i) and "
                             "number of names (%i)" % (len(pos), len(names)))

        if names is not None:
            missing = [name for name in self.names if name not in names]
            if missing:
                raise ValueError("The following sensors are missing: %r" % missing)
            index = np.array([names.index(name) for name in self.names])
            pos = pos[index]
        elif len(pos) != len(self.locs):
            raise ValueError("If names are not specified pos must specify "
                             "exactly one position per channel")
        self.locs[:] = pos

    @property
    def values(self):
        return self.names


def as_sensor(obj):
    "Coerce to Sensor instance"
    if isinstance(obj, Sensor):
        return obj
    elif isinstance(obj, NDVar) and obj.has_dim('sensor'):
        return obj.sensor
    elif hasattr(obj, 'pos') and hasattr(obj, 'ch_names') and hasattr(obj, 'kind'):
        return Sensor(obj.pos, obj.ch_names, sysname=obj.kind)
    else:
        raise TypeError("Can't get sensors from %r" % (obj,))


def _point_graph(coords, dist_threshold):
    "Connectivity graph for points based on distance"
    n = len(coords)
    dist = pdist(coords)

    # construct vertex pairs corresponding to dist
    graph = np.empty((len(dist), 2), np.uint32)
    i0 = 0
    for vert, di in enumerate(xrange(n - 1, 0, -1)):
        i1 = i0 + di
        graph[i0:i1, 0] = vert
        graph[i0:i1, 1] = np.arange(vert + 1, n)
        i0 = i1

    return graph[dist < dist_threshold]


def _matrix_graph(matrix):
    "Create connectivity from matrix"
    coo = matrix.tocoo()
    assert np.all(coo.data)
    edges = {(min(a, b), max(a, b)) for a, b in izip(coo.col, coo.row) if a != b}
    return np.array(sorted(edges), np.uint32)


def _tri_graph(tris):
    """Create connectivity graph from triangles

    Parameters
    ----------
    tris : array_like, (n_tris, 3)
        Triangles.

    Returns
    -------
    edges : array (n_edges, 2)
        All edges between vertices of tris.
    """
    pairs = set()
    for tri in tris:
        a, b, c = sorted(tri)
        pairs.add((a, b))
        pairs.add((a, c))
        pairs.add((b, c))
    return np.array(sorted(pairs), np.uint32)


def _mne_tri_soure_space_graph(source_space, vertices_list):
    "Connectivity graph for a triangulated mne source space"
    i = 0
    graphs = []
    for ss, verts in izip(source_space, vertices_list):
        if len(verts) == 0:
            continue

        # graph for the whole source space
        src_vertices = ss['vertno']
        tris = ss['use_tris']
        graph = _tri_graph(tris)

        # select relevant edges
        if not np.array_equal(verts, src_vertices):
            if not np.all(np.in1d(verts, src_vertices)):
                raise RuntimeError("Not all vertices are in the source space")
            edge_in_use = np.logical_and(np.in1d(graph[:, 0], verts),
                                         np.in1d(graph[:, 1], verts))
            graph = graph[edge_in_use]

        # reassign vertex ids based on present vertices
        if len(verts) != verts.max() + 1:
            graph = (np.digitize(graph.ravel(), verts, True)
                     .reshape(graph.shape).astype(np.uint32))

        # account for index of previous source spaces
        if i > 0:
            graph += i
        i += len(verts)

        graphs.append(graph)
    return np.vstack(graphs)


class SourceSpace(Dimension):
    """MNE source space dimension.

    Parameters
    ----------
    vertno : list of array
        The vertex identities of the dipoles in the source space (left and
        right hemisphere separately).
    subject : str
        The mri-subject name.
    src : str
        The kind of source space used (e.g., 'ico-4').
    subjects_dir : str
        The path to the subjects_dir (needed to locate the source space
        file).
    parc : None | str
        Add a parcellation to the source space to identify vertex location.
        Only applies to ico source spaces, default is 'aparc'.
    connectivity : None | sparse matrix
        Cached source space connectivity.

    Notes
    -----
    besides numpy indexing, the following indexes are possible:

     - mne Label objects
     - 'lh' or 'rh' to select an entire hemisphere

    """
    name = 'source'
    adjacent = False
    _src_pattern = os.path.join('{subjects_dir}', '{subject}', 'bem',
                                '{subject}-{src}-src.fif')

    def __init__(self, vertno, subject=None, src=None, subjects_dir=None,
                 parc='aparc', connectivity=None):
        match = re.match("(ico|vol)-(\d)", src)
        if match:
            kind, grade = match.groups()
            grade = int(grade)
        else:
            raise ValueError("Unrecognized src value %r" % src)

        self.vertno = vertno
        self.subject = subject
        self.src = src
        self.kind = kind
        self.grade = grade
        self.subjects_dir = subjects_dir
        self._connectivity = connectivity
        self._n_vert = sum(len(v) for v in vertno)
        if kind == 'ico':
            self.lh_vertno = vertno[0]
            self.rh_vertno = vertno[1]
            self.lh_n = len(self.lh_vertno)
            self.rh_n = len(self.rh_vertno)
            self.set_parc(parc)

    def __getstate__(self):
        state = {'vertno': self.vertno, 'subject': self.subject,
                 'src': self.src, 'subjects_dir': self.subjects_dir,
                 'parc': self.parc}
        return state

    def __setstate__(self, state):
        vertno = state['vertno']
        subject = state['subject']
        src = state.get('src', None)
        parc = state.get('parc', None)
        subjects_dir = state.get('subjects_dir', None)
        self.__init__(vertno, subject, src, subjects_dir, parc)

    def __repr__(self):
        ns = ', '.join(str(len(v)) for v in self.vertno)
        return "<SourceSpace [%s], %r, %r>" % (ns, self.subject, self.src)

    def __len__(self):
        return self._n_vert

    def __eq__(self, other):
        is_equal = (Dimension.__eq__(self, other)
                    and self.subject == other.subject
                    and len(self) == len(other)
                    and all(np.array_equal(s, o) for s, o in
                            izip(self.vertno, other.vertno)))
        return is_equal

    def __getitem__(self, index):
        arange = np.arange(len(self))
        int_index = arange[index]
        bool_index = np.in1d(arange, int_index, True)

        # vertno
        boundaries = np.cumsum(tuple(chain((0,), (len(v) for v in self.vertno))))
        vertno = [v[bool_index[boundaries[i]:boundaries[i + 1]]]
                  for i, v in enumerate(self.vertno)]

        # parc
        if self.parc is None:
            parc = None
        else:
            parc = self.parc[index]

        dim = SourceSpace(vertno, self.subject, self.src, self.subjects_dir,
                          parc, _subgraph_edges(self._connectivity, int_index))
        return dim

    def _axis_formatter(self):
        return FormatStrFormatter('%i')

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : Dataset
            A dataset with variables describing cluster properties along this
            dimension: "n_sources".
        """
        if np.any(np.sum(x, 1) == 0):
            raise ValueError("Empty cluster")

        ds = Dataset()

        # no clusters
        if len(x) == 0:
            ds['n_sources'] = Var([])
            ds['hemi'] = Factor([])
            if self.parc is not None:
                ds['location'] = Factor([])
            return ds

        # n sources
        ds['n_sources'] = Var(x.sum(1))

        if self.kind == 'vol':
            return ds

        # hemi
        hemis = []
        for x_ in x:
            where = np.flatnonzero(x_)
            src_in_lh = (where < self.lh_n)
            if np.all(src_in_lh):
                hemis.append('lh')
            elif np.any(src_in_lh):
                hemis.append('bh')
            else:
                hemis.append('rh')
        ds['hemi'] = Factor(hemis)

        # location
        if self.parc is not None:
            locations = []
            for x_ in x:
                parc_entries = self.parc[x_]
                argmax = np.argmax(np.bincount(parc_entries.x))
                location = parc_entries[argmax]
                locations.append(location)
            ds['location'] = Factor(locations)

        return ds

    def _diminfo(self):
        ns = ', '.join(str(len(v)) for v in self.vertno)
        return "SourceSpace (MNE) [%s], %r, %r>" % (ns, self.subject, self.src)

    def connectivity(self, disconnect_parc=False):
        """Create source space connectivity

        Parameters
        ----------
        disconnect_parc : bool
            Reduce connectivity to label-internal connections.

        Returns
        -------
        connetivity : array of int, (n_pairs, 2)
            array of sorted [src, dst] pairs, with all src < dts.
        """
        if self._connectivity is None:
            if self.src is None or self.subject is None or self.subjects_dir is None:
                err = ("In order for a SourceSpace dimension to provide "
                       "connectivity information it needs to be initialized with "
                       "src, subject and subjects_dir parameters")
                raise ValueError(err)

            src = self.get_source_space()
            if self.kind == 'vol':
                coords = src[0]['rr'][self.vertno[0]]
                dist_threshold = self.grade * 0.0011
                connectivity = _point_graph(coords, dist_threshold)
            elif self.kind == 'ico':
                connectivity = _mne_tri_soure_space_graph(src, self.vertno)
            else:
                msg = "Connectivity for %r source space" % self.kind
                raise NotImplementedError(msg)

            if connectivity.max() >= len(self):
                raise RuntimeError("SourceSpace connectivity failed")
            self._connectivity = connectivity
        else:
            connectivity = self._connectivity

        if disconnect_parc:
            parc = self.parc
            if parc is None:
                raise RuntimeError("SourceSpace has no parcellation (use "
                                   ".set_parc())")
            idx = np.array([parc[s] == parc[d] for s, d in connectivity])
            connectivity = connectivity[idx]

        return connectivity

    def circular_index(self, seeds, extent=0.05, name="globe"):
        """Returns an index into all vertices within extent of seed

        Parameters
        ----------
        seeds : array_like, (3,) | (n, 3)
            Seed location(s) around which to build index.
        extent :

        Returns
        -------
        roi : NDVar, ('source',)
            Index into the spherical area around seeds.
        """
        seeds = np.atleast_2d(seeds)
        dist = cdist(self.coordinates, seeds)
        mindist = np.min(dist, 1)
        x = mindist < extent
        dims = (self,)
        info = {'seeds': seeds, 'extent': extent}
        return NDVar(x, dims, info, name)

    @LazyProperty
    def coordinates(self):
        sss = self.get_source_space()
        coords = (ss['rr'][v] for ss, v in izip(sss, self.vertno))
        coords = np.vstack(coords)
        return coords

    def dimindex(self, arg):
        if isinstance(arg, (mne.Label, mne.label.BiHemiLabel)):
            return self._dimindex_label(arg)
        elif isinstance(arg, basestring):
            if arg == 'lh':
                if self.lh_n:
                    return slice(None, self.lh_n)
                else:
                    raise IndexError("lh is empty")
            elif arg == 'rh':
                if self.rh_n:
                    return slice(self.lh_n, None)
                else:
                    raise IndexError("rh is empty")
            else:
                return self._dimindex_label(arg)
        elif isinstance(arg, SourceSpace):
            sv = self.vertno
            ov = arg.vertno
            if all(np.array_equal(s, o) for s, o in izip(sv, ov)):
                return full_slice
            else:
                idxs = tuple(np.in1d(s, o, True) for s, o in izip(sv, ov))
                index = np.hstack(idxs)
                return index
        elif isinstance(arg, SEQUENCE_TYPES):
            return self.parc.isin(arg)
        else:
            return super(SourceSpace, self).dimindex(arg)

    def _dimindex_label(self, label):
        if isinstance(label, basestring):
            if self.parc is None:
                raise RuntimeError("SourceSpace has no parcellation")
            elif label not in self.parc:
                err = ("SourceSpace parcellation has no label called %r"
                       % label)
                raise KeyError(err)
            idx = self.parc == label
        elif label.hemi == 'both':
            lh_idx = self._dimindex_hemilabel(label.lh)
            rh_idx = self._dimindex_hemilabel(label.rh)
            idx = np.hstack((lh_idx, rh_idx))
        else:
            idx = np.zeros(len(self), dtype=np.bool8)
            idx_part = self._dimindex_hemilabel(label)
            if label.hemi == 'lh':
                idx[:self.lh_n] = idx_part
            elif label.hemi == 'rh':
                idx[self.lh_n:] = idx_part
            else:
                err = "Unknown value for label.hemi: %s" % repr(label.hemi)
                raise ValueError(err)

        return idx

    def _dimindex_hemilabel(self, label):
        if label.hemi == 'lh':
            stc_vertices = self.vertno[0]
        else:
            stc_vertices = self.vertno[1]
        idx = np.in1d(stc_vertices, label.vertices, True)
        return idx

    def distances(self):
        "Surface distances between source space vertices"
        dist = -np.ones((self._n_vert, self._n_vert))
        sss = self.get_source_space()
        i0 = 0
        for vertices, ss in izip(self.vertno, sss):
            if ss['dist'] is None:
                raise RuntimeError("Source-space does not contain distances")
            i = i0 + len(vertices)
            dist[i0:i, i0:i] = ss['dist'][vertices, vertices[:, None]].toarray()
            i0 = i
        return dist

    def get_source_space(self):
        "Read the corresponding MNE source space"
        path = self._src_pattern.format(subjects_dir=self.subjects_dir,
                                        subject=self.subject, src=self.src)
        src = mne.read_source_spaces(path)
        return src

    def index_for_label(self, label):
        """Returns the index for a label

        Parameters
        ----------
        label : str | Label | BiHemiLabel
            The name of a region in the current parcellation, or a Label object
            (as created for example by mne.read_label). If the label does not
            match any sources in the SourceEstimate, a ValueError is raised.

        Returns
        -------
        index : NDVar of bool
            Index into the source space dim that corresponds to the label.
        """
        idx = self._dimindex_label(label)
        if isinstance(label, basestring):
            name = label
        else:
            name = label.name
        return NDVar(idx, (self,), {}, name)

    def intersect(self, other, check_dims=True):
        """Create a Source dimension that is the intersection with dim

        Parameters
        ----------
        dim : Source
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : Source
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.subject != other.subject:
            raise ValueError("Source spaces can not be compared because they "
                             "are defined on different MRI subjects (%s, %s). "
                             "Consider using eelbrain.morph_source_space()."
                             % (self.subject, other.subject))
        elif self.src != other.src:
            raise ValueError("Source spaces can not be compared because they "
                             "are defined with different spatial decimation "
                             "parameters (%s, %s)." % (self.src, other.src))
        elif self.subjects_dir != other.subjects_dir:
            raise ValueError("Source spaces can not be compared because they "
                             "have differing subjects_dir parameters:\n%s\n%s"
                             % (self.subjects_dir, other.subjects_dir))

        index = np.hstack(np.in1d(s, o) for s, o
                          in izip(self.vertno, other.vertno))
        return self[index]

    def _mask_label(self):
        "Create a Label that masks the areas not covered in this SourceSpace"
        lh = rh = None
        sss = self.get_source_space()
        if self.lh_n:
            lh_verts = np.setdiff1d(sss[0]['vertno'], self.lh_vertno)
            if len(lh_verts):
                lh = mne.Label(lh_verts, hemi='lh', color=(0, 0, 0)).fill(sss, 'unknown')
        if self.rh_n:
            rh_verts = np.setdiff1d(sss[1]['vertno'], self.rh_vertno)
            if len(rh_verts):
                rh = mne.Label(rh_verts, hemi='rh', color=(0, 0, 0)).fill(sss, 'unknown')
        return lh, rh

    def set_parc(self, parc):
        """Set the source space parcellation

        Parameters
        ----------
        parc : None | str | Factor
            Add a parcellation to the source space to identify vertex location.
            Can be specified as Factor assigning a label to each source, or a
            string specifying a freesurfer parcellation (stored as *.annot
            files with the MRI). Only applies to ico source spaces, default is
            'aparc'.
        """
        if parc is None:
            parc_ = None
        elif isfactor(parc):
            if len(parc) != len(self):
                raise ValueError("Wrong length (%i)" % len(parc))
            parc_ = parc
        elif isinstance(parc, basestring):
            if self.kind == 'ico':
                fname = os.path.join(self.subjects_dir, self.subject, 'label', '%%s.%s.annot' % parc)
                vert_codes_lh, ctab_lh, names_lh = read_annot(fname % 'lh')
                vert_codes_rh, ctab_rh, names_rh = read_annot(fname % 'rh')
                x_lh = vert_codes_lh[self.lh_vertno]
                x_rh = vert_codes_rh[self.rh_vertno]
                x_rh += x_lh.max() + 1
                names = chain(('%s-lh' % name for name in names_lh),
                              ('%s-rh' % name for name in names_rh))
                parc_ = Factor(np.hstack((x_lh, x_rh)), parc,
                               labels={i: name for i, name in enumerate(names)})
            else:
                raise NotImplementedError
        else:
            raise ValueError("Parc needs to be string, got %s" % repr(parc))

        self.parc = parc_

    @property
    def values(self):
        raise NotImplementedError


_uts_tol = 0.000001  # tolerance for deciding if time values are equal

class UTS(Dimension):
    """Dimension object for representing uniform time series

    Parameters
    ----------
    tmin : scalar
        First time point (inclusive).
    tstep : scalar
        Time step between samples.
    nsamples : int
        Number of samples.

    Notes
    -----
    Special indexing:

    (tstart, tstop) : tuple
        Restrict the time to the indicated window (either end-point can be
        None).

    """
    name = 'time'
    unit = 's'
    _axis_unit = 'ms'

    def __init__(self, tmin, tstep, nsamples):
        self.tmin = tmin
        self.tstep = tstep
        self.nsamples = nsamples = int(nsamples)
        self.x = self.times = tmin + np.arange(nsamples) * tstep
        self.tmax = self.times[-1]
        self.tstop = self.tmin + tstep * nsamples

    @classmethod
    def from_int(cls, first, last, sfreq):
        """Create a UTS dimension from sample index and sampling frequency

        Parameters
        ----------
        first : int
            Index of the first sample, relative to 0.
        last : int
            Index of the last sample, relative to 0.
        sfreq : scalar
            Sampling frequency, in Hz.
        """
        tmin = first / sfreq
        nsamples = last - first + 1
        tstep = 1. / sfreq
        return cls(tmin, tstep, nsamples)

    def __getstate__(self):
        state = {'tmin': self.tmin,
                 'tstep': self.tstep,
                 'nsamples': self.nsamples}
        return state

    def __setstate__(self, state):
        tmin = state['tmin']
        tstep = state['tstep']
        nsamples = state['nsamples']
        self.__init__(tmin, tstep, nsamples)

    def __repr__(self):
        return "UTS(%s, %s, %s)" % (self.tmin, self.tstep, self.nsamples)

    def _axis_im_extent(self):
        return self.tmin - 0.5 * self.tstep, self.tmax + 0.5 * self.tstep

    def _axis_formatter(self):
        return FuncFormatter(lambda x, pos: '%i' % round(1e3 * x))

    def _diminfo(self):
        name = self.name.capitalize()
        tmax = self.times[-1] + self.tstep
        sfreq = 1. / self.tstep
        info = '%s %.3f - %.3f s, %s Hz' % (name, self.tmin, tmax, sfreq)
        return info

    def __len__(self):
        return len(self.times)

    def __eq__(self, other):
        is_equal = (Dimension.__eq__(self, other)
                    and self.tmin == other.tmin
                    and self.tstep == other.tstep
                    and self.nsamples == other.nsamples)
        return is_equal

    def __contains__(self, index):
        return self.tmin - self.tstep < index < self.tstop

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.times[index]
        elif not isinstance(index, slice):
            # convert index to slice
            index = np.arange(len(self))[index]
            start = index[0]
            steps = np.unique(np.diff(index))
            if len(steps) > 1:
                raise NotImplementedError("non-uniform time series")
            step = steps[0]
            stop = index[-1] + step
            index = slice(start, stop, step)

        if isinstance(index, slice):
            if index.start is None:
                start = 0
            else:
                start = index.start

            if index.stop is None:
                stop = len(self)
            else:
                stop = index.stop

            tmin = self.times[start]
            nsamples = stop - start
            if nsamples < 0:
                raise IndexError("Time index out of range: %s." % repr(index))

            if index.step is None:
                tstep = self.tstep
            else:
                tstep = self.tstep * index.step
        else:
            raise TypeError("Unsupported index: %r" % index)

        return UTS(tmin, tstep, nsamples)

    def _cluster_bounds(self, x):
        """Cluster start and stop in samples

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.
        """
        # find indices of cluster extent
        row, col = np.nonzero(x)
        try:
            ts = [col[row == i][[0, -1]] for i in xrange(len(x))]
        except IndexError:
            raise ValueError("Empty cluster")
        ts = np.array(ts)
        return ts

    def _cluster_properties(self, x):
        """Find cluster properties for this dimension

        Parameters
        ----------
        x : array of bool, (n_clusters, len(self))
            The cluster extents, with different clusters stacked along the
            first axis.

        Returns
        -------
        cluster_properties : Dataset
            A dataset with variables describing cluster properties along this
            dimension: "tstart", "tstop", "duration".
        """
        ds = Dataset()

        # no clusters
        if len(x) == 0:
            ds['tstart'] = Var([])
            ds['tstop'] = Var([])
            ds['duration'] = Var([])
            return ds

        # create time values
        bounds = self._cluster_bounds(x)
        tmin = self.times[bounds[:, 0]]
        tmax = self.times[bounds[:, 1]]
        ds['tstart'] = Var(tmin)
        ds['tstop'] = Var(tmax + self.tstep)
        ds['duration'] = ds.eval("tstop - tstart")
        return ds

    def dimindex(self, arg):
        if np.isscalar(arg):
            i = int(round((arg - self.tmin) / self.tstep))
            if i < 0 or i >= self.nsamples:
                raise ValueError("Time index %s out of range (%s, %s)"
                                 % (arg, self.tmin, self.tmax))
            return i
        elif isinstance(arg, UTS):
            if self.tmin == arg.tmin:
                start = None
                stop = arg.nsamples
            elif arg.tmin < self.tmin:
                err = ("The index time dimension starts before the reference "
                       "time dimension")
                raise DimensionMismatchError(err)
            else:
                start_float = (arg.tmin - self.tmin) / self.tstep
                start = int(round(start_float))
                if abs(start_float - start) > _uts_tol:
                    err = ("The index time dimension contains values not "
                           "contained in the reference time dimension")
                    raise DimensionMismatchError(err)
                stop = start + arg.nsamples

            if self.tstep == arg.tstep:
                step = None
            elif self.tstep > arg.tstep:
                err = ("The index time dimension has a higher sampling rate "
                       "than the reference time dimension")
                raise DimensionMismatchError(err)
            else:
                step_float = arg.tstep / self.tstep
                step = int(round(step_float))
                if abs(step_float - step) > _uts_tol:
                    err = ("The index time dimension contains values not "
                           "contained in the reference time dimension")
                    raise DimensionMismatchError(err)

            if stop == self.nsamples:
                stop = None

            return slice(start, stop, step)
        elif isinstance(arg, tuple):
            if len(arg) != 2:
                raise ValueError("Tuple indexes signify intervals for uniform "
                                 "time-series (UTS) dimension and need to be "
                                 "exactly of length 2 (got %s)" % repr(arg))
            tstart, tstop = arg
            return self._slice(tstart, tstop)
        else:
            return super(UTS, self).dimindex(arg)

    def index(self, time, rnd='closest'):
        """Find the index for a time point

        Parameters
        ----------
        time : scalar
            Time point for which to find an index.
        rnd : 'down' | 'closest' | 'up'
            Rounding: how to handle time values that do not have an exact
            match. Round 'up', 'down', or to the 'closest' neighbor.

        Returns
        -------
        i : int
            Index of ``time``, rounded according to ``rnd``.
        """
        if rnd == 'closest':
            return int(round((time - self.tmin) / self.tstep))
        else:
            i, _ = find_time_point(self.times, time, rnd)
            return i

    def intersect(self, dim, check_dims=True):
        """Create a UTS dimension that is the intersection with dim

        Parameters
        ----------
        dim : UTS
            Dimension to intersect with.
        check_dims : bool
            Check dimensions for consistency (not applicaple to this subclass).

        Returns
        -------
        intersection : UTS
            The intersection with dim (returns itself if dim and self are
            equal)
        """
        if self.tstep == dim.tstep:
            tstep = self.tstep
        else:
            raise NotImplementedError("Intersection of UTS with unequal tstep :(")

        tmin_diff = abs(self.tmin - dim.tmin) / tstep
        if abs(tmin_diff - round(tmin_diff)) > _uts_tol:
            raise DimensionMismatchError("UTS dimensions have different times")
        tmin = max(self.tmin, dim.tmin)

        tmax = min(self.tmax, dim.tmax)
        nsamples = int(round((tmax - tmin) / tstep)) + 1
        if nsamples <= 0:
            raise DimensionMismatchError("UTS dimensions don't overlap")

        return UTS(tmin, tstep, nsamples)

    def _slice(self, tstart, tstop):
        "Create a slice into the time axis"
        if (tstart is not None) and (tstop is not None) and (tstart >= tstop):
            raise ValueError("tstart must be smaller than tstop")

        if tstart is None:
            start = None
        elif tstart <= self.tmin - self.tstep:
            raise ValueError("Value out of range: tstart=%s" % tstart)
        else:
            start_float = (tstart - self.tmin) / self.tstep
            start = int(start_float)
            if start_float - start > 0.000001:
                start += 1

        if tstop is None:
            stop = None
        elif tstop > self.tstop:
            raise ValueError("Value out of range: tstop=%s" % tstop)
        else:
            stop_float = (tstop - self.tmin) / self.tstep
            stop = int(stop_float)
            if stop_float - stop > 0.000001:
                stop += 1

        s = slice(start, stop)
        return s

    @property
    def values(self):
        return self.times


def intersect_dims(dims1, dims2, check_dims=True):
    """Find the intersection between two multidimensional spaces

    Parameters
    ----------
    dims1, dims2 : tuple of dimension objects
        Two spaces involving the same dimensions with overlapping values.
    check_dims : bool
        Check dimensions for consistency (e.g., channel locations in a Sensor
        dimension). Default is ``True``. Set to ``False`` to ignore non-fatal
        mismatches.

    Returns
    -------
    dims : tuple of Dimension objects
        Intersection of dims1 and dims2.
    """
    return tuple(d1.intersect(d2, check_dims=check_dims) for d1, d2 in zip(dims1, dims2))


# ---NDVar functions---

def corr(x, dim='sensor', obs='time', name=None):
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
        NDVar with correlation coefficients.
    """
    dim_obj = x.get_dim(dim)

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
    return NDVar(y, (dim_obj,), info, name)


def fft(ndvar, name=None):
    axis = ndvar.get_axis('time')
    uts = ndvar.get_dim('time')
    x = np.abs(np.fft.rfft(ndvar.x, axis=axis))
    freq = Ordered('frequency', np.fft.rfftfreq(len(uts), uts.tstep), 'Hz')
    dims = ndvar.dims[:axis] + (freq,) + ndvar.dims[axis + 1:]
    info = cs.set_info_cs(ndvar.info, cs.default_info('Amplitude'))
    return NDVar(x, dims, info, name)


def cwt_morlet(Y, freqs, use_fft=True, n_cycles=3.0, zero_mean=False,
               out='magnitude'):
    """Time frequency decomposition with Morlet wavelets (mne-python)

    Parameters
    ----------
    Y : NDVar with time dimension
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

    if not Y.get_axis('time') == Y.ndim - 1:
        raise NotImplementedError
    x = Y.x
    x = x.reshape((np.prod(x.shape[:-1]), x.shape[-1]))
    Fs = 1. / Y.time.tstep
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

    new_shape = Y.x.shape[:-1]
    dims = Y.dims[:-1]
    if fdim is not None:
        new_shape += (len(freqs),)
        dims += (fdim,)
    new_shape += Y.x.shape[-1:]
    dims += Y.dims[-1:]

    x = x.reshape(new_shape)
    info = cs.set_info_cs(Y.info, cs.default_info('A'))
    out = NDVar(x, dims, info, Y.name)
    return out


def resample(data, sfreq, npad=100, window='boxcar'):
    """Resample an NDVar with 'time' dimension after properly filtering it

    Parameters
    ----------
    data : NDVar
        Ndvar which should be resampled.
    sfreq : scalar
        New sampling frequency.
    npad : int
        Number of samples to use at the beginning and end for padding.
    window : string | tuple
        See scipy.signal.resample for description.

    Notes
    -----
    requires mne-python
    """
    axis = data.get_axis('time')
    old_sfreq = 1.0 / data.time.tstep
    x = mne.filter.resample(data.x, sfreq, old_sfreq, npad, axis, window)
    tstep = 1. / sfreq
    time = UTS(data.time.tmin, tstep, x.shape[axis])
    dims = data.dims[:axis] + (time,) + data.dims[axis + 1:]
    return NDVar(x, dims=dims, info=data.info, name=data.name)
