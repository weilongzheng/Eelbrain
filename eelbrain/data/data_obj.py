# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
'''
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

'''

from __future__ import division

import collections
import itertools
import cPickle as pickle
import operator
import os
import re
from warnings import warn

import mne
import numpy as np
from numpy import dot
import scipy
import scipy.stats
from scipy.linalg import inv
from scipy.optimize import leastsq
from scipy.sparse import coo_matrix
from scipy.spatial.distance import pdist, squareform

from .. import fmtxt
from .. import ui
from ..utils import LazyProperty, natsorted
from . import colorspaces as cs
from .stats import cihw


__all__ = ('Datalist', 'Dataset', 'Factor', 'Var', 'NDVar', 'combine', 'align',
           'align1', 'cwt_morlet', 'resample', 'cellname', 'Celltable')

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


_pickled_ds_wildcard = ("Pickled Dataset (*.pickled)", '*.pickled')
_tex_wildcard = ("TeX (*.tex)", '*.tex')
_tsv_wildcard = ("Plain Text Tab Separated Values (*.txt)", '*.txt')
_txt_wildcard = ("Plain Text (*.txt)", '*.txt')


class DimensionMismatchError(Exception):
    pass



def _effect_eye(n):
    """
    Returns effect coding for n categories. E.g.::

        >>> _effect_eye(4)
        array([[ 1,  0,  0],
               [ 0,  1,  0],
               [ 0,  0,  1],
               [-1, -1, -1]])

    """
    X = np.empty((n, n - 1), dtype=np.int8)
    X[:n - 1] = np.eye(n - 1, dtype=np.int8)
    X[n - 1] = -1
    return X


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
    else:
        return str(cell)


def rank(A, tol=1e-8):
    """
    Rank of a matrix, from
    http://mail.scipy.org/pipermail/numpy-discussion/2008-February/031218.html

    """
    s = np.linalg.svd(A, compute_uv=0)
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


def isbalanced(X):
    """
    returns True if X is balanced, False otherwise.

    X : categorial
        categorial model (Factor or Interaction)

    """
    if ismodel(X):
        return all(isbalanced(e) for e in X.effects)
    else:
        ns = (np.sum(X == c) for c in X.cells)
        return len(np.unique(ns)) <= 1

def iscategorial(Y):
    "factors as well as interactions are categorial"
    if isfactor(Y):
        return True
    elif isinteraction(Y):
        return Y.is_categorial
    else:
        return False

def isdataobject(Y):
    dataob = ["model", "var", "ndvar", "factor", "interaction", "nonbasic",
              "nested", "list"]
    return hasattr(Y, '_stype_') and  Y._stype_ in dataob

def isdataset(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == 'dataset'

def iseffect(Y):
    effectnames = ["factor", "var", "interaction", "nonbasic", "nested"]
    return hasattr(Y, '_stype_') and  Y._stype_ in effectnames

def isdatalist(Y, contains=None, test_all=True):
    """Test whether Y is a Datalist instance

    Parameters
    ----------
    contains : None | class
        Test whether the content is instances of a specific class.
    test_all : bool
        If contains is provided, test all items' class (otherwise just test the
        first item).
    """
    is_dl = isinstance(Y, Datalist)
    if is_dl and contains:
        if test_all:
            is_dl = all(isinstance(item, contains) for item in Y)
        else:
            is_dl = isinstance(Y[0], contains)
    return is_dl

def isfactor(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "factor"

def isinteraction(Y):
    return hasattr(Y, '_stype_') and Y._stype_ == "interaction"

def ismodel(X):
    return hasattr(X, '_stype_') and X._stype_ == "model"

def isnested(Y):
    "Determine whether Y is nested"
    return hasattr(Y, '_stype_') and Y._stype_ == "nested"

def isnestedin(item, item2):
    "Returns True if item is nested in item2, False otherwise"
    if hasattr(item, 'nestedin'):
        return item.nestedin and (item2 in find_factors(item.nestedin))
    else:
        return False

def isndvar(Y):
    "Determine whether Y is an NDVar"
    return hasattr(Y, '_stype_') and Y._stype_ == "ndvar"

def isnumeric(Y):
    "Determine wether Y is numeric (a Var or an NDVar)"
    return hasattr(Y, '_stype_') and Y._stype_ in ["ndvar", "var"]

def isuv(Y):
    "Determine whether Y is univariate (a Var or a Factor)"
    return hasattr(Y, '_stype_') and Y._stype_ in ["factor", "var"]

def isvar(Y):
    "Determine whether Y is a Var"
    return hasattr(Y, '_stype_') and Y._stype_ == "var"

def isboolvar(Y):
    "Determine whether Y is a Var whose data type is boolean"
    if not isvar(Y):
        return False
    isbool = Y.x.dtype.kind == 'b'
    return isbool

def isintvar(Y):
    "Determine whether Y is a Var whose data type is integer"
    if not isvar(Y):
        return False
    # http://stackoverflow.com/a/934652/166700
    isint = Y.x.dtype.kind in 'iu'
    return isint


def hasrandom(Y):
    """True if Y is or contains a random effect, False otherwise"""
    if isfactor(Y):
        return Y.random
    elif isinteraction(Y):
        for e in Y.base:
            if isfactor(e) and e.random:
                return True
    elif ismodel(Y):
        return any(map(hasrandom, Y.effects))
    return False


def ascategorial(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Parameter was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if iscategorial(Y):
        pass
    else:
        Y = asfactor(Y)

    if sub is not None:
        return Y[sub]
    else:
        return Y

def asdataobject(Y, sub=None, ds=None):
    "Convert to any data object or numpy array."
    if isinstance(Y, str):
        if ds is None:
            err = ("Data object was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if isdataobject(Y):
        pass
    elif isinstance(Y, np.ndarray):
        pass
    else:
        Y = Datalist(Y)

    if sub is not None:
        Y = Y[sub]
    return Y

def asfactor(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Factor was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if isfactor(Y):
        pass
    elif hasattr(Y, 'as_factor'):
        Y = Y.as_factor()
    else:
        Y = Factor(Y)

    if sub is not None:
        return Y[sub]
    else:
        return Y

def asmodel(X, sub=None, ds=None):
    if isinstance(X, str):
        if ds is None:
            err = ("Model was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        X = ds.eval(X)

    if ismodel(X):
        pass
    elif isinstance(X, (list, tuple)):
        X = Model(*X)
    else:
        X = Model(X)

    if sub is not None:
        return X[sub]
    else:
        return X

def asndvar(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Ndvar was specified as string, but no Dataset was "
                   "specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    # convert MNE objects
    if isinstance(Y, mne.Epochs):
        from .load.fiff import epochs_ndvar
        Y = epochs_ndvar(Y)
    elif isinstance(Y, mne.fiff.Evoked):
        from .load.fiff import evoked_ndvar
        Y = evoked_ndvar(Y)

    if not isndvar(Y):
        raise TypeError("NDVar required")

    if sub is not None:
        return Y[sub]
    else:
        return Y

def asnumeric(Y, sub=None, ds=None):
    "Var, NDVar"
    if isinstance(Y, str):
        if ds is None:
            err = ("Numeric argument was specified as string, but no Dataset "
                   "was specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if not isnumeric(Y):
        raise TypeError("Numeric argument required (Var or NDVar)")

    if sub is not None:
        return Y[sub]
    else:
        return Y

def assub(sub, ds=None):
    "Interpret the sub argument."
    if isinstance(sub, str):
        if ds is None:
            err = ("the sub parameter was specified as string, but no Dataset "
                   "was specified")
            raise TypeError(err)
        sub = ds.eval(sub)
    return sub

def asvar(Y, sub=None, ds=None):
    if isinstance(Y, str):
        if ds is None:
            err = ("Var was specified as string, but no Dataset was specified")
            raise TypeError(err)
        Y = ds.eval(Y)

    if isvar(Y):
        pass
    else:
        Y = Var(Y)

    if sub is not None:
        return Y[sub]
    else:
        return Y


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
        return Factor([''], rep=n, name=name)
    elif isvar(obj):
        return Var(np.empty(n) * np.NaN, name=name)
    elif isndvar(obj):
        shape = (n,) + obj.shape[1:]
        return NDVar(np.empty(shape) * np.NaN, dims=obj.dims, name=name)
    elif isdatalist(obj):
        return Datalist([None] * n, name=name)
    else:
        err = "Type not supported: %s" % type(obj)
        raise TypeError(err)


# --- sorting ---

def align(d1, d2, out='data', i1='index', i2='index'):
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
    i1, i2 : str | array-like (dtype=int)
        Indexes for cases in d1 and d2.
        If d1 and d2 are datasets, i1 and i2 can be keys for variables in d1 and
        d2. If d1 an d2 are other data objects, i1 and i2 have to be actual indices
        (array-like)
    out : 'data' | 'index'
        **'data'**: returns the two aligned data objects. **'index'**: returns two
        indices index1 and index2 which can be used to align the datasets with
        ``ds1[index1]; ds2[index2]``.


    Examples
    --------

    see examples/datasets/align.py

    """
    i1 = asvar(i1, ds=d1)
    i2 = asvar(i2, ds=d2)

    if len(i1) > len(i1.values):
        raise ValueError('Non-unique index in i1 for %r' % d1.name)
    if len(i2) > len(i2.values):
        raise ValueError('Non-unique index in i2 for %r' % d2.name)

    idx1 = []
    idx2 = []
    for i, idx in enumerate(i1):
        if idx in i2:
            idx1.append(i)
            where2 = i2.index(idx)[0]
            idx2.append(where2)

    if out == 'data':
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
    idx : index array, len = n2
        index to which d should be aligned.
    d_idx : str | index array, len = n1
        Indices of cases in d. If d is a Dataset, d_idx can be a name in d.
    out : 'data' | 'index' | 'bool'
        Return a subset of d, an array of numerical indices into d, or a
        boolean array into d.
    """
    idx = asvar(idx)
    d_idx = asvar(d_idx, ds=d)

    where = np.in1d(d_idx, idx, True)
    if out == 'bool':
        return where
    elif out == 'index':
        return np.nonzero(where)
    elif out == 'data':
        return d[where]
    else:
        ValueError("Invalid value for out parameter: %r" % out)


class Celltable(object):
    """Divide Y into cells defined by X.

    Attributes
    ----------
    .Y, .X,
        Y and X after sub was applied.
    .sub, .match:
        Input arguments.
    .cells : list of str
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
                 cat=None, ds=None):
        """Divide Y into cells defined by X.

        Parameters
        ----------
        Y : data-object
            dependent measurement
        X : categorial
            Model (Factor or Interaction) for dividing Y.
        match :
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


        Examples
        --------
        Split a repeated-measure variable Y into cells defined by the
        interaction of A and B::

            >>> c = Celltable(Y, A % B, match=subject)

        """
        self.sub = sub
        sub = assub(sub, ds)
        if X is not None:
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
                            cat = list(cat)
                            for i in xrange(len(cat)):
                                if cat[i] is None:
                                    cat[i] = cells.pop(0)
                            cat = tuple(cat)
                    else:
                        err = ("Categories can only be specified as None if X "
                               "contains exactly as many cells as categories are "
                               "required (%i)." % len(cat))
                        raise ValueError(err)

                # apply cat
                sort_idx = X.sort_idx(order=cat)
                X = X[sort_idx]
                if sub is None:
                    sub = sort_idx
                else:
                    imax = max(len(sub), np.max(sub))
                    sub = np.arange(imax)[sub][sort_idx]

        Y = asdataobject(Y, sub, ds)

        if match is not None:
            match = asfactor(match, sub, ds)
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
        self.n_cases = len(Y)

        # extract cell data
        self.data = {}
        self.data_indexes = {}
        if X is None:
            self.data[None] = Y
            self.data_indexes[None] = slice(None)
            self.cells = [None]
            self.n_cells = 1
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
                func = cihw
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


def combine(items, name=None, check_dims=True):
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

    Notes
    -----
    For datasets, missing variables are filled in with empty values ('' for
    factors, NaN for variables).
    """
    if name is None:
        names = filter(None, (item.name for item in items))
        name = os.path.commonprefix(names) or None

    item0 = items[0]
    if isdataset(item0):
        # find all keys and data types
        keys = item0.keys()
        sample = dict(item0)
        for item in items:
            for key in item.keys():
                if key not in keys:
                    keys.append(key)
                    sample[key] = item[key]
        # create new Dataset
        out = Dataset(name=name)
        for key in keys:
            pieces = [ds[key] if key in ds else
                      _empty_like(sample[key], ds.n_cases) for ds in items]
            out[key] = combine(pieces)
        return out
    elif isvar(item0):
        x = np.hstack(i.x for i in items)
        return Var(x, name=name)
    elif isfactor(item0):
        if all(f._labels == item0._labels for f in items[1:]):
            x = np.hstack(f.x for f in items)
            kwargs = item0._child_kwargs(name=name)
        else:
            x = sum((i.as_labels() for i in items), [])
            kwargs = dict(name=name, random=item0.random)
        return Factor(x, **kwargs)
    elif isndvar(item0):
        has_case = np.array([v.has_case for v in items])
        if np.all(has_case):
            has_case = True
            all_dims = (item.dims[1:] for item in items)
        elif np.all(has_case == False):
            has_case = False
            all_dims = (item.dims for item in items)
        else:
            err = ("Some items have a 'case' dimension, others do not")
            raise DimensionMismatchError(err)

        dims = reduce(lambda x, y: intersect_dims(x, y, check_dims), all_dims)
        idx = {d.name: d for d in dims}
        items = [item.sub(**idx) for item in items]
        if has_case:
            x = np.concatenate([v.x for v in items], axis=0)
        else:
            x = np.array([v.x for v in items])
        dims = ('case',) + dims
        return NDVar(x, dims=dims, name=name, info=item0.info)
    elif isdatalist(item0):
        out = sum(items[1:], item0)
        out.name = name
        return out
    else:
        err = ("Objects of type %s can not be combined." % type(item0))
        raise TypeError(err)


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
            if (len(f) == len(item)) and np.all(item == f):
                return True
        return False

    def index(self, item):
        for i, f in enumerate(self):
            if (len(f) == len(item)) and np.all(item == f):
                return i
        raise ValueError("Factor %r not in EffectList" % item.name)

    def names(self):
        return [e.name if isuv(e) else repr(e) for e in self]



class Var(object):
    """
    Container for scalar data.

    While :py:class:`Var` objects support a few basic operations in a
    :py:mod:`numpy`-like fashion (``+``, ``-``, ``*``, ``/``, ``//``), their
    :py:attr:`Var.x` attribute provides access to the corresponding
    :py:class:`numpy.array` which can be used for anything more complicated.
    :py:attr:`Var.x` can be read and modified, but should not be replaced.

    """
    _stype_ = "var"
    def __init__(self, x, name=None):
        """Represents a univariate variable.

        Parameters
        ----------
        x : array_like
            Data; is converted with ``np.asarray(x)``. Multidimensional arrays
            are flattened as long as only 1 dimension is longer than 1.
        name : str | None
            Name of the variable
        """
        x = np.asarray(x)
        if x.ndim > 1:
            if np.count_nonzero(i > 1 for i in x.shape) <= 1:
                x = np.ravel(x)
            else:
                err = ("X needs to be one-dimensional. Use NDVar class for "
                       "data with more than one dimension.")
                raise ValueError(err)
        self.__setstate__((x, name))

    def __setstate__(self, state):
        x, name = state
        # raw
        self.name = name
        self.x = x
        # constants
        self._n_cases = len(x)
        self.df = 1
        self.random = False

    def __getstate__(self):
        return (self.x, self.name)

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

        return "Var(%s)" % ', '.join(args)

    def __str__(self):
        return self.__repr__(True)

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
            return Var(x, self.name)
        elif isvar(index):
            index = index.x

        x = self.x[index]
        if np.iterable(x):
            return Var(x, self.name)
        else:
            return x

    def __setitem__(self, index, value):
        self.x[index] = value

    def __contains__(self, value):
        return value in self.x

    # numeric ---
    def __add__(self, other):
        if isdataobject(other):
            # ??? should Var + Var return sum or Model?
            return Model(self, other)
        else:
            x = self.x + other
            if np.isscalar(other):
                name = '%s+%s' % (self.name, other)
            else:
                name = self.name

            return Var(x, name=name)

    def __sub__(self, other):
        "subtract: values are assumed to be ordered. Otherwise use .sub method."
        if np.isscalar(other):
            return Var(self.x - other,
                       name='%s-%s' % (self.name, other))
        elif len(other) != len(self):
            err = ("Objects have different length (%i vs "
                   "%i)" % (len(self), len(other)))
            raise ValueError(err)
        else:
            x = self.x - other.x
            n1, n2 = self.name, other.name
            if n1 == n2:
                name = n1
            else:
                name = "%s-%s" % (n1, n2)
            return Var(x, name)

    def __mul__(self, other):
        if iscategorial(other):
            return Model(self, other, self % other)
        elif isvar(other):
            x = self.x * other.x
            name = '%s*%s' % (self.name, other.name)
        else:  #  np.isscalar(other)
            x = self.x * other
            other_name = str(other)
            if len(other_name) < 12:
                name = '%s*%s' % (self.name, other_name)
            else:
                name = self.name

        return Var(x, name=name)

    def __floordiv__(self, other):
        if isvar(other):
            x = self.x // other.x
            name = '%s//%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x // other
            name = '%s//%s' % (self.name, other)
        else:
            x = self.x // other
            name = '%s//%s' % (self.name, '?')
        return Var(x, name=name)

    def __mod__(self, other):
        if  ismodel(other):
            return Model(self) % other
        elif isdataobject(other):
            return Interaction((self, other))
        elif isvar(other):
            other = other.x
            other_name = other.name
        else:
            other_name = str(other)[:10]

        name = '{name}%{other}'
        name = name.format(name=self.name, other=other_name)
        return Var(self.x % other, name=name)

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

    def __div__(self, other):
        """
        type of other:
        scalar:
            returns var divided by other
        Factor:
            returns a separate slope for each level of the Factor; needed for
            ANCOVA

        """
        if np.isscalar(other):
            return Var(self.x / other,
                       name='%s/%s' % (self.name, other))
        elif isvar(other):
            return Var(self.x / other.x,
                       name='%s/%s' % (self.name, other.name))
        else:
            categories = other
            if not hasattr(categories, 'as_dummy_complete'):
                raise NotImplementedError
            dummy_factor = categories.as_dummy_complete
            codes = dummy_factor * self.as_effects
            # center
            means = codes.sum(0) / dummy_factor.sum(0)
            codes -= dummy_factor * means
            # create effect
            name = '%s per %s' % (self.name, categories.name)
            labels = categories.dummy_complete_labels
            out = NonbasicEffect(codes, [self, categories], name,
                                  beta_labels=labels)
            return out

    def argmax(self):
        """:fun:`numpy.argmax`"""
        return np.argmax(self.x)

    def argmin(self):
        """:fun:`numpy.argmin`"""
        return np.argmin(self.x)

    def argsort(self, kind='quicksort'):
        """:fun:`numpy.argsort`

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
    def as_effects(self):
        "for effect initialization"
        return self.centered()[:, None]

    def as_factor(self, name=None, labels='%r'):
        """
        Convert the Var into a Factor

        Parameters
        ----------
        name : None | str
            Name for the Factor. If None (default), it will be the Var's name.
        labels : dict | str
            Dictionary mapping values to labels, or format string for
            converting values into labels (default: ``'%r'``).
        """
        if name is None:
            name = self.name

        if type(labels) is not dict:
            fmt = labels
            labels = {}
            for value in np.unique(self.x):
                labels[value] = fmt % value

        f = Factor(self.x, name=name, labels=labels)
        return f

    def centered(self):
        return self.x - self.x.mean()

    def copy(self, name='{name}'):
        "returns a deep copy of itself"
        x = self.x.copy()
        name = name.format(name=self.name)
        return Var(x, name=name)

    def compress(self, X, func=np.mean, name='{name}'):
        "Deprecated. Use .aggregate()."
        warn("Var.compress s deprecated; use Var.aggregate instead"
             "(with identical functionality).", DeprecationWarning)
        self.aggregate(X, func, name)

    def aggregate(self, X, func=np.mean, name='{name}'):
        """Summarize cases within cells of X

        Parameters
        ----------
        X : categorial
            Model defining cells in which to aggregate.
        func : callable
            Function that converts arrays into scalars, used to summarize data
            within each cell of X.
        name : str
            Name of the output Var ("{name}" is converted to the current Var's
            name).

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

        x = np.array(x)
        name = name.format(name=self.name)
        out = Var(x, name=name)
        return out

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
        return Var(y, name)

    @classmethod
    def from_dict(cls, base, values, name=None, default=0):
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
        Y = cls([values.get(b, default) for b in base], name=name)
        return Y

    @classmethod
    def from_apply(cls, base, func, name='{func}({name})'):
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
        base_name = getattr(base, 'name', 'x')
        if isvar(base) or isndvar(base):
            base = base.x

        if isinstance(func, np.ufunc):
            x = func(base)
        elif getattr(base, 'ndim', 1) > 1:
            x = func(base.reshape((len(base), -1)), axis=1)
        else:
            x = np.array([func(val) for val in base])

        name = name.format(func=func.__name__, name=base_name)
        return cls(x, name=name)

    def index(self, *values):
        """
        v.index(*values) returns an array of indices of where v has one of
        *values.

        """
        idx = []
        for v in values:
            where = np.where(self == v)[0]
            idx.extend(where)
        return sorted(idx)

    def isany(self, *values):
        return np.any([self.x == v for v in values], axis=0)

    def isin(self, values):
        return np.any([self.x == v for v in values], axis=0)

    def isnot(self, *values):
        return np.all([self.x != v for v in values], axis=0)

    def max(self):
        "Returns the highest value"
        return self.x.max()

    def mean(self):
        "Returns the mean"
        return self.x.mean()

    def min(self):
        "Returns the smallest value"
        return self.x.min()

    def repeat(self, repeats, name='{name}'):
        "Analogous to :py:func:`numpy.repeat`"
        return Var(self.x.repeat(repeats), name=name.format(name=self.name))

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
        return Model(self, other, self % other)

    def __mod__(self, other):
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
        count : array of int,  len = len(self)
            Cumulative count of value in self.

        Examples
        --------
        >>> a = Factor('abc', tile=3)
        >>> a
        Factor(['a', 'b', 'c', 'a', 'b', 'c', 'a', 'b', 'c'])
        >>> a.count('a')
        array([0, 0, 0, 1, 1, 1, 2, 2, 2])
        """
        count = np.cumsum(self == value) + start
        return count

    def index(self, cell):
        "``e.index(cell)`` returns an array of indices where e equals cell"
        return np.nonzero(self == cell)[0]

    def index_opt(self, cell):
        """Find an optimized index for a given cell.

        Returns
        -------
        index : slice | array
            If possible, a ``slice`` object is returned. Otherwise, an array
            of indices (as with ``e.index(cell)``).
        """
        index = self.index(cell)
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
        if order is None:
            cells = self.cells
            idx = np.empty(len(self), dtype=np.uint32)
        else:
            cells = order
            idx = -np.ones(len(self), dtype=np.uint32)

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
    """
    Container for categorial data.

    """
    _stype_ = "factor"
    def __init__(self, x, name=None, random=False, rep=1, tile=1, labels={}):
        """
        Parameters
        ----------
        x : iterator
            Sequence of Factor values (see also the ``labels`` kwarg).
        name : str
            Name of the Factor.
        random : bool
            Treat Factor as random factor (for ANOVA; default is False).
        rep : int
            Repeat each element in ``x`` ``rep`` many times.
        tile : int
            Repeat x as a whole ``tile`` many times.
        labels : dict or None
            If provided, these labels are used to replace values in x when
            constructing the labels dictionary. All labels for values of
            x not in ``labels`` are constructed using ``str(value)``.


        Examples
        --------
        The most obvious way to initialize a Factor is a list of strings::

            >>> Factor(['in', 'in', 'in', 'out', 'out', 'out'])
            Factor(['in', 'in', 'in', 'out', 'out', 'out'])

        The same can be achieved with a list of integers plus a labels dict::

            >>> Factor([1, 1, 1, 0, 0, 0], labels={1: 'in', 0: 'out'})
            Factor(['in', 'in', 'in', 'out', 'out', 'out'])

        Since the Factor initialization simply iterates over the ``x``
        argument, a Factor with one-character codes can also be initialized
        with a single string::

            >>> Factor('iiiooo')
            Factor(['i', 'i', 'i', 'o', 'o', 'o'])

        """
        state = {'name': name, 'random': random}
        labels_ = state['labels'] = {}  # {code -> label}

        try:
            n_cases = len(x)
        except TypeError:  # for generators:
            x = tuple(x)
            n_cases = len(x)

        # convert x to codes
        codes = {}  # {label -> code}
        x_ = np.empty(n_cases, dtype=np.uint16)
        for i, value in enumerate(x):
            label = labels.get(value, value)
            if not isinstance(label, unicode):
                label = str(label)
            if label in codes:
                code = codes.get(label)
            else:  # new code
                code = max(labels_) + 1 if labels_ else 0
                labels_[code] = label
                codes[label] = code

            x_[i] = code

        if rep > 1:
            x_ = x_.repeat(rep)

        if tile > 1:
            x_ = np.tile(x_, tile)

        state['x'] = x_
        self.__setstate__(state)

    def __setstate__(self, state):
        self.x = x = state['x']
        self.name = state['name']
        self.random = state['random']
        self._labels = labels = state['labels']
        self._codes = {lbl: code for code, lbl in labels.iteritems()}
        self._n_cases = len(x)

    def __getstate__(self):
        state = {'x': self.x,
                 'name': self.name,
                 'random': self.random,
                 'labels': self._labels}
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
            return Factor(x, **self._child_kwargs())
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

            if code >= 65535:
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
        return self.x == self._encode_(other)

    def __ne__(self, other):
        return self.x != self._encode_(other)

    def _encode_(self, Y):
        if isinstance(Y, basestring):
            return self._codes.get(Y, -1)
        else:
            out = np.empty(len(Y), dtype=self.x.dtype)
            for i, v in enumerate(Y):
                out[i] = self._codes.get(v, -1)
            return out

    def __call__(self, other):
        """
        Create a nested effect. A factor A is nested in another factor B if
        each level of A only occurs together with one level of B.

        """
        return NestedEffect(self, other)

    def _child_kwargs(self, name='{name}'):
        kwargs = dict(labels=self._labels,
                      name=name.format(name=self.name),
                      random=self.random)
        return kwargs

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

    def as_labels(self):
        return [self._labels[v] for v in self.x]

    @property
    def beta_labels(self):
        cells = self.cells
        txt = '{0}=={1}'
        return [txt.format(cells[i], cells[-1]) for i in range(len(cells) - 1)]

    @property
    def cells(self):
        return natsorted(self._labels.values())

    def compress(self, X, name='{name}'):
        "Deprecated. Use .aggregate()."
        warn("Factor.compress s deprecated; use Factor.aggregate instead"
             "(with identical functionality).", DeprecationWarning)
        self.aggregate(X, name)

    def aggregate(self, X, name='{name}'):
        """
        Summarize the Factor by collapsing within cells in `X`.

        Raises an error if there are cells that contain more than one value.

        Parameters
        ----------
        X : categorial
            A categorial model defining cells to collapse.

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
                    err = ("ambiguous cell: Factor %r has multiple values for "
                           "cell %r" % (self.name, cell))
                    raise ValueError(err)
                else:
                    x.append(x_i[0])

        x = np.array(x)
        name = name.format(name=self.name)
        out = Factor(x, name=name, labels=self._labels, random=self.random)
        return out

    def copy(self, name='{name}', rep=1, tile=1):
        "returns a deep copy of itself"
        f = Factor(self.x.copy(), rep=rep, tile=tile,
                   **self._child_kwargs(name))
        return f

    @property
    def df(self):
        return max(0, len(self._labels) - 1)

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
        if not all(isinstance(v, basestring) for v in values):
            err = "Factor indexes need to be str, got %r" % str(values)
            raise ValueError(err)
        return self.isin(values)

    def isin(self, values):
        """Find the index of entries matching one of the ``values``

        Returns
        -------
        index : array of bool
            For each case True if the value is in values, else False.

        Examples
        --------
        >>> a = Factor('aabbcc')
        >>> b.isany(('b', 'c'))
        array([False, False,  True,  True,  True,  True], dtype=bool)
        """
        is_v = [self.x == self._codes.get(v, np.nan) for v in values]
        return np.any(is_v, 0)

    def isnot(self, *values):
        """Find the index of entries not in ``values``

        Returns
        -------
        index : array of bool
            For each case False if the value is in values, else True.
        """
        if not all(isinstance(v, basestring) for v in values):
            err = "Factor indexes need to be str, got %r" % str(values)
            raise ValueError(err)
        return self.isnotin(values)

    def isnotin(self, values):
        """Find the index of entries not in ``values``

        Returns
        -------
        index : array of bool
            For each case False if the value is in values, else True.
        """
        is_not_v = [self.x != self._codes.get(v, np.nan) for v in values]
        if is_not_v:
            return np.all(is_not_v, axis=0)
        else:
            return np.ones(len(self), dtype=bool)

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

    def project(self, target, name='{name}'):
        """
        Project the Factor onto an index array ``target``

        Example::

            >>> f = Factor('abc')
            >>> f.as_labels()
            ['a', 'b', 'c']
            >>> fp = f.project([1,2,1,2,0,0])
            >>> fp.as_labels()
            ['b', 'c', 'b', 'c', 'a', 'a']

        """
        if isvar(target):
            target = target.x
        x = self.x[target]
        return Factor(x, **self._child_kwargs(name))

    def repeat(self, repeats, name='{name}'):
        "Repeat elements of a Factor (analogous to :py:func:`numpy.repeat`)"
        return Factor(self.x.repeat(repeats), **self._child_kwargs(name))



class NDVar(object):
    "Container for n-dimensional data."
    _stype_ = "ndvar"
    def __init__(self, x, dims=('case',), info={}, name=None):
        """
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

    # numeric ---
    def _align(self, other):
        "align data from 2 ndvars"
        if isvar(other):
            return self.dims, self.x, self._ialign(other)
        elif isndvar(other):
            i_self = list(self.dimnames)
            dims = list(self.dims)
            i_other = []

            for dim in i_self:
                if dim in other.dimnames:
                    i_other.append(dim)
                else:
                    i_other.append(None)

            for dim in other.dimnames:
                if dim in i_self:
                    pass
                else:
                    i_self.append(None)
                    i_other.append(dim)
                    dims.append(other.get_dim(dim))

            x_self = self.get_data(i_self)
            x_other = other.get_data(i_other)
            return dims, x_self, x_other
        else:
            raise TypeError

    def _ialign(self, other):
        "align for self-modifying operations (+=, ...)"
        if isvar(other):
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
        if isnumeric(other):
            dims, x_self, x_other = self._align(other)
            x = x_self + x_other
            name = '%s+%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x + other
            dims = self.dims
            name = '%s+%s' % (self.name, str(other))
        else:
            raise ValueError("can't add %r" % other)
        return NDVar(x, dims=dims, name=name, info=self.info)

    def __iadd__(self, other):
        self.x += self._ialign(other)
        return self

    def __div__(self, other):
        if isnumeric(other):
            dims, x_self, x_other = self._align(other)
            x = x_self / x_other
            name = '%s/%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x / other
            dims = self.dims
            name = '%s/%s' % (self.name, str(other))
        else:
            raise ValueError("can't subtract %r" % other)
        return NDVar(x, dims=dims, name=name, info=self.info)

    def __mul__(self, other):
        if isnumeric(other):
            dims, x_self, x_other = self._align(other)
            x = x_self * x_other
            name = '%s*%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x * other
            dims = self.dims
            name = '%s*%s' % (self.name, str(other))
        else:
            raise ValueError("can't subtract %r" % other)
        return NDVar(x, dims=dims, name=name, info=self.info)

    def __sub__(self, other):  # TODO: use dims
        if isnumeric(other):
            dims, x_self, x_other = self._align(other)
            x = x_self - x_other
            name = '%s-%s' % (self.name, other.name)
        elif np.isscalar(other):
            x = self.x - other
            dims = self.dims
            name = '%s-%s' % (self.name, str(other))
        else:
            raise ValueError("can't subtract %r" % other)
        return NDVar(x, dims=dims, name=name, info=self.info)

    def __isub__(self, other):
        self.x -= self._ialign(other)
        return self

    def __rsub__(self, other):
        x = other - self.x
        return NDVar(x, self.dims, self.info, name=self.name)

    # container ---
    def __getitem__(self, index):
        if isvar(index):
            index = index.x

        if np.iterable(index) or isinstance(index, slice):
            x = self.x[index]
            if x.shape[1:] != self.x.shape[1:]:
                err = ("Use .sub() method for indexing dimensions other than "
                       "case")
                raise NotImplementedError(err)
            return NDVar(x, dims=self.dims, name=self.name, info=self.info)
        else:
            index = int(index)
            x = self.x[index]
            if self.ndim == 1:
                return x

            dims = self.dims[1:]
            if self.name:
                name = '%s_%i' % (self.name, index)
            else:
                name = None
            return NDVar(x, dims=dims, name=name, info=self.info)

    def __len__(self):
        return self._len

    def __repr__(self):
        rep = '<NDVar %(name)r: %(dims)s>'
        if self.has_case:
            dims = [(self._len, 'case')]
        else:
            dims = []
        dims.extend([(len(dim), dim._dimrepr_()) for dim in self._truedims])

        dims = ' X '.join('%i (%s)' % fmt for fmt in dims)
        args = dict(dims=dims, name=self.name or '')
        return rep % args

    def assert_dims(self, dims):
        if self.dimnames != dims:
            err = "Dimensions of %r do not match %r" % (self, dims)
            raise DimensionMismatchError(err)

    def compress(self, X, func=np.mean, name='{name}'):
        "Deprecated. Use .aggregate()."
        warn("NDVar.compress s deprecated; use NDVar.aggregate instead"
             "(with identical functionality).", DeprecationWarning)
        self.aggregate(X, func, name)

    def aggregate(self, *dims, **regions):
        r"""Aggregate data in the NDVar

        This method has two possible signature.

          - Aggregating data over cases in specified cells is described in the
            Parameters section below.
          - Aggregating data over dimensions (or slices in dimensions) is
            performed as described here:

        dimension:
            A whole dimension is specified as string argument. This
            dimension is collapsed over the whole range.
        slice:
            A range within a dimension is specified through a keyword-argument.
            Only the data in the specified range is included. Use like the
            :py:meth:`.sub` method.

        .. warning::
            Data is collapsed over the different dimensions in turn using the
            provided function with an axis argument. For certain functions
            this is not equivalent to collapsing over several axes concurrently
            (e.g., variance `np.var`).

        **additional kwargs:**

        func : callable
            Function used to collapse the data. Needs to accept an "axis"
            kwarg (default: np.mean)
        name : str
            Name for the new NDVar. Default: "{func}({name})".

        Parameters
        ----------
        X : categorial
            Categorial whose cells define which cases to aggregate.
        func : function with axis argument
            Function that is used to create a summary of the cases falling
            into each cell of X. The function needs to accept the data as
            first argument and ``axis`` as keyword-argument. Default is
            ``numpy.mean``.
        name : str
            Name for the resulting NDVar. ``'{name}'`` is formatted to the
            current NDVar's ``.name``.

        Returns
        -------
        aggregated_ndvar : NDVar
            NDVar containing the aggregated data.

        Examples
        --------
        Assuming ``data`` is a normal time series. Get the average in a time
        window::

        >>> Y = data.aggregate(time=(.1, .2))

        Get the peak in a time window::

        >>> Y = data.aggregate(time=(.1, .2), func=np.max)

        Assuming MEG is an NDVar with dimensions time and sensor. Get the
        average across sensors 5, 6, and 8 in a time window::

        >>> ROI = [5, 6, 8]
        >>> Y = MEG.aggregate(sensor=ROI, time=(.1, .2))

        Get the peak in the same data:

        >>> ROI = [5, 6, 8]
        >>> Y = MEG.aggregate(sensor=ROI, time=(.1, .2), func=np.max)

        Get the RMS over all sensors

        >>> MEG_RMS = MEG.aggregate('sensor', func=rms)
        """
        # special case:  aggregate cases in cells
        if len(dims) and not isinstance(dims[0], basestring):
            if not self.has_case:
                raise DimensionMismatchError("%r has no case dimension" % self)
            elif 'case' in regions:
                raise ValueError("Case index specified twice")
            X = dims[0]
            if len(X) != len(self):
                err = ("Length mismatch: %i (Var) != %i "
                       "(X)" % (len(self), len(X)))
                raise ValueError(err)

            if len(dims) > 1:
                func = dims[1]
            else:
                func = regions.pop('func',
                                   self.info.get('summary_func', np.mean))

            if len(dims) > 2:
                name = dims[2]
            else:
                name = regions.get('name', '{name}')

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

            x = np.array(x)
            name = name.format(name=self.name)
            out = NDVar(x, self.dims, info=info, name=name)
            return out

        # normal case: aggregate in dimensions
        func = regions.pop('func', self.info.get('summary_func', np.mean))
        name = regions.pop('name', '{func}({name})')
        name = name.format(func=func.__name__, name=self.name)
        if len(dims) + len(regions) == 0:
            dims = ('case',)

        if regions:
            dims = list(dims)
            for dim in sorted(regions):
                if not np.isscalar(regions[dim]) and not dim in dims:
                    dims.append(dim)
            data = self.sub(**regions)
            return data.aggregate(*dims, func=func, name=name)
        else:
            x = self.x
            axes = [self._dim_2_ax[dim] for dim in np.unique(dims)]
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
                return Var(x, name=name)
            else:
                return NDVar(x, dims=dims, name=name, info=info)

    def copy(self, name='{name}'):
        "returns a deep copy of itself"
        x = self.x.copy()
        name = name.format(name=self.name)
        info = self.info.copy()
        return self.__class__(x, dims=self.dims, name=name,
                              info=info)

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
        if isinstance(dim, str) and dim == 'case':
            dim = Var(np.arange(len(self)), 'case')

        return dim

    def get_dims(self, names):
        "Returns a tuple with the requested Dimension objects"
        return tuple(self.get_dim(name) for name in names)

    def has_dim(self, name):
        return name in self._dim_2_ax

    def repeat(self, repeats, dim='case', name='{name}'):
        """
        Analogous to :py:func:`numpy.repeat`

        """
        ax = self.get_axis(dim)
        x = self.x.repeat(repeats, axis=ax)

        repdim = self.dims[ax]
        if not isinstance(repdim, str):
            repdim = repdim.repeat(repeats)

        dims = self.dims[:ax] + (repdim,) + self.dims[ax + 1:]
        info = self.info.copy()
        name = name.format(name=self.name)
        return NDVar(x, dims, info=info, name=name)

    def summary(self, *dims, **regions):
        "Deprecated. Use .aggregate() instead with the same signature."
        return self.aggregate(*dims, **regions)

    def sub(self, **kwargs):
        """Retrieve a slice through the NDVar.

        Returns an NDVar object with a slice of the current NDVar's data.
        The slice is specified using kwargs, with dimensions as keywords and
        indexes as values, e.g.::

            >>> Y.sub(time = 1)

        returns a slice for time point 1 (second). For dimensions whose values
        change monotonically, a tuple can be used to specify a window::

            >>> Y.sub(time = (.2, .6))

        returns a slice containing all values for times .2 seconds to .6
        seconds.

        The name of the new NDVar can be set with a ``name`` parameter. The
        default is the name of the current NDVar.
        """
        var_name = kwargs.pop('name', self.name)
        info = self.info.copy()
        dims = list(self.dims)
        index = [slice(None)] * len(dims)

        for name, arg in kwargs.iteritems():
            if arg is None:
                continue

            dimax = self.get_axis(name)
            dim = self.dims[dimax]

            if hasattr(dim, 'dimindex'):
                idx = dim.dimindex(arg)
            else:
                idx = arg

            index[dimax] = idx
            if np.isscalar(idx):
                dims[dimax] = None
                info[name] = arg
            elif isinstance(dim, str):
                dims[dimax] = dim
            else:
                dims[dimax] = dim[idx]

        # slice the data
        x = self.x
        i_max = len(index) - 1
        for i, idx in enumerate(reversed(index)):
            if isinstance(idx, slice) and idx == slice(None):
                continue
            i_cur = i_max - i
            x = x[(slice(None),) * i_cur + (idx,)]

        # create NDVar
        dims = tuple(dim for dim in dims if dim is not None)
        return NDVar(x, dims=dims, name=var_name, info=info)

    def subdata(self, **kwargs):
        "Deprecated. Use .sub() method (with identical functionality)."
        warn("NDVar.subdata is deprecated; use NDVar.sub instead "
             "(with identical functionality).", DeprecationWarning)
        return self.sub(**kwargs)


class Datalist(list):
    """
    :py:class:`list` subclass for including lists in in a Dataset.

    The subclass adds certain methods that makes indexing behavior more
    similar to numpy and other data objects. The following methods are
    modified:

    __getitem__:
        Add support for list and array indexes.
    __init__:
        Store a name attribute.

    """
    _stype_ = 'list'
    def __init__(self, items=None, name=None):
        self.name = name
        if items:
            super(Datalist, self).__init__(items)
        else:
            super(Datalist, self).__init__()

    def __repr__(self):
        return "Datalist(%s)" % super(Datalist, self).__repr__()

    def __getitem__(self, index):
        if isinstance(index, (int, slice)):
            return list.__getitem__(self, index)

        index = np.array(index)
        if issubclass(index.dtype.type, np.bool_):
            N = len(self)
            assert len(index) == N
            return Datalist(self[i] for i in xrange(N) if index[i])
        elif issubclass(index.dtype.type, np.integer):
            return Datalist(self[i] for i in index)
        else:
            err = ("Unsupported type of index for Datalist: %r" % index)
            raise TypeError(err)

    def __add__(self, other):
        lst = super(Datalist, self).__add__(other)
        return Datalist(lst, name=self.name)

    def compress(self, X, merge='mean'):
        "Deprecated. Use .aggregate()."
        warn("Var.compress s deprecated; use Var.aggregate instead"
             "(with identical functionality).", DeprecationWarning)
        self.aggregate(X, merge)

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

        x = Datalist()
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

        return x




class Dataset(collections.OrderedDict):
    """
    A Dataset is a dictionary that represents a data table.

    Attributes
    ----------
    n_cases : None | int
        The number of cases in the Dataset (corresponding to the number of
        rows in the table representation). None if no variables have been
        added.
    n_items : int
        The number of items (variables) in the Dataset (corresponding to the
        number of columns in the table representation).

    Methods
    -------
    add:
        Add a variable to the Dataset.
    as_table:
        Create a data table representation of the Dataset.
    aggregate:
        For a given categorial description, reduce the number of cases in the
        Dataset per cell to one (e.g., average by subject and condition).
    copy:
        Create a shallow copy (i.e., return a new Dataset containing the same
        data objects).
    export:
        Save the Dataset in different formats.
    eval:
        Evaluate an expression in the Dataset's namespace.
    index:
        Add an index to the Dataset (to keep track of cases when selecting
        subsets).
    itercases:
        Iterate through each case as a dictionary.
    subset:
        Create a Dataset form a subset of cases (slightly faster than normal
        indexing).

    See Also
    --------
    collections.OrderedDict: superclass

    Notes
    -----
    A Dataset represents a data table as a {variable_name: value_list}
    dictionary. Each variable corresponds to a column, and each index in the
    value list corresponds to a row, or case.

    The Dataset class inherits most of its behavior from its superclass
    :py:class:`collections.OrderedDict`.
    Dictionary keys are enforced to be :py:class:`str` objects and should
    preferably correspond to the variable names.
    An exception is the Dataset's length, which reflects the number of cases
    in the Dataset (i.e., the number of rows; the number of items can be
    retrieved as  :py:attr:`Dataset.n_items`).


    **Accessing Data:**

    Standard indexing with *strings* is used to access the contained Var and
    Factor objects. Nesting is possible:

    - ``ds['var1']`` --> ``var1``.
    - ``ds['var1',]`` --> ``[var1]``.
    - ``ds['var1', 'var2']`` --> ``[var1, var2]``

    When indexing numerically, the first index defines cases (rows):

    - ``ds[1]`` --> row 1
    - ``ds[1:5]`` == ``ds[1,2,3,4]`` --> rows 1 through 4
    - ``ds[1, 5, 6, 9]`` == ``ds[[1, 5, 6, 9]]`` --> rows 1, 5, 6 and 9

    The second index accesses columns, so case indexing can be combined with
    column indexing:

     - ``ds[:4, :2]`` --> first 4 rows,

    The ``.get_case()`` method or iteration over the Dataset
    retrieve individual cases/rows as {name: value} dictionaries.


    **Naming:**

    While Var and Factor objects themselves need not be named, they need
    to be named when added to a Dataset. This can be done by a) adding a
    name when initializing the Dataset::

        >>> ds = Dataset(('v1', var1), ('v2', var2))

    or b) by adding the Var or Factor with a key::

        >>> ds['v3'] = var3

    If a Var/Factor that is added to a Dataset does not have a name, the new
    key is automatically assigned to the Var/Factor's ``.name`` attribute.

    """
    _stype_ = "dataset"
    def __init__(self, *items, **kwargs):
        """Store multiple variables covering to the same observations

        Datasets can be initialize with data-objects, or with
        ('name', data-object) tuples.::

            >>> ds = Dataset(var1, var2)
            >>> ds = Dataset(('v1', var1), ('v2', var2))

        The Dataset stores the input items themselves, without making a copy().


        Parameters
        ----------

        name : str
            name describing the Dataset
        info : dict
            info dictionary, can contain arbitrary entries and can be accessad
            as ``.info`` attribute after initialization.

        """
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

        self.n_cases = None
        super(Dataset, self).__init__(args)
        self.__setstate__(kwargs)

    def __setstate__(self, kwargs):
        self.name = kwargs.get('name', None)
        self.info = kwargs.get('info', {})
        self._caption = kwargs.get('caption', None)

    def __reduce__(self):
        args = tuple(self.items())
        kwargs = {'name': self.name, 'info': self.info,
                  'caption': self._caption}
        return self.__class__, args, kwargs

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
            return Dataset(*(self[item] for item in index))

        if isinstance(index, tuple):
            i0 = index[0]
            if isinstance(i0, basestring):
                return self[i0][index[1:]]

            if len(index) != 2:
                raise KeyError("Invalid index for Dataset: %r" % index)
            i1 = index[1]
            if isinstance(i1, basestring):
                return self[i1][i0]
            elif np.iterable(i1) and all(isinstance(item, basestring) for item
                                         in i1):
                keys = i1
            else:
                keys = Datalist(self.keys())[i1]
                if isinstance(keys, basestring):
                    return self[i1][i0]

            subds = Dataset(*((k, self[k][i0]) for k in keys))
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

    def __setitem__(self, name, item, overwrite=True):
        if isinstance(name, str):
            # test if name already exists
            if (not overwrite) and (name in self):
                raise KeyError("Dataset already contains variable of name %r" % name)

            # coerce item to data-object
            if isdataobject(item) or isinstance(object, Datalist):
                if not item.name:
                    item.name = name
            elif isinstance(item, (list, tuple)):
                item = Datalist(item, name=name)
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

            super(Dataset, self).__setitem__(name, item)
        elif isinstance(name, tuple):
            if len(name) != 2:
                raise NotImplementedError("More than 2 index components.")
            key, idx = name
            if key in self:
                self[key][idx] = item
            elif isinstance(item, basestring):
                if idx.start is None and idx.stop is None:
                    self[key] = Factor([item], rep=self.n_cases)
                else:
                    err = ("Can only add Factor with general value "
                           "(ds['name',:] = ...")
                    raise NotImplementedError(err)
            else:
                raise NotImplementedError
        else:
            raise TypeError("Dataset indexes need to be strings")

    def __str__(self):
        return unicode(self).encode('utf-8')

    def __unicode__(self):
        if sum(map(isuv, self.values())) == 0:
            return self.__repr__()

        maxn = preferences['dataset_str_n_cases']
        txt = unicode(self.as_table(cases=maxn, fmt='%.5g', midrule=True))
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
        """
        ``ds.add(item)`` -> ``ds[item.name] = item``

        unless the Dataset already contains a variable named item.name, in
        which case a KeyError is raised. In order to replace existing
        variables, set ``replace`` to True::

            >>> ds.add(item, True)

        """
        if not isdataobject(item):
            raise ValueError("Not a valid data-object: %r" % item)
        elif (item.name in self) and not replace:
            raise KeyError("Dataset already contains variable named %r" % item.name)
        else:
            self[item.name] = item

    def as_table(self, cases=0, fmt='%.6g', sfmt='%s', match=None,
                 sort=False, header=True, midrule=False, count=False,
                 title=None, caption=None, ifmt='%s', bfmt='%s',
                 f_fmt='deprecated'):
        r"""
        Create a fmtxt.Table containing all Vars and Factors in the Dataset.
        Can be used for exporting in different formats such as csv.

        Parameters
        ----------
        cases : int
            number of cases to include (0 includes all; negative number works
            like negative indexing).
        count : bool
            Add an initial column containing the case number.
        fmt : str
            Format string for float variables (default ``'%.6g'``).
        sfmt : str | None
            Formatting for strings (None -> code; default ``'%s'``).
        match : None | Factor
            Create repeated-measurement table.
        header : bool
            Include the varibale names as a header row.
        midrule : bool
            print a midrule after table header.
        sort : bool
            Sort the columns alphabetically.
        title : None | str
            Title for the table.
        caption : None | str
            Caption for the table (default is the Dataset's caption).
        ifmt : str
            Formatting for integers (default ``'%s'``).
        bfmt : str
            Formatting for booleans (default ``'%s'``).
        """
        if f_fmt != 'deprecated':
            msg = "The f_fmt parameter is deprecated. Use sfmt instead."
            warn(msg, DeprecationWarning)
            sfmt = f_fmt

        if cases < 1:
            cases = self.n_cases + cases
            if cases < 0:
                raise ValueError("Can't get table for fewer than 0 cases")
        else:
            cases = min(cases, self.n_cases)

        keys = [k for k, v in self.iteritems() if isuv(v)]
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

        for i in xrange(cases):
            if count:
                table.cell(i)

            for v, fmt_ in zip(values, fmts):
                if fmt_ is None:
                    table.cell(v.x[i], fmt='%i')
                else:
                    table.cell(v[i], fmt=fmt_)

        return table

    def export(self, fn=None, fmt='%.10g', header=True, sort=False):
        """This method is deprecated. Use .save(), .save_pickled(),
        .save_txt() or .save_tex() instead.
        """
        msg = ("The Dataset.export() method is deprecated. Use .save(), "
               ".save_pickled(), .save_txt() or .save_tex() instead.")
        warn(msg, DeprecationWarning)

        if not isinstance(fn, basestring):
            fn = ui.ask_saveas(filetypes=[("Tab-separated values", '*.txt'),
                                          ("Tex table", '*.tex'),
                                          ("Pickle", '*.pickled')])
            if fn:
                print 'saving %r' % fn
            else:
                return

        ext = os.path.splitext(fn)[1][1:]
        if ext == 'pickled':
            with open(fn, 'wb') as fid:
                pickle.dump(self, fid)
        else:
            table = self.as_table(fmt=fmt, header=header, sort=sort)
            if ext in ['txt', 'tsv']:
                table.save_tsv(fn, fmt=fmt)
            elif ext == 'tex':
                table.save_tex(fn)
            else:
                table.save_tsv(fn, fmt=fmt)

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
            err = ("Eval needs expression of type unicode or str. Got "
                   "%s" % type(expression))
            raise TypeError(err)
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

    def compress(self, X, drop_empty=True, name='{name}', count='n',
                 drop_bad=False, drop=()):
        "Deprecated. Use .aggregate()."
        warn("Dataset.compress s deprecated; use Dataset.aggregate instead"
             "(with identical functionality).", DeprecationWarning)
        return self.aggregate(X, drop_empty, name, count, drop_bad, drop)

    def aggregate(self, X, drop_empty=True, name='{name}', count='n',
                  drop_bad=False, drop=()):
        """
        Return a Dataset with one case for each cell in X.

        Parameters
        ----------
        X : None | str | categorial
            Model defining cells to which to reduce cases. For None, the
            Dataset is reduced to a single case.
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

        Notes
        -----
        Handle mne Epoch objects by creating a list with an mne Evoked object
        for each cell.
        """
        if not drop_empty:
            raise NotImplementedError('drop_empty = False')
        if X:
            X = ascategorial(X, ds=self)
            self._check_n_cases(X, empty_ok=False)
        else:
            X = Factor('a' * self.n_cases)

        ds = Dataset(name=name.format(name=self.name), info=self.info)

        if count:
            x = filter(None, (np.sum(X == cell) for cell in X.cells))
            ds[count] = Var(x)

        for k, v in self.iteritems():
            if k in drop:
                continue
            try:
                if hasattr(v, 'aggregate'):
                    ds[k] = v.aggregate(X)
                else:
                    from mne import Epochs
                    if isinstance(v, Epochs):
                        evokeds = []
                        for cell in X.cells:
                            idx = (X == cell)
                            evoked = v[idx].average()
                            evokeds.append(evoked)
                        ds[k] = evokeds
                    else:
                        err = ("Unsupported value type: %s" % type(v))
                        raise TypeError(err)
            except:
                if drop_bad:
                    pass
                else:
                    raise

        return ds

    def copy(self):
        "ds.copy() returns an shallow copy of ds"
        ds = Dataset(name=self.name, info=self.info.copy())
        ds.update(self)
        return ds

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
    def N(self):
        warn("Dataset.N s deprecated; use Dataset.n_cases instead",
             DeprecationWarning)
        return self.n_cases

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

    def repeat(self, n, name='{name}'):
        """
        Analogous to :py:func:`numpy.repeat`. Returns a new Dataset with each
        row repeated ``n`` times.

        """
        ds = Dataset(name=name.format(name=self.name))
        for k, v in self.iteritems():
            ds[k] = v.repeat(n)
        return ds

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

    def save_tex(self, path=None, fmt='%.3g'):
        """Save the Dataset as TeX table.

        Parameters
        ----------
        path : None | str
            Target file name (if ``None`` is supplied, a save file dialog is
            displayed). If no extension is specified, '.tex' is appended.
        fmt : format string
            Formatting for scalar values.
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

        table = self.as_table(fmt=fmt, header=True)
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

        name = name.format(name=self.name)
        info = self.info.copy()

        if isvar(index):
            index = index.x

        ds = Dataset(name=name, info=info)
        for k, v in self.iteritems():
            ds[k] = v[index]

        return ds

    def subset(self, index, name='{name}'):
        "Deprecated: use .sub() method with identical functionality."
        warn("Dataset.subset is deprecated; use Dataset.sub instead"
             "(with identical functionality).", DeprecationWarning)
        return self.sub(index, name)

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

        if info:
            self.info.update(ds.info)



class Interaction(_Effect):
    """
    Represents an Interaction effect.


    Attributes
    ----------

    factors :
        List of all factors (i.e. nonbasic effects are broken up into
        factors).
    base :
        All effects.

    """
    _stype_ = "interaction"
    def __init__(self, base):
        """
        Usually not initialized directly but through operations on
        factors/vars.

        Parameters
        ----------
        base : list
            List of data-objects that form the basis of the interaction.

        """
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
            elif b._stype_ == "nested":  # TODO: nested effects
                self.base.append(b)
                if b.nestedin not in self.nestedin:
                    self.nestedin.append(b.nestedin)
            else:
                raise TypeError("Invalid type for Interaction: %r" % type(b))

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
        self.cells = list(itertools.product(*(f.cells for f in factors)))

        # effect coding
        codelist = [f.as_effects for f in self.base]
        codes = reduce(_effect_interaction, codelist)
        self.as_effects = codes
        self.beta_labels = ['?'] * codes.shape[1]  # TODO:

    def __repr__(self):
        names = [str(f.name) for f in self.base]
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
        return self.base.__contains__(item)

    def __iter__(self):
        for i in xrange(len(self)):
            yield tuple(b[i] for b in self.base)

    # numeric ---
    def __eq__(self, other):
        X = tuple((f == cell) for f, cell in zip (self.base, other))
        return np.all(X, axis=0)

    def __ne__(self, other):
        X = tuple((f != cell) for f, cell in zip (self.base, other))
        return np.any(X, axis=0)

    def as_factor(self):
        name = self.name.replace(' ', '')
        x = self.as_labels()
        return Factor(x, name)

    def as_cells(self):
        """All values as a list of tuples."""
        return [case for case in self]

    def as_labels(self, delim=' '):
        """All values as a list of strings.

        Parameters
        ----------
        delim : str
            Delimiter with which to join the elements of cells.
        """
        return map(delim.join, self)

    def compress(self, X):
        "Deprecated. Use .aggregate()."
        warn("Interaction.compress s deprecated; use Interaction.aggregate "
             "instead (with identical functionality).", DeprecationWarning)
        self.aggregate(X)

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
        self.I1 = X == i1;                self.I2 = X == i2

        if sub is not None:
            self.I1 = self.I1 * sub
            self.I2 = self.I2 * sub

        m1 = match.x[self.I1];          m2 = match.x[self.I2]
        self.s1 = np.argsort(m1);       self.s2 = np.argsort(m2)
        assert np.all(np.unique(m1) == np.unique(m2))
        self.name = "{n}({x1}-{x2})".format(n='{0}',
                                            x1=X.cells[i1],
                                            x2=X.cells[i2])

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
            return Factor(y1, Y.name, random=Y.random, labels=Y.cells,
                          sort=False)
        else:
            return Var(y1, Y.name)

    @property
    def N(self):
        return np.sum(self.I1)


def box_cox_transform(X, p, name=True):
    """
    :returns: a variable with the Box-Cox transform applied to X. With p==0,
        this is the log of X; otherwise (X**p - 1) / p

    :arg Var X: Source variable
    :arg float p: Parameter for Box-Cox transform

    """
    if isvar(X):
        if name is True:
            name = "Box-Cox(%s)" % X.name
        X = X.x
    else:
        if name is True:
            name = "Box-Cox(x)"

    if p == 0:
        y = np.log(X)
    else:
        y = (X ** p - 1) / p

    return Var(y, name=name)



def split(Y, n=2, name='{name}_{split}'):
    """
    Returns a Factor splitting Y in n categories with equal number of cases
    (e.g. n=2 for a median split)

    Y : array-like
        variable to split
    n : int
        number of categories
    name : str |

    """
    if isinstance(Y, Var):
        y = Y.x

    d = 100. / n
    percentile = np.arange(d, 100., d)
    values = [scipy.stats.scoreatpercentile(y, p) for p in percentile]
    x = np.zeros(len(y))
    for v in values:
        x += y > v

    fmt = {'name': Y.name}
    if n == 2:
        fmt['split'] = "mediansplit"
    else:
        fmt['split'] = "split%i" % n

    name = name.format(fmt)
    return Factor(x, name=name)



class NestedEffect(object):
    _stype_ = "nested"
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


class NonbasicEffect(object):
    _stype_ = "nonbasic"
    def __init__(self, effect_codes, factors, name, nestedin=[],
                 beta_labels=None):
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



class Model(object):
    """
    stores a list of effects which constitute a model for an ANOVA.

    a Model's data is exhausted by its. .effects list; all the rest are
    @properties.

    Accessing effects:
     - as list in Model.effects
     - with name as Model[name]

    """
    _stype_ = "model"
    def __init__(self, *x):
        """

        Parameters : factors | effects | models
            factors and secondary effects contained in the Model.
            Can also contain models, in which case all the models' effects
            will be added.

        """
        if len(x) == 0:
            raise ValueError("Model needs to be initialized with effects")

        # try to find effects in input
        self.effects = effects = EffectList()
        self._n_cases = n_cases = len(x[0])
        for e in x:
            # check that all effects have same number of cases
            if len(e) != n_cases:
                e0 = effects[0]
                err = ("All effects contained in a Model need to describe"
                       " the same number of cases. %r has %i cases, %r has"
                       " %i cases." % (e0.name, len(e0), e.name, len(e)))
                raise ValueError(err)

            #
            if iseffect(e):
                effects.append(e)
            elif ismodel(e):
                effects += e.effects
            else:
                err = ("Model needs to be initialized with effects (vars, "
                       "factors, interactions, ...) and/or models (got %s)"
                       % type(e))
                raise TypeError(err)

        # beta indices
        self.beta_index = beta_index = {}
        i = 1
        for e in effects:
            k = i + e.df
            beta_index[e] = slice(i, k)
            i = k

        # dfs
        self.df_total = df_total = n_cases - 1  # 1=intercept
        self.df = df = sum(e.df for e in effects)
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
            return "Model(%s)" % x

    def __str__(self):
        return str(self.get_table(cases=50))

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
            return Model(*(x[sub] for x in self.effects))

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
        return Model(*out)

    # numeric ---
    def __add__(self, other):
        return Model(self, other)

    def __mul__(self, other):
        return Model(self, other, self % other)

    def __mod__(self, other):
        out = []
        for e_self in self.effects:
            for e_other in Model(other).effects:
                out.append(e_self % e_other)
        return Model(*out)

    # repr ---
    @property
    def model_eq(self):
        return self.name

    def get_table(self, cases='all'):
        """
        :returns: the full model as a table.
        :rtype: :class:`fmtxt.Table`

        :arg cases: maximum number of cases (lines) to display.

        """
        full_model = self.full
        if cases == 'all':
            cases = len(full_model)
        else:
            cases = min(cases, len(full_model))
        n_cols = full_model.shape[1]
        table = fmtxt.Table('l' * n_cols)
        table.cell("Intercept")
        for e in self.effects:
            table.cell(e.name, width=e.df)

        # rules
        i = 2
        for e in self.effects:
            j = i + e.df - 1
            if e.df > 1:
                table.midrule((i, j))
            i = j + 1

        # data
        for line in full_model[:cases]:
            for i in line:
                table.cell(i)

        if cases < len(full_model):
            table.cell('...')
        return table

    # coding
    @LazyProperty
    def as_effects(self):
        return np.hstack((e.as_effects for e in self.effects))

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
        beta = dot(self.Xsinv, Y.x)
        return beta

    @LazyProperty
    def full(self):
        "returns the full model including an intercept"
        out = np.empty((self._n_cases, self.df + 1))

        # intercept
        out[:, 0] = np.ones(self._n_cases)
        self.full_index = {'I': slice(0, 1)}

        # effects
        i = 1
        for e in self.effects:
            j = i + e.df
            out[:, i:j] = e.as_effects
            self.full_index[e] = slice(i, j)
            i = j
        return out

    # checking model properties
    def check(self, v=True):
        "shortcut to check linear independence and orthogonality"
        return self.lin_indep(v) + self.orthogonal(v)

    def lin_indep(self, v=True):
        "Checks the Model for linear independence of its factors"
        msg = []
        ne = len(self.effects)
        codes = [e.as_effects for e in self.effects]
#        allok = True
        for i in range(ne):
            for j in range(i + 1, ne):
#                ok = True
                e1 = self.effects[i]
                e2 = self.effects[j]
                X = np.hstack((codes[i], codes[j]))
#                V0 = np.zeros(self._n_cases)
                # trash, trash, rank, trash = np.linalg.lstsq(X, V0)
                if rank(X) < X.shape[1]:
#                    ok = False
#                    allok = False
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

    # category access
    @property
    def unique(self):
        full = self.full
        unique_indexes = np.unique([tuple(i) for i in full])
        return unique_indexes

    @property
    def n_cat(self):
        return len(self.unique)

    def iter_cat(self):
        full = self.full
        for i in self.unique:
            cat = np.all(full == i, axis=1)
            yield cat

    def repeat(self, n):
        "Analogous to numpy repeat method"
        effects = [e.repeat(n) for e in self.effects]
        out = Model(effects)
        return out

    @LazyProperty
    def Xsinv(self):
        X = self.full
        XT = X.T
        Xsinv = dot(inv(dot(XT, X)), XT)
        return Xsinv


# ---NDVar dimensions---

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
            raise ValueError("Invalid argument snap=%r" % snap)

        if (t_next - time) < (time - t_prev):
            i = i_next
            time = t_next
        else:
            i = i_prev
            time = t_prev
    return i, time



class Dimension(object):
    """
    Base class for dimensions.
    """
    name = 'Dimension'
    adjacent = True

    def __getstate__(self):
        raise NotImplementedError

    def __setstate__(self, state):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError

    def __getitem__(self, index):
        """
         - int -> label or value for that location
         - [int] -> Dimension object with 1 location
         - [int, ...] -> Dimension object
        """
        raise NotImplementedError

    def _dimrepr_(self):
        return str(self.name)

    def dimindex(self, arg):
        raise NotImplementedError

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


class Scalar(Dimension):
    def __init__(self, name, values, unit=None):
        "Simple scalar dimension"
        if len(np.unique(values)) < len(values):
            raise ValueError("Dimension can not have duplicate values")
        self.name = name
        self.x = self.values = np.asarray(values)
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

    def _dimrepr_(self):
        values = str(list(self.values))
        r = '%r: %s' % (self.name, values)
        return r

    def __len__(self):
        return len(self.values)

    def __getitem__(self, index):
        if isinstance(index, int):
            return self.values[index]

        values = self.values[index]
        return Scalar(self.name, values)

    def dimindex(self, arg):
        if isinstance(arg, self.__class__):
            s_idx, a_idx = np.nonzero(self.values[:, None] == arg.values)
            arg = s_idx[np.argsort(a_idx)]
        elif np.isscalar(arg):
            nz = np.nonzero(self.values == arg)[0]
            if len(nz):
                arg = nz[0]
            else:
                raise ValueError("%s has no value %s" % (self.name, arg))
        else:
            arg = [self.dimindex(a) for a in arg]
        return arg

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

    def dimindex(self, arg):
        if isinstance(arg, tuple):
            start, stop = arg
            arg = np.logical_and(self.values >= start, self.values < stop)
        else:
            arg = super(Ordered, self).dimindex(arg)
        return arg


class Sensor(Dimension):
    """Dimension class for representing sensor information

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

    """
    name = 'sensor'
    adjacent = False

    def __init__(self, locs, names=None, groups=None, sysname=None,
                 proj2d='z root', connect_dist=1.75):
        """
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
        connect_dist : None | scalar
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor.

        Examples
        --------
        >>> sensors = [(0,  0,   0),
                       (0, -.25, -.45)]
        >>> sensor_dim = Sensor(sensors, names=["Cz", "Pz"])

        """
        self.sysname = sysname
        self.default_proj2d = proj2d
        self._connect_dist = connect_dist

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
        self._triangulations = {}

        # groups
        self.groups = groups

    def __getstate__(self):
        state = {'proj2d': self.default_proj2d,
                 'groups': self.groups,
                 'locs': self.locs,
                 'names': self.names,
                 'sysname': self.sysname}
        return state

    def __setstate__(self, state):
        locs = state['locs']
        names = state['names']
        groups = state['groups']
        sysname = state['sysname']
        proj2d = state['proj2d']

        self.__init__(locs, names, groups, sysname, proj2d)

    def __repr__(self):
        return "<Sensor n=%i, name=%r>" % (self.n, self.sysname)

    def __len__(self):
        return self.n

    def __getitem__(self, index):
        index = self.dimindex(index)
        if np.isscalar(index):
            return self.names[index]
        elif len(index) == 0:
            return None
        else:
            locs = self.locs[index]
            names = self.names[index]
            # TODO: groups
            return Sensor(locs, names, sysname=self.sysname,
                          proj2d=self.default_proj2d)

    def dimindex(self, arg):
        "Convert dimension indexes into numpy indexes"
        if isinstance(arg, basestring):
            idx = self.channel_idx[arg]
        elif isinstance(arg, Sensor):
            idx = np.array([self.names.index(name) for name in arg.names])
        elif np.iterable(arg):
            if (isinstance(arg, np.ndarray) and
                        issubclass(arg.dtype.type, (np.bool_, np.integer))):
                idx = arg
            else:
                idx = map(self._dimindex_map, arg)
        else:
            idx = arg
        return idx

    def _dimindex_map(self, name):
        "Convert any index to a proper int"
        if isinstance(name, basestring):
            return self.channel_idx[name]
        else:
            return int(name)

    def connectivity(self, connect_dist=None):
        """Construct a connectivity matrix in COOrdinate format

        Parameters
        ----------
        connect_dist : None | scalar
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor. If
            None, the default specified on initialization is used.

        Returns
        -------
        connectivity : scipy.sparse.coo_matrix
            Connectivity.

        See Also
        --------
        .neighbors() : Neighboring sensors for each sensor in a dictionary.
        """
        mult = connect_dist or self._connect_dist
        nb = self.neighbors(mult)

        # sparse representation (remove duplicate connections)
        nbs = collections.defaultdict(set)
        for k, vals in nb.iteritems():
            for v in vals:
                if k < v:
                    nbs[k].add(v)
                else:
                    nbs[v].add(k)

        # sparse matrix
        row = []
        col = []
        for k in sorted(nbs):
            v = sorted(nbs[k])
            row.extend([k] * len(v))
            col.extend(sorted(v))
        data = np.ones(len(row))
        N = len(self)
        connectivity = coo_matrix((data, (row, col)), shape=(N, N))
        return connectivity

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

    def get_locs_2d(self, proj='default', extent=1):
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
        """
        if proj == 'default':
            proj = self.default_proj2d

        if proj is None:
            proj = 'z+'

        index = (proj, extent)
        if index in self._transformed:
            return self._transformed[index]

        if proj in ['cone', 'lower cone', 'z root']:

            # fit the 3d sensor locations to a sphere with center (cx, cy, cz)
            # and radius r

            # error function
            def err(params):
                r, cx, cy, cz = params
                return   (self.locs[:, 0] - cx) ** 2 \
                       + (self.locs[:, 1] - cy) ** 2 \
                       + (self.locs[:, 2] - cz) ** 2 \
                       - r ** 2

            # initial guess of sphere parameters (radius and center)
            params = (1, 0, 0, 0)
            # do fit
            (r, cx, cy, cz), _ = leastsq(err, params)

            # center the sensor locations based on the sphere and scale to
            # radius 1
            sphere_center = np.array((cx, cy, cz))
#            logging.debug("Sensor sphere projection: %r, %r" % (sphere_center, r))
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
                z = max(locs3d[:, 2]) - locs3d[:, 2]  # distance form top
                r = np.sqrt(z)  # desired 2d radius
                r_xy = np.sqrt(np.sum(locs3d[:, :2] ** 2, 1))  # current radius in xy
                idx = (r_xy != 0)  # avoid zero division
                F = r[idx] / r_xy[idx]  # stretching Factor accounting for current r
                locs2d[idx, :] *= F[:, None]

        else:
            pattern = re.compile('([xyz])([+-])')
            match = pattern.match(proj.lower())
            if match:
                ax = match.group(1)
                sign = match.group(2)
                if ax == 'x':
                    locs2d = np.copy(self.locs[:, 1:])
                    if sign == '-':
                        locs2d[:, 0] = -locs2d[:, 0]
                elif ax == 'y':
                    locs2d = np.copy(self.locs[:, [0, 2]])
                    if sign == '+':
                        locs2d[:, 0] = -locs2d[:, 0]
                elif ax == 'z':
                    locs2d = np.copy(self.locs[:, :2])
                    if sign == '-':
                        locs2d[:, 1] = -locs2d[:, 1]
            else:
                raise ValueError("invalid proj kwarg: %r" % proj)

        # correct extent
        if extent:
            locs2d -= np.min(locs2d, axis=0)  # move to bottom left
            locs2d /= (np.max(locs2d) / extent)  # scale to extent
            locs2d += (extent - np.max(locs2d, axis=0)) / 2  # center

        # save for future access
        self._transformed[index] = locs2d
        return locs2d

    def get_tri(self, proj, resolution, frame):
        """
        Returns delaunay triangulation and meshgrid objects
        (for projecting sensor maps to ims)

        Based on matplotlib.mlab.griddata function
        """
        locs = self.get_locs_2d(proj)
        from matplotlib.delaunay import Triangulation
        tri = Triangulation(locs[:, 0], locs[:, 1])

        emin = -frame
        emax = 1 + frame
        x = np.linspace(emin, emax, resolution)
        xi, yi = np.meshgrid(x, x)

        return tri, xi, yi

    def get_im_for_topo(self, Z, proj='default', res=100, frame=.03, interp='linear'):
        """
        Returns an im for an arrray in sensor space X

        Based on matplotlib.mlab.griddata function
        """
        if proj == 'default':
            proj = self.default_proj2d

        index = (proj, res, frame)

        tri, xi, yi = self._triangulations.setdefault(index, self.get_tri(*index))

        if interp == 'nn':
            interp = tri.nn_interpolator(Z)
            zo = interp(xi, yi)
        elif interp == 'linear':
            interp = tri.linear_interpolator(Z)
            zo = interp[yi.min():yi.max():complex(0, yi.shape[0]),
                        xi.min():xi.max():complex(0, xi.shape[1])]
        else:
            raise ValueError("interp keyword must be one of"
            " 'linear' (for linear interpolation) or 'nn'"
            " (for natural neighbor interpolation). Default is 'nn'.")
        # mask points on grid outside convex hull of input data.
        if np.any(np.isnan(zo)):
            zo = np.ma.masked_where(np.isnan(zo), zo)
        return zo

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
            return slice(None)

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

    def neighbors(self, connect_dist=None):
        """Find neighboring sensors.

        Parameters
        ----------
        connect_dist : None | scalar
            For each sensor, neighbors are defined as those sensors within
            ``connect_dist`` times the distance of the closest neighbor. If
            None, the default specified on initialization is used.

        Returns
        -------
        neighbors : dict
            Dictionaries whose keys are sensor indices, and whose values are
            lists of neighbors represented as sensor indices.

        See Also
        --------
        .connectivity() : neighbor connectivity as sparse matrix
        """
        connect_dist = connect_dist or self._connect_dist
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


class SourceSpace(Dimension):
    """
    Indexing
    --------

    besides numpy indexing, the following indexes are possible:

     - mne Label objects
     - 'lh' or 'rh' to select an entire hemisphere

    """
    name = 'source'
    adjacent = False

    def __init__(self, vertno, subject=None, src=None, subjects_dir=None,
                 connectivity=None):
        """Create mne source space dimension.

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
        connectivity : None | sparse matrix
            Cached source space connectivity.
        """
        self.vertno = vertno
        self.lh_vertno = vertno[0]
        self.rh_vertno = vertno[1]
        self.lh_n = len(self.lh_vertno)
        self.rh_n = len(self.rh_vertno)
        self.subject = subject
        self.src = src
        self.subjects_dir = subjects_dir
        self._connectivity = connectivity

    def __getstate__(self):
        state = {'vertno': self.vertno, 'subject': self.subject,
                 'src': self.src, 'subjects_dir': self.subjects_dir}
        return state

    def __setstate__(self, state):
        vertno = state['vertno']
        subject = state['subject']
        src = state.get('src', None)
        subjects_dir = state.get('subjects_dir', None)
        self.__init__(vertno, subject, src, subjects_dir)

    def __repr__(self):
        return "<dim SourceSpace: %i (lh), %i (rh)>" % (self.lh_n, self.rh_n)

    def __len__(self):
        return self.lh_n + self.rh_n

    def __getitem__(self, index):
        if self._connectivity is None:
            connectivity = None
        else:
            c = self._connectivity
            int_index = np.arange(len(self))[index]
            idx = np.logical_and(np.in1d(c.row, int_index, True),
                                 np.in1d(c.col, int_index, True))
            row = c.row[idx]
            col = c.col[idx]
            data = c.data[idx]
            connectivity = coo_matrix((data, (row, col)))

        vert = np.hstack(self.vertno)
        hemi = np.zeros(len(vert))
        hemi[self.lh_n:] = 1

        vert = vert[index]
        hemi = hemi[index]

        new_vert = (vert[hemi == 0], vert[hemi == 1])
        dim = SourceSpace(new_vert, self.subject, self.src, self.subjects_dir,
                          connectivity)
        return dim

    def connectivity(self):
        """Create source space connectivity
        """
        if self._connectivity is not None:
            return self._connectivity


        if any(x is None for x in (self.src, self.subject, self.subjects_dir)):
            err = ("In order for a SourceSpace dimension to provide "
                   "connectivity information it needs to be initialized with "
                   "src, subject and subjects_dir parameters")
            raise ValueError(err)

        import mne
        pattern = os.path.join('{subjects_dir}', '{subject}', 'bem',
                               '{subject}-{src}-src.fif')
        path = pattern.format(subjects_dir=self.subjects_dir,
                              subject=self.subject, src=self.src)
        src = mne.read_source_spaces(path)

        lh_vertno = np.intersect1d(self.vertno[0], src[0]['vertno'], True)
        pt_in_use = np.in1d(src[0]['use_tris'], lh_vertno).reshape((-1, 3))
        tri_in_use = np.any(pt_in_use, axis=1)
        lh_tris = src[0]['use_tris'][tri_in_use]

        rh_vertno = np.intersect1d(self.vertno[1], src[1]['vertno'], True)
        pt_in_use = np.in1d(src[1]['use_tris'], rh_vertno).reshape((-1, 3))
        tri_in_use = np.any(pt_in_use, axis=1)
        rh_tris = src[1]['use_tris'][tri_in_use]

        tris = np.vstack((lh_tris, rh_tris))
        c = mne.spatial_tris_connectivity(tris)
        self._connectivity = c
        return c

    def dimindex(self, obj):
        import mne
        if isinstance(obj, (mne.Label, mne.label.BiHemiLabel)):
            return self.label_index(obj)
        elif isinstance(obj, str):
            if obj == 'lh':
                if self.lh_n:
                    return slice(None, self.lh_n)
                else:
                    raise IndexError("lh is empty")
            if obj == 'rh':
                if self.rh_n:
                    return slice(self.lh_n, None)
                else:
                    raise IndexError("rh is empty")
            else:
                raise IndexError('%r' % obj)
        elif isinstance(obj, SourceSpace):
            if (np.array_equal(self.lh_vertno, obj.lh_vertno) and
                np.array_equal(self.rh_vertno, obj.rh_vertno)):
                return slice(None)
            else:
                lh_index = np.in1d(self.lh_vertno, obj.lh_vertno, True)
                rh_index = np.in1d(self.rh_vertno, obj.rh_vertno, True)
                index = np.hstack((lh_index, rh_index))
                return index
        else:
            return obj

    def _hemilabel_index(self, label):
        if label.hemi == 'lh':
            stc_vertices = self.vertno[0]
            base = 0
        else:
            stc_vertices = self.vertno[1]
            base = len(self.vertno[0])

        idx = np.nonzero(map(label.vertices.__contains__, stc_vertices))[0]
        return idx + base

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
            raise ValueError("Different subject")
        elif self.src != other.src:
            raise ValueError("Different src")
        elif self.subjects_dir != other.subjects_dir:
            raise ValueError("Different subjects_dir")
        lh = np.intersect1d(self.lh_vertno, other.lh_vertno, True)
        rh = np.intersect1d(self.rh_vertno, other.rh_vertno, True)
        out = SourceSpace([lh, rh], self.subject, self.src, self.subjects_dir)
        return out

    def label_index(self, label):
        """Returns the index for a label

        Parameters
        ----------
        label : Label | BiHemiLabel
            The label (as created for example by mne.read_label). If the label
            does not match any sources in the SourceEstimate, a ValueError is
            raised.
        """
        if label.hemi == 'both':
            lh_idx = self._hemilabel_index(label.lh)
            rh_idx = self._hemilabel_index(label.rh)
            idx = np.hstack((lh_idx, rh_idx))
        else:
            idx = self._hemilabel_index(label)

        if len(idx) == 0:
            raise ValueError('No vertices match the label in the stc file')

        return idx


class UTS(Dimension):
    """Dimension object for representing uniform time series

    Special Indexing
    ----------------

    (tstart, tstop) : tuple
        Restrict the time to the indicated window (either end-point can be
        None).

    """
    name = 'time'
    unit = 's'

    def __init__(self, tmin, tstep, nsamples):
        """UTS dimension

        Parameters
        ----------
        tmin : scalar
            First time point (inclusive).
        tstep : scalar
            Time step between samples.
        nsamples : int
            Number of samples.
        """
        self.tmin = tmin
        self.tstep = tstep
        self.nsamples = nsamples = int(nsamples)
        self.x = self.times = tmin + np.arange(nsamples) * tstep
        self.tmax = self.times[-1]

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

    def _dimrepr_(self):
        tmax = self.times[-1]
        sfreq = 1. / self.tstep
        r = '%r: %.3f - %.3f s, %s Hz' % (self.name, self.tmin, tmax, sfreq)
        return r

    def __len__(self):
        return len(self.times)

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

            if index.step is None:
                tstep = self.tstep
            else:
                tstep = self.tstep * index.step
        else:
            err = ("Unupported index: %r" % index)
            raise TypeError(err)

        return UTS(tmin, tstep, nsamples)

    def dimindex(self, arg):
        if np.isscalar(arg):
            i, _ = find_time_point(self.times, arg)
            return i
        elif isinstance(arg, UTS):
            idxs = (self.times[:, None] == arg.times)
            if not np.all(np.any(idxs, 0)):
                err = ("The index dimension has time values not in self")
                raise DimensionMismatchError(err)
            start = self.dimindex(arg.tmin)
            step = int(round(arg.tstep / self.tstep))
            stop = start + len(arg) * step
            arg = slice(start, stop, step)
            return arg
        elif isinstance(arg, tuple) and len(arg) == 2:
            tstart, tstop = arg
            if (tstart is not None) and (tstop is not None) and (tstart >= tstop):
                raise ValueError("tstart must be smaller than tstop")

            if tstart is None:
                start = None
            else:
                start, _ = find_time_point(self.times, tstart, 'up')

            if tstop is None:
                stop = None
            else:
                stop, _ = find_time_point(self.times, tstop, 'up')

            s = slice(start, stop)
            return s
        else:
            return arg

    def index(self, time, rnd='closest'):
        """Find the index for a time point

        Parameters
        ----------
        time : scalar
            Time point for which to find an index.
        rnd : 'down' | 'closest' | 'up'
            Rounding: how to handle time values that do not have an exact
            match. Round 'up', 'down', or to the 'closest' neighbor.
        """
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
        idx = self.times[:, None] == dim.times
        if np.all(np.any(idx, 0)):
            return dim
        elif np.all(np.any(idx, 1)):
            return self

        times = np.intersect1d(self.times, dim.times)
        tmin = times[0]
        # guess tstep :(
        tstep_est = np.mean(np.unique(np.diff(times)))
        x = int(round(tstep_est / self.tstep))
        tstep = self.tstep * x
        nsamples = len(times)
        return UTS(tmin, tstep, nsamples)


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

def corr(x, dim='sensor', obs='time', neighbors=None, name='{name}_r_nbr'):
    """Calculate Neighbor correlation

    Parameter
    ---------
    x : NDVar
        The data.
    dim : str
        Dimension over which to correlate neighbors.
    """
    dim_obj = x.get_dim(dim)
    neighbors = neighbors or dim_obj.neighbors()

    data = x.get_data((dim, obs))
    cc = np.corrcoef(data)
    y = np.zeros(len(dim_obj))
    for i in xrange(len(dim_obj)):
        y[i] = np.mean(cc[i, neighbors[i]])

    xname = x.name or ''
    name = name.format(name=xname)
    info = cs.set_info_cs(x.info, cs.stat_info('r'))
    out = NDVar(y, (dim_obj,), info=info, name=name)
    return out


def cwt_morlet(Y, freqs, use_fft=True, n_cycles=7.0, zero_mean=False,
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
        Compute convolution with FFT or temoral convolution.
    n_cycles: float | array of float
        Number of cycles. Fixed number or one per frequency.
    zero_mean : bool
        Make sure the wavelets are zero mean.
    out : 'complex' | 'magnitude' | 'phase'
        xxx

    Returns
    -------
    tfr : 3D array
        Time Frequency Decompositions (n_signals x n_frequencies x n_times)
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
    import mne
    axis = data.get_axis('time')
    old_sfreq = 1.0 / data.time.tstep
    x = mne.filter.resample(data.x, sfreq, old_sfreq, npad, axis, window)
    tstep = 1. / sfreq
    time = UTS(data.time.tmin, tstep, x.shape[axis])
    dims = data.dims[:axis] + (time,) + data.dims[axis + 1:]
    return NDVar(x, dims=dims, info=data.info, name=data.name)
