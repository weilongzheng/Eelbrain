'''Statistical tests for NDVars

Common Attributes
-----------------

The following attributes are always present. For ANOVA, they are lists with the
corresponding items for different effects.

t/f/... : NDVar
    Map of the statistical parameter.
p_uncorrected : NDVar
    Map of uncorrected p values.
p : NDVar | None
    Map of corrected p values (None if no correct was applied).
clusters : Dataset | None
    Table of all the clusters found (None if no clusters were found, or if no
    clustering was performed).
'''
from __future__ import division

from datetime import timedelta
from itertools import chain, izip
from math import ceil, floor
from multiprocessing import Process, Queue, cpu_count
from multiprocessing.sharedctypes import RawArray
import operator
import os
import re
from time import time as current_time

import numpy as np
import scipy.stats
from scipy import ndimage

from .. import _colorspaces as _cs
from .._utils import logger, LazyProperty
from .._data_obj import (ascategorial, asmodel, asndvar, asvar, assub, Dataset,
                         NDVar, Var, Celltable, cellname, combine, Categorial,
                         UTS)
from .glm import _nd_anova
from .opt import merge_labels
from .permutation import resample, _resample_params
from .stats import ftest_f, ftest_p
from .test import star_factor


__test__ = False

# toggle multiprocessing for _ClusterDist
multiprocessing = 1


class _Result(object):
    _state_common = ('Y', 'match', 'sub', 'samples', 'name', 'pmin', '_cdist')
    _state_specific = ()

    @property
    def _attributes(self):
        return self._state_common + self._state_specific

    def __getstate__(self):
        state = {name: getattr(self, name, None) for name in self._attributes}
        return state

    def __setstate__(self, state):
        for k, v in state.iteritems():
            setattr(self, k, v)
        self._expand_state()

    def __repr__(self):
        temp = "<%s %%s>" % self.__class__.__name__

        args = self._repr_test_args()
        if self.sub:
            args.append(', sub=%r' % self.sub)
        if self._cdist:
            args += self._cdist._repr_test_args(self.pmin)
            args += self._cdist._repr_clusters()

        out = temp % ', '.join(args)
        return out

    def _repr_test_args(self):
        """List of strings describing parameters unique to the test, to be
        joined by comma
        """
        raise NotImplementedError()

    def _expand_state(self):
        "override to create secondary results"
        cdist = self._cdist
        if cdist is None:
            self.tfce_map = None
            self.p = None
        else:
            self.tfce_map = cdist.tfce_map
            self.p = cdist.probability_map

    def masked_parameter_map(self, pmin=0.05, **sub):
        """Create a copy of the parameter map masked by significance

        Parameters
        ----------
        pmin : None | scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, pmin=None includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map wherever p <= pmin
            and 0 everywhere else.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.masked_parameter_map(pmin, **sub)

    @LazyProperty
    def clusters(self):
        if self._cdist is None:
            return None
        else:
            return self._clusters(None, True)

    def _clusters(self, pmin=None, maps=False, **sub):
        """Find significant regions as clusters

        Parameters
        ----------
        pmin : None | scalar, 1 >= p  >= 0
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps : bool
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default False).

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if self._cdist is None:
            err = ("Test results have no clustering (set samples to an int "
                   " >= 0 to find clusters")
            raise RuntimeError(err)
        return self._cdist.clusters(pmin, maps, **sub)

    def find_peaks(self):
        """Find peaks in a threshold-free cluster distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.find_peaks()

    def compute_probability_map(self, **sub):
        """Compute a probability map

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        return self._cdist.compute_probability_map(**sub)


class t_contrast_rel(_Result):

    _state_specific = ('X', 'contrast', 't')

    def __init__(self, Y, X, contrast, match=None, sub=None, ds=None,
                 samples=None, pmin=None, tmin=None, tfce=False, tstart=None,
                 tstop=None, dist_dim=(), parc=(), dist_tstep=None,
                 **criteria):
        """Contrast with t-values from multiple comparisons

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : categorial
            Model containing the cells which are compared with the contrast.
        contrast : str
            Contrast specification: see Notes.
        match : Factor
            Match cases for a repeated measures test.
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use a t-value equivalent to an
            uncorrected p-value for a related samples t-test (with df =
            len(match.cells) - 1).
        tmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.

        Notes
        -----
        Contrast definitions can contain:

         - comparisons using ">" and "<", e.g. ``"cell1 > cell0"``.
         - numpy functions, e.g. ``min(...)``.
         - prefixing a function or comparison with ``+`` or ``-`` makes the
           relevant comparison one-tailed by setting all values of the opposite
           sign to zero (e.g., ```"+a>b"``` sets all data points where a<b to
           0.

        So for example, to find cluster where both of two pairwise comparisons
        are reliable, one could use ``"min(a1 > a0, b1 > b0)"``

        If X is an interaction, interaction cells are specified with "|", e.g.
        ``"a1 | b > a0 | b"``.
        """
        test_name = "t-contrast"
        ct = Celltable(Y, X, match, sub, ds=ds, coercion=asndvar)
        indexes = ct.data_indexes

        # setup contrast
        contrast_ = _parse_t_contrast(contrast)
        n_buffers, cells_in_contrast = _t_contrast_rel_properties(contrast_)
        pcells, mcells = _t_contrast_rel_expand_cells(cells_in_contrast, ct.cells)
        tail_ = contrast_[1]
        if tail_ is None:
            tail = 0
        elif tail_ == '+':
            tail = 1
        elif tail_ == '-':
            tail = -1
        else:
            raise RuntimeError("Invalid tail in parse: %s" % repr(tail_))

        # buffer memory allocation
        shape = ct.Y.shape[1:]
        buff = np.empty((n_buffers,) + shape)

        # original data
        data = _t_contrast_rel_data(ct.Y.x, indexes, pcells, mcells)
        tmap = _t_contrast_rel(contrast_, data, buff)
        del buff
        dims = ct.Y.dims[1:]
        t = NDVar(tmap, dims, {}, 't')

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                df = len(ct.match.cells) - 1
                threshold = _ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't', test_name,
                                 tstart, tstop, criteria, dist_dim, parc,
                                 dist_tstep)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                # buffer memory allocation
                shape = cdist.Y_perm.shape[1:]
                buff = np.empty((n_buffers,) + shape)
                tmap_ = np.empty(shape)
                for Y_ in resample(cdist.Y_perm, samples, unit=ct.match):
                    data = _t_contrast_rel_data(Y_.x, indexes, pcells, mcells)
                    _t_contrast_rel(contrast_, data, buff, tmap_)
                    cdist.add_perm(tmap_)

        # store attributes
        self.Y = ct.Y.name
        self.X = ct.X.name
        self.contrast = contrast
        if ct.match:
            self.match = ct.match.name
        else:
            self.match = None
        if sub is None or isinstance(sub, basestring):
            self.sub = sub
        else:
            self.sub = "<array>"
        self.samples = samples
        self.pmin = pmin
        self.tmin = tmin
        self.tfce = tfce
        self.name = test_name
        self.t = t
        self._cdist = cdist

        self._expand_state()

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X), repr(self.contrast)]
        if self.match:
            args.append('match=%r' % self.match)
        return args


def _parse_cell(cell_name):
    "Parse a cell name for t_contrast"
    cell = tuple(s.strip() for s in cell_name.split('|'))
    if len(cell) == 1:
        return cell[0]
    else:
        return cell


def _parse_t_contrast(contrast):
    """Parse a string specifying a t-contrast into nested instruction tuples

    Parameters
    ----------
    contrast : str
        Contrast specification string.

    Returns
    -------
    compiled_contrast : tuple
        Nested tuple composed of:
        Comparisons:  ``('comp', tail, c1, c0)`` and
        Functions:  ``('func', tail, [arg1, arg2, ...])``
        where ``arg1`` etc. are in turn comparisons and functions.
    """
    depth = 0
    start = 0
    if not '(' in contrast:
        m = re.match("\s*([+-]*)\s*([\w\|*]+)\s*([<>])\s*([\w\|*]+)", contrast)
        if m:
            clip, c1, direction, c0 = m.groups()
            if direction == '<':
                c1, c0 = c0, c1
            c1 = _parse_cell(c1)
            c0 = _parse_cell(c0)
            return ('comp', clip or None, c1, c0)

    for i, c in enumerate(contrast):
        if c == '(':
            if depth == 0:
                prefix = contrast[start:i]
                i_open = i + 1
                items = []
            depth += 1
        elif c == ',':
            if depth == 0:
                raise
            elif depth == 1:
                item = _parse_t_contrast(contrast[i_open:i])
                items.append(item)
                i_open = i + 1
        elif c == ')':
            depth -= 1
            if depth == 0:
                item = _parse_t_contrast(contrast[i_open:i])
                items.append(item)

                m = re.match("\s*([+-]*)\s*(\w+)", prefix)
                if m is None:
                    raise ValueError("uninterpretable prefix: %r" % prefix)
                clip, func_name = m.groups()
                func = getattr(np, func_name)

                return ('func', clip or None, func, items)
            elif depth == -1:
                err = "Invalid ')' at position %i of %r" % (i, contrast)
                raise ValueError(err)


def _t_contrast_rel_properties(item):
    """Find properties of a compiled t-contrast

    Parameters
    ----------
    item : tuple
        Contrast specification.

    Returns
    -------
    n_buffers : int
        Number of buffer maps needed.
    cells : set
        names of all cells that occur in the contrast.
    """
    if item[0] == 'func':
        _, _, _, items_ = item
        local_buffers = len(items_)
        cells = set()
        for i, item_ in enumerate(items_):
            available_buffers = local_buffers - i - 1
            needed_buffers, cells_ = _t_contrast_rel_properties(item_)
            additional_buffers = needed_buffers - available_buffers
            if additional_buffers > 0:
                local_buffers += additional_buffers
            cells.update(cells_)
        return local_buffers, cells
    else:
        return 0, set(item[2:])


def _t_contrast_rel_expand_cells(cells, all_cells):
    """Find cells that are an average of other cells

    Parameters
    ----------
    cells : set
        Cells occurring in the contrast.
    all_cells : tuple
        All cells in the data.

    Returns
    -------
    primary_cells : set
        All cells that occur directly in the data.
    mean_cells : dict
        ``{name: components}`` dictionary (components being a tuple with all
        cells to be averaged).
    """
    # check all cells have same number of components
    ns = set(1 if isinstance(cell, str) else len(cell) for cell in all_cells)
    ns.update(1 if isinstance(cell, str) else len(cell) for cell in cells)
    if len(ns) > 1:
        msg = ("Not all cells have the same number of components: %s" %
               str(tuple(cells) + tuple(all_cells)))
        raise ValueError(msg)

    primary_cells = set()
    mean_cells = {}
    for cell in cells:
        if cell in all_cells:
            primary_cells.add(cell)
        elif isinstance(cell, str):
            if cell != '*':
                raise ValueError("%s not in all_cells" % repr(cell))
            mean_cells[cell] = all_cells
            primary_cells.update(all_cells)
        elif not '*' in cell:
            msg = "Contrast contains cell not in data: %s" % repr(cell)
            raise ValueError(msg)
        else:
            # find cells that should be averaged ("base")
            base = tuple(cell_ for cell_ in all_cells if
                         all(i in (i_, '*') for i, i_ in izip(cell, cell_)))
            if len(base) == 0:
                raise ValueError("No cells in data match %s" % repr(cell))
            mean_cells[cell] = base
            primary_cells.update(base)

    return primary_cells, mean_cells


def _t_contrast_rel_data(y, indexes, cells, mean_cells):
    "Create {cell: data} dictionary"
    data = {}
    for cell in cells:
        index = indexes[cell]
        data[cell] = y[index]

    for name, cells_ in mean_cells.iteritems():
        cell = cells_[0]
        x = data[cell].copy()
        for cell in cells_[1:]:
            x += data[cell]
        x /= len(cells_)
        data[name] = x

    return data


def _t_contrast_rel(item, data, buff=None, out=None):
    "Execute a t_contrast (recursive)"
    if item[0] == 'func':
        _, clip, func, items_ = item
        tmaps = buff[:len(items_)]
        for i, item_ in enumerate(items_):
            if buff is None:
                buff_ = None
            else:
                buff_ = buff[i + 1:]
            _t_contrast_rel(item_, data, buff_, tmaps[i])
        tmap = func(tmaps, axis=0, out=out)
    else:
        _, clip, c1, c0 = item
        tmap = _t_rel(data[c1], data[c0], out)

    if clip is not None:
        if clip == '+':
            a_min = 0
            a_max = tmap.max() + 1
        elif clip == '-':
            a_min = tmap.min() - 1
            a_max = 0
        tmap.clip(a_min, a_max, tmap)

    return tmap


class corr(_Result):
    """Correlation

    Attributes
    ----------
    r : NDVar
        Correlation (with threshold contours).
    """
    _state_specific = ('X', 'norm', 'n', 'df', 'r')

    def __init__(self, Y, X, norm=None, sub=None, ds=None, samples=None,
                 pmin=None, rmin=None, tfce=False, tstart=None, tstop=None,
                 match=None, dist_dim=(), parc=(), dist_tstep=None,
                 **criteria):
        """Correlation.

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : continuous
            The continuous predictor variable.
        norm : None | categorial
            Categories in which to normalize (z-score) X.
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use an r-value equivalent to an
            uncorrected p-value.
        rmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        match : None | categorial
            When permuting data, only shuffle the cases within the categories
            of match.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        sub = assub(sub, ds)
        Y = asndvar(Y, sub=sub, ds=ds)
        if not Y.has_case:
            msg = ("Dependent variable needs case dimension")
            raise ValueError(msg)
        X = asvar(X, sub=sub, ds=ds)
        if norm is not None:
            norm = ascategorial(norm, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        name = "%s corr %s" % (Y.name, X.name)

        # Normalize by z-scoring the data for each subject
        # normalization is done before the permutation b/c we are interested in
        # the variance associated with each subject for the z-scoring.
        Y = Y.copy()
        if norm is not None:
#             Y.x = Y.x.reshape((n, -1))
            for cell in norm.cells:
                idx = (norm == cell)
                Y.x[idx] = scipy.stats.zscore(Y.x[idx], None)

        # subtract the mean from Y and X so that this can be omitted during
        # permutation
        Y -= Y.summary('case')
        X = X - X.mean()

        n = len(Y)
        df = n - 2

        rmap = _corr(Y.x, X.x)

        if samples is None:
            cdist = None
            r0, r1, r2 = _rtest_r((.05, .01, .001), df)
            info = _cs.stat_info('r', r0, r1, r2)
        else:
            # threshold
            if sum((pmin is not None, rmin is not None, tfce)) > 1:
                msg = "Only one of pmin, rmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = _rtest_r(pmin, df)
            elif rmin is not None:
                threshold = abs(rmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            cdist = _ClusterDist(Y, samples, threshold, 0, 'r', name,
                                 tstart, tstop, criteria, dist_dim, parc,
                                 dist_tstep)
            cdist.add_original(rmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples, unit=match):
                    rmap_ = _corr(Y_.x, X.x)
                    cdist.add_perm(rmap_)
            info = _cs.stat_info('r', threshold)

        # compile results
        dims = Y.dims[1:]
        r = NDVar(rmap, dims, info, name)

        # store attributes
        self.Y = Y.name
        self.X = X.name
        self.norm = None if norm is None else norm.name
        if sub is None or isinstance(sub, basestring):
            self.sub = sub
        else:
            self.sub = "<array>"
        if match:
            self.match = match.name
        else:
            self.match = None
        self.samples = samples
        self.pmin = pmin
        self.rmin = rmin
        self.tfce = tfce
        self.name = name
        self._cdist = cdist

        self.n = n
        self.df = df
        self.r = r

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        r = self.r

        # uncorrected probability
        pmap = _rtest_p(r.x, self.df)
        info = _cs.sig_info()
        p_uncorrected = NDVar(pmap, r.dims, info, 'p_uncorrected')
        self.p_uncorrected = p_uncorrected

        self.r_p_uncorrected = [[r, r]]
        if self.samples:
            self.r_p = self._default_plot_obj = [[r, self.p]]
        else:
            self._default_plot_obj = self.r_p_uncorrected

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X)]
        if self.norm:
            args.append('norm=%r' % self.norm)
        return args


def _corr(y, x):
    """Correlation parameter map

    Parameters
    ----------
    y : array_like, shape = (n_cases, ...)
        Dependent variable with case in the first axis and case mean zero.
    x : array_like, shape = (n_cases, )
        Covariate.

    Returns
    -------
    r : array, shape = (...)
        The correlation. Occurrence of NaN due to 0 variance in either y or x
        are replaced with 0.
    """
    x = x.reshape((len(x),) + (1,) * (y.ndim - 1))
    r = np.sum(y * x, axis=0) / (np.sqrt(np.sum(y ** 2, axis=0)) *
                                 np.sqrt(np.sum(x ** 2, axis=0)))
    # replace NaN values
    isnan = np.isnan(r)
    if np.any(isnan):
        if np.isscalar(r):
            r = 0
        else:
            r[isnan] = 0
    return r

def _corr_alt(y, x):
    n = len(y)
    cov = np.sum(x * y, axis=0) / (n - 1)
    r = cov / (np.std(x, axis=0) * np.std(y, axis=0))
    return r


def _rtest_p(r, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    r = np.asanyarray(r)
    t = r * np.sqrt(df / (1 - r ** 2))
    p = _ttest_p(t, df)
    return p


def _rtest_r(p, df):
    # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
    p = np.asanyarray(p)
    t = _ttest_t(p, df)
    r = t / np.sqrt(df + t ** 2)
    return r


class ttest_1samp(_Result):
    """Element-wise one sample t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    _state_specific = ('popmean', 'tail', 'n', 'df', 't', 'y', 'diff')

    def __init__(self, Y, popmean=0, match=None, sub=None, ds=None, tail=0,
                 samples=None, pmin=None, tmin=None, tfce=False, tstart=None,
                 tstop=None, dist_dim=(), parc=(), dist_tstep=None,
                 **criteria):
        """Element-wise one sample t-test

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        popmean : scalar
            Value to compare Y against (default is 0).
        match : None | categorial
            Combine data for these categories before testing.
        sub : None | index-array
            Perform test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use a t-value equivalent to an
            uncorrected p-value.
        tmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        ct = Celltable(Y, match=match, sub=sub, ds=ds, coercion=asndvar)

        test_name = '1-Sample t-Test'
        n = len(ct.Y)
        df = n - 1
        y = ct.Y.summary()
        tmap = _t_1samp(ct.Y.x, popmean)
        if popmean:
            diff = y - popmean
            if np.any(diff < 0):
                diff.info['cmap'] = 'xpolar'
        else:
            diff = y

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = _ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            if popmean:
                y_perm = ct.Y - popmean
            else:
                y_perm = ct.Y
            n_samples, samples = _resample_params(len(y_perm), samples)
            cdist = _ClusterDist(y_perm, n_samples, threshold, tail, 't',
                                 test_name, tstart, tstop, criteria, dist_dim,
                                 parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples, sign_flip=True):
                    tmap_ = _t_1samp(Y_.x, 0)
                    cdist.add_perm(tmap_)

        # NDVar map of t-values
        dims = ct.Y.dims[1:]
        t0, t1, t2 = _ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        # store attributes
        self.Y = ct.Y.name
        self.popmean = popmean
        if ct.match:
            self.match = ct.match.name
        else:
            self.match = None
        if sub is None or isinstance(sub, basestring):
            self.sub = sub
        else:
            self.sub = "<unsaved array>"
        self.tail = tail
        self.samples = samples
        self.pmin = pmin
        self.tmin = tmin
        self.tfce = tfce

        self.name = test_name
        self.n = n
        self.df = df

        self.y = y
        self.diff = diff
        self.t = t
        self._cdist = cdist

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        t = self.t
        pmap = _ttest_p(t.x, self.df, self.tail)
        info = _cs.set_info_cs(t.info, _cs.sig_info())
        p_uncorr = NDVar(pmap, t.dims, info=info, name='p')
        self.p_uncorrected = p_uncorr

        diff_p_uncorrected = [self.diff, t]
        self.diff_p_uncorrected = [diff_p_uncorrected]

        if self.samples:
            diff_p = [self.diff, self.p]
            self.diff_p = self._default_plot_obj = [diff_p]
        else:
            self._default_plot_obj = self.diff_p_uncorrected

    def _repr_test_args(self):
        args = [repr(self.Y)]
        if self.popmean:
            args.append(repr(self.popmean))
        if self.match:
            args.append('match=%r' % self.match)
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args


class ttest_ind(_Result):
    """Element-wise independent samples t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    _state_specific = ('X', 'c1', 'c0', 'tail', 't', 'n1', 'n0', 'df', 'c1_mean',
                       'c0_mean')

    def __init__(self, Y, X, c1=None, c0=None, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=None, tmin=None, tfce=False,
                 tstart=None, tstop=None, dist_dim=(), parc=(),
                 dist_tstep=None, **criteria):
        """Element-wise t-test

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : categorial
            Model containing the cells which should be compared.
        c1 : str | tuple | None
            Test condition (cell of X). Can be None is X only contains two
            cells.
        c0 : str | tuple | None
            Control condition (cell of X). Can be None if X only contains two
            cells.
        match : None | categorial
            Combine cases with the same cell on X % match for testing.
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold p value for forming clusters. None for threshold-free
            cluster enhancement.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds, coercion=asndvar)
        c1, c0 = ct.cat

        test_name = 'Independent Samples t-Test'
        n1 = len(ct.data[c1])
        N = len(ct.Y)
        n0 = N - n1
        df = N - 2
        tmap = _t_ind(ct.Y.x, n1, n0)

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = _ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't',
                                 test_name, tstart, tstop, criteria, dist_dim,
                                 parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                for Y_ in resample(cdist.Y_perm, samples):
                    tmap_ = _t_ind(Y_.x, n1, n0)
                    cdist.add_perm(tmap_)

        dims = ct.Y.dims[1:]

        t0, t1, t2 = _ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        info = _cs.set_info_cs(ct.Y.info, info)
        t = NDVar(tmap, dims, info=info, name='T')

        c1_mean = ct.data[c1].summary(name=cellname(c1))
        c0_mean = ct.data[c0].summary(name=cellname(c0))

        # store attributes
        self.Y = ct.Y.name
        self.X = ct.X.name
        self.c0 = c0
        self.c1 = c1
        if ct.match:
            self.match = ct.match.name
        else:
            self.match = None
        if sub is None or isinstance(sub, basestring):
            self.sub = sub
        else:
            self.sub = "<unsaved array>"
        self.tail = tail
        self.samples = samples
        self.pmin = pmin
        self.tmin = tmin
        self.tfce = tfce

        self.name = test_name
        self.n1 = n1
        self.n0 = n0
        self.df = df

        self.c1_mean = c1_mean
        self.c0_mean = c0_mean
        self.t = t
        self._cdist = cdist

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        c1_mean = self.c1_mean
        c0_mean = self.c0_mean
        t = self.t

        # difference
        diff = c1_mean - c0_mean
        if np.any(diff.x < 0):
            diff.info['cmap'] = 'xpolar'
        self.difference = diff

        # uncorrected p
        pmap = _ttest_p(t.x, self.df, self.tail)
        info = _cs.set_info_cs(t.info, _cs.sig_info())
        p_uncorr = NDVar(pmap, t.dims, info=info, name='p')
        self.p_uncorrected = p_uncorr

        # composites
        diff_p_uncorrected = [diff, t]
        self.diff_p_uncorrected = [diff_p_uncorrected]
        self.all_uncorrected = [c1_mean, c0_mean, diff_p_uncorrected]
        if self.samples:
            diff_p = [diff, self.p]
            self.diff_p = [diff_p]
            self.all = [c1_mean, c0_mean, diff_p]
            self._default_plot_obj = self.all
        else:
            self._default_plot_obj = self.all_uncorrected

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X), "%r (n=%i)" % (self.c1, self.n1),
                "%r (n=%i)" % (self.c0, self.n0)]
        if self.match:
            args.append('match=%r' % self.match)
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args


class ttest_rel(_Result):
    """Element-wise related samples t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    _state_specific = ('X', 'c1', 'c0', 'tail', 't', 'n', 'df', 'c1_mean',
                       'c0_mean')

    def __init__(self, Y, X, c1=None, c0=None, match=None, sub=None, ds=None,
                 tail=0, samples=None, pmin=None, tmin=None, tfce=False,
                 tstart=None, tstop=None, dist_dim=(), parc=(),
                 dist_tstep=None, **criteria):
        """Element-wise t-test

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        X : categorial
            Model containing the cells which should be compared.
        c1 : str | tuple | None
            Test condition (cell of X). Can be omitted (or ``None``) if X only
            contains two cells.
        c0 : str | tuple | None
            Control condition (cell of X). Can be omitted (or ``None``) if X
            only contains two cells.
        match : categorial
            Units within which measurements are related (e.g. 'subject' in a
            within-subject comparison).
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        tail : 0 | 1 | -1
            Which tail of the t-distribution to consider:
            0: both (two-tailed);
            1: upper tail (one-tailed);
            -1: lower tail (one-tailed).
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use a t-value equivalent to an
            uncorrected p-value.
        tmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.

        Notes
        -----
        In the permutation cluster test, permutations are done within the
        categories of ``match``.
        """
        if match is None:
            msg = ("The `match` argument needs to be specified for a related "
                   "samples t-test.")
            raise TypeError(msg)
        ct = Celltable(Y, X, match, sub, cat=(c1, c0), ds=ds, coercion=asndvar)
        c1, c0 = ct.cat
        if not ct.all_within:
            err = ("conditions %r and %r do not have the same values on "
                   "%r" % (c1, c0, ct.match.name))
            raise ValueError(err)

        test_name = 'Related Samples t-Test'
        n = len(ct.Y) // 2
        if n <= 2:
            raise ValueError("Not enough observations for t-test (n=%i)" % n)
        df = n - 1
        tmap = _t_rel(ct.Y.x[:n], ct.Y.x[n:])

        if samples is None:
            cdist = None
        else:
            # threshold
            if sum((pmin is not None, tmin is not None, tfce)) > 1:
                msg = "Only one of pmin, tmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                threshold = _ttest_t(pmin, df, tail)
            elif tmin is not None:
                threshold = abs(tmin)
            elif tfce:
                threshold = 'tfce'
            else:
                threshold = None

            cdist = _ClusterDist(ct.Y, samples, threshold, tail, 't',
                                 test_name, tstart, tstop, criteria, dist_dim,
                                 parc, dist_tstep)
            cdist.add_original(tmap)
            if cdist.n_clusters and samples:
                tmap_ = np.empty(cdist.Y_perm.shape[1:])
                for Y_ in resample(cdist.Y_perm, samples, unit=ct.match):
                    _t_rel(Y_.x[:n], Y_.x[n:], tmap_)
                    cdist.add_perm(tmap_)

        dims = ct.Y.dims[1:]
        t0, t1, t2 = _ttest_t((.05, .01, .001), df, tail)
        info = _cs.stat_info('t', t0, t1, t2, tail)
        t = NDVar(tmap, dims, info=info, name='T')

        c1_mean = ct.data[c1].summary(name=cellname(c1))
        c0_mean = ct.data[c0].summary(name=cellname(c0))

        # store attributes
        self.Y = ct.Y.name
        self.X = ct.X.name
        self.c0 = c0
        self.c1 = c1
        if ct.match:
            self.match = ct.match.name
        else:
            self.match = None
        if sub is None or isinstance(sub, basestring):
            self.sub = sub
        else:
            self.sub = "<unsaved array>"
        self.tail = tail
        self.samples = samples
        self.pmin = pmin
        self.tmin = tmin
        self.tfce = tfce

        self.name = test_name
        self.n = n
        self.df = df

        self.c1_mean = c1_mean
        self.c0_mean = c0_mean
        self.t = t
        self._cdist = cdist

        self._expand_state()

    def _expand_state(self):
        _Result._expand_state(self)

        cdist = self._cdist
        t = self.t

        # difference
        diff = self.c1_mean - self.c0_mean
        if np.any(diff.x < 0):
            diff.info['cmap'] = 'xpolar'
        self.difference = diff

        # uncorrected p
        pmap = _ttest_p(t.x, self.df, self.tail)
        info = _cs.sig_info()
        info['test'] = self.name
        p_uncorr = NDVar(pmap, t.dims, info=info, name='p')
        self.p_uncorrected = p_uncorr

        # composites
        diff_p_uncorr = [diff, t]
        self.difference_p_uncorrected = [diff_p_uncorr]
        self.uncorrected = [self.c1_mean, self.c0_mean, diff_p_uncorr]
        if self.samples:
            diff_p_corr = [diff, cdist.probability_map]
            self.difference_p = [diff_p_corr]
            self._default_plot_obj = [self.c1_mean, self.c0_mean, diff_p_corr]
        else:
            self._default_plot_obj = self.uncorrected

    def _repr_test_args(self):
        args = [repr(self.Y), repr(self.X), repr(self.c1), repr(self.c0),
                "%r (n=%i)" % (self.match, self.n)]
        if self.tail:
            args.append("tail=%i" % self.tail)
        return args


def _t_1samp(a, popmean):
    "Based on scipy.stats.ttest_1samp"
    n = len(a)
    if np.prod(a.shape) > 2 ** 25:
        a_flat = a.reshape((n, -1))
        n_samp = a_flat.shape[1]
        step = int(floor(2 ** 25 / n))
        t_flat = np.empty(n_samp)
        for i in xrange(0, n_samp, step):
            t_flat[i:i + step] = _t_1samp(a_flat[:, i:i + step], popmean)
        t = t_flat.reshape(a.shape[1:])
        return t

    d = np.mean(a, 0) - popmean
    v = np.var(a, 0, ddof=1)
    denom = np.sqrt(v / n)
    t = np.divide(d, denom)
    return t


def _t_ind(x, n1, n2, equal_var=True):
    "Based on scipy.stats.ttest_ind"
    a = x[:n1]
    b = x[n1:]
    v1 = np.var(a, 0, ddof=1)
    v2 = np.var(b, 0, ddof=1)

    if equal_var:
        df = n1 + n2 - 2
        svar = ((n1 - 1) * v1 + (n2 - 1) * v2) / float(df)
        denom = np.sqrt(svar * (1.0 / n1 + 1.0 / n2))
    else:
        vn1 = v1 / n1
        vn2 = v2 / n2
        denom = np.sqrt(vn1 + vn2)

    d = np.mean(a, 0) - np.mean(b, 0)
    t = np.divide(d, denom)
    return t


def _t_rel(y1, y0, out=None, buff=None):
    """
    Calculates the T statistic on two related samples.

    Parameters
    ----------
    y1, y0 : array_like, shape (n_cases, ...)
        Dependent variable for the two samples.
    out : None | array, shape (...)
        array in which to place the result.
    buff : None | array, shape (n_cases, ...)
        Array to serve as buffer for the difference y1 - y0.

    Returns
    -------
    t : array, shape (...)
        t-statistic.

    Notes
    -----
    Based on scipy.stats.ttest_rel
    df = n - 1
    """
    assert(y1.shape == y0.shape)
    n_subjects = len(y1)
    shape = y1.shape[1:]
    n_tests = np.product(shape)

    if np.log2(n_tests) > 13:
        y1 = y1.reshape((n_subjects, n_tests))
        y0 = y0.reshape((n_subjects, n_tests))
        if out is None:
            out = np.empty(shape)
        out_flat = out.reshape(n_tests)
        step = 2 ** 13
        for i in xrange(0, n_tests, step):
            i1 = i + step
            if buff is None:
                buff_ = None
            else:
                buff_ = buff[:, i:i1]
            _t_rel(y1[:, i:i1], y0[:, i:i1], out_flat[i:i1], buff_)
        return out

    if buff is None:
        buff = np.empty(y1.shape)
    d = np.subtract(y1, y0, buff)
    # out = mean(d) / sqrt(var(d) / n_subjects)
    out = d.mean(0, out=out)
    denom = d.var(0, ddof=1)
    denom /= n_subjects
    np.sqrt(denom, out=denom)
    out /= denom
    return out


def _ttest_p(t, df, tail=0):
    """Two tailed probability

    Parameters
    ----------
    t : array_like
        T values.
    df : int
        Degrees of freedom.
    tail : 0 | 1 | -1
        Which tail of the t-distribution to consider:
        0: both (two-tailed);
        1: upper tail (one-tailed);
        -1: lower tail (one-tailed).
    """
    t = np.asanyarray(t)
    if tail == 0:
        t = np.abs(t)
    elif tail == -1:
        t = -t
    elif tail != 1:
        raise ValueError("tail=%r" % tail)
    p = scipy.stats.t.sf(t, df)
    if tail == 0:
        p *= 2
    return p


def _ttest_t(p, df, tail=0):
    """Positive t value for a given probability

    Parameters
    ----------
    p : array_like
        Probability.
    df : int
        Degrees of freedom.
    tail : 0 | 1 | -1
        One- or two-tailed t-distribution (the return value is always positive):
        0: two-tailed;
        1 or -1: one-tailed).
    """
    p = np.asanyarray(p)
    if tail == 0:
        p = p / 2
    t = scipy.stats.t.isf(p, df)
    return t


class _MultiEffectResult(_Result):

    def __repr__(self):
        temp = "<%s %%s>" % self.__class__.__name__

        args = [repr(self.Y), repr(self.X)]
        if self.sub:
            args.append(', sub=%r' % self.sub)
        if self._cdist:
            cdist = self._cdist[0]
            args += cdist._repr_test_args(self.pmin)
            for cdist in self._cdist:
                effect_args = cdist._repr_clusters()
                args += ["%r: %s" % (cdist.name, ', '.join(effect_args))]

        out = temp % ', '.join(args)
        return out

    def _expand_state(self):
        self.effects = tuple(e.name for e in self._effects)

        # clusters
        cdists = self._cdist
        if cdists is not None:
            self.tfce_maps = [cdist.tfce_map for cdist in cdists]
            self.probability_maps = [cdist.probability_map for cdist in cdists]

    def compute_probability_map(self, effect=0, **sub):
        """Compute a probability map

        Parameters
        ----------
        effect : int | str
            Index or name of the effect from which to use the parameter map.

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        elif isinstance(effect, basestring):
            effect = self.effects.index(effect)
        return self._cdist[effect].compute_probability_map(**sub)

    def masked_parameter_map(self, effect=0, pmin=0.05, **sub):
        """Create a copy of the parameter map masked by significance

        Parameters
        ----------
        effect : int | str
            Index or name of the effect from which to use the parameter map.
        pmin : None | scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, pmin=None includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map wherever p <= pmin
            and 0 everywhere else.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        elif isinstance(effect, basestring):
            effect = self.effects.index(effect)
        return self._cdist[effect].masked_parameter_map(pmin, **sub)

    def _clusters(self, pmin=None, maps=False, **sub):
        """Find significant regions in a TFCE distribution

        Parameters
        ----------
        pmin : None | scalar, 1 >= p  >= 0
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps : bool
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default False).

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if self._cdist is None:
            err = ("Test results have no clustering (set samples to an int "
                   " >= 0 to find clusters")
            raise RuntimeError(err)
        dss = []
        info = {}
        for cdist in self._cdist:
            ds = cdist.clusters(pmin, maps, **sub)
            ds[:, 'effect'] = cdist.name
            if 'clusters' in ds.info:
                info['%s clusters' % cdist.name] = ds.info.pop('clusters')
            dss.append(ds)
        out = combine(dss)
        out.info.update(info)
        return out

    def find_peaks(self):
        """Find peaks in a TFCE distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self._cdist is None:
            err = "Method only applies to results with samples > 0"
            raise RuntimeError(err)
        dss = []
        for cdist in self._cdist:
            ds = cdist.find_peaks()
            ds[:, 'effect'] = cdist.name
            dss.append(ds)
        return combine(dss)


class anova(_MultiEffectResult):
    """Element-wise ANOVA

    Attributes
    ----------
    effects : tuple of str
        Names of all the effects as they occur in the ``.clusters`` Dataset.
    clusters : None | Dataset
        When performing a cluster permutation test, a Dataset of all clusters.
    f : list
        Maps of f values with probability contours.
    p : list
        Maps of p values.
    """
    _state_specific = ('X', 'pmin', '_effects', '_dfs_denom', 'f')

    def __init__(self, Y, X, sub=None, ds=None, samples=None, pmin=None,
                 fmin=None, tfce=False, tstart=None, tstop=None, match=None,
                 dist_dim=(), parc=(), dist_tstep=None, **criteria):
        """ANOVA with cluster permutation test

        Parameters
        ----------
        Y : NDVar
            Measurements (dependent variable)
        X : categorial
            Model
        sub : None | index-array
            Perform the test with a subset of the data.
        ds : None | Dataset
            If a Dataset is specified, all data-objects can be specified as
            names of Dataset variables.
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed. Use 0 to compute clusters without performing
            any permutations.
        pmin : None | scalar (0 < pmin < 1)
            Threshold for forming clusters:  use an f-value equivalent to an
            uncorrected p-value.
        fmin : None | scalar
            Threshold for forming clusters.
        tfce : bool
            Use threshold-free cluster enhancement (Smith & Nichols, 2009).
            Default is False.
        replacement : bool
            whether random samples should be drawn with replacement or
            without
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        match : None | categorial
            When permuting data, only shuffle the cases within the categories
            of match.
        mintime : scalar
            Minimum duration for clusters (in seconds).
        minsource : int
            Minimum number of sources per cluster.
        """
        sub = assub(sub, ds)
        Y = asndvar(Y, sub, ds)
        X = asmodel(X, sub, ds)
        if match is not None:
            match = ascategorial(match, sub, ds)

        lm = _nd_anova(X)
        effects = lm.effects
        dfs_denom = lm.dfs_denom
        fmaps = lm.map(Y.x)

        if samples is None:
            cdists = None
        else:
            # threshold
            if sum((pmin is not None, fmin is not None, tfce)) > 1:
                msg = "Only one of pmin, fmin and tfce can be specified"
                raise ValueError(msg)
            elif pmin is not None:
                thresholds = (ftest_f(pmin, e.df, df_den) for e, df_den in
                              izip(effects, dfs_denom))
            elif fmin is not None:
                thresholds = (abs(fmin) for _ in xrange(len(effects)))
            elif tfce:
                thresholds = ('tfce' for _ in xrange(len(effects)))
            else:
                thresholds = (None for _ in xrange(len(effects)))

            n_workers = max(1, int(ceil(cpu_count() / len(effects))))
            cdists = [_ClusterDist(Y, samples, thresh, 1, 'F', e.name, tstart,
                                   tstop, criteria, dist_dim, parc,
                                   dist_tstep, n_workers)
                      for e, thresh in izip(effects, thresholds)]

            # Find clusters in the actual data
            n_clusters = 0
            for cdist, fmap in izip(cdists, fmaps):
                cdist.add_original(fmap)
                n_clusters += cdist.n_clusters

            if n_clusters and samples:
                fmaps_ = lm.preallocate(cdist.Y_perm.shape)
                for Y_ in resample(cdist.Y_perm, samples, unit=match):
                    lm.map(Y_.x)
                    for cdist, fmap in izip(cdists, fmaps_):
                        if cdist.n_clusters:
                            cdist.add_perm(fmap)

        # create ndvars
        dims = Y.dims[1:]

        f = []
        for e, fmap, df_den in izip(effects, fmaps, dfs_denom):
            f0, f1, f2 = ftest_f((0.05, 0.01, 0.001), e.df, df_den)
            info = _cs.stat_info('f', f0, f1, f2)
            f_ = NDVar(fmap, dims, info, e.name)
            f.append(f_)

        # store attributes
        self.Y = Y.name
        self.X = X.name
        if match:
            self.match = match.name
        else:
            self.match = None
        if sub is None or isinstance(sub, basestring):
            self.sub = sub
        else:
            self.sub = "<unsaved array>"
        self.samples = samples
        self.pmin = pmin

        self.name = "ANOVA"
        self._effects = effects
        self._dfs_denom = dfs_denom
        self.f = f

        self._cdist = cdists

        self._expand_state()

    def _expand_state(self):
        # backwards compatibility
        if hasattr(self, 'effects'):
            self._effects = self.effects

        _MultiEffectResult._expand_state(self)

        # backwards compatibility
        if hasattr(self, 'df_den'):
            df_den_temp = {e.name: df for e, df in self.df_den.iteritems()}
            del self.df_den
            self._dfs_denom = tuple(df_den_temp[e] for e in self.effects)

        # f-maps with clusters
        pmin = self.pmin or 0.05
        if self.samples:
            f_and_clusters = []
            for e, fmap, df_den, cdist in izip(self._effects, self.f,
                                               self._dfs_denom, self._cdist):
                # create f-map with cluster threshold
                f0 = ftest_f(pmin, e.df, df_den)
                info = _cs.stat_info('f', f0)
                f_ = NDVar(fmap.x, fmap.dims, info, e.name)
                # add overlay with cluster
                if cdist.probability_map is not None:
                    f_and_clusters.append([f_, cdist.probability_map])
                else:
                    f_and_clusters.append([f_])
            self.f_probability = f_and_clusters

        # uncorrected probability
        p_uncorr = []
        for e, f, df_den in izip(self._effects, self.f, self._dfs_denom):
            info = _cs.sig_info()
            pmap = ftest_p(f.x, e.df, df_den)
            p_ = NDVar(pmap, f.dims, info, e.name)
            p_uncorr.append(p_)
        self.p_uncorrected = p_uncorr

        if self.samples:
            self._default_plot_obj = f_and_clusters
        else:
            self._default_plot_obj = self.f


def _label_clusters(pmap, out, bin_buff, int_buff, threshold, tail, struct,
                    all_adjacent, flat_shape, conn, criteria):
    """Find clusters on a statistical parameter map

    Parameters
    ----------
    pmap : array
        Statistical parameter map (non-adjacent dimension on the first
        axis).
    out : array of int
        Buffer for the cluster id map (will be modified).

    Returns
    -------
    cluster_ids : tuple
        Identifiers of the clusters that survive the minimum duration
        criterion.
    """
    if tail >= 0:
        bin_map_above = np.greater(pmap, threshold, bin_buff)
        cids = _label_clusters_binary(bin_map_above, out, struct, all_adjacent,
                                      flat_shape, conn, criteria)

    if tail <= 0:
        bin_map_below = np.less(pmap, -threshold, bin_buff)
        if tail < 0:
            cids = _label_clusters_binary(bin_map_below, out, struct,
                                          all_adjacent, flat_shape, conn,
                                          criteria)
        else:
            cids_l = _label_clusters_binary(bin_map_below, int_buff, struct,
                                            all_adjacent, flat_shape, conn,
                                            criteria)
            x = int(out.max())  # apparently np.uint64 + int makes a float
            int_buff[bin_map_below] += x
            out += int_buff
            cids = np.concatenate((cids, cids_l + x))

    return cids


def _label_clusters_binary(bin_map, out, struct, all_adjacent, flat_shape,
                           conn, criteria):
    """Label clusters in a binary array

    Parameters
    ----------
    bin_map : np.ndarray
        Binary map of where the parameter map exceeds the threshold for a
        cluster (non-adjacent dimension on the first axis).
    out : np.ndarray
        Array in which to label the clusters.
    struct : np.ndarray
        Struct to use for scipy.ndimage.label
    all_adjacent : bool
        Whether all dimensions have line-graph connectivity.
    flat_shape : tuple
        Shape for making bin_map 2-dimensional.
    conn : dict
        Connectivity (if first dimension is not a line graph).
    criteria : None | list
        Cluster size criteria, list of (axes, v) tuples. Collapse over axes
        and apply v minimum length).

    Returns
    -------
    cluster_ids : np.ndarray
        Sorted identifiers of the clusters that survive the selection criteria.
    """
    # find clusters
    n = ndimage.label(bin_map, struct, out)
    # n is 1 even when no cluster is found
    if n == 1 and out.max() == 0:
        n = 0

    if all_adjacent or n <= 1:
        cids = np.arange(1, n + 1)
    else:
        cmap_flat = out.reshape(flat_shape)
        cids = merge_labels(cmap_flat, n, conn)

    # apply minimum cluster size criteria
    if criteria:
        rm_cids = set()
        for axes, v in criteria:
            rm_cids.update(i for i in cids if
                           np.count_nonzero(np.equal(out, i).any(axes)) < v)
        cids = np.setdiff1d(cids, rm_cids)

    return cids


def _tfce(pmap, out, tail, bin_buff, int_buff, struct, all_adjacent,
          flat_shape, conn):
    dh = 0.1
    E = 0.5
    H = 2.0

    if tail <= 0:
        hs = np.arange(-dh, pmap.min(), -dh)
    if tail >= 0:
        upper = np.arange(dh, pmap.max(), dh)
        if tail == 0:
            hs = np.hstack((hs, upper))
        else:
            hs = upper

    out.fill(0)

    # label clusters in slices at different heights
    # fill each cluster with total section value
    # each point's value is the vertical sum
    for h in hs:
        if h > 0:
            np.greater_equal(pmap, h, bin_buff)
            h_factor = h ** H
        else:
            np.less_equal(pmap, h, bin_buff)
            h_factor = (-h) ** H

        c_ids = _label_clusters_binary(bin_buff, int_buff, struct,
                                       all_adjacent, flat_shape, conn, None)
        for id_ in c_ids:
            np.equal(int_buff, id_, bin_buff)
            v = np.count_nonzero(bin_buff) ** E * h_factor
            out[bin_buff] += v

    return out


def _clustering_worker(in_queue, out_queue, shape, threshold, tail, struct,
                       all_adjacent, flat_shape, conn, criteria, parc):
    os.nice(20)

    # allocate memory buffers
    cmap = np.empty(shape, np.uint32)
    bin_buff = np.empty(shape, np.bool_)
    int_buff = np.empty(shape, np.uint32)
    if parc is not None:
        out = np.empty(len(parc))

    while True:
        pmap = in_queue.get()
        if pmap is None:
            break
        cids = _label_clusters(pmap, cmap, bin_buff, int_buff, threshold, tail,
                               struct, all_adjacent, flat_shape, conn, criteria)
        if parc is not None:
            out.fill(0)
            for i, idx in enumerate(parc):
                clusters_v = ndimage.sum(pmap[idx], cmap[idx], cids)
                if len(clusters_v):
                    np.abs(clusters_v, clusters_v)
                    out[i] = clusters_v.max()
            out_queue.put(out)
        elif len(cids):
            clusters_v = ndimage.sum(pmap, cmap, cids)
            np.abs(clusters_v, clusters_v)
            out_queue.put(clusters_v.max())
        else:
            out_queue.put(0)


def _tfce_worker(in_queue, out_queue, shape, tail, struct, all_adjacent,
                 flat_shape, conn, stacked_shape, max_axes):
    os.nice(20)

    # allocate memory buffers
    tfce_map = np.empty(shape)
    tfce_map_stacked = tfce_map.reshape(stacked_shape)
    bin_buff = np.empty(shape, np.bool_)
    int_buff = np.empty(shape, np.uint32)

    while True:
        pmap = in_queue.get()
        if pmap is None:
            break
        _tfce(pmap, tfce_map, tail, bin_buff, int_buff, struct, all_adjacent,
              flat_shape, conn)
        out = tfce_map_stacked.max(max_axes)
        out_queue.put(out)


def _dist_worker(ct_dist, dist_shape, in_queue):
    "Worker that accumulates values and places them into the distribution"
    n = reduce(operator.mul, dist_shape)
    dist = np.frombuffer(ct_dist, np.float64, n)
    dist.shape = dist_shape
    for i in xrange(dist_shape[0]):
        dist[i] = in_queue.get()


class _ClusterDist:
    """Accumulate information on a cluster statistic.

    Notes
    -----
    Use of the _ClusterDist proceeds in 3 steps:

    - initialize the _ClusterDist object: ``cdist = _ClusterDist(...)``
    - use a copy of Y cropped to the time window of interest:
      ``Y = cdist.Y_perm``
    - add the actual statistical map with ``cdist.add_original(pmap)``
    - if any clusters are found (``if cdist.n_clusters``):

      - proceed to add statistical maps from permuted data with
        ``cdist.add_perm(pmap)``.
    """
    def __init__(self, Y, N, threshold, tail=0, meas='?', name=None,
                 tstart=None, tstop=None, criteria={}, dist_dim=(), parc=(),
                 dist_tstep=None, n_workers=cpu_count()):
        """Accumulate information on a cluster statistic.

        Parameters
        ----------
        Y : NDVar
            Dependent variable.
        N : int
            Number of permutations.
        threshold : None | scalar > 0 | 'tfce'
            Threshold for finding clusters. None for forming distribution of
            largest value in parameter map. 'TFCE' for threshold-free cluster
            enhancement.
        tail : 1 | 0 | -1
            Which tail(s) of the distribution to consider. 0 is two-tailed,
            whereas 1 only considers positive values and -1 only considers
            negative values.
        meas : str
            Label for the parameter measurement (e.g., 't' for t-values).
        name : None | str
            Name for the comparison.
        tstart, tstop : None | scalar
            Restrict the time window for finding clusters (None: use the whole
            epoch).
        criteria : dict
            Dictionary with threshold criteria for cluster size: 'mintime'
            (seconds) and 'minsource' (n_sources).
        dist_dim : str | sequence of str
            Collect permutation extrema for all points in this dimension(s)
            instead of only collecting the overall maximum. This allows
            deriving p-values for regions of interest from the same set of
            permutations. Threshold-free distributions only.
        parc : str | sequence of str
            Collect permutation extrema for all regions of the parcellation of
            this dimension(s). For threshold-based test, the regions are
            disconnected.
        dist_tstep : None | scalar [seconds]
            Instead of collecting the distribution for the maximum across time,
            collect the maximum in several time bins. The value of tstep has to
            divide the time between tstart and tstop in even sections. TFCE
            only.
        n_workers : int
            Number of clustering workers (for threshold based clusters and
            TFCE). Negative numbers are added to the cpu-count, 0 to disable
            multiprocessing.
        """
        assert Y.has_case
        if threshold is None:
            kind = 'raw'
        elif isinstance(threshold, str):
            if threshold.lower() == 'tfce':
                kind = 'tfce'
            else:
                raise ValueError("Invalid value for pmin: %s" % repr(threshold))
        else:
            try:
                threshold = float(threshold)
            except:
                raise TypeError("Invalid value for pmin: %s" % repr(threshold))

            if threshold > 0:
                kind = 'cluster'
            else:
                raise ValueError("Invalid value for pmin: %s" % repr(threshold))

        # adapt arguments
        if isinstance(dist_dim, basestring):
            dist_dim = (dist_dim,)
        elif dist_dim is None:
            dist_dim = ()

        if isinstance(parc, basestring):
            parc = (parc,)
        elif parc is None:
            parc = ()

        # prepare temporal cropping
        if (tstart is None) and (tstop is None):
            self.crop = False
            Y_perm = Y
        else:
            t_ax = Y.get_axis('time') - 1
            self.crop = True
            Y_perm = Y.sub(time=(tstart, tstop))
            t_slice = Y.time._slice(tstart, tstop)
            self._crop_idx = (slice(None),) * t_ax + (t_slice,)
            self._uncropped_shape = Y.shape[1:]

        # cluster map properties
        ndim = Y_perm.ndim - 1
        shape = Y_perm.shape[1:]
        cmap_dims = Y_perm.dims[1:]

        # prepare adjacency
        struct = ndimage.generate_binary_structure(ndim, 1)
        adjacent = [d.adjacent for d in Y_perm.dims[1:]]
        all_adjacent = all(adjacent)
        if all_adjacent:
            nad_ax = 0
            connectivity = None
            flat_shape = None
        else:
            if sum(adjacent) < len(adjacent) - 1:
                err = ("more than one non-adjacent dimension")
                raise NotImplementedError(err)
            nad_ax = adjacent.index(False)
            struct[::2] = False
            # prepare flattening (cropped) maps with swapped axes
            if nad_ax:
                shape = list(shape)
                shape[0], shape[nad_ax] = shape[nad_ax], shape[0]
                shape = tuple(shape)
                cmap_dims = list(cmap_dims)
                cmap_dims[0], cmap_dims[nad_ax] = cmap_dims[nad_ax], cmap_dims[0]
                cmap_dims = tuple(cmap_dims)
            flat_shape = (shape[0], np.prod(shape[1:]))

            # prepare connectivity
            nad_dim = cmap_dims[0]
            disconnect_parc = (nad_dim.name in parc)
            connectivity = nad_dim.connectivity(disconnect_parc)

        # prepare cluster minimum size criteria
        if criteria:
            criteria_ = []
            for k, v in criteria.iteritems():
                if k == 'mintime':
                    ax = Y.get_axis('time') - 1
                    v = int(ceil(v / Y.time.tstep))
                else:
                    m = re.match('min(\w+)', k)
                    if m:
                        ax = Y.get_axis(m.group(1)) - 1
                    else:
                        raise ValueError("Unknown argument: %s" % k)

                if nad_ax:
                    if ax == 0:
                        ax = nad_ax
                    elif ax == nad_ax:
                        ax = 0

                axes = tuple(i for i in xrange(ndim) if i != ax)
                criteria_.append((axes, v))

            if kind != 'cluster':
                # here so that invalid keywords raise explicitly
                err = ("Can not use cluster size criteria when doing "
                       "threshold free cluster evaluation")
                raise ValueError(err)
        else:
            criteria_ = None

        # prepare distribution
        N = int(N)
        if (dist_dim or parc or dist_tstep):
            # raise for incompatible cases
            if (dist_dim or dist_tstep) and kind == 'cluster':
                err = ("The dist_dim and dist_tstep parameters only apply to "
                       "threshold-free cluster distributions.")
                raise ValueError(err)

            # check all dims are in order
            if dist_tstep and not Y.has_dim('time'):
                msg = "dist_tstep specified but data has no time dimension"
                raise ValueError(msg)
            dim_names = tuple(dim.name for dim in Y_perm.dims[1:])
            err = tuple(name for name in chain(dist_dim, parc) if name not in
                        dim_names)
            if err:
                if len(err) == 1:
                    msg = ("%r is contained in dist_dim or parc but is not a "
                           "valid dimension in the input ndvar" % err)
                else:
                    msg = ("%r are contained in dist_dim or parc but are not "
                           "valid dimensions in the input ndvar" % str(err))
                raise ValueError(msg)
            duplicates = set(dist_dim)
            duplicates.intersection_update(parc)
            if duplicates:
                msg = ("%s were specified as dist_dim as well as parc. Each "
                       "dimension can only be either dist_dim or parc.")
                raise ValueError(msg)

            # find parameters for aggregating dist
            dist_shape = [N]
            dist_dims = ['case']
            cmap_reshape = []  # reshape value map for dist_tstep before .max()
            max_axes = []  # v_map.max(max_axes)
            reshaped_ax_shift = 0  # number of inserted axes after reshaping cmap
            parc_indexes = None  # (ax, parc-Factor) tuples
            for i, dim in enumerate(cmap_dims):
                if dim.name in dist_dim:  # keep the dimension
                    length = len(dim)
                    dist_shape.append(length)
                    dist_dims.append(dim)
                    cmap_reshape.append(length)
                elif dim.name in parc:
                    if not hasattr(dim, 'parc'):
                        msg = "%r dimension has no parcellation" % dim.name
                        raise NotImplementedError(msg)
                    elif i != 0:
                        msg = "parc that is not non-adjacent axis"
                        raise NotImplementedError(msg)
                    parc_ = dim.parc
                    parc_dim = Categorial(dim.name, parc_.cells)
                    length = len(parc_dim)
                    dist_shape.append(length)
                    dist_dims.append(parc_dim)
                    cmap_reshape.append(len(dim))
                    indexes = [parc_ == cell for cell in parc_.cells]
                    parc_indexes = np.array(indexes)
                elif dim.name == 'time' and dist_tstep:
                    step = int(round(dist_tstep / dim.tstep))
                    if dim.nsamples % step != 0:
                        err = ("dist_tstep={} does not divide time into even "
                               "parts ({} samples / {}).")
                        err = err.format(dist_tstep, dim.nsamples, step)
                        raise ValueError(err)
                    n_times = int(dim.nsamples / step)

                    dist_shape.append(n_times)
                    dist_dims.append(UTS(dim.tmin, dist_tstep, n_times))
                    cmap_reshape.append(step)
                    cmap_reshape.append(n_times)
                    max_axes.append(i + reshaped_ax_shift)
                    reshaped_ax_shift += 1
                else:
                    cmap_reshape.append(len(dim))
                    max_axes.append(i + reshaped_ax_shift)

            dist_shape = tuple(dist_shape)
            dist_dims = tuple(dist_dims)
            cmap_reshape = tuple(cmap_reshape)
            max_axes = tuple(max_axes)
        else:
            dist_shape = (N,)
            dist_dims = None
            cmap_reshape = None
            max_axes = None
            parc_indexes = None

        # multiprocessing
        if n_workers:
            if multiprocessing and N > 1 and kind != 'raw':
                if n_workers < 0:
                    n_workers = max(1, cpu_count() + n_workers)
            else:
                n_workers = 0

        self.kind = kind
        self.Y_perm = Y_perm
        self.dims = Y_perm.dims
        self._cmap_dims = cmap_dims
        self.shape = shape
        self._flat_shape = flat_shape
        self._connectivity = connectivity
        self.N = N
        self._dist_shape = dist_shape
        self._dist_dims = dist_dims
        self._cmap_reshape = cmap_reshape
        self._max_axes = max_axes
        self._parc = parc_indexes
        self.dist = None
        self._i = N
        self.threshold = threshold
        self.tail = tail
        self._all_adjacent = all_adjacent
        self._nad_ax = nad_ax
        self._struct = struct
        self.tstart = tstart
        self.tstop = tstop
        self.dist_dim = dist_dim
        self.parc = parc
        self.dist_tstep = dist_tstep
        self.meas = meas
        self.name = name
        self._criteria = criteria_
        self.criteria = criteria
        self.has_original = False
        self._has_buffers = False
        self.dt_perm = None
        self._dist_i = N
        self._n_workers = n_workers

        if kind != 'raw':
            self._allocate_memory_buffers()

    def _allocate_memory_buffers(self):
        "Pre-allocate memory buffers used for cluster processing"
        if self._has_buffers:
            return

        shape = self.shape
        self._bin_buff = np.empty(shape, dtype=np.bool8)
        self._int_buff = np.empty(shape, dtype=np.uint32)
        if self.kind == 'cluster' and self.tail == 0:
            self._int_buff2 = np.empty(shape, dtype=np.uint32)
        else:
            self._int_buff2 = None

        self._has_buffers = True
        if self._i == 0 or self.kind == 'cluster':
            return

        # only for TFCE
        self._float_buff = np.empty(shape)
        if self._cmap_reshape is None:
            self._cmap_stacked = self._float_buff
        else:
            self._cmap_stacked = self._float_buff.reshape(self._cmap_reshape)

    def _clear_memory_buffers(self):
        "Remove memory buffers used for cluster processing"
        for name in ('_bin_buff', '_int_buff', '_float_buff', '_int_buff2'):
            if hasattr(self, name):
                delattr(self, name)
        self._has_buffers = False

    def _init_permutation(self):
        "Permutation is only performed when clusters are found"
        if self._n_workers:
            n = reduce(operator.mul, self._dist_shape)
            ct_dist = RawArray('d', n)
            dist = np.frombuffer(ct_dist, np.float64, n)
            dist.shape = self._dist_shape
            self._spawn_workers(ct_dist)
        else:
            dist = np.zeros(self._dist_shape)

        self.dist = dist

    def _spawn_workers(self, ct_dist):
        "Spawn workers for multiprocessing"
        logger.debug("Setting up worker processes...")
        self._dist_queue = dist_queue = Queue()
        self._pmap_queue = pmap_queue = Queue()

        # clustering workers
        shape = self.shape
        tail = self.tail
        struct = self._struct
        all_adjacent = self._all_adjacent
        flat_shape = self._flat_shape
        conn = self._connectivity
        parc = self._parc
        if self.kind == 'cluster':
            criteria = self._criteria
            target = _clustering_worker
            threshold = self.threshold
            args = (pmap_queue, dist_queue, shape, threshold, tail, struct,
                    all_adjacent, flat_shape, conn, criteria, parc)
        else:
            stacked_shape = self._cmap_reshape
            max_axes = self._max_axes
            target = _tfce_worker
            args = (pmap_queue, dist_queue, shape, tail, struct, all_adjacent,
                    flat_shape, conn, stacked_shape, max_axes)

        self._workers = []
        for _ in xrange(self._n_workers):
            w = Process(target=target, args=args)
            w.start()
            self._workers.append(w)

        # distribution worker
        args = (ct_dist, self._dist_shape, dist_queue)
        w = Process(target=_dist_worker, args=args)
        w.start()
        self._dist_worker = w

    def __repr__(self):
        items = []
        if self.has_original:
            dt = timedelta(seconds=round(self.dt_original))
            items.append("%i clusters (%s)" % (self.n_clusters, dt))

            if self.N > 0 and self.n_clusters > 0:
                if self._i == 0:
                    dt = timedelta(seconds=round(self.dt_perm))
                    item = "%i permutations (%s)" % (self.N, dt)
                else:
                    item = "%i of %i permutations" % (self.N - self._i, self.N)
                items.append(item)
        else:
            items.append("no data")

        return "<ClusterDist: %s>" % ', '.join(items)

    def __getstate__(self):
        if self._i > 0:
            err = ("Cannot pickle cluster distribution before all permu"
                   "tations have been added.")
            raise RuntimeError(err)
        attrs = ('name', 'meas',
                 # settings ...
                 'kind', 'threshold', 'tail', 'criteria', 'N', 'tstart',
                 'tstop', 'dist_dim', 'dist_tstep',
                  # data properties ...
                 'dims', 'shape', '_all_adjacent', '_nad_ax', '_struct',
                 '_flat_shape', '_connectivity', '_criteria', '_cmap_dims',
                 # results ...
                 'dt_original', 'dt_perm', 'n_clusters',
                 '_dist_shape', '_dist_dims', 'dist',
                 '_original_param_map', '_original_cluster_map', '_cids')
        state = {name: getattr(self, name) for name in attrs}
        return state

    def __setstate__(self, state):
        if '_connectivity_src' in state:
            # backwards compatibility
            state['_connectivity'] = np.hstack((state.pop('_connectivity_src'),
                                                state.pop('_connectivity_dst')))

        for k, v in state.iteritems():
            setattr(self, k, v)
        self._i = 0
        self.has_original = True
        self._has_buffers = False
        self._finalize()

    def _repr_test_args(self, pmin):
        "Argument representation for TestResult repr"
        args = ['samples=%r' % self.N]
        if pmin:
            args.append("pmin=%r" % pmin)
        if self.tstart:
            args.append("tstart=%r" % self.tstart)
        if self.tstop:
            args.append("tstop=%r" % self.tstop)
        if self.dist_dim:
            args.append("dist_dim=%r" % self.dist_dim)
        if self.dist_tstep:
            args.append("dist_tstep=%r" % self.dist_tstep)
        for item in self.criteria.iteritems():
            args.append("%s=%r" % item)
        return args

    def _repr_clusters(self):
        info = []
        if self.kind == 'cluster':
            if self.n_clusters == 0:
                info.append("no clusters")
            else:
                info.append("%i clusters" % self.n_clusters)

        if self.n_clusters and self.N:
            info.append("p >= %.3f" % self.probability_map.min())

        return info

    def _crop(self, im):
        if self.crop:
            return im[self._crop_idx]
        else:
            return im

    @LazyProperty
    def _default_plot_obj(self):
        if self.N:
            return [[self.self.parameter_map, self.probability_map]]
        else:
            return [[self.self.parameter_map]]

    def _finalize(self):
        "Package results and delete temporary data"
        # prepare container for clusters
        self._allocate_memory_buffers()
        dims = self.dims
        param_contours = {}
        if self.kind == 'cluster':
            if self.tail >= 0:
                param_contours[self.threshold] = (0.7, 0.7, 0)
            if self.tail <= 0:
                param_contours[-self.threshold] = (0.7, 0, 0.7)

        # original parameter-map
        param_map = self._original_param_map

        # TFCE map
        if self.kind == 'tfce':
            stat_map = self._original_cluster_map
            x = stat_map.swapaxes(0, self._nad_ax)
            tfce_map_ = NDVar(x, dims[1:], {}, self.name)
        else:
            stat_map = self._original_param_map
            tfce_map_ = None

        # cluster map
        if self.kind == 'cluster':
            cluster_map = self._original_cluster_map
            x = cluster_map.swapaxes(0, self._nad_ax)
            cluster_map_ = NDVar(x, dims[1:], {}, self.name)
        else:
            cluster_map_ = None

        # original parameter map
        info = _cs.stat_info(self.meas, contours=param_contours)
        if self._nad_ax:
            param_map = param_map.swapaxes(0, self._nad_ax)
        param_map_ = NDVar(param_map, dims[1:], info, self.name)

        # store attributes
        self.tfce_map = tfce_map_
        self.parameter_map = param_map_
        self.cluster_map = cluster_map_

        self._clear_memory_buffers()

    def _find_peaks(self, x, out=None):
        """Find peaks (local maxima, including plateaus) in x

        Returns
        -------
        out : array (x.shape, bool)
            Boolean array which is True only on local maxima. The borders are
            treated as lower than the rest of x (i.e., local maxima can touch
            the border).
        """
        if out is None:
            out = np.empty(x.shape, np.bool8)
        out.fill(True)

        # move through each axis in both directions and discard descending
        # slope. Do most computationally intensive axis last.
        for ax in xrange(x.ndim - 1, -1, -1):
            if ax == 0 and not self._all_adjacent:
                shape = (len(x), -1)
                xsa = x.reshape(shape)
                outsa = out.reshape(shape)
                axlen = xsa.shape[1]

                conn_src = self._connectivity[:, 0]
                conn_dst = self._connectivity[:, 1]
                for i in xrange(axlen):
                    data = xsa[:, i]
                    outslice = outsa[:, i]
                    if not np.any(outslice):
                        continue

                    # find all points under a slope
                    sign = np.sign(data[conn_src] - data[conn_dst])
                    no = set(conn_src[sign < 0])
                    no.update(conn_dst[sign > 0])

                    # expand to equal points
                    border = no
                    while border:
                        # forward
                        idx = np.in1d(conn_src, border)
                        conn_dst_sub = conn_dst[idx]
                        eq = np.equal(data[conn_src[idx]], data[conn_dst_sub])
                        new = set(conn_dst_sub[eq])
                        # backward
                        idx = np.in1d(conn_dst, border)
                        conn_src_sub = conn_src[idx]
                        eq = np.equal(data[conn_src_sub], data[conn_dst[idx]])
                        new.update(conn_src_sub[eq])

                        # update
                        new.difference_update(no)
                        no.update(new)
                        border = new

                    # mark vertices or whole isoline
                    if no:
                        outslice[list(no)] = False
                    elif not np.all(outslice):
                        outslice.fill(False)
            else:
                if x.ndim == 1:
                    xsa = x[:, None]
                    outsa = out[:, None]
                else:
                    xsa = x.swapaxes(0, ax)
                    outsa = out.swapaxes(0, ax)
                axlen = len(xsa)

                kernel = np.empty(xsa.shape[1:], dtype=np.bool8)

                diff = np.diff(xsa, 1, 0)

                # forward
                kernel.fill(True)
                for i in xrange(axlen - 1):
                    kernel[diff[i] > 0] = True
                    kernel[diff[i] < 0] = False
                    nodiff = diff[i] == 0
                    kernel[nodiff] *= outsa[i + 1][nodiff]
                    outsa[i + 1] *= kernel

                # backward
                kernel.fill(True)
                for i in xrange(axlen - 2, -1, -1):
                    kernel[diff[i] < 0] = True
                    kernel[diff[i] > 0] = False
                    nodiff = diff[i] == 0
                    kernel[nodiff] *= outsa[i][nodiff]
                    outsa[i] *= kernel

        return out

    def _uncrop(self, im, background=0):
        if self.crop:
            im_ = np.empty(self._uncropped_shape, dtype=im.dtype)
            im_[:] = background
            im_[self._crop_idx] = im
            return im_
        else:
            return im

    def add_original(self, param_map):
        """Add the original statistical parameter map.

        Parameters
        ----------
        param_map : array
            Parameter map of the statistic of interest (uncropped).
        """
        if self.has_original:
            raise RuntimeError("Original pmap already added")

        logger.debug("Adding original parameter map...")
        t0 = current_time()
        param_map = self._crop(param_map)
        if self._nad_ax:
            param_map = param_map.swapaxes(0, self._nad_ax)

        if self.kind == 'tfce':
            original_cluster_map = np.empty(self.shape)
            _tfce(param_map, original_cluster_map, self.tail,
                  self._bin_buff, self._int_buff, self._struct,
                  self._all_adjacent, self._flat_shape, self._connectivity)
            cids = None
            n_clusters = True
        elif self.kind == 'cluster':
            original_cluster_map = buff = np.empty(self.shape, dtype=np.uint32)
            cids = _label_clusters(param_map, buff, self._bin_buff,
                                   self._int_buff2, self.threshold,
                                   self.tail, self._struct, self._all_adjacent,
                                   self._flat_shape, self._connectivity,
                                   self._criteria)
            n_clusters = len(cids)
            # clean original cluster map
            idx = (np.in1d(original_cluster_map, cids, invert=True)
                   .reshape(original_cluster_map.shape))
            original_cluster_map[idx] = 0
        else:
            original_cluster_map = param_map
            cids = None
            n_clusters = True

        t1 = current_time()
        self._original_cluster_map = original_cluster_map
        self._cids = cids
        self.n_clusters = n_clusters
        self.has_original = True
        self.dt_original = t1 - t0
        self._t0 = t1
        self._original_param_map = param_map
        if self.N and n_clusters:
            self._init_permutation()
        else:
            self._i = 0  # set to 0 so it can be saved
            self._finalize()

    def add_perm(self, pmap):
        """Add the statistical parameter map from permuted data.

        Parameters
        ----------
        pmap : array
            Parameter map of the statistic of interest.
        """
        if self._i <= 0:
            raise RuntimeError("Too many permutations added to _ClusterDist")
        self._i -= 1
        finished = self._i == 0

        if self._nad_ax:
            pmap = pmap.swapaxes(0, self._nad_ax)

        if self._n_workers:
            # log
            dt = current_time() - self._t0
            elapsed = timedelta(seconds=round(dt))
            logger.info("%s: putting %i" % (elapsed, self._i))
            # place data in queue
            self._pmap_queue.put(pmap)

            if finished:
                logger.info("Waiting for cluster distribution...")
                for _ in xrange(self._n_workers):
                    self._pmap_queue.put(None)
                for w in self._workers:
                    w.join()
                self._dist_worker.join()
                logger.info("Done")
        else:
            if self.kind == 'tfce':
                _tfce(pmap, self._float_buff, self.tail, self._bin_buff,
                      self._int_buff, self._struct, self._all_adjacent,
                      self._flat_shape, self._connectivity)
                v = self._cmap_stacked.max(self._max_axes)
            elif self.kind == 'cluster':
                cmap = self._int_buff
                cids = _label_clusters(pmap, cmap, self._bin_buff,
                                       self._int_buff2, self.threshold,
                                       self.tail, self._struct,
                                       self._all_adjacent, self._flat_shape,
                                       self._connectivity, self._criteria)
                if self._parc is not None:
                    v = np.empty(len(self._parc))
                    v.fill(0)
                    for i, idx in enumerate(self._parc):
                        clusters_v = ndimage.sum(pmap[idx], cmap[idx], cids)
                        if len(clusters_v):
                            np.abs(clusters_v, clusters_v)
                            v[i] = clusters_v.max()
                elif len(cids):
                    clusters_v = ndimage.sum(pmap, cmap, cids)
                    np.abs(clusters_v, clusters_v)
                    v = clusters_v.max()
                else:
                    v = 0
            else:
                pmap_ = pmap.reshape(self._cmap_reshape)
                if self.tail == 0:
                    v = np.abs(pmap_, pmap_).max(self._max_axes)
                elif self.tail > 0:
                    v = pmap_.max(self._max_axes)
                else:
                    v = -pmap_.min(self._max_axes)

                if self._parc is not None:
                    v = [v[idx].max() for idx in self._parc]

            self.dist[self._i] = v
            # log
            n_done = self.N - self._i
            dt = current_time() - self._t0
            elapsed = timedelta(seconds=round(dt))
            avg = timedelta(seconds=dt / n_done)
            logger.info("%s: Sample %i, avg time: %s" % (elapsed, n_done, avg))

        # catch last permutation
        if finished:
            self.dt_perm = current_time() - self._t0
            self._finalize()

    def _cluster_properties(self, cluster_map, cids):
        """Create a Dataset with cluster properties

        Parameters
        ----------
        cluster_map : NDVar
            NDVar in which clusters are marked by bearing the same number.
        cids : array_like of int
            Numbers specifying the clusters (must occur in cluster_map) which
            should be analyzed.

        Returns
        -------
        cluster_properties : Dataset
            Cluster properties. Which properties are included depends on the
            dimensions.
        """
        ndim = cluster_map.ndim
        n_clusters = len(cids)

        # setup compression
        compression = []
        for ax, dim in enumerate(cluster_map.dims):
            extents = np.empty((n_clusters, len(dim)), dtype=np.bool_)
            axes = tuple(i for i in xrange(ndim) if i != ax)
            compression.append((ax, dim, axes, extents))

        # find extents for all clusters
        c_mask = np.empty(cluster_map.shape, np.bool_)
        for i, cid in enumerate(cids):
            np.equal(cluster_map, cid, c_mask)
            for ax, dim, axes, extents in compression:
                np.any(c_mask, axes, extents[i])

        # prepare Dataset
        ds = Dataset()
        ds['id'] = Var(cids)

        for ax, dim, axes, extents in compression:
            properties = dim._cluster_properties(extents)
            if properties is not None:
                ds.update(properties)

        return ds

    def clusters(self, pmin=None, maps=True, **sub):
        """Find significant clusters

        Parameters
        ----------
        pmin : None | scalar, 1 >= p  >= 0
            Threshold p-value for clusters (for thresholded cluster tests the
            default is 1, for others 0.05).
        maps : bool
            Include in the output a map of every cluster (can be memory
            intensive if there are large statistical maps and/or many
            clusters; default True).
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        ds : Dataset
            Dataset with information about the clusters.
        """
        if pmin is None:
            if self.kind != 'cluster':
                pmin = 0.05
        if pmin is not None and self.N == 0:
            msg = ("Can not determine p values in distribution without "
                   "permutations.")
            if self.kind == 'cluster':
                msg += " Find clusters with pmin=None."
            raise RuntimeError(msg)

        if sub:
            param_map = self.parameter_map.sub(**sub)
        else:
            param_map = self.parameter_map

        if self.kind == 'cluster':
            if sub:
                cluster_map = self.cluster_map.sub(**sub)
                cids = np.setdiff1d(cluster_map.x, [0])
            else:
                cluster_map = self.cluster_map
                cids = np.array(self._cids)

            if len(cids):
                # measure original clusters
                cluster_v = ndimage.sum(param_map.x, cluster_map.x, cids)

                # p-values
                if self.N:
                    # p-values: "the proportion of random partitions that
                    # resulted in a larger test statistic than the observed
                    # one" (179)
                    dist = self._aggregate_dist(**sub)
                    n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                    cluster_p = n_larger / self.N

                    # select clusters
                    if pmin is not None:
                        idx = cluster_p <= pmin
                        cids = cids[idx]
                        cluster_p = cluster_p[idx]
                        cluster_v = cluster_v[idx]

                    # p-value corrected across parc
                    if sub:
                        dist = self._aggregate_dist()
                        n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                        cluster_p_corr = n_larger / self.N
            else:
                cluster_v = cluster_p = cluster_p_corr = []

            ds = self._cluster_properties(cluster_map, cids)
            ds['v'] = Var(cluster_v)
            if self.N:
                ds['p'] = Var(cluster_p)
                if sub:
                    ds['p_parc'] = Var(cluster_p_corr)

            threshold = self.threshold
        else:
            p_map = self.compute_probability_map(**sub)
            bin_map = np.less_equal(p_map.x, pmin)
            shape = p_map.shape

            # threshold for maps
            if maps:
                values = np.abs(param_map.x)[bin_map]
                if len(values):
                    threshold = values.min() / 2
                else:
                    threshold = 1.

            # find clusters
            c_map = np.empty(shape, np.uint32)  # cluster map
            # reshape to internal shape for labelling
            bin_map_is = bin_map.swapaxes(0, self._nad_ax)
            c_map_is = c_map.swapaxes(0, self._nad_ax)
            if not self._all_adjacent:
                ishape = bin_map_is.shape  # internal shape
                flat_shape = (ishape[0], np.prod(ishape[1:]))
            else:
                ishape = shape
                flat_shape = None
            cids = _label_clusters_binary(bin_map_is, c_map_is, self._struct,
                                          self._all_adjacent, flat_shape,
                                          self._connectivity, None)

            # Dataset with cluster info
            cluster_map = NDVar(c_map, p_map.dims, {}, "clusters")
            ds = self._cluster_properties(cluster_map, cids)
            ds.info['clusters'] = cluster_map
            min_pos = ndimage.minimum_position(p_map.x, c_map, cids)
            ds['p'] = Var([p_map.x[pos] for pos in min_pos])

        if 'p' in ds:
            ds['sig'] = star_factor(ds['p'])

        # expand clusters
        if maps:
            shape = (ds.n_cases,) + param_map.shape
            c_maps = np.empty(shape, dtype=param_map.x.dtype)
            c_mask = np.empty(param_map.shape, dtype=np.bool_)
            for i, cid in enumerate(cids):
                np.equal(cluster_map.x, cid, c_mask)
                np.multiply(param_map.x, c_mask, c_maps[i])

            # package ndvar
            dims = ('case',) + param_map.dims
            param_contours = {}
            if self.tail >= 0:
                param_contours[threshold] = (0.7, 0.7, 0)
            if self.tail <= 0:
                param_contours[-threshold] = (0.7, 0, 0.7)
            info = _cs.stat_info(self.meas, contours=param_contours,
                                 summary_func=np.sum)
            ds['cluster'] = NDVar(c_maps, dims, info=info)
        else:
            ds.info['clusters'] = self.cluster_map

        return ds

    def find_peaks(self):
        """Find peaks in a TFCE distribution

        Returns
        -------
        ds : Dataset
            Dataset with information about the peaks.
        """
        if self.kind == 'cluster':
            raise RuntimeError("Not a threshold-free distribution")

        self._allocate_memory_buffers()
        param_map = self._original_param_map
        probability_map = self.probability_map.x.swapaxes(0, self._nad_ax)

        peaks = self._find_peaks(self._original_cluster_map)
        peak_map = self._int_buff
        peak_ids = _label_clusters_binary(peaks, peak_map, self._struct,
                                          self._all_adjacent, self._flat_shape,
                                          self._connectivity, None)

        ds = Dataset()
        ds['id'] = Var(peak_ids)
        v = ds.add_empty_var('v')
        if self.N:
            p = ds.add_empty_var('p')

        for i, id_ in enumerate(peak_ids):
            idx = np.equal(peak_map, id_, self._bin_buff)
            v[i] = param_map[idx][0]
            if self.N:
                p[i] = probability_map[idx][0]

        self._clear_memory_buffers()
        return ds

    def _aggregate_dist(self, **sub):
        """Aggregate permutation distribution to one value per permutation

        Parameters
        ----------
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        dist : array, shape = (N,)
            Maximum value for each permutation in the given region.
        """
        if sub:
            dist_ = NDVar(self.dist, self._dist_dims)
            dist_sub = dist_.sub(**sub)
            dist = dist_sub.x
        else:
            dist = self.dist

        if dist.ndim > 1:
            axes = tuple(xrange(1, dist.ndim))
            dist = dist.max(axes)

        return dist

    def compute_probability_map(self, **sub):
        """Compute a probability map

        Parameters
        ----------
        [dimname] : index
            Limit the data for the distribution.

        Returns
        -------
        probability : NDVar
            Map of p-values.
        """
        if not self.N:
            raise RuntimeError("Can't compute probability without permutations")

        if self.kind == 'cluster':
            cpmap = np.ones(self.shape)
            if self.n_clusters:
                cids = self._cids
                dist = self._aggregate_dist(**sub)
                cluster_map = self._original_cluster_map
                param_map = self._original_param_map

                # measure clusters
                cluster_v = ndimage.sum(param_map, cluster_map, cids)

                # p-values: "the proportion of random partitions that resulted
                # in a larger test statistic than the observed one" (179)
                n_larger = np.sum(dist > np.abs(cluster_v[:, None]), 1)
                cluster_p = n_larger / self.N

                c_mask = np.empty(self.shape, dtype=np.bool8)
                for i, cid in enumerate(cids):
                    np.equal(cluster_map, cid, c_mask)
                    cpmap[c_mask] = cluster_p[i]
            # revert to original shape
            if self._nad_ax:
                cpmap = cpmap.swapaxes(0, self._nad_ax)

            dims = self.dims[1:]
        else:
            if self.kind == 'tfce':
                stat_map = self.tfce_map
            else:
                if self.tail == 0:
                    stat_map = self.parameter_map.abs()
                elif self.tail < 0:
                    stat_map = -self.parameter_map
                else:
                    stat_map = self.parameter_map

            dist = self._aggregate_dist(**sub)
            if sub:
                stat_map = stat_map.sub(**sub)

            idx = np.empty(stat_map.shape, dtype=np.bool8)
            cpmap = np.zeros(stat_map.shape)
            for v in dist:
                cpmap += np.greater(v, stat_map.x, idx)
            cpmap /= self.N
            dims = stat_map.dims

        info = _cs.cluster_pmap_info()
        return NDVar(cpmap, dims, info, self.name)

    def masked_parameter_map(self, pmin=0.05, **sub):
        """Create a copy of the parameter map masked by significance

        Parameters
        ----------
        pmin : None | scalar
            Threshold p-value for masking (default 0.05). For threshold-based
            cluster tests, pmin=None includes all clusters regardless of their
            p-value.

        Returns
        -------
        masked_map : NDVar
            NDVar with data from the original parameter map wherever p <= pmin
            and 0 everywhere else.
        """
        if sub:
            param_map = self.parameter_map.sub(**sub)
        else:
            param_map = self.parameter_map.copy()

        if pmin is None:
            if self.kind != 'cluster':
                msg = "pmin can only be None for thresholded cluster tests"
                raise ValueError(msg)
            c_mask = self.cluster_map.x != 0
        else:
            probability_map = self.compute_probability_map(**sub)
            c_mask = np.less_equal(probability_map.x, pmin)
        param_map.x *= c_mask
        return param_map

    @LazyProperty
    def probability_map(self):
        if self.N:
            return self.compute_probability_map()
        else:
            return None
