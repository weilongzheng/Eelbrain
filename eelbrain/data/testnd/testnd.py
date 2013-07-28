'''Statistical tests for ndvars'''
from __future__ import division

from math import ceil

import numpy as np
import scipy.stats
from scipy.stats import percentileofscore
from scipy import ndimage
from scipy.ndimage import binary_closing, binary_erosion, binary_dilation
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import connected_components

from ... import fmtxt
from .. import colorspaces as _cs
from ..data_obj import (ascategorial, asmodel, asndvar, asvar, assub, dataset,
                        factor, ndvar, var, Celltable, cellname, combine)
from ..test import glm as _glm
from ..test.test import resample


__all__ = ['ttest', 'f_oneway', 'anova', 'cluster_anova', 'corr',
           'cluster_corr', 'clean_time_axis']
__test__ = False


def clean_time_axis(pmap, dtmin=0.02, below=None, above=None, null=0):
    """Clean a parameter map by requiring a threshold value for a minimum time
    window.

    Parameters
    ----------
    pmap : ndvar
        Parameter map with time axis.
    dtmin : scalar
        Minimum duration required
    below : scalar | None
        Threshold value for finding clusters: find clusters of values below
        this threshold.
    above : scalar | None
        As ``below``, but for finding clusters above a threshold.
    null : scalar
        Value to substitute outside of clusters.

    Returns
    -------
    cleaned_map : ndvar
        A copy of pmap with all values that do not belong to a cluster set to
        null.
    """
    if below is None and above is None:
        raise TypeError("Need to specify either above or below.")
    elif below is None:
        passes_t = pmap.x >= above
    elif above is None:
        passes_t = pmap.x <= below
    else:
        passes_t = np.logical_and(pmap.x >= above, pmap.x <= below)

    ax = pmap.get_axis('time')
    di_min = int(ceil(dtmin / pmap.time.tstep))
    struct_shape = (1,) * ax + (di_min,) + (1,) * (pmap.ndim - ax - 1)
    struct = np.ones(struct_shape, dtype=int)

    cores = binary_erosion(passes_t, struct)
    keep = binary_dilation(cores, struct)
    x = np.where(keep, pmap.x, null)

    info = pmap.info.copy()
    cleaned = ndvar(x, pmap.dims, info, pmap.name)
    return cleaned



class corr:
    """
    Attributes
    ----------

    r : ndvar
        Correlation (with threshold contours).

    """
    def __init__(self, Y, X, norm=None, sub=None, ds=None,
                 contours={.05: (.8, .2, .0), .01: (1., .6, .0), .001: (1., 1., .0)}):
        """

        Y : ndvar
            Dependent variable.
        X : continuous | None
            The continuous predictor variable.
        norm : None | categorial
            Categories in which to normalize (z-score) X.

        """
        sub = assub(sub, ds)
        Y = asndvar(Y, sub=sub, ds=ds)
        X = asvar(X, sub=sub, ds=ds)
        if norm is not None:
            norm = ascategorial(norm, sub, ds)

        if not Y.has_case:
            msg = ("Dependent variable needs case dimension")
            raise ValueError(msg)

        y = Y.x.reshape((len(Y), -1))
        if norm is not None:
            y = y.copy()
            for cell in norm.cells:
                idx = (norm == cell)
                y[idx] = scipy.stats.mstats.zscore(y[idx])

        n = len(X)
        x = X.x.reshape((n, -1))

        # covariance
        m_x = np.mean(x)
        if np.isnan(m_x):
            raise ValueError("np.mean(x) is nan")
        x -= m_x
        y -= np.mean(y, axis=0)
        cov = np.sum(x * y, axis=0) / (n - 1)

        # correlation
        r = cov / (np.std(x, axis=0) * np.std(y, axis=0))

        # p-value calculation
        # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
        pcont = {}
        df = n - 2
        for p, color in contours.iteritems():
            t = scipy.stats.distributions.t.isf(p, df)
            r_p = t / np.sqrt(n - 2 + t ** 2)
            pcont[r_p] = color
            pcont[-r_p] = color

        dims = Y.dims[1:]
        shape = Y.x.shape[1:]
        info = Y.info.copy()
        info.update(cmap='xpolar', vmax=1, contours=pcont)
        r = ndvar(r.reshape(shape), dims=dims, info=info)

        # store results
        self.name = "%s corr %s" % (Y.name, X.name)
        self.r = r
        self.all = r


class cluster_corr:
    """
    Attributes
    ----------

    r : ndvar
        Correlation (with threshold contours).

    """
    def __init__(self, Y, X, norm=None, sub=None, ds=None,
                 contours={.05: (.8, .2, .0), .01: (1., .6, .0), .001: (1., 1., .0)},
                 tp=.1, samples=1000, replacement=False,
                 tstart=None, tstop=None, close_time=0):
        """

        Y : ndvar
            Dependent variable.
        X : continuous | None
            The continuous predictor variable.
        norm : None | categorial
            Categories in which to normalize (z-score) X.

        """
        sub = assub(sub, ds)
        Y = asndvar(Y, sub=sub, ds=ds)
        X = asvar(X, sub=sub, ds=ds)
        if norm is not None:
            norm = ascategorial(norm, sub, ds)

        self.name = name = "%s corr %s" % (Y.name, X.name)

        # calculate threshold
        # http://en.wikipedia.org/wiki/Pearson_product-moment_correlation_coefficient#Inference
        self.n = n = len(X)
        df = n - 2
        tt = scipy.stats.distributions.t.isf(tp, df)
        tr = tt / np.sqrt(df + tt ** 2)

        # normalization is done before the permutation b/c we are interested in
        # the variance associated with each subject for the z-scoring.
        Y = Y.copy()
        Y.x = Y.x.reshape((n, -1))
        if norm is not None:
            for cell in norm.cells:
                idx = (norm == cell)
                Y.x[idx] = scipy.stats.mstats.zscore(Y.x[idx])

        x = X.x.reshape((n, -1))
        m_x = np.mean(x)
        if np.isnan(m_x):
            raise ValueError("np.mean(x) is nan")
        self.x = x - m_x

        cdist = _ClusterDist(Y, samples, t_upper=tr, t_lower=-tr,
                            tstart=tstart, tstop=tstop, close_time=close_time,
                            meas='r', name=name)
        r = self._corr(Y)
        cdist.add_original(r)

        if cdist.n_clusters:
            for Yrs in resample(Y, samples, replacement=replacement):
                r = self._corr(Yrs)
                cdist.add_perm(r)

        self.cdist = cdist
        self.r_map = cdist.pmap
        self.all = [[self.r_map, cdist.cpmap]]
        self.clusters = cdist.clusters

    def _corr(self, Y):
        n = self.n
        x = self.x

        # covariance
        y = Y.x - np.mean(Y.x, axis=0)
        cov = np.sum(x * y, axis=0) / (n - 1)

        # correlation
        r = cov / (np.std(x, axis=0) * np.std(y, axis=0))
        return r

    def as_table(self, pmax=1.):
        table = self.cdist.as_table(pmax=pmax)
        return table


class ttest:
    """Element-wise t-test

    Attributes
    ----------
    all :
        c1, c0, [c0 - c1, P]
    p_val :
        [c0 - c1, P]
    """
    def __init__(self, Y='meg', X=None, c1=None, c0=0, match=None, sub=None,
                 ds=None, samples=None, pmin=0.1, tstart=None, tstop=None,):
        """Element-wise t-test

        Parameters
        ----------
        Y : ndvar
            Dependent variable.
        X : categorial | None
            Model; None if the grand average should be tested against a
            constant.
        c1 : str | tuple | None
            Test condition (cell of X).
        c0 : str | tuple | scalar
            Control condition (cell of X or constant to test against).
        match : factor
            Match cases for a repeated measures t-test.
        sub : index-array
            perform test with a subset of the data
        ds : dataset
            If a dataset is specified, all data-objects can be specified as
            names of dataset variables
        samples : None | int
            Number of samples for permutation cluster test. For None, no
            clusters are formed.
        pmin : scalar (0 < pmin < 1)
            Threshold p value for forming clusters in permutation cluster test.
        tstart, tstop : None | scalar
            Restrict time window for permutation cluster test.
        """
        if c1 is None:
            if len(X.cells) == 1:
                c1 = X.cells[0]
            elif len(X.cells) == 2:
                c1, c0 = X.cells
            else:
                err = ("If X has more than 2 categories, c1 and c0 must be "
                       "explicitly specified.")
                raise ValueError(err)

        if isinstance(c0, (str, tuple)):
            cat = (c1, c0)
        else:
            cat = (c1,)
        ct = Celltable(Y, X, match, sub, cat=cat, ds=ds)

        if isinstance(c0, (basestring, tuple)):  # two samples
            c1_mean = ct.data[c1].summary(name=cellname(c1))
            c0_mean = ct.data[c0].summary(name=cellname(c0))
            diff = c1_mean - c0_mean
            if ct.all_within:
                test_name = 'Related Samples t-Test'
                n = len(ct.Y) / 2
                df = n - 1
                T = _t_rel(ct.Y)
                P = _ttest_p(T, df)
                if samples:
                    tmin = _ttest_t(pmin, df)
                    cdist = _ClusterDist(ct.Y, samples, tmin, -tmin, 't', test_name,
                                        tstart, tstop)
                    cdist.add_original(T)
                    if cdist.n_clusters:
                        for Y_ in resample(cdist.Y_perm, samples, replacement=False):
                            tmap = _t_rel(Y_.x)
                            cdist.add_perm(tmap)
            else:
                test_name = 'Independent Samples t-Test'
                n1 = len(ct.data[c1])
                N = len(ct.Y)
                n2 = N - n1
                df = N - 2
                T = _t_ind(ct.Y.x, n1, n2)
                P = _ttest_p(T, df)
                if samples:
                    tmin = _ttest_t(pmin, df)
                    cdist = _ClusterDist(ct.Y, samples, tmin, -tmin, 't', test_name,
                                        tstart, tstop)
                    cdist.add_original(T)
                    if cdist.n_clusters:
                        for Y_ in resample(cdist.Y_perm, samples, replacement=False):
                            tmap = _t_ind(Y_.x, n1, n2)
                            cdist.add_perm(tmap)
                n = (n1, n2)
        elif np.isscalar(c0):  # one sample
            c1_data = ct.data[c1]
            x = c1_data.x
            c1_mean = c1_data.summary()
            c0_mean = None

            # compute T and P
            if np.prod(x.shape) > 2 ** 25:
                ax = np.argmax(x.shape[1:]) + 1
                x = x.swapaxes(ax, 1)
                mod_len = x.shape[1]
                fix_shape = x.shape[0:1] + x.shape[2:]
                N = 2 ** 25 // np.prod(fix_shape)
                res = [scipy.stats.ttest_1samp(x[:, i:i + N], popmean=c0, axis=0)
                       for i in xrange(0, mod_len, N)]
                T = np.vstack((v[0].swapaxes(ax, 1) for v in res))
                P = np.vstack((v[1].swapaxes(ax, 1) for v in res))
            else:
                T, P = scipy.stats.ttest_1samp(x, popmean=c0, axis=0)

            n = len(c1_data)
            df = n - 1
            test_name = '1-Sample t-Test'
            if c0:
                diff = c1_mean - c0
            else:
                diff = c1_mean
        else:
            raise ValueError('invalid c0: %r. Must be string or scalar.' % c0)

        dims = ct.Y.dims[1:]

        info = _cs.set_info_cs(ct.Y.info, _cs.sig_info())
        info['test'] = test_name
        P = ndvar(P, dims, info=info, name='p')

        info = _cs.set_info_cs(ct.Y.info, _cs.default_info('T', vmin=0))
        T = ndvar(T, dims, info=info, name='T')

        # diff
        if np.any(diff < 0):
            diff.info['cmap'] = 'xpolar'

        # add Y.name to dataset name
        Yname = getattr(Y, 'name', None)
        if Yname:
            test_name = ' of '.join((test_name, Yname))

        # store attributes
        self.t = T
        self.p = P
        self.n = n
        self.df = df
        self.name = test_name
        self.c1_mean = c1_mean
        if c0_mean:
            self.c0_mean = c0_mean

        self.diff = diff
        self.p_val = [[diff, P]]

        if c0_mean:
            all_uncorrected = [c1_mean, c0_mean] + self.p_val
        elif c0:
            all_uncorrected = [c1_mean] + self.p_val
        else:
            all_uncorrected = self.p_val

        if samples:
            self.diff_cl = [diff, cdist.cpmap]
            self.all = [c1_mean, c0_mean, self.diff_cl]
            self.cdist = cdist
            self.clusters = cdist.clusters
        else:
            self.all = all_uncorrected


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


def _t_rel(Y):
    """
    Calculates the T statistic on two related samples.

    Parameters
    ----------
    Y : array
        Dependent variable in right input format: The first half and second
        half of the data represent the two samples; in each subjects

    Returns
    -------
    t : array
        t-statistic

    Notes
    -----
    Based on scipy.stats.ttest_rel
    df = n - 1
    """
    n = len(Y) / 2
    a = Y[:n]
    b = Y[n:]
    d = (a - b).astype(np.float64)
    v = np.var(d, 0, ddof=1)
    dm = np.mean(d, 0)
    denom = np.sqrt(v / n)
    t = np.divide(dm, denom)
    return t


def _ttest_p(t, df):
    "Two tailed probability"
    p = scipy.stats.t.sf(np.abs(t), df) * 2
    return p


def _ttest_t(p, df):
    "Positive t value for two tailed probability"
    t = scipy.stats.distributions.t.isf(p / 2, df)
    return t


class f_oneway:
    def __init__(self, Y='MEG', X='condition', sub=None, ds=None,
                 p=.05, contours={.01: '.5', .001: '0'}):
        """
        uses scipy.stats.f_oneway

        """
        sub = assub(sub, ds)
        Y = asndvar(Y, sub, ds)
        X = ascategorial(X, sub, ds)

        Ys = [Y[X == c] for c in X.cells]
        Ys = [y.x.reshape((y.x.shape[0], -1)) for y in Ys]
        N = Ys[0].shape[1]

        Ps = []
        for i in xrange(N):
            groups = (y[:, i] for y in Ys)
            F, p = scipy.stats.f_oneway(*groups)
            Ps.append(p)
        test_name = 'One-way ANOVA'

        dims = Y.dims[1:]
        Ps = np.reshape(Ps, tuple(len(dim) for dim in dims))

        info = _cs.set_info_cs(Y.info, _cs.sig_info(p, contours))
        info['test'] = test_name
        p = ndvar(Ps, dims, info=info, name=X.name)

        # store results
        self.name = "anova"
        self.p = p
        self.all = p



class anova:
    """
    Attributes
    ----------

    Y : ndvar
        Dependent variable.
    X : model
        Model.
    p_maps : {effect -> ndvar}
        Maps of p-values.
    all : [ndvar]
        List of all p-maps.

    """
    def __init__(self, Y, X, sub=None, ds=None, p=.05,
                 contours={.01: '.5', .001: '0'}):
        sub = assub(sub, ds)
        Y = self.Y = asndvar(Y, sub, ds)
        X = self.X = asmodel(X, sub, ds)

        self.name = "anova"

        fitter = _glm.lm_fitter(X)

        info = _cs.set_info_cs(Y.info, _cs.sig_info(p, contours))
        kwargs = dict(dims=Y.dims[1:], info=info)

        self.all = []
        self.p_maps = {}
        for e, _, Ps in fitter.map(Y.x):
            name = e.name
            P = ndvar(Ps, name=name, **kwargs)
            self.all.append(P)
            self.p_maps[e] = P


class cluster_anova:
    """
    Attributes
    ----------

    Y : ndvar
        Dependent variable.
    X : model
        Model.

    all other attributes are dictionaries mapping effects from X.effects to
    results

    F_maps : {effect -> ndvar{
        Maps of F-values.

    """
    def __init__(self, Y, X, t=.1, samples=1000, replacement=False,
                 tstart=None, tstop=None, close_time=0, sub=None, ds=None):
        """ANOVA with cluster permutation test

        Parameters
        ----------
        Y : ndvar
            Measurements (dependent variable)
        X : categorial
            Model
        t : scalar
            Threshold (uncorrected p-value) to use for finding clusters
        samples : int
            Number of samples to estimate parameter distributions
        replacement : bool
            whether random samples should be drawn with replacement or
            without
        tstart, tstop : None | scalar
            Time window for clusters.
            **None**: use the whole epoch;
            **scalar**: use only a part of the epoch

            .. Note:: implementation is not optimal: F-values are still
                computed but ignored.

        close_time : scalar
            Close gaps in clusters that are smaller than this interval. Assumes
            that Y is a uniform time series.
        sub : index
            Apply analysis to a subset of cases in Y, X


        .. FIXME:: connectivity for >2 dimensional data. Currently, adjacent
            samples are connected.

        """
        sub = assub(sub, ds)
        Y = self.Y = asndvar(Y, sub, ds)
        X = self.X = asmodel(X, sub, ds)

        lm = _glm.lm_fitter(X)

        # get F-thresholds from p-threshold
        tF = {}
        if lm.full_model:
            for e in lm.E_MS:
                effects_d = lm.E_MS[e]
                if effects_d:
                    df_d = sum(ed.df for ed in effects_d)
                    tF[e] = scipy.stats.distributions.f.isf(t, e.df, df_d)
        else:
            df_d = X.df_error
            tF = {e: scipy.stats.distributions.f.isf(t, e.df, df_d)
                  for e in X.effects}

        # Estimate statistic distributions from permuted Ys
        kwargs = dict(meas='F', tstart=tstart, tstop=tstop,
                      close_time=close_time)
        cdists = {e: _ClusterDist(Y, samples, tF[e], name=e.name, **kwargs)
                  for e in tF}

        # Find clusters in the actual data
        test0 = lm.map(Y.x, p=False)
        self.effects = []
        self.F_maps = {}
        for e, F in test0:
            self.effects.append(e)
            dist = cdists[e]
            dist.add_original(F)
            self.F_maps[e] = dist.P

        for Y_ in resample(Y, samples, replacement=replacement):
            for e, F in lm.map(Y_.x, p=False):
                cdists[e].add_perm(F)

        self.name = "ANOVA Permutation Cluster Test"
        self.tF = tF
        self.cdists = cdists

        dss = []
        for e in self.effects:
            name = e.name
            ds = cdists[e].clusters
            ds['effect'] = factor([name], rep=ds.n_cases)
            dss.append(ds)
        ds = combine(dss)
        self.clusters = ds

        self.all = [[self.F_maps[e]] + self.cdists[e].clusters
                    for e in self.X.effects if e in self.F_maps]


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
    def __init__(self, Y, N, t_upper, t_lower=None, meas='?', name=None,
                 tstart=None, tstop=None, close_time=0):
        """Accumulate information on a cluster statistic.

        Parameters
        ----------
        Y : ndvar
            Dependent variable.
        N : int
            Number of permutations.
        t_upper, t_lower : None | scalar
            Positive and negative thresholds for finding clusters. If None,
            no clusters with the corresponding sign are counted.
        meas : str
            Label for the parameter measurement (e.g., 't' for t-values).
        name : None | str
            Name for the comparison.
        tstart, tstop : None | scalar
            Restrict the time window for finding clusters (None: use the whole
            epoch).
        close_time : scalar
            Close gaps in clusters that are smaller than this interval. Assumes
            that Y is a uniform time series.
        """
        assert Y.has_case
        if t_lower is not None:
            if t_lower >= 0:
                raise ValueError("t_lower needs to be < 0; is %s" % t_lower)
        if t_upper is not None:
            if t_upper <= 0:
                raise ValueError("t_upper needs to be > 0; is %s" % t_upper)
        if (t_lower is not None) and (t_upper is not None):
            if t_lower != -t_upper:
                err = ("If t_upper and t_lower are defined, t_upp has to be "
                       "-t_lower")
                raise ValueError(err)

        # prepare gap closing
        if close_time:
            raise NotImplementedError
            time = Y.get_dim('time')
            self._close = np.ones(round(close_time / time.tstep))
        else:
            self._close = None

        # prepare cropping
        if (tstart is None) and (tstop is None):
            self.crop = False
            Y_perm = Y
        else:
            self.crop = True
            Y_perm = Y.subdata(time=(tstart, tstop))
            istart = 0 if tstart is None else Y.time.index(tstart, 'up')
            istop = istart + len(Y_perm.time)
            t_ax = Y.get_axis('time') - 1
            self._crop_idx = (slice(None),) * t_ax + (slice(istart, istop),)
            self._uncropped_shape = Y.shape[1:]

        # prepare adjacency
        adjacent = [d.adjacent for d in Y_perm.dims[1:]]
        self._all_adjacent = all_adjacent = all(adjacent)
        if not all_adjacent:
            if sum(adjacent) < len(adjacent) - 1:
                err = ("more than one non-adjacent dimension")
                raise NotImplementedError(err)
            self._nad_ax = ax = adjacent.index(False)
            self._conn = Y_perm.dims[ax + 1].connectivity()
            struct = ndimage.generate_binary_structure(2, 1)
            struct[::2] = False
            self._struct = struct
            # flattening and reshaping pmaps with swapped axes
            shape = Y_perm.shape[1:]
            if ax:
                shape = list(shape)
                shape[0], shape[ax] = shape[ax], shape[0]
                shape = tuple(shape)
            self._orig_shape = shape
            self._flat_shape = (shape[0], -1)

        self.Y = Y
        self.Y_perm = Y_perm
        self.dist = np.zeros(N)
        self._i = int(N)
        self.t_upper = t_upper
        self.t_lower = t_lower
        self.tstart = tstart
        self.tstop = tstop
        self.meas = meas
        self.name = name

    def _crop(self, im):
        if self.crop:
            return im[self._crop_idx]
        else:
            return im

    def _finalize(self):
        if self._i < 0:
            raise RuntimeError("Too many permutations added to _ClusterDist")

        # retrieve original clusters
        pmap = self._original_pmap
        pmap_ = self._crop(pmap)
        cmap = self._cluster_im
        cids = self._cids

        if not self.n_clusters:
            self.clusters = None
            return

        # measure original clusters
        cluster_v = ndimage.sum(pmap_, cmap, cids)
        cluster_p = np.array([1 - percentileofscore(self.dist, abs(v), 'mean')
                              / 100 for v in cluster_v])
        sort_idx = np.argsort(cluster_p)

        # prepare container for clusters
        ds = dataset()
        ds['p'] = var(cluster_p[sort_idx])
        ds['v'] = var(cluster_v[sort_idx])

        # time window
        time = self.Y_perm.get_dim('time') if self.Y.has_dim('time') else None
        if time is not None:
            time_ax = self.Y.get_axis('time') - 1
            tstart = []
            tstop = []

        # create cluster ndvars
        cpmap = np.ones_like(pmap_)
        cmaps = np.empty((self.n_clusters,) + pmap.shape, dtype=pmap.dtype)
        boundaries = ndimage.find_objects(cmap)
        for i, ci in enumerate(sort_idx):
            v = cluster_v[ci]
            p = cluster_p[ci]
            cid = cids[ci]

            # update cluster maps
            c_mask = (cmap == cid)
            cpmap[c_mask] = p
            cmaps[i] = self._uncrop(pmap_ * c_mask)

            # extract cluster properties
            bounds = boundaries[cid - 1]
            if time is not None:
                t_slice = bounds[time_ax]
                tstart.append(time.times[t_slice.start])
                if t_slice.stop == len(time):
                    tstop.append(time.times[-1] + time.tstep)
                else:
                    tstop.append(time.times[t_slice.stop])

        dims = self.Y.dims
        contours = {self.t_lower: (0.7, 0, 0.7), self.t_upper: (0.7, 0.7, 0)}
        info = _cs.stat_info(self.meas, contours=contours, summary_func=np.sum)
        ds['cluster'] = ndvar(cmaps, dims=dims, info=info)

        if time is not None:
            ds['tstart'] = var(tstart)
            ds['tstop'] = var(tstop)
        self.clusters = ds

        # cluster probability map
        cpmap = self._uncrop(cpmap, 1)
        info = _cs.cluster_pmap_info()
        self.cpmap = ndvar(cpmap, dims=dims[1:], name=self.name, info=info)

        # statistic parameter map
        info = _cs.stat_info(self.meas, contours=contours)
        self.pmap = ndvar(pmap, dims=dims[1:], name=self.name, info=info)

        self.all = [[self.pmap, self.cpmap]]

    def _label_clusters(self, pmap):
        """Find clusters on a statistical parameter map

        Parameters
        ----------
        pmap : array
            Statistical parameter map (flattened if the data contains
            non-adjacent dimensions).

        Returns
        -------
        cluster_map : array
            Array of same shape as pmap with clusters labeled.
        clusters : tuple
            Identifiers of the clusters.
        """
        if self.t_upper is not None:
            bin_map_above = (pmap > self.t_upper)
            cmap, cids = self._label_clusters_1tailed(bin_map_above)

        if self.t_lower is not None:
            bin_map_below = (pmap < self.t_lower)
            if self.t_upper is None:
                cmap, cids = self._label_clusters_1tailed(bin_map_below)
            else:
                cmap_l, cids_l = self._label_clusters_1tailed(bin_map_below)
                x = cmap.max()
#                 cmap_l += x * bin_map_below  # faster?
                cmap_l[bin_map_below] += x
                cmap += cmap_l
                cids.update(c + x for c in cids_l)

        return cmap, tuple(cids)

    def _label_clusters_1tailed(self, bin_map):
        """
        Parameters
        ----------
        bin_map : array
            Binary map of where the parameter map exceeds the threshold for a
            cluster.

        Returns
        -------
        cluster_map : array
            Array of same shape as bin_map with clusters labeled.
        cluster_ids : iterator over int
            Identifiers of the clusters.
        """
        # manipulate morphology
        if self._close is not None:
            bin_map = bin_map | binary_closing(bin_map, self._close)

        # find clusters
        if self._all_adjacent:
            cmap, n = ndimage.label(bin_map)
            return cmap, set(xrange(1, n + 1))
        else:
            c = self._conn
            cmap, n = ndimage.label(bin_map, self._struct)
            cids = set(xrange(1, n + 1))
            n_chan = len(cmap)

            for i in xrange(bin_map.shape[1]):
                if len(np.setdiff1d(cmap[:, i], np.zeros(1), False)) <= 1:
                    continue

                idx = np.flatnonzero(cmap[:, i])
                c_idx = np.logical_and(np.in1d(c.row, idx), np.in1d(c.col, idx))
                row = c.row[c_idx]
                col = c.col[c_idx]
                data = c.data[c_idx]
                n = np.max(idx)
                c_ = coo_matrix((data, (row, col)), shape=c.shape)
                n_, lbl_map = connected_components(c_, False)
                if n_ == n_chan:
                    continue
                labels_ = np.flatnonzero(np.bincount(lbl_map) > 1)
                for lbl in labels_:
                    idx_ = lbl_map == lbl
                    merge = np.unique(cmap[idx_, i])

                    # merge labels
                    idx_ = reduce(np.logical_or, (cmap == m for m in merge))
                    cmap[idx_] = merge[0]
                    cids.difference_update(merge[1:])

                if len(cids) == 1:
                    break

            return cmap, cids

    def _uncrop(self, im, background=0):
        if self.crop:
            im_ = np.empty(self._uncropped_shape, dtype=im.dtype)
            im_[:] = background
            im_[self._crop_idx] = im
            return im_
        else:
            return im

    def add_original(self, pmap):
        """Add the originl statistical parameter map.

        Parameters
        ----------
        pmap : array
            Parameter map of the statistic of interest (uncropped).
        """
        if hasattr(self, '_cluster_im'):
            raise RuntimeError("Original pmap already added")

        pmap_ = self._crop(pmap)
        if not self._all_adjacent:
            pmap_ = pmap_.swapaxes(0, self._nad_ax)
            pmap_ = pmap_.reshape(self._flat_shape)
        cmap, cids = self._label_clusters(pmap_)
        if not self._all_adjacent:  # return cmap to proper shape
            cmap = cmap.reshape(self._orig_shape)
            cmap = cmap.swapaxes(0, self._nad_ax)

        self._cluster_im = cmap
        self._original_pmap = pmap
        self._cids = cids
        self.n_clusters = len(cids)
        if self.n_clusters == 0:
            self._finalize()

    def add_perm(self, pmap):
        """Add the statistical parameter map from permuted data.

        Parameters
        ----------
        pmap : array
            Parameter map of the statistic of interest.
        """
        self._i -= 1

        if not self._all_adjacent:
            pmap = pmap.swapaxes(0, self._nad_ax)
            pmap = pmap.reshape(self._flat_shape)

        cmap, cids = self._label_clusters(pmap)
        if cids:
            clusters_v = ndimage.sum(pmap, cmap, cids)
            self.dist[self._i] = np.max(np.abs(clusters_v))

        if self._i == 0:
            self._finalize()

    def as_table(self, pmax=1.):
        cols = 'll'
        headings = ('#', 'p')
        time = self.Y.get_dim('time') if self.Y.has_dim('time') else None
        if time is not None:
            time_ax = self.Y.get_axis('time') - 1
            any_axes = tuple(i for i in xrange(self.Y.ndim - 1) if i != time_ax)
            cols += 'l'
            headings += ('time interval',)

        table = fmtxt.Table(cols)
        table.cells(*headings)
        table.midrule()

        i = 0
        for c in self.clusters:
            p = c.info['p']
            if p <= pmax:
                table.cell(i)
                i += 1
                table.cell(p)

                if time is not None:
                    nz = np.flatnonzero(np.any(c.x, axis=any_axes))
                    tstart = time[nz.min()]
                    tstop = time[nz.max()]
                    interval = '%.3f - %.3f s' % (tstart, tstop)
                    table.cell(interval)

        return table
