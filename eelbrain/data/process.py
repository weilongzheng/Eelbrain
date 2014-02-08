'''
Created on Feb 24, 2012

@author: christian
'''

import numpy as _np

try:
    import mdp as _mdp
except:
    pass

from .data_obj import isfactor, Var, isvar, NDVar, isndvar, asnumeric


def rm_pca(ds, rm=[], source='MEG', target='MEG', baseline=None):
    """
    Perform PCA and remove certain components. Use gui.pca to find components
    initially.

    baseline : None | tuple(tstart, tend)
        Baseline correction after subtracting the PCA components. The baseline
        is specified as a (tstart, tend) tuple. None includes all values until the
        start/end of the epoch (see :func:`.rm_baseline`).

    """
    if not rm:
        raise ValueError("No components selected")
    if isinstance(source, basestring):
        source = ds[source]

    rm = sorted(rm)
    n_comp = max(rm) + 1

    node = PCA(source, n_comp=n_comp)
    ds[target] = node.subtract(rm, name=target, baseline=baseline)



class PCA:
    "Object that performs PCA for the local function and the gui"
    def __init__(self, source, n_comp=None):
        """
        source : NDVar
            source data
        n_comp : None | int
            number of components to compute (None = all)

        """
        self.source = source
        data = self._source_data = source.get_data(('case', 'time', 'sensor'))
        self._source_dtype = data.dtype
        data = data.astype(_np.float64)

        # do the pca
        node = self.node = _mdp.nodes.PCANode(output_dim=n_comp)
        for epoch in data:
            node.train(epoch)
        node.stop_training()

        # project into the pca space
        n_epochs, n_t, n_sensors = data.shape
        old_data = data.reshape((n_epochs * n_t, n_sensors))
        self._proj = self.node.execute(old_data)


    def get_component(self, i):
        dims = self.source.get_dims(('sensor',))
        data = self.node.v.T[i, ]
        name = 'comp_%i' % i
        ndvar = NDVar(data, dims, name=name)
        return ndvar

    def get_load(self, i):
        time = self.source.get_dim('time')
        dims = ('case', time,)
        shape = (len(self.source), len(time))
        data = self._proj[:, i].reshape(shape)
        name = 'comp_%i_load' % i
        ndvar = NDVar(data, dims=dims, name=name)
        return ndvar

    def subtract(self, components, baseline=(None, 0), name='{name}'):
        """
        returns a copy of the source NDVar with the principal
        components specified in ``components`` removed.

        Arguments:

        components : list of ints
            list of components to remove
        baseline : True | False | (int|None, int|None)
            Baseline correction after subtracting the components. True -> use the
            settings stored in the NDVar.info; False -> do not apply any
            baseline correction; a new baseline can be specified with a tuple of
            two time values or None (use all values until the end of the epoch).

        """
        data = self._source_data
        proj = self._proj

        # flatten retained components
        for i in xrange(proj.shape[1]):
            if i not in components:
                proj[:, i] = 0

        # remove the components
        rm_comp_data = self.node.inverse(proj)
        new_data = data - rm_comp_data.reshape(data.shape)
        new_data = new_data.astype(self._source_dtype)

        # create the output new NDVar
        dims = ('case',) + self.source.get_dims(('time', 'sensor'))
        info = self.source.info
        name = name.format(name=self.source.name)
        out = NDVar(new_data, dims=dims, info=info, name=name)
        if baseline:
            tstart, tend = baseline
            out = rm_baseline(out, tstart, tend)
        return out



def mark_by_threshold(ds, x='MEG', threshold=2e-12, above=False,
                      below=None, target='accept', bad_chs=None):
    """
    Marks epochs based on a threshold criterion (any sensor exceeding the
    threshold at any time)

    Parameters
    ----------
    ds : Dataset
        Dataset containing the data.
    x : NDVar
        Dependent variable (data which should be thresholded).
    threshold : scalar
        The threshold value. Examples: 1.25e-11 to detect saturated channels;
        2e-12: for conservative MEG rejection.
    above, below: True, False, None
        How to mark segments that do (above) or do not (below) exceed the
        threshold: True->good; False->bad; None->don't change
    target : Factor | str
        Factor (or its name) in which the result is stored. If ``target`` is
        a string and the Dataset does not contain a Factor by that name, it is
        created.
    bad_chs : None | list of str, int
        Channels to ignore when thresholding.
    """
    x = asnumeric(x, ds=ds)
    if bad_chs:
        idx = x.sensor.index(bad_chs)
        x = x.sub(sensor=idx)

    # get the Factor on which to store results
    if isfactor(target) or isvar(target):
        assert len(target) == ds.n_cases
    elif isinstance(target, basestring):
        if target in ds:
            target = ds[target]
        else:
            target_x = _np.ones(ds.n_cases, dtype=bool)
            target = Var(target_x, name=target)
            ds.add(target)
    else:
        raise ValueError("target needs to be a Factor")

    # do the thresholding
    if isndvar(x):
        for ID in xrange(ds.n_cases):
            data = x[ID]
            v = _np.max(_np.abs(data.x))

            if v > threshold:
                if above is not None:
                    target[ID] = above
            elif below is not None:
                target[ID] = below
    else:
        for ID in xrange(ds.n_cases):
            v = x[ID]

            if v > threshold:
                if above is not None:
                    target[ID] = above
            elif below is not None:
                target[ID] = below



def rm_baseline(ndvar, tstart=None, tend=0, name='{name}'):
    """
    returns an NDVar object with baseline correction applied.

    ndvar : NDVar
        the source data
    tstart : scalar | None
        the beginning of the baseline period (None -> the start of the epoch)
    tend : scalar | None
        the end of the baseline  period (None -> the end of the epoch)
    name : str
        name for the new NDVar

    """
    subdata = ndvar.sub(time=(tstart, tend))
    baseline = subdata.aggregate('time')

    t_ax = subdata.get_axis('time')
    index = (slice(None),) * t_ax + (None,)
    bl_data = baseline.x[index]

    dims = ndvar.dims
    data = ndvar.x - bl_data
    name = name.format(name=ndvar.name)
    return NDVar(data, dims=dims, info=ndvar.info, name=name)


