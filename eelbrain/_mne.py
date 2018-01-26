from itertools import izip
from math import ceil, floor
import os
import re
import warnings

import numpy as np
import scipy as sp
from scipy.spatial.distance import cdist

import mne
from mne.label import Label, BiHemiLabel
from mne.utils import get_subjects_dir

from ._data_obj import NDVar, SourceSpace, VolumeSourceSpace
from ._ndvar import set_parc


ICO_N_VERTICES = (12, 42, 162, 642, 2562, 10242, 40962)
ICO_SLICE_SUBJECTS = ('fsaverage', 'fsaverage_sym')


def assert_subject_exists(subject, subjects_dir):
    if not os.path.exists(os.path.join(subjects_dir, subject)):
        raise IOError("Subject %s does not exist in subjects_dir %s" %
                      (subject, subjects_dir))


def complete_source_space(ndvar, fill=0.):
    """Fill in missing vertices on an NDVar with a partial source space

    Parameters
    ----------
    ndvar : NDVar  (..., source, ...)
        NDVar with SourceSpace dimension that is missing some vertices.
    fill : scalar
        Value to fill in for missing vertices.

    Returns
    -------
    completed_ndvar : NDVar
        Copy of ``ndvar`` with its SourceSpace dimension completed.
    """
    source = ndvar.get_dim('source')
    axis = ndvar.get_axis('source')
    lh_vertices, rh_vertices = source_space_vertices(source.kind, source.grade,
                                                     source.subject,
                                                     source.subjects_dir)
    shape = list(ndvar.shape)
    shape[axis] = len(lh_vertices) + len(rh_vertices)
    x = np.empty(shape, ndvar.x.dtype)
    x.fill(fill)
    lh_index = np.in1d(lh_vertices, source.lh_vertices, True)
    rh_index = np.in1d(rh_vertices, source.rh_vertices, True)
    index = (slice(None,),) * axis + (np.concatenate((lh_index, rh_index)),)
    x[index] = ndvar.x
    dims = list(ndvar.dims)
    parc = None if source.parc is None else source.parc.name
    dims[axis] = SourceSpace((lh_vertices, rh_vertices), source.subject,
                             source.src, source.subjects_dir, parc)
    return NDVar(x, dims, ndvar.info.copy(), ndvar.name)


def source_space_vertices(kind, grade, subject, subjects_dir):
    """Vertices in ico-``grade`` source space"""
    if kind == 'ico' and subject in ICO_SLICE_SUBJECTS:
        n = ICO_N_VERTICES[grade]
        return np.arange(n), np.arange(n)
    path = SourceSpace._SRC_PATH.format(subjects_dir=subjects_dir,
                                        subject=subject, src='ico-%i' % grade)
    if os.path.exists(path):
        src_to = mne.read_source_spaces(path)
        return src_to[0]['vertno'], src_to[1]['vertno']
    elif kind != 'ico':
        raise NotImplementedError("Can't infer vertices for non-ico source space")
    else:
        return mne.grade_to_vertices(subject, grade, subjects_dir)


def _vertices_equal(v1, v0):
    "Test whether v1 and v0 are equal"
    return np.array_equal(v1[0], v0[0]) and np.array_equal(v1[1], v0[1])


def shift_mne_epoch_trigger(epochs, trigger_shift, min_shift=None, max_shift=None):
    """Shift the trigger in an MNE Epochs object

    Parameters
    ----------
    epochs : mne.Epochs
        Epochs object.
    trigger_shift : scalar sequence
        For each event in ``epochs`` the amount of time by which to shift the
        trigger (in seconds).
    min_shift : scalar (optional)
        Minimum time shift, used to crop data (default is
        ``min(trigger_shift)``).
    max_shift : scalar (optional)
        Maximum time shift, used to crop data (default is
        ``max(trigger_shift)``).

    Returns
    -------
    shifted_epochs : mne.EpochsArray
        Epochs object in which data and timing information is shifted to set
        the time point t=0 to the new trigger position. Data is temporally
        cropped so that only time points with information on all the epochs
        are contained in ``shifted_epochs``.
    """
    data = epochs.get_data()
    tstep = 1. / epochs.info['sfreq']
    shifts = [int(round(x / tstep)) for x in trigger_shift]
    if min_shift is None:
        min_shift = min(shifts)
    else:
        min_shift = int(floor(min_shift / tstep))
        if any(shift < min_shift for shift in shifts):
            invalid = (i for i, shift in enumerate(shifts) if shift < min_shift)
            raise ValueError("The post_baseline_trigger_shift is smaller than "
                             "min_shift at the following events %s" %
                             ', '.join(map(str, invalid)))

    if max_shift is None:
        max_shift = max(shifts)
    else:
        max_shift = int(ceil(max_shift / tstep))
        if any(shift > max_shift for shift in shifts):
            invalid = (i for i, shift in enumerate(shifts) if shift > max_shift)
            raise ValueError("The post_baseline_trigger_shift is greater than "
                             "max_shift at the following events %s" %
                             ', '.join(map(str, invalid)))

    x, y, z = data.shape
    start_offset = -min_shift
    stop_offset = z - max_shift
    new_shape = (x, y, z - (max_shift - min_shift))
    new_data = np.empty(new_shape, data.dtype)
    for i, shift in enumerate(shifts):
        new_data[i] = data[i, :, start_offset + shift:stop_offset + shift]
    tmin = epochs.tmin + (start_offset / float(epochs.info['sfreq']))
    return mne.EpochsArray(new_data, epochs.info, epochs.events, tmin,
                           epochs.event_id)


def labels_from_clusters(clusters, names=None):
    """Create Labels from source space clusters

    Parameters
    ----------
    clusters : NDVar
        NDVar which is non-zero on the cluster. Can have a case dimension to
        define multiple labels (one label per case).
    names : None | list of str | str
        Label names corresponding to clusters (default is "cluster%i").

    Returns
    -------
    labels : list of mne.Label
        One label for each cluster.

    See Also
    --------
    NDVar.label_clusters : clusters from thresholding data
    """
    from mne.label import _n_colors

    if isinstance(names, basestring):
        names = [names]

    source = clusters.source
    source_space = clusters.source.get_source_space()
    subject = source.subject
    collapse = tuple(dim for dim in clusters.dimnames if dim not in ('case', 'source'))
    if collapse:
        clusters_index = clusters.any(collapse)
    else:
        clusters_index = clusters != 0

    if clusters_index.has_case:
        n_clusters = len(clusters)
    else:
        n_clusters = 1
        clusters_index = (clusters_index,)

    if names is None:
        names = ("cluster%i" % i for i in xrange(n_clusters))
    elif len(names) != n_clusters:
        err = "Number of names difference from number of clusters."
        raise ValueError(err)

    colors = _n_colors(n_clusters)
    labels = []
    for cluster, color, name in izip(clusters_index, colors, names):
        lh_vertices = source.lh_vertices[cluster.x[:source.lh_n]]
        rh_vertices = source.rh_vertices[cluster.x[source.lh_n:]]
        if len(lh_vertices) and len(rh_vertices):
            lh = Label(lh_vertices, hemi='lh', name=name + '-lh',
                       subject=subject, color=color).fill(source_space)
            rh = Label(rh_vertices, hemi='rh', name=name + '-rh',
                       subject=subject, color=color).fill(source_space)
            label = BiHemiLabel(lh, rh, name, color)
        elif len(lh_vertices):
            label = Label(lh_vertices, hemi='lh', name=name + '-lh',
                          subject=subject, color=color).fill(source_space)
        elif len(rh_vertices):
            label = Label(rh_vertices, hemi='rh', name=name + '-lh',
                          subject=subject, color=color).fill(source_space)
        else:
            raise ValueError("Empty Cluster")
        labels.append(label)

    return labels


def labels_from_mni_coords(seeds, extent=30., subject='fsaverage',
                           surface='white', mask=None, subjects_dir=None,
                           parc=None):
    """Create a parcellation from seed coordinates in MNI space

    Parameters
    ----------
    seeds : dict
        Seed coordinates. Keys are label names, including -hemi tags. values
        are seeds (array_like of shape (3,) or (3, n_seeds)).
    extent : scalar
        Extent of the label in millimeters (maximum distance from the seed).
    subject : str
        MRI-subject to use (default 'fsaverage').
    surface : str
        Surface to use (default 'white').
    mask : None | str
        A parcellation used to mask the parcellation under construction.
    subjects_dir : str
        SUBJECTS_DIR.
    parc : None | str
        Name of the parcellation under construction (only used for error
        messages).
    """
    name_re = re.compile("^\w+-(lh|rh)$")
    matches = {name: name_re.match(name) for name in seeds}
    invalid = sorted(name for name, m in matches.iteritems() if m is None)
    if invalid:
        raise ValueError(
            "Invalid seed names in parc %r: %s; seed names need to conform to "
            "the 'xxx-lh' or 'xxx-rh' scheme so that the proper hemisphere can "
            "be selected" % (parc, ', '.join(map(repr, sorted(invalid)))))

    # load surfaces
    subjects_dir = get_subjects_dir(subjects_dir)
    fpath = os.path.join(subjects_dir, subject, 'surf', '.'.join(('%s', surface)))
    surfs = {hemi: mne.read_surface(fpath % hemi) for hemi in ('lh', 'rh')}

    # prepare seed properties for mne.grow_labels
    vertices = []
    names = []
    hemis = []
    for name, coords_ in seeds.iteritems():
        coords = np.atleast_2d(coords_)
        if coords.ndim != 2 or coords.shape[1] != 3:
            raise ValueError("Invalid coordinate specification for seed %r in "
                             "parc %r: %r. Seeds need to be specified as "
                             "arrays with shape (3,) or (n_seeds, 3)."
                             % (name, parc, coords_))
        hemi = matches[name].group(1)
        seed_verts = []
        for coord in coords:
            dist = np.sqrt(np.sum((surfs[hemi][0] - coord) ** 2, axis=1))
            seed_verts.append(np.argmin(dist))
        vertices.append(seed_verts)
        names.append(name)
        hemis.append(hemi == 'rh')

    # grow labels
    labels = mne.grow_labels(subject, vertices, extent, hemis, subjects_dir,
                             1, False, names, surface)

    # apply mask
    if mask is not None:
        mlabels = mne.read_labels_from_annot(subject, mask,
                                             subjects_dir=subjects_dir)
        unknown = {l.hemi: l for l in mlabels if l.name.startswith('unknown-')}

        for label in labels:
            rm = unknown[label.hemi]
            if np.any(np.in1d(label.vertices, rm.vertices)):
                label.vertices = np.setdiff1d(label.vertices, rm.vertices, True)

    return labels


def morph_source_space(ndvar, subject_to, vertices_to=None, morph_mat=None,
                       copy=False, parc=True, xhemi=False, mask=None):
    """Morph source estimate to a different MRI subject

    Parameters
    ----------
    ndvar : NDVar
        NDVar with SourceSpace dimension.
    subject_to : string
        Name of the subject on which to morph.
    vertices_to : None | list of array of int | 'lh' | 'rh'
        The vertices on the destination subject's brain. If ndvar contains a
        whole source space, vertices_to can be automatically loaded, although
        providing them as argument can speed up processing by a second or two.
        Use 'lh' or 'rh' to target vertices from only one hemisphere.
    morph_mat : None | sparse matrix
        The morphing matrix. If ndvar contains a whole source space, the morph
        matrix can be automatically loaded, although providing a cached matrix
        can speed up processing by a second or two.
    copy : bool
        Make sure that the data of ``morphed_ndvar`` is separate from
        ``ndvar`` (default False).
    parc : bool | str
        Parcellation for target source space. The default is to keep the
        parcellation from ``ndvar``. Set to ``False`` to load no parcellation.
        If the annotation files are missing for the target subject an IOError
        is raised.
    xhemi : bool
        Mirror hemispheres (i.e., project data from the left hemisphere to the
        right hemisphere and vice versa).
    mask : bool
        Remove sources in "unknown-" labels (default is True unless ``ndvar``
        contains sources with "unknown-" label or ``vertices_to`` is specified).

    Returns
    -------
    morphed_ndvar : NDVar
        NDVar morphed to the destination subject.

    Notes
    -----
    This function is used to make sure a number of different NDVars are defined
    on the same MRI subject and handles scaled MRIs efficiently. If the MRI
    subject on which ``ndvar`` is defined is a scaled copy of ``subject_to``,
    by default a shallow copy of ``ndvar`` is returned. That means that it is
    not safe to assume that ``morphed_ndvar`` can be modified in place without
    altering ``ndvar``. To make sure the date of the output is independent from
    the data of the input, set the argument ``copy=True``.
    """
    source = ndvar.get_dim('source')
    subjects_dir = source.subjects_dir
    subject_from = source.subject
    src = source.src
    if isinstance(source, VolumeSourceSpace):
        source_to = VolumeSourceSpace.from_file(subjects_dir, subject_to, src, None)
        assert np.all(source_to.vertices[0] == source.vertices[0])
        dims = list(ndvar.dims)
        dims[ndvar.get_axis('source')] = source_to
        return NDVar(ndvar.x, dims, ndvar.info.copy(), ndvar.name)
    has_lh_out = bool(source.rh_n if xhemi else source.lh_n)
    has_rh_out = bool(source.lh_n if xhemi else source.rh_n)
    assert_subject_exists(subject_to, subjects_dir)
    if vertices_to in (None, 'lh', 'rh'):
        default_vertices = source_space_vertices(source.kind, source.grade, subject_to, subjects_dir)
        lh_out = vertices_to == 'lh' or (vertices_to is None and has_lh_out)
        rh_out = vertices_to == 'rh' or (vertices_to is None and has_rh_out)
        vertices_to = [default_vertices[0] if lh_out else np.empty(0, int),
                       default_vertices[1] if rh_out else np.empty(0, int)]
        if mask is None:
            if source.parc is None:
                mask = False
            else:
                mask = not np.any(source.parc.startswith('unknown-'))
    elif not isinstance(vertices_to, list) or not len(vertices_to) == 2:
        raise ValueError('vertices_to must be a list of length 2, got %r' %
                         (vertices_to,))

    # check that requested data is available
    n_to_lh = len(vertices_to[0])
    n_to_rh = len(vertices_to[1])
    if n_to_lh and not has_lh_out:
        raise ValueError("Data on the left hemisphere was requested in "
                         "vertices_to but is not available in ndvar")
    elif n_to_rh and not has_rh_out:
        raise ValueError("Data on the right hemisphere was requested in "
                         "vertices_to but is not available in ndvar")
    elif n_to_lh == 0 and n_to_rh == 0:
        raise ValueError("No target vertices")

    # parc for new source space
    if parc is True:
        parc_to = None if source.parc is None else source.parc.name
    else:
        parc_to = parc
    if mask and parc_to is None:
        raise ValueError("Can't mask source space without parcellation...")
    # check that annot files are available
    if parc_to:
        fname = SourceSpace._ANNOT_PATH.format(
            subjects_dir=subjects_dir, subject=subject_to, hemi='%s',
            parc=parc_to)
        fnames = tuple(fname % hemi for hemi in ('lh', 'rh'))
        missing = tuple(fname for fname in fnames if not os.path.exists(fname))
        if missing:
            raise IOError(
                "Annotation files are missing for parc=%r for target subject "
                "%s. Use the parc parameter to change the parcellation. The "
                "following files are missing:\n%s" %
                (parc_to, subject_to, '\n'.join(missing)))
    # catch in == out
    if subject_from == subject_to and _vertices_equal(source.vertices, vertices_to):
        if copy:
            ndvar = ndvar.copy()
        if parc is not True:
            set_parc(ndvar, parc)
        return ndvar
    # find target source space
    source_to = SourceSpace(vertices_to, subject_to, src, subjects_dir, parc_to)
    if mask is True:
        index = np.invert(source_to.parc.startswith('unknown-'))
        source_to = source_to[index]
    elif mask not in (None, False):
        raise TypeError("mask=%r" % (mask,))

    axis = ndvar.get_axis('source')
    x = ndvar.x

    # check whether it is a scaled brain
    do_morph = True
    if not xhemi:
        cfg_path = os.path.join(subjects_dir, subject_from,
                                'MRI scaling parameters.cfg')
        if os.path.exists(cfg_path):
            cfg = mne.coreg.read_mri_cfg(subject_from, subjects_dir)
            subject_from = cfg['subject_from']
            if (subject_to == subject_from and
                    _vertices_equal(source_to.vertices, source.vertices)):
                if copy:
                    x_ = x.copy()
                else:
                    x_ = x
                do_morph = False

    if do_morph:
        if morph_mat is None:
            with warnings.catch_warnings():
                warnings.filterwarnings('ignore', '\d+/\d+ vertices not included in smoothing', module='mne')
                morph_mat = mne.compute_morph_matrix(
                    subject_from, subject_to, source.vertices, source_to.vertices,
                    None, subjects_dir, xhemi=xhemi)
        elif not sp.sparse.issparse(morph_mat):
            raise ValueError('morph_mat must be a sparse matrix')
        elif not sum(len(v) for v in source_to.vertices) == morph_mat.shape[0]:
            raise ValueError('morph_mat.shape[0] must match number of '
                             'vertices in vertices_to')

        # flatten data
        if axis != 0:
            x = x.swapaxes(0, axis)
        n_sources = len(x)
        if not n_sources == morph_mat.shape[1]:
            raise ValueError('ndvar source dimension length must be the same '
                             'as morph_mat.shape[0]')
        if ndvar.ndim > 2:
            shape = x.shape
            x = x.reshape((n_sources, -1))

        # apply morph matrix
        x_ = morph_mat * x

        # restore data shape
        if ndvar.ndim > 2:
            shape_ = (len(x_),) + shape[1:]
            x_ = x_.reshape(shape_)
        if axis != 0:
            x_ = x_.swapaxes(axis, 0)

    # package output NDVar
    dims = ndvar.dims[:axis] + (source_to,) + ndvar.dims[axis + 1:]
    info = ndvar.info.copy()
    return NDVar(x_, dims, info, ndvar.name)


# label operations ---

def dissolve_label(labels, source, targets, subjects_dir=None,
                   hemi='both'):
    """
    Assign every point from source to the target that is closest to it.

    Parameters
    ----------
    labels : list
        List of labels (as returned by mne.read_annot).
    source : str
        Name of the source label (without hemi affix).
    targets : list of str
        List of target label names (without hemi affix).
    subjects_dir : str
        subjects_dir.
    hemi : 'both', 'lh', 'rh'
        Hemisphere(s) for which to dissolve the label.

    Notes
    -----
    Modifies ``labels`` in-place, returns None.
    """
    subjects_dir = get_subjects_dir(subjects_dir)
    subject = labels[0].subject
    if hemi == 'both':
        hemis = ('lh', 'rh')
    elif hemi == 'lh' or hemi == 'rh':
        hemis = (hemi,)
    else:
        raise ValueError("hemi=%r" % hemi)

    idx = {l.name: i for i, l in enumerate(labels)}

    rm = set()
    for hemi in hemis:
        fpath = os.path.join(subjects_dir, subject, 'surf', hemi + '.inflated')
        points, _ = mne.read_surface(fpath)

        src_name = '-'.join((source, hemi))
        src_idx = idx[src_name]
        rm.add(src_idx)
        src_label = labels[src_idx]
        tgt_names = ['-'.join((name, hemi)) for name in targets]
        tgt_idxs = [idx[name] for name in tgt_names]
        tgt_labels = [labels[i] for i in tgt_idxs]
        tgt_points = [points[label.vertices] for label in tgt_labels]

        vert_by_tgt = {i: [] for i in xrange(len(targets))}
        for src_vert in src_label.vertices:
            point = points[src_vert:src_vert + 1]
            dist = [cdist(point, pts).min() for pts in tgt_points]
            tgt = np.argmin(dist)
            vert_by_tgt[tgt].append(src_vert)

        for i, label in enumerate(tgt_labels):
            new_vertices = vert_by_tgt[i]
            label.vertices = np.union1d(label.vertices, new_vertices)

    for i in sorted(rm, reverse=True):
        del labels[i]


def rename_label(labels, old, new):
    """Rename a label in a parcellation

    Parameters
    ----------
    labels : list of Label
        The labels
    old, new : str
        Old and new names without hemi affix.

    Notes
    -----
    Modifies ``labels`` in-place, returns None.
    """
    delim = '-'
    for hemi in ('lh', 'rh'):
        old_ = delim.join((old, hemi))
        for label in labels:
            if label.name == old_:
                new_ = delim.join((new, hemi))
                label.name = new_


def combination_label(name, exp, labels, subjects_dir):
    """Create a label based on combination of existing labels

    Parameters
    ----------
    name : str
        Name for the new label (without -hemi tag to create labels for both
        hemispheres).
    exp : str
        Boolean expression containing label names, + and - (all without -hemi
        tags).
    labels : dict
        {name: label} dictionary.
    subjects_dir : str
        Freesurfer SUBJECTS_DIR (used for splitting labels).

    Returns
    -------
    labels : list
        List of labels, one or two depending on what hemispheres are included.
    """
    m = re.match("([\w.]+)-([lr]h)", name)
    if m:
        name = m.group(1)
        hemis = (m.group(2),)
    else:
        hemis = ('lh', 'rh')

    # splitting labels function
    def split(label, parts=2):
        return mne.split_label(label, parts, subjects_dir=subjects_dir)

    # execute recombination
    out = []
    env = {'split': split}
    for hemi in hemis:
        local_env = {k[:-3].replace('.', '_'): v for k, v in labels.iteritems()
                     if k.endswith(hemi)}
        try:
            label = eval(exp.replace('.', '_'), env, local_env)
        except Exception as exc:
            raise ValueError("Invalid label expression: %r\n%s" % (exp, exc))
        label.name = '%s-%s' % (name, hemi)
        out.append(label)

    return out


def xhemi(ndvar, mask=None, hemi='lh', parc=True):
    """Project data from both hemispheres to ``hemi`` of fsaverage_sym

    Project data from both hemispheres to the same hemisphere for
    interhemisphere comparisons. The fsaverage_sym brain is a symmetric
    version of fsaverage to facilitate interhemisphere comparisons. It is
    included with FreeSurfer > 5.1 and can be obtained as described `here
    <http://surfer.nmr.mgh.harvard.edu/fswiki/Xhemi>`_. For statistical
    comparisons between hemispheres, use of the symmetric ``fsaverage_sym``
    model is recommended to minimize bias [1]_.

    Parameters
    ----------
    ndvar : NDVar
        NDVar with SourceSpace dimension.
    mask : bool
        Remove sources in "unknown-" labels (default is True unless ``ndvar``
        contains sources with "unknown-" label).
    hemi : 'lh' | 'rh'
        Hemisphere onto which to morph the data.
    parc : bool | str
        Parcellation for target source space; True to use same as in ``ndvar``
        (default).

    Returns
    -------
    lh : NDVAr
        Data from the left hemisphere on ``hemi`` of ``fsaverage_sym``.
    rh : NDVar
        Data from the right hemisphere on ``hemi`` of ``fsaverage_sym``.

    References
    ----------
    .. [1] Greve D. N., Van der Haegen L., Cai Q., Stufflebeam S., Sabuncu M.
           R., Fischl B., Brysbaert M.
           A Surface-based Analysis of Language Lateralization and Cortical
           Asymmetry. Journal of Cognitive Neuroscience 25(9), 1477-1492, 2013.
    """
    other_hemi = 'rh' if hemi == 'lh' else 'lh'

    if ndvar.source.subject == 'fsaverage_sym':
        ndvar_sym = ndvar
    else:
        ndvar_sym = morph_source_space(ndvar, 'fsaverage_sym', parc=parc, mask=mask)

    vert_lh, vert_rh = ndvar_sym.source.vertices
    vert_from = [[], vert_rh] if hemi == 'lh' else [vert_lh, []]
    vert_to = [vert_lh, []] if hemi == 'lh' else [[], vert_rh]
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', '\d+/\d+ vertices not included in smoothing', module='mne')
        morph_mat = mne.compute_morph_matrix(
            'fsaverage_sym', 'fsaverage_sym', vert_from, vert_to,
            subjects_dir=ndvar.source.subjects_dir, xhemi=True)

    out_same = ndvar_sym.sub(source=hemi)
    out_other = morph_source_space(
        ndvar_sym.sub(source=other_hemi), 'fsaverage_sym',
        out_same.source.vertices, morph_mat, parc=parc, xhemi=True, mask=mask)

    if hemi == 'lh':
        return out_same, out_other
    else:
        return out_other, out_same
