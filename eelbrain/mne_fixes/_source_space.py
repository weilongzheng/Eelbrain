from mne import SourceSpaces
import numpy as np


def merge_volume_source_space(sss, name='Volume'):
    """Merge multiple volume source spaces into one

    Notes
    -----
    Only intended to work properly if ``sss`` is the output of a single
    :func:`mne.setup_volume_source_space` call.
    """
    if len(sss) <= 1:
        return sss
    assert all(ss['type'] == 'vol' for ss in sss)
    ss = sss[0]
    ss['vertno'] = np.unique(np.concatenate([ss['vertno'] for ss in sss]))
    ss['inuse'] = reduce(np.logical_or, (ss['inuse'] for ss in sss))
    ss['nuse'] = ss['inuse'].sum()
    ss['seg_name'] = str(name)
    del ss['neighbor_vert']
    return SourceSpaces([ss], sss.info)
