# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
import os

from nose.tools import eq_, ok_, assert_raises
import numpy as np
from numpy.testing import assert_array_equal

from eelbrain import Dataset, Factor, Var, MneExperiment
from ..._utils.testing import assert_dataobj_equal, TempDir


SUBJECT = 'CheeseMonger'
SUBJECTS = ['R%04i' % i for i in (1, 11, 111, 1111)]
SAMPLINGRATE = 1000.
TRIGGERS = np.tile(np.arange(1, 5), 2)
I_START = np.arange(1001, 1441, 55)


class BaseExperiment(MneExperiment):

    path_version = 1

    sessions = 'file'


class EventExperiment(MneExperiment):

    path_version = 1

    trigger_shift = 0.03

    sessions = 'cheese'

    variables = {'kind': {(1, 2, 3, 4): 'cheese', (11, 12, 13, 14): 'pet'},
                 'name': {1: 'Leicester', 2: 'Tilsit', 3: 'Caerphilly',
                          4: 'Bel Paese'},
                 'backorder': {(1, 4): 'no', (2, 3): 'yes'},
                 'taste': {(1, 2): 'good', 'default': 'bad'}}

    epochs = {'cheese': {'sel': "kind == 'cheese'",
                         'tmin': -0.2},
              'cheese-leicester': {'base': 'cheese',
                                   'tmin': -0.1,
                                   'sel': "name == 'Leicester'"},
              'cheese-tilsit': {'base': 'cheese',
                                'sel': "name == 'Tilsit"}}

    defaults = {'model': 'name'}


class EventExperimentTriggerShiftDict(EventExperiment):
    "Test trigger shift as dictionary"
    trigger_shift = {SUBJECT: 0.04}


def gen_triggers():
    raw = Var([], info={'sfreq': SAMPLINGRATE})
    ds = Dataset(info={'subject': SUBJECT, 'raw': raw, 'sfreq': SAMPLINGRATE})
    ds['trigger'] = Var(TRIGGERS)
    ds['i_start'] = Var(I_START)
    return ds


def test_mne_experiment_templates():
    "Test MneExperiment template formatting"
    tempdir = TempDir()
    e = BaseExperiment(tempdir, False)

    # Don't create dirs without root
    ok_(e.get('raw-file', mkdir=True).endswith('-raw.fif'))

    # model
    eq_(e.get('model', model='a % b'), 'a%b')
    eq_(e.get('model', model='b % a'), 'a%b')
    assert_raises(ValueError, e.set, model='a*b')
    assert_raises(ValueError, e.set, model='log(a)')

    # compounds
    eq_(e.get('src_kind'), '0-40 bestreg free-3-dSPM')
    e.set_inv('fixed')
    eq_(e.get('src_kind'), '0-40 bestreg fixed-3-dSPM')
    e.set(cov='noreg')
    eq_(e.get('src_kind'), '0-40 noreg fixed-3-dSPM')
    e.set(raw='1-40')
    eq_(e.get('src_kind'), '1-40 noreg fixed-3-dSPM')
    e.set(src='ico-5')
    eq_(e.get('src_kind'), '1-40 noreg ico-5 fixed-3-dSPM')
    e.set(src='ico-4')
    eq_(e.get('src_kind'), '1-40 noreg fixed-3-dSPM')

    # find terminal field names
    eq_(e.find_keys('raw-file'), {'root', 'subject', 'session'})
    eq_(e.find_keys('evoked-file', False),
        {'subject', 'session', 'modality', 'raw', 'epoch', 'rej',
         'equalize_evoked_count', 'model', })

    # inv
    SNR1 = 1.
    SNR2 = 1. / 2**2
    SNR3 = 1. / 3**2

    def set_inv(inv, make_kw, apply_kw, *args):
        e.reset()
        e.set(inv=inv)
        eq_(e._params['make_inv_kw'], make_kw)
        eq_(e._params['apply_inv_kw'], apply_kw)
        e.reset()
        e.set_inv(*args)
        eq_(e.get('inv'), inv)
        eq_(e._params['make_inv_kw'], make_kw)
        eq_(e._params['apply_inv_kw'], apply_kw)

    yield (set_inv, 'free-3-MNE',
           {'loose': 1, 'depth': 0.8},
           {'method': 'MNE', 'lambda2': SNR3},
           'free', 3, 'MNE', .8)
    yield (set_inv, 'free-3-dSPM-0.2-pick_normal',
           {'loose': 1, 'depth': 0.2},
           {'method': 'dSPM', 'lambda2': SNR3, 'pick_normal': True},
           'free', 3, 'dSPM', .2, True)
    yield (set_inv, 'fixed-2-MNE-0.2',
           {'fixed': True, 'loose': None, 'depth': 0.2},
           {'method': 'MNE', 'lambda2': SNR2},
           'fixed', 2, 'MNE', .2)
    yield (set_inv, 'fixed-2-MNE-pick_normal',
           {'fixed': True, 'loose': None, 'depth': 0.8},
           {'method': 'MNE', 'lambda2': SNR2, 'pick_normal': True},
           'fixed', 2, 'MNE', None, True)
    yield (set_inv, 'loose.5-3-sLORETA',
           {'loose': 0.5, 'depth': 0.8},
           {'method': 'sLORETA', 'lambda2': SNR3},
           0.5, 3, 'sLORETA')
    yield (set_inv, 'fixed-1-MNE-0',
           {'fixed': True, 'loose': None, 'depth': None},
           {'method': 'MNE', 'lambda2': SNR1},
           'fixed', 1, 'MNE', 0)

    assert_raises(ValueError, e.set_inv, 'free', -3, 'dSPM')
    assert_raises(ValueError, e.set, inv='free-3-mne')
    assert_raises(ValueError, e.set, inv='free-3-MNE-2')


def test_test_experiment():
    "Test event labeling with the EventExperiment subclass of MneExperiment"
    e = EventExperiment()

    # test defaults
    eq_(e.get('session'), 'cheese')
    eq_(e.get('model'), 'name')

    # test event labeling
    ds = e.label_events(gen_triggers())
    name = Factor([e.variables['name'][t] for t in TRIGGERS], name='name')
    assert_dataobj_equal(ds['name'], name)
    tgt = ds['trigger'].as_factor(e.variables['backorder'], 'backorder')
    assert_dataobj_equal(ds['backorder'], tgt)
    tgt = ds['trigger'].as_factor(e.variables['taste'], 'taste')
    assert_dataobj_equal(ds['taste'], tgt)
    assert_array_equal(ds['i_start'], I_START)
    assert_array_equal(ds['subject'] == SUBJECT, True)

    # tests disabled (trigger-shift applied in load_events):
    # ---
    #  assert_equal(ds['i_start'], I_START + round(0.03 * SAMPLINGRATE))
    # # test without trigger shift
    # e.trigger_shift = 0
    # ds = e.label_events(gen_triggers())
    # assert_equal(ds['i_start'], I_START)
    # # trigger shift dict
    # e2 = EventExperimentTriggerShiftDict('', False)
    # ds = e2.label_events(gen_triggers())
    # assert_equal(ds['i_start'], I_START + round(0.04 * SAMPLINGRATE))

    # epochs
    eq_(e._epochs['cheese'].tmin, -0.2)
    eq_(e._epochs['cheese-leicester'].tmin, -0.1)
    eq_(e._epochs['cheese-tilsit'].tmin, -0.2)


class FileExperiment(MneExperiment):

    path_version = 1

    auto_delete_cache = 'disable'

    groups = {'gsub': SUBJECTS[1:],
              'gexc': {'exclude': SUBJECTS[0]},
              'gexc2': {'base': 'gexc', 'exclude': SUBJECTS[-1]}}

    sessions = 'file'


class FileExperimentDefaults(FileExperiment):

    defaults = {'session': 'file',
                'group': 'gsub'}


def test_file_handling():
    "Test MneExperiment with actual files"
    tempdir = TempDir()
    for subject in SUBJECTS:
        sdir = os.path.join(tempdir, 'meg', subject)
        os.makedirs(sdir)

    e = FileExperiment(tempdir)

    eq_(e.get('subject'), SUBJECTS[0])
    eq_([s for s in e.iter(group='all')], SUBJECTS)
    eq_([s for s in e.iter(group='gsub')], SUBJECTS[1:])
    eq_([s for s in e.iter(group='gexc')], SUBJECTS[1:])
    eq_([s for s in e.iter(group='gexc2')], SUBJECTS[1:-1])
    eq_(e.get('subject'), SUBJECTS[1])
    eq_(e.get('subject', group='all'), SUBJECTS[1])
    e.set(SUBJECTS[0])
    eq_(e.get('subject'), SUBJECTS[0])
    eq_(e.get('subject', group='gsub'), SUBJECTS[1])

    e = FileExperimentDefaults(tempdir)
    eq_(e.get('group'), 'gsub')
    eq_(e.get('subject'), SUBJECTS[1])
