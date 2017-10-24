# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
"""Test MneExperiment using mne-python sample data"""
import imp
from os.path import join, realpath

from nose.tools import eq_, assert_raises

from eelbrain import *
from eelbrain._exceptions import DefinitionError

from ..._utils.testing import (
    TempDir, assert_dataobj_equal, requires_mne_sample_data)


@requires_mne_sample_data
def test_sample():
    set_log_level('warning', 'mne')

    # import from file:  http://stackoverflow.com/a/67692/166700
    e_path = realpath(join(__file__, '..', '..', '..', '..', 'examples',
                           'experiment', 'sample_experiment.py'))
    e_module = imp.load_source('sample_experiment', e_path)

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 3, 2)

    root = join(tempdir, 'SampleExperiment')
    e = e_module.SampleExperiment(root)

    eq_(e.get('subject'), 'R0000')
    eq_(e.get('subject', subject='R0002'), 'R0002')

    # evoked cache invalidated by change in bads
    e.set('R0001', rej='')
    ds = e.load_evoked()
    eq_(ds[0, 'evoked'].info['bads'], [])
    e.make_bad_channels(['MEG 0331'])
    ds = e.load_evoked()
    eq_(ds[0, 'evoked'].info['bads'], ['MEG 0331'])

    e.set(rej='man')
    sds = []
    for _ in e:
        e.make_rej(auto=2.5e-12)
        sds.append(e.load_evoked())

    ds = e.load_evoked('all')
    assert_dataobj_equal(combine(sds), ds)

    # e._report_subject_info() broke with non-alphabetic subject order
    subjects = e.get_field_values('subject')
    ds = Dataset()
    ds['subject'] = Factor(reversed(subjects))
    ds['n'] = Var(range(3))
    s_table = e._report_subject_info(ds, '')

    # duplicate subject
    class BadExperiment(e_module.SampleExperiment):
        groups = {'group': ('R0001', 'R0002', 'R0002')}
    assert_raises(DefinitionError, BadExperiment, root)

    # non-existing subject
    class BadExperiment(e_module.SampleExperiment):
        groups = {'group': ('R0001', 'R0003', 'R0002')}
    assert_raises(DefinitionError, BadExperiment, root)

    # unsorted subjects
    class Experiment(e_module.SampleExperiment):
        groups = {'group': ('R0002', 'R0000', 'R0001')}
    e = Experiment(root)
    eq_([s for s in e], ['R0000', 'R0001', 'R0002'])


@requires_mne_sample_data
def test_samples_sesssions():
    set_log_level('warning', 'mne')

    e_path = realpath(join(__file__, '..', '..', '..', '..', 'examples',
                           'experiment', 'sample_experiment_sessions.py'))
    e_module = imp.load_source('sample_experiment_sessions', e_path)

    tempdir = TempDir()
    datasets.setup_samples_experiment(tempdir, 2, 1, 2)

    root = join(tempdir, 'SampleExperiment')
    e = e_module.SampleExperiment(root)
    # bad channels
    e.make_bad_channels('0111')
    eq_(e.load_bad_channels(), ['MEG 0111'])
    eq_(e.load_bad_channels(session='sample2'), [])
    e.show_bad_channels()
    e.merge_bad_channels()
    eq_(e.load_bad_channels(session='sample2'), ['MEG 0111'])
    e.show_bad_channels()

    # rejection
    for _ in e:
        for epoch in ('target1', 'target2'):
            e.set(epoch=epoch)
            e.make_rej(auto=2e-12)

    ds = e.load_evoked('R0000', epoch='target2')
    e.set(session='sample1')
    ds2 = e.load_evoked('R0000')
    assert_dataobj_equal(ds2, ds)

    # super-epoch
    ds1 = e.load_epochs(epoch='target1')
    ds2 = e.load_epochs(epoch='target2')
    ds_super = e.load_epochs(epoch='super')
    assert_dataobj_equal(ds_super['meg'], combine((ds1['meg'], ds2['meg'])))
