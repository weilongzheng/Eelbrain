# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
from os.path import join

import mne
from nose.tools import eq_, ok_, assert_false
from numpy.testing import assert_array_equal

from eelbrain import gui, load, set_log_level
from eelbrain._utils.testing import TempDir, gui_test
from eelbrain._wxgui.select_epochs import Document, Model


@gui_test
def test_select_epochs():
    "Test Select-Epochs GUI Document"
    set_log_level('warning', 'mne')

    data_path = mne.datasets.testing.data_path()
    raw_path = join(data_path, 'MEG', 'sample', 'sample_audvis_trunc_raw.fif')
    raw = mne.io.Raw(raw_path, preload=True).pick_types('mag', stim=True)
    ds = load.fiff.events(raw)
    ds['meg'] = load.fiff.epochs(ds, tmax=0.1)

    tempdir = TempDir()
    path = join(tempdir, 'rej.pickled')

    # Test Document
    # =============
    # create a file
    doc = Document(ds, 'meg')
    doc.set_path(path)
    doc.set_case(1, False, 'tag', None)
    doc.set_case(2, None, None, ['2'])
    doc.set_bad_channels([1])
    # check modifications
    eq_(doc.accept[1], False)
    eq_(doc.tag[1], 'tag')
    eq_(doc.interpolate[1], [])
    eq_(doc.interpolate[2], ['2'])
    eq_(doc.bad_channels, [1])
    assert_array_equal(doc.accept[2:], True)
    # save
    doc.save()

    # check the file
    ds_ = load.unpickle(path)
    eq_(doc.epochs.sensor._array_index(ds_.info['bad_channels']), [1])

    # load the file
    doc = Document(ds, 'meg', path=path)
    # modification checks
    eq_(doc.accept[1], False)
    eq_(doc.tag[1], 'tag')
    eq_(doc.interpolate[1], [])
    eq_(doc.interpolate[2], ['2'])
    eq_(doc.bad_channels, [1])
    assert_array_equal(doc.accept[2:], True)

    # Test Model
    # ==========
    doc = Document(ds, 'meg')
    model = Model(doc)

    # accept
    model.set_case(0, False, None, None)
    eq_(doc.accept[0], False)
    model.history.undo()
    eq_(doc.accept[0], True)
    model.history.redo()
    eq_(doc.accept[0], False)

    # interpolate
    model.toggle_interpolation(2, '3')
    eq_(doc.interpolate[2], ['3'])
    model.toggle_interpolation(2, '4')
    eq_(doc.interpolate[2], ['3', '4'])
    model.toggle_interpolation(2, '3')
    eq_(doc.interpolate[2], ['4'])
    model.toggle_interpolation(3, '3')
    eq_(doc.interpolate[2], ['4'])
    eq_(doc.interpolate[3], ['3'])
    model.history.undo()
    model.history.undo()
    eq_(doc.interpolate[2], ['3', '4'])
    eq_(doc.interpolate[3], [])
    model.history.redo()
    eq_(doc.interpolate[2], ['4'])

    # bad channels
    model.set_bad_channels([1])
    model.set_bad_channels([1, 10])
    eq_(doc.bad_channels, [1, 10])
    model.history.undo()
    eq_(doc.bad_channels, [1])
    model.history.redo()
    eq_(doc.bad_channels, [1, 10])

    # reload to reset
    model.load(path)
    # tests
    eq_(doc.accept[1], False)
    eq_(doc.tag[1], 'tag')
    eq_(doc.interpolate[1], [])
    eq_(doc.interpolate[2], ['2'])
    eq_(doc.bad_channels, [1])
    assert_array_equal(doc.accept[2:], True)

    # Test GUI
    # ========
    frame = gui.select_epochs(ds, nplots=9)
    assert_false(frame.CanBackward())
    ok_(frame.CanForward())
    frame.OnForward(None)
    frame.SetVLim(1e-12)
