from nose.tools import eq_

from eelbrain import datasets, plot


def test_linestack():
    "Test plot.LineStack()"
    ds = datasets.get_uts()
    p = plot.LineStack(ds[:10, 'uts'], show=False)
    ax = p.figure.axes[0]
    eq_(ax.get_xlim(), (-0.2, .79))
    p.set_xlim(.1, .4)
    eq_(ax.get_xlim(), (.1, .4))
    p.close()
