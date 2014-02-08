'''
Created on Jul 22, 2013

@author: christian
'''
from eelbrain.data import datasets, plot


def test_plot_topomap():
    "Test plot.Topomap"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.Topomap('utsnd.aggregate(time=(0.075, 0.125))', ds=ds)
    p.close()
    p = plot.Topomap('utsnd.aggregate(time=(0.075, 0.125))', 'A%B', ds=ds)
    p.close()

def test_plot_butterfly():
    "Test plot.TopoButterfly"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.TopoButterfly('utsnd', ds=ds)
    p.close()
    p = plot.TopoButterfly('utsnd', 'A%B', ds=ds)
    p.close()

def test_plot_array():
    "Test plot.TopoArray"
    plot.configure_backend(False, False)
    ds = datasets.get_rand(utsnd=True)
    p = plot.TopoArray('utsnd', ds=ds)
    p.close()
    p = plot.TopoArray('utsnd', 'A%B', ds=ds)
    p.close()
