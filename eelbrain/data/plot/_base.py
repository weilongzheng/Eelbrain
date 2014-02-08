"""
Matplotlib and Backends
=======================

Some Eelbrain figures come with a toolbar with additional functionality. This
toolbar requires a running wxPython application. This is automatically
available if the plot is created from within a wxPython application such as
PyShell or Eelbrain's WxTerm. In IPython, the wx gui can be made available by
invoking it with::

    $ ipython --gui=wx

In order to use a different matplotlib backend (or the wx backend with the
matplotlib toolbar), a call to::

    >>> plot.configure_backend(False)

determines the frame used for any subsequently created plots.


Implementation
==============

Plotting is implemented hierarchically in 3 different types of
functions/classes:

top-level (public names)
    Top-level functions or classes have public names create an entire figure.
    Some classes also retain the figure and provide methods for manipulating
    it.

_ax_
    Functions beginning with _ax_ organize an axes object. They do not
    create their own axes object (this is provided by the top-level function),
    but change axes formatting such as labels and extent.

_plt_
    Functions beginning with _plt_ only plot data to a given axes object
    without explicitly changing aspects of the axes themselves.


Top-level plotters can be called with nested lists of data-objects (NDVar
instances). They create a separate axes for each list element. Axes
themselves can have multiple layers (e.g., a difference map visualized through
a colormap, and significance levels indicated by contours).


Example: t-test
---------------

For example, the default plot for testnd.ttest() results is the
following list (assuming the test compares A and B):

``[A, B, [diff(A,B), p(A, B)]]``

where ``diff(...)`` is a difference map and ``p(...)`` is a map of p-values.
The main plot function creates a separate axes object for each list element:

- ``A``
- ``B``
- ``[diff(A,B), p(A, B)]``

Each of these element is then plotted with the corresponding _ax_ function.
The _ax_ function calls _plt_ for each of its input elements. Thus, the
functions executed are:

#. plot([A, B, [diff(A,B), p(A, B)]])
#. ---> _ax_(A)
#. ------> _plt_(A)
#. ---> _ax_(B)
#. ------> _plt_(B)
#. ---> _ax_([diff(A,B), p(A, B)])
#. ------> _plt_(diff(A,B))
#. ------> _plt_(p(A, B))

"""
from __future__ import division

from itertools import chain
import math
import os
import shutil
import subprocess
import tempfile

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import PIL

from ...utils import LazyProperty
from ...utils.subp import cmd_exists
from ...fmtxt import texify
from ..colorspaces import symmetric_cmaps, zerobased_cmaps
from ..data_obj import ascategorial, asndvar, DimensionMismatchError


# defaults
defaults = {'DPI': 72, 'maxw': 16, 'maxh': 10}
backend = {'block': False,  # plt.show() parameter for mpl_figure
           'frame': 'mpl'}
if mpl.get_backend().lower().startswith('wx'):
    try:
        from ...wxutils.mpl_canvas import CanvasFrame
        backend['frame'] = 'wx'
    except:
        pass

# store callback figures (they need to be preserved)
figs = []

# constants
default_cmap = None
default_meas = '?'


def configure_backend(frame=True, block=False):
    """Set basic configuration parameters

    Parameters
    ----------
    frame : bool
        Use a custom frame with a different toolbar from matplotlib.
    block : bool
        pyplot.show() parameters: Block the interpreter when showing a figure.
        This parameter only applies when using the Matplotlib wx-backend and
        frame.
    """
    backend['frame'] = 'wx' if frame else 'mpl'
    backend['block'] = block


_unit = {'time': 'ms'}
_conversion = {'time': {'ms': 1e3}}
_fmt = {'time': '%i'}


def _convert(x, dimname):
    """Convert known units from SI for plotting

    Parameters
    ----------
    x : scalar | array_like
        Value in SI.
    dimname : str
        Name of the dimension (e.g., "time")
    """
    unit = _unit[dimname]
    u = _conversion[dimname][unit]
    return x * u


def _ticklabel(tick, dimname, unit=False):
    "Convert value to string if default units exist"
    fmt = _fmt.get(dimname, None)
    label = _convert(tick, dimname)
    if fmt:
        label = fmt % label
    if unit is True:
        unit = _unit[dimname]
    if unit:
        label = ' '.join((label, unit))
    return label


def _ticklabels(ticks, dimname):
    "Convert values to strings if default units exist"
    fmt = _fmt.get(dimname, None)
    ticklabels = _convert(ticks, dimname)
    if fmt:
        ticklabels = [fmt % lbl for lbl in ticklabels]
    return ticklabels


def _axlabel(dimname, label=True):
    """Find an axis label

    Parameters
    ----------
    dimname : str
        Name of the dimension.
    label : None | True | str
        Label argument.

    Returns
    -------
    label : str | None
        Returns the default axis label if label==True, otherwise the label
        argument.
    """
    if label is True:
        unit = _unit[dimname]
        name = dimname.capitalize()
        label = "%s [%s]" % (name, unit)
    return label


def _x_axis(ndvar, dimname, label=True):
    dim = ndvar.get_dim(dimname)
    values = _convert(dim.x, dimname)
    label = _axlabel(dimname, label)
    return values, label


def find_ct_args(ndvar, overlay, contours={}):
    """Construct a dict with kwargs for a contour plot

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.
    overlay : bool
        Whether the NDVar is plotted as a first layer or as an overlay.
    contours : dict
        Externally specified contours as {meas: {level: color}} mapping.

    Returns
    -------
    ct_args : dict
        {level: color} mapping for contour plots.

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if overlay:
        kind = ndvar.info.get('overlay', ('contours',))
    else:
        kind = ndvar.info.get('base', ())

    ct_args = {}
    if 'contours' in kind:
        info_ct = ndvar.info.get('contours', None)
        if overlay:
            info_ct = ndvar.info.get('overlay_contours', info_ct)
        else:
            info_ct = ndvar.info.get('base_contours', info_ct)

        if info_ct:
            ct_args.update(info_ct)

    meas = ndvar.info.get('meas', default_meas)
    if meas in contours:
        ct_args.update(contours[meas])

    return ct_args


def find_im_args(ndvar, overlay, vlims={}, cmaps={}):
    """Construct a dict with kwargs for an im plot

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.
    overlay : bool
        Whether the NDVar is plotted as a first layer or as an overlay.
    vlims : dict
        {(meas, cmap): (vmax, vmin)} mapping to replace v-limits based on the
        ndvar.info dict.
    cmaps : dict
        {meas: cmap} mapping to replace the cmap in the ndvar.info dict.

    Returns
    -------
    im_args : dict
        Arguments for the im plot (cmap, vmin, vmax).

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if overlay:
        kind = ndvar.info.get('overlay', ('contours',))
    else:
        kind = ndvar.info.get('base', ('im',))

    if 'im' in kind:
        if 'meas' in ndvar.info:
            meas = ndvar.info['meas']
        else:
            meas = default_meas

        if meas in cmaps:
            cmap = cmaps[meas]
        elif 'cmap' in ndvar.info:
            cmap = ndvar.info['cmap']
        else:
            cmap = default_cmap

        key = (meas, cmap)
        if key in vlims:
            vmin, vmax = vlims[key]
        else:
            vmin, vmax = find_vlim_args(ndvar)
            vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cmap)
        im_args = dict(cmap=cmap, vmin=vmin, vmax=vmax)
    else:
        im_args = None

    return im_args


def find_uts_args(ndvar, overlay, color=None):
    """Construct a dict with kwargs for a uts plot

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.
    overlay : bool
        Whether the NDVar is plotted as a first layer or as an overlay.
    vlims : dict
        Vmax and vmin values by (meas, cmap).

    Returns
    -------
    uts_args : dict
        Arguments for a uts plot (color).

    Notes
    -----
    The NDVar's info dict contains default arguments that determine how the
    NDVar is plotted as base and as overlay. In case of insufficient
    information, defaults apply. On the other hand, defaults can be overridden
    by providing specific arguments to plotting functions.
    """
    if overlay:
        kind = ndvar.info.get('overlay', ())
    else:
        kind = ndvar.info.get('base', ('trace',))

    if 'trace' in kind:
        args = {}
        color = color or ndvar.info.get('color', None)
        if color is not None:
            args['color'] = color
    else:
        args = None

    return args


def find_uts_hlines(ndvar):
    """Find horizontal lines for uts plots (based on contours)

    Parameters
    ----------
    ndvar : NDVar
        Data to be plotted.

    Returns
    -------
    h_lines : iterator
        Iterator over (y, kwa) tuples.
    """
    contours = ndvar.info.get('contours', None)
    if contours:
        for level in sorted(contours):
            args = contours[level]
            if isinstance(args, dict):
                yield level, args.copy()
            else:
                yield level, {'color': args}


def find_uts_ax_vlim(layers, vlims={}):
    """Find y axis limits for uts axes

    Parameters
    ----------
    layers : list of NDVar
        Data to be plotted.
    vlims : dict
        Vmax and vmin values by (meas, cmap).

    Returns
    -------
    bottom : None | scalar
        Lowest value on y axis.
    top : None | scalar
        Highest value on y axis.
    """
    bottom = None
    top = None
    overlay = False
    for ndvar in layers:
        if overlay:
            kind = ndvar.info.get('overlay', ())
        else:
            kind = ndvar.info.get('base', ('trace',))
        overlay = True

        if 'trace' not in kind:
            continue

        meas = ndvar.info.get('meas', default_meas)
        cmap = ndvar.info.get('cmap', default_cmap)
        key = (meas, cmap)
        if key in vlims:
            bottom_, top_ = vlims[key]
            if bottom is None:
                bottom = bottom_
            elif bottom_ != bottom:
                raise RuntimeError("Double vlim specification")
            if top is None:
                top = top_
            elif top_ != top:
                raise RuntimeError("Double vlim specification")

    return bottom, top


def find_fig_vlims(plots, range_by_measure=False):
    """Find vmin and vmax parameters for every (meas, cmap) combination

    Parameters
    ----------
    plots : nested list of NDVar
        Unpacked plot data.
    range_by_measure : bool
        Constrain the vmax - vmin range such that the range is constant within
        measure (for uts plots).

    Returns
    -------
    vlims : dict
        Dictionary of im limits: {(meas, cmap): (vmin, vmax)}.
    """
    vlims = {}  # (meas, cmap): (vmin, vmax)
    for ndvar in chain(*plots):
        vmin, vmax = find_vlim_args(ndvar)
        meas = ndvar.info.get('meas', '?')
        cmap = ndvar.info.get('cmap', None)
        key = (meas, cmap)
        if key in vlims:
            vmin_, vmax_ = vlims[key]
            vmin = min(vmin, vmin_)
            vmax = max(vmax, vmax_)
        vmin, vmax = fix_vlim_for_cmap(vmin, vmax, cmap)
        vlims[key] = (vmin, vmax)

    if range_by_measure:
        range_ = {}
        for (meas, cmap), (vmin, vmax) in vlims.iteritems():
            r = vmax - vmin
            range_[meas] = max(range_.get(meas, 0), r)
        for key in vlims.keys():
            meas, cmap = key
            vmin, vmax = vlims[key]
            diff = range_[meas] - (vmax - vmin)
            if diff:
                if cmap in zerobased_cmaps:
                    vmax += diff
                else:
                    diff /= 2
                    vmax += diff
                    vmin -= diff
                vlims[key] = vmin, vmax

    return vlims


def find_vlim_args(ndvar, vmin=None, vmax=None):
    if vmax is None:
        vmax = ndvar.info.get('vmax', None)
        if vmin is None:
            vmin = ndvar.info.get('vmin', None)

    if vmax is None:
        xmax = ndvar.x.max()
        xmin = ndvar.x.min()
        abs_max = max(abs(xmax), abs(xmin)) or 1e-14
        scale = math.floor(np.log10(abs_max))
        vmax = math.ceil(xmax * 10 ** -scale) * 10 ** scale
        vmin = math.floor(xmin * 10 ** -scale) * 10 ** scale

    return vmin, vmax


def fix_vlim_for_cmap(vmin, vmax, cmap):
    "Fix the vmin value to yield an appropriate range for the cmap"
    if cmap in symmetric_cmaps:
        if vmax is None and vmin is None:
            pass
        elif vmin is None:
            vmax = abs(vmax)
            vmin = -vmax
        elif vmax is None:
            vmax = abs(vmin)
            vmin = -vmax
        else:
            vmax = max(abs(vmax), abs(vmin))
            vmin = -vmax
    elif cmap in zerobased_cmaps:
        vmin = 0
    return vmin, vmax


def unpack_epochs_arg(Y, ndim, Xax=None, ds=None, levels=1):
    """Unpack the first argument to top-level NDVar plotting functions

    low level functions (``_plt_...``) work with two levels:

     - list of lists axes
     - list of layers

    Ndvar plotting functions above 1-d UTS level should support the following
    API:

     - simple NDVar: summary ``plot(meg)``
     - list of ndvars: summary for each ``plot(meg.as_list())``
     - NDVar and Xax argument: summary for each  ``plot(meg, Xax=subject)
     - nested list of layers (e.g., ttest results: [c1, c0, [c1-c0, p]])

    Parameters
    ----------
    Y : NDVar | list
        the first argument.
    ndim : int
        The number of dimensions needed for the plotting function.
    Xax : None | categorial
        A model to divide Y into different axes. Xax is currently applied on
        the first level, i.e., it assumes that Y's first dimension is cases.
    ds : None | Dataset
        Dataset containing data objects which are provided as str.
    levels : int
        Current levels of nesting (0 is the lowest level where the output is a
        list of layers).

    Returns
    -------
    data : list of list of NDVar
        The processed data to plot.
    """
    if isinstance(Y, basestring):
        Y = ds.eval(Y)

    if Xax is not None and isinstance(Y, (tuple, list)):
        err = ("Xax can only be used to divide Y into different axes if Y is "
               "a single NDVar (got a %s)." % Y.__class__.__name__)
        raise TypeError(err)

    if isinstance(Xax, str) and Xax.startswith('.'):
        dimname = Xax[1:]
        dim = Y.get_dim(dimname)
        unit = getattr(dim, 'unit', '')
        name = dimname.capitalize() + ' = %s'
        if unit:
            name += ' ' + unit
        Y = [Y.sub(name=name % v, **{dimname: v}) for v in dim.values]
    elif Xax is not None:
        Xax = ascategorial(Xax, ds=ds)
        Ys = []
        for cell in Xax.cells:
            v = Y[Xax == cell]
            v.name = cell
            Ys.append(v)
        Y = Ys
    else:
        Y = getattr(Y, '_default_plot_obj', getattr(Y, 'all', Y))
        if not isinstance(Y, (tuple, list)):
            Y = [Y]

    if levels > 0:
        return [unpack_epochs_arg(v, ndim, None, ds, levels - 1) for v in Y]
    else:
        # every value needs to be a NDVar
        out = []
        for ndvar in Y:
            ndvar = asndvar(ndvar, ds=ds)

            if ndvar.ndim == ndim + 1:
                if ndvar.has_case:
                    if len(ndvar) == 1:
                        ndvar = ndvar.aggregate(name='{name}')
                    else:
                        ndvar = ndvar.aggregate()

            if ndvar.ndim != ndim:
                err = ("Plot requires ndim=%i; %r ndim==%i" %
                       (ndim, ndvar, ndvar.ndim))
                raise DimensionMismatchError(err)

            out.append(ndvar)
        return out


def str2tex(txt):
    """If matplotlib usetex is enabled, replace tex sensitive characters in the
    string.
    """
    if txt and plt.rcParams['text.usetex']:
        return texify(txt)
    else:
        return txt


class mpl_figure:
    "cf. wxutils.mpl_canvas"
    def __init__(self, **fig_kwargs):
        "creates self.figure and self.canvas attributes and returns the figure"
        self.figure = plt.figure(**fig_kwargs)
        self.canvas = self.figure.canvas
        figs.append(self)

    def Close(self):
        plt.close(self.figure)

    def SetStatusText(self, text):
        pass

    def Show(self):
        if mpl.get_backend() == 'WXAgg':
            plt.show(block=backend['block'])

    def redraw(self, axes=[], artists=[]):
        "Adapted duplicate of mpl_canvas.FigureCanvasPanel"
        self.canvas.restore_region(self._background)
        for ax in axes:
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.canvas.blit(extent)
        for artist in artists:
            ax = artist.get_axes()
            ax.draw_artist(ax)
            extent = ax.get_window_extent()
            self.canvas.blit(extent)

    def store_canvas(self):
        self._background = self.canvas.copy_from_bbox(self.figure.bbox)


# MARK: figure composition

def _loc(name, size=(0, 0), title_space=0, frame=.01):
    """
    takes a loc argument and returns x,y of bottom left edge

    """
    if isinstance(name, basestring):
        y, x = name.split()
    # interpret x
    elif len(name) == 2:
        x, y = name
    else:
        raise NotImplementedError("loc needs to be string or len=2 tuple/list")
    if isinstance(x, basestring):
        if x == 'left':
            x = frame
        elif x in ['middle', 'center', 'centre']:
            x = .5 - size[0] / 2
        elif x == 'right':
            x = 1 - frame - size[0]
        else:
            raise ValueError(x)
    # interpret y
    if isinstance(y, basestring):
        if y in ['top', 'upper']:
            y = 1 - frame - title_space - size[1]
        elif y in ['middle', 'center', 'centre']:
            y = .5 - title_space / 2. - size[1] / 2.
        elif y in ['lower', 'bottom']:
            y = frame
        else:
            raise ValueError(y)
    return x, y


class eelfigure(object):
    """
    Parent class for eelbrain figures.

    In order to subclass:

     - find desired figure properties and then use them to initialize
       the eelfigure superclass; then use the
       :py:attr:`eelfigure.figure` and :py:attr:`eelfigure.canvas` attributes.
     - end the initialization by calling `eelfigure._show()`
     - add the :py:meth:`_fill_toolbar` method


    """
    def __init__(self, title="Eelbrain Figure", nax=None, layout_kwa={},
                 ax_aspect=2, axh_default=2, fig_kwa={}, ax_kwa={},
                 figtitle=None):
        """

        Parameters
        ----------
        title : str
            Frame title.
        nax : None | int
            Number of axes to produce layout for. If None, no layout is
            produced.
        layout_kwargs : dict
            Arguments to produce a layout (optional).
        """
        if figtitle:
            title = '%s: %s' % (title, figtitle)

        # layout
        if nax is not None:
            layout = dict(ax_aspect=ax_aspect, axh_default=axh_default)
            layout.update(layout_kwa)
            self._layout = Layout(nax, **layout)
            fig_kwa = fig_kwa.copy()
            fig_kwa.update(self._layout.fig_kwa)
        else:
            self._layout = None

        # find the right frame
        self._is_wx = False
        frame_kind = backend['frame']
        if frame_kind == 'wx':
            import wx
            app = wx.GetApp()
            if app is None:
                frame_kind = 'mpl'
            elif hasattr(app, 'shell'):
                parent = app.shell
            else:
                parent = app.GetTopWindow()
            frame = CanvasFrame(parent, title=title, eelfigure=self, **fig_kwa)
            self._is_wx = True

        if frame_kind == 'mpl':
            frame = mpl_figure(**fig_kwa)
        elif frame_kind != 'wx':
            err = "Invalid backend 'frame' parameter: %r" % frame_kind
            raise RuntimeError(err)

        figure = frame.figure
        if figtitle:
            figure.suptitle(figtitle)

        # store attributes
        self._frame = frame
        self.figure = figure
        self.canvas = frame.canvas
        self._subplots = None
        self._ax_kwa = ax_kwa

        self.canvas.mpl_connect('motion_notify_event', self._on_motion)
        self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)

    def _get_statusbar_text(self, event):
        "subclass to add figure-specific content"
        return '%s'

    @LazyProperty
    def _axes(self):
        if self._layout is None:
            return []
        else:
            ncol = self._layout.ncol
            nrow = self._layout.nrow
            kw = self._ax_kwa
            return [self.figure.add_subplot(nrow, ncol, i + 1, **kw)
                    for i in xrange(self._layout.nax)]

    def _on_leave_axes(self, event):
        "update the status bar when the cursor leaves axes"
        self._frame.SetStatusText(':-)')

    def _on_motion(self, event):
        "update the status bar for mouse movement"
        ax = event.inaxes
        if ax:
            y_fmt = getattr(ax, 'y_fmt', 'y = %.3g')
            x_fmt = getattr(ax, 'x_fmt', 'x = %.3g')
            # update status bar
            y_txt = y_fmt % event.ydata
            x_txt = x_fmt % event.xdata
            pos_txt = ',  '.join((x_txt, y_txt))

            txt = self._get_statusbar_text(event)
            self._frame.SetStatusText(txt % pos_txt)

    def _show(self, tight=True):
        if tight:
            self.figure.tight_layout()

        self.draw()
        self._frame.Show()

    def _fill_toolbar(self, tb):
        """
        Subclasses should add their toolbar items in this function which
        is called by CanvasFrame.FillToolBar()

        """
        pass

    def close(self):
        "Close the figure."
        self._frame.Close()

    def draw(self):
        "(Re-)draw the figure (after making manual changes)."
        self._frame.canvas.draw()



class subplot_figure(eelfigure):
    def _show(self, figtitle=None):
        self.figure.tight_layout()
        if figtitle:
            t = self.figure.suptitle(figtitle)
            trans = self.figure.transFigure.inverted()
            bbox = trans.transform(t.get_window_extent(self.figure.canvas.renderer))
            print bbox
            t_bottom = bbox[0, 1]
            self.figure.subplots_adjust(top=1 - 2 * (1 - t_bottom))

        super(subplot_figure, self)._show()


class Layout():
    """Create layouts for figures with several axes of the same size
    """
    def __init__(self, nax, h=None, w=None, axh=None, axw=None, nrow=None,
                 ncol=None, ax_aspect=1.5, axh_default=1, dpi=None):
        """Create a grid of axes based on variable parameters.

        Parameters
        ----------
        nax : int
            Number of axes required.
        h, w : scalar
            Height and width of the figure.
        axh, axw : scalar
            Height and width of the axes.
        nrow, ncol : None | int
            Limit number of rows/columns. If both are  None, a square layout is
            produced
        ax_aspect : scalar
            Width / height aspect of the axes.
        axh_default : scalar
            The default axes height if it can not be determined from the other
            parameters.
        """
        if h and axh:
            if h < axh:
                raise ValueError("h < axh")
        if w and axw:
            if w < axw:
                raise ValueError("w < axw")

        if nrow is None and ncol is None:
            if w and axw:
                ncol = math.floor(w / axw)
                nrow = math.ceil(nax / ncol)
                if h:
                    axh = axh or h / nrow
                elif axh:
                    h = axh * nrow
                else:
                    axh = axw / ax_aspect
                    h = axh * nrow
            elif h and axh:
                nrow = math.floor(h / axh)
                ncol = math.ceil(nax / nrow)
                if w:
                    axw = axw or w / ncol
                elif axw:
                    w = axw * ncol
                else:
                    axw = axh * ax_aspect
                    w = axw * ncol
            elif w:
                if axh:
                    ncol = round(w / (axh * ax_aspect))
                else:
                    ncol = round(w / (axh_default * ax_aspect))
                ncol = min(nax, ncol)
                axw = w / ncol
                nrow = math.ceil(nax / ncol)
                if h:
                    axh = h / nrow
                else:
                    if not axh:
                        axh = axw / ax_aspect
                    h = nrow * axh
            elif h:
                if axw:
                    nrow = round(h / (axw / ax_aspect))
                else:
                    nrow = round(h / (axh_default))
                nrow = min(nax, nrow)
                axh = h / nrow
                ncol = math.ceil(nrow / nax)
                if w:
                    axw = w / ncol
                else:
                    if not axw:
                        axw = axh * ax_aspect
                    w = ncol * axw
            elif axh or axw:
                axh = axh or axw / ax_aspect
                axw = axw or axh * ax_aspect
                ncol = min(nax, math.floor(defaults['maxw'] / axw))
                nrow = math.ceil(nax / ncol)
                h = nrow * axh
                w = ncol * axw
            else:
                maxh = defaults['maxh']
                maxw = defaults['maxw']

                # try default
                axh = axh_default
                axw = axh_default * ax_aspect
                ncol = min(nax, maxw / axw)
                nrow = math.ceil(nax / ncol)
                h = axh * nrow
                if h > maxh:
                    col_to_row_ratio = maxw / (ax_aspect * maxh)
                    # nax = ncol * nrow
                    # nax = (col_to_row * nrow) * nrow
                    nrow = math.ceil(math.sqrt(nax / col_to_row_ratio))
                    ncol = math.ceil(nax / nrow)
                    h = maxh
                    axh = h / nrow
                    w = maxw
                    axw = w / ncol
                else:
                    w = axw * ncol
        else:
            if nrow is None:
                ncol = min(nax, ncol)
                nrow = int(math.ceil(nax / ncol))
            elif ncol is None:
                nrow = min(nax, nrow)
                ncol = int(math.ceil(nax / nrow))

            if h:
                axh = axh or h / nrow
            if w:
                axw = axw or w / ncol

            if not axw and not axh:
                axh = axh_default

            if axh and not axw:
                axw = axh * ax_aspect
            elif axw and not axh:
                axh = axh = axw / ax_aspect

        w = w or axw * ncol
        h = h or axh * nrow

        self.nax = nax
        self.h = h
        self.w = w
        self.axh = axh
        self.axw = axw
        self.nrow = int(nrow)
        self.ncol = int(ncol)
        self.fig_kwa = dict(figsize=(w, h), dpi=dpi or defaults['DPI'])


class legend(eelfigure):
    def __init__(self, handles, labels, dpi=90, figsize=(2, 2)):
        fig_kwa = dict(dpi=dpi, figsize=figsize)
        super(legend, self).__init__(title="Legend", fig_kwa=fig_kwa)

        self.legend = self.figure.legend(handles, labels, loc=2)

        self._show(tight=False)


class ImageTiler(object):
    """
    Create tiled images and animations from individual image files.

    """
    def __init__(self, ext='.png', nrow=1, ncol=1, nt=1, dest=None):
        """
        Parameters
        ----------
        ext : str
            Extension to append to generated file names.
        nrow : int
            Number of rows of tiles in a frame.
        ncol : int
            Number of columns of tiles in a frame.
        nt : int
            Number of time points in the animation.
        dest : str(directory)
            Directory in which to place files. If None, a temporary directory
            is created and removed upon deletion of the ImageTiler instance.
        """
        if dest is None:
            self.dir = tempfile.mkdtemp()
        else:
            if not os.path.exists(dest):
                os.makedirs(dest)
            self.dir = dest

        # find number of digits necessary to name images
        row_fmt = '%%0%id' % (np.floor(np.log10(nrow)) + 1)
        col_fmt = '%%0%id' % (np.floor(np.log10(ncol)) + 1)
        t_fmt = '%%0%id' % (np.floor(np.log10(nt)) + 1)
        self._tile_fmt = 'tile_%s_%s_%s%s' % (row_fmt, col_fmt, t_fmt, ext)
        self._frame_fmt = 'frame_%s%s' % (t_fmt, ext)

        self.dest = dest
        self.ncol = ncol
        self.nrow = nrow
        self.nt = nt

    def __del__(self):
        if self.dest is None:
            shutil.rmtree(self.dir)

    def get_tile_fname(self, col=0, row=0, t=0):
        if col >= self.ncol:
            raise ValueError("col: %i >= ncol" % col)
        if row >= self.nrow:
            raise ValueError("row: %i >= nrow" % row)
        if t >= self.nt:
            raise ValueError("t: %i >= nt" % t)

        if self.ncol == 1 and self.nrow == 1:
            return self.get_frame_fname(t)

        fname = self._tile_fmt % (col, row, t)
        return os.path.join(self.dir, fname)

    def get_frame_fname(self, t=0, dirname=None):
        if t >= self.nt:
            raise ValueError("t: %i >= nt" % t)

        if dirname is None:
            dirname = self.dir

        fname = self._frame_fmt % (t,)
        return os.path.join(dirname, fname)

    def make_frame(self, t=0, redo=False):
        """Produce a single frame."""
        dest = self.get_frame_fname(t)

        if os.path.exists(dest):
            if redo:
                os.remove(dest)
            else:
                return

        # collect tiles
        images = []
        colw = [0] * self.ncol
        rowh = [0] * self.nrow
        for r in xrange(self.nrow):
            row = []
            for c in xrange(self.ncol):
                fname = self.get_tile_fname(c, r, t)
                if os.path.exists(fname):
                    im = PIL.Image.open(fname)
                    colw[c] = max(colw[c], im.size[0])
                    rowh[r] = max(rowh[r], im.size[1])
                else:
                    im = None
                row.append(im)
            images.append(row)

        cpos = np.cumsum([0] + colw)
        rpos = np.cumsum([0] + rowh)
        out = PIL.Image.new('RGB', (cpos[-1], rpos[-1]))
        for r, row in enumerate(images):
            for c, im in enumerate(row):
                if im is None:
                    pass
                else:
                    out.paste(im, (cpos[c], rpos[r]))
        out.save(dest)

    def make_frames(self):
        for t in xrange(self.nt):
            self.make_frame(t=t)

    def make_movie(self, dest, framerate=10, codec='mpeg4'):
        """Make all frames and export a movie"""
        dest = os.path.expanduser(dest)
        dest = os.path.abspath(dest)
        root, ext = os.path.splitext(dest)
        dirname = os.path.dirname(dest)
        if ext not in ['.mov', '.avi']:
            if len(ext) == 4:
                dest = root + '.mov'
            else:
                dest = dest + '.mov'

        if not cmd_exists('ffmpeg'):
            err = ("Need ffmpeg for saving movies. Download from "
                   "http://ffmpeg.org/download.html")
            raise RuntimeError(err)
        elif os.path.exists(dest):
            os.remove(dest)
        elif not os.path.exists(dirname):
            os.mkdir(dirname)

        self.make_frames()

        # make the movie
        frame_name = self._frame_fmt
        cmd = ['ffmpeg',  # ?!? order of options matters
               '-f', 'image2',  # force format
               '-r', str(framerate),  # framerate
               '-i', frame_name,
               '-c', codec,
               '-sameq', dest,
               '-pass', '2'  #
               ]
        sp = subprocess.Popen(cmd, cwd=self.dir, stdout=subprocess.PIPE,
                              stderr=subprocess.PIPE)
        stdout, stderr = sp.communicate()
        if not os.path.exists(dest):
            raise RuntimeError("ffmpeg failed:\n" + stderr)

    def save_frame(self, dest, t=0, overwrite=False):
        if not overwrite and os.path.exists(dest):
            raise IOError("File already exists: %r" % dest)
        self.make_frame(t=t)
        fname = self.get_frame_fname(t)
        im = PIL.Image.open(fname)
        im.save(dest)
