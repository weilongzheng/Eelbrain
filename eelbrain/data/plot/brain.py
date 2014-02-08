'''
Plot :class:`NDVar` objects containing source estimates with mayavi/pysurfer.

.. autosummary::
   :toctree: generated

   activation
   dspm
   stat
   STC

'''
# author: Christian Brodbeck
from __future__ import division

import os

import numpy as np
from mayavi import mlab
import surfer

from ... import ui
from .. import testnd
from . import _base


__all__ = ['activation', 'dspm', 'STC', 'stat']


def idx(i):
    return int(round(i))


def dspm(source_estimate, fmin=13, fmid=18, fmax=22, surf='smoothwm',
         hemi='both'):
    """
    Plot a source estimate with coloring for dSPM values.

    Parameters
    ----------
    source_estimate : NDVar, dims = ([case,] source, [time])
        The source estimates. If stc contains a case dimension, the average
        across cases is taken.
    fmin, fmid, fmax : scalar
        Start-, mid- and endpoint for the color gradient.
    surf : 'smoothwm' | ...
        Freesurfer surface.
    hemi : 'lh' | 'rh' | 'both'
        Which hemisphere to plot.
    """
    if not (fmin < fmid < fmax):
        raise ValueError("Invalid colormap, we need fmin < fmid < fmax")

    n = 256
    lut = np.zeros((n, 4), dtype=np.uint8)
    i0 = idx(n / 2)
    imin = idx((fmin / fmax) * i0)
    min_n = i0 - imin
    min_p = i0 + imin
    imid = idx((fmid / fmax) * i0)
    mid_n = i0 - imid
    mid_p = i0 + imid

    # red end
    lut[i0:, 0] = 255
    lut[mid_p:, 1] = np.linspace(0, 255, n - mid_p)

    # blue end
    lut[:i0, 2] = 255
    lut[:mid_n, 0] = np.linspace(127, 0, mid_n)
    lut[:mid_n, 1] = np.linspace(127, 0, mid_n)

    # alpha
    lut[:mid_n, 3] = 255
    lut[mid_n:min_n, 3] = np.linspace(255, 0, min_n - mid_n)
    lut[min_n:min_p, 3] = 0
    lut[min_p:mid_p, 3] = np.linspace(0, 255, mid_p - min_p)
    lut[mid_p:, 3] = 255

    if source_estimate.has_case:
        source_estimate = source_estimate.aggregate()

    return STC(source_estimate, lut, min=-fmax, max=fmax, surf=surf,
               colorbar=False, hemi=hemi)



def stat(p_map, param_map=None, p0=0.05, p1=0.01, solid=False, surf='smoothwm',
         hemi='both', dtmin=0.01):
    """
    Plot a statistic in source space.

    Parameters
    ----------
    p_map : NDVar
        Statistic to plot (normally a map of p values).
    param_map : NDVar
        Statistical parameter covering the same data points as p_map. Used
        only for incorporating the directionality of the effect into the plot.
    p0, p1 : scalar
        Threshold p-values for the color map.
    solid : bool
        Use solid color patches between p0 and p1 (else: blend transparency
        between p0 and p1)
    surf : 'smoothwm' | ...
        Freesurfer surface.
    hemi : 'lh' | 'rh' | 'both'
        Which hemisphere to plot.
    dtmin : scalar | None
        Temporal cutoff: minimum duration of p < p0 to display.
    """
    if dtmin and p_map.has_dim('time'):
        p_map = testnd.clean_time_axis(p_map, dtmin, below=p0, null=1)
    pmap, lut, vmax = p_lut(p_map, param_map, p0=p0, p1=p1, solid=solid)
    plot = STC(pmap, colormap=lut, min=-vmax, max=vmax, colorbar=False,
               surf=surf, hemi=hemi)
    return plot


def activation(source_estimate, a_thresh=None, act_max=None, surf='smoothwm',
               hemi='both'):
    """
    Plot activation in source space.

    Parameters
    ----------
    source_estimate : NDVar
        An NDVar describing activation in source space.
    a_thresh : scalar | None
        the point at which alpha transparency is 50%. When None,
        a_thresh = one standard deviation above and below the mean.
    act_max : scalar | None
        the upper range of activation values. values are clipped above this range.
        When None, act_max = two standard deviations above and below the mean.
    surf : 'smoothwm' | ...
        Freesurfer surface.
    hemi : 'lh' | 'rh' | 'both'
        Which hemisphere to plot.
    """
    x = source_estimate.x.mean()
    std = source_estimate.x.std()

    if a_thresh is None:
        a_thresh = x + std
    if act_max is None:
        act_max = x + 2 * std

    lut = colorize_activation(a_thresh=a_thresh, act_max=act_max)
    if source_estimate.has_case:
        source_estimate = source_estimate.aggregate()

    plot = STC(source_estimate, colormap=lut, min=-act_max, max=act_max,
               colorbar=False, surf=surf, hemi=hemi)
    return plot


class STC:
    """
    Plot a source space NDVar.

    See also
    --------
    activation : plot activation in source space
    stat : plot statistics in source space
    """
    def __init__(self, v, colormap='hot', min=0, max=30, surf='smoothwm',
                 figsize=(500, 500), colorbar=True, hemi='both'):
        """
        Parameters
        ----------
        v : NDVar, dims = (source, [time])
            Ndvar to plot. Must contain a source dimension, and can optionally
            contain a time dimension.
        colormap : str | array
            Colormap name or look up table.
        min, max : scalar
            Endpoints of the colormap.
        surf : 'smoothwm' | ...
            Freesurfer surface.
        figsize : tuple (int, int)
            Size of the mayavi figure in pixels.
        colorbar : bool
            Add a colorbar to the figure.
        hemi : 'both' | 'l[eft]' | 'r[ight]'
            Only plot one hemisphere.
        """
        hemi = hemi.lower()
        if hemi.startswith('l'):
            v = v.sub(source='lh')
        elif hemi.startswith('r'):
            v = v.sub(source='rh')

        self.fig = fig = mlab.figure(size=figsize)
        self.lh = self.rh = None
        b_kwargs = dict(surf=surf, figure=fig, curv=True)
        d_kwargs = dict(colormap=colormap, alpha=1, smoothing_steps=20,
                        time_label='%.3g s', min=min, max=max,
                        colorbar=colorbar)
        if v.has_dim('time'):
            d_kwargs['time'] = v.time.times

        if v.source.lh_n:
            self.lh = self._hemi(v, 'lh', b_kwargs, d_kwargs)
            self._lh_visible = True
            d_kwargs['time_label'] = None
        else:
            self._lh_visible = False

        if v.source.rh_n:
            self.rh = self._hemi(v, 'rh', b_kwargs, d_kwargs)
            self._rh_visible = True
        else:
            self._rh_visible = False

        # Brian object with decorators
        self._dec_hemi = self.lh or self.rh

        # time
        self._time_index = 0
        if v.has_dim('time'):
            self._time = v.get_dim('time')
        else:
            self._time = False

    def _hemi(self, v, hemi, b_kwargs, d_kwargs):
        brain = surfer.Brain(v.source.subject, hemi, **b_kwargs)
        data = v.sub(source=hemi).x
        vert = v.source.vertno[hemi == 'rh']
        brain.add_data(data, vertices=vert, **d_kwargs)
        return brain

    def animate(self, tstart=None, tstop=None, tstep=None, view=None,
                save_frames=False, save_mov=False, framerate=10,
                codec='mpeg4'):
        """
        cycle through time points and optionally save each image. Saving the
        animation (``save_mov``) requires `ffmpeg <http://ffmpeg.org>`_

        Parameters
        ----------
        tstart, tstop, tstep | scalar
            Start, end and step time for the animation.
        save_frames : str(path)
            Path to save frames to. Should contain '%s' for frame index.
            Extension determines format (mayavi supported formats).
        save_mov : str(path)
            save the movie
        """
        # find time points
        if tstep is None:
            times = self._time.x
            if tstart is not None:
                times = times[times >= tstart]
            if tstop is not None:
                times = times[times <= tstop]
        else:
            if tstart is None:
                tstart = self._time.x.min()
            if tstop is None:
                tstop = self._time.x.max()
            # make range inclusive
            if not (tstop - tstart) % tstep:
                tstop += tstep / 2
            times = np.arange(tstart, tstop, tstep)

        if view is None:
            pass
        elif isinstance(view, str):
            self.show_view(view)
            view = None
        else:
            # convert unnested list entries
            isstr = [isinstance(v, str) for v in view]
            for i in np.nonzero(isstr)[0]:
                view[i] = [view[i]]

        nt = len(times)
        if view is None:
            nrow = 1
            ncol = 1
        else:
            nrow = len(view)
            ncols = map(len, view)
            ncol = ncols[0]
            for n in ncols[1:]:
                assert n == ncol

        tiler = _base.ImageTiler('.png', nrow, ncol, nt)

        self.fig.scene.disable_render = True
        if view is None:
            self._make_view_frames(tiler, times)
        else:
            time_label_shown = False
            for r, row in enumerate(view):
                for c, view_ in enumerate(row):
                    self.show_view(view_)
                    if time_label_shown:
                        self._dec_hemi.texts['time_label'].visible = False
                    else:
                        self._dec_hemi.texts['time_label'].visible = True
                        time_label_shown = True
                    self._make_view_frames(tiler, times, r, c)

        tiler.make_movie(save_mov, framerate, codec)
        self.fig.scene.disable_render = False

    def _make_view_frames(self, tiler, times, row=0, col=0):
        "Make all frames for a single view"
        for i, t in enumerate(times):
            self.set_time(t)
            fname = tiler.get_tile_fname(col, row, i)
            self.save_frame(fname)

    def close(self):
        if self.lh is None:
            self.rh.close()
        else:
            self.lh.close()

    def save_frame(self, fname, view=None):
        """
        Save an image with one or more views on the current brain.

        Parameters
        ----------
        fname : str(path)
            Destination of the image file.
        view : str | list
            View(s) to include in the image.

        """
        if view is None:
            self.fig.scene.save(fname)
            return
        elif isinstance(view, str):
            self.show_view(view)
            self.fig.scene.save(fname)
            return

        # convert unnested list entries
        isstr = [isinstance(v, str) for v in view]
        for i in np.nonzero(isstr)[0]:
            view[i] = [view[i]]

        nrow = len(view)
        ncols = map(len, view)
        ncol = ncols[0]
        for n in ncols[1:]:
            assert n == ncol

        _, ext = os.path.splitext(fname)
        im = _base.ImageTiler(ext, nrow, ncol)
        time_label_shown = False
        for r, row in enumerate(view):
            for c, view_ in enumerate(row):
                if time_label_shown:
                    self.show_time_label(False)
                else:
                    self.show_time_label(True)
                    time_label_shown = True
                tile_fname = im.get_tile_fname(c, r)
                self.save_frame(tile_fname, view_)

        im.make_frame(fname, redo=True)

    def save_movie(self, fname=None, view=1, tstep=0.01, tstart=None,
                   tstop=None):
        """Save a movie

        Parameters
        ----------
        fname : str | None
            Path where to save the file (if None, open a save-as dialog).
        view : list | int
            list of view names, or an int describing a preset (see below).

        Notes
        -----
        View presets:

        1. lateral and medial for both hemispheres, for inflated surface
        2. lateral and medial for both hemispheres, plus caudal and ventral,
           for smoothwm surface
        """
        if not fname:
            fname = ui.ask_saveas("Save Movie", "Save Movie",
                                  [('Movie file (*.mov)', '*.mov')])
            if not fname:
                return

        if isinstance(view, int):
            if view == 1:
                view = [['lh lateral', 'rh lateral'],
                        ['rh medial', 'lh medial']]
            elif view == 2:
                view = [['lh lateral', 'rh lateral', 'caudal'],
                        ['rh medial', 'lh medial', 'ventral']]
            else:
                raise ValueError("Unknown view code: %i" % view)

        self.animate(tstart, tstop, tstep, view, save_mov=fname)

    def set_time(self, time):
        "set the time frame displayed (in seconds)"
        if self._lh_visible:
            self.lh.set_time(time)
            i = self.lh.data["time_idx"]
            if self._rh_visible:
                self.rh.set_data_time_index(i)
        elif self._rh_visible:
            self.rh.set_time(time)

    def show(self, hemi='lh'):
        """
        Change the visible hemispheres.

        Parameters
        ----------
        hemi : 'lh' | 'rh' | 'both' | None
            Hemisphere(s) to show. Show both hemispheres, or only the left or
            the right hemisphere. None hides both hemispheres.

        """
        if hemi == 'both':
            self._show_hemi('lh')
            self._show_hemi('rh')
        elif hemi == 'lh':
            self._show_hemi('lh')
            self._show_hemi('rh', False)
        elif hemi == 'rh':
            self._show_hemi('lh', False)
            self._show_hemi('rh')
        elif hemi is None:
            self._show_hemi('lh', False)
            self._show_hemi('rh', False)
        else:
            err = ("Invalid parameter: hemi = %r" % hemi)
            raise ValueError(err)

    def _show_hemi(self, hemi, show=True):
        "Show or hide one hemisphere"
        visible_attr = '_%s_visible' % hemi
        if getattr(self, visible_attr) == show:
            return

        if hemi == 'lh':
            brain = self.lh
        elif hemi == 'rh':
            brain = self.rh
        else:
            err = ("Invalid parameter: hemi = %r" % hemi)
            raise ValueError(err)

        if brain is None:
            return

        brain._geo_mesh.visible = show
        setattr(self, visible_attr, show)
        if hasattr(brain, "data"):
            surf = brain.data['surface']
            surf.visible = show

            # update time index
            if show and (brain.data['time_idx'] != self._time_index):
                brain.set_data_time_index(self._time_index)

    def show_time_label(self, show=True):
        "Show or hide the time label."
        texts = self._dec_hemi.texts
        if 'time_label' in texts:
            if texts['time_label'].visible != show:
                texts['time_label'].visible = show

    def show_view(self, view):
        """
        Parameters
        ----------
        view : str
            PySurfer show_view argument.
            'lateral', 'medial' and 'parietal' need a hemisphere prefix (e.g.,
            'lh lateral') for 'lateral' and 'medial', the opposite hemisphere
            will be hidden.
        """
        if view.endswith(('frontal', 'lateral', 'medial', 'parietal')):
            hemi, view = view.split()
            if hemi == 'lh':
                plot = self.lh
            elif hemi == 'rh':
                plot = self.rh
            else:
                err = ("The first segment of view needs to specify a "
                       "hemisphere ('lh'/'rh', not %r)." % hemi)
                raise ValueError(err)
        elif self.lh is None:
            plot = self.rh
        else:
            plot = self.lh

        if view in ('lateral', 'medial'):
            self.show(hemi)
        else:
            self.show('both')

        plot.show_view(view)



def p_lut(pmap, tmap, p0=0.05, p1=0.01, n=256, solid=False):
    """Creat a color look up table (lut) for p-values

    Parameters
    ----------
    pmap : NDVar
        Map of p-values.
    tmap : NDVar
        Map of signed statistic (only used to code the sign of each p-value).
    p0 : scalar
        Highest p-vale that should be visible.
    p1 : scalar
        P-value where the colormap changes from ramping alpha to ramping color.
    n : int
        Number of color categories in the lut.
    solid : bool
        Instead of ramping alpha/color hue, create steps at p0 and p1.
    """
    if p1 >= p0:
        raise ValueError("p1 needs to be smaller than p0.")

    pstep = 2 * p0 / idx(n / 2 - 1)

    # max p-value that needs to be represented
    vmax = p0 + pstep

    # bring interesting p-values to the range [pstep vmax]
    pmap = vmax - pmap

    # set uninteresting values to zero
    pmap.x.clip(0, vmax, pmap.x)

    # add sign to p-values
    if tmap is not None:
        pmap.x *= np.sign(tmap.x)

    # http://docs.enthought.com/mayavi/mayavi/auto/example_custom_colormap.html
    lut = np.zeros((n, 4), dtype=np.uint8)

    middle = n / 2
    p0n = idx(p0 / pstep)
    p0p = n - p0n
    p1n = idx(p1 / pstep)
    p1p = n - p1n

    # negative colors
    lut[:middle, 2] = 255
    lut[:p1n, 0] = np.linspace(255, 0, p1n)

    # positive colors
    lut[middle:, 0] = 255
    lut[p1p:, 1] = np.linspace(0, 255, n - p1p)

    # alpha
    if solid:
        lut[:p0n, 3] = 255
        lut[p0p:, 3] = 255
    else:
        lut[:p1n, 3] = 255
        lut[p1n:p0n, 3] = np.linspace(255, 0, p0n - p1n)
        lut[p0p:p1p, 3] = np.linspace(0, 255, p1p - p0p)
        lut[p1p:, 3] = 255

    return pmap, lut, vmax



def colorize_activation(a_thresh=3, act_max=8):
    """
    Creates a lookup table containing a color map for plotting stc activation.

    Parameters
    ----------

    a_thresh : int
        a_thresh is point at which values gain 50% visibility.
        from a_thresh and beyond, alpha = 100% visibility.

    act_max : int
        the upper range of activation values. values are clipped above this range. When None,
        act_max = two standard deviations above and below the mean.


    Notes
    -----

    midpoint is 127
    act_max is the upper bound range of activation.

    for colors:

    - negative is blue.
    - super negative is purple.
    - positive is red.
    - super positive is yellow.

    """

    values = np.linspace(-act_max, act_max, 256)
    trans_uidx = np.argmin(np.abs(values - a_thresh))
    trans_lidx = np.argmin(np.abs(values + a_thresh))

    # Transparent ramping
    lut = np.zeros((256, 4), dtype=np.uint8)
    lut[127:trans_uidx, 3] = np.linspace(0, 128, trans_uidx - 127)
    lut[trans_lidx:127, 3] = np.linspace(128, 0, 127 - trans_lidx)
    lut[trans_uidx:, 3] = 255
    lut[:trans_lidx, 3] = 255

    # negative -> Blue
    lut[:127, 2] = 255
    # super negative -> Purple
    lut[:trans_lidx, 0] = np.linspace(0, 255, trans_lidx)
    # positive -> Red
    lut[127:, 0] = 255
    # super positive -> Yellow
    lut[trans_uidx:, 1] = np.linspace(0, 255, 256 - trans_uidx)

    return lut
