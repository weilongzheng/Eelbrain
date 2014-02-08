'''
Created on Feb 17, 2012

@author: christian
'''


import logging
import math
import os
import time

import mne
import numpy as np
import wx
from wx.lib.dialogs import ScrolledMessageDialog

from ..data.data_obj import Dataset, Var, corr, asndvar
from ..data import load, save
from ..data import plot
from ..data import process
from ..data.plot._base import eelfigure, find_fig_vlims
from ..data.plot.utsnd import _ax_bfly_epoch
from ..data.plot.nuts import _plt_bin_nuts
from .. import ui
from ..wxutils import mpl_canvas, Icon
from . import ID

__all__ = ['SelectEpochs', 'pca']


class SelectEpochs(eelfigure):
    """
    Interfaces showing individual cases in an NDVar as butterfly plots, with
    the option to interactively manipulate a boolean list (i.e., select cases).

    LMB on any butterfly plot:
        (de-)select this case.
    't' on any butterfly plot:
        Open a Topomap for the current time point.
    'b' on any butterfly plot:
        Open a full butterfly plot for the current epoch.
    'c' on any epoch butterfly plot:
        Open a plot with the correlation of each channel with its neighbors.
    left/right arrow keys:
        Cycle through pages.
    """
    def __init__(self, ds, data='meg', target='accept', blink=True, path=None,
                 bad_chs=None, nplots=16, plotw=3, ploth=1.5, fill=True,
                 color=None, mark=None, mcolor='r', mean=True, topo=True,
                 vlim=None, dpi=60):
        """
        Plots all cases in the collection segment and allows visual selection
        of cases. The selection can be retrieved through the get_selection
        Method.

        Parameters
        ----------
        ds : Dataset | mne.Epochs
            Dataset on which to perform the selection. Can also be mne.Epochs
            instance.
        data : str | NDVar
            Epoch data as case by sensor by time NDVar (or its name in ds).
        target : str | Var
            Boolean variable indicating for each epoch whether it is accepted
            (True) or rejected (False). If a str with no corresponding variable
            in ds, a new variable is created.
        blink : bool | str | blink epochs (see load.eyelink)
            Overlay eye tracker blink data on the epoch plots. If
            True and ds contains eyetracker information ('t_edf' variable and
            .info['edf'], as added by ``load.eyelink.events(path, ds=ds)``),
            the blink epochs are extracted automatically. Can also be str (name
            in ds) or blink epochs directly.
        path : None | str
            Path to the rejection file. If the file already exists, its values
            are read and applied immediately. If the file does not exist, this
            will be the default path for saving the file (valid extensions are
            .txt and .pickled).
        bad_chs : None | list
            Bad channels (are excluded from plotting and automatic rejection,
            and saved in ds.info['bad_chs'] when the selection is pickled).
        nplots : int | tuple  (n_row, n_col)
            Number of plots (including topo plots and mean). If an int n, a
            square arrangement of n plots is produced. Alternatively, an
            (n_row, n_col) tuple can be specified.
        plotw, ploth : scalar
            Width and height of each plot in inches.
        fill : bool
            Only show the range in the butterfly plots, instead of all traces.
            This is faster for data with many channels.
        color : None | matplotlib color
            Color for primary data (defaultis black).
        mark : None | index for sensor dim
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        mean : bool
            Plot the page mean on each page.
        topo : bool
            Show a dynamically updated topographic plot at the bottom right.
        vlim : scalar
            y-limit of the butterfly plots.
        dpi : scalar
            Figure DPI (determines figure size).


        Examples
        --------
        >>> SelectEpochs(my_dataset)
        [... visual selection of cases ...]
        >>> cases = my_dataset['reject'] == False
        >>> pruned_dataset = my_dataset[cases]

        """
    # interpret plotting args
        # allow ds to be mne.Epochs
        if isinstance(ds, mne.Epochs):
            epochs = ds
            if not epochs.preload:
                err = ("Need Epochs with preloaded data (preload=True)")
                raise ValueError(err)
            ds = Dataset()
            ds[data] = epochs
            ds['trigger'] = Var(epochs.events[:, 2])


        # variable keeping track of selection
        data = asndvar(data, ds=ds)
        self._data = data

        if isinstance(blink, basestring):
            blink = ds.get(blink, None)
        elif blink == True:
            if 'edf' in ds.info:
                tmin = data.time.tmin
                tmax = data.time.tmax
                _, blink = load.eyelink.artifact_epochs(ds, tmin=tmin, tmax=tmax,
                                                        esacc=False)
            else:
                msg = ("No eye tracker data was found in ds.info['edf']. Use "
                       "load.eyelink.add_edf(ds) to add an eye tracker file "
                       "to a Dataset ds.")
                wx.MessageBox(msg, "Eye Tracker Data Not Found")
                blink = None

        self._blink = blink if blink else None

        if isinstance(nplots, int):
            if nplots == 1:
                mean = False
            elif nplots < 1:
                raise ValueError("nplots needs to be >= 1; got %r" % nplots)
            nax = nplots + bool(mean) + bool(topo)
            nrow = math.ceil(math.sqrt(nax))
            ncol = int(math.ceil(nax / nrow))
            nrow = int(nrow)
            n_per_page = nplots
        else:
            nrow, ncol = nplots
            nax = ncol * nrow
            if nax == 1:
                mean = False
                topo = False
            elif nax == 2:
                mean = False
            elif nax < 1:
                err = ("nplots=%s: Need at least one plot." % str(nplots))
                raise ValueError(err)
            n_per_page = nax - bool(topo) - bool(mean)

        if isinstance(target, basestring):
            self._target_name = target
            if target in ds:
                target = ds[target]
            else:
                x = np.ones(ds.n_cases, dtype=bool)
                target = Var(x, name=target)
                ds.add(target)
        else:
            self._target_name = target.name
        self._target = target
        self._saved_target = target.copy()
        self._plot_mean = mean
        self._plot_topo = topo
        self._topo_fig = None
        self._path = path

    # prepare segments
        self._nplots = (nrow, ncol)
        self._n_per_page = n_per_page
        self._n_pages = n_pages = int(math.ceil(ds.n_cases / n_per_page))

        # get a list of IDS for each page
        self._segs_by_page = []
        for i in xrange(n_pages):
            start = i * n_per_page
            stop = min((i + 1) * n_per_page, ds.n_cases)
            self._segs_by_page.append(range(start, stop))

    # init wx frame
        fig_kwa = dict(figsize=(plotw * ncol, ploth * nrow), dpi=dpi)
        super(self.__class__, self).__init__("SelectEpochs", None,
                                             fig_kwa=fig_kwa)
        self._frame.Bind(wx.EVT_CLOSE, self._OnClose)  # , source=None, id=-1, id2=-1

    # setup figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05, top=.95,
                                    hspace=.5)
        # connect canvas
        self.canvas.mpl_connect('button_press_event', self._on_click)
        self.canvas.mpl_connect('key_release_event', self._on_key)
        if self._is_wx:
            self.canvas.mpl_connect('axes_leave_event', self._on_leave_axes)

    # compile plot kwargs:
        self._vlims = find_fig_vlims([[data]])
        self._set_plot_style(fill, color, mark, mcolor)

    # finalize
        self._set_bad_chs(bad_chs, reset=True)
        self._ds = ds
        if path and os.path.exists(path):
            self._load_selection(path)

        self.show_page(0)
        if vlim is not None:
            self.set_vlim(vlim)
        self._frame.store_canvas()
        self._show(tight=False)
        self._UpdateTitle()

    def _fill_toolbar(self, tb):
        tb.AddSeparator()

        # --> select page
        txt = wx.StaticText(tb, -1, "Page:")
        tb.AddControl(txt)
        pages = []
        for i in xrange(self._n_pages):
            istart = self._segs_by_page[i][0]
            if i == self._n_pages - 1:
                pages.append('%i: %i..%i' % (i, istart, len(self._data)))
            else:
                pages.append('%i: %i...' % (i, istart))
        c = self._page_choice = wx.Choice(tb, -1, choices=pages)
        tb.AddControl(c)
        tb.Bind(wx.EVT_CHOICE, self._OnPageChoice)

        # forward / backward
        self._backTool = tb.AddLabelTool(wx.ID_BACKWARD, "Back",
                                         Icon("tango/actions/go-previous"))
        tb.Bind(wx.EVT_TOOL, self._OnBackward, id=wx.ID_BACKWARD)
        self._nextTool = tb.AddLabelTool(wx.ID_FORWARD, "Next",
                                         Icon("tango/actions/go-next"))
        tb.Bind(wx.EVT_TOOL, self._OnForward, id=wx.ID_FORWARD)
        if self._n_pages < 2:
            tb.EnableTool(wx.ID_FORWARD, False)
            tb.EnableTool(wx.ID_BACKWARD, False)
        tb.AddSeparator()

        # Thresholding
        btn = wx.Button(tb, ID.THRESHOLD, "Threshold")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnThreshold)

        # exclude channels
        btn = wx.Button(tb, ID.EXCLUDE_CHANNELS, "Exclude Channel")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnExcludeChannel)

        # save rejection
        btn = wx.Button(tb, ID.SAVE_REJECTION, "Save")
        btn.SetHelpText("Save the epoch selection to a file.")
        tb.AddControl(btn)
        btn.Bind(wx.EVT_BUTTON, self._OnSaveSelection)

    def auto_reject(self, threshold=2e-12, above=False, below=True):
        process.mark_by_threshold(self._ds, x=self._data,
                                  threshold=threshold, above=above,
                                  below=below, target=self._target,
                                  bad_chs=self._bad_chs)

        self._refresh()
        self._UpdateTitle()

    def _get_page_mean_seg(self, sensor=None):
        seg_IDs = self._segs_by_page[self._current_page_i]
        index = np.zeros(self._ds.n_cases, dtype=bool)
        index[seg_IDs] = True
        index = np.logical_and(index, self._target)
        mseg = self._data.aggregate(case=index)
        mseg = mseg.sub(sensor=sensor)
        return mseg

    def _get_statusbar_text(self, event):
        "called by parent class to get figure-specific status bar text"
        ax = event.inaxes
        if ax and (ax.ID > -2):  # topomap ID is -2
            t = event.xdata
            tseg = self._get_topo_seg(ax, t)
            if ax.ID >= 0:  # single trial plot
                txt = 'Epoch %i,   ' % ax.segID + '%s'
            elif  ax.ID == -1:  # mean plot
                txt = "Page average,   %s"
            # update internal topomap plot
            plot.topo._ax_topomap(self._topo_ax, [tseg], **self._topo_kwargs)
            self._frame.redraw(axes=[self._topo_ax])
            # update external topomap plot
            if self._topo_fig:
                pass

            return txt
        else:
            return '%s'

    def _get_topo_seg(self, ax, t):
        name = '%%s, %.3f s' % t
        ax_id = ax.ID
        if ax_id == -1:
            tseg = self._mean_seg.sub(time=t, name=name % 'Page Average')
        elif ax_id >= 0:
            seg = self._case_segs[ax_id]
            tseg = seg.sub(time=t, name=name % 'Epoch %i' % ax.segID)
        else:
            raise IndexError("ax_id needs to be >= -1, not %i" % ax_id)

        self._tseg = tseg
        return tseg

    def _update_mean(self):
        mseg = self._get_page_mean_seg()
        self._mean_plot.set_data(mseg)
        self._frame.redraw(axes=[self._mean_ax])

    def set_ax_state(self, axID, state):
        """Set the state (accept/reject) of one axis.

        Parameters
        ----------
        axID : int
            Axis ID (index of the axis on the current page)
        state : bool
            Accept (True) or reject (False).
        """
        ax = self._case_axes[axID]
        h = self._case_plots[axID]
        h.set_state(state)
        ax._epoch_state = state

        self._frame.redraw(artists=[h.ax])
        self._update_mean()

    def get_bad_chs(self, name=True):
        """Get the channels currently set as bad

        Parameters
        ----------
        name : bool
            Return channel names (otherwise the channel index is returned).

        Returns
        -------
        bad_chs : None | list of int, str
            Channels currenty excluded.
        """
        if name:
            return [self._data.sensor.names[i] for i in self._bad_chs]
        else:
            return self._bad_chs[:]

    def invert_selection(self, axID):
        "ID refers to ax-ID in the display"
        # find current selection
        ax = self._case_axes[axID]
        epochID = ax.segID
        state = not self._target[epochID]
        self._target[epochID] = state

        # update plot
        self.set_ax_state(axID, state)
        self._UpdateTitle()

    def load_selection(self, path):
        try:
            self._load_selection(path)
        except Exception as ex:
            msg = str(ex)
            title = "Error Loading Rejections"
            dlg = ScrolledMessageDialog(self, msg, title)
            dlg.ShowModal()
            dlg.Destroy()
        self._refresh()

    def _load_selection(self, path):
        _, ext = os.path.splitext(path)
        if ext == '.pickled':
            ds = load.unpickle(path)
        elif ext == '.txt':
            ds = load.txt.tsv(path)
        else:
            raise ValueError("Unknown file extension for rejections: %r" % ext)

        if np.all(ds['trigger'] == self._ds['trigger']):
            self._target[:] = ds['accept']
            if 'bad_chs' in ds.info:
                bad_chs = ds.info['bad_chs']
                self._set_bad_chs(bad_chs, reset=True)
            self._path = path
            self._saved_target[:] = ds['accept']
        else:
            err = ("Event IDs of the file don't match the current data's "
                   "event IDs")
            cap = "Error Loading Rejection"
            wx.MessageBox(err, cap, wx.ICON_ERROR)

    def open_topomap(self):
        if self._topo_fig:
            pass
        else:
            fig = plot.Topomap(self._tseg, sensors='name')
            self._topo_fig = fig

    def save_rejection(self, path):
        """Save the rejection list as a file

        Parameters
        ----------
        path : str
            Path under which to save. The extension determines the way file
            (*.pickled -> pickled Dataset; *.txt -> tsv)
        """
        # find dest path
        root, ext = os.path.splitext(path)
        if ext == '':
            ext = '.pickled'
        path = root + ext

        # create Dataset to save
        accept = self._target
        if accept.name != 'accept':
            accept = accept.copy('accept')
        ds = Dataset(self._ds['trigger'], accept,
                     info={'bad_chs': self.get_bad_chs()})

        if ext == '.pickled':
            save.pickle(ds, path)
        elif ext == '.txt':
            ds.save_txt(path)
        else:
            raise ValueError("Unsupported extension: %r" % ext)
        self._saved_target[:] = self._target
        self._UpdateTitle()

    def set_bad_chs(self, bad_chs=None, reset=False):
        """Set the channels to treat as bad (i.e., exclude)

        Parameters
        ----------
        bad_chs : None | list of str, int
            List of channels to treat as bad (as name or index).
        reset : bool
            Reset previously set bad channels to good.
        """
        self._set_bad_chs(bad_chs, reset=reset)
        self._refresh()

    def _set_bad_chs(self, bad_chs, reset=False):
        "Set the self._bad_chs value, but don't refresh the plot"
        if reset:
            self._bad_chs = []

        if bad_chs is None:
            return

        bad_chs = self._data.sensor.dimindex(bad_chs)
        for ch in bad_chs:
            if ch in self._bad_chs:
                continue

            self._bad_chs.append(ch)

    def set_plot_style(self, fill=True, color=None, mark=None, mcolor='r'):
        """Select channels to mark in the butterfly plots.

        Parameters
        ----------
        fill : bool
            Only show the range in the butterfly plots, instead of all traces.
            This is faster for data with many channels.
        color : None | matplotlib color
            Color for primary data (defaultis black).
        mark : None | index for sensor dim
            Sensors to plot as individual traces with a separate color.
        mcolor : matplotlib color
            Color for marked traces.
        """
        self._set_plot_style(fill, color, mark, mcolor)
        self.show_page()

    def _set_plot_style(self, fill, color, mark, mcolor):
        if fill or mark is None:
            traces = not bool(fill)
        else:
            traces = self._data.sensor.index(mark, names=True)

        self._bfly_kwargs = {'plot_range': fill, 'traces': traces,
                             'color': color, 'mark': mark, 'mcolor': mcolor,
                             'vlims':self._vlims}
        self._topo_kwargs = {'vlims':self._vlims}

    def set_vlim(self, vlim):
        """Set the value limits (butterfly plot y axes and topomap colormaps)

        Parameters
        ----------
        vlim : scalar | (scalar, scalar)
            For symmetric limits the positive vmax, for asymmetric limits a
            (vmin, vmax) tuple.
        """
        for p in self._case_plots:
            p.set_ylim(vlim)
        self._mean_plot.set_ylim(vlim)
        if np.isscalar(vlim):
            vlim = (-vlim, vlim)
        for key in self._vlims:
            self._vlims[key] = vlim
        self.canvas.draw()

    def show_page(self, page=None):
        "Dislay a specific page (start counting with 0)"
        wx.BeginBusyCursor()
        t0 = time.time()
        if page is None:
            page = self._current_page_i
        else:
            self._current_page_i = page
            self._page_choice.Select(page)
            self._nextTool.Enable(page < self._n_pages - 1)
            self._backTool.Enable(page > 0)

        self.figure.clf()
        nrow, ncol = self._nplots
        seg_IDs = self._segs_by_page[page]

        self._current_bad_chs = self._bad_chs[:]
        if self._bad_chs:
            sens_idx = self._data.sensor.index(self._bad_chs)
        else:
            sens_idx = None

        # segment plots
        self._case_plots = []
        self._case_axes = []
        self._case_segs = []
        for i, ID in enumerate(seg_IDs):
            case = self._data.sub(case=ID, sensor=sens_idx,
                                      name='Epoch %i' % ID)
            state = self._target[ID]
            ax = self.figure.add_subplot(nrow, ncol, i + 1, xticks=[0], yticks=[])  # , 'axis_off')
            ax._epoch_state = state
#            ax.set_axis_off()
            h = _ax_bfly_epoch(ax, case, xlabel=None, ylabel=None, state=state,
                               **self._bfly_kwargs)
            if self._blink is not None:
                _plt_bin_nuts(ax, self._blink[ID], color=(0.99, 0.76, 0.21))

            ax.ID = i
            ax.segID = ID
            self._case_plots.append(h)
            self._case_axes.append(ax)
            self._case_segs.append(case)


        # mean plot
        if self._plot_mean:
            plot_i = nrow * ncol
            ax = self._mean_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            ax.ID = -1

            mseg = self._mean_seg = self._get_page_mean_seg(sensor=sens_idx)
            self._mean_plot = _ax_bfly_epoch(ax, mseg, **self._bfly_kwargs)

        # topomap
        if self._plot_topo:
            plot_i = nrow * ncol - self._plot_mean
            ax = self._topo_ax = self.figure.add_subplot(nrow, ncol, plot_i)
            ax.ID = -2
            ax.set_axis_off()

        self.canvas.draw()
        dt = time.time() - t0
        logging.debug('Page draw took %.1f seconds.', dt)
        wx.EndBusyCursor()

    def _OnBackward(self, event):
        "turns the page backward"
        if self._current_page_i == 0:
            self.show_page(self._n_pages - 1)
        else:
            self.show_page(self._current_page_i - 1)

    def _on_click(self, event):
        "called by mouse clicks"
        logging.debug('click: ')
        ax = event.inaxes
        if ax:
            if ax.ID >= 0:
                self.invert_selection(ax.ID)
            elif ax.ID == -2:
                self.open_topomap()

    def _OnClose(self, event):
        "Ask to save unsaved changes"
        if event.CanVeto() and self._has_unsaved_changes:
            msg = ("The current document has unsaved changes. Would you like "
                   "to save them?")
            cap = ("Saved Unsaved Changes?")
            cmd = wx.MessageBox(msg, cap,
                                wx.YES | wx.NO | wx.CANCEL | wx.YES_DEFAULT)
            if cmd == wx.YES:
                if self._OnSaveSelection(event) != wx.ID_OK:
                    return
            elif cmd == wx.CANCEL:
                return
            elif cmd != wx.NO:
                raise RuntimeError("Unknown answer: %r" % cmd)

        event.Skip()

    def _on_key(self, event):
        ax = event.inaxes
        ax_id = getattr(ax, 'ID', None)
        if (event.key == 't'):
            if ax_id == -2:
                return
            tseg = self._get_topo_seg(ax, t=event.xdata)
            plot.Topomap(tseg, sensors='name')
        elif (event.key == 'b'):
            if ax_id == -1:
                seg = self._mean_seg
            elif ax_id >= 0:
                seg = self._case_segs[ax_id]
            else:
                return
            plot.TopoButterfly(seg)
        elif (event.key == 'c'):
            if ax_id == -2:
                return
            elif ax_id == -1:
                seg = self._mean_seg
                name = 'Page Mean Neighbor Correlation'
            else:
                seg = self._case_segs[ax_id]
                name = 'Epoch %i Neighbor Correlation' % ax.segID
            cseg = corr(seg, name=name)
            plot.Topomap(cseg, sensors='name')
        elif event.key == 'right' and self._current_page_i < self._n_pages - 1:
            self.show_page(self._current_page_i + 1)
        elif event.key == 'left' and self._current_page_i > 0:
            self.show_page(self._current_page_i - 1)

    def _OnExcludeChannel(self, event):
        chs = ', '.join(self.get_bad_chs()) or 'None'
        msg = ("Currently excluded: %s\n"
               "Exclude a channels by name; exclude several channels\n"
               "separated by comma." % chs)
        dlg = wx.TextEntryDialog(self._frame, msg, "Exclude Channels")
        if dlg.ShowModal() != wx.ID_OK:
            return

        chs = map(unicode.strip, dlg.GetValue().split(','))
        chs = filter(None, chs)
        bads = [ch for ch in chs if ch not in self._data.sensor.channel_idx]
        if bads:
            names = ', '.join(map(repr, map(str, bads)))
            if len(bads) > 1:
                name_form = 'names'
            else:
                name_form = 'name'
            msg = ("Invalid channel %s: %s. Use full name (including "
                   "leading '0's)" % (name_form, names))
            title = "Error Excluding Channels"
            dlg = wx.MessageDialog(self._frame, msg, title, wx.OK | wx.ICON_ERROR)
            dlg.ShowModal()
            dlg.Destroy()
        else:
            self.set_bad_chs(chs)

    def _OnForward(self, event):
        "turns the page forward"
        if self._current_page_i < self._n_pages - 1:
            self.show_page(self._current_page_i + 1)
        else:
            self.show_page(0)

    def _on_leave_axes(self, event):
        sb = self.GetStatusBar()
        sb.SetStatusText("", 0)

    def _OnPageChoice(self, event):
        "called by the page Choice control"
        page = self._page_choice.GetSelection()
        self.show_page(page)

    def _OnSaveSelection(self, event):
        msg = ("Save the epoch selection to a file.")
        if self._path:
            default_dir, default_name = os.path.split(self._path)
        else:
            default_dir = ''
            default_name = ''
        wildcard = "Pickle (*.pickled)|*.pickled|Text (*.txt)|*.txt"
        dlg = wx.FileDialog(self._frame, msg, default_dir, default_name,
                            wildcard, wx.FD_SAVE | wx.FD_OVERWRITE_PROMPT)
        rcode = dlg.ShowModal()
        if rcode == wx.ID_OK:
            path = dlg.GetPath()
            self.save_rejection(path)

        dlg.Destroy()
        return rcode

    def _refresh(self, event=None):
        "updates the states of the segments on the current page"
        if self._current_bad_chs == self._bad_chs:
            for ax in self._case_axes:
                state = self._target[ax.segID]
                if state != ax._epoch_state:
                    self.set_ax_state(ax.ID, state)
        else:
            self.show_page()

        self._UpdateTitle()

    def save_all_pages(self, fname=None):
        if fname is None:
            msg = ("Save all pages. Index is inserted before extension. "
                   "If no extension is provided, pdf is used.")
            fname = ui.ask_saveas("title", msg, None)
            if not fname:
                return

        head, tail = os.path.split(fname)
        if os.path.extsep in tail:
            root, ext = os.path.splitext(tail)
        else:
            root = fname
            ext = '.pdf'
        fntemp = os.path.join(head, root) + '_%i' + ext

        i = self.current_page_i
        self.figure.savefig(fntemp % i)
        pages = range(self.n)
        pages.remove(i)
        prog = ui.progress_monitor(self.n - 1, "Saving Figures", "Saving Figures")
        for i in pages:
            self.show_page(i)
            self.figure.savefig(fntemp % i)
            prog.advance()
        prog.terminate()

    def _OnThreshold(self, event):
        threshold = None
        above = False
        below = True

        msg = "What value should be used to threshold the data?"
        while threshold is None:
            dlg = wx.TextEntryDialog(self._frame, msg, "Choose Threshold", "2e-12")
            if dlg.ShowModal() == wx.ID_OK:
                value = dlg.GetValue()
                try:
                    threshold = float(value)
                except ValueError:
                    ui.message("Invalid Entry", "%r is not a valid entry. Need "
                               "a float." % value, '!')
            else:
                return

        self.auto_reject(threshold, above, below)

    def _UpdateTitle(self):
        if self._path:
            fname = os.path.basename(self._path)
            if np.all(self._saved_target == self._target):
                title = '%s' % fname
                self._has_unsaved_changes = False
            else:
                title = '* %s' % fname
                self._has_unsaved_changes = True
        else:
            title = 'Untitled'
            self._has_unsaved_changes = True

        self._frame.SetTitle(title)



class pca(mpl_canvas.CanvasFrame):
    def __init__(self, dataset, Y='MEG', nplots=(7, 10), dpi=50, figsize=(20, 12)):
        """
        Performs PCA and opens a GUI for removing individual components.

        Y : NDVar | str
            dependent variable
        timecourse : int
            number of (randomly picked) segments for whichthe time-course is
            displayed

        """
        if isinstance(Y, basestring):
            Y = dataset[Y]

        self._dataset = dataset
        self._Y = Y

    # prepare plots:
        self._topo_kwargs = {}

    # wx stuff
        parent = wx.GetApp().shell
        title = "PCA of %r" % Y.name
        mpl_canvas.CanvasFrame.__init__(self, parent, title, statusbar=False,
                                        figsize=figsize, dpi=dpi)
        # connect
        self.canvas.mpl_connect('button_press_event', self.OnClick)
        # figure
        self.figure.subplots_adjust(left=.01, right=.99, bottom=.05, top=.95,
                                    hspace=.2)

    # do the PCA
        pca = self.pca = process.PCA(Y)

    # plot the components
        self._components = []
        self._rm_comp = []
        npy, npx = nplots
        title_temp = '%i'
        for i in xrange(np.prod(nplots)):
            name = title_temp % i
            comp = pca.get_component(i)
            ax = self.figure.add_subplot(npy, npx, i + 1, xticks=[], yticks=[])
            ax.Id = i
            plot.topo._ax_topomap(ax, [comp])
            ax.set_title(name)
            ax.set_frame_on(1)
            ax.set_axis_on()
            self._components.append(comp)

    # finalize
        self.canvas.store_canvas()
        self.Show()

    def _init_FillToolBar(self, tb):
        tb.AddSeparator()

        # remove
        btn = wx.Button(tb, ID.PCA_REMOVE, "Remove Selected Components")
        tb.AddControl(btn)
        self.Bind(wx.EVT_BUTTON, self.OnRemove, id=ID.PCA_REMOVE)

        mpl_canvas.CanvasFrame._init_FillToolBar(self, tb)

    def OnClick(self, event):
        ax = event.inaxes
        if ax:
            Id = ax.Id
            if Id in self._rm_comp:
                self._rm_comp.remove(Id)
                ax.set_axis_bgcolor('white')
            else:
                self._rm_comp.append(Id)
                ax.set_axis_bgcolor('r')
            self.canvas.redraw(axes=[ax])

    def OnRemove(self, event):
        target = None
        rm = sorted(self._rm_comp)
        while not target:
            dlg = wx.TextEntryDialog(self, "What name should the new NDVar be assigned in the Dataset?",
                                     "Choose Name for New Variable", "%s" % self._Y.name)
            if dlg.ShowModal() == wx.ID_OK:
                newname = str(dlg.GetValue())
                if newname in self._dataset:
                    msg = ("The Dataset already contains an item named %r. "
                           "Should it be replaced? The item is:\n\n%r" %
                            (newname, self._dataset[newname]))
                    answer = ui.ask("Replace %r?" % newname, msg)
                    if answer is True:
                        target = newname
                    elif answer is None:
                        pass
                    else:
                        return
                else:
                    target = newname
            else:
                return

        # if we made it down here, remove the component:
        self._dataset[target] = self.pca.subtract(rm, name=target)

        self.Close()
