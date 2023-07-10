import numpy as np
import matplotlib.pylab as plt

from . import hist_mst

def set_plot_default(use_bold=True):
    """Sets the default fonts for the matplotlib plots.

    Parameters
    ----------
    use_bold : bool
        Determines whether the fonts used are in bold.
    """
    plt.rc('text', usetex=True)
    plt.rc('font', family='serif')
    if use_bold == True:
        plt.rcParams['text.latex.preamble'] = [r'\boldmath']


def plot_histogram_line(x_mid, histogram, x_edges=None, ax=None, color='C0',
                        alpha=1., linewidth=1., linestyle='-'):
    """Outputs the histogram line boxes without using the histogram function.

    Parameters
    ----------
    x_mid : array_like
        Midpoint of each histogram bar.
    histogram : array_like
        Histogram height.
    x_edges : array_like
        The edges of each histogram bar. Ideal for unequal spacings between histograms.
    ax : class
        The matplotlib plotting axis.
    color : str
        The color of the histogram error plot.
    alpha : float
        The transparency fraction of the histogram error plot.
    linewidth : float
        The linewidth of histogram line.
    linestyle : str
        The linestyle used.
    """
    x_hist = []
    y_hist = []
    if x_edges is None:
        _dx = x_mid[1]-x_mid[0]
        for i in range(0,len(histogram)):
            x_hist.append(x_mid[i] - 0.5*_dx)
            x_hist.append(x_mid[i] + 0.5*_dx)
            y_hist.append(histogram[i])
            y_hist.append(histogram[i])
    else:
        for i in range(0, len(histogram)):
            x_hist.append(x_edges[i])
            x_hist.append(x_edges[i+1])
            y_hist.append(histogram[i])
            y_hist.append(histogram[i])
    x_hist, y_hist = np.array(x_hist), np.array(y_hist)
    if ax == None:
        plt.plot(x_hist, y_hist, color=color, alpha=alpha,
                 linewidth=linewidth, linestyle=linestyle)
    else:
        ax.plot(x_hist, y_hist, color=color, alpha=alpha,
                linewidth=linewidth, linestyle=linestyle)


def plot_histogram_confidence(x_mid, histogram_min, histogram_max, x_edges=None,
                              ax=None, color='C0', alpha=0.25):
    """Plots the confidence envelopes for a histogram.

    Parameters
    ----------
    x_mid : array_like
        Midpoint of each histogram bar.
    histogram_min : array_like
        Histogram minimum envelope.
    histogram_max : array_like
        Histogram maximum envelope.
    x_edges : array_like
        The edges of each histogram bar. Ideal for unequal spacings between histograms.
    ax : class
        The matplotlib plotting axis.
    color : str
        The color of the histogram error plot.
    alpha : float
        The transparency fraction of the histogram error plot.
    """
    if x_edges is None:
        _dx = x_mid[1]-x_mid[0]
        for i in range(0, len(histogram_min)):
            if ax == None:
                plt.fill_between(np.array([x_mid[i]-0.5*_dx, x_mid[i]+0.5*_dx]),
                                np.array([histogram_min[i], histogram_min[i]]),
                                np.array([histogram_max[i], histogram_max[i]]),
                                color=color,alpha=alpha)
            else:
                ax.fill_between(np.array([x_mid[i]-0.5*_dx, x_mid[i]+0.5*_dx]),
                                np.array([histogram_min[i], histogram_min[i]]),
                                np.array([histogram_max[i], histogram_max[i]]),
                                color=color,alpha=alpha)
    else:
        for i in range(0, len(histogram_min)):
            if ax == None:
                plt.fill_between(np.array([x_edges[i], x_edges[i+1]]),
                                np.array([histogram_min[i], histogram_min[i]]),
                                np.array([histogram_max[i], histogram_max[i]]),
                                color=color,alpha=alpha)
            else:
                ax.fill_between(np.array([x_edges[i], x_edges[i+1]]),
                                np.array([histogram_min[i], histogram_min[i]]),
                                np.array([histogram_max[i], histogram_max[i]]),
                                color=color,alpha=alpha)


def plot_histogram_error(x_mid, histogram, histogram_error, x_edges=None, ax=None,
                         color='C0', alpha=0.25):
    """Plots the errorbars envelopes for a histogram.

    Parameters
    ----------
    x_mid : array_like
        Midpoint of each histogram bar.
    histogram : array_like
        Histogram.
    histogram_error : array_like
        The error associated to each histogram bar.
    x_edges : array_like
        The edges of each histogram bar. Ideal for unequal spacings between histograms.
    ax : class
        The matplotlib plotting axis.
    color : str
        The color of the histogram error plot.
    alpha : float
        The transparency fraction of the histogram error plot.
    """
    plot_histogram_confidence(x_mid, histogram-histogram_error, histogram+histogram_error,
                              x_edges=x_edges, ax=ax, color=color, alpha=alpha)


class PlotHistMST:

    def _get_rotate_colors(self):
        """Cycles through the default matplotlib colours, i.e. C0 - C9.
        """
        color = 'C'+str(self.rotate_colors)
        if self.rotate_colors + 1 == 10:
            self.rotate_colors = 0
        else:
            self.rotate_colors += 1
        return color

    def _get_rotate_linestyles(self):
        """Cycles through linestyles "-", "--", "-." and ":".
        """
        linestyle = self.linestyle_all[self.rotate_linestyle]
        if self.rotate_linestyle + 1 == 4:
            self.rotate_linestyle = 0
        else:
            self.rotate_linestyle += 1
        return linestyle

    def __init__(self):
        """Initialises the plotting class.
        """
        # tracking data
        self.num_data = 0
        # line information
        self.colors = []
        self.alphas = []
        self.alphas_envelope = []
        self.labels = []
        self.linestyles = []
        self.linewidths = []
        self.binned_data = []
        self.need_envelopes = []
        # colors and styles
        self.rotate_colors = 0
        self.rotate_linestyle = 0
        self.linestyle_all = ['-', '--', '-.', ':']
        # Type of data used
        self.use_sqrt_s = None
        self.uselog = None
        # Binning information
        self.d_min = None
        self.d_max = None
        self.l_min = None
        self.l_max = None
        self.b_min = None
        self.b_max = None
        self.s_min = None
        self.s_max = None
        self.l_edges = None
        self.b_edges = None
        self.s_edges = None
        self.usenorm = None

    def read_mst(self, mst_hist, color=None, linewidth=2., linestyle=None, alpha=0.8,
                 label=None, alpha_envelope=0.3):
        """Input minimum spanning tree statistics.

        Parameters
        ----------
        mst_hist : dict
            Binned MST dictionary, given in the format outputted by HistMST.
        color : str
            Color of the MST histogram.
        linewidth : float
            Width of line used.
        linestyle : str
            Linestyle used.
        alpha : float
            The transparency of the line.
        label : str
            label used in the legend, this will only appear in the legend if this is given.
        alpha_envelope : float
            The transparency of envelopes (usually the standard deviation of the counts in each bin).
        """
        if self.usenorm is None:
            self.usenorm = mst_hist['usenorm']
        self.binned_data.append(mst_hist)
        self.num_data += 1
        delta_d = mst_hist['x_d'][1]-mst_hist['x_d'][0]
        self.d_min = mst_hist['x_d'][0] - delta_d/2.
        self.d_max = mst_hist['x_d'][-1] + delta_d/2.
        if mst_hist['uselog'] == False:
            self.uselog = False
            delta_l = mst_hist['x_l'][1]-mst_hist['x_l'][0]
            self.l_min = mst_hist['x_l'][0] - delta_l/2.
            self.l_max = mst_hist['x_l'][-1] + delta_l/2.
            delta_b = mst_hist['x_b'][1]-mst_hist['x_l'][0]
            self.b_min = mst_hist['x_b'][0] - delta_b/2.
            self.b_max = mst_hist['x_b'][-1] + delta_b/2.
            self.l_edges = np.linspace(self.l_min, self.l_max, len(mst_hist['x_l'])+1)
            self.b_edges = np.linspace(self.b_min, self.b_max, len(mst_hist['x_b'])+1)
        else:
            self.uselog = True
            logdelta_l = np.log10(mst_hist['x_l'][1])-np.log10(mst_hist['x_l'][0])
            self.l_min = 10.**(np.log10(mst_hist['x_l'][0]) - logdelta_l/2.)
            self.l_max = 10.**(np.log10(mst_hist['x_l'][-1]) + logdelta_l/2.)
            logdelta_b = np.log10(mst_hist['x_b'][1])-np.log10(mst_hist['x_b'][0])
            self.b_min = 10.**(np.log10(mst_hist['x_b'][0]) - logdelta_b/2.)
            self.b_max = 10.**(np.log10(mst_hist['x_b'][-1]) + logdelta_b/2.)
            logl_edges = np.linspace(np.log10(self.l_min), np.log10(self.l_max), len(mst_hist['x_l'])+1)
            logb_edges = np.linspace(np.log10(self.b_min), np.log10(self.b_max), len(mst_hist['x_b'])+1)
            self.l_edges = 10.**logl_edges
            self.b_edges = 10.**logb_edges
        delta_s = mst_hist['x_s'][1]-mst_hist['x_s'][0]
        self.s_min = 0.
        self.s_max = 1.
        if mst_hist['use_sqrt_s'] == True:
            self.use_sqrt_s = True
        if color is None:
            self.colors.append(self._get_rotate_colors())
        else:
            self.colors.append(color)
        self.alphas.append(alpha)
        self.alphas_envelope.append(alpha_envelope)
        self.linewidths.append(linewidth)
        if linestyle is None:
            self.linestyles.append(self._get_rotate_linestyles())
        else:
            self.linestyles.append(linestyle)
        self.labels.append(label)
        self.need_envelopes.append(mst_hist['isgroup'])

    def plot(self, usebox=True, saveas=None, fontsize=16, figsize=(16, 4), subplot_setup='4x1',
             units=None, showenvelopes=True, showsigma=2, usecomp=False, usemean=True, height_ratios=[2, 1],
             usefraction=False, whichcomp=0, plotzeroline=True, legend=True, subplot_adjust_top=0.85,
             legend_fontsize=14, legend_column=4, xlabels=[None, None, None, None], dpi=None, plt_output='show'):
        """Outputs the final plot of the MST statistics.

        Parameters
        ----------
        usebox : bool
            For l, b and s this sets whether to use boxes for the histogram or to simple plot as line
            from the bin centre.
        saveas : str
            If not None, this will save the plot with the name provided.
        fontsize : int
            Fontsize of axis labels.
        figsize : tuple
            Dimensions of the figure.
        subplot_setup : string
            Subplot setup: 4x1 or 2x2.
        units : str
            Units of l and b MST statistics, if None is supplied then we assume it is unitless.
        showenvelopes : bool
            This determines whether to plot data with input standard deviation.
        showsigma : int
            Number of sigma errorbars to plot.
        usecomp : bool
            Determines whether to include comparison subplots.
        usemean : bool
            For comparison plots, the determines whether to use the mean of input distributions.
        height_ratios : tuple
            Height ratio between main plots and comparison plots.
        usefraction : bool
            If true comparison plots are given by the fractional difference to a reference data otherwise
            they are given as absolute differences.
        whichcomp : int
            Determines which data should be the reference data that all other data is compared to. Only used
            if usemean == False.
        plotzeroline : bool
            Plots horizontal zero line for subplot.
        legend : bool
            Determines whether to include a legend.
        subplot_adjust_top : float
            If legend == True then the figure's top is adjusted by the input value.
        legend_fontsize : int
            Size of legend text.
        legend_column : int
            Number of keys in each line of the legend.
        xlabels : list
            List of string labels to replace the default if not set to None.
        dpi : int
            Pixels per inch for non-vector images.
        plt_output : string
            Output type: closed or show.
        """
        # setting up figure
        if self.usenorm == True:
            ylabel = '\\bar{N}'
        else:
            ylabel = 'N'
        if usecomp == True and self.num_data > 1:
            if subplot_setup == '4x1':
                f, ((ax1, ax2, ax3, ax4),
                    (ax15, ax25, ax35, ax45)) = plt.subplots(2, 4, figsize=figsize,
                                                             gridspec_kw={'height_ratios': height_ratios})
            elif subplot_setup == '2x2':
                f, ((ax1, ax2), (ax15, ax25),
                    (ax3, ax4), (ax35, ax45)) = plt.subplots(4, 2, figsize=figsize,
                                                             gridspec_kw={'height_ratios': height_ratios + height_ratios})
            else:
                pass
            if xlabels[0] is None:
                ax15.set_xlabel(r'$d$', fontsize=fontsize)
            else:
                ax15.set_xlabel(xlabels[0], fontsize=fontsize)
            if units is None:
                if xlabels[1] is None:
                    ax25.set_xlabel(r'$l$', fontsize=fontsize)
                else:
                    ax25.set_xlabel(xlabels[1], fontsize=fontsize)
                if xlabels[2] is None:
                    ax35.set_xlabel(r'$b$', fontsize=fontsize)
                else:
                    ax35.set_xlabel(xlabels[2], fontsize=fontsize)
            else:
                if xlabels[1] is None:
                    ax25.set_xlabel(r'$l$ $[%s]$' %(units), fontsize=fontsize)
                else:
                    ax25.set_xlabel(xlabels[1]+r' $[%s]$' %(units), fontsize=fontsize)
                if xlabels[2] is None:
                    ax35.set_xlabel(r'$b$ $[%s]$' %(units), fontsize=fontsize)
                else:
                    ax35.set_xlabel(xlabels[2]+r' $[%s]$' %(units), fontsize=fontsize)
            if self.use_sqrt_s == True:
                if xlabels[3] is None:
                    ax45.set_xlabel(r'$\sqrt{1-s}$', fontsize=fontsize)
                else:
                    ax45.set_xlabel(xlabels[3], fontsize=fontsize)
            else:
                if xlabels[3] is None:
                    ax45.set_xlabel(r'$s$', fontsize=fontsize)
                else:
                    ax45.set_xlabel(xlabels[3], fontsize=fontsize)
            if usefraction == False:
                ax15.set_ylabel(r'$\Delta %s$' %(ylabel), fontsize=fontsize)
                ax25.set_ylabel(r'$\Delta %s$' %(ylabel), fontsize=fontsize)
                ax35.set_ylabel(r'$\Delta %s$' %(ylabel), fontsize=fontsize)
                ax45.set_ylabel(r'$\Delta %s$' %(ylabel), fontsize=fontsize)
            else:
                if usemean == False:
                    if self.labels[0] is None:
                        comp_label = str(0)
                    else:
                        comp_label = str(self.labels[0])
                else:
                    comp_label = '{\\rm Mean}'
                ax15.set_ylabel(r'$\frac{%s}{%s_{%s}}-1$' %(ylabel, ylabel, comp_label), fontsize=fontsize)
                ax25.set_ylabel(r'$\frac{%s}{%s_{%s}}-1$' %(ylabel, ylabel, comp_label), fontsize=fontsize)
                ax35.set_ylabel(r'$\frac{%s}{%s_{%s}}-1$' %(ylabel, ylabel, comp_label), fontsize=fontsize)
                ax45.set_ylabel(r'$\frac{%s}{%s_{%s}}-1$' %(ylabel, ylabel, comp_label), fontsize=fontsize)
        else:
            if subplot_setup == '4x1':
                f, (ax1, ax2, ax3, ax4) = plt.subplots(1, 4, figsize=figsize)
            elif subplot_setup == '2x2':
                f, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=figsize)
            else:
                pass
            ax1.set_xlabel(r'$d$', fontsize=fontsize)
            if units is None:
                ax2.set_xlabel(r'$l$', fontsize=fontsize)
                ax3.set_xlabel(r'$b$', fontsize=fontsize)
            else:
                ax2.set_xlabel(r'$l$ $[%s]$' %(units), fontsize=fontsize)
                ax3.set_xlabel(r'$b$ $[%s]$' %(units), fontsize=fontsize)
            if self.use_sqrt_s == True:
                ax4.set_xlabel(r'$\sqrt{1-s}$', fontsize=fontsize)
            else:
                ax4.set_xlabel(r'$s$', fontsize=fontsize)
        ax1.set_ylabel(r'$%s$' %(ylabel), fontsize=fontsize)
        ax2.set_ylabel(r'$%s$' %(ylabel), fontsize=fontsize)
        ax3.set_ylabel(r'$%s$' %(ylabel), fontsize=fontsize)
        ax4.set_ylabel(r'$%s$' %(ylabel), fontsize=fontsize)
        if self.uselog == True:
            ax2.set_xscale('log')
            ax3.set_xscale('log')
            if usecomp == True and self.num_data > 1:
                ax25.set_xscale('log')
                ax35.set_xscale('log')
        # Main Plots
        if usecomp == True and self.num_data > 1:
            if usemean == False:
                y_d_comp = self.binned_data[whichcomp]['y_d']
                y_l_comp = self.binned_data[whichcomp]['y_l']
                y_b_comp = self.binned_data[whichcomp]['y_b']
                y_s_comp = self.binned_data[whichcomp]['y_s']
            else:
                y_d_all = []
                y_l_all = []
                y_b_all = []
                y_s_all = []
                for i in range(0, len(self.binned_data)):
                    y_d_all.append(self.binned_data[i]['y_d'])
                    y_l_all.append(self.binned_data[i]['y_l'])
                    y_b_all.append(self.binned_data[i]['y_b'])
                    y_s_all.append(self.binned_data[i]['y_s'])
                y_d_all = np.array(y_d_all)
                y_l_all = np.array(y_l_all)
                y_b_all = np.array(y_b_all)
                y_s_all = np.array(y_s_all)
                y_d_comp = np.mean(y_d_all, axis=0)
                y_l_comp = np.mean(y_l_all, axis=0)
                y_b_comp = np.mean(y_b_all, axis=0)
                y_s_comp = np.mean(y_s_all, axis=0)
            if usefraction == True:
                c1 = np.where(y_d_comp != 0.)[0]
                c2 = np.where(y_l_comp != 0.)[0]
                c3 = np.where(y_b_comp != 0.)[0]
                c4 = np.where(y_s_comp != 0.)[0]
        for i in range(0, len(self.binned_data)):
            plot_histogram_line(self.binned_data[i]['x_d'], self.binned_data[i]['y_d'],
                                ax=ax1, color=self.colors[i], alpha=self.alphas[i],
                                linewidth=self.linewidths[i], linestyle=self.linestyles[i])
            if showenvelopes == True and self.need_envelopes[i] == True:
                for sigma in range(0, showsigma):
                    plot_histogram_error(self.binned_data[i]['x_d'], self.binned_data[i]['y_d'], (1+sigma)*self.binned_data[i]['y_d_std'],
                                         ax=ax1, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
            if usebox == True:
                plot_histogram_line(self.binned_data[i]['x_l'], self.binned_data[i]['y_l'], x_edges=self.l_edges,
                                    ax=ax2, color=self.colors[i], alpha=self.alphas[i],
                                    linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                plot_histogram_line(self.binned_data[i]['x_b'], self.binned_data[i]['y_b'], x_edges=self.b_edges,
                                    ax=ax3, color=self.colors[i], alpha=self.alphas[i],
                                    linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                plot_histogram_line(self.binned_data[i]['x_s'], self.binned_data[i]['y_s'],
                                    ax=ax4, color=self.colors[i], alpha=self.alphas[i],
                                    linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                if showenvelopes == True and self.need_envelopes[i] == True:
                    for sigma in range(0, showsigma):
                        plot_histogram_error(self.binned_data[i]['x_l'], self.binned_data[i]['y_l'], (1+sigma)*self.binned_data[i]['y_l_std'],
                                             x_edges=self.l_edges, ax=ax2, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                        plot_histogram_error(self.binned_data[i]['x_b'], self.binned_data[i]['y_b'], (1+sigma)*self.binned_data[i]['y_b_std'],
                                             x_edges=self.b_edges, ax=ax3, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                        plot_histogram_error(self.binned_data[i]['x_s'], self.binned_data[i]['y_s'], (1+sigma)*self.binned_data[i]['y_s_std'],
                                             ax=ax4, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
            else:
                ax2.plot(self.binned_data[i]['x_l'], self.binned_data[i]['y_l'],
                         color=self.colors[i], alpha=self.alphas[i],
                         linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                ax3.plot(self.binned_data[i]['x_b'], self.binned_data[i]['y_b'],
                         color=self.colors[i], alpha=self.alphas[i],
                         linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                ax4.plot(self.binned_data[i]['x_s'], self.binned_data[i]['y_s'],
                         color=self.colors[i], alpha=self.alphas[i],
                         linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                if showenvelopes == True and self.need_envelopes[i] == True:
                    for sigma in range(0, showsigma):
                        ax2.fill_between(self.binned_data[i]['x_l'], self.binned_data[i]['y_l']-(1+sigma)*self.binned_data[i]['y_l_std'],
                                         self.binned_data[i]['y_l']+(1+sigma)*self.binned_data[i]['y_l_std'], color=self.colors[i],
                                         alpha=self.alphas_envelope[i]/(1+sigma))
                        ax3.fill_between(self.binned_data[i]['x_b'], self.binned_data[i]['y_b']-(1+sigma)*self.binned_data[i]['y_b_std'],
                                         self.binned_data[i]['y_b']+(1+sigma)*self.binned_data[i]['y_b_std'], color=self.colors[i],
                                         alpha=self.alphas_envelope[i]/(1+sigma))
                        ax4.fill_between(self.binned_data[i]['x_s'], self.binned_data[i]['y_s']-(1+sigma)*self.binned_data[i]['y_s_std'],
                                         self.binned_data[i]['y_s']+(1+sigma)*self.binned_data[i]['y_s_std'], color=self.colors[i],
                                         alpha=self.alphas_envelope[i]/(1+sigma))
            if usecomp == True and self.num_data > 1:
                if usefraction == False:
                    plot_histogram_line(self.binned_data[i]['x_d'], self.binned_data[i]['y_d']-y_d_comp,
                                        ax=ax15, color=self.colors[i], alpha=self.alphas[i],
                                        linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                    if showenvelopes == True and self.need_envelopes[i] == True:
                        for sigma in range(0, showsigma):
                            plot_histogram_error(self.binned_data[i]['x_d'], self.binned_data[i]['y_d']-y_d_comp, (1+sigma)*self.binned_data[i]['y_d_std'],
                                                 ax=ax15, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                    if usebox == True:
                        plot_histogram_line(self.binned_data[i]['x_l'], self.binned_data[i]['y_l']-y_l_comp,
                                            x_edges=self.l_edges, ax=ax25, color=self.colors[i], alpha=self.alphas[i],
                                            linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        plot_histogram_line(self.binned_data[i]['x_b'], self.binned_data[i]['y_b']-y_b_comp,
                                            x_edges=self.b_edges, ax=ax35, color=self.colors[i], alpha=self.alphas[i],
                                            linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        plot_histogram_line(self.binned_data[i]['x_s'], self.binned_data[i]['y_s']-y_s_comp,
                                            ax=ax45, color=self.colors[i], alpha=self.alphas[i],
                                            linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        if showenvelopes == True and self.need_envelopes[i] == True:
                            for sigma in range(0, showsigma):
                                plot_histogram_error(self.binned_data[i]['x_l'], self.binned_data[i]['y_l']-y_l_comp,
                                                     (1+sigma)*self.binned_data[i]['y_l_std'], x_edges=self.l_edges,
                                                     ax=ax25, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                                plot_histogram_error(self.binned_data[i]['x_b'], self.binned_data[i]['y_b']-y_b_comp,
                                                     (1+sigma)*self.binned_data[i]['y_b_std'], x_edges=self.b_edges,
                                                     ax=ax35, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                                plot_histogram_error(self.binned_data[i]['x_s'], self.binned_data[i]['y_s']-y_s_comp,
                                                     (1+sigma)*self.binned_data[i]['y_s_std'],
                                                     ax=ax45, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                    else:
                        ax25.plot(self.binned_data[i]['x_l'], self.binned_data[i]['y_l']-y_l_comp,
                                  color=self.colors[i], alpha=self.alphas[i],
                                  linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        ax35.plot(self.binned_data[i]['x_b'], self.binned_data[i]['y_b']-y_b_comp,
                                  color=self.colors[i], alpha=self.alphas[i],
                                  linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        ax45.plot(self.binned_data[i]['x_s'], self.binned_data[i]['y_s']-y_s_comp,
                                  color=self.colors[i], alpha=self.alphas[i],
                                  linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        if showenvelopes == True and self.need_envelopes[i] == True:
                            for sigma in range(0, showsigma):
                                ax25.fill_between(self.binned_data[i]['x_l'], self.binned_data[i]['y_l']-(1+sigma)*self.binned_data[i]['y_l_std']-y_l_comp,
                                                  self.binned_data[i]['y_l']+(1+sigma)*self.binned_data[i]['y_l_std']-y_l_comp,
                                                  color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                                ax35.fill_between(self.binned_data[i]['x_b'], self.binned_data[i]['y_b']-(1+sigma)*self.binned_data[i]['y_b_std']-y_b_comp,
                                                  self.binned_data[i]['y_b']+(1+sigma)*self.binned_data[i]['y_b_std']-y_b_comp,
                                                  color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                                ax45.fill_between(self.binned_data[i]['x_s'], self.binned_data[i]['y_s']-(1+sigma)*self.binned_data[i]['y_s_std']-y_s_comp,
                                                  self.binned_data[i]['y_s']+(1+sigma)*self.binned_data[i]['y_s_std']-y_s_comp, color=self.colors[i],
                                                  alpha=self.alphas_envelope[i]/(1+sigma))
                else:
                    plot_histogram_line(self.binned_data[i]['x_d'][c1], self.binned_data[i]['y_d'][c1]/y_d_comp[c1]-1.,
                                        ax=ax15, color=self.colors[i], alpha=self.alphas[i],
                                        linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                    if showenvelopes == True and self.need_envelopes[i] == True:
                        for sigma in range(0, showsigma):
                            plot_histogram_confidence(self.binned_data[i]['x_d'][c1],
                                                      (self.binned_data[i]['y_d'][c1]-(1+sigma)*self.binned_data[i]['y_d_std'][c1])/y_d_comp[c1]-1,
                                                      (self.binned_data[i]['y_d'][c1]+(1+sigma)*self.binned_data[i]['y_d_std'][c1])/y_d_comp[c1]-1,
                                                      ax=ax15, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                    if usebox == True:
                        plot_histogram_line(self.binned_data[i]['x_l'][c2], self.binned_data[i]['y_l'][c2]/y_l_comp[c2]-1.,
                                            x_edges=self.l_edges, ax=ax25, color=self.colors[i], alpha=self.alphas[i],
                                            linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        plot_histogram_line(self.binned_data[i]['x_b'][c3], self.binned_data[i]['y_b'][c3]/y_b_comp[c3]-1.,
                                            x_edges=self.b_edges, ax=ax35, color=self.colors[i], alpha=self.alphas[i],
                                            linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        plot_histogram_line(self.binned_data[i]['x_s'][c4], self.binned_data[i]['y_s'][c4]/y_s_comp[c4]-1,
                                            ax=ax45, color=self.colors[i], alpha=self.alphas[i],
                                            linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        if showenvelopes == True and self.need_envelopes[i] == True:
                            for sigma in range(0, showsigma):
                                plot_histogram_confidence(self.binned_data[i]['x_l'][c2],
                                                          (self.binned_data[i]['y_l'][c2]-(1+sigma)*self.binned_data[i]['y_l_std'][c2])/y_l_comp[c2]-1,
                                                          (self.binned_data[i]['y_l'][c2]+(1+sigma)*self.binned_data[i]['y_l_std'][c2])/y_l_comp[c2]-1,
                                                          x_edges=self.l_edges, ax=ax25, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                                plot_histogram_confidence(self.binned_data[i]['x_b'][c3],
                                                          (self.binned_data[i]['y_b'][c3]-(1+sigma)*self.binned_data[i]['y_b_std'][c3])/y_b_comp[c3]-1,
                                                          (self.binned_data[i]['y_b'][c3]+(1+sigma)*self.binned_data[i]['y_b_std'][c3])/y_b_comp[c3]-1,
                                                          x_edges=self.b_edges, ax=ax35, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                                plot_histogram_confidence(self.binned_data[i]['x_s'][c4],
                                                          (self.binned_data[i]['y_s'][c4]-(1+sigma)*self.binned_data[i]['y_s_std'][c4])/y_s_comp[c4]-1,
                                                          (self.binned_data[i]['y_s'][c4]+(1+sigma)*self.binned_data[i]['y_s_std'][c4])/y_s_comp[c4]-1,
                                                          ax=ax45, color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma))
                    else:
                        ax25.plot(self.binned_data[i]['x_l'][c2], self.binned_data[i]['y_l'][c2]/y_l_comp[c2]-1.,
                                 color=self.colors[i], alpha=self.alphas[i],
                                 linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        ax35.plot(self.binned_data[i]['x_b'][c3], self.binned_data[i]['y_b'][c3]/y_b_comp[c3]-1.,
                                 color=self.colors[i], alpha=self.alphas[i],
                                 linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        ax45.plot(self.binned_data[i]['x_s'][c4], self.binned_data[i]['y_s'][c4]/y_s_comp[c4]-1.,
                                 color=self.colors[i], alpha=self.alphas[i],
                                 linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        if showenvelopes == True and self.need_envelopes[i] == True:
                            for sigma in range(0, showsigma):
                                ax25.fill_between(self.binned_data[i]['x_l'][c2],
                                                  (self.binned_data[i]['y_l'][c2]-(1+sigma)*self.binned_data[i]['y_l_std'][c2])/y_l_comp[c2]-1.,
                                                  (self.binned_data[i]['y_l'][c2]+(1+sigma)*self.binned_data[i]['y_l_std'][c2])/y_l_comp[c2]-1.,
                                                  color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma),
                                                  linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                                ax35.fill_between(self.binned_data[i]['x_b'][c3],
                                                  (self.binned_data[i]['y_b'][c3]-(1+sigma)*self.binned_data[i]['y_b_std'][c3])/y_b_comp[c3]-1.,
                                                  (self.binned_data[i]['y_b'][c3]+(1+sigma)*self.binned_data[i]['y_b_std'][c3])/y_b_comp[c3]-1.,
                                                  color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma),
                                                  linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                                ax45.fill_between(self.binned_data[i]['x_s'][c4],
                                                  (self.binned_data[i]['y_s'][c4]-(1+sigma)*self.binned_data[i]['y_s_std'][c4])/y_s_comp[c4]-1.,
                                                  (self.binned_data[i]['y_s'][c4]+(1+sigma)*self.binned_data[i]['y_s_std'][c4])/y_s_comp[c4]-1.,
                                                  color=self.colors[i], alpha=self.alphas_envelope[i]/(1+sigma),
                                                  linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                if plotzeroline == True:
                    ax15.axhline(0., color='k', linestyle=':', linewidth=1.)
                    ax25.axhline(0., color='k', linestyle=':', linewidth=1.)
                    ax35.axhline(0., color='k', linestyle=':', linewidth=1.)
                    ax45.axhline(0., color='k', linestyle=':', linewidth=1.)
        ax1.set_xlim(self.d_min, self.d_max)
        ax2.set_xlim(self.l_min, self.l_max)
        ax3.set_xlim(self.b_min, self.b_max)
        ax4.set_xlim(self.s_min, self.s_max)
        ax1.set_ylim(0., None)
        ax2.set_ylim(0., None)
        ax3.set_ylim(0., None)
        ax4.set_ylim(0., None)
        if usecomp == True and self.num_data > 1:
            ax1.set_xticks([])
            ax2.set_xticks([])
            ax3.set_xticks([])
            ax4.set_xticks([])
            ax15.set_xlim(self.d_min, self.d_max)
            ax25.set_xlim(self.l_min, self.l_max)
            ax35.set_xlim(self.b_min, self.b_max)
            ax45.set_xlim(self.s_min, self.s_max)
        plt.tight_layout()
        if legend == True:
            lines = []
            labels = []
            for i in range(0, len(self.binned_data)):
                if self.labels[i] is None:
                    pass
                else:
                    if showenvelopes == True and self.need_envelopes[i] == True:
                        line = ax1.fill_between([], [], [], color=self.colors[i], alpha=self.alphas_envelope[i])
                        lines.append(line)
                        labels.append(self.labels[i])
                    else:
                        line, = ax1.plot([], [], color=self.colors[i], alpha=self.alphas[i],
                                         linewidth=self.linewidths[i], linestyle=self.linestyles[i])
                        lines.append(line)
                        labels.append(self.labels[i])
            f.subplots_adjust(top=subplot_adjust_top)
            plt.figlegend(handles=lines, labels=labels, loc='upper center', fontsize=legend_fontsize, ncol=legend_column, frameon=False)
        if saveas is None:
            pass
        else:
            if dpi is None:
                plt.savefig(saveas)
            else:
                plt.savefig(saveas, dpi=dpi)
        if plt_output == 'show':
            plt.show()
        elif plt_output == 'close':
            plt.close()
        else:
            pass

    def clean(self):
        """Resets PlotHistMST variables.
        """
        self.__init__()
