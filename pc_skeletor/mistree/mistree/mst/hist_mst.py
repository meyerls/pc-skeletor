import numpy as np
import matplotlib.pylab as plt


def bin_data(data, minimum=None, maximum=None, bin_size=None, bin_number=100, normalised=True):
    """Returns the (normalised) number count of a data set with values within defined bins.

    Parameters
    ----------
    data : array_like
        The data to be binned.
    minimum : float
        A given minimum for the bin edges.
    maximum : float
        A given maximum for the bin edges
    bin_size : float
        The size of the bins.
    bin_number : int
        The number of bins to be used.
    normalised : bool
        Tf true will return a normalised histogram, if false returns a number counts.

    Returns
    -------
    bin_centres : array_like
        The central value of each bin.
    binned_data : array_like
        The binned number count (or normalised histogram) of the data set.
    """
    if minimum is None:
        minimum = np.min(data)
    if maximum is None:
        maximum = np.max(data)
    if bin_size is None:
        _bin_edge = np.linspace(minimum, maximum, bin_number+1)
    if bin_size is not None:
        _bin_edge = np.arange(minimum, maximum+bin_size, bin_size)
        condition = np.where(_bin_edge <= maximum)[0]
        _bin_edge = _bin_edge[condition]
    binned_data, _bin_edges = np.histogram(data, bins=_bin_edge, density=normalised)
    bin_centres = 0.5 * (_bin_edge[1:] + _bin_edge[:-1])
    return bin_centres, binned_data


class HistMST:

    def __init__(self):
        """Initialises MST histogram class.
        """
        # bin edges information
        self.uselog = None
        self.use_s_sqrt = None
        self.usenorm = None
        self.d_min = None
        self.d_max = None
        self.num_d_bins = None
        self.l_min = None
        self.l_max = None
        self.num_l_bins = None
        self.b_min = None
        self.b_max = None
        self.num_b_bins = None
        self.s_min = None
        self.s_max = None
        self.num_s_bins = None
        self.logl_min = None
        self.logl_max = None
        self.logb_min = None
        self.logb_max = None
        self.group_mode = False

    def setup(self, uselog=False, use_sqrt_s=True, usenorm=True, d_min=0.5, d_max=6.5, num_d_bins=6,
              l_min=0., l_max=None, num_l_bins=100, b_min=0., b_max=None, num_b_bins=100,
              s_min=0., s_max=1., num_s_bins=50, logl_min=None, logl_max=None, logb_min=None, logb_max=None):
        """Setups bin sizes for the MST statistics.

        Parameters
        ----------
        uselog : bool
            Determines whether to use log bins for l and b.
        use_sqrt_s : bool
            Determines whether to use the sqrt(1-s) projection of s or just s itself.
        usenorm : bool
            Determines whether to normalise the histograms.
        d_min : float
            Minimum for degree bins (use half integer values).
        d_max : float
            Maximum for degree bins (use half integer values).
        num_d_bins : int
            Number of bins for the distribution of degree, this should be equal to d_max - d_min.
        l_min : float
            Minimum for edge length bins.
        l_max : float
            Maximum for edge length bins.
        num_l_bins : int
            Number of bins for the distribution of edge lengths.
        b_min : float
            Minimum for branch length bins.
        b_max : float
            Maximum for branch length bins.
        num_b_bins : int
            Number of bins for the distribution of branch lengths.
        s_min : float
            Minimum for branch shape bins.
        s_max : float
            Maximum for branch shape bins.
        num_s_bins : int
            Number of bins for the distribution of branch shapes.
        logl_min : float
            Minimum of edge lengths in log to base 10.
        logl_max : float
            Maximum of edge lengths in log to base 10.
        logb_min : float
            Minimum of branch lengths in log to base 10.
        logb_max : float
            Maximum of branch lengths in log to base 10.
        """
        self.uselog = uselog
        self.use_sqrt_s= use_sqrt_s
        self.usenorm = usenorm
        self.d_min = d_min
        self.d_max = d_max
        self.num_d_bins = num_d_bins
        self.l_min = l_min
        self.l_max = l_max
        self.num_l_bins = num_l_bins
        self.b_min = b_min
        self.b_max = b_max
        self.num_b_bins = num_b_bins
        self.s_min = s_min
        self.s_max = s_max
        self.num_s_bins = num_s_bins
        self.logl_min = logl_min
        self.logl_max = logl_max
        self.logb_min = logb_min
        self.logb_max = logb_max

    def get_hist(self, d, l, b, s):
        """Bins the MST distribution which is returned as a dictionary.

        Parameters
        ----------
        d : array_like
            Distribution of degree.
        l : array_like
            Distribution of edge length.
        b : array_like
            Distribution of branch length.
        s : array_like
            Distribution of branch shape.

        Returns
        -------
        mst_hist : dict
            Dictionary of MST binned histograms.
        """
        # find minimum and maximum
        if self.l_max is None:
            self.l_max = 1.05*l.max()
        if self.b_max is None:
            self.b_max = 1.05*b.max()
        if self.logl_min is None:
            self.logl_min = np.log10(0.95*l.min())
        if self.logl_max is None:
            self.logl_max = np.log10(1.05*l.max())
        if self.logb_min is None:
            self.logb_min = np.log10(0.95*b.min())
        if self.logb_max is None:
            self.logb_max = np.log10(1.05*b.max())
        # bin mst statistics
        x_d, y_d = bin_data(d, minimum=self.d_min, maximum=self.d_max, bin_number=self.num_d_bins, normalised=self.usenorm)
        if self.uselog == False:
            x_l, y_l = bin_data(l, minimum=self.l_min, maximum=self.l_max, bin_number=self.num_l_bins, normalised=self.usenorm)
            x_b, y_b = bin_data(b, minimum=self.b_min, maximum=self.b_max, bin_number=self.num_b_bins, normalised=self.usenorm)
        else:
            x_logl, y_l = bin_data(np.log10(l), minimum=self.logl_min, maximum=self.logl_max, bin_number=self.num_l_bins, normalised=self.usenorm)
            x_logb, y_b = bin_data(np.log10(b), minimum=self.logb_min, maximum=self.logb_max, bin_number=self.num_b_bins, normalised=self.usenorm)
            x_l = 10.**x_logl
            x_b = 10.**x_logb
        if self.use_sqrt_s == False:
            x_s, y_s = bin_data(s, minimum=self.s_min, maximum=self.s_max, bin_number=self.num_s_bins, normalised=self.usenorm)
        else:
            x_s, y_s = bin_data(np.sqrt(1.-s), minimum=self.s_min, maximum=self.s_max, bin_number=self.num_s_bins, normalised=self.usenorm)
        mst_hist = {
            "uselog" : self.uselog, "use_sqrt_s" : self.use_sqrt_s, "usenorm" : self.usenorm, "isgroup" : False,
            "x_d" : x_d, "y_d" : y_d, "x_l" : x_l, "y_l" : y_l, "x_b" : x_b, "y_b" : y_b, "x_s" : x_s, "y_s" : y_s
        }
        if self.group_mode == True:
            self.x_d = mst_hist['x_d']
            self.x_l = mst_hist['x_l']
            self.x_b = mst_hist['x_b']
            self.x_s = mst_hist['x_s']
            self.group_y_d.append(mst_hist['y_d'])
            self.group_y_l.append(mst_hist['y_l'])
            self.group_y_b.append(mst_hist['y_b'])
            self.group_y_s.append(mst_hist['y_s'])
        return mst_hist

    def start_group(self):
        """Begins group mode for calculating the mean and standard deviation of the MST
        from different realisations of data points coming from the same model.
        """
        self.group_mode = True
        self.group_y_d = []
        self.group_y_l = []
        self.group_y_b = []
        self.group_y_s = []

    def end_group(self):
        """Ends group mode for calculating the mean and standard deviation of the MST.

        Returns
        -------
        mst_hist : dict
            Dictionary of the mean and standard deviation of the MST binned histograms.
        """
        self.group_mode = False
        self.group_y_d = np.array(self.group_y_d)
        self.group_y_l = np.array(self.group_y_l)
        self.group_y_b = np.array(self.group_y_b)
        self.group_y_s = np.array(self.group_y_s)
        y_d_mean = np.mean(self.group_y_d, axis=0)
        y_l_mean = np.mean(self.group_y_l, axis=0)
        y_b_mean = np.mean(self.group_y_b, axis=0)
        y_s_mean = np.mean(self.group_y_s, axis=0)
        y_d_std = np.std(self.group_y_d, axis=0)
        y_l_std = np.std(self.group_y_l, axis=0)
        y_b_std = np.std(self.group_y_b, axis=0)
        y_s_std = np.std(self.group_y_s, axis=0)
        mst_hist = {
            "uselog" : self.uselog, "use_sqrt_s" : self.use_sqrt_s,
            "usenorm" : self.usenorm, "isgroup" : True,
            "x_d" : self.x_d, "y_d" : y_d_mean, "y_d_std" : y_d_std,
            "x_l" : self.x_l, "y_l" : y_l_mean, "y_l_std" : y_l_std,
            "x_b" : self.x_b, "y_b" : y_b_mean, "y_b_std" : y_b_std,
            "x_s" : self.x_s, "y_s" : y_s_mean, "y_s_std" : y_s_std,
        }
        return mst_hist

    def clean(self):
        """Resets HistMST variables.
        """
        self.__init__()
