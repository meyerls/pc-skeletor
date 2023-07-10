import numpy as np
import mistree as mist


def test_PlotHistMST_rotate():
    pmst = mist.PlotHistMST()
    pmst._get_rotate_colors()
    assert pmst.rotate_colors == 1
    pmst._get_rotate_linestyles()
    assert pmst.rotate_linestyle == 1


def test_PlotHistMST_plot():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup()
    mst_dict = hmst.get_hist(d, l, b, s)
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(plt_output='close')


def test_PlotHistMST_plot_usebox():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup()
    mst_dict = hmst.get_hist(d, l, b, s)
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(usebox=False, plt_output='close')


def test_PlotHistMST_plot_contour_extra_option():
    hmst = mist.HistMST()
    hmst.setup()
    hmst.start_group()
    for i in range(0, 10):
        x = np.random.random_sample(100)
        y = np.random.random_sample(100)
        mst = mist.GetMST(x=x, y=y)
        d, l, b, s = mst.get_stats()
        mst_dict = hmst.get_hist(d, l, b, s)
    mst_dict = hmst.end_group()
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(plt_output='close')
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(usebox=False, plt_output='close')
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(xlabels=['a', 'b', 'c', 'd'], plt_output='close')
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(usefraction=True, plt_output='close')
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(usefraction=True, usemean=False, plt_output='close')
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_dict)
    pmst.plot(units=r'^{\circ}', plt_output='close')


def test_PlotHistMST_plot_comparison():
    x_lf, y_lf, z_lf = mist.get_levy_flight(5000)
    x_alf, y_alf, z_alf = mist.get_adjusted_levy_flight(5000)
    mst = mist.GetMST(x=x_lf, y=y_lf, z=z_lf)
    d_lf, l_lf, b_lf, s_lf = mst.get_stats()
    mst = mist.GetMST(x=x_alf, y=y_alf, z=z_alf)
    d_alf, l_alf, b_alf, s_alf = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup(uselog=True)
    hist_lf = hmst.get_hist(d_lf, l_lf, b_lf, s_lf)
    x_alf, y_alf, z_alf = mist.get_adjusted_levy_flight(5000)
    mst = mist.GetMST(x=x_alf, y=y_alf, z=z_alf)
    d_alf, l_alf, b_alf, s_alf = mst.get_stats()
    hist_alf = hmst.get_hist(d_alf, l_alf, b_alf, s_alf)
    pmst = mist.PlotHistMST()
    pmst.read_mst(hist_lf, label='Levy Flight')
    pmst.read_mst(hist_alf, label='Adjusted Levy Flight')
    pmst.plot(usecomp=True, plt_output='close')
    pmst = mist.PlotHistMST()
    pmst.read_mst(hist_lf, label='Levy Flight')
    pmst.read_mst(hist_alf, label='Adjusted Levy Flight')
    pmst.plot(usebox=False, usecomp=True, plt_output='close')


def test_PlotHistMST_plot_comparison_envelope():
    x_lf, y_lf, z_lf = mist.get_levy_flight(5000)
    mst = mist.GetMST(x=x_lf, y=y_lf, z=z_lf)
    d_lf, l_lf, b_lf, s_lf = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup(uselog=True)
    hist_lf = hmst.get_hist(d_lf, l_lf, b_lf, s_lf)
    hmst.start_group()
    for i in range(0, 10):
        x_alf, y_alf, z_alf = mist.get_adjusted_levy_flight(5000)
        mst = mist.GetMST(x=x_alf, y=y_alf, z=z_alf)
        d_alf, l_alf, b_alf, s_alf = mst.get_stats()
        _hist_alf = hmst.get_hist(d_alf, l_alf, b_alf, s_alf)
    hist_alf_group = hmst.end_group()
    pmst = mist.PlotHistMST()
    pmst.read_mst(hist_lf, label='Levy Flight')
    pmst.read_mst(hist_alf_group, label='Adjusted Levy Flight')
    pmst.plot(usecomp=True, plt_output='close')


def test_PlotHistMST_read_mst():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup()
    mst_hist = hmst.get_hist(d, l, b, s)
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_hist, color='dodgerblue', linewidth=1., linestyle=':', alpha=0.5,
                  label='check', alpha_envelope=0.5)
    assert pmst.colors[0] == 'dodgerblue'
    assert pmst.linewidths[0] == 1.
    assert pmst.linestyles[0] == ':'
    assert pmst.alphas[0] == 0.5
    assert pmst.labels[0] == 'check'
    assert pmst.alphas_envelope[0] == 0.5
    assert pmst.need_envelopes[0] == False
    assert pmst.use_sqrt_s == True


def test_PlotHistMST_read_mst_uselog():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup(uselog=True)
    mst_hist = hmst.get_hist(d, l, b, s)
    pmst = mist.PlotHistMST()
    pmst.read_mst(mst_hist, color='dodgerblue', linewidth=1., linestyle=':', alpha=0.5,
                  label='check', alpha_envelope=0.5)
    assert pmst.colors[0] == 'dodgerblue'
    assert pmst.linewidths[0] == 1.
    assert pmst.linestyles[0] == ':'
    assert pmst.alphas[0] == 0.5
    assert pmst.labels[0] == 'check'
    assert pmst.alphas_envelope[0] == 0.5
    assert pmst.need_envelopes[0] == False
    assert pmst.use_sqrt_s == True
