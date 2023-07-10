import numpy as np
import mistree as mist


def test_bin_data_bin_no_normalise():
    data = np.random.random_sample(1000)
    bin_x, bin_y = mist.bin_data(data, minimum=0., maximum=1., bin_size=None, bin_number=100, normalised=False)
    assert np.sum(bin_y) == 1000


def test_bin_data_bin_normalise():
    data = np.random.random_sample(1000)
    bin_x, bin_y = mist.bin_data(data, minimum=0., maximum=1., bin_size=None, bin_number=100, normalised=True)
    assert round(np.sum(bin_y)/100., 1) == 1.


def test_bin_data_bin_dx():
    data = np.random.random_sample(1000)
    bin_x, bin_y = mist.bin_data(data, minimum=0., maximum=1., bin_size=None, bin_number=100, normalised=True)
    bin_x1, bin_y1 = mist.bin_data(data, minimum=0., maximum=1., bin_size=1./100.,  bin_number=100, normalised=True)
    condition = np.where(bin_y == bin_y1)[0]
    assert len(condition) == 100


def test_HistMST_setup():
    hmst = mist.HistMST()
    hmst.setup(uselog=True)
    assert hmst.uselog == True
    hmst.setup(use_sqrt_s=False)
    assert hmst.use_sqrt_s == False
    hmst.setup(usenorm=False)
    assert hmst.usenorm == False
    hmst.setup(d_min = 1.)
    assert hmst.d_min == 1.
    hmst.setup(d_max = 6.)
    assert hmst.d_max == 6.
    hmst.setup(num_d_bins=7)
    assert hmst.num_d_bins == 7
    hmst.setup(l_min = 1.)
    assert hmst.l_min == 1.
    hmst.setup(l_max = 6.)
    assert hmst.l_max == 6.
    hmst.setup(num_l_bins=7)
    assert hmst.num_l_bins == 7
    hmst.setup(b_min = 1.)
    assert hmst.b_min == 1.
    hmst.setup(b_max = 6.)
    assert hmst.b_max == 6.
    hmst.setup(num_b_bins=7)
    assert hmst.num_b_bins == 7
    hmst.setup(s_min = 1.)
    assert hmst.s_min == 1.
    hmst.setup(s_max = 6.)
    assert hmst.s_max == 6.
    hmst.setup(num_s_bins=7)
    assert hmst.num_s_bins == 7


def test_get_hist():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    hmst = mist.HistMST()
    hmst.setup()
    mst_hist = hmst.get_hist(d, l, b, s)
    assert mst_hist['isgroup'] == False
    assert len(mst_hist['x_d']) == 6
    assert len(mst_hist['y_d']) == 6
    assert len(mst_hist['x_l']) == 100
    assert len(mst_hist['y_l']) == 100
    assert len(mst_hist['x_b']) == 100
    assert len(mst_hist['y_b']) == 100
    assert len(mst_hist['x_s']) == 50
    assert len(mst_hist['y_s']) == 50


def test_start_group():
    hmst = mist.HistMST()
    hmst.setup()
    hmst.start_group()
    assert hmst.group_mode == True


def test_end_group():
    hmst = mist.HistMST()
    hmst.setup()
    hmst.start_group()
    for i in range(0, 10):
        x = np.random.random_sample(100)
        y = np.random.random_sample(100)
        mst = mist.GetMST(x=x, y=y)
        d, l, b, s = mst.get_stats()
        _mst_hist = hmst.get_hist(d, l, b, s)
    mst_hist = hmst.end_group()
    assert hmst.group_mode == False
    assert mst_hist['isgroup'] == True
    assert len(mst_hist['y_d_std']) == 6
    assert len(mst_hist['y_l_std']) == 100
    assert len(mst_hist['y_b_std']) == 100
    assert len(mst_hist['y_s_std']) == 50
