import numpy as np
import mistree as mist


def test_GetMST_init():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    assert mst._mode == '2D'
    z = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y, z=z)
    assert mst._mode == '3D'
    mst = mist.GetMST(phi=x, theta=y)
    assert mst._mode == 'tomographic'
    mst = mist.GetMST(phi=x, theta=y, r=z)
    assert mst._mode == 'spherical polar'
    mst = mist.GetMST(ra=x, dec=y)
    assert mst._mode == 'tomographic celestial'
    mst = mist.GetMST(ra=x, dec=y, r=z)
    assert mst._mode == 'spherical polar celestial'


def test_GetMST_knn():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    mst.define_k_neighbours(10)
    assert mst.k_neighbours == 10


def test_GetMST_scal_cut():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    mst.scale_cut(0.2)
    assert mst.scale_cut_length == 0.2


def test_GetMST_construct_mst_2d():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is None
    assert mst.phi is None
    assert mst.theta is None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is None
    assert mst.edge_phi is None
    assert mst.edge_theta is None
    assert mst.edge_index is not None


def test_GetMST_construct_mst_2d_scale_cut():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    mst.scale_cut(0.2)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is None
    assert mst.phi is None
    assert mst.theta is None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is None
    assert mst.edge_phi is None
    assert mst.edge_theta is None
    assert mst.edge_index is not None


def test_GetMST_construct_mst_3d():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    z = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is not None
    assert mst.phi is None
    assert mst.theta is None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is not None
    assert mst.edge_phi is None
    assert mst.edge_theta is None
    assert mst.edge_index is not None


def test_GetMST_construct_mst_tomo():
    phi = 360.*np.random.random_sample(100)
    theta = 180.*np.random.random_sample(100)
    mst = mist.GetMST(phi=phi, theta=theta)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is not None
    assert mst.phi is not None
    assert mst.theta is not None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is not None
    assert mst.edge_phi is not None
    assert mst.edge_theta is not None
    assert mst.edge_index is not None


def test_GetMST_construct_mst_spherical():
    r = np.random.random_sample(100)
    phi = 360.*np.random.random_sample(100)
    theta = 180.*np.random.random_sample(100)
    mst = mist.GetMST(phi=phi, theta=theta, r=r)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is not None
    assert mst.phi is not None
    assert mst.theta is not None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is not None
    assert mst.edge_phi is not None
    assert mst.edge_theta is not None
    assert mst.edge_index is not None


def test_GetMST_construct_mst_celestial_tomo():
    ra = 360.*np.random.random_sample(100)
    dec = 180.*np.random.random_sample(100) - 90.
    mst = mist.GetMST(ra=ra, dec=dec)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is not None
    assert mst.phi is not None
    assert mst.theta is not None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is not None
    assert mst.edge_phi is not None
    assert mst.edge_theta is not None
    assert mst.edge_index is not None


def test_GetMST_construct_mst_celestial_spherical():
    r = np.random.random_sample(100)
    ra = 360.*np.random.random_sample(100) - 90.
    dec = 180.*np.random.random_sample(100)
    mst = mist.GetMST(ra=ra, dec=dec, r=r)
    mst.construct_mst()
    assert mst.x is not None
    assert mst.y is not None
    assert mst.z is not None
    assert mst.phi is not None
    assert mst.theta is not None
    assert mst.edge_length is not None
    assert mst.edge_x is not None
    assert mst.edge_y is not None
    assert mst.edge_z is not None
    assert mst.edge_phi is not None
    assert mst.edge_theta is not None
    assert mst.edge_index is not None


def test_GetMST_get_degree():
    mst = mist.GetMST()
    mst.edge_index = np.array([[0, 0, 4, 4, 2], [1, 2, 2, 3, 3]])
    mst.x = np.random.random_sample(5)
    mst.get_degree()
    degree2 = np.array([2., 1., 3., 2., 2.])
    condition = np.where(mst.degree == degree2)[0]
    assert len(condition) == 5


def test_GetMST_get_mean_degree_for_edges():
    mst = mist.GetMST()
    mst.edge_index = np.array([[0, 0, 4, 4, 2], [1, 2, 2, 3, 3]])
    mst.x = np.random.random_sample(5)
    mst.get_degree()
    mst.get_mean_degree_for_edges()
    mean_degree2 = np.array([1.5, 2.5, 2.5, 2., 2.5])
    condition = np.where(mst.mean_degree == mean_degree2)[0]
    assert len(condition) == 5


def test_GetMST_get_degree_for_edges():
    mst = mist.GetMST()
    mst.edge_index = np.array([[0, 0, 4, 4, 2], [1, 2, 2, 3, 3]])
    mst.x = np.random.random_sample(5)
    mst.get_degree()
    edge_degree = mst.get_degree_for_edges()
    edge_degree2 = np.array([[2., 2., 2., 2., 3.], [1., 3., 3., 2., 2.]])
    condition = np.where((mst.edge_degree[0] == edge_degree2[0]) &
                         (mst.edge_degree[1] == edge_degree2[1]))[0]
    assert len(condition) == 5


def test_GetMST_get_branches_2d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches(box_size=None, sub_divisions=1)
    b1 = np.copy(mst.branch_length)
    mst.get_branches(box_size=None, sub_divisions=2)
    b2 = np.copy(mst.branch_length)
    condition = np.where(np.sort(b1) == np.sort(b2))[0]
    assert len(condition) == len(b1)



def test_GetMST_get_branches_3d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches(box_size=None, sub_divisions=1)
    b1 = np.copy(mst.branch_length)
    mst.get_branches(box_size=None, sub_divisions=2)
    b2 = np.copy(mst.branch_length)
    condition = np.where(np.sort(b1) == np.sort(b2))[0]
    assert len(condition) == len(b1)


def test_GetMST_get_branches_2d_box():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches(box_size=None, sub_divisions=1)
    b1 = np.copy(mst.branch_length)
    mst.get_branches(box_size=1, sub_divisions=1)
    b2 = np.copy(mst.branch_length)
    condition = np.where(np.sort(b1) == np.sort(b2))[0]
    assert len(condition) == len(b1)


def test_GetMST_get_branches_3d_box():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches(box_size=None, sub_divisions=1)
    b1 = np.copy(mst.branch_length)
    mst.get_branches(box_size=1, sub_divisions=1)
    b2 = np.copy(mst.branch_length)
    condition = np.where(np.sort(b1) == np.sort(b2))[0]
    assert len(condition) == len(b1)


def test_GetMST_get_branches_3d_box():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches(box_size=None, sub_divisions=1)
    b1 = np.copy(mst.branch_length)
    mst.get_branches(box_size=1, sub_divisions=1)
    b2 = np.copy(mst.branch_length)
    condition = np.where(np.sort(b1) == np.sort(b2))[0]
    assert len(condition) == len(b1)


def test_GetMST_branch_edge_count():
    mst = mist.GetMST()
    mst.branch_index = [[1, 2, 3], [1, 2], [3, 4, 5, 6, 6]]
    mst.get_branch_edge_count()
    condition = np.where(mst.branch_edge_count == np.array([3, 2, 5]))[0]
    assert len(condition) == 3


def test_GetMST_get_branch_shape():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches(box_size=None, sub_divisions=1)
    mst.get_branch_shape()
    assert len(mst.branch_length) == len(mst.branch_shape)


def test_GetMST_get_stats_vs_density_2d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    dx = 0.1
    box_size = 1.
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    output = mst.get_stats_vs_density(dx, box_size)
    assert len(output) == 5


def test_GetMST_get_stats_vs_density_3d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    dx = 0.1
    box_size = 1.
    mst = mist.GetMST(x=x, y=y, z=z)
    d, l, b, s = mst.get_stats()
    output = mst.get_stats_vs_density(dx, box_size)
    assert len(output) == 5


def test_GetMST_output_stats_no_index():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    dx = 0.1
    box_size = 1.
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    output = mst.output_stats()
    assert len(output) == 4


def test_GetMST_output_stats_index():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    dx = 0.1
    box_size = 1.
    mst = mist.GetMST(x=x, y=y)
    d, l, b, s = mst.get_stats()
    output = mst.output_stats(include_index=True)
    assert len(output) == 6


def test_GetMST_get_stats_2D():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y)
    mst.get_stats(partitions=2)
    mst.get_stats(partitions=2, include_index=True)
    mst.clean()


def test_GetMST_get_stats_3D():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    z = np.random.random_sample(100)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.get_stats(partitions=2)
    mst.get_stats(partitions=2, include_index=True)
    mst.clean()


def test_GetMST_get_stats_tomo():
    phi = 360.*np.random.random_sample(100)
    theta = 180.*np.random.random_sample(100)
    mst = mist.GetMST(phi=phi, theta=theta)
    mst.get_stats(partitions=2)
    mst.get_stats(partitions=2, include_index=True)
    mst.clean()


def test_GetMST_get_stats_spherical():
    r = np.random.random_sample(100)
    phi = 360.*np.random.random_sample(100)
    theta = 180.*np.random.random_sample(100)
    mst = mist.GetMST(r=r, phi=phi, theta=theta)
    mst.get_stats(partitions=2)
    mst.get_stats(partitions=2, include_index=True)
    mst.clean()


def test_GetMST_get_stats_tomo_celestial():
    ra = 360.*np.random.random_sample(100)
    dec = 180.*np.random.random_sample(100) - 90.
    mst = mist.GetMST(ra=ra, dec=dec)
    mst.get_stats(partitions=2)
    mst.get_stats(partitions=2, include_index=True)
    mst.clean()


def test_GetMST_get_stats_spherical_celestial():
    r = np.random.random_sample(100)
    ra = 360.*np.random.random_sample(100)
    dec = 180.*np.random.random_sample(100) - 90.
    mst = mist.GetMST(r=r, ra=ra, dec=dec)
    mst.get_stats(partitions=2)
    mst.get_stats(partitions=2, include_index=True)
    mst.clean()
