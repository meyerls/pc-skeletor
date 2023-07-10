import numpy as np
import mistree as mist


def test_construct_mst_2d():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    edge_length, edge_x, edge_y, edge_index = mist.construct_mst(x, y, two_dimensions=True)
    assert len(edge_length) == 99


def test_construct_mst_3d():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    z = np.random.random_sample(100)
    edge_length, edge_x, edge_y, edge_z, edge_index = mist.construct_mst(x, y, z=z, two_dimensions=False)
    assert len(edge_length) == 99


def test_construct_mst_k_neighbours():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    edge_length, edge_x, edge_y, edge_index = mist.construct_mst(x, y, two_dimensions=True)
    edge_length2, edge_x2, edge_2, edge_index2 = mist.construct_mst(x, y, k_neighbours=30, two_dimensions=True)
    condition = np.where(np.sort(edge_length) == np.sort(edge_length2))[0]
    assert len(condition) == len(edge_length)


def test_construct_mst_2d_scale_cut():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    edge_length, edge_x, edge_y, edge_index, num_removed_edges = mist.construct_mst(x, y, two_dimensions=True, scale_cut_length=0.01)
    condition = np.where(edge_length >= 0.01)[0]
    assert len(condition) == len(edge_length)


def test_construct_mst_3d_scale_cut():
    x = np.random.random_sample(100)
    y = np.random.random_sample(100)
    z = np.random.random_sample(100)
    edge_length, edge_x, edge_y, edge_z, edge_index, num_removed_edges = mist.construct_mst(x, y, z=z, two_dimensions=False, scale_cut_length=0.01)
    condition = np.where(edge_length >= 0.01)[0]
    assert len(condition) == len(edge_length)


def test_construct_mst_tomo_scale_cut():
    phi = 360.*np.random.random_sample(100)
    theta = 180.*np.random.random_sample(100)
    x, y, z = mist.spherical_2_unit_sphere(phi, theta)
    edge_length, edge_x, edge_y, edge_z, edge_index, num_removed_edges = mist.construct_mst(x, y, z=z, two_dimensions=False, scale_cut_length=0.2)
    condition = np.where(edge_length >= 0.2)[0]
    assert len(condition) == len(edge_length)
