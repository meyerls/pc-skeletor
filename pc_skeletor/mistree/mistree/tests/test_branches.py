import numpy as np
import mistree as mist


def test_get_branch_index():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rejected = mist.get_branch_index(mst.edge_index, mst.edge_degree)
    assert len(branch_index_rejected) == 0
    x = np.random.random_sample(500)
    y = np.random.random_sample(500)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rejected = mist.get_branch_index(mst.edge_index, mst.edge_degree, branch_cutting_frequency=10)
    assert len(branch_index_rejected) == 0


def test_get_branch_index_sub_divide_2d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rej = mist.get_branch_index(mst.edge_index, mst.edge_degree)
    branch_index_sub, branch_index_sub_rej = mist.get_branch_index_sub_divide(2, mst.edge_index, mst.edge_degree,
                                                                              box_size=1., edge_x=mst.edge_x, edge_y=mst.edge_y)
    assert len(branch_index_sub_rej) == 0
    count = 0
    for i in range(0, len(branch_index)):
        for j in range(0, len(branch_index_sub)):
            if np.array_equal(np.array(sorted(branch_index[i])), np.array(sorted(branch_index_sub[j]))) == True:
                count += 1
            else:
                pass
    assert count == len(branch_index)
    x = np.random.random_sample(500)
    y = np.random.random_sample(500)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index_sub, branch_index_sub_rej = mist.get_branch_index_sub_divide(2, mst.edge_index, mst.edge_degree,
                                                                              box_size=1., edge_x=mst.edge_x, edge_y=mst.edge_y,
                                                                              branch_cutting_frequency=10)



def test_get_branch_index_sub_divide_3d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rej = mist.get_branch_index(mst.edge_index, mst.edge_degree)
    branch_index_sub, branch_index_sub_rej = mist.get_branch_index_sub_divide(2, mst.edge_index, mst.edge_degree,
                                                                              box_size=1., edge_x=mst.edge_x, edge_y=mst.edge_y, edge_z=mst.edge_z,
                                                                              two_dimension=False)
    assert len(branch_index_sub_rej) == 0
    count = 0
    for i in range(0, len(branch_index)):
        for j in range(0, len(branch_index_sub)):
            if np.array_equal(np.array(sorted(branch_index[i])), np.array(sorted(branch_index_sub[j]))) == True:
                count += 1
            else:
                pass
    assert count == len(branch_index)


def test_get_branch_index_sub_divide_3d_no_box_size():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rej = mist.get_branch_index(mst.edge_index, mst.edge_degree)
    branch_index_sub, branch_index_sub_rej = mist.get_branch_index_sub_divide(2, mst.edge_index, mst.edge_degree,
                                                                              edge_x=mst.edge_x, edge_y=mst.edge_y, edge_z=mst.edge_z)
    assert len(branch_index_sub_rej) == 0
    count = 0
    for i in range(0, len(branch_index)):
        for j in range(0, len(branch_index_sub)):
            if np.array_equal(np.array(sorted(branch_index[i])), np.array(sorted(branch_index_sub[j]))) == True:
                count += 1
            else:
                pass
    assert count == len(branch_index)


def test_get_branch_index_sub_divide_spherical_coords():
    phi = 2.*np.pi*np.random.random_sample(50)
    theta = np.pi*np.random.random_sample(50)
    mst = mist.GetMST(phi=phi, theta=theta)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rej = mist.get_branch_index(mst.edge_index, mst.edge_degree)
    branch_index_sub, branch_index_sub_rej = mist.get_branch_index_sub_divide(2, mst.edge_index, mst.edge_degree, phi=phi, theta=theta,
                                                                              mode='spherical', edge_phi=mst.edge_phi, edge_theta=mst.edge_theta, two_dimension=False)
    assert len(branch_index_sub_rej) == 0
    count = 0
    for i in range(0, len(branch_index)):
        for j in range(0, len(branch_index_sub)):
            if np.array_equal(np.array(sorted(branch_index[i])), np.array(sorted(branch_index_sub[j]))) == True:
                count += 1
            else:
                pass
    assert count == len(branch_index)


def test_get_branch_end_index():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    branch_index, branch_index_rej = mist.get_branch_index(mst.edge_index, mst.edge_degree)
    branch_end = mist.get_branch_end_index(mst.edge_index, mst.edge_degree, branch_index)
    condition = np.where((mst.degree[branch_end[0]] != 2.) & (mst.degree[branch_end[1]] != 2.))[0]
    assert len(condition) == len(branch_end[0])


def test_get_branch_edge_count():
    branch_index = [[1], [2, 2], [3, 3, 3], [5, 5, 5, 5, 5]]
    branch_edge_count = mist.get_branch_edge_count(branch_index)
    condition = np.where(np.array([1, 2, 3, 5]) == np.array(branch_edge_count))[0]
    assert len(condition) == 4


def test_get_branch_shape_2d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches()
    branch_shape = mist.get_branch_shape(mst.edge_index, mst.edge_degree, mst.branch_index,
                                         mst.branch_length, mode='2D', x=x, y=y, z=None)
    assert len(mst.branch_length) == len(branch_shape)


def test_get_branch_shape_3d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y, z=z)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches()
    branch_shape = mist.get_branch_shape(mst.edge_index, mst.edge_degree, mst.branch_index,
                                         mst.branch_length, mode='3D', x=x, y=y, z=z)
    assert len(mst.branch_length) == len(branch_shape)


def test_get_branch_shape_tomo():
    phi = np.random.random_sample(50)
    theta = np.random.random_sample(50)
    mst = mist.GetMST(phi=phi, theta=theta)
    mst.construct_mst()
    mst.get_degree()
    mst.get_degree_for_edges()
    mst.get_branches()
    branch_shape = mist.get_branch_shape(mst.edge_index, mst.edge_degree, mst.branch_index,
                                         mst.branch_length, mode='tomographic', x=mst.x, y=mst.y, z=mst.z)
    assert len(mst.branch_length) == len(branch_shape)
    condition = np.where(mst.branch_length <= np.pi)[0]
    assert len(condition) == len(mst.branch_length)
