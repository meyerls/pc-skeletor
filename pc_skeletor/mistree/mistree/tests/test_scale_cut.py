import numpy as np
import mistree as mist


def test_graph_scale_cut():
    ind1 = np.array([0, 1, 2, 3])
    ind2 = np.array([1, 2, 3, 0])
    edge_length = np.array([1., 2., 3., 4.])
    graph = mist.data2graph(ind1, ind2, edge_length, 4)
    scale_cut_length = 2.5
    num_nodes = 4
    graph_cut, index1, index2, num_removed_edges_fraction = mist.graph_scale_cut(graph, scale_cut_length, num_nodes)
    assert int(num_nodes*num_removed_edges_fraction) == 2


def test_k_nearest_neighbour_scale_cut_2d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    x, y, knn, num_removed_edges_fraction = mist.k_nearest_neighbour_scale_cut(x, y, 0.05, 10)
    index1, index2, edge_length = mist.graph2data(knn)
    condition = np.where(edge_length >= 0.05)[0]
    assert len(condition) == len(edge_length)


def test_k_nearest_neighbour_scale_cut_3d():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    z = np.random.random_sample(50)
    x, y, z, knn, num_removed_edges_fraction = mist.k_nearest_neighbour_scale_cut(x, y, 0.05, 10, z=z)
    index1, index2, edge_length = mist.graph2data(knn)
    condition = np.where(edge_length >= 0.05)[0]
    assert len(condition) == len(edge_length)
