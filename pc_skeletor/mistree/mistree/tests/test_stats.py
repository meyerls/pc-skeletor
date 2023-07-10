import numpy as np
import mistree as mist


def test_get_graph_degree():
    edge_index = np.array([[0, 0, 4, 4, 2], [1, 2, 2, 3, 3]])
    number_of_nodes = 5
    degree = mist.get_graph_degree(edge_index, number_of_nodes)
    degree2 = np.array([2., 1., 3., 2., 2.])
    condition = np.where(degree == degree2)[0]
    assert len(condition) == 5


def test_get_mean_degree_for_edges():
    edge_index = np.array([[0, 0, 4, 4, 2], [1, 2, 2, 3, 3]])
    number_of_nodes = 5
    degree = mist.get_graph_degree(edge_index, number_of_nodes)
    mean_degree = mist.get_mean_degree_for_edges(edge_index, degree)
    mean_degree2 = np.array([1.5, 2.5, 2.5, 2., 2.5])
    condition = np.where(mean_degree == mean_degree2)[0]
    assert len(condition) == 5


def test_get_degree_for_edges():
    edge_index = np.array([[0, 0, 4, 4, 2], [1, 2, 2, 3, 3]])
    number_of_nodes = 5
    degree = mist.get_graph_degree(edge_index, number_of_nodes)
    edge_degree = mist.get_degree_for_edges(edge_index, degree)
    edge_degree2 = np.array([[2., 2., 2., 2., 3.], [1., 3., 3., 2., 2.]])
    condition = np.where((edge_degree[0] == edge_degree2[0]) &
                         (edge_degree[1] == edge_degree2[1]))[0]
    assert len(condition) == 5
