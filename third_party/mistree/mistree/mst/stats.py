# 'stats.py' contains functions for determining statistical properties (degree
# quantities at present) of an input graph.

import numpy as np
from . import utility_mst


def get_graph_degree(edge_index, number_of_nodes):
    """Finds the degree of each node in a constructed graph.

    Parameters
    ----------
    edge_index : array
        A 2 dimensional array, containing the node indexes on either side of each edge.
    number_of_nodes : int
        The number of nodes in the graph.

    Returns
    -------
    degree : array
        The degree for each node in the graph.
    """
    index1, index2 = edge_index[0], edge_index[1]
    _number_of_edges = len(index1)
    degree = utility_mst.get_degree_for_index(index1, index2, number_of_nodes, _number_of_edges)
    return degree


def get_mean_degree_for_edges(edge_index, degree):
    """Finds the mean degree for each edge. This is done by finding the mean of the degree at each end of the edges.

    Parameters
    ----------
    edge_index : array
        A 2 dimensional array, containing the node indexes on either side of each edge.
    degree : array
        The degree for each node in the graph.

    Returns
    -------
    mean_degree : array
        The mean degree for each edge.
    """
    index1, index2 = edge_index[0], edge_index[1]
    mean_degree = 0.5 * (degree[index1] + degree[index2])
    return mean_degree


def get_degree_for_edges(edge_index, degree):
    """Gets the degree of the nodes at each end of the edge.

    Parameters
    ----------
    edge_index : array
        The node index of the ends of each edge.
    degree : array
        The degree for each node in the graph.

    Returns
    -------
    edge_degree : array
        The degree for each node in an edge.
    """
    index1, index2 = edge_index[0], edge_index[1]
    edge_degree = np.array([degree[index1], degree[index2]])
    return edge_degree
