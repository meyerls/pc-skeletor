# 'scale_cut.py' applies a scale cut to an input graph, removing edges below a
# given scale cut limit.

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import kneighbors_graph
from . import graph as gr


def graph_scale_cut(graph, scale_cut_length, num_nodes):
    """Will remove all edges in the graph below the scale_cut_length.

    Parameters
    ----------
    graph : csr_matrix
        A sparse matrix of the edges in a graph and corresponding node indexes.
    scale_cut_length : float
        A minimum length scale.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    graph_cut : csr_matrix
        The original graph with distances below the scale_cut_length removed.
    index1, index2 : array
        The node indexes of each end of the edges in the graph.
    num_removed_edges : int
        Number of removed edges.
    """
    index1, index2, distances = gr.graph2data(graph)
    condition = np.where((distances >= scale_cut_length))[0]
    num_removed_edges_fraction = float(len(index1) - len(condition))/float(len(index1))
    index1, index2, distances = index1[condition], index2[condition], distances[condition]
    graph_cut = gr.data2graph(index1, index2, distances, num_nodes)
    return graph_cut, index1, index2, num_removed_edges_fraction


def k_nearest_neighbour_scale_cut(x, y, scale_cut_length, k_neighbours, z=None):
    """Iteratively removes edges below the scale_cut_length of a k_nearest_neighbour graph.

    Parameters
    ----------
    x, y, (z) : array
        2D or 3D coordinates of the positions of the nodes.
    scale_cut_length : float
        The minimum allowed length in the k_nearest_neighbour_graph.
    k_neighbours : int
        The number of nearest neighbours to consider when creating the k-nearest neighbour graph.
    Returns
    -------
    x, y, (z) : array
        The 2D or 3D coordinates of the positions of the nodes.
    knn : csr_matrix
        A sparse scale cut k_nearest_neighbour_graph.
    """
    if z is None:
        vertices = np.array([x, y]).T
    else:
        vertices = np.array([x, y, z]).T
    knn = kneighbors_graph(vertices, n_neighbors=k_neighbours, mode='distance')
    knn, index1, index2, num_removed_edges_fraction = graph_scale_cut(knn, scale_cut_length, len(x))
    if z is None:
        return x, y, knn, num_removed_edges_fraction
    else:
        return x, y, z, knn, num_removed_edges_fraction
