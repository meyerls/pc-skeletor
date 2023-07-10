# 'construct.py' contains the basic minimum spanning tree algorithms for
# constructing the tree and infering statistics.

import numpy as np
from scipy.sparse.csgraph import minimum_spanning_tree
from sklearn.neighbors import kneighbors_graph

from . import scale_cut
from . import tomo


def construct_mst(x, y, z=None, k_neighbours=20, two_dimensions=False, scale_cut_length=0., is_tomo=False):
    """Constructs the minimum spanning tree (MST) of an input data set.

    Parameters
    ----------
    x, y, z : array
        The coordinate positions of the nodes.
    k_neighbours : integer, optional
        The number of nearest neighbours to consider when creating the k-nearest neighbour graph.
    two_dimensions : bool, optional
        If true assumes the data set is two dimensional.
    scale_cut_length : float, optional
        The minimum allowed length in the k_nearest_neighbour_graph.
    is_tomo : bool, optional
        Should be true if the entered x, y and z coordinates are on a unit sphere.

    Returns
    -------
    edge_length : array
        The length of the edges included in the MST.
    edge_x, edge_y, edge_z : array
        The positional coordinates of the ends of each edge.
    edge_index : array
        The node index of the ends of each edge.
    num_removed_edges : int, optional
        Number of removed edges. Only given and applicable if scale_cut_length is given.
    """
    if two_dimensions is True:
        vertices = np.array([x, y]).T
    else:
        vertices = np.array([x, y, z]).T
    if scale_cut_length != 0.:
        if two_dimensions is True:
            x, y, k_nearest_neighbour_graph, num_removed_edges = \
                scale_cut.k_nearest_neighbour_scale_cut(x, y, scale_cut_length, k_neighbours)
        else:
            x, y, z, k_nearest_neighbour_graph, num_removed_edges = \
                scale_cut.k_nearest_neighbour_scale_cut(x, y, scale_cut_length, k_neighbours, z=z)
    else:
        k_nearest_neighbour_graph = kneighbors_graph(vertices, n_neighbors=k_neighbours, mode='distance')
    if is_tomo is True:
        k_nearest_neighbour_graph = tomo.convert_tomo_knn_length2angle(k_nearest_neighbour_graph, len(x))
    tree = minimum_spanning_tree(k_nearest_neighbour_graph, overwrite=True)
    tree = tree.tocoo()
    edge_length = tree.data
    index1 = tree.row
    index2 = tree.col
    x1 = x[index1]
    x2 = x[index2]
    edge_x = np.array([x1, x2])
    y1 = y[index1]
    y2 = y[index2]
    edge_y = np.array([y1, y2])
    if two_dimensions is False:
        z1 = z[index1]
        z2 = z[index2]
        edge_z = np.array([z1, z2])
    edge_index = np.array([index1, index2])
    if scale_cut_length == 0.:
        if two_dimensions is True:
            return edge_length, edge_x, edge_y, edge_index
        else:
            return edge_length, edge_x, edge_y, edge_z, edge_index
    else:
        if two_dimensions is True:
            return edge_length, edge_x, edge_y, edge_index, num_removed_edges
        else:
            return edge_length, edge_x, edge_y, edge_z, edge_index, num_removed_edges
