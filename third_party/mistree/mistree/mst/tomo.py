# 'tomo.py' converts tomographic k-nearest neighbour graphs from being defined
# across a unitary sphere to perpendical angles across a sphere.

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[:-3] + '/coordinates/')

from . import graph
import coordinate_utility


def convert_tomo_knn_length2angle(k_nearest_neighbour_graph, number_of_nodes):
    """Converts the k-nearest neighbour graph from 3D lengths on a unit sphere to angular distances.

    Parameters
    ----------
    k_nearest_neighbour_graph : csr_matrix
        A sparse matrix of the nearest k neighbours.
    number_of_nodes : int
        The number of nodes from which the nearest neighbour graph is constructed.

    Returns
    -------
    k_nearest_neighbour_graph_angle : csr_matrix
        The same as input with edges returned in angles (radians).
    """
    index1, index2, distances = graph.graph2data(k_nearest_neighbour_graph)
    distances_angles = coordinate_utility.perpendicular_distance_2_angle(distances)
    k_nearest_neighbour_graph_angle = graph.data2graph(index1, index2, distances_angles, number_of_nodes)
    return k_nearest_neighbour_graph_angle
