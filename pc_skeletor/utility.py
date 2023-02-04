#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

from functools import wraps
import time
import os

import scipy.sparse.linalg as sla
from scipy import sparse
import matplotlib.pyplot as plt
import imageio
import open3d as o3d
import numpy as np


def timeit(func):
    @wraps(func)
    def timeit_wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()
        total_time = end_time - start_time
        print(f'Function {func.__name__}{args} {kwargs} Took {total_time:.4f} seconds')
        return result

    return timeit_wrapper


def points2pcd(points):
    colors = [[0, 0, 1] for i in range(points.shape[0])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def load_pcd(filename, normalize=False):
    pcd = o3d.io.read_point_cloud(filename)

    return pcd


def normalize_pcd(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    scale = 1.6 / max((bbox.max_bound - bbox.min_bound))
    pcd = pcd.translate(-pcd.get_center())
    pcd_normalized = pcd.scale(scale, center=[0, 0, 0])

    return pcd_normalized


def display_inlier_outlier(cloud, ind):
    inlier_cloud = cloud.select_by_index(ind)
    outlier_cloud = cloud.select_by_index(ind, invert=True)

    print("Showing outliers (red) and inliers (gray): ")
    outlier_cloud.paint_uniform_color([1, 0, 0])
    inlier_cloud.paint_uniform_color([0, 0, 1])
    o3d.visualization.draw_geometries([inlier_cloud, outlier_cloud],
                                      zoom=0.3412,
                                      front=[0.4257, -0.2125, -0.8795],
                                      lookat=[2.6172, 2.0475, 1.532],
                                      up=[-0.0694, -0.9768, 0.2024])


def generate_gif(filenames, output_name):
    """
    Generates GIF from a list of image file paths

    :param filenames: a list of image file paths
    :param output_path: output folder to save gif
    :return:
    """
    images = []
    for filename in filenames:
        images.append(imageio.imread(filename))
    imageio.mimsave(os.path.join(os.path.dirname(filenames[0]), '{}.gif'.format(output_name)), images, format='GIF')


def simplifyGraph(G):
    '''
    Loop over the graph until all nodes of degree 2 have been removed and their incident edges fused
     https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges
     '''

    g = G.copy()

    while any(degree == 2 for _, degree in g.degree):

        keept_node_pos = []
        keept_node_idx = []
        g0 = g.copy()  # <- simply changing g itself would cause error `dictionary changed size during iteration`
        for node, degree in g.degree():
            if degree == 2:

                if g.is_directed():  # <-for directed graphs
                    a0, b0 = list(g0.in_edges(node))[0]
                    a1, b1 = list(g0.out_edges(node))[0]

                else:
                    edges = g0.edges(node)
                    edges = list(edges.__iter__())
                    a0, b0 = edges[0]
                    a1, b1 = edges[1]

                e0 = a0 if a0 != node else b0
                e1 = a1 if a1 != node else b1

                g0.remove_node(node)
                g0.add_edge(e0, e1)
            else:
                keept_node_pos.append(g.nodes[node]['pos'])
                keept_node_idx.append(node)
        g = g0

    return g, keept_node_pos, keept_node_idx


def least_squares_sparse(pcd_points, laplacian, laplacian_weighting, positional_weighting, debug=False):
    # Define Weights
    I = sparse.eye(pcd_points.shape[0])
    WL = I * laplacian_weighting
    WH = sparse.diags(positional_weighting)

    A = sparse.vstack([laplacian.dot(WL), WH]).tocsc()
    b = np.vstack([np.zeros((pcd_points.shape[0], 3)), WH.dot(pcd_points)])

    A_new = A.T @ A

    if debug:
        plt.spy(A_new, ms=0.1)
        plt.title('A_new: A.T @ A')
        plt.show()

    permc_method = 'COLAMD'

    x = sla.spsolve(A_new, A.T @ b[:, 0], permc_spec=permc_method)
    y = sla.spsolve(A_new, A.T @ b[:, 1], permc_spec=permc_method)
    z = sla.spsolve(A_new, A.T @ b[:, 2], permc_spec=permc_method)

    return np.vstack([x, y, z]).T
