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
from typing import Union, List

import scipy.sparse.linalg as sla
from scipy import sparse
import matplotlib.pyplot as plt
import imageio
import open3d as o3d
import numpy as np


def visualize(geometry: list,
              width: int = 1920,
              height: int = 1080,
              background_color: tuple = (0, 0, 0),
              point_size: float = 0.1,
              line_width: float = 1.,
              camera: Union[dict, bool] = False,
              window_name: str = 'Open3D',
              filename: Union[str, bool] = False):
    '''
    Nice Visualization of the geometry in open3D's main visualizer.

    :param geometry: list of open3d gemotries
    :param width: window width for the visualizer
    :param height: window height for the visualizer
    :param background_color: tuple of background color (r,g,b). Between 0 and 1.
    :param point_size: Point cloud point size
    :param line_width: line set line width
    :param camera: A json/dict view control for the visualizer. Simply ctrl + c inside a window to retain this view point.
    :param window_name: Name of the window
    :param filename: Default is False: Otherwise specify the file path to save the rendered image
    :return:
    '''
    vis = o3d.visualization.Visualizer()

    vis.create_window(window_name=window_name, width=width, height=height)
    opt = vis.get_render_option()
    opt.point_size = point_size
    opt.line_width = line_width
    opt.background_color = background_color
    vis.clear_geometries()

    for g in geometry:
        vis.add_geometry(g)

    if camera:
        ctr = vis.get_view_control()
        ctr.set_front(camera['trajectory'][0]['front'])
        ctr.set_lookat(camera['trajectory'][0]['lookat'])
        ctr.set_up(camera['trajectory'][0]['up'])
        ctr.set_zoom(camera['trajectory'][0]['zoom'])
    opt.light_on = False

    if filename:
        vis.capture_screen_image(filename=filename, do_render=True)
    else:
        vis.run()
    vis.close()


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


def points2pcd(points: np.ndarray):
    '''
    Convert a numpy array to an open3d point cloud. Just for convenience to avoid converting it every single time.
    Assigns blue color uniformly to the point cloud.

    :param points: Nx3 array with xyz location of points
    :return: a blue open3d.geometry.PointCloud()
    '''

    colors = [[0, 0, 1] for i in range(points.shape[0])]
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    return pcd


def load_pcd(filename):
    '''
    Wrapper to load open3D point Cloud.

    :param filename: path to file
    :return: a open3d.geometry.PointCloud()
    '''
    pcd = o3d.io.read_point_cloud(filename)

    return pcd


def normalize_pcd(pcd):
    bbox = pcd.get_axis_aligned_bounding_box()
    scale = 1.6 / max((bbox.max_bound - bbox.min_bound))
    pcd = pcd.translate(-pcd.get_center())
    pcd_normalized = pcd.scale(scale, center=[0, 0, 0])

    return pcd_normalized


def display_inlier_outlier(cloud: o3d.geometry.PointCloud, ind: List[int]):
    '''
    It separates the inlier and outlier points from the input cloud and displays them using the Open3D visualization library.

    :param cloud: The input point cloud data as an Open3D PointCloud object.
    :param ind:  A list of indices representing the inlier points in the input cloud
    :return:
    '''
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


def simplify_graph(G):
    """
    The simplifyGraph function simplifies a given graph by removing nodes of degree 2 and fusing their incident edges.
    Source:  https://stackoverflow.com/questions/53353335/networkx-remove-node-and-reconnect-edges

    :param G: A NetworkX graph object to be simplified
    :return: A tuple consisting of the simplified NetworkX graph object, a list of positions of kept nodes, and a list of indices of kept nodes.
    """

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


def least_squares_sparse(pcd_points: np.ndarray, laplacian: sparse.csr_matrix, laplacian_weighting: float,
                         positional_weighting: np.ndarray, debug: bool = False):
    """
    Solve a sparse least squares problem to reconstruct a point cloud with smooth geometry.

    Given a set of point cloud data `pcd_points`, a Laplacian matrix `laplacian`, a weighting factor `laplacian_weighting`
    to adjust the importance of smoothness, and a weighting vector `positional_weighting` to adjust the importance of
    each point's position.

    Parameters:
        pcd_points (numpy.ndarray): An array of shape (N, 3) containing the original point cloud data.
        laplacian (scipy.sparse.csr_matrix): A sparse Laplacian matrix of shape (N, N) that encodes the geometry of the point cloud.
        laplacian_weighting (float): A scalar weighting factor for the Laplacian matrix to adjust the importance of smoothness.
        positional_weighting (numpy.ndarray): A vector of shape (N,) that encodes the weighting of each point's position in the reconstruction.
        debug (bool, optional): If True, plot a sparsity pattern of the matrix `A_new` for performance debugging.


    :param pcd_points: An array of shape (N, 3) containing the original point cloud data.
    :param laplacian: A sparse Laplacian matrix of shape (N, N) that encodes the geometry of the point cloud.
    :param laplacian_weighting: A scalar weighting factor for the Laplacian matrix to adjust the importance of smoothness.
    :param positional_weighting: A vector of shape (N,) that encodes the weighting of each point's position in the reconstruction.
    :param debug: If True, plot a sparsity pattern of the matrix `A_new` for performance debugging.
    :return:  An array of shape (N, 3) containing the reconstructed point cloud.
    """

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
