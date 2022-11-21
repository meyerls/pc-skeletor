#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
import os
import timeit

# Libs
import open3d.visualization as o3d
import robust_laplacian
import numpy as np
import scipy.sparse.linalg as sla
from scipy import sparse
import matplotlib.pyplot as plt
from dgl.geometry import farthest_point_sampler
import torch

# Own modules
from pc_skeletor.utility import *
from pc_skeletor.download import *


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


class Skeletonizer(object):
    '''
    1. Laplacian based contraction.
        Paper: https://taiya.github.io/pubs/cao2010cloudcontr.pdf
        Code: https://github.com/taiya/cloudcontr
    2. Source:
        Paper: https://www.cs.sfu.ca/~haoz/pubs/huang_sig13_l1skel.pdf
        Code: https://github.com/HongqiangWei/L1-Skeleton
    '''

    def __init__(self, point_cloud, debug, down_sample=0.005):
        if isinstance(point_cloud, str):
            self.pcd = load_pcd(filename=point_cloud, normalize=False)
        else:
            self.pcd = point_cloud

        # Pointcloud to apply skeletonization algorithm
        self.pcd = point_cloud

        self.debug = debug

        self.down_sample = down_sample  # hq = 0.005, lq = 0.03

        # Intermediate pcd results
        self.contracted_point_cloud = None
        self.sceleton = None

    def save(self, result_folder):
        os.makedirs(result_folder, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(result_folder, '01_point_cloud_contracted' + '.ply'),
                                 self.contracted_point_cloud)
        o3d.io.write_point_cloud(os.path.join(result_folder, '02_sceleton' + '.ply'),
                                 self.sceleton)

    def init_laplacian_weights(self, num_pcd_points):
        '''
        Initialize parameters for laplacian based contraction. This is a suggestion! For a best practice these values
        should be computed automatically and not with hard decision.

        :param branch_points:
        :return:
        '''
        positional_init_weights = 1

        self.MAX_LAPLACE_CONTRACTION_WEIGHT = 1024  # 2048
        self.MAX_CONTRACTION_WEIGHT = 1024  # 2048

        if num_pcd_points < 1000:
            s_l = 1
            volume_ratio_quit = 0.01
        elif num_pcd_points < 1e4:
            s_l = 2
            volume_ratio_quit = 0.007
        elif num_pcd_points < 1e5:
            s_l = 5
            volume_ratio_quit = 0.005
        elif num_pcd_points < 0.5 * 1e6:
            s_l = 5
            volume_ratio_quit = 0.004
        elif num_pcd_points < 1e6:
            s_l = 5
            volume_ratio_quit = 0.003
        else:
            # self.MAX_LAPLACE_CONTRACTION_WEIGHT = 2048 * 2
            s_l = 8
            volume_ratio_quit = 0.001

        return positional_init_weights, s_l, volume_ratio_quit

    @timeit
    def laplacian_contraction(self, point_cloud, down_sample, iterations=15) -> o3d.geometry.PointCloud:
        '''
        Laplacian based contraction on point clouds.

        Heart of the algorithm is the computation of the cotangent laplacian (What is cotangent laplacian?
        http://rodolphe-vaillant.fr/entry/101/definition-laplacian-matrix-for-triangle-meshes).

        Computation of the discrete laplacian is done with the package/library robust_laplacian and can be found
        here: https://github.com/nmwsharp/robust-laplacians-py. On how to compute the laplacian please refer to
        http://www.cs.cmu.edu/~kmcrane/Projects/NonmanifoldLaplace/NonmanifoldLaplace.pdf.

        :param point_cloud: original point cloud
        :param down_sample: down sample (equivalent voxel size)
        :return: contracted point cloud -> skeleton
        '''

        # Down sampling point cloud for faster contraction.
        pcd = point_cloud.voxel_down_sample(down_sample)
        pcd_points = np.asarray(pcd.points)

        positional_weights, amplification, volume_ratio_quit = self.init_laplacian_weights(
            num_pcd_points=pcd_points.shape[0])

        print('PCD #Points: ', pcd_points.shape[0], '\n')
        print('Stepwise amplification factor: ', amplification)

        M_list = []
        for i in range(iterations):
            print('Iteration: ', i)

            # 1. Calculate laplacian of point cloud (mean curvature computation)
            # Build point cloud Laplacian
            L, M = robust_laplacian.point_cloud_laplacian(pcd_points, mollify_factor=1e-5, n_neighbors=30)

            # 2. init or update weights
            if i == 0:
                # Init weights, weighted by the mass matrix
                positional_weights = positional_weights * np.ones(M.shape[0])
                laplacian_weigths = 1 / (100 * np.mean(M.diagonal()))  # 1 * np.sqrt(np.mean(M.diagonal())) # 10 ** -3
            else:
                # Update laplacian weights with amplification factor
                laplacian_weigths *= amplification
                # Update positional weights with the ration of the first Mass matrix and the current one.
                positional_weights = positional_weights * (M_list[0] / M.diagonal()) / 20
            M_list.append(M.diagonal())

            # Clip weights
            laplacian_weigths = np.clip(laplacian_weigths, 1, self.MAX_LAPLACE_CONTRACTION_WEIGHT)
            positional_weights = np.clip(positional_weights, 1, self.MAX_CONTRACTION_WEIGHT)

            print('Laplacian Weight: ', laplacian_weigths)
            print('Mean Positional Weight: ', np.mean(positional_weights))

            # 3. Least squares meshes
            pcd_points = least_squares_sparse(pcd_points=pcd_points,
                                              laplacian=L,
                                              laplacian_weighting=laplacian_weigths,
                                              positional_weighting=positional_weights,
                                              debug=self.debug)

            if self.debug:
                skeleton = points2pcd(pcd_points)
                open3d.visualization.draw_geometries([pcd, skeleton])

            # 4. Check for termination condition. Based on the volume
            volume_ratio = np.mean(M_list[-1]) / np.mean(M_list[0])
            print('Volume ratio :', volume_ratio, '\n')
            if volume_ratio < volume_ratio_quit:
                break

        print('Contraction is Done.')

        return pcd_points

    def extract(self, method='Laplacian'):

        if method == 'Laplacian':
            contracted_point_cloud = self.laplacian_contraction(point_cloud=self.pcd, down_sample=self.down_sample)
        elif method == 'L1':
            raise NotImplementedError('Up to now only Laplacian based contraction!')
        else:
            raise NotImplementedError('Up to now only Laplacian based contraction!')

        # Convert to o3d point cloud and prepare with torch for dgl
        self.contracted_point_cloud = points2pcd(contracted_point_cloud)
        pcd2torch = torch.asarray(contracted_point_cloud)[np.newaxis, ...]

        # Compute points for farthest point sampling
        self.fps_points = int(contracted_point_cloud.shape[0] * 0.05)
        self.fps_points = max(self.fps_points, 15)

        # Sample with farthest point sampling
        point_idx = farthest_point_sampler(pcd2torch, self.fps_points)

        # convert back to o3d point cloud object
        branch_contracted_points_fps = pcd2torch[0, point_idx].numpy()[0]
        self.sceleton = points2pcd(branch_contracted_points_fps)

        return self.contracted_point_cloud, self.sceleton


if __name__ == '__main__':
    downloader = Dataset()
    downloader.download_tree_dataset()

    pcd_file = downloader.file_path
    pcd = o3d.io.read_point_cloud(pcd_file)
    output_folder = './data/'
    skeletor = Skeletonizer(point_cloud=pcd,
                            down_sample=0.01,
                            debug=False)
    sceleton = skeletor.extract(method='Laplacian')
    skeletor.save(output_folder)
