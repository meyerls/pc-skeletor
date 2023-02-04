#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

# Built-in/Generic Imports
from typing import Tuple
from copy import copy
import typing

# Libs
import open3d.visualization as o3d
import robust_laplacian
import networkx as nx
import mistree as mist
from dgl.geometry import farthest_point_sampler
import torch
from scipy.spatial.transform import Rotation as R

# Own modules
from pc_skeletor.utility import *
from pc_skeletor.download import *


class Skeletonizer(object):
    '''
    1. Laplacian based contraction.
        Paper: https://taiya.github.io/pubs/cao2010cloudcontr.pdf
        Code: https://github.com/taiya/cloudcontr
    2. Source:
        Paper: https://www.cs.sfu.ca/~haoz/pubs/huang_sig13_l1skel.pdf
        Code: https://github.com/HongqiangWei/L1-Skeleton
    '''

    def __init__(self, point_cloud: [o3d.geometry.PointCloud, str], debug: bool, down_sample: float = 0.005):
        '''

        :param point_cloud:
        :param debug:
        :param down_sample:

        :param point_cloud: point cloud or string path to point cloud
        :param debug: Boolean, causes several pyplot windows to get shown to debug the process.
        :param down_sample: Performs voxel_down_sample on the given point cloudb before running contraction.
                            Smaller values will result in smaller voxels, which will take longer to run.
        '''
        if isinstance(point_cloud, str):
            self.pcd = load_pcd(filename=point_cloud, normalize=False)
        else:
            self.pcd = point_cloud

        self.debug = debug

        self.down_sample = down_sample  # hq = 0.005, lq = 0.03

        # Intermediate pcd results
        self.contracted_point_cloud = None
        self.sceleton = None

        self.graph_k_n = 15

    def save(self, result_folder: str):
        os.makedirs(result_folder, exist_ok=True)
        o3d.io.write_point_cloud(os.path.join(result_folder, '01_point_cloud_contracted' + '.ply'),
                                 self.contracted_point_cloud)
        o3d.io.write_point_cloud(os.path.join(result_folder, '02_sceleton' + '.ply'),
                                 self.sceleton)

    def init_laplacian_weights(self, num_pcd_points: int, config: typing.Dict) -> Tuple[float, float, float]:
        '''
        Initialize parameters for laplacian based contraction. This is a suggestion! For a best practice these values
        should be computed automatically and not with hard decision.

        :param num_pcd_points:
        :return:
        '''
        positional_init_weights = 1

        self.MAX_LAPLACE_CONTRACTION_WEIGHT = config["MAX_LAPLACE_CONTRACTION_WEIGHT"]  # 2048
        self.MAX_POSITIONAL_WEIGHT = config["MAX_POSITIONAL_WEIGHT"]  # 2048
        self.init_laplacian_scale_factor = config["INIT_LAPLACIAN_SCALE"]

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
    def laplacian_contraction(self, point_cloud: o3d.geometry.PointCloud,
                              down_sample: float,
                              config: typing.Dict,
                              iterations: int = 15) -> o3d.geometry.PointCloud:
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
            num_pcd_points=pcd_points.shape[0], config=config)

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
                laplacian_weigths = 1 / (self.init_laplacian_scale_factor * np.mean(
                    M.diagonal()))  # 1 * np.sqrt(np.mean(M.diagonal())) # 10 ** -3
            else:
                # Update laplacian weights with amplification factor
                laplacian_weigths *= amplification
                # Update positional weights with the ration of the first Mass matrix and the current one.
                positional_weights = positional_weights * (M_list[0] / M.diagonal()) / 20
            M_list.append(M.diagonal())

            # Clip weights
            laplacian_weigths = np.clip(laplacian_weigths, 1, self.MAX_LAPLACE_CONTRACTION_WEIGHT)
            positional_weights = np.clip(positional_weights, 1, self.MAX_POSITIONAL_WEIGHT)

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

    def visualize(self):
        o3d.visualization.draw_geometries([self.pcd, self.sceleton])

    def animate(self, init_rot: np.ndarray = np.eye(3),
                steps: int = 360,
                point_size: float = 1.0,
                out: [str, None] = None):
        """
            Creates an animation of a point cloud. The point cloud is simply rotated by 360 Degree in multpile steps.

            :param init_rot: Inital rotation to align pcd for visualization
            :param steps: animation rotates 36o degree and is divided into #steps .
            :param point_size: point size of point cloud points.
            :param output_folder: folder where the rendered images are saved to. If None, no images will be saved.

            :return:
            """

        output_folder = os.path.join(out, './tmp_{}'.format(time.time_ns()))
        os.mkdir(output_folder)

        # Load PCD
        orig = copy(self.pcd)
        orig.rotate(init_rot)
        scel = copy(self.sceleton)
        scel.paint_uniform_color([0, 0, 1])
        scel.rotate(init_rot)

        pcd = copy(orig)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=2000, height=1000)
        # vis.set_full_screen(True)
        vis.add_geometry(pcd)

        ctl = vis.get_view_control()
        ctl.set_zoom(0.6)

        print("Point size: ", point_size)
        # Set smaller point size. Default is 5.0
        vis.get_render_option().point_size = point_size
        vis.get_render_option().light_on = False
        vis.update_renderer()

        # Calculate rotation matrix for every step. Must only be calculated once as rotations are added up in the point cloud
        Rot_mat = R.from_euler('y', np.deg2rad(360 / steps)).as_matrix()

        image_path_list = []

        pcd_idx = 0

        for i in range(steps):

            orig.rotate(Rot_mat)
            scel.rotate(Rot_mat)

            if pcd_idx == 0:
                pcd.points = orig.points
                pcd.colors = orig.colors
                pcd.normals = orig.normals
            if pcd_idx == 1:
                pcd.points = scel.points
                pcd.colors = scel.colors
            vis.update_geometry(pcd)
            vis.poll_events()
            vis.update_renderer()

            if ((i % 40) == 0) and i != 0:
                pcd_idx = (pcd_idx + 1) % 2

            if True:
                current_image_path = "{}/img_%04d.jpg".format(output_folder) % i
                vis.capture_screen_image(current_image_path)
                image_path_list.append(current_image_path)
        vis.destroy_window()

        generate_gif(filenames=image_path_list, output_name='skeleton_animation')

    def extract(self, method: str = 'Laplacian',
                config: typing.Dict = {
                    "MAX_LAPLACE_CONTRACTION_WEIGHT": 1024,
                    "MAX_POSITIONAL_WEIGHT": 1024,
                    "INIT_LAPLACIAN_SCALE": 100
                }) -> Tuple[
        o3d.geometry.PointCloud, o3d.geometry.PointCloud]:

        if method == 'Laplacian':
            contracted_point_cloud = self.laplacian_contraction(point_cloud=self.pcd,
                                                                down_sample=self.down_sample,
                                                                config=config)
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

        # Connectivity with MST
        mst = mist.GetMST(x=branch_contracted_points_fps[:, 0], y=branch_contracted_points_fps[:, 1],
                          z=branch_contracted_points_fps[:, 2])
        d, l, b, s, l_index, b_index = mst.get_stats(include_index=True, k_neighbours=self.graph_k_n)

        # Convert to Graph
        mst = nx.Graph(l_index.T.tolist())
        for idx in range(mst.number_of_nodes()):
            mst.nodes[idx]['pos'] = branch_contracted_points_fps[idx].T

        G_simplified, node_pos, node_idx = simplifyGraph(mst)
        skeleton_cleaned = o3d.geometry.PointCloud(o3d.utility.Vector3dVector(np.vstack(node_pos)))
        skeleton_cleaned.paint_uniform_color([0, 0, 1])
        skeleton_cleaned_points = np.asarray(skeleton_cleaned.points)

        mapping = {}
        for node in G_simplified:
            pcd_idx = np.where(skeleton_cleaned_points == G_simplified.nodes[node]['pos'])[0][0]
            mapping.update({node: pcd_idx})

        self.graph = nx.relabel_nodes(G_simplified, mapping)

        self.skeleton_graph = o3d.geometry.LineSet()
        self.skeleton_graph.points = o3d.utility.Vector3dVector(skeleton_cleaned_points)
        self.skeleton_graph.lines = o3d.utility.Vector2iVector(list((self.graph.edges())))

        return self.sceleton, self.graph, self.skeleton_graph


if __name__ == '__main__':
    # Download test tree dataset
    downloader = Dataset()
    downloader.download_tree_dataset()

    # Init tree skeletonizer
    skeletor = Skeletonizer(point_cloud=downloader.file_path,
                            down_sample=0.01,
                            debug=False)
    laplacian_config = {"MAX_LAPLACE_CONTRACTION_WEIGHT": 1024,
                        "MAX_POSITIONAL_WEIGHT": 1024,
                        "INIT_LAPLACIAN_SCALE": 100}
    skeleton, graph, skeleton_graph = skeletor.extract(method='Laplacian', config=laplacian_config)
    output_folder = './data/'
    # save results
    skeletor.save(result_folder=output_folder)
    # Make animation of original point cloud and skeletonization
    skeletor.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=200, out=output_folder)
    # Interactive visualization
    skeletor.visualize()
