#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Copyright (c) 2022 Lukas Meyer
Licensed under the MIT License.
See LICENSE file for more information.
"""

import logging
from copy import copy

from scipy.spatial.transform import Rotation as R
import networkx as nx

from pc_skeletor.utility import *


class SkeletonBase(object):
    def __init__(self, verbose: bool = False, debug: bool = False):
        self.verbose: bool = verbose
        self.debug: bool = debug

        if self.verbose:
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.DEBUG)
        else:
            logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)

        # Skeleton output
        self.pcd: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.contracted_point_cloud: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.skeleton: o3d.geometry.PointCloud = o3d.geometry.PointCloud()
        self.graph: nx.Graph = nx.Graph()
        self.topology: o3d.geometry.LineSet = o3d.geometry.LineSet()

    def extract_skeleton(self):
        pass

    def extract_topology(self):
        pass

    def save(self, *args):
        pass

    def animate(self,
                init_rot: np.ndarray = np.eye(3),
                steps: int = 360,
                point_size: float = 1.0,
                output: [str, None] = None):
        """
            Creates an animation of a point cloud. The point cloud is simply rotated by 360 Degree in multpile steps.

            :param init_rot: Inital rotation to align pcd for visualization
            :param steps: animation rotates 36o degree and is divided into #steps .
            :param point_size: point size of point cloud points.
            :param output_folder: folder where the rendered images are saved to. If None, no images will be saved.

            :return:
        """
        output_folder = os.path.join(output, './tmp_{}'.format(time.time_ns()))
        os.mkdir(output_folder)

        # Load PCD
        orig = copy(self.pcd)
        orig.rotate(init_rot, center=[0, 0, 0])

        skel = copy(self.contracted_point_cloud)
        skel.paint_uniform_color([0, 0, 1])
        skel.rotate(init_rot, center=[0, 0, 0])

        topo = copy(self.topology)
        topo.paint_uniform_color([0, 0, 0])
        topo.rotate(init_rot, center=[0, 0, 0])

        pcd = copy(orig)

        vis = o3d.visualization.Visualizer()
        vis.create_window(width=1920, height=1080)
        vis.add_geometry(pcd)
        vis.add_geometry(topo)

        ctl = vis.get_view_control()
        ctl.set_zoom(0.6)

        # Set smaller point size. Default is 5.0
        vis.get_render_option().point_size = point_size
        vis.get_render_option().line_width = 15
        vis.get_render_option().light_on = False
        vis.update_renderer()

        # Calculate rotation matrix for every step. Must only be calculated once as rotations are added up in the point cloud
        Rot_mat = R.from_euler('y', np.deg2rad(360 / steps)).as_matrix()

        image_path_list = []

        pcd_idx = 0

        for i in range(steps):
            orig.rotate(Rot_mat, center=[0, 0, 0])
            skel.rotate(Rot_mat, center=[0, 0, 0])
            topo.rotate(Rot_mat, center=[0, 0, 0])

            if pcd_idx == 0:
                pcd.points = orig.points
                pcd.colors = orig.colors
                pcd.normals = orig.normals
            if pcd_idx == 1:
                pcd.points = skel.points
                pcd.colors = skel.colors
            vis.update_geometry(pcd)
            vis.update_geometry(topo)
            vis.poll_events()
            vis.update_renderer()

            if ((i % 30) == 0) and i != 0:
                pcd_idx = (pcd_idx + 1) % 2

            if True:
                current_image_path = "{}/img_%04d.jpg".format(output_folder) % i
                vis.capture_screen_image(current_image_path)
                image_path_list.append(current_image_path)
        vis.destroy_window()

        generate_gif(filenames=image_path_list, output_name='skeleton_animation')

