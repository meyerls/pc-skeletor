import open3d as o3d
import pytest
from copy import deepcopy
import networkx as nx
from pc_skeletor import SLBC, LBC
from pc_skeletor import Dataset

downloader = Dataset()
trunk_pcd_path, branch_pcd_path = downloader.download_semantic_tree_dataset()

pcd_trunk = o3d.io.read_point_cloud(trunk_pcd_path)
pcd_branch = o3d.io.read_point_cloud(branch_pcd_path)
pcd = pcd_trunk + pcd_branch

def test_slbc():
    global pcd

    pcd_trunk_test = deepcopy(pcd_trunk)
    pcd_branch_test = deepcopy(pcd_branch)

    # Semantic Laplacian-based Contraction
    s_lbc = SLBC(point_cloud={'trunk': pcd_trunk_test, 'branches': pcd_branch_test},
                 semantic_weighting=30.,
                 init_contraction=1,
                 init_attraction=0.5,
                 down_sample=0.3)
    s_lbc.extract_skeleton()
    s_lbc.extract_topology()

    assert isinstance(s_lbc.contracted_point_cloud,
                      o3d.geometry.PointCloud), 'Contracted point cloud is not of type o3d.geometry.PointCloud!'
    assert isinstance(s_lbc.skeleton, o3d.geometry.PointCloud), 'skeleton is not of type o3d.geometry.PointCloud!'
    assert isinstance(s_lbc.topology, o3d.geometry.LineSet), 'Extracted topology is not of type o3d.geometry.LineSet!'
    assert isinstance(s_lbc.topology_graph, nx.Graph), 'Extracted topology graph is not of type nx.Graph'
    assert isinstance(s_lbc.skeleton_graph, nx.Graph), 'Extracted skeleton graph is not of type nx.Graph'
