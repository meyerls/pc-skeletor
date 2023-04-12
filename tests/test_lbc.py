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


@pytest.mark.parametrize('init_contraction', [0.0, 0.00001, 10], ids=lambda param: 'init_contraction({})'.format(param))
@pytest.mark.parametrize('init_attraction', [0.01, 10], ids=lambda param: 'init_attraction({})'.format(param))
@pytest.mark.parametrize('max_contraction', [2 ** 2, 2 ** 18], ids=lambda param: 'max_contraction({})'.format(param))
@pytest.mark.parametrize('max_attraction', [2 ** 2, 2 ** 18], ids=lambda param: 'max_attraction({})'.format(param))
@pytest.mark.parametrize('termination_ratio', [1, 0.005, 0, -1], ids=lambda param: 'max_attraction({})'.format(param))
@pytest.mark.parametrize('max_iteration_steps', [20], ids=lambda param: 'max_iteration_steps({})'.format(param))
def test_lpc_init_parameters(init_contraction,
                             init_attraction,
                             max_contraction,
                             max_attraction,
                             termination_ratio,
                             max_iteration_steps):
    global pcd

    pcd_test = deepcopy(pcd)

    # Laplacian-based Contraction
    lbc = LBC(point_cloud=pcd_test,
              init_contraction=init_contraction,
              init_attraction=init_attraction,
              max_contraction=max_contraction,
              max_attraction=max_attraction,
              step_wise_contraction_amplification='auto',
              termination_ratio=termination_ratio,
              max_iteration_steps=max_iteration_steps,
              down_sample=0.08,
              filter_nb_neighbors=False,
              filter_std_ratio=False,
              debug=False,
              verbose=False)
    lbc.extract_skeleton()
    lbc.extract_topology()

    assert isinstance(lbc.contracted_point_cloud,
                      o3d.geometry.PointCloud), 'Contracted point cloud is not of type o3d.geometry.PointCloud!'
    assert isinstance(lbc.skeleton, o3d.geometry.PointCloud), 'skeleton is not of type o3d.geometry.PointCloud!'
    assert isinstance(lbc.topology, o3d.geometry.LineSet), 'Extracted topology is not of type o3d.geometry.LineSet!'
    assert isinstance(lbc.topology_graph, nx.Graph), 'Extracted topology graph is not of type nx.Graph'
    assert isinstance(lbc.skeleton_graph, nx.Graph), 'Extracted skeleton graph is not of type nx.Graph'


@pytest.mark.parametrize(
    "init_contraction,  init_attraction, max_contraction, max_attraction, termination_ratio, max_iteration_steps", [
        (100000., 0.00001, 2048, 1024, 0.005, 20),
        (0.0, 0.0, 2048, 1024, 0.005, 20),
        (1.0, 0.0, 2048, 1024, 0.005, 20),
        (0.0, 0.0, 2048, 1024, 0.005, 20),
        (0.0, 0.01, 4, 4, 0.005, 0),
        (1e-05, 0.01, 4, 4, 0.005, 0),
    ])
def test_lpc_init_parameters_failure(init_contraction,
                                     init_attraction,
                                     max_contraction,
                                     max_attraction,
                                     termination_ratio,
                                     max_iteration_steps):
    global pcd

    pcd_test = deepcopy(pcd)
    with pytest.raises(ValueError) as e_info:
        # Laplacian-based Contraction
        lbc = LBC(point_cloud=pcd_test,
                  init_contraction=init_contraction,
                  init_attraction=init_attraction,
                  max_contraction=max_contraction,
                  max_attraction=max_attraction,
                  step_wise_contraction_amplification='auto',
                  termination_ratio=termination_ratio,
                  max_iteration_steps=max_iteration_steps,
                  down_sample=0.08,
                  filter_nb_neighbors=False,
                  filter_std_ratio=False,
                  debug=False,
                  verbose=False)
