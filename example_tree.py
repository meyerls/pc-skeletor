import open3d as o3d
import numpy as np

from pc_skeletor import SLBC, LBC
from pc_skeletor import Dataset

if __name__ == "__main__":
    downloader = Dataset()
    trunk_pcd_path, branch_pcd_path = downloader.download_semantic_tree_dataset()

    pcd_trunk = o3d.io.read_point_cloud(trunk_pcd_path)
    pcd_branch = o3d.io.read_point_cloud(branch_pcd_path)
    pcd = pcd_trunk + pcd_branch

    # Laplacian-based Contraction
    lbc = LBC(point_cloud=pcd,
              init_contraction=2,
              init_attraction=0.2,
              down_sample=0.005)
    lbc.extract_skeleton()
    lbc.extract_topology()
    #lbc.visualize()
    lbc.save('./output')
    lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=500, output='./output')

    # Semantic Laplacian-based Contraction
    s_lbc = SLBC(point_cloud={'trunk': pcd_trunk, 'branches': pcd_branch},
                 semantic_weighting=30.,
                 init_contraction=2,
                 init_attraction=0.2,
                 down_sample=0.005)
    s_lbc.extract_skeleton()
    s_lbc.extract_topology()
    #s_lbc.visualize()
    s_lbc.save('./output')
    s_lbc.animate(init_rot=np.asarray([[1, 0, 0], [0, 0, 1], [0, 1, 0]]), steps=500, output='./output')
