import pc_skeletor
from pc_skeletor import skeletor
from pc_skeletor.download import Dataset
import numpy as np

# Download test tree dataset
downloader = Dataset()
downloader.download_tree_dataset()

# Init tree skeletonizer
skeletor = skeletor.Skeletonizer(point_cloud=downloader.file_path,
                                 down_sample=0.01,
                                 debug=False)
laplacian_config = {"MAX_LAPLACE_CONTRACTION_WEIGHT": 1024,
                    "MAX_POSITIONAL_WEIGHT": 1024,
                    "INIT_LAPLACIAN_SCALE": 100}
skeleton, graph, skeleton_graph = skeletor.extract(method='Laplacian', config=laplacian_config)
# save results
skeletor.save(result_folder='./data/')
# Make animation of original point cloud and skeleton
skeletor.animate(init_rot=np.asarray([[1, 0, 0],
                                      [0, 0, 1],
                                      [0, 1, 0]]), steps=200, out='./data/')
# Interactive visualization
skeletor.visualize()
