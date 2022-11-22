# Own modules
from skeletor import *
from download import *

if __name__ == "__main__":
    downloader = Dataset()
    downloader.download_tree_dataset()

    # Init tree skeletonizer
    skeletor = Skeletonizer(point_cloud=downloader.file_path,
                            down_sample=0.05,
                            debug=False)
    laplacian_config = {"MAX_LAPLACE_CONTRACTION_WEIGHT": 1024,
                        "MAX_POSITIONAL_WEIGHT": 1024,
                        "INIT_LAPLACIAN_SCALE": 100}
    sceleton = skeletor.extract(method='Laplacian', config=laplacian_config)
