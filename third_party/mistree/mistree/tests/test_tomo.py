import numpy as np
import mistree as mist


def test_convert_tomo_knn_length2angle():
    phi = 360.*np.random.random_sample(50)
    theta = 180.*np.random.random_sample(50)
    x, y, z = mist.spherical_2_unit_sphere(phi, theta)
    x, y, z, knn, num_removed_edges_fraction = mist.k_nearest_neighbour_scale_cut(x, y, 0., 10, z=z)
    knn_tomo = mist.convert_tomo_knn_length2angle(knn, 50)
    index1, index2, distances = mist.graph2data(knn)
    index1, index2, angle_distances = mist.graph2data(knn_tomo)
    condition = np.where(angle_distances == 2. * np.arcsin(distances / 2.))[0]
    assert len(condition) == len(distances)
