import numpy as np
import mistree as mist


def test_partition_data():
    groups = mist.mst.partition.partition_data(10, 2)
    assert len(np.unique(groups)) == 2
    groups = mist.mst.partition.partition_data(10, 5)
    assert len(np.unique(groups)) == 5


def test_get_index_for_group():
    groups = np.array([0., 1., 0., 1., 2.])
    index = mist.mst.partition.get_index_for_group(groups, 0)
    assert index[0] == 0
    assert index[1] == 2
    index = mist.mst.partition.get_index_for_group(groups, 1)
    assert index[0] == 1
    assert index[1] == 3
    index = mist.mst.partition.get_index_for_group(groups, 2)
    assert index[0] == 4
