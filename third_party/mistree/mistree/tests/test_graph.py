import numpy as np
import mistree as mist


def test_graph_conversion():
    x = np.random.random_sample(50)
    y = np.random.random_sample(50)
    mst = mist.GetMST(x=x, y=y)
    mst.construct_mst()
    graph = mist.data2graph(mst.edge_index[0], mst.edge_index[1], mst.edge_length, 50)
    index1, index2, edge_length = mist.graph2data(graph)
    condition = np.where(np.sort(index1) == np.sort(mst.edge_index[0]))[0]
    assert len(condition) == len(mst.edge_index[0])
    condition = np.where(np.sort(index2) == np.sort(mst.edge_index[1]))[0]
    assert len(condition) == len(mst.edge_index[1])
    condition = np.where(np.sort(edge_length) == np.sort(mst.edge_length))[0]
    assert len(condition) == len(mst.edge_length)
