#'graph.py' contains functions for switching how the edges of a graph are stored,
# i.e. either as a csr_matrix or a 2 dimensional array.

from scipy.sparse import csr_matrix


def graph2data(graph):
    """Returns the index and data of a sparse matrix.

    Parameters
    ----------
    graph : csr_matrix
        A sparse matrix of the edges in a graph and corresponding node indexes.

    Returns
    -------
    index1, index2 : array
        Arrays of the node indices of the graph.
    weights : array
        An array of the weights of each edge in the graph.
    """
    graph = graph.tocoo()
    weights = graph.data
    index1 = graph.row
    index2 = graph.col
    return index1, index2, weights


def data2graph(index1, index2, weights, num_nodes):
    """Returns the sparse matrix of a graph given the indices and data.

    Parameters
    ----------
    index1, index2 : array
        Arrays of the node indices of the graph.
    weights : array
        An array of the weights of each edge in the graph.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    graph : csr_matrix
        A sparse matrix of the edges in a graph and corresponding node indexes.
    """
    graph = csr_matrix((weights, (index1, index2)), shape=(num_nodes, num_nodes))
    return graph
