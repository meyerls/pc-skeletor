#'partition.py' is used to partition input data sets.

import numpy as np


def partition_data(total_sample_size, num_groups):
    """Returns an array with values that correspond to the assigned groupings.

    Parameters
    ----------
    total_sample_size : int
        Length of the original data set.
    num_groups: int
        The number of groups/partitions the data is divided into.

    Returns
    -------
    groups : array
        An array of the same size as the total sample, indicating which group
        each element is assigned to. Unassigned points are given a group of -1,
        which occurs if the number of groups does not exactly divide the total_sample_size
    """
    individual_sample_size = total_sample_size/num_groups
    indices = np.arange(0, total_sample_size, 1)
    groups = np.zeros(len(indices))
    for i in range(1, num_groups):
        groups[int(individual_sample_size*i):int(individual_sample_size*(i+1))] = i
    if total_sample_size % num_groups != 0:
        groups[individual_sample_size*num_groups:] = -1
    randoms = np.random.random_sample(len(groups))
    ordered_randoms_and_reordered_groups = np.array(sorted(zip(randoms, groups)))
    groups = ordered_randoms_and_reordered_groups[:, 1]
    return groups


def get_index_for_group(groups, which_group):
    """Returns the corresponding indices for a specified group/partition.

    Paramaters
    ----------
    groups : array
        An array of the assigned groupings for each element of a given data set.
    which_group : int
        The specific group we need the indices for.

    Returns
    -------
    group_indexes : array
        The indexes for a specified group.
    """
    group_indexes = np.where(groups == float(which_group))[0]
    return group_indexes
