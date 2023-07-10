# 'branches.py' finds and measures the length and shapes of branches in from a
# given constructed MST.

import os
import sys

dir_path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(dir_path[:-3] + '/coordinates/')

import numpy as np
import coordinate_utility


def get_branch_index(edge_index, edge_degree, branch_cutting_frequency=1000):
    """Finds the branch indexes for each branch in the MST.

    Parameters
    ----------
    edge_index : array
        The node index of the ends of each edge.
    edge_degree : array
        The degree for the ends of each edge.
    branch_cutting_frequency : int, optional
        An optimisation parameter, used to remove edges that have already been placed into a branch.
        This significantly improves the speed of the algorithm as branches that are already constructed
        are now removed from the branch finder.

    Returns
    -------
    branch_index : list
        A list of branches where each branch is a list of the edge index of edges contained in each branch.
    branch_index_rejected : list
        A list of branches that have not been completed. This will occur only if a subset of the edge indexes
        of the full tree is provided.
    """
    degree1 = edge_degree[0]
    degree2 = edge_degree[1]
    index1 = edge_index[0]
    index2 = edge_index[1]
    condition = np.where((degree1 == 2.) & (degree2 == 2.))[0]
    index_branch_mid = condition
    index_branch_mid1 = index1[index_branch_mid]
    index_branch_mid2 = index2[index_branch_mid]
    condition = np.where(((degree1 == 2.) & (degree2 != 2.)) | ((degree1 != 2.) & (degree2 == 2.)))[0]
    index_branch_end = condition
    index_branch_end1 = index1[index_branch_end]
    index_branch_end2 = index2[index_branch_end]
    degree_branch_end1 = degree1[index_branch_end]
    degree_branch_end2 = degree2[index_branch_end]
    check_mid = np.ones(len(index_branch_mid))
    check_end = np.ones(len(index_branch_end))
    branch_index = []
    branch_index_rejected = []
    mask_end = np.ones(index_branch_end.shape, dtype=bool)
    mask_mid = np.ones(index_branch_mid.shape, dtype=bool)
    count = 0
    item = 0
    while item < len(index_branch_end):
        if check_end[item] == 1.:
            check_end[item] = 0.
            done = 0.
            _twig = []
            _twig.append(index_branch_end[item])
            if degree_branch_end1[item] == 2.:
                node_index = index_branch_end1[item]
            elif degree_branch_end2[item] == 2.:
                node_index = index_branch_end2[item]
            else:
                assert ValueError("branch edge incorrect.")
            mask_end[item] = False
            while done == 0.:
                condition = np.where(((check_mid == 1.) & (index_branch_mid1 == node_index)) |
                                     ((check_mid == 1.) & (index_branch_mid2 == node_index)))[0]
                if len(condition) == 0:
                    condition = np.where(((check_end == 1.) & (index_branch_end1 == node_index)) |
                                         ((check_end == 1.) & (index_branch_end2 == node_index)))[0]
                    if len(condition) == 0:
                        branch_index_rejected = branch_index_rejected + \
                                                np.ndarray.tolist(np.ndarray.flatten(np.array(_twig)))
                        done = 1.
                    else:
                        check_end[condition] = 0.
                        _twig.append(index_branch_end[condition][0])
                        done = 1.
                        mask_end[condition] = False
                        branch_index.append(np.ndarray.tolist(np.ndarray.flatten(np.array(_twig))))
                else:
                    if len(condition) == 1:
                        check_mid[condition] = 0.
                        _twig.append(index_branch_mid[condition][0])
                        if index_branch_mid1[condition] == node_index:
                            node_index = index_branch_mid2[condition]
                        elif index_branch_mid2[condition] == node_index:
                            node_index = index_branch_mid1[condition]
                        else:
                            assert ValueError("Identification error.")
                        mask_mid[condition] = False
                    else:
                        assert ValueError("Found more than one vertex.")
        else:
            pass
        if count % branch_cutting_frequency == 0 and count != 0:
            index_branch_end = index_branch_end[mask_end]
            check_end = check_end[mask_end]
            index_branch_end1 = index_branch_end1[mask_end]
            index_branch_end2 = index_branch_end2[mask_end]
            degree_branch_end1 = degree_branch_end1[mask_end]
            degree_branch_end2 = degree_branch_end2[mask_end]
            index_branch_mid = index_branch_mid[mask_mid]
            check_mid = check_mid[mask_mid]
            index_branch_mid1 = index_branch_mid1[mask_mid]
            index_branch_mid2 = index_branch_mid2[mask_mid]
            mask_end = mask_end[mask_end]
            mask_mid = mask_mid[mask_mid]
            count = count + 1
            item = 0
        elif count % 1001 == 0:
            count = count + 1
            item = item + 1
        elif item == len(index_branch_end) - 1:
            index_branch_end = index_branch_end[mask_end]
            check_end = check_end[mask_end]
            index_branch_end1 = index_branch_end1[mask_end]
            index_branch_end2 = index_branch_end2[mask_end]
            degree_branch_end1 = degree_branch_end1[mask_end]
            degree_branch_end2 = degree_branch_end2[mask_end]
            index_branch_mid = index_branch_mid[mask_mid]
            check_mid = check_mid[mask_mid]
            index_branch_mid1 = index_branch_mid1[mask_mid]
            index_branch_mid2 = index_branch_mid2[mask_mid]
            mask_end = mask_end[mask_end]
            mask_mid = mask_mid[mask_mid]
            count = count + 1
            item = 0
        else:
            count = count + 1
            item = item + 1
    branch_index_rejected = branch_index_rejected + np.ndarray.tolist(np.ndarray.flatten(np.array(index_branch_mid)))
    branch_index = [np.ndarray.tolist(np.hstack(np.array(branch_index[i]))) for i in range(0, len(branch_index))]
    if len(branch_index_rejected) != 0:
        branch_index_rejected = np.ndarray.tolist(np.hstack(np.array(branch_index_rejected)))
    return branch_index, branch_index_rejected


def get_branch_index_sub_divide(sub_divisions, edge_index, edge_degree, box_size=None, edge_x=None, edge_y=None,
                                edge_z=None, phi=None, theta=None, edge_phi=None, edge_theta=None,
                                branch_cutting_frequency=1000, mode='Euclidean', two_dimension=True):
    """ Finds the length of branches for large sets of data where a rapid increase in speed is achieved by subdividing
    the full data set and finding branches in each sub division and then completing branches that straddle across the
    sub divides.

    Parameters
    ----------
    sub_divisions : int
        The number of divisions used to divide the data set in each axis. A significant boost in speed is achieved.
    edge_degree : array
        The degree for the ends of each edge.
    edge_index : array
        The node index of the ends of each edge.
    box_size : {'None', float}, optional
        The size of the '2D' or '3D' box. If undefined, this will be determined by the maximum x value.
    edge_x, edge_y, edge_z : array
        The cartesian coordinates of the nodes at each end of every edge.
    phi, theta : array
        The spherical coordinates of the nodes on the sphere.
    edge_phi, edge_theta : array
        The spherical coordinates of the nodes at each end of every edge.
    branch_cutting_frequency : int
        An optimisation parameter, used to remove edges that have already been placed into a branch.
        This significantly improves the speed of the algorithm as branches that are already constructed
        are now removed from the branch finder.
    mode : {'2D', '3D', string}, optional
        '2D', '3D' or assumed to be in spherical coordinates.
    two_dimension : bool, optional
        Determines whether the data set is 2D of 3D.

    Returns
    -------
    branch_index : list
        A list of branches where each branch is a list of the edge index of edges contained in each branch.
    branch_index_rejected : list
        A list of branches that have not been completed. This will occur only if a subset of the edge indexes
        of the full tree is provided.
    """
    if mode == 'Euclidean':
        x_min, x_max, y_min, y_max = np.min(edge_x), np.max(edge_x), np.min(edge_y), np.max(edge_y)
        dx, dy = (x_max - x_min) / float(sub_divisions), (y_max - y_min) / float(sub_divisions)
        xx = np.arange(x_min, x_max + dx, dx)
        yy = np.arange(y_min, y_max + dy, dy)
        xx_mid = 0.5 * (xx[1:len(xx)] + xx[0:len(xx) - 1])
        yy_mid = 0.5 * (yy[1:len(yy)] + yy[0:len(yy) - 1])
        if two_dimension is True:
            x_div, y_div = np.meshgrid(xx_mid, yy_mid, indexing='ij')
            x_div = np.ndarray.flatten(x_div)
            y_div = np.ndarray.flatten(y_div)
        else:
            z_min, z_max = np.min(edge_z), np.max(edge_z)
            dz = (z_max - z_min) / float(sub_divisions)
            zz = np.arange(z_min, z_max + dz, dz)
            zz_mid = 0.5 * (zz[1:len(zz)] + zz[0:len(zz) - 1])
            x_div, y_div, z_div = np.meshgrid(xx_mid, yy_mid, zz_mid, indexing='ij')
            x_div = np.ndarray.flatten(x_div)
            y_div = np.ndarray.flatten(y_div)
            z_div = np.ndarray.flatten(z_div)
    else:
        phi_min, phi_max, theta_min, theta_max = np.min(phi), np.max(phi), np.min(theta), np.max(theta)
        dphi, dtheta = (phi_max - phi_min) / float(sub_divisions), (theta_max - theta_min) / float(sub_divisions)
        phiphi = np.arange(phi_min, phi_max + dphi, dphi)
        thetatheta = np.arange(theta_min, theta_max + dtheta, dtheta)
        pp = 0.5 * (phiphi[1:len(phiphi)] + phiphi[0:len(phiphi) - 1])
        tt = 0.5 * (thetatheta[1:len(thetatheta)] + thetatheta[0:len(thetatheta) - 1])
        phi_div, theta_div = np.meshgrid(pp, tt, indexing='ij')
        phi_div = np.ndarray.flatten(phi_div)
        theta_div = np.ndarray.flatten(theta_div)
    branch_index_total = []
    branch_index_rejected_total = []
    if mode == 'Euclidean':
        length = len(x_div)
        total_mask = np.ones(len(edge_x[0]))
    else:
        length = len(phi_div)
        total_mask = np.ones(len(edge_phi[0]))
    for k in range(0, length):
        if mode == 'Euclidean' and two_dimension is True:
            xd, yd = x_div[k], y_div[k]
            condition = np.where((total_mask == 1.) & ((edge_x[0] >= xd - dx / 2.) | (edge_x[0] <= xd + dx / 2.) |
                                 (edge_y[0] >= yd - dx / 2.) | (edge_y[0] <= yd + dx / 2.)))[0]
        elif mode == 'Euclidean' and two_dimension is False:
            xd, yd, zd = x_div[k], y_div[k], z_div[k]
            condition = np.where((total_mask == 1.) & ((edge_x[0] >= xd - dx / 2.) | (edge_x[0] <= xd + dx / 2.) |
                                 (edge_y[0] >= yd - dx / 2.) | (edge_y[0] < yd + dx / 2.) | (edge_z[0] >= zd - dx / 2.)
                                 & (edge_z[0] <= zd + dx / 2.)))[0]
        else:
            pd, td = phi_div[k], theta_div[k]
            condition = np.where((total_mask == 1.) & ((edge_phi[0] >= pd - dphi / 2.) | (edge_phi[0] <= pd + dphi / 2.)
                                 | (edge_theta[0] >= td - dtheta / 2.) | (edge_theta[0] <= td + dtheta / 2.)))[0]
        edge_degree_cut = np.array([edge_degree[0][condition], edge_degree[1][condition]])
        edge_index_cut = np.array([edge_index[0][condition], edge_index[1][condition]])
        branch_index_cut, branch_index_rejected_cut = \
            get_branch_index(edge_index_cut, edge_degree_cut, branch_cutting_frequency=branch_cutting_frequency)
        branch_index_cut_corrected = [np.ndarray.tolist(condition[i]) for i in branch_index_cut]
        branch_index_total = branch_index_total + branch_index_cut_corrected
        total_mask[[item for sublist in branch_index_total for item in sublist]] = 0.
        if len(branch_index_rejected_cut) != 0:
            branch_index_rejected_total = branch_index_rejected_total + \
                                          np.ndarray.tolist(condition[branch_index_rejected_cut])
        total_mask[branch_index_rejected_total] = 0.
    branch_index_rejected_total = np.array(branch_index_rejected_total)
    branch_index_rejected_total = np.unique(branch_index_rejected_total)
    if len(branch_index_rejected_total) == 0:
        pass
    else:
        edge_degree_rejected = np.array([edge_degree[0][branch_index_rejected_total],
                                         edge_degree[1][branch_index_rejected_total]])
        edge_index_rejected = np.array([edge_index[0][branch_index_rejected_total],
                                        edge_index[1][branch_index_rejected_total]])
        branch_index_left_over, branch_index_rejected_left_over = \
            get_branch_index(edge_index_rejected, edge_degree_rejected, branch_cutting_frequency)
        branch_index_left_over_corrected = \
            [np.ndarray.tolist(branch_index_rejected_total[i]) for i in branch_index_left_over]
        branch_index_total = branch_index_total + branch_index_left_over_corrected
        branch_index_rejected_total = branch_index_rejected_left_over
    return branch_index_total, branch_index_rejected_total


def get_branch_end_index(edge_index, edge_degree, branch_index):
    """Gets the index of the nodes at the extreme end of each branch.

    Parameters
    ----------
    edge_index : array
        The node index of the ends of each edge.
    edge_degree : array
        The degree for the ends of each edge.
    branch_index : list
        A list of branches. Listing the indices of edges within each branch.

    Returns
    -------
    branch_end : array
        The index of the nodes at the ends of each branch.
    """
    branch_edge_index_end1 = [i[0] for i in branch_index]
    branch_edge_index_end2 = [i[len(i) - 1] for i in branch_index]
    edge_degree_end12 = edge_degree[1][branch_edge_index_end1]
    index11 = edge_index[0][branch_edge_index_end1]
    index12 = edge_index[1][branch_edge_index_end1]
    condition = np.where(edge_degree_end12 != 2.)[0]
    branch_index_end1 = np.copy(index11)
    branch_index_end1[condition] = index12[condition]
    edge_degree22 = edge_degree[1][branch_edge_index_end2]
    index21 = edge_index[0][branch_edge_index_end2]
    index22 = edge_index[1][branch_edge_index_end2]
    condition = np.where(edge_degree22 != 2.)[0]
    branch_index_end2 = np.copy(index21)
    branch_index_end2[condition] = index22[condition]
    branch_end = np.array([branch_index_end1, branch_index_end2])
    return branch_end


def get_branch_edge_count(branch_index):
    """Finds the number of edges included in each branch.
    """
    branch_edge_count = [float(len(i)) for i in branch_index]
    branch_edge_count = np.array(branch_edge_count)
    return branch_edge_count


def get_branch_shape(edge_index, edge_degree, branch_index, branch_length, mode='2D', x=None, y=None, z=None):
    """Finds the shape of all branches. This is simply the straight line distance between the two ends divided by
    the branch length.
    """
    branch_index_end = get_branch_end_index(edge_index, edge_degree, branch_index)
    branch_index_end1, branch_index_end2 = branch_index_end[0], branch_index_end[1]
    if mode == '2D':
        dx = abs(x[branch_index_end1] - x[branch_index_end2])
        dy = abs(y[branch_index_end1] - y[branch_index_end2])
        branch_end_length = np.sqrt((dx ** 2.) + (dy ** 2.))
    elif mode == '3D' or mode == 'spherical polar' or mode == 'spherical polar celestial':
        dx = abs(x[branch_index_end1] - x[branch_index_end2])
        dy = abs(y[branch_index_end1] - y[branch_index_end2])
        dz = abs(z[branch_index_end1] - z[branch_index_end2])
        branch_end_length = np.sqrt((dx ** 2.) + (dy ** 2.) + (dz ** 2.))
    elif mode == 'tomographic' or mode == 'tomographic celestial':
        dx = abs(x[branch_index_end1] - x[branch_index_end2])
        dy = abs(y[branch_index_end1] - y[branch_index_end2])
        dz = abs(z[branch_index_end1] - z[branch_index_end2])
        branch_end_length = np.sqrt((dx ** 2.) + (dy ** 2.) + (dz ** 2.))
        branch_end_length = coordinate_utility.perpendicular_distance_2_angle(branch_end_length)
    branch_shape = branch_end_length/branch_length
    return branch_shape
