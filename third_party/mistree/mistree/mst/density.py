# 'density.py' compares the density of particles in cells to another variable.

import numpy as np
from . import utility_density


def variable_vs_density(x, y, dx, x_param, y_param, param, box_size, z=None, z_param=None, mode='2D',
                        get_density=False):
    """ Finds the relationship between a variable and the density of points.

    Parameters
    ----------
    x, y, z : array
        The positions of points.
    x_param, y_param, z_param : array
        The positions associated with the measured parameter.
    param : array
        An array of the input parameter.
    dx : float
        The size of each cell in each axis.
    box_size : float
        The size of the box along one axis.
    mode : {'2D', '3D'}
        Specified whether the data is two or three dimensional
    get_density : bool, optional
        If true will output the density of particles.

    Returns
    -------
    mean_param : array
        The mean of the parameter in each cell.
    density : array
        The density of points in each cell.

    To do
    -----
    Extend for use on tomographic or spherical polar data sets.
    """
    xx = np.arange(dx / 2., box_size + dx / 2., dx)
    condition = np.where(xx < box_size)[0]
    xx = xx[condition]
    length = len(condition)
    if mode == '2D':
        x_div, y_div = np.meshgrid(xx, xx, indexing='ij')
        x_div, y_div = np.ndarray.flatten(x_div), np.ndarray.flatten(y_div)
        mean_param = utility_density.get_param_2d(x_param, y_param, param, dx, length, len(x_div), len(x_param))
        if get_density is True:
            counts = utility_density.get_counts_2d(x, y, dx, length, len(x_div), len(x))
            density = counts / np.mean(counts)
            return mean_param, density
        else:
            return mean_param
    else:
        x_div, y_div, z_div = np.meshgrid(xx, xx, xx, indexing='ij')
        x_div, y_div, z_div = np.ndarray.flatten(x_div), np.ndarray.flatten(y_div), np.ndarray.flatten(z_div)
        mean_param = utility_density.get_param_3d(x_param, y_param, z_param, param, dx, length, len(x_div),
                                                  len(x_param))
        if get_density is True:
            counts = utility_density.get_counts_3d(x, y, z, dx, length, len(x_div), len(x))
            density = counts / np.mean(counts)
            return mean_param, density
        else:
            return mean_param
