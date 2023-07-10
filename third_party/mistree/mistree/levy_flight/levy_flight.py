# 'levy_flight.py' has a number of functions that can generate random levy
# flight-like distributions.

import numpy as np
from . import utility_random_walk as random_walk


def _get_randoms_sphere(size):
    """Generate random points on a sphere.

    Parameters
    ----------
    size : int
        Size of random sample.

    Returns
    -------
    phi : array
        Longitude coordinate.
    theta :array
        Latitude coordinate.
    """
    if size == 1:
        phi = 2.*np.pi*np.random.random_sample(1)[0]
        u = np.random.random_sample(1)[0]
        theta = np.arccos(1.-2.*u)
    else:
        phi = 2.*np.pi*np.random.random_sample(size)
        u = np.random.random_sample(size)
        theta = np.arccos(1.-2.*u)
    return phi, theta


def get_random_flight(steps, mode='3D', box_size=75., periodic=True):
    """Generates a random realisation of a 'Levy flight'-like distribution. The random step sizes are defined by the
    user.

    Parameters
    ----------
    steps : array
        Distribution of step sizes defined by the user.
    mode : {'2D', '3D'}, optional
        Defines whether the distribution is defined in 2D or 3D cartesian coordinates.
    box_size : float
        Length of the periodic box across one axis.
    periodic : bool
        If True then this sets periodic boundary conditions on the Levy flight realisation. If False, the box_size
        parameter is ignored.

    Returns
    -------
    x, y, z : array
        Distribution of random walk particles. z is only outputed if this is a 3D distribution.

    Internal
    --------
    _x_start, _y_start, _z_start : array
        Starting position of the distribution.
    _size : int
        The size of the distribution, 1 + size of the step_sizes array.
    """
    _size = len(steps)
    if periodic is False:
        box_size = None
    if mode == '2D':
        if box_size is None:
            _x_start, _y_start = 0., 0.
            _periodic = 0
            box_size = 0.
        else:
            _x_start = np.random.uniform(0., box_size, 1)[0]
            _y_start = np.random.uniform(0., box_size, 1)[0]
            _periodic = 1
        _phi = np.random.uniform(0., 2. * np.pi, _size)
        x, y = random_walk.random_walk_fast_2d(steps, _phi, box_size, _x_start, _y_start, _periodic, _size)
        return x, y
    elif mode == '3D':
        if box_size is None:
            _x_start, _y_start, _z_start = 0., 0., 0.
            _periodic = 0
            box_size = 0.
        else:
            _x_start = np.random.uniform(0., box_size, 1)[0]
            _y_start = np.random.uniform(0., box_size, 1)[0]
            _z_start = np.random.uniform(0., box_size, 1)[0]
            _periodic = 1
        _phi, _theta = _get_randoms_sphere(_size)
        x, y, z = random_walk.random_walk_fast_3d(steps, _phi, _theta, box_size, _x_start, _y_start, _z_start,
                                                  _periodic, _size)
        return x, y, z


def get_levy_flight(size, mode='3D', periodic=True, box_size=75., t_0=0.2, alpha=1.5):
    """Generates a random realisation of a Levy flight distribution.

    Parameters
    ----------
    size : int
        Number of points.
    mode : {'2D', '3D'}, optional
        Defines whether the distribution is defined in 2D or 3D cartesian coordinates.
    box_size : float, optional
        Length of the periodic box across one axis.
    periodic : bool, optional
        If True then this sets periodic boundary condition on the Levy flight realisation. If False, the box_size parameter is ignored.
    t_0, alpha : float, optional
        Are parameters of the Levy flight model.

    Returns
    -------
    x, y, z : array
        Distribution of Levy flight particles. z is only outputted if this is a 3D distribution.

    Internal
    --------
    _u : array
        Uniform random numbers between 0 and 1.
    _steps : array
        Random step sizes.
    _x_start, _y_start, _z_start : float
        Starting position of the distribution.
    """
    _u = np.random.uniform(0., 1., size - 1)
    _steps = t_0 / (1. - _u) ** (1. / alpha)
    if periodic is False:
        box_size = None
    if mode == '2D':
        x, y = get_random_flight(_steps, mode='2D', box_size=box_size)
        return x, y
    elif mode == '3D':
        x, y, z = get_random_flight(_steps, mode='3D', box_size=box_size)
        return x, y, z
    else:
        print("Error: Mode is unsupported. Only allowed modes are '2D' or '3D'.")


def get_adjusted_levy_flight(size, mode='3D', periodic=True, box_size=75.,
                             t_0=0.325, t_s=0.015, alpha=1.5, beta=0.45, gamma=1.3):
    """Generates a random realisation of an adjusted Levy flight (ALF) distribution.

    Parameters
    ----------
    size : int
        Number of points.
    mode : {'2D', '3D'}, optional
        Defines whether the distribution is defined in 2D or 3D cartesian coordinates.
    box_size : float, optional
        Length of the periodic box across one axis.
    periodic : bool, optional
        If True then this sets periodic boundary condition on the Levy flight realisation. If False, the box_size parameter is ignored.
    t_0, t_s, alpha, beta, gamma : float
        Are parameters of the adjusted levy walk model.

    Returns
    -------
    x, y, z : array
        Distribution of adjusted Levy flight particles. z is only outputted if this is a 3D distribution.

    Internal
    --------
    _u : array
        Uniform random numbers between 0 and 1.
    _steps : array
        Random step sizes.
    _x_start, _y_start, _z_start : float
        Starting position of the distribution.
    """
    if gamma is None:
        # If gamma is not given, then it can be calculated by imposing the condition that the probability distribution
        # function remains smooth across t_0.
        gamma = alpha*((1.-beta)/beta)*((t_0-t_s)/t_0)
    else:
        pass
    _u = np.random.uniform(0., 1., size - 1)
    _steps = (t_0 - t_s) * (_u / beta) ** (1. / gamma) + t_s
    condition = np.where(_u >= beta)[0]
    _steps[condition] = t_0*(1.+((beta-_u[condition])/(1.-beta)))**(-1./alpha)
    if periodic is False:
        box_size = None
    if mode == '2D':
        x, y = get_random_flight(_steps, mode='2D', box_size=box_size)
        return x, y
    elif mode == '3D':
        x, y, z = get_random_flight(_steps, mode='3D', box_size=box_size)
        return x, y, z
    else:
        print("Error: Mode is unsupported. Only allowed modes are '2D' or '3D'.")
