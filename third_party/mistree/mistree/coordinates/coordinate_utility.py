# 'coordinate_utility.py' contains a number of coordinate transformation functions.

import numpy as np


def spherical_2_cartesian(r, phi, theta, units='degrees'):
    """Converts spherical polar coordinates into cartesian coordinates.

    Parameters
    ----------
    r : array
        Radial distance.
    phi : array
        Longitudinal coordinates (radians = [0, 2*pi]).
    theta : array
        Latitude coordinates (radians = [0, pi]).
    units : str
        Units of phi and theta given in either degrees or radians.

    Returns
    -------
    x, y, z : array
        'euclidean': euclidean coordinates.
    """
    phi = np.copy(phi)
    theta = np.copy(theta)
    if units == 'degrees':
        phi, theta = np.deg2rad(phi), np.deg2rad(theta)
    elif units == 'radians':
        pass
    else:
        raise AssertionError("Unexpected value entered for 'units', only supports either degrees or radians", units)
    x = r * np.cos(phi) * np.sin(theta)
    y = r * np.sin(phi) * np.sin(theta)
    z = r * np.cos(theta)
    return x, y, z


def celestial_2_cartesian(r, ra, dec, units='degrees', output='both'):
    """Converts spherical polar coordinates into cartesian coordinates.

    Parameters
    ----------
    r : array
        Radial distance.
    ra : array
        Longitudinal celestial coordinates.
    dec : array
        Latitude celestial coordinates.
    units : str
        Units of ra and dec given in either degrees or radians.
    output : {'cartesian', 'both'}, optional
        Determines whether to output only the cartesian or both cartesian and spherical coordinates.

    Returns
    -------
    phi, theta : array
        'spherical': spherical polar coordinates.
    x, y, z : array
        'cartesian': euclidean coordinates.
    """
    phi = np.copy(ra)
    theta = np.copy(dec)
    if units == 'degrees':
        phi, theta = np.deg2rad(phi), np.deg2rad(theta)
    elif units == 'radians':
        pass
    else:
        raise AssertionError("Unexpected value entered for 'units', only supports either degrees or radians", units)
    theta = np.pi / 2. - theta
    if output == 'spherical':
        return phi, theta
    else:
        x = r * np.cos(phi) * np.sin(theta)
        y = r * np.sin(phi) * np.sin(theta)
        z = r * np.cos(theta)
        if output == 'cartesian':
            return x, y, z
        elif output == 'both':
            return phi, theta, x, y, z
        else:
            raise AssertionError("Unexpected value entered for 'output', should be either 'cartesian' or 'both'.",
                                 output)


def spherical_2_unit_sphere(phi, theta, units='degrees'):
    """Project coordinates on a sphere into cartesian coordinates on a unit sphere.

    Parameters
    ----------
    phi : array
        Longitudinal coordinates (radians = [0, 2*pi]).
    theta : array
        Latitude coordinates (radians = [0, pi]).
    units : {'degrees', 'radians'}, optional
        Units of phi and theta given in either 'degrees' or 'radians'.

    Returns
    -------
    x, y, z : array
        cartesian coordinates.
    """
    if np.isscalar(phi) is True:
        return spherical_2_cartesian(1., phi, theta, units=units)
    else:
        return spherical_2_cartesian(np.ones(len(phi)), phi, theta, units=units)


def celestial_2_unit_sphere(ra, dec, units='degrees', output='both'):
    """Project coordinates on a sphere into cartesian coordinates on a unit sphere.

    Parameters
    ----------
    ra : array
        Longitudinal celestial coordinates.
    dec : array
        Latitude celestial coordinates.
    units : {'degrees', 'radians'}, optional
        Units of ra and dec given in either 'degrees' or 'radians'.
    output : {'cartesian', 'both'}, optional
        Determines whether to output only the euclidean or both euclidean and spherical coordinates.

    Returns
    -------
    phi, theta : array
        'spherical': spherical polar coordinates.
    x, y, z : array
        'cartesian': cartesian coordinates.
    """
    if np.isscalar(ra) is True:
        return celestial_2_cartesian(1., ra, dec, units=units, output=output)
    else:
        return celestial_2_cartesian(np.ones(len(ra)), ra, dec, units=units, output=output)


def perpendicular_distance_2_angle(distance):
    """Converts distances on a unit sphere to angular distances projected across a unit sphere.

    Parameters
    ----------
    distance : array
        Perpendicular distances across (i.e. going on the surface) of a unit sphere.

    Returns
    -------
    angular_distance : array
        The angular distance of points across a unit sphere.
    """
    angular_distance = 2. * np.arcsin(distance / 2.)
    return angular_distance
