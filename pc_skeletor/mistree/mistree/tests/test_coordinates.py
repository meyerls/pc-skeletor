import numpy as np
import mistree as mist


def test_spherical_2_cartesian():
    x, y, z = mist.spherical_2_cartesian(1., 0., 0.)
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.spherical_2_cartesian(1., 90., 90.)
    assert round(x, 6) == 0.
    assert y == 1.
    assert round(z, 6) == 0.
    x, y, z = mist.spherical_2_cartesian(1., 90., 45.)
    assert round(x, 6) == 0.
    assert round(y, 6) == round(1./np.sqrt(2.), 6)
    assert round(z, 6) == round(1./np.sqrt(2.), 6)
    x, y, z = mist.spherical_2_cartesian(1., 0., 0., units='radians')
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.spherical_2_cartesian(1., np.pi/2., np.pi/2., units='radians')
    assert round(x, 6) == 0.
    assert y == 1.
    assert round(z, 6) == 0.
    x, y, z = mist.spherical_2_cartesian(1., np.pi/2., np.pi/4., units='radians')
    assert round(x, 6) == 0.
    assert round(y, 6) == round(1./np.sqrt(2.), 6)
    assert round(z, 6) == round(1./np.sqrt(2.), 6)
    x, y, z = mist.spherical_2_cartesian(np.random.random_sample(100),
                                         360.*np.random.random_sample(100),
                                         180.*np.random.random_sample(100))
    assert len(x) == 100
    assert len(y) == len(x)
    assert len(z) == len(y)


def test_celestial_2_cartesian():
    p = mist.celestial_2_cartesian(1., 0., 0.)
    assert len(p) == 5
    p = mist.celestial_2_cartesian(1., 0., 0., output='cartesian')
    assert len(p) == 3
    p = mist.celestial_2_cartesian(1., 0., 0., output='spherical')
    assert len(p) == 2
    phi, theta, x, y, z = mist.celestial_2_cartesian(1., 0., 90.)
    assert phi == 0.
    assert theta == 0.
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.celestial_2_cartesian(1., 0., 90., output='cartesian')
    assert x == 0.
    assert y == 0.
    assert z == 1.
    phi, theta, x, y, z = mist.celestial_2_cartesian(1., 0., 0.)
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/2., 6)
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    x, y, z = mist.celestial_2_cartesian(1., 0., 0., output='cartesian')
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    phi, theta, x, y, z = mist.celestial_2_cartesian(1., 0., 45.)
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/4., 6)
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    x, y, z = mist.celestial_2_cartesian(1., 0., 45., output='cartesian')
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    phi, theta, x, y, z = mist.celestial_2_cartesian(1., 0., np.pi/2., units='radians')
    assert phi == 0.
    assert theta == 0.
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.celestial_2_cartesian(1., 0., np.pi/2., output='cartesian', units='radians')
    assert x == 0.
    assert y == 0.
    assert z == 1.
    phi, theta, x, y, z = mist.celestial_2_cartesian(1., 0., 0., units='radians')
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/2., 6)
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    x, y, z = mist.celestial_2_cartesian(1., 0., 0., output='cartesian', units='radians')
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    phi, theta, x, y, z = mist.celestial_2_cartesian(1., 0., np.pi/4., units='radians')
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/4., 6)
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    x, y, z = mist.celestial_2_cartesian(1., 0., np.pi/4., output='cartesian', units='radians')
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    phi, theta, x, y, z = mist.celestial_2_cartesian(np.random.random_sample(100),
                                                     360.*np.random.random_sample(100),
                                                     180.*np.random.random_sample(100) - 90.)
    assert len(phi) == 100
    assert len(theta) == len(phi)
    assert len(x) == len(theta)
    assert len(y) == len(x)
    assert len(z) == len(x)


def test_spherical_2_unit_sphere():
    x, y, z = mist.spherical_2_unit_sphere(0., 0.)
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.spherical_2_unit_sphere(0., 90.)
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    x, y, z = mist.spherical_2_unit_sphere(90., 90.)
    assert round(x, 6) == 0.
    assert y == 1.
    assert round(z, 6) == 0.
    x, y, z = mist.spherical_2_unit_sphere(45., 45.)
    assert x == 0.5
    assert round(y, 6) == 0.5
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    x, y, z = mist.spherical_2_unit_sphere(0., 0., units='radians')
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.spherical_2_unit_sphere(0., np.pi/2., units='radians')
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    x, y, z = mist.spherical_2_unit_sphere(np.pi/2., np.pi/2., units='radians')
    assert round(x, 6) == 0.
    assert y == 1.
    assert round(z, 6) == 0.
    x, y, z = mist.spherical_2_unit_sphere(np.pi/4., np.pi/4., units='radians')
    assert x == 0.5
    assert round(y, 6) == 0.5
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    x, y, z = mist.spherical_2_unit_sphere(180.*np.random.random_sample(100), 360.*np.random.random_sample(100))
    assert len(x) == len(y)
    assert len(y) == len(z)


def test_celestial_2_unit_sphere():
    p = mist.celestial_2_unit_sphere(0., 0.)
    assert len(p) == 5
    p = mist.celestial_2_unit_sphere(0., 0., output='cartesian')
    assert len(p) == 3
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(0., 90.)
    assert phi == 0.
    assert theta == 0.
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.celestial_2_unit_sphere(0., 90., output='cartesian')
    assert x == 0.
    assert y == 0.
    assert z == 1.
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(0., 0.)
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/2., 6)
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    x, y, z = mist.celestial_2_unit_sphere(0., 0., output='cartesian')
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(0., 45.)
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/4., 6)
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    x, y, z = mist.celestial_2_unit_sphere(0., 45., output='cartesian')
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(0., np.pi/2., units='radians')
    assert phi == 0.
    assert theta == 0.
    assert x == 0.
    assert y == 0.
    assert z == 1.
    x, y, z = mist.celestial_2_unit_sphere(0., np.pi/2., output='cartesian', units='radians')
    assert x == 0.
    assert y == 0.
    assert z == 1.
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(0., 0., units='radians')
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/2., 6)
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    x, y, z = mist.celestial_2_unit_sphere(0., 0., output='cartesian', units='radians')
    assert x == 1.
    assert y == 0.
    assert round(z, 6) == 0.
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(0., np.pi/4., units='radians')
    assert phi == 0.
    assert round(theta, 6) == round(np.pi/4., 6)
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    x, y, z = mist.celestial_2_unit_sphere(0., np.pi/4., output='cartesian', units='radians')
    assert round(x, 6) == round(np.sqrt(2.)/2., 6)
    assert y == 0.
    assert round(z, 6) == round(np.sqrt(2.)/2., 6)
    phi, theta, x, y, z = mist.celestial_2_unit_sphere(360.*np.random.random_sample(100),
                                                       180.*np.random.random_sample(100) - 90.)
    assert len(phi) == 100
    assert len(theta) == len(phi)
    assert len(x) == len(theta)
    assert len(y) == len(x)
    assert len(z) == len(x)


def test_perpendicular_distance_2_angle():
    assert round(mist.perpendicular_distance_2_angle(2.), 6) == round(np.pi, 6)
    assert round(mist.perpendicular_distance_2_angle(np.sqrt(2.)), 6) == round(np.pi/2., 6)
