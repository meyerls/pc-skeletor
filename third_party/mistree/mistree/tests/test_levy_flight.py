import numpy as np
import mistree as mist


def test_get_random_flight():
    steps = np.random.random_sample(100)
    p = mist.get_random_flight(steps)
    assert len(p) == 3
    p = mist.get_random_flight(steps, mode='2D')
    assert len(p) == 2
    box_size = 1.
    x, y, z = mist.get_random_flight(steps, box_size=box_size)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size) &
                         (z >= 0.) & (z <= box_size))[0]
    assert len(condition) == len(x)
    box_size = 1.
    steps = 10.*np.ones(10)
    x, y, z = mist.get_random_flight(steps, box_size=box_size, periodic=False)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size) &
                         (z >= 0.) & (z <= box_size))[0]
    assert len(condition) != len(x)


def test_get_levy_flight():
    p = mist.get_levy_flight(10)
    assert len(p) == 3
    p = mist.get_levy_flight(10, mode='2D')
    assert len(p) == 2
    x, y, z = mist.get_levy_flight(10)
    assert len(x) == 10
    assert len(y) == len(x)
    assert len(z) == len(y)
    box_size = 1.
    x, y, z = mist.get_levy_flight(10, box_size=box_size)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size) &
                         (z >= 0.) & (z <= box_size))[0]
    assert len(condition) == len(x)
    box_size = 0.1
    x, y = mist.get_levy_flight(10, mode='2D', box_size=box_size, periodic=False)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size))[0]
    assert len(condition) != len(x)
    box_size = 0.1
    x, y, z = mist.get_levy_flight(10, box_size=box_size, periodic=False)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size) &
                         (z >= 0.) & (z <= box_size))[0]
    assert len(condition) != len(x)


def test_get_adjusted_levy_flight():
    p = mist.get_adjusted_levy_flight(10)
    assert len(p) == 3
    p = mist.get_adjusted_levy_flight(10, mode='2D')
    assert len(p) == 2
    x, y, z = mist.get_adjusted_levy_flight(10)
    assert len(x) == 10
    assert len(y) == len(x)
    assert len(z) == len(y)
    box_size = 1.
    x, y, z = mist.get_adjusted_levy_flight(10, box_size=box_size)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size) &
                         (z >= 0.) & (z <= box_size))[0]
    assert len(condition) == len(x)
    box_size = 0.01
    x, y, z = mist.get_adjusted_levy_flight(10, box_size=box_size, periodic=False)
    condition = np.where((x >= 0.) & (x <= box_size) &
                         (y >= 0.) & (y <= box_size) &
                         (z >= 0.) & (z <= box_size))[0]
    assert len(condition) != len(x)
    x, y, z = mist.get_adjusted_levy_flight(10, box_size=box_size, gamma=None)
