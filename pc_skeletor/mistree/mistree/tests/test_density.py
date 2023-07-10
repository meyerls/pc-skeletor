import numpy as np
import mistree as mist


def test_variable_vs_density_2d():
    xgrid = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    ygrid = np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5], [0.5, 1.5, 2.5]])
    den = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for i in range(0, len(xgrid)):
        for j in range(0, len(xgrid[0])):
            _x = np.random.random_sample(den[i, j]) + float(i)
            _y = np.random.random_sample(den[i, j]) + float(j)
            _p = den[i, j]*np.ones(den[i, j])
            if i == 0 and j == 0:
                x, y, p = _x, _y, _p
            else:
                x = np.concatenate([x, _x])
                y = np.concatenate([y, _y])
                p = np.concatenate([p, _p])
    dx = 1.
    den2 = mist.variable_vs_density(x, y, dx, x, y, p, 3., z=None, z_param=None, mode='2D', get_density=False)
    condition = np.where(den.flatten().astype(float) == den2)[0]
    assert len(condition) == 9


def test_variable_vs_density_2d_param():
    xgrid = np.array([[0.5, 0.5, 0.5], [1.5, 1.5, 1.5], [2.5, 2.5, 2.5]])
    ygrid = np.array([[0.5, 1.5, 2.5], [0.5, 1.5, 2.5], [0.5, 1.5, 2.5]])
    den = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    for i in range(0, len(xgrid)):
        for j in range(0, len(xgrid[0])):
            _x = np.random.random_sample(den[i, j]) + float(i)
            _y = np.random.random_sample(den[i, j]) + float(j)
            _p = den[i, j]*np.ones(den[i, j])
            if i == 0 and j == 0:
                x, y, p = _x, _y, _p
            else:
                x = np.concatenate([x, _x])
                y = np.concatenate([y, _y])
                p = np.concatenate([p, _p])
    dx = 1.
    _, den2 = mist.variable_vs_density(x, y, dx, x, y, p, 3., z=None, z_param=None, mode='2D', get_density=True)
    den = den.flatten().astype(float)/np.mean(den)
    den2 = np.round(den2.flatten().astype(float), decimals=1)
    condition = np.where(den == den2)[0]
    assert len(condition) == 9


def test_variable_vs_density_3d():
    xgrid = np.array([[[0.5, 0.5], [1.5, 1.5]], [[0.5, 0.5], [1.5, 1.5]]])
    ygrid = np.array([[[0.5, 1.5], [0.5, 1.5]], [[0.5, 1.5], [0.5, 1.5]]])
    zgrid = np.array([[[0.5, 0.5], [0.5, 0.5]], [[1.5, 1.5], [1.5, 1.5]]])

    den = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    for i in range(0, len(xgrid)):
        for j in range(0, len(xgrid[0])):
            for k in range(0, len(xgrid[0, 0])):
                _x = np.random.random_sample(den[i, j, k]) + float(i)
                _y = np.random.random_sample(den[i, j, k]) + float(j)
                _z = np.random.random_sample(den[i, j, k]) + float(k)
                _p = den[i, j, k]*np.ones(den[i, j, k])
                if i == 0 and j == 0 and k ==0:
                    x, y, z, p = _x, _y, _z, _p
                else:
                    x = np.concatenate([x, _x])
                    y = np.concatenate([y, _y])
                    z = np.concatenate([z, _z])
                    p = np.concatenate([p, _p])
    dx = 1.
    den2 = mist.variable_vs_density(x, y, dx, x, y, p, 2., z=z, z_param=z, mode='3D', get_density=False)
    condition = np.where(den.flatten().astype(float) == den2)[0]
    assert len(condition) == 8


def test_variable_vs_density_3d_param():
    xgrid = np.array([[[0.5, 0.5], [1.5, 1.5]], [[0.5, 0.5], [1.5, 1.5]]])
    ygrid = np.array([[[0.5, 1.5], [0.5, 1.5]], [[0.5, 1.5], [0.5, 1.5]]])
    zgrid = np.array([[[0.5, 0.5], [0.5, 0.5]], [[1.5, 1.5], [1.5, 1.5]]])

    den = np.array([[[1, 2], [3, 4]], [[5, 6], [7, 8]]])

    for i in range(0, len(xgrid)):
        for j in range(0, len(xgrid[0])):
            for k in range(0, len(xgrid[0, 0])):
                _x = np.random.random_sample(den[i, j, k]) + float(i)
                _y = np.random.random_sample(den[i, j, k]) + float(j)
                _z = np.random.random_sample(den[i, j, k]) + float(k)
                _p = den[i, j, k]*np.ones(den[i, j, k])
                if i == 0 and j == 0 and k ==0:
                    x, y, z, p = _x, _y, _z, _p
                else:
                    x = np.concatenate([x, _x])
                    y = np.concatenate([y, _y])
                    z = np.concatenate([z, _z])
                    p = np.concatenate([p, _p])
    dx = 1.
    _, den2 = mist.variable_vs_density(x, y, dx, x, y, p, 2., z=z, z_param=z, mode='3D', get_density=True)
    den = np.round(den.flatten().astype(float)/np.mean(den), decimals=2)
    den2 = np.round(den2.flatten().astype(float), decimals=2)
    condition = np.where(den == den2)[0]
    assert len(condition) == 8
