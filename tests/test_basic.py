import pytest
import numpy as np
from time import time

X = np.random.rand(20, 2)
y = np.random.rand(20)

X_new = np.random.rand(40, 2)


def common(model):
    print(repr(model), end=' ')
    init = time()
    model.fit(X, y)
    y_new = model.predict(X_new)

    assert y_new.shape == (40, )
    print('Passed', 'Time:', np.round(time()-init, 3), 'seconds')


def common_nsgp(model):
    print(repr(model), end=' ')
    init = time()
    model.fit(X, y, **{'ECM': X@X.T})
    y_new = model.predict(X_new)

    assert y_new.shape == (40, )
    assert y_new.sum() == y_new.sum()  # No NaN
    print('Passed', 'Time:', np.round(time()-init, 3), 'seconds')


def test_basic():
    from polire import (IDW, Spline, Trend, GP, Kriging,
                        NaturalNeighbor, SpatialAverage,
                        CustomInterpolator, NSGP)
    from sklearn.linear_model import LinearRegression

    common(IDW())
    common(Spline())
    common(Trend())
    common(GP())
    common(Kriging())
    common(NaturalNeighbor())
    common(SpatialAverage())
    common(CustomInterpolator(LinearRegression(normalize=True)))
    common_nsgp(NSGP())


if __name__ == '__main__':
    test_basic()
