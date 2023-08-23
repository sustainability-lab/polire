import pytest
import numpy as np
from time import time
from polire import (
    IDW,
    Spline,
    Trend,
    # GP,
    Kriging,
    NaturalNeighbor,
    SpatialAverage,
    CustomInterpolator,
    # NSGP,
)
from sklearn.linear_model import LinearRegression

X = np.random.rand(20, 2)
y = np.random.rand(20)

X_new = np.random.rand(40, 2)


@pytest.mark.parametrize(
    "model",
    [
        IDW(),
        Spline(),
        Trend(),
        # GP(),
        Kriging(),
        NaturalNeighbor(),
        SpatialAverage(),
        CustomInterpolator(LinearRegression()),
        # NSGP(),
    ],
)
def test_fit_predict(model):
    init = time()
    model.fit(X, y)
    y_new = model.predict(X_new)

    assert y_new.shape == (40,)
    print("Passed", "Time:", np.round(time() - init, 3), "seconds")


@pytest.mark.skip(reason="Temporarily disabled")
def test_nsgp():
    model = NSGP()
    init = time()
    model.fit(X, y, **{"ECM": X @ X.T})
    y_new = model.predict(X_new)

    assert y_new.shape == (40,)
    assert y_new.sum() == y_new.sum()  # No NaN
    print("Passed", "Time:", np.round(time() - init, 3), "seconds")
