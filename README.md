![example workflow](https://github.com/patel-zeel/polire/actions/workflows/python-package.yml/badge.svg)


## Polire

The word "interpolation" has Latin origin and is composed of two words - Inter meaning between and Polire meaning to polish.

This repository is a collection of several spatial interpolation algorithms. 

## Examples
### Minimal example of interpolation
```python
import numpy as np
from polire import Kriging

# Data
X = np.random.rand(10, 2) # Spatial 2D points
y = np.random.rand(10) # Observations
X_new = np.random.rand(100, 2) # New spatial points

# Fit
model = Kriging()
model.fit(X, y)

# Predict
y_new = model.predict(X_new)
```

### Supported Interpolation Methods
```python
from polire import (
    Kriging, # Best spatial unbiased predictor
    GP, # Gaussian process interpolator from GPy
    IDW, # Inverse distance weighting
    SpatialAverage,
    Spline,
    Trend,
    Random, # Predict uniformly within the observation range, a reasonable baseline
    NaturalNeighbor,
    CustomInterpolator # Supports any regressor from Scikit-learn
)
```

### Use GP kernels from GPy and regressors from sklearn
```python
from sklearn.linear_model import LinearRegression # or any Scikit-learn regressor
from GPy.kern import Matern32 # or any other GPy kernel

from polire import GP, CustomInterpolator

# GP model
model = GP(Matern32(input_dim=2))

# Sklearn model
model = CustomInterpolator(LinearRegression(normalize = True))
```

## More info

Contributors:  [S Deepak Narayanan](https://github.com/sdeepaknarayanan), [Zeel B Patel](https://github.com/patel-zeel), [Apoorv Agnihotri](https://github.com/apoorvagnihotri), and [Nipun Batra](https://github.com/nipunbatra).

This project is a part of Sustainability Lab at IIT Gandhinagar.

Acknowledgement to sklearn template for helping to package into a PiPy package.


