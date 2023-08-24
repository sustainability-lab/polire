![Tests](https://github.com/sustainability-lab/polire/actions/workflows/tests.yml/badge.svg)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![Coverage](https://coveralls.io/repos/github/sustainability-lab/polire/badge.svg?branch=master)](https://coveralls.io/github/sustainability-lab/polire?branch=master)

## Polire

```python
pip install polire
```


The word "interpolation" has a Latin origin and is composed of two words - Inter, meaning between, and Polire, meaning to polish.


This repository is a collection of several spatial interpolation algorithms. 

## Examples
Please refer to [the documentation](https://sustainability-lab.github.io/polire/) to check out practical examples on real datasets.

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

### Use GP kernels from GPy (temporarily unavailable)
```python
from GPy.kern import Matern32 # or any other GPy kernel

# GP model
model = GP(Matern32(input_dim=2))
```

### Regressors from sklearn
```py
from sklearn.linear_model import LinearRegression # or any Scikit-learn regressor
from polire import GP, CustomInterpolator

# Sklearn model
model = CustomInterpolator(LinearRegression())
```

### Extract spatial features from spatio-temporal dataset
```python
# X and X_new are datasets as numpy arrays with first three dimensions as longitude, latitute and time.
# y is corresponding observations with X

from polire.preprocessing import SpatialFeatures
spatial = SpatialFeatures(n_closest=10)
Features = spatial.fit_transform(X, y)
Features_new = spatial.transform(X_new)
```

## Citation

If you use this library, please cite the following paper:

```
@inproceedings{10.1145/3384419.3430407,
author = {Narayanan, S Deepak and Patel, Zeel B and Agnihotri, Apoorv and Batra, Nipun},
title = {A Toolkit for Spatial Interpolation and Sensor Placement},
year = {2020},
isbn = {9781450375900},
publisher = {Association for Computing Machinery},
address = {New York, NY, USA},
url = {https://doi.org/10.1145/3384419.3430407},
doi = {10.1145/3384419.3430407},
booktitle = {Proceedings of the 18th Conference on Embedded Networked Sensor Systems},
pages = {653–654},
numpages = {2},
location = {Virtual Event, Japan},
series = {SenSys '20}
}
```