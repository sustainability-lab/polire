# imports
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from polire import CustomInterpolator
import xgboost
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import Matern

# sample data
X = [[0, 0], [0, 3], [3, 0], [3, 3]]
y = [0, 1.5, 1.5, 3]
X = np.array(X)
y = np.array(y)

for r in [
    CustomInterpolator(xgboost.XGBRegressor()),
    CustomInterpolator(RandomForestRegressor()),
    CustomInterpolator(LinearRegression(normalize=True)),
    CustomInterpolator(KNeighborsRegressor(n_neighbors=3, weights="distance")),
    CustomInterpolator(GaussianProcessRegressor(
        normalize_y=True, kernel=Matern()))
]:

    r.fit(X, y)
    Z = r.predict_grid((0, 3), (0, 3)).reshape(100, 100)
    sns.heatmap(Z)
    plt.title(r)
    plt.show()
    plt.close()
