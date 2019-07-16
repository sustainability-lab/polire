# imports
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from interpolation.random.random import Random
from interpolation.trend.trend import Trend
from interpolation.spline.bspline import BSpline

# sample data
X = [[0, 0], [0, 3], [3, 0], [3, 3]]
y = [0, 1.5, 1.5, 3]
X = np.array(X)
y = np.array(y)

for r in [Random(), BSpline(kx=1, ky=1), Trend()]:
    r.fit(X, y)
    y_pred = r.predict(0, 3, 0, 3)
    Z = y_pred
    sns.heatmap(Z)
    plt.title(r.__class__)
    plt.show()
    plt.close()
