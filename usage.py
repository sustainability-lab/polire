# imports
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

from polire.random.random import Random
from polire.trend.trend import Trend
from polire.spline.bspline import BSpline
from polire.idw.idw import Idw
# sample data
X = [[0, 0], [0, 3], [3, 0], [3, 3]]
y = [0, 1.5, 1.5, 3]
X = np.array(X)
y = np.array(y)

def test_grid():
    # Gridded interpolation testing
    for r in [Random(), BSpline(kx=1, ky=1), Trend(), Idw()]:
        r.fit(X, y)
        y_pred = r.predict_grid((0, 3), (0, 3))
        Z = y_pred
        sns.heatmap(Z)
        plt.title(r.__class__)
        plt.show()
        plt.close()
def test_point():
    # Pointwise interpolation testing
    for r in [Random(), BSpline(kx=1, ky=1), Trend(), Idw()]:
        r.fit(X,y)
        test_data = [[0,0],[0,3],[3,0],[3,3],[1,1],[1.5,1.5],[2,2],[2.5,2.5],[4,4]]
        y_pred = r.predict(np.array(test_data))
        print(r.__class__)
        print(y_pred)
if __name__=='__main__':
    print("Testing Gridded Interpolation")
    test_grid()
    print("\nTesting Pointwise Interpolation")
    test_point()
    