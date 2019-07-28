import numpy as np

from ..base import Base


class SpatialAverage(Base):
    """
    Class to interpolate by fitting a XGBoost Regressor to given
    data.
    """

    def __init__(self,
    radius=100,
    resolution="standard", 
    coordinate_type="Euclidean", **kwargs):
        super().__init__(resolution, coordinate_type)
        self.radius = radius

    def _fit(self, X, y):
        """Function for fitting.
        This function is not supposed to be called directly.
        """
        self.X = X
        self.y = y
        return self

    def _predict_grid(self, x1lim, x2lim):
        """Function for grid interpolation.
        This function is not supposed to be called directly.
        """
        # getting the boundaries for interpolation
        x1min, x1max = x1lim
        x2min, x2max = x2lim
        
        # building the grid
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)
        X1, X2 = np.meshgrid(x1, x2)
        return self._average(np.asarray([X1.ravel(), X2.ravel()]).T)

    def _predict(self, X):
        """Function for interpolation on specific points.
        This function is not supposed to be called directly.
        """
        return self._average(X)

    def _average(self, X):
        y_pred = []
        for ix in range(X.shape[0]):
            dist = np.linalg.norm(self.X - X[ix, :], 2, axis=1)
            mask = self.radius >= dist
            # print ('mask', mask)
            points_within_rad = mask.sum()
            # print ('points_within_rad', points_within_rad)
            y_pred.append(sum(self.y[mask]) / points_within_rad)

        return np.asarray(y_pred)
