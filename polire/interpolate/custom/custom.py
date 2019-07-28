import numpy as np

from ..base import Base
from ...utils import RESOLUTION_DOC, COORDINATE_DOC

# common docstring across classes


class CustomInterpolator(Base):
    f"""
    Class to interpolate by fitting a sklearn type Regressor to
    the given data.

    Parameters
    ----------
    regressor: class definition,
        This variable is used to pass in the Regressor we would like
        to use for interpolation. The regressor sould be sklearn type
        regressor. Example from sklearn.ensemble -> RandomForestRegressor
    
    {RESOLUTION_DOC}

    {COORDINATE_DOC}
    
    reg_kwargs: dict, optional
        This is a dictionary that is passed into the Regressor initialization.
        Use this to change the behaviour of the passed regressor. Default = empty dict

    Attributes
    ----------
    reg : object
        Object of the `regressor` class passed.
    """

    def __init__(
        self,
        regressor,
        resolution="standard",
        coordinate_type="Euclidean",
        reg_kwargs={},
    ):
        super().__init__(resolution, coordinate_type)
        self.reg = regressor(**reg_kwargs)

    def _fit(self, X, y):
        """Function for fitting.
        This function is not supposed to be called directly.
        """
        self.reg.fit(X, y)
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
        return self.reg.predict(np.asarray([X1.ravel(), X2.ravel()]).T)

    def _predict(self, X):
        """Function for interpolation on specific points.
        This function is not supposed to be called directly.
        """
        return self.reg.predict(X)

    def __repr__(self):
        return self.__class__.__name__ + "." + self.reg.__class__.__name__
