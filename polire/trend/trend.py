import numpy as np
from scipy.optimize import curve_fit

from ..base import Base
from .polynomials import _create_polynomial


class Trend(Base):
    """Class to interpolate by fitting a curve to the data points
    available using `scipy`'s `curve_fit`.
    
    Parameters
    ----------
    order: int, default 1
        Selects the order of the polynomial to best fit.
        Possible values 0 <= order <= 2.

    custom_poly: functor, default None
        If you would like to fit to your custom function,
        _set order to None_ and then pass a functor.
        See Example functor passing below
        ```python
        def func(X, a, b, c):
            x1, x2 = X
            return np.log(a) + b*np.log(x1) + c*np.log(x2)
        t = Trend(order=None, custom_poly=func)
        ```
    """

    def __init__(
        self,
        order=1,
        custom_poly=None,
        resolution="standard",
        coordinate_type="Euclidean",
    ):
        super().__init__(resolution, coordinate_type)
        self.order = order
        # setting the polynomial to fit our data
        if _create_polynomial(order) is not None:
            self.func = _create_polynomial(order)
        else:
            if custom_poly is not None:
                self.func = custom_poly
            else:
                raise ValueError("Arguments passed are not valid")

    def _fit(self, X, y):
        """Function for fitting trend interpolation.
        This function is not supposed to be called directly.
        """
        # fitting the curve using scipy
        self.popt, self.pcov = curve_fit(self.func, (X[:, 0], X[:, 1]), y)
        return self

    def _predict(self, lims):
        """Function for trend interpolation.
        This function is not supposed to be called directly.
        """
        x1min, x1max, x2min, x2max = lims
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)
        X1, X2 = np.meshgrid(x1, x2)
        return self.func((X1, X2), *self.popt)
