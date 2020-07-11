"""
This is a module for Kriging Interpolation
"""
import numpy as np 
from ..base import Base
from pykrige.ok import OrdinaryKriging
from pykrige.uk import UniversalKriging


class Kriging(Base):
    """A class that is declared for performing Kriging interpolation.
    Kriging interpolation (usually) works on the principle of finding the 
    best unbiased predictor. Ordinary Kriging, for an example, involves finding out the
    best unbaised linear predictor. 

    Parameters
    ----------
    type : str, optional
    This parameter defines the type of Kriging under consideration. This 
    implementation uses PyKrige package  (https://github.com/bsmurphy/PyKrige).
    The user needs to choose between "Ordinary" and "Universal".

    plotting: boolean, optional
    This parameter plots the fit semivariogram. We use PyKrige's inbuilt plotter for the same.s
    
    variogram_model : str, optional
    Specifies which variogram model to use; may be one of the following:
    linear, power, gaussian, spherical, exponential, hole-effect.
    Default is linear variogram model. To utilize a custom variogram model,
    specify 'custom'; you must also provide variogram_parameters and
    variogram_function. Note that the hole-effect model is only technically
    correct for one-dimensional problems.

    require_variance : Boolean, optional
    This variable returns the uncertainity in the interpolated values using Kriging
    interpolation. If this is True, kindly call the attribute return_variance, of this class
    to retreive the computed variances. False is the default value.d
    
	nlags: int, optional
	Number of lags to be considered for semivariogram. As in PyKrige, we set default to be 6.
    """

    def __init__(
        self,
        type = "Ordinary",
        plotting = False,
        variogram_model = 'linear',
        require_variance = False,
        resolution = "standard",
        coordinate_type = "Eucledian",
        nlags = 6
    ):

        super().__init__(resolution, coordinate_type)
        self.variogram_model = variogram_model
        self.ok = None
        self.uk = None
        self.type = type 
        self.plotting = plotting
        self.coordinate_type = None
        self.require_variance = require_variance
        self.variance = None
    	
        if self.coordinate_type == 'Geographic':
            self.coordinate_type = 'geographic'
        elif self.coordinate_type == 'Euclidean':
            self.coordinate_type = 'euclidean'
        else:
            raise NotImplementedError("Only Geographic and Euclidean Coordinates are available")

        self.nlags = nlags


    def _fit(self, X, y):
        """This method of the Kriging Class is used to fit Kriging interpolation model to
        the train data. This function shouldn't be called directly."""
        if self.type == "Ordinary":
            self.ok = OrdinaryKriging(
                X[:,0],
                X[:,1],
                y,
                variogram_model = self.variogram_model,
                enable_plotting = self.plotting,
                coordinates_type = self.coordinate_type,
                nlags = self.nlags
            )

        elif self.type == "Universal":
            self.uk = UniversalKriging(
                X[:,0],
                X[:,1],
                y,
                variogram_model = self.variogram_model,
                enable_plotting = self.plotting,
            )

        else:
            raise ValueError("Choose either Universal or Ordinary - Given argument is neither")


        return self

    def _predict_grid(self, x1lim, x2lim):
        """The function that is called to return the interpolated data in Kriging Interpolation
        in a grid. This method shouldn't be called directly"""
        lims = (*x1lim, *x2lim)
        x1min, x1max, x2min, x2max = lims
        x1 = np.linspace(x1min, x1max, self.resolution)
        x2 = np.linspace(x2min, x2max, self.resolution)

        if self.ok is not None:
            predictions, self.variance = self.ok.execute(
            style = 'grid',
            xpoints = x1,
            ypoints = x2
            )
        
        else:
            predictions, self.variance = self.uk.execute(
            style = 'grid',
            xpoints = x1,
            ypoints = x2
            )
            
        return  predictions

    def _predict(self, X):
        """This function should be called to return the interpolated data in kriging
        in a pointwise manner. This method shouldn't be called directly."""
        if self.ok is not None:
            predictions, self.variance = self.ok.execute(
            style = 'points',
            xpoints = X[:,0],
            ypoints = X[:,1]
            )
        
        else:
            predictions, self.variance = self.uk.execute(
            style = 'points',
            xpoints = X[:,0],
            ypoints = X[:,1]
            )
            
        return  predictions

    def return_variance(self):
        """This method of the Kriging class returns the variance at the interpolated
        points if the user chooses to use this option at the beginning of the interpolation"""
        if self.require_variance:
            return self.variance

        else:
            print("Variance not asked for, while instantiating the object. Returning None")
            return None

















