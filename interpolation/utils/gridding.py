""" Standard Utility Script for Gridding Data
    1. Contains all the common functions that 
        will be employed across various different interpolators

"""
import numpy as np

def make_grid(self, x, y, res, offset=0.2):

        """ This function returns the grid to perform interpolation on.
           This function is used inside the fit() attribute of the idw class.
        
        Parameters
        ----------
        x: array-like, shape(n_samples,)
            The first coordinate values of all points where
            ground truth is available
        y: array-like, shape(n_samples,)
            The second coordinate values of all points where
            ground truth is available
        res: int
            The resolution value
        offset: float, optional
            A value between 0 and 0.5 that specifies the extra interpolation to be done
            Default is 0.2
        
        Returns
        -------
        xx : {array-like, 2D}, shape (n_samples, n_samples)
        yy : {array-like, 2D}, shape (n_samples, n_samples)
        """
        y_min = y.min() - offset
        y_max = y.max()+ offset
        x_min = x.min()-offset
        x_max = x.max()+offset
        x_arr = np.linspace(x_min,x_max,resolution)
        y_arr = np.linspace(y_min,y_max,resolution)
        xx,yy = np.meshgrid(x_arr,y_arr)  
        return xx,yy
