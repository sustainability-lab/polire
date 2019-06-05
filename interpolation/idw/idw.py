
"""
This is a module for IDW Spatial Interpolation
"""
import numpy as np
import pandas as pd
from utils import make_grid

class idw():
    """ A class that is declared for performing IDW Interpolation.
    For more information on how this method works, kindly refer to
    https://en.wikipedia.org/wiki/Inverse_distance_weighting

    Parameters
    ----------
    exponent : positive float, optional
        The rate of fall of values from source data points.
        Higher the exponent, lower is the value when we move
        across space. Default value is 2.
    resolution: str, optional
        Decides the smoothness of the interpolation. Note that
        interpolation is done over a grid. Higher the resolution
        means more grid cells and more time for interpolation.
        Default value is 'standard'
    coordinate_type: str, optional
        Decides the distance metric to be used, while performing
        interpolation. Euclidean by default.     
    """
    def __init__(self, exponent = 2, resolution = 'standard', coordinate_type='Euclidean'):
        
        self.exponent = exponent
        self.resolution = resolution
        self.coordinate_type = coordinate_types
        self.interpolated_values = None
        self.x_grid = None
        self.y_grid = None

    def fit(self, X, y):
        """ The function call to fit the model on the given data. 
        Parameters
        ----------
        X: {array-like, 2D matrix}, shape(n_samples, 2)
            The set of all coordinates, where we have ground truth
            values
        y: array-like, shape(n_samples,)
            The set of all the ground truth values using which
            we perform interpolation

        Returns
        -------
        self : object
            Returns self
        """

         if self.coordinate_type == 'latlong_small':
        """
            Use the conversions and projections for small changes in LatLong
        """
            print ("To be done later")
            return self

        if self.coordinate_type == 'latlong_large':
            """
                Code to be written after understanding all the projections.
            """
            print ("To be done later")
            return self

        if self.coordinate_type=="Euclidean":
            
            X = deepcopy(np.c_[X,y])

            # Makes the grid. Check for more on the make_grid function inside utils.py
            if self.resolution=='high':
                xx,yy = make_grid(X,y,1000)
                
            if self.resolution=='low':
                xx,yy = make_grid(X,y,10)
                
            if self.resolution=='standard':
                xx,yy = make_grid(X,y,100)


                   
            new = []
            # This list stores all the tuples that contain points, and the closest point to the point in the 2D Grid that we are concerned with.

            for points in X:
                # Check the closest point in the grid, corresponding to the source data point.
                # We assume that the source is located at those exact coordinates on the grid.
                # IDW is done using these source data point locations, while computing distance.

                min_dist = np.inf
                val = 0
                for j in range(len(yy)):
                    temp = yy[j,0]
                    for i in range(len(xx[0])):
                        # Just checks for the closest location
                        dist = np.linalg.norm(np.array([xx[0][i],temp]) - points[:2])
                        if dist<min_dist:
                            min_dist = dist
                            val = (i,j)     
                new.append((points,val))
            # New now contains all the points that we're concerned with.

            new_grid = np.zeros((len(xx),len(yy)))

            for i in range(len(new)):
                # Storing the source data point values into the corresponding grid locations.
                # As from above, we store all the values needed in the new_list. 
                x = new[i][1][0]
                y = new[i][1][1]
                new_grid[x][y] = new[i][0][2]
            x_nz,y_nz = np.nonzero(new_grid)

            # This list_nz contains all the non -zero points or the source data points that we will have
            list_nz = []
            for i in range(len(x_nz)):
                list_nz.append((x_nz[i],y_nz[i]))

            final = deepcopy(new_grid)
            ## The final grid that we create will have all the needed interpolated values
            
            for i in range(len(xx[0])):
                for j in range(len(yy)):
                    normalise = 0
                    if (i,j) in list_nz:
                        ## These are the source data points. We can't interpolate here. 
                        # The point estimates have to be preserved
                        continue
                    else:
                        for elem in range(len(x_nz)):
                            # Looping through every non-zero location
                            source = np.array([x_nz[elem],y_nz[elem]])
                            target = np.array([xx[0][i],yy[j][0]])
                            ## Here we use the dist to compute the interpolated value, based on the exponent that the user provides. 
                            dist = (np.abs(xx[0][source[0]] - target[0])**self.exponent + np.abs(yy[source[1]][0] - target[1])**self.exponent)**(1/self.exponent)
                            final[i][j]+=new_grid[x_nz[elem],y_nz[elem]]/dist
                            normalise+=1/(dist)
                    # We divide by the normalise factor, as per the definiton of the inverse distance weighted interpolation
                    final[i][j]/=normalise
            self.interpolated_values = final
            self.x_grid = xx
            self.y_grid = yy
        
        # Return the self object -- This is useful if we want to do something like = object.fit().predict() etc., 
        return self

    def predict(self, X):
        """ The function call to predict using the interpolated data
        Parameters
        ----------
        X: {array-like, 2D matrix}, shape(n_samples, 2)
            The set of all coordinates, where we have ground truth
            values
        

        Returns
        -------
        y: array-like, shape(n_samples,)
            The set of all the ground truth values using which
            we perform interpolation 
        """
        if self.coordinate_type == 'Euclidean':
            for i in range(self.x_grid[0]):
                # Will add the code soon enough!
        
        
            
                
