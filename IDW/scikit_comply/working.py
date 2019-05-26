
"""
This is a module for IDW Spatial Interpolation
"""
import numpy as np
import pandas as pd

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
	 	return "To be done later"

        if self.coordinate_type == 'latlong_large':
            """
                Code to be written after understanding all the projections.
            """
            return "To be done later"

        if self.coordinate_type=="Euclidean":
        
            X = deepcopy(X[:,0])
            y = deepcopy(y[:,1])

            if self.resolution=='high':
                xx,yy = self.make_grid(X,y,1000)
                
            if self.resolution=='low':
                xx,yy = self.make_grid(X,y,10)
                
            if self.resolution=='standard':
                xx,yy = self.make_grid(X,y,100)
            
            new = []
            new_arr = deepcopy(X)
            for points in new_arr:
                min_dist = np.inf
                val = 0
                for j in range(len(yy)):
                    temp = yy[j][0]
                    for i in range(len(xx[0])):
                        dist = np.linalg.norm(np.array([xx[0][i],temp]) - points[:2])
                        if dist<min_dist:
                            min_dist = dist
                            val = (i,j)
                new.append((points,val))
            new_grid = np.zeros((len(xx),len(yy)))
            for i in range(len(new)):
                x = new[i][1][0]
                y = new[i][1][1]
                new_grid[x][y] = new[i][0][2]
            x_nz,y_nz = np.nonzero(new_grid)
            list_nz = []
            for i in range(len(x_nz)):
                list_nz.append((x_nz[i],y_nz[i]))
            final = np.copy(new_grid)
            for i in range(len(xx[0])):
                for j in range(len(yy)):
                    normalise = 0
                    if (i,j) in list_nz:
                        continue
                    else:
                        for elem in range(len(x_nz)):
                            source = np.array([x_nz[elem],y_nz[elem]])
                            target = np.array([xx[0][i],yy[j][0]])
                            dist = (np.abs(xx[0][source[0]] - target[0])**exponent + np.abs(yy[source[1]][0] - target[1])**exponent)**(1/exponent)
                            final[i][j]+=new_grid[x_nz[elem],y_nz[elem]]/dist
                            normalise+=1/(dist)
                    final[i][j]/=normalise
            self.interpolated_values = final
        return self
        
    