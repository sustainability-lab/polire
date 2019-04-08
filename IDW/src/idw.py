"""
	This module implements Inverse Distance Weightes Interpolation
"""
# Author(s) S Deepak Narayanan <deepak.narayanan@iitgn.ac.in>
## License - MIT License


import numpy as np 
import pandas as pd 



def make_grid(x_coordinates, y_coordinates, resolution, offset=0.2):
	"""
		Here x_coordinates are the x_values for which we have the data
		Here y_coordinates are teh y_values for which we have the data
		Resolution refers to how smooth one wants the grid values to be.
		Higher the resolution, more finer will be the change in values.
		This function just makes a NumPy grid and returns it.

		Parameters
		x_coordinates : array of all the x_coordinates of points where data is present
		y_coordinates : array of all the y_coordinates of points where data is present
		resolution : How fine a grid we want. (Values are: 'standard','low','high')
		offset : Determines the extent of Grid - How much extension beyond the least X and Y coordinates - floating point value

		Returns:
			xx, yy - Standard Output of NumPy Meshgrid 
	"""
	y_min = y_coordinates.min() - offset
    y_max = y_coordinates.max()+ offset
    x_min = x_coordinates.min()-offset
    x_max = x_coordinates.max()+offset
    x_arr = np.linspace(x_min,x_max,resolution)
    y_arr = np.linspace(y_min,y_max,resolution)
    xx,yy = np.meshgrid(x_arr,y_arr)  
    return xx,yy

def idw_interpolation(data, exponent = 2, resolution='standard',coordinate_type='euclidean',verbose=False):

	"""
		Performs IDW Interpolation.

		Parameters:
		
		data: Assumed to be a Numpy Ndarray that has the locations in the first two columns and values in the last column
		Example: data = [[1,2,4],[5,6,7],[8,9,10]]

		exponent: Its a positive integer bigger than 1 and assumed to be even.
		Dictates how fast the value at any given location falls off, with respect to distance.

		resolution: A string : Either "standard", "low" or "high"
		Decides how smooth the interpolation is going to be.

		coordinate_type: A string: "euclidean", "latlong_large" or "latlong_small"
		Decides how to perform the interpolation. Distance vary across coordinates - this ensures we account for all of that.

		verbose: Boolean - True or False
		If True prints out progress, else nothing.

		Returns 
		A NumPy 2D Array of all the interpolated values at all the locations within a given grid.
	"""
	 if coordinate_type == 'latlong_small':
	 	"""
	 		Use the conversions and projections for small changes in LatLong
 		"""
	 	return "To be done later"

     if coordinate_type == 'latlong_large':
	    """
	        Code to be written after understanding all the projections.
	    """
        return "To be done later"

    if coordinate_type=="euclidean":
    	
        
        X = dataset[:,0]
        y = dataset[:,1]
        if resolution=='high':
            xx,yy = make_grid(X,y,1000)
            
        if resolution=='low':
            xx,yy = make_grid(X,y,10)
            
        if resolution=='standard':
            xx,yy = make_grid(X,y,100)
        
        new = []
        new_arr = dataset
        for points in new_arr:
            mindist = np.inf
            val = 0
            for j in range(len(yy)):
                temp = yy[j][0]
                for i in range(len(xx[0])):
                    dist = np.linalg.norm(np.array([xx[0][i],temp]) - points[:2])
                    if dist<mindist:
                        mindist = dist
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
                    """
                    Could potentially have a divide by zero error here
                    Use a try except clause
                    """
                    for elem in range(len(x_nz)):
                        source = np.array([x_nz[elem],y_nz[elem]])
                        target = np.array([xx[0][i],yy[j][0]])
                        dist = (np.abs(xx[0][source[0]] - target[0])**exponent + np.abs(yy[source[1]][0] - target[1])**exponent)**(1/exponent)
                        final[i][j]+=new_grid[x_nz[elem],y_nz[elem]]/dist
                        normalise+=1/(dist)
                final[i][j]/=normalise
    
    return final
 

