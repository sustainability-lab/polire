"""
This is a module for Natural Neighbors Interpolation
"""

import numpy as np 
from scipy.spatial import Voronoi, voronoi_plot_2d
import matplotlib.pyplot as plt
from ..base import Base
from shapely.geometry import Point
from shapely.geometry.polygon import Polygon
from math import atan2

def order_poly(vertices):
    """This function essentially is used to order the vertices
    of the Voronoi polygon in a clockwise manner. 
    
    Arguments
    ---------
    vertices : {array-like, 2D matrix} 
    This contains the list of vertices of the Polygon to be sorted
    
    Returns
    -------
    new_vertices : {array-like, 2D matrix}
    All the vertices reordered in a clockwise manner
    """
    mean_x = np.mean(vertices[:,0])
    mean_y = np.mean(vertices[:,1])
    
    def condition(x):
        """This is the condition to be used while sorting. We convert the coordinates
        to Polar and sort the points
        """
        return atan2(x[0] - mean_x, x[1] - mean_y)*180/np.pi
    return sorted(vertices, key=condition)

class Natural_neighbor(Base):
	"""Class used for natural neighbors interpolation. This method is an implementation first
	proposed by Sibson et al. [1] in 1981. We use the weights derived using the work in [1]
	and leave it for future addition, the use of Laplace Weights [2].

	Parameters
	----------
	weights: str, optional
		This defines the type of weights to be used for natural neighbor interpolation.
		We use Sibson Weights, and plan to add Laplace weights in the future
		Default value is "sibson"

	display: Boolean, optional
		True value displays the voronoi tesselation to the user after fitting the model.
		Default value is False.

	Notes
	-----
		The way in which part of the code is used is in the assumption that
		we can 
	References
	----------
	[1]  Sibson, R. (1981). "A brief description of natural neighbor interpolation (Chapter 2)". In V. Barnett (ed.). Interpolating Multivariate Data. Chichester: John Wiley. pp. 21–36.
	[2]  V.V. Belikov; V.D. Ivanov; V.K. Kontorovich; S.A. Korytnik; A.Y. Semenov (1997). "The non-Sibsonian interpolation: A new method of interpolation of the values of a function on an arbitrary set of points". Computational mathematics and mathematical physics. 37 (1): 9–15.
	[3]  N.H. Christ; R. Friedberg, R.; T.D. Lee (1982). "Weights of links and plaquettes in a random lattice". Nuclear Physics B. 210 (3): 337–346.
	"""

	def __init__(
		self,
		weights = "sibson",
		display = False,
		resolution = "standard",
		coordinate_type = "Eucledian",
	):
		super().__init__(resolution, coordinate_type)
		self.weights = weights
		self.X = None
		self.y = None
		self.result = None
		self.voronoi = None
		self.regions = [] # This variable stores the original Voronoi regions
		self.vertex_poly_map = dict() # This variable stores the polygon to data point map

	def _fit(self, X, y):
		"""This function is for the natural neighbors interpolation method.
		This is not expected to be called directly.
		"""
		self.X = X
		self.y = y
		self.voronoi = Voronoi(X, incremental = True)
		regions = self.voronoi.regions

		for i in regions:
			if i!=[] and -1 not in i:
				# -1 corresponds to unbounded region - we can't have this in interpolation
				# and the function returns an empty list anyways
				self.regions.append(i)

		self.vertex_poly_map = {i:0 for i in range(len(X))}
		
		for i in range(len(X)):
			point = Point(X[i,0], X[i,1])
			for j in range(len(self.regions)):
				p = Polygon(order_poly(self.voronoi.vertices[self.regions[j]]))
				if p.contains(point):
					self.vertex_poly_map[i] = p

		# if self.display:
		# 	voronoi_plot_2d(vor)
		# 	plt.show()



