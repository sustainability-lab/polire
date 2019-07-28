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
from copy import deepcopy

def order_poly(vertices):
    """This function essentially is used to order the vertices
    of the Voronoi polygon in a clockwise manner. This ensures
    that Shapely doesn't produce Polygon objects that are potentially
    non-convex and non-zero area.
    
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
	This is for contributors:
		The way in which part of the code is used is in the assumption that
		we use the data's ordering to find its voronoi partitions. 
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
		self.vertices = None #This variable stored the voronoi partition's vertices
		self.regions = [] # This variable stores the original Voronoi regions
		self.vertex_poly_map = dict() # This variable stores the polygon to data point map

	def _fit(self, X, y):
		"""This function is for the natural neighbors interpolation method.
		This is not expected to be called directly.
		"""
		self.X = X
		self.y = y
		self.voronoi = Voronoi(X, incremental = True)
		self.vertices = self.voronoi.vertices
		regions = self.voronoi.regions

		for i in regions:
			if i!=[] and -1 not in i:
				# -1 corresponds to unbounded region - we can't have this in interpolation
				# and the function returns an empty list anyways
				self.regions.append(i)

		self.vertex_poly_map = {i:0 for i in range(len(X))}
		
		# Assigning each data point to its corresponding polygon
		for i in range(len(X)):
			point = Point(X[i,0], X[i,1])
			for j in range(len(self.regions)):
				p = Polygon(order_poly(self.voronoi.vertices[self.regions[j]]))
				if p.contains(point):
					self.vertex_poly_map[i] = p
		# Remove all the data points that do not contribute to Nearest Neighhbor interpolation
		for i in range(len(vertex_poly_map)):
			if self.vertex_poly_map[i]==0:
				self.vertex_poly_map.pop(i,None)


		if self.display:
			voronoi_plot_2d(vor)
			plt.show()

	def _predict_grid(self, x1lim, x2lim):
		raise NotImplementedError

	def _predict(self, X):
		"""The function taht is called to predict the interpolated data in Natural Neighbors
		interpolation. This should not be called directly.
		"""
		result = np.zeros(len(X))

		for index in range(len(X)):
			vor = deepcopy(self.voronoi)
			vor.add_points(np.array([X[index]]))
			# We exploit the incremental processing of Scipy's Voronoi.
			# We create a copy to ensure that the original copy is preserved.
			new_regions = vor.regions
			new_vertices = vor.vertices
			final_regions = []

			for i in new_regions:
				if i!=[] and -1 not in i:
					final_regions.append(i)

			new = [] # this stores the newly created voronoi partitions
			for i in range(len(new_vertices)):
				if new_vertices[i] not in self.vertices:
					new.append(new_vertices[i])
			new = np.array(new)

			weights = {}	#Weights that we use for interpolation

			new_polygon = Polygon(order_poly(new))
			new_polygon_area = new_polygon.area

			for i in self.vertex_poly_map:
				if new_polygon.intersects(vertex_poly_map[i]):
					weights[i] = (new_polygon.intersection(self.vertex_poly_map[i])).area/new_polygon_area

			prediction = np.array([self.y[i]*weights[i] for i in weights]).sum()
			result[index] = prediction
			del vor, weights, new_polygon, new_polygon_area

		return result




