"""
A module to have different distance metrics for spatial interpolation
"""
import math
import numpy as np


def haversine(test_point, training_locations):
    """
    Arguments
    ---------
    One test point
    Multiple Train Points
    
    Long Lat Order
    """
    test_point = test_point.reshape(1, 2)
    difference = (test_point - training_locations) * np.pi / 180
    test_point_lat = test_point[:, 1] * np.pi / 180
    training_locations_lat = training_locations[:, 1] * np.pi / 180
    
    a = np.sin(difference[:, 0] / 2)**2 * np.cos(test_point_lat) * np.cos(training_locations_lat) +\
        np.sin(difference[:, 1] / 2)**2 
    radius = 6371
    c = 2 * np.arcsin(np.sqrt(a))
    return radius * c

def euclidean(X1, X2):
    return np.linalg.norm(X1 - X2, 2, axis=1)


