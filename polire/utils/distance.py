"""
A module to have different distance metrics for spatial interpolation
"""
import math
import numpy as np


def haversine(X1, X2):
    """
    This function computes the distance in kilometers when the inputs are of the form
    (Longitude, Latitude). Typically, when we want to do spatial interpolation over 
    longer distances the curvature of the earth becomes significant
    and hence it is better to use haversine distance. 
    Attributes
    ----------
    X1: Input Data point. Typically a numpy array
    X2: Inpute Data point. Typically a numpy array

    Returns
    -------
    result: Distance in Kilometres
    """
    # distance between latitudes 
    # and longitudes 
    lon1, lat1 = X1
    lon2, lat2 = X2
    dLat = (lat2 - lat1) * math.pi / 180.0
    dLon = (lon2 - lon1) * math.pi / 180.0

    # convert to radians 
    lat1 = (lat1) * math.pi / 180.0
    lat2 = (lat2) * math.pi / 180.0

    # apply formulae 
    a = (pow(math.sin(dLat / 2), 2) + 
         pow(math.sin(dLon / 2), 2) * 
             math.cos(lat1) * math.cos(lat2)); 
    rad = 6371
    c = 2 * math.asin(math.sqrt(a)) 
    return rad * c 

def euclidean(X1, X2):
    return np.linalg.norm(X1 - X2)


