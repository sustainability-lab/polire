"""
A module to have different distance metrics for spatial interpolation
"""
import numpy as np
from scipy.spatial.distance import cdist


def haversine(X1, X2):
    """
    Inspired from https://stackoverflow.com/a/29546836/13330701
    """
    lon1, lat1, lon2, lat2 = map(
        np.radians, [X1[:, 0, None], X1[:, 1, None], X2[:, 0, None], X2[:, 1, None]])

    dlon = lon2.T - lon1
    dlat = lat2.T - lat1

    a = np.sin(dlat/2.0)**2 + np.cos(lat1)@np.cos(lat2.T) * np.sin(dlon/2.0)**2

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def euclidean(X1, X2):
    # return np.linalg.norm(X1 - X2, 2, axis=1)
    return cdist(X1, X2)
