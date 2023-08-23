"""
A module to have different distance metrics for spatial interpolation
"""
import numpy as np
from scipy.spatial.distance import cdist


def haversine(X1, X2):
    """
    Arguments
    ---------
    One test point
    Multiple Train Points

    Long Lat Order
    """

    # Non-vectorized version
    # X1 = X1.reshape(1, 2)
    # difference = (X1 - X2) * np.pi / 180
    # test_point_lat = X1[:, 1] * np.pi / 180
    # training_locations_lat = X2[:, 1] * np.pi / 180

    # a = np.sin(difference[:, 0] / 2)**2 * np.cos(test_point_lat) * np.cos(training_locations_lat) +\
    #     np.sin(difference[:, 1] / 2)**2
    # radius = 6371
    # c = 2 * np.arcsin(np.sqrt(a))
    # return radius * c

    # Vectorized code
    lon1, lat1, lon2, lat2 = map(
        np.radians,
        [X1[:, 0, None], X1[:, 1, None], X2[:, 0, None], X2[:, 1, None]],
    )

    dlon = lon2.T - lon1
    dlat = lat2.T - lat1

    a = (
        np.sin(dlat / 2.0) ** 2
        + np.cos(lat1) @ np.cos(lat2.T) * np.sin(dlon / 2.0) ** 2
    )

    c = 2 * np.arcsin(np.sqrt(a))
    km = 6371 * c
    return km


def euclidean(X1, X2):
    # return np.linalg.norm(X1 - X2, 2, axis=1)
    return cdist(X1, X2)
