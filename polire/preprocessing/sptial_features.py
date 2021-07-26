import numpy as np
from ..idw.idw import IDW
from ..utils.distance import haversine, euclidean


class SpatialFeatures:
    """Generate spatial features from N-closest locations

    Args:
        n_closest : 'N' closest locations

        idw : To use idw output as one of the feature

        idw_exponent : Exponent to be used in idw (if idw is False, ignore)

        coordinate_type : 'Eucleadian' or 'Geographic' (if idw is False, ignore)

        resolution : 'low', 'standard' or 'high' (if idw is False, ignore)
    """

    def __init__(self, n_closest: int = 5, idw: bool = True, idw_exponent: float = 2,
                 coordinate_type: str = 'Euclidean', resolution: str = 'standard') -> None:

        self.n_closest = n_closest
        self.idw = idw
        self.idw_exponent = idw_exponent
        self.coordinate_type = coordinate_type
        self.resolution = resolution
        if self.coordinate_type == 'Eucledian':
            self.distance = euclidean
        elif self.coordinate_type == 'Geographic':
            self.distance = haversine
        else:
            raise NotImplementedError(
                '"'+self.coordinate_type+'" is not implemented yet or invalid')

    def fit(self, X: np.ndarray, y: np.ndarray) -> object:
        """[summary]

        Args:
            X : Reference X data (longitude, latitude, time, ...)
            y : Reference y data

        Returns:
            self
        """
        self.X = X
        self.y = y

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform features

        Args:
            X (np.ndarray): (longitude, latitude, time, ...)

        Raises:
            Exception: If not already fitted

        Returns:
            np.ndarray: Transformed features
        """
        try:
            self.X
        except AttributeError:
            raise Exception("Not fitted yet. first call the 'fit' method")

        Xflag = False
        if np.all(X == self.X):
            Xflag = True

        F = np.empty((X.shape[0], (X.shape[1] - 3) +
                     self.n_closest*2 + self.idw)) * np.nan
        for t in np.unique(X[:, 2]):            # Iterating over time
            mask = X[:, 2] == t                 # rows with time t
            trn_mask = self.X[:, 2] == t
            X_local = X[mask]
            self_X_local = self.X[trn_mask]

            lonlat = X_local[:, :2]             # locs
            self_lonlat = self_X_local[:, :2]   # Reference locs
            dst = self.distance(lonlat, self_lonlat)
            if Xflag:
                idx = dst.argsort()[:, 1:self.n_closest+1]
            else:
                idx = dst.argsort()[:, :self.n_closest]

            # Feature set 1: closest distances
            f1 = dst[np.arange(lonlat.shape[0])[:, None], idx]

            self_y_local = self.y[trn_mask]     # Train obs
            ymat = self_y_local[:, None].repeat(lonlat.shape[0], 1).T
            # Feature set 2: closest observations
            f2 = ymat[np.arange(lonlat.shape[0])[:, None], idx]

            if self.idw:
                def for_each_row(i):
                    i = i[0]
                    model = IDW(exponent=self.idw_exponent)
                    model.resolution = self.resolution
                    model.coordinate_type = self.coordinate_type
                    model.fit(self_lonlat[idx[i]], self_y_local[idx[i]])
                    return model.predict(lonlat[i][None, :])

                # Feature set 3: IDW observation
                f3 = np.apply_along_axis(
                    for_each_row, axis=1, arr=np.arange(lonlat.shape[0]).reshape(-1, 1))
                F[mask] = np.concatenate([X_local[:, 3:], f1, f2, f3], axis=1)
            else:
                F[mask] = np.concatenate([X_local[:, 3:], f1, f2], axis=1)

        return F

    def fit_transform(self, X: np.ndarray, y: np.ndarray):
        self.fit(X, y)
        return self.transform(X)
