import numpy as np

from sklearn.ensemble import RandomForestRegressor
from ..utils.distance import haversine, euclidean
from ..base import Base
from ..idw.idw import IDW


class RFSI(Base):
    """'Random Forest Spatial Interpolation (RFSI)' originally developed by Aleksandar Sekuli´c et al.
    paper link: https://www.mdpi.com/2072-4292/12/10/1687

    Make sure that first two features represent (longitude, latitude)
    """

    def __init__(self, n_obs=5, idw=True, exponent=2, rf_kwargs={},
                 resolution="standard", coordinate_type="Euclidean"):
        super().__init__(resolution, coordinate_type)
        self.n_obs = n_obs
        self.idw = idw
        self.exponent = exponent
        self.rf_kwargs = rf_kwargs
        if self.coordinate_type == 'Geographic':
            self.distance = haversine
        elif self.coordinate_type == 'Euclidean':
            self.distance = euclidean
        else:
            raise NotImplementedError(
                "Only Geographic and Euclidean Coordinates are available")

    def _fit(self, X, y):
        self.X = X
        self.y = y
        self.F = self.get_n_obs_fetures()
        self.model = RandomForestRegressor(**self.rf_kwargs)
        self.model.fit(self.F, self.y)
        return self

    def get_n_obs_fetures(self, X=None):
        self.lonlat = self.X[:, :2]
        if (X is None) or np.all(X == self.X):
            X = self.X
            lonlat = self.lonlat
            dst = np.round(self.distance(lonlat, self.lonlat))
            idx = dst.argsort()[:, 1:self.n_obs+1]
        else:
            lonlat = X[:, :2]
            dst = np.round(self.distance(lonlat, self.lonlat))
            idx = dst.argsort()[:, :self.n_obs]

        f1 = dst[np.arange(lonlat.shape[0])[:, None], idx]

        ymat = self.y[:, None].repeat(lonlat.shape[0], 1).T
        # print(lonlat.shape, idx.shape)
        f2 = ymat[np.arange(lonlat.shape[0])[:, None], idx]
        if self.idw:
            def for_each_row(i):
                i = i[0]
                model = IDW(exponent=self.exponent)
                model.resolution = self.resolution
                model.coordinate_type = self.coordinate_type
                model.fit(self.lonlat[idx[i]], self.y[idx[i]])
                return model.predict(lonlat[i][None, :])
            f3 = np.apply_along_axis(
                for_each_row, axis=1, arr=np.arange(X.shape[0]).reshape(-1, 1))
            return np.concatenate([X, f1, f2, f3], axis=1)

        return np.concatenate([X, f1, f2], axis=1)

    def _predict(self, X):
        F = self.get_n_obs_fetures(X)
        return self.model.predict(F)
