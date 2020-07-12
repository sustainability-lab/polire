from ..base import Base
from GPy.models import GPRegression
from GPy.kern import Matern32, Matern52, RBF

class Stationary(Base):
    """
    Matern32 kernel for sensor placement
    """
    def __init__(self, n_restarts, kernel_name, verbose=True):
        super().__init__(verbose)
        self.__n_restarts = n_restarts
        self.__kernel_name = kernel_name
        
    def _Kernel(self, S1, S2=None):
        return self.__model.kern.K(S1, S2)
    
    def _fit(self, X, y, ECM=None):
        self._X = X
        self._y = y
        
        kern_dict = {'m32': Matern32(input_dim=self._X.shape[1], active_dims=list(range(self._X.shape[1])), ARD=True), 
                 'm52': Matern52(input_dim=self._X.shape[1], active_dims=list(range(self._X.shape[1])), ARD=True), 
                 'rbf': RBF(input_dim=self._X.shape[1], active_dims=list(range(self._X.shape[1])), ARD=True)}
        
        
        self.__model = GPRegression(X, y, kern_dict[self.__kernel_name])
        self.__model.optimize_restarts(self.__n_restarts)
        return self
        
    def _predict(self, X, return_cov=True):
        if not return_cov:
            return self.__model.predict(X)[0]
        return self.__model.predict(X, full_cov=True)