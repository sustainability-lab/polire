from ..base import Base
import numpy as np
import multiprocessing as mp
import psutil
from GPy.kern import Matern32, Matern52, RBF
from scipy.optimize import least_squares

class NottDuns(Base):
    """
    A class to learn Nott and Dunsmuir's non-stationary kernel. For more information, refer to
    https://academic.oup.com/biomet/article-abstract/89/4/819/242307
    
    Parameters
    ----------
    N : int, default=10
        Number of nearby points to learn each kernel locally
    
    eta : int, default=1
        A hyperparameter used in weight function
        
    loc_kernel : str, default='m32', ('m32', 'm52' or 'rbf')
        type of kernel to be used
    """
    
    def __init__(self, N=10, eta=1, kernel_name='m32', verbose=True):
        super().__init__(verbose)
        self.__N = N + 1 # Number of datapoints for local kernel learning
        self.__eta = eta # Eta hyperparameter for weighting function
        self.__kernel_name = kernel_name
        self.__param_dict = {'N': self.__N, 
                             'eta': self.__eta,
                             'kernel_name': self.__kernel_name}
    
    def get_all_params(self):
        """
        Returns class parameters
        """
        return self.__param_dict
        
    def get_param(self, param):
        """
        Returns the value of a parameter
        """
        return self.__param_dict[param]
    
    def __calculate_dmat(self):
        self.__dmat = np.zeros((self._X.shape[0],self._X.shape[0]))
        for i in range(self._X.shape[0]):
            for j in range(i, self._X.shape[0]):
                self.__dmat[i, j] = np.linalg.norm(self._X[i] - self._X[j])
                self.__dmat[j, i] = self.__dmat[i, j]
    
    def __get_close_locs(self):
        self.__calculate_dmat() # Distance matrix
        return [self.__dmat[i].argsort()[:self.__N] for i in range(self._X.shape[0])]
    
    def __weight_func(self, S):
        return np.exp(-(1/self.__eta) * ((S - self._X)**2).sum(axis=1))
    
    def _model(self, loc):
        def __D_z(sj):
            pj = self._Gamma[sj, sj]
            return sum(np.meshgrid(pj, pj)) - 2 * self._Gamma[np.ix_(sj, sj)]

        def __obfunc(x):
            kernel = kern_dict[self.__kernel_name]
            kernel.variance = x[0]
            kernel.lengthscale = x[1]
            kern_vals = kernel.K(self._X[self.__close_locs[loc]])
            term = (__D_z(self.__close_locs[loc]) - kern_vals)/kern_vals
            return np.sum(term**2)
        
        # ARD can be added
        kern_dict = {'m32': Matern32(input_dim=self._X.shape[1], active_dims=list(range(self._X.shape[1]))), 
                 'm52': Matern52(input_dim=self._X.shape[1], active_dims=list(range(self._X.shape[1]))), 
                 'rbf': RBF(input_dim=self._X.shape[1], active_dims=list(range(self._X.shape[1])))}

        kernel = kern_dict[self.__kernel_name]
        var, ls = least_squares(__obfunc, [1, 1]).x
        kernel.variance = var
        kernel.lengthscale = ls
        return kernel.K
          
    def _c_inv(self, kern_func):
        return np.linalg.pinv(kern_func(self._X))
        
    def __learnLocal(self):
        self._verbose_print('Training local kernels. This may take a few moments')
        
        job = mp.Pool(psutil.cpu_count())
        self.__kernels = job.map(self._model, list(range(self._X.shape[0]))) 
        self.__C_inv = job.map(self._c_inv, self.__kernels)
        job.close()
        
        self._verbose_print('Training complete')
    
    def _Kernel(self, S1, S2=None):
        """
        This function is for the NottDuns Class.
        This is not expected to be called directly.
        """
        S2exists = True
        if np.all(S1 == S2) or S2 is None:
            S2exists = False
            S2 = S1
        
        assert S1.shape[1] == self._X.shape[1]
        assert S2.shape[1] == self._X.shape[1]
        
        # Calculating Weights & c_mats
        self.__v_s1 = np.zeros((S1.shape[0], self._X.shape[0]))
        self.__v_s2 = np.zeros((S2.shape[0], self._X.shape[0]))
        self.__c_mat_s1 = np.zeros((self._X.shape[0], S1.shape[0], self._X.shape[0]))
        self.__c_mat_s2 = np.zeros((self._X.shape[0], self._X.shape[0], S2.shape[0]))
        self.__c_mat_s1s2 = np.zeros((self._X.shape[0], S1.shape[0], S2.shape[0]))
        
        if S2exists:
            for s1i, s1 in enumerate(S1):
                s_vec = self.__weight_func(s1)
                self.__v_s1[s1i, :] = s_vec/s_vec.sum()
            for s2i, s2 in enumerate(S2):
                s_vec = self.__weight_func(s2)
                self.__v_s2[s2i, :] = s_vec/s_vec.sum()
            for i in range(self._X.shape[0]):
                self.__c_mat_s1[i, :, :] = self.__kernels[i](S1, self._X)
                self.__c_mat_s2[i, :, :] = self.__kernels[i](self._X, S2)
                self.__c_mat_s1s2[i, :, :] = self.__kernels[i](S1, S2)
        else:
            for s1i, s1 in enumerate(S1):
                s_vec = self.__weight_func(s1)
                self.__v_s1[s1i, :] = s_vec/s_vec.sum()
            self.__v_s2 = self.__v_s1
            for i in range(self._X.shape[0]):
                self.__c_mat_s1[i, :, :] = self.__kernels[i](S1, self._X)
                self.__c_mat_s2[i, :, :] = self.__c_mat_s1[i, :, :].T
                self.__c_mat_s1s2[i, :, :] = self.__kernels[i](S1)
        
        # Calculating main covariance function
        first_term = np.zeros((self._X.shape[0], self._X.shape[0], S1.shape[0], S2.shape[0]), dtype='float64')
        for i in range(self._X.shape[0]):
            for j in range(self._X.shape[0]):
                first_term[i, j, :, :] = (self.__c_mat_s1[i, :, :]\
                                         .dot(self.__C_inv[i])\
                                         .dot(self._Gamma)\
                                         .dot(self.__C_inv[j])\
                                         .dot(self.__c_mat_s2[j, :, :]))*\
                                        (self.__v_s1[:, i].reshape(-1, 1)\
                                         .dot(self.__v_s2[:, j].reshape(1, -1)))
            
        second_term = np.zeros((self._X.shape[0], S1.shape[0], S2.shape[0]))
        for i in range(self._X.shape[0]):
            second_term[i, :, :] =  np.sqrt(self.__v_s1[:, i].reshape(-1,1).dot(self.__v_s2[:, i].reshape(1,-1))) *\
                               (self.__c_mat_s1s2[i, :, :] - self.__c_mat_s1[i, :, :].\
                                                                     dot(self.__C_inv[i]).\
                                                                     dot(self.__c_mat_s2[i, :, :]))
        
        return first_term.sum(axis=(0,1)) + second_term.sum(axis=0)
    
    def _fit(self, X, y, ECM):
        """
        This function is for the NottDuns Class.
        This is not expected to be called directly.
        """
        
        assert type(ECM) == type(np.zeros((1,1))), 'ECM must be a numpy array'
        assert ECM.shape[0] == ECM.shape[1] == X.shape[0], 'ECM must have ('+str(X.shape[0])+', '+str(X.shape[0])+') shape'
        
        self._X = X # training fetures
        self._y = y # Training values
        self._Gamma = ECM # Empirical Covariance Matrix
        self.__param_dict['X'] = X
        self.__param_dict['y'] = y
        self.__param_dict['ECM'] = ECM
        
        self.__close_locs = self.__get_close_locs() # Get closest N locations for each train location
        self.__learnLocal() # Learning local kernels
        return self
        
    def _predict(self, X, return_cov):
        """
        This function is for the NottDuns Class.
        This is not expected to be called directly.
        """
        if self._KX_inv is None:
            self._KX_inv = np.linalg.pinv(self._Kernel(self._X, self._X))
        KX_test = self._Kernel(X, self._X)
        pred_mean = KX_test\
                                 .dot(self._KX_inv)\
                                 .dot(self._y - self._y.mean()) + self._y.mean()
        if return_cov:
            pred_var = self._Kernel(X, X) - KX_test.dot(self._KX_inv).dot(KX_test.T)
            return (pred_mean, pred_cov)
        return pred_mean