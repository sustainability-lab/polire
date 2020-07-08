import numpy as np

class Base:
    """
    Base class for placement
    """
    def __init__(self, verbose):
        self.__verbose = verbose
        self._KX_inv = None # inverse of Kernel(X, X)
        self.cov_np = None
        
    def _verbose_print(self, args):
        if self.__verbose:
            print(args)
    
    def fit(self, X, y, ECM=None):
        """
        Fit method to learn a kernel from available data
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature vectors
            n_samples: Number of locations in context of placement or samples
            n_features: Number of co-ordinates or features

        y : np.ndarray, shape (n_samples, )
            Training values

        ECM (Empirical Covariance Matrix) : np.ndarray, shape (n_locations, n_locations)
            Estimation of Emprical Covariance Matrix
            (This is needed only in Nott and Dunsmuir's Non-stationary kernel)
            --> One way to calculate it is given at,
            http://math.mit.edu/~liewang/ECM.pdf
            --> It can be fitted with scikit-learn library as well,
            https://scikit-learn.org/stable/modules/generated/sklearn.covariance.EmpiricalCovariance.html
        """
        assert type(X) == type(np.zeros((1,1))), 'X must be a numpy array'
        assert type(y) == type(np.zeros((1,1))), 'y must be a numpy array'
        assert X.shape[0] == y.shape[0], 'X and y must be of same size'
        assert len(X.shape) == 2, 'X must be a 2D array. use X.reshape(-1,1) if X is 1D'
        
        self._fit(X, y, ECM)
        self.__fitted = True # Setting fit process over
        
    def predict(self, X, return_cov=False):
        """
        Predict method to return predicted mean and variance
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Feature vectors
            n_samples: Number of locations in context of placement or samples
            n_features: Number of co-ordinates or features
        """
        
        return self._predict(X, return_cov)
    
    def Kernel(self, X, Y=None):
        """
        Covariance function
        
        Parameters
        ----------
        X, Y : np.ndarray, shape (n_locations, n_dimentions)
            X, Y are arrays passed to the function to get covariance matrix
            n_locations : Number of locations
            n_features : Number of co-ordinates
        """
        return self._Kernel(X, Y)
    
    def place(self, X, N=1, method = 'MI', random_state=None, committee=None):
        """
        A method for placement from Krause et al. JMLR, 2008 Paper, can be found at,
        http://jmlr.org/papers/volume9/krause08a/krause08a.pdf
        
        We have tried to keep notations of the paper intact in the code
        
        Parameters
        ----------
        X : np.ndarray, shape (n_samples, n_features)
            Locations of interest for placement
            n_samples: Number of locations in context of placement or samples
            n_features: Number of co-ordinates or features
            
        N : int, default=1
            Number of placements to be done
            
        Method : str, default='MI', ('MI', 'Var', 'Rand')
            Method to select N sensors
            
            MI : Use Maximum Mutual Information for placement
            Var : Use Maximum Variance for placement 
            
        committee : list, default=None
            A list of learners to be used to run QBC.
            Mandatory to pass when using method = "QBC".
            [cite]: https://doi.org/10.1145/130385.130417
        """
        assert self.__fitted, 'First call fit method to learn a kernel'
        assert type(X) == type(np.zeros((1,1))), 'X must be a numpy array'
        assert len(X.shape) == 2, 'X must be a 2D array. use X.reshape(-1,1) if X is 1D'
        assert self._X.shape[1] == X.shape[1], 'X must have (*, '+str(self.X.shape[1])+') shape'
        
        if self.cov_np is None:
            self.cov_np = self.Kernel(X)
        
        A = [] # selected indices
        
        if method == 'MI': # Mutual Information
            self.MI = [] # Making global to enable debugging
            for selection in range(N):
                selected = None
                delta_old = -np.inf
                location_bag = set(range(X.shape[0]))-set(A)
                for Y_ind in location_bag:
                    y = [Y_ind]
                    A_bar = list(location_bag - set(y))

                    if len(A) == 0:
                        numer = self.cov_np[y, y]
                    else:
                        numer = self.cov_np[y, y] - self.cov_np[np.ix_(y, A)]\
                        .dot(np.linalg.pinv(self.cov_np[np.ix_(A, A)]))\
                        .dot(self.cov_np[np.ix_(A, y)])
                        
                    if len(A) +1 == X.shape[0]:
                        denom = self.cov_np[y, y]
                    else:
                        denom = self.cov_np[y, y] - self.cov_np[np.ix_(y, A_bar)]\
                        .dot(np.linalg.pinv(self.cov_np[np.ix_(A_bar, A_bar)]))\
                        .dot(self.cov_np[np.ix_(A_bar, y)])
                    delta = numer/denom
                    if delta > delta_old:
                        selected = Y_ind
                        delta_old = delta
                A.append(selected)
                self.MI.append(delta_old.squeeze())
        
        if method == 'QBC': # QBC
            if committee is None:
                raise NotImplementedError("committee needs to passed.")
            y_preds = [learners.predict(X) for learners in committee]
            y_preds = np.asarray(y_preds).T
            std = y_preds.std(axis=1)
            A = std.argsort()[-N:][::-1]

        if method == 'Var': # Variance
            self.Var = [] # Making global to enable debugging
            for selection in range(N):
                selected = None
                delta_old = -np.inf
                location_bag = set(range(X.shape[0]))-set(A)
                for Y_ind in location_bag:
                    y = [Y_ind]
                    A_bar = list(location_bag - set(y))

                    if len(A) == 0:
                        numer = self.cov_np[y, y]
                    else:
                        numer = self.cov_np[y, y] - self.cov_np[np.ix_(y, A)]\
                        .dot(np.linalg.pinv(self.cov_np[np.ix_(A, A)]))\
                        .dot(self.cov_np[np.ix_(A, y)])
                    if numer > delta_old:
                        selected = Y_ind
                        delta_old = numer
                A.append(selected)
                self.Var.append(delta_old.squeeze())
        
        if method == 'Rand': # Random placement
            self.MI = [] # Making global to enable debugging
            np.random.seed(random_state)
            selected = np.random.choice(range(X.shape[0]), size=N, replace=False)
            for end in range(N):
                y = [end]
                A_bar = selected[end+1:]

                if len(A) == 0:
                    numer = self.cov_np[y, y]
                else:
                    numer = self.cov_np[y, y] - self.cov_np[np.ix_(y, A)]\
                    .dot(np.linalg.pinv(self.cov_np[np.ix_(A, A)]))\
                    .dot(self.cov_np[np.ix_(A, y)])

                if len(A) +1 == X.shape[0]:
                    denom = self.cov_np[y, y]
                else:
                    denom = self.cov_np[y, y] - self.cov_np[np.ix_(y, A_bar)]\
                    .dot(np.linalg.pinv(self.cov_np[np.ix_(A_bar, A_bar)]))\
                    .dot(self.cov_np[np.ix_(A_bar, y)])
                delta = numer/denom
                self.MI.append(delta.squeeze())
                
                A = selected[:end+1]
        
        return (A, X[A, :])
