import numpy as np
from scipy.spatial.distance import cdist
np.cat = np.concatenate


X = np.round(np.random.rand(10, 2)*10)
X1 = np.round(np.random.rand(20, 2))
X2 = np.round(np.random.rand(20, 5))
print(np.cat([X1, X2], axis=1).shape)
