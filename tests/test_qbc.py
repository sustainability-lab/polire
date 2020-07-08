import unittest
import numpy as np
import sys

sys.path.append('..')

from polire.placement.base import Base

class TestSum(unittest.TestCase):
    def test_qbc_diagreeing_committee(self):
        """
        Test that it can sum a list of integers
        """
        # let us select 2 data points [0, 3]
        X = np.array([[0, 3]]).T

        class Learner:
            def __init__(self, ix):
                self.ix = ix
            def predict(self, num):
                return num if ix == 0 else -num

        committee = [Learner(0), Learner(1)] # creating learners that predict y = x or y = -x
        object_to_be_tested = Base(verbose=False)
        placed = object_to_be_tested.place(X, method="qbc", committee=committee)
        self.assertEqual(placed, 1)

if __name__ == '__main__':
    unittest.main()
