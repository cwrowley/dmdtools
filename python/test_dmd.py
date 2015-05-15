import unittest
import numpy as np
from dmdtools import *

class TestDMD(unittest.TestCase):
    def check_2d(self, f, **kwargs):
        eigvals = [0.5, 3.0]
        A = np.diag(eigvals)
        x1 = [1,2]
        x2 = [-1,1]
        X = np.array([x1, x2]).T
        Y = np.dot(A, X)
        modes, evals = f(X, Y, **kwargs)
        np.testing.assert_array_almost_equal(np.sort(evals), eigvals)
        # TODO: test modes as well

    def test_2d(self):
        self.check_2d(dmd)

    def test_2d_kernel_none(self):
        self.check_2d(kdmd, kernel=None)

    def test_2d_kernel_0(self):
        self.check_2d(kdmd, kernel=0)

    def test_2d_kernel_5(self):
        # TODO: fix this test.  Probably shouldn't recover the eigenvalues of A
        self.check_2d(kdmd, kernel=5)

if __name__ == "__main__":
    unittest.main()
