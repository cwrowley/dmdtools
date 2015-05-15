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

    def test_2d_custom_kernel(self):
        def custom_kernel(X, Y):
            return np.dot(X.T, Y)
        self.check_2d(kdmd, kernel=custom_kernel)

    def check_2d_random_data(self, n_data, f, **kwargs):

        eigvals = [0.9, 0.5]
        eigvecs = np.array([[1., 0.], [0., 1.]])
        A = np.diag(eigvals)

        np.random.seed(0)  # set the seed to 0 for reproducability
        X = np.random.randn(2, n_data)
        Y = A.dot(X)

        modes, evals = f(X, Y, **kwargs)

        # Check that we have an eigenvalue near 1
        self.assertAlmostEqual(min(abs(evals - 1)), 0)

        # Check that we have an eigenvalue near the true values
        for eigval in eigvals:
            self.assertAlmostEqual(min(abs(evals - eigval)), 0)

        # Check that all modes BUT the ones associated
        # with the system eigenvalues are 0
        for ii in range(0, len(evals)):
            try:  # Either the mode is 0 in norm
                np.testing.assert_array_almost_equal(modes[:, ii], np.zeros(2))
            except AssertionError:  # OR we match an eigval/eigvec
                # Find which system eigenvalue we best match
                index = np.argmin(abs(eigvals - evals[ii]))

                # Check that the "best fit" eigenvalue is a good match
                self.assertAlmostEqual(eigvals[index], evals[ii])

                # AND, once normalize, we match an eigenvector
                np.testing.assert_array_almost_equal(
                    abs(modes[:, ii]/np.linalg.norm(modes[:, ii])),
                    eigvecs[:, index])

    def test_2d_kernel_powers_random(self):
        n_data = 100
        for kernel_value in range(1, 5):
            self.check_2d_random_data(n_data, kdmd, kernel=kernel_value)


if __name__ == "__main__":
    unittest.main()
