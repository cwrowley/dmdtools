from nose.tools import *
import dmdtools
import numpy as np
import pylab as pyl

def check_eigenvectors(v1, v2, tol=1e-10):
    """ Check whether two eigenvectors are equivalent
    """

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    err = v2 - (v1.conj().T.dot(v2))*v1
    print np.linalg.norm(err), v1, v2, err
    assert np.linalg.norm(err) < tol

def test_constructors():
    """ Check that the constructor works
    """
    
    poly = dmdtools.PolyKernel(10)
    poly = dmdtools.PolyKernel(10, 1.0)
    poly = dmdtools.PolyKernel(999, 0.1234)

    dmdtools.KDMD(poly)
    dmdtools.KDMD(poly, 10)
    dmdtools.KDMD(poly, 10, True)
    dmdtools.KDMD(poly, 10, True, True)

def test_dmd_computation():
    """  Check all variants of DMD on a toy problem
    """
    for exact in [True, False]:
        for total in [True, False]:
            yield check_kdmd_computation_simple, exact, total

def check_kdmd_computation_simple(exact, total):
    """ Check DMD computations on a problem where the 
        true solution is known.  All variants of DMD
        should identify the original eigenvalues
    """

    num_snapshots = 50

    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    X = np.random.randn(2, num_snapshots)
    Y = A.dot(X)

    ADMD = Y.dot(np.linalg.pinv(X))
    vals, vecs = np.linalg.eig(ADMD)
    inds = np.argsort(np.abs(vals))[::-1]
    vals = vals[inds]
    vecs = vecs[:, inds]

    # DMD class with rank 2
    poly = dmdtools.PolyKernel(5)
    KDMD = dmdtools.KDMD(poly, None, exact, total)
    KDMD.fit(X, Y)

    for ii in range(len(vals)):
        dmd_vals, dmd_modes = dmdtools.sort_modes_evals(KDMD, sortby="closest",
                                                   target=vals[ii])
        assert np.abs(dmd_vals[0] - vals[ii]) < 1e-10
        check_eigenvectors(vecs[:, ii], dmd_modes[:, 0])

