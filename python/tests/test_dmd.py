from nose.tools import *
import dmdtools
import numpy as np


def check_eigenvectors(v1, v2, tol=1e-10):
    """ Check whether two eigenvectors are equivalent
    """

    v1 /= np.linalg.norm(v1)
    v2 /= np.linalg.norm(v2)

    err = v2 - (v1.conj().T.dot(v2))*v1
    assert np.linalg.norm(err) < tol

def test_constructors():
    """ Check that the constructor works
    """

    dmdtools.DMD()
    dmdtools.DMD(10)
    dmdtools.DMD(10, True)
    dmdtools.DMD(10, True, True)

    # Check the arguments are copied properly
    args = {"n_rank":131, "exact":True, "total":False}
    tmp = dmdtools.DMD(**args)
    assert tmp.n_rank == args["n_rank"]
    assert tmp.total == args["total"]
    assert tmp.exact == args["exact"]

def test_dmd_computation():
    """  Check all variants of DMD on a toy problem
    """
    for exact in [True, False]:
        for total in [True, False]:
            yield check_dmd_computation_simple, exact, total
            yield check_dmd_computation_simple_timeseries, exact, total

def check_dmd_computation_simple(exact, total):
    """ Check DMD computations on a problem where the 
        true solution is known.  All variants of DMD
        should give identical outputs.
    """

    num_snapshots = 10

    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    X = np.random.randn(2, num_snapshots)
    Y = A.dot(X)

    ADMD = Y.dot(np.linalg.pinv(X))
    vals, vecs = np.linalg.eig(ADMD)
    inds = np.argsort(np.abs(vals))[::-1]
    vals = vals[inds]
    vecs = vecs[:, inds]

    # DMD class with rank 2
    DMD = dmdtools.DMD(2, exact, total)
    DMD.fit(X, Y)
    dmd_vals, dmd_modes = DMD.get_mode_pairs(sortby="LM")

    for ii in range(len(vals)):
        assert np.abs(dmd_vals[ii] - vals[ii]) < 1e-10
        check_eigenvectors(vecs[:, ii], dmd_modes[:, ii])


def check_dmd_computation_simple_timeseries(exact, total):
    """ Check DMD computations on a problem where the 
        true solution is known.  All variants of DMD
        should give identical outputs.
    """

    num_snapshots = 10

    A = np.array([[0.9, 0.1], [0.0, 0.8]])
    # Create a time series of data
    data = np.zeros((2, num_snapshots+1))
    data[:, 0] = np.random.randn(2)

    for ii in range(num_snapshots):
        data[:, ii + 1] = A.dot(data[:, ii])

    X = data[:, :-1]
    Y = data[:, 1:]
    ADMD = Y.dot(np.linalg.pinv(X))
    vals, vecs = np.linalg.eig(ADMD)
    inds = np.argsort(np.abs(vals))[::-1]
    vals = vals[inds]
    vecs = vecs[:, inds]

    # DMD class with rank 2
    DMD = dmdtools.DMD(2, exact, total)
    DMD.fit(data)
    dmd_vals, dmd_modes = DMD.get_mode_pairs(sortby="LM")

    for ii in range(len(vals)):
        assert np.abs(dmd_vals[ii] - vals[ii]) < 1e-10
        check_eigenvectors(vecs[:, ii], dmd_modes[:, ii])
