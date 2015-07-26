""" Implements Dynamic Mode Decomposition, kernel DMD, exact DMD,
    and related algorithms given sets of data
"""

import numpy as np
# import scipy.spatial


class DMD(object):
    """ Dynamic Mode Decomposition (DMD)

    Dynamically relevant dimensionality reduction using the Proper Orthogonal
    Decomposition (POD) in conjunction with a least-squares solver.  This
    approach extracts a set of modes with a fixed temporal behavior (i.e.,
    exponential growth/decay).

    This implementation uses the numpy implementation of the singular value
    decomposition (SVD), and therefore requires the entire data set to fit in
    memory.  This algorithm runs in O(NM^2) time, where N is the size of a
    snapshot and M is the number of snapshots assuming N>M.

    Due to the similarities in implementation, this code can compute the modes
    associated with three variants of DMD that have appeared in the literature,
    and a forth that is a logical combination of existing approaches:

    1) Projected DMD (see Tu et al., 2014)
    2) Exact DMD (see Tu et al., 2014)
    3) Total least squares DMD (see Hemati & Rowley, 2015)
    4) Projected total least squares DMD (not published, but a logical
        combination of the Tu and Hemati papers)


    Parameters
    ----------
    n_rank : int or None, optional
        Number of POD modes to retain in the when performing DMD.  n_rank is
        an upper bound on the rank of the resulting DMD matrix.

        If n_rank is None (default), then all of the POD modes will
        be retained.

    exact : bool, optional
        If false (default), compute the DMD modes using projected DMD
        If true, compute the DMD modes using exact DMD

        See Tu et al., 2014 for details.

    total : bool, optional
        If false (default), compute the ``standard'' DMD modes
        If true, compute the total least squares DMD modes

        See Hemati & Rowley, 2015 for details.

    Properties
    ----------
    evals : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (None if not computed)

    modes: array, shape (n_dim, n_rank) or None
       The DMD modes associated with the eigenvalues in evals

    basis : array, shape (n_dim, n_rank) or None
       The basis vectors used to construct the modes.  If exact=False,
       these are the POD modes ordered by energy.

    Atilde : array, shape (n_rank, n_rank) or None
       The "DMD matrix" used in mode computation

    Notes
    -----
    Implements the DMD algorithms as presented in:

    Tu et al. On Dynamic Mode Decomposition: Theory and Applications,
        Journal of Computational Dynamics 1(2), pp. 391-421 (2014).

    Total least squares DMD is defined in:

    Hemati and Rowley, De-biasing the dynamic mode decomposition
        for applied Koopman spectral analysis, arXiv:1502.03854 (2015).

    For projected DMD as defined in Tu et al., exact=False and total=False
    For exact DMD as defined in Tu et al., exact=True and total=False
    For total least squares DMD in Hemati & Rowley, exact=True and total=True

    """

    def __init__(self, n_rank=None, exact=False, total=False):
        self.n_rank = n_rank
        self.exact = exact
        self.total = total

        # Internal variables
        self._basis = None  # spatial basis vectors
        self._mode_coeffs = None  # DMD mode coefficients
        self._evals = None  # DMD eigenvalues
        self._Atilde = None  # The full DMD matrix

    @property
    def modes(self):
        return self._basis.dot(self._mode_coeffs)

    @property
    def evals(self):
        return self._evals

    @property
    def basis(self):
        return self._basis

    @property
    def Atilde(self):
        return self._Atilde

    def fit(self, X, Y=None):
        """ Fit a DMD model with the data in X (and Y)

        Parameters
        ----------
        X : array, shape (n_dim, n_snapshots)
            Data set where n_snapshots is the number of snapshots and
            n_dim is the size of each snapshot.  Note that spatially
            distributed data should be ``flattened'' to a vector.

            If Y is None, then the columns of X must contain a time-series
            of data taken with a fixed sampling interval.

        Y : array, shape (n_dim, n_snapsots)
            Data set containing the updated snapshots of X after a fixed
            time interval has elapsed.

        Returns
        -------
        self : object
            Returns this object containing the computed modes and eigenvalues
        """

        if Y is None:
            Y = X[:, 1:]
            X = X[:, :-1]

        #  Max rank is either the specified value or determined by matrix size
        if self.n_rank is not None:
            n_rank = min(self.n_rank, X.shape[0], X.shape[1])
        else:
            n_rank = min(X.shape)

        # ====== Total Least Squares DMD: Project onto shared subspace ========
        if self.total:
            # Compute V using the method of snapshots
            sig2, V_stacked = np.linalg.eigh(X.T.dot(X) + Y.T.dot(Y))
            inds = np.argsort(sig2)[::-1]  # sort by eigenvalue

            V_stacked = V_stacked[:, inds[:n_rank]]  # truncate to n_rank

            # Compute the "clean" data sets
            proj_Vh = V_stacked.dot(V_stacked.T)
            X = X.dot(proj_Vh)
            Y = Y.dot(proj_Vh)

        # ===== Dynamic Mode Decomposition Computation ======
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        if self.n_rank is not None:
            U = U[:, :n_rank]
            S = S[:n_rank]
            Vh = Vh[:n_rank, :]

        # Compute the DMD matrix using the pseudoinverse of X
        self._Atilde = U.T.dot(Y).dot(Vh.T)/S

        # Eigensolve gives modes and eigenvalues
        self._evals, self._mode_coeffs = np.linalg.eig(self._Atilde)

        # Two options: exact modes or projected modes
        if self.exact:
            self._basis = Y.dot(Vh.T)/S
        else:
            self._basis = U

        return self


class KDMD(object):
    """ Kernel Dynamic Mode Decomposition (KDMD)

    Dynamically relevent dimensionality reduction using kernel-based methods
    to implicitly choose a larger subset of observable space than used by
    standard DMD.  This approach extracts a set of modes with fixed temoral
    behaviors (i.e., exponential growth or decay) and embeds the data
    in an approximate Koopman eigenfunction coordinate system.

    This implementation uses the numpy implementation of the singular value
    decomposition (SVD), and therefore requires the entire data set to fit in
    memory.  This algorithm runs in O(NM^2) time, where N is the size of a
    snapshot and M is the number of snapshots assuming N>M.

    Due to the similarities in implementation, this code computes four
    variants of Kernel DMD, only one of which has appeared in the literature.

    1) Kernel DMD (see Williams, Rowley, & Kevrekidis, 2015)
    2) Exact Kernel DMD (modes are based on the Y data rather than the X data)
    3) Total least squares kernel DMD (a combination of Williams 2014 and
        Hemati 2015)
    4) Exact, TLS, kernel DMD (a combination of Williams 2014 and Hemati 2015)


    Parameters
    ----------
    kernel_fun : function or functor (array, array) -> square array
        A kernel function that computes the inner products of data arranged
        in an array with snapshots along each *COLUMN* when the __call__
        method is evaluated.

    n_rank : int or None, optional
        Number of features to retain in the when performing DMD.  n_rank is
        an upper bound on the rank of the resulting DMD matrix.

        If n_rank is None (default), then n_snapshot modes will be retained.

    exact : bool, optional
        If false (default), compute the KDMD modes using the X data
        If true, compute the KDMD modes using the Y data

        See Tu et al., 2014 and Williams, Rowley, & Kevrekidis 2014
        for details.

    total : bool, optional
        If false (default), compute the ``standard'' KDMD modes
        If true, compute the total least squares KDMD modes

        See Hemati & Rowley, 2015 and Williams, Rowley,
        & Kevrekidis, 2015 for details.

    Properties
    ----------
    evals : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (None if not computed)

    modes: array, shape (n_dim, n_rank) or None
       The DMD modes associated with the eigenvalues in evals

    Phi : array, shape (n_rank, n_snapshots) or None
       An embedding of the X data

    Atilde : array, shape (n_rank, n_rank) or None
       The "KDMD matrix" used in mode computation

    Notes
    -----
    Implements the DMD algorithms as presented in:

    Williams, Rowley, and Kevrekidis.  A Kernel-Based Approach to
        Data-Driven Koopman Spectral Analysis, arXiv:1411.2260 (2014)

    Augmented with ideas from:

    Hemati and Rowley, De-biasing the dynamic mode decomposition
        for applied Koopman spectral analysis, arXiv:1502.03854 (2015).

    For kernel DMD as defined in Williams, exact=False and total= False
    """

    def __init__(self, kernel_fun, n_rank=None, exact=False, total=False):
        self.kernel_fun = kernel_fun
        self.n_rank = n_rank
        self.exact = exact
        self.total = total

        self._modes = None
        self._evals = None
        self._Phi = None
        self._Atilde = None
        self._G = None
        self._A = None

    @property
    def modes(self):
        return self._modes

    @property
    def evals(self):
        return self._evals

    @property
    def basis(self):
        return self._basis

    @property
    def Atilde(self):
        return self._Atilde

    def fit(self, X, Y=None):
        """ Fit a DMD model with the data in X (and Y)

        Parameters
        ----------
        X : array, shape (n_dim, n_snapshots)
            Data set where n_snapshots is the number of snapshots and
            n_dim is the size of each snapshot.  Note that spatially
            distributed data should be ``flattened'' to a vector.

            If Y is None, then the columns of X must contain a time-series
            of data taken with a fixed sampling interval.

        Y : array, shape (n_dim, n_snapsots)
            Data set containing the updated snapshots of X after a fixed
            time interval has elapsed.

        Returns
        -------
        self : object
            Returns this object containing the computed modes and eigenvalues
        """

        if Y is None:
            # Efficiently compute A, G, and optionally Gy
            # given a time series of data

            Gfull = self.kernel_fun(X, X)
            G = Gfull[:-1, :-1]
            A = Gfull[1:, :-1]

            if self.total:
                Gy = Gfull[1:, 1:]

            Y = X[:, 1:]
            X = X[:, :-1]
        else:  # Paired data

            try:
                gram_tuple = self.kernel_fun.compute_products(X, Y, self.total)

                if self.total:
                    G, A, Gy = gram_tuple
                else:
                    G, A = gram_tuple

            except AttributeError:
                G = self.kernel_fun(X, X)
                A = self.kernel_fun(Y, X)

                if self.total:
                    Gy = self.kernel_fun(Y, Y)

        # Rank is determined either by the specified value or
        # the number of snapshots
        if self.n_rank is not None:
            n_rank = min(self.n_rank, X.shape[1])
        else:
            n_rank = X.shape[1]

        # ====== Total Least Squares DMD: Project onto shared subspace ========
        if self.total:
            # Compute V using the method of snapshots

            sig2, V_stacked = np.linalg.eigh(G + Gy)
            inds = np.argsort(sig2)[::-1]  # sort by eigenvalue
            V_stacked = V_stacked[:, inds[:n_rank]]  # truncate to n_rank

            # Compute the "clean" data sets
            proj_Vh = V_stacked.dot(V_stacked.T)
            G = proj_Vh.dot(G).dot(proj_Vh)
            A = proj_Vh.dot(A).dot(proj_Vh)
            X = X.dot(proj_Vh)
            Y = Y.dot(proj_Vh)

        # ===== Kernel Dynamic Mode Decomposition Computation ======
        self._A = A
        self._G = G
        S2, U = np.linalg.eigh(G)
        inds = np.argsort(S2)[::-1]
        U = U[:, inds[:n_rank]]
        S2 = S2[inds[:n_rank]]
        self._Atilde = U.T.dot(A).dot(U)/S2

        # Eigensolve gives modes and eigenvalues
        self._evals, vecs = np.linalg.eig(self._Atilde)
        self._PhiX = (U.dot(vecs)).T

        # Two options: exact modes or projected modes
        if self.exact:
            PhiY = ((A.dot(U)/S2).dot(vecs)).T
            self._modes = Y.dot(np.linalg.pinv(PhiY))
        else:
            self._modes = X.dot(np.linalg.pinv(self._PhiX))

        return self


class PolyKernel(object):
    """ Implements a simple polynomial kernel

    This class is meant as an example for implementing kernels.

    Parameters
    ----------
    alpha : int
        The power used in the polynomial kernel
    epsilon : double, optional
        Scaling parameter in the kernel, default is 1.
    """

    def __init__(self, alpha, epsilon=1.0):
        self.alpha = alpha
        self.epsilon = epsilon

    def __call__(self, X, Y):
        return (1.0 + X.T.dot(Y)/self.epsilon)**self.alpha

    def compute_products(self, X, Y, Gy=False):
        """
        Compute the inner products X^T*X, Y^T*X, and if needed Y^T*Y.

        For a polynomial kernel, this code is no more efficient than
        computing the terms individually.  Other kernels require
        knowledge of the complete data set, and must use this.

        Note: If this method is not implemented, the KDMD code will
        manually compute the inner products using the __call__ method.
        """

        if Gy:
            return self(X, X), self(Y, X), self(Y, Y)
        else:
            return self(X, X), self(Y, X)


def sort_modes_evals(dmd_class, k=None, sortby="LM", target=None):
    """ Sort and return the DMD or KDMD modes and eigenvalues

    Paramters
    ---------
    dmd_class : object
       A DMD-like object with evals and modes properties

    k : int, optional
        The number of DMD mode/eigenvalue pairs to return.
        None (default) returns all of them.

    sortby : string, optional
       How to sort the eigenvalues and modes.  Options are

       "LM"   : Largest eigenvalue magnitudes come first
       "closest" : Sort by distance from argument target

    target : complex double, optional
       If "closest" is chosen, sort by distance from this eigenvalue
    """

    evals = dmd_class.evals
    modes = dmd_class.modes

    if k is None:
        k = len(evals)

    if evals is None or modes is None:
        raise RuntimeError("DMD modes have not yet been computed.")

    if sortby == "LM":
        inds = np.argsort(np.abs(evals))[::-1]
    elif sortby == "closest":
        inds = np.argsort(np.abs(evals - target))
    else:
        raise NotImplementedError("Cannot sort by " + sortby)

    evals = evals[inds]
    modes = modes[:, inds]

    return evals[:k], modes[:, :k]
