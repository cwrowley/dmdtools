""" Implements Dynamic Mode Decomposition, kernel DMD, exact DMD,
and related algorithms given sets of data
"""

import numpy as np
import scipy.spatial
import warnings


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

        If n_rank is None (default), then all of the POD modes will be retained.

    exact : bool, optional
        If false (default), compute the DMD modes using projected DMD
        If true, compute the DMD modes using exact DMD

        See Tu et al., 2014 for details.

    total : bool, optional
        If false (default), compute the ``standard'' DMD modes
        If true, compute the total least squares DMD modes

        See Hemati & Rowley, 2015 for details.

    Attributes
    ----------
    Atilde_ : array, shape (n_rank, n_rank) or None
       The DMD matrix.  Mostly used for testing purposes.

    modes_ : array, shape (n_dim, n_rank) or None
       The DMD modes associated with nonzero eigenvalues (if computed)
       or None otherwise.  The number of rows, n_dim, is determined during
       the fitting step.

    evals_ : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (or None if not yet computed)

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
        self.modes_ = None
        self.evals_ = None


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

        # ====== Total Least Squares DMD: Project onto shared subspace ========
        if self.total:
            # Compute V using the method of snapshots
            sig2, V_stacked = np.linalg.eigh(X.T.dot(X) + Y.T.dot(Y))
            inds = np.argsort(sig2)[::-1]  # sort by eigenvalue
            V_stacked = V_stacked[:, inds[:self.n_rank]]  # truncate to n_rank

            # Compute the "clean" data sets
            proj_Vh = V_stacked.dot(V_stacked.T)
            X = X.dot(proj_Vh)
            Y = Y.dot(proj_Vh)

        # ===== Dynamic Mode Decomposition Computation ======
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        if self.n_rank is not None:
            U = U[:, :self.n_rank]
            S = S[:self.n_rank]
            Vh = Vh[:self.n_rank, :]

        # Compute the DMD matrix using the pseudoinverse of X
        self.Atilde_ = U.T.dot(Y).dot(Vh.T)/S

        # Eigensolve gives modes and eigenvalues
        self.evals_, vecs = np.linalg.eig(self.Atilde_)

        # Two options: exact modes or projected modes
        if self.exact:
            self.modes_ = (Y.dot(Vh.T)/S).dot(vecs)/self.evals_
        else:
            self.modes_ = U.dot(vecs)

        return self

    def get_mode_pairs(self, k=None, sortby="none", target=None):
        """ Returns the DMD modes and eigenvalues.

        Paramters
        ---------
        k : int, optional
            The number of DMD mode/eigenvalue pairs to return.
            None (default) returns all of them.

        sortby : string, optional
            How to sort the eigenvalues and modes.  Options are

                "none" : No sorting, which is the default
                "LM"   : Largest eigenvalue magnitudes come first
                "closest" : Sort by distance from argument target

        target : complex double, optional
            If "closest" is chosen, sort by distance from this eigenvalue
        """

        if self.evals_ is None or self.modes_ is None:
            raise RuntimeError("DMD modes have not yet been computed.")

        if sortby == "none":
            inds = np.arange(len(self.evals_))
        elif sortby == "LM":
            inds = np.argsort(np.abs(self.evals_))[::-1]
        elif sortby == "closest":
            inds = np.argsort(np.abs(self.evals_ - target))
        else:
            raise NotImplementedError("Cannot sort by " + sortby)

        self.evals_ = self.evals_[inds]
        self.modes_ = self.modes_[:, inds]

        return self.evals_[:k], self.modes_[:, :k]



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

    Attributes
    ----------
    Atilde_ : array, shape (n_rank, n_rank) or None
       The DMD matrix.  Mostly used for testing purposes.

    modes_ : array, shape (n_dim, n_rank) or None
       The DMD modes associated with nonzero eigenvalues (if computed)
       or None otherwise.  The number of rows, n_dim, is determined during
       the fitting step.

    evals_ : array, shape (n_rank,) or None
       The eigenvalues associated with each mode (or None if not yet computed)

    PhiX_ : array, shape (n_rank, n_snapshots) or None
       An embedding of the X data 

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
        self.modes_ = None
        self.evals_ = None


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

        G = self.kernel_fun(X, X)
        A = self.kernel_fun(Y, X)

        # ====== Total Least Squares DMD: Project onto shared subspace ========
        if self.total:
            # Compute V using the method of snapshots
            Gy = self.kernel_fun(Y, Y)
            sig2, V_stacked = np.linalg.eigh(G + Gy)
            inds = np.argsort(sig2)[::-1]  # sort by eigenvalue
            V_stacked = V_stacked[:, inds[:self.n_rank]]  # truncate to n_rank

            # Compute the "clean" data sets
            proj_Vh = V_stacked.dot(V_stacked.T)
            G = proj_Vh.dot(G).dot(proj_Vh)
            A = proj_Vh.dot(A).dot(proj_Vh)
            X = X.dot(proj_Vh)
            Y = Y.dot(proj_Vh)

        # ===== Kernel Dynamic Mode Decomposition Computation ======
        S2, U = np.linalg.eigh(G)

        if self.n_rank is not None:
            U = U[:, :self.n_rank]
            S2 = S2[:self.n_rank]

        self.Atilde_ = U.T.dot(A).dot(U)/S2

        # Eigensolve gives modes and eigenvalues
        self.evals_, vecs = np.linalg.eig(self.Atilde_)
        self.PhiX_ = ((G.dot(U)/S2).dot(vecs)).T
        self.PhiY_ = ((A.dot(U)/S2).dot(vecs)).T

        # Two options: exact modes or projected modes
        if self.exact:
            self.modes_ = Y.dot(np.linalg.pinv(self.PhiY_))
        else:
            self.modes_ = X.dot(np.linalg.pinv(self.PhiX_))

        return self

    def get_mode_pairs(self, k=None, sortby="none", target=None):
        """ Returns the DMD modes and eigenvalues.

        Paramters
        ---------
        k : int, optional
            The number of DMD mode/eigenvalue pairs to return.
            None (default) returns all of them.

        sortby : string, optional
            How to sort the eigenvalues and modes.  Options are

                "none" : No sorting, which is the default
                "LM"   : Largest eigenvalue magnitudes come first
        "closest" : Sort by distance from the target eigenvalues

        target : complex double, optional
            If "closest" is chosen, distance from this number determines
            the sorting order.
        """

        if self.evals_ is None or self.modes_ is None:
            raise RuntimeError("DMD modes have not yet been computed.")


        if sortby == "none":
            inds = np.arange(len(self.evals_))
        elif sortby == "LM":
            inds = np.argsort(np.abs(self.evals_))[::-1]
        elif sortby == "closest":
            inds = np.argsort(np.abs(self.evals_ - target))
        else:
            raise NotImplementedError("Cannot sort by " + sortby)

        self.evals_ = self.evals_[inds]
        self.modes_ = self.modes_[:, inds]
        self.PhiX_ = self.PhiX_[inds, :]

        return self.evals_[:k], self.modes_[:, :k]

    def get_embedding_pairs(self, k=None, sortby="none", target=None):
        """ Returns the eigenvalues and embedding of X data

        Paramters
        ---------
        k : int, optional
            The number of DMD mode/eigenvalue pairs to return.
            None (default) returns all of them.

        sortby : string, optional
            How to sort the eigenvalues and modes.  Options are

                "none" : No sorting, which is the default
                "LM"   : Largest eigenvalue magnitudes come first
                "closest" : Sort by distance from the target eigenvalues

        target : complex double, optional
            If "closest" is chosen, distance from this number determines
            the sorting order.
        """

        if self.evals_ is None or self.PhiX_ is None:
            raise RuntimeError("KDMD embedding has not yet been computed.")

        self.get_dmd_pairs(k, sortby, target)  # sort the pairs

        return self.evals_[:k], self.PhiX_[:k, :]


class PolyKernel(object):
    """ Implements a simple polynomial kernel

    Parameters
    ----------
    alpha : int 
        The power used in the polynomial kernel
    epsilon : double, optional
        Scaling parameter in the kernel, default is 1.
    """

    def __init__(self, alpha, epsilon=1.0):
        self.alpha = alpha
        self.epsilon = 1.0

    def __call__(self, X, Y):
        return (1.0 + X.T.dot(Y)/self.epsilon)**self.alpha

