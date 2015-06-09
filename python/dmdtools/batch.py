""" Implements Dynamic Mode Decomposition, kernel DMD, exact DMD,
and related algorithms given sets of data
"""

import numpy as np
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


    def __init__(self, n_rank=None, exact):
        self.n_rank = n_rank
        self.exact = exact
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

            # Compute the "clean" data sets
            proj_Vh = V_stacked.T.dot(V_stacked)
            X = X.dot(proj_Vh)
            Y = Y.dot(proj_Vh)

        # ===== Dynamic Mode Decomposition Computation ======
        U, S, Vh = np.linalg.svd(X, full_matrices=False)

        if self.n_rank is not None:
            U = U[:, :self.n_rank]
            S = S[:self.n_rank]
            Vh = Vh[:self.n_rank, :]

        # Compute the DMD matrix using the pseudoinverse of X
        Atilde = U.T.dot(Y).dot(Vh.T)/S

        # Eigensolve gives modes and eigenvalues
        self.evals_, vecs = np.linalg.eig(Atilde)

        # Two options: exact modes or projected modes
        if self.exact:
            self.modes_ = (Y.dot(Vh.T)/S).dot(vecs)/self.evals
        else:
            self.modes_ = U.dot(vecs)

        return self

    def get_dmd_pairs(self, k=None, sortby="none"):
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
        """

        if self.evals_ is None or self.modes_ is None:
            raise RuntimeError("DMD modes have not yet been computed.")

        if sortby == "LM":
            inds = np.argsort(np.abs(self.evals_))[::-1]
            self.evals_ = self.evals_[inds]
            self.modes_ = self.modes_[:, inds]

        return self.evals_[:k], self.modes_[:, :k]
