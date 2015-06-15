""" Theis module contains code that computes regularized DMD

    Note: these algorithms are MUCH more expensive than standard DMD.
          Also, they are a work is progress...so they may not even work!
"""

import numpy as np
import scipy.linalg
import warnings


class RegularizedDMD(object):
    """ Dynamic Mode Decomposition with regularization (DMDr)

    This class implements Dynamic Mode Decomposition with regularization.  The
    goal is to extract the same ``dynamically relevant'' features DMD does, but
    in a more robust manner.  In general, these algorithms produce results that
    are slighly biased, but also less sensitive to variance in the data.

    Three types of regularization are employed:

    1) Tikhonov regularization, which penalizes DMD matrices with large
        Frobenius norms.  This is numerically cheap and can sometimes
        be justified using Bayesian arguments.

    2) Entrywise L1 norm, which penalizes DMD matrices for not being sparse.

    3) Nuclear (trace) norm, which penalizes matrices for not being low-rank.
        This is useful for minimizing the number of ``important'' DMD modes,
        and tends to extract different structures than DMD on its own.

    Parameters
    ----------
    gamma : double, optional
        The relative importance of fitting the data vs the regularization.
        DMDr solves the optimization problem:

        min_A  ||Y - AX||^2 + gamma g(A),

        where A is the DMD matrix, Y and X are the data, g is the regularization
        function, and gamma is this parameter.  By default, gamma = 1.

    regularization : string, optional
        Determines the type of regularization to use.  Options are:

        "tik" : Tikhonov regularization (default)
        "L1"  : Entrywise L1 regularization
        "trace": Nuclear (trace) norm regularization

    cutoff : (double, double), optional
        tuple with the stopping parameters for the ADMM algorithm used for L1
        and trace minimization.  Can be supplied with Tikhonov regularization,
        but will be ignored.  Default values (1e-3, 1e-3)

    rho : double, optional
        Double with the regularization parameter used in the ADMM

    max_iter : int, optional
        Maximum number of iteration if ADMM is used

    tau : double, optional
        Factor by which rho is adjusted to balance the primal
        and dual problems.  Default is 2.0

    Attributes
    ----------
    modes_ : array, shape (n_dim, n_snapshots) or None
       The DMD modes associated with nonzero eigenvalues (if computed)
       or None otherwise.  The number of rows, n_dim, is determined during
       the fitting step.

    evals_ : array, shape (n_snapshots,) or None
       The eigenvalues associated with each mode (or None if not yet computed)

    Atilde_ : array, shape (n_snapshots, n_snapshots) 
       The matrix computed by regularized DMD (or None if not yet computed)
    Notes
    -----
    L1 and nuclear norm regularization is computed using the alternating
    direction method of multipliers (ADMM), and are much more expensive than
    simple Tikhonov regularization.
    """


    def __init__(self, gamma=1.0, regularization="tik",
                 cutoff=(1e-3,1e-3), rho=1.0, max_iter=1000, tau=2.0):
        self.gamma = gamma
        self.regularization = regularization
        self.cutoff = cutoff
        self.rho = rho
        self.max_iter = max_iter
        self.tau = tau

        self.modes_ = None
        self.evals_ = None
        self.Atilde_ = None


    def fit(self, X, Y, X_unpaired = None, X0=None, Y0=None):
        """ Fit a regularized DMD model with the data in X (and Y)

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

        X_unpaired : array or None
            Unpaired snapshots to be incorporated into the DMD model

        X0 : array, optional
        Y0 : array, optional
            Initial guesses for the primal and dual variables

        Returns
        -------
        self : object
            Returns this object containing the computed modes and eigenvalues
        """

        # Include any unpaired snapshots
        if X_unpaired is not None:
            X_full = np.c_[X, X_unpaired]
        else:
            X_full = X

        # Project onto a POD basis
        sig2, V = np.linalg.eigh(X_full.T.dot(X_full))
        sig = np.sqrt(np.abs(sig2))
        U = X_full.dot(V)/sig

        print U.shape, X_full.shape

        # Project X and Y onto a POD basis
        X = U.T.dot(X)
        Y = U.T.dot(Y)

        print Y.shape

        # We compute A transpose
        if self.regularization == "tik":
            solver = TikhonovSolver(self.gamma)
            solver.set_lhs(X.dot(X.T))
            AT = solver.solve(X.dot(Y.T))
        else:
            solver = ADMM(self.gamma, self.rho, self.cutoff[0],
                          self.cutoff[1], self.regularization,
                          self.max_iter, self.tau)

            AT, Ydual = solver.solve(X.T, Y.T, X0, Y0)

            # ADMM is expensive...save these in case needed
            self.X0 = AT.copy()
            self.Y0 = Ydual.copy()

        # Compute modes and vals
        self.Atilde_ = AT.T
        self.evals_, dmd_vecs = np.linalg.eig(self.Atilde_)
        self.modes_ = U.dot(dmd_vecs)

        return self

    def get_mode_pairs(self, k=None, sortby="none"):
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


class TikhonovSolver(object):
    """ Solver for linear systems with Tikhonov regularization

    This class solves the normal equation for a least squares problem 
    with Tikhonov regularization. It is most effective when 
    multiple ``right hand sides'' are used since Cholesky factorization is 
    stored.

    Specifically, this solves the (A + gamma I)X = B, where A must be a 
    symmetric positive semi-definite matrix.


    Parameters
    ----------
    rho : double
        The regularization parameter.

    Attributes
    ----------

    Achol_ : object
        The Cholesky decomposition of the A matrix
    """

    def __init__(self, rho):
        self.rho = rho
        self.Achol_ = None

    def set_lhs(self, A, rho=None):
        """ Sets and decomposes the left hand side. 
            Optionally change rho as well.
        """

        if rho is not None:
            self.rho = rho

        self.Achol_ = scipy.linalg.cho_factor(A + self.rho*np.eye(A.shape[1]))

        return self

    def solve(self, B):
        """ Solve the Tikhonov regularized problem with the given rhs
        """
        return scipy.linalg.cho_solve(self.Achol_, B)


class ADMM(object):
    """ Implementation of the ADMM algorithm for regularized least-squares

    This class implements the Alternating Direction Method of Multipliers (ADMM)
    algorithm for regularized least square problems.  For details,
    see Boyd, 2011.

    In particular, we are interested in problems of the form:

        min_X ||AX - B||_F^2 + gamma g(X),

    where g is either the entrywise L1 norm or the nuclear norm.

    Parameters
    ----------
    gamma : double
        The regularization parameter (default = 1.0).

    rho : double
        The regularization parameter used in ADMM (default = 1.0).

    epsilon_primal : double
        Stopping criterion for the DMD matrix (default = 10^-6)

    epsilon_dual : double
        Stopping criterion for the dual problem (default = 10^-6)

    regularization : string
       The type of regularization to impose.  Options are "L1" for entrywise L1
       and "trace" for nuclear norm (default="L1")

    max_iter : int
       The maximum number of iterations before we give up
   
    tau : double
       The amount rho is adjusted to balance the primal and dual problems

    Attributes
    ----------
    mu : double
        The threshold used to determine when rho should be adjusted

    References
    ----------
    Boyd et al.,  "Distributed optimization and statistical learning via
        the alternating direction method of multipliers."
        Foundations and Trends in Machine Learning 3.1 (2011): 1-122.
    """

    def __init__(self, gamma=1.0, rho=1.0, epsilon_primal=1e-3,
                 epsilon_dual=1e-3, regularization="L1", 
                 max_iter=int(1e4), tau=2.0):

        self.gamma = gamma
        self.rho = rho
        self.epsilon_primal = epsilon_primal
        self.epsilon_dual = epsilon_dual
        self.regularization = regularization
        self.max_iter = int(max_iter)
        self.tau = tau

        self.mu = 10.0  # when to do the parameter update

    def solve(self, A, B, X0=None, Y0=None):
        """ Compute the regularized least squares solution. 

        Parameters
        ----------
        A : array
            The left hand side matrix in the least squares problem
        B : array
            The right hand side matrix in the least squares problem
        X0 : array, optional
            An initial guess of the primal solution.  
            If none, defaults to tikhonov solution
        Y0 : array, optional
            An initial guess of the dual solution.  If none, defaults to zeros
        """

        ATA_ = A.T.dot(A)
        ATB_ = A.T.dot(B)

        # Tikhonov regularization is a necessary step, setup the solver
        tik_solver = TikhonovSolver(self.rho)
        tik_solver.set_lhs(ATA_)

        # Setup the initial guess
        if X0  is None:
            X = tik_solver.solve(ATB_)
        else:
            X = X0.copy()

        if Y0 is None:
            Y = np.zeros(X.shape)
        else:
            Y = Y0.copy()

        Z = X.copy()


        # Main computationa loop
        for ii in xrange(self.max_iter):
            Zold = Z.copy()
            Yold = Y.copy()
            Xold = X.copy()
            # Update X
            X = tik_solver.solve(ATB_ + self.rho*Z - Y)

            # Update Z
            if self.regularization == "trace":  # SVD based thresholding
                Z = X + Y/self.rho
                U, S, Vh = np.linalg.svd(Z, full_matrices=False)
                
                S -= self.gamma/self.rho
                Sold = S.copy()
                S[S < 0] = 0.0
                Z = (U*S).dot(Vh)

            elif self.regularization == "L1":  # L1
                Ztmp = X + Y/self.rho
                thresh_val = self.gamma/self.rho
                Z = np.piecewise(Ztmp, [Ztmp < -0.5*thresh_val,
                                        Ztmp > 0.5*thresh_val,
                                        abs(Ztmp) <= 0.5*thresh_val],
                                 [lambda z: z + 0.5*thresh_val,
                                  lambda z: z - 0.5*thresh_val,
                                  0.0])
            else:
                raise NotImplementedError("Regularization of type"+
                                          self.regularization+
                                          "is not currently implemented")

            # Update the multiplier
            Y += self.rho*(X - Z)

            res_primal = np.linalg.norm(X - Z, "fro")
            res_dual = self.rho*np.linalg.norm(Z - Zold, "fro")

            if res_primal > self.mu*res_dual:
                self.rho *= self.tau
                tik_solver.set_lhs(ATA_, self.rho)  # Recompute Tikhonov lhs
            elif res_dual > self.mu*res_primal:
                self.rho /= self.tau
                tik_solver.set_lhs(ATA_, self.rho)  # Recompute Tikhonov lhs
                

            print ii, res_primal, res_dual, self.epsilon_primal*max(np.linalg.norm(X, "fro"), np.linalg.norm(Z, "fro")), self.epsilon_dual*np.linalg.norm(Y, "fro"), self.rho

            # Check that the procedure has terminated
            if res_primal < self.epsilon_primal*max(np.linalg.norm(X, "fro"),
                                                    np.linalg.norm(Z, "fro")):
                if res_dual < self.epsilon_dual*np.linalg.norm(Y, "fro"):
                    return X, Y
        else:
            raise RuntimeError("ADMM failed to converge")

