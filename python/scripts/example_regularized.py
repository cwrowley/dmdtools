""" An example highlighting the impact of regularization on DMD

    Returns
    -------
    Outputs a plot comparing the true, DMD, and TLS-DMD eigenvalues
"""
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import dmdtools

m = 500  # Number of snapshots
n = 50  # Apparent dimension

# True underlying system
n_true = 6
Alow = np.diag(np.array([0.9, 0.81, 0.95, 0.7, 0.6, 0.88]) * 
               np.exp(np.array([1j, 1.2j, 1.2j, 0.5j, 0.2j, 1j])))
A = Alow
np.random.seed(0)
Q = np.linalg.qr(np.random.randn(n, n_true))[0]
#A = Q.dot(Alow).dot(Q.T)


# Create data from a linear system
X = np.random.randn(n_true, m) + 1j*np.random.randn(n_true, m)
X = A.dot(A.dot(X))
Y = A.dot(X)

# Hide all but 4 entries 
#X = X[:3, :]
#Y = Y[:3, :]
X = Q.dot(X)
Y = Q.dot(Y)



X = np.r_[X.real, X.imag]
Y = np.r_[Y.real, Y.imag]

X += 5e-1*(np.random.randn(X.shape[0], X.shape[1]))
Y += 5e-1*(np.random.randn(Y.shape[0], Y.shape[1]))

#X = np.r_[X, X**2]
#Y = np.r_[Y, Y**2]

# Try with DMD
DMD = dmdtools.DMD(total=False, n_rank=13)
DMD.fit(X, Y)
dmd_vals, dmd_modes = DMD.get_mode_pairs()

fig = plt.figure(1)
th = np.linspace(0, 2*np.pi, 101)
plt.plot(np.cos(th), np.sin(th), '-', color='0.75', lw=4)
plt.plot(np.diag(Alow).real, np.diag(Alow).imag, 'k.', ms=14)
plt.plot(dmd_vals.real, dmd_vals.imag, 'x', ms=14)


DMDr = dmdtools.RegularizedDMD(1.0, "trace", rho=1.0, cutoff=(1e-4, 1e-4), max_iter=2000)
DMDr.fit(X, Y)
dmdr_vals, dmdr_modes = DMDr.get_mode_pairs()

plt.plot(dmdr_vals.real, dmdr_vals.imag, '+', ms=14)

plt.show()




