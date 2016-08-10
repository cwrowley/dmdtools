""" An example highlighting the difference between TLS-DMD and DMD

    TLS-DMD is a total least squares variant of DMD, which can produce
    superior results when the data provided to the method are noisy.
    This example is meant to highlight the difference between the two
    methods on a simple problem where the true solution is already known.

    Returns
    -------
    Outputs a plot comparing the true, DMD, and TLS-DMD eigenvalues
"""
import sys
sys.path.append('..')

import numpy as np
import matplotlib.pyplot as plt
import dmdtools

if __name__ == "__main__":
    np.random.seed(0)

    # ======== System Parameters =======
    n_rank = 2  # True rank of the system
    n = 250  # Number of states
    m = 1000  # Number of snapshots
    std = 5e-1  # standard deviation of the noise

    # The true system is 2 dimensional and oscillatory
    Alow = np.diag(np.exp([1j, 0.65j]))
    data = np.zeros((n_rank, m+1), dtype="complex")
    data[:, 0] = np.random.randn(n_rank) + 1j*np.random.randn(n_rank)

    for ii in xrange(m):
        data[:, ii+1] = Alow.dot(data[:, ii])
    Q = np.linalg.qr(np.random.randn(n, 2))[0]

    data = Q.dot(data)
    data = np.r_[data.real, data.imag]  # Split and stack real and image parts

    # Add noise to the data
    noisy_data = data + std*np.random.randn(data.shape[0], data.shape[1])

    # Create a new figure for output
    fig = plt.figure(1)
    th = np.linspace(0, 2*np.pi, 101)
    plt.plot(np.cos(th), np.sin(th), '-', color='0.75', lw=4)
    plt.plot(np.diag(Alow).real, np.diag(Alow).imag, 'ko', ms=14)

    # Note:  n_rank is doubled because we only deal with real numbers
    dmd = dmdtools.DMD(n_rank*2, False, False)  # "standard" DMD
    dmd = dmd.fit(noisy_data)
    dmd_vals, dmd_modes = dmd.get_mode_pairs(sortby="LM")

    # Plot the DMD eigenvalues
    plt.plot(dmd_vals.real, dmd_vals.imag, 'rv', ms=14)

    # With TLS DMD
    tlsdmd = dmdtools.DMD(n_rank*2, False, True)  # "standard" DMD
    tlsdmd = tlsdmd.fit(noisy_data)
    tlsdmd_vals, tlsdmd_modes = tlsdmd.get_mode_pairs(sortby="LM")

    # Plot the DMD eigenvalues
    plt.plot(tlsdmd_vals.real, tlsdmd_vals.imag, 'b^', ms=14)
    plt.xlabel("$\Re(\mu)$")
    plt.ylabel("$\Im(\mu)$")
    plt.legend(["Unit Circle", "True", "DMD", "TLS-DMD"], "lower left")
    plt.xlim([0, 1])
    plt.ylim([0, 1])
    plt.gca().set_aspect("equal")
    plt.title("DMD vs TLS-DMD")
    plt.savefig("tls_dmd_comparison.pdf")
    plt.show()

