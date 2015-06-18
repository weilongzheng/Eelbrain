# Author: Christian Brodbeck <christianbrodbeck@nyu.edu>
#cython: boundscheck=False, wraparound=False

import numpy as np
cimport numpy as np
from libc.math cimport exp, log, sqrt


ctypedef np.int64_t int64
ctypedef np.float64_t float64

def gaussian_smoother(np.ndarray[float64, ndim=2] dist, double fwhm):
    """Create a gaussian smoothing matrix

    Parameters
    ----------
    dist : array (float64)
        Distances; dist[i, j] should provide the distance for any vertex pair i,
         j. Distances < 0 indicate absence of a connection.
    fwhm : float
        The full width at half maximum of the kernel.

    Returns
    -------
    kernel : array (float64)
        Gaussian smoothing kernel, with same shape as dist.
    """
    cdef int64 source, target
    cdef long n_vertices = len(dist)
    cdef double std = fwhm / (2 * (sqrt(2 * log(2))))
    cdef double a = 1. / (std * sqrt(2 * np.pi))
    cdef np.ndarray out = np.empty((n_vertices, n_vertices), np.float64)

    if dist.shape[1] != n_vertices:
        raise ValueError("dist needs to be rectangular, got shape")

    for target in range(n_vertices):
        for source in range(n_vertices):
            if dist[target, source] < 0:
                out[target, source] = 0
            else:
                out[target, source] = a * exp(- (dist[target, source] / std) ** 2 / 2)

    return out
