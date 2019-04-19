cimport cython
import numpy as np
cimport numpy as np
from libc.math cimport log
from cython.parallel import prange, parallel

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mutual_likelihood_score_parallel(const float[:,:] mu1, const float[:,:] mu2, const float[:,:] sigma_sq1, const float[:,:] sigma_sq2):
    cdef int m = mu1.shape[0]
    cdef int n = mu2.shape[0]
    cdef int d = mu1.shape[1]
    assert mu1.shape[1] == mu2.shape[1]
    assert sigma_sq1.shape[1] == sigma_sq2.shape[1]
    assert mu1.shape[0] == sigma_sq1.shape[0]
    assert mu2.shape[0] == sigma_sq2.shape[0]
    
    cdef float dist
    cdef float s_sum
    cdef float [:,:] result = np.empty(shape=(m,n), dtype=np.float32)
    cdef int i, j, k

    for i in prange(m, nogil=True):
        for j in range(n):
            dist = 0
            for k in range(d):
               s_sum = sigma_sq1[i,k]  + sigma_sq2[j,k]
               dist = dist + (mu1[i,k] - mu2[j,k]) * (mu1[i,k] - mu2[j,k]) / s_sum + log(s_sum)
            result[i,j] = -dist
    return result

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def mutual_likelihood_score(const double[:,:] mu1, const double[:,:] mu2, const double[:,:] sigma_sq1, const double[:,:] sigma_sq2):
    cdef int m = mu1.shape[0]
    cdef int n = mu2.shape[0]
    cdef int d = mu1.shape[1]
    assert mu1.shape[1] == mu2.shape[1]
    assert sigma_sq1.shape[1] == sigma_sq2.shape[1]
    assert mu1.shape[0] == sigma_sq1.shape[0]
    assert mu2.shape[0] == sigma_sq2.shape[0]
    
    cdef double dist
    cdef double s_sum
    cdef double [:,:] result = np.empty(shape=(m,n))
    cdef int i, j, k

    for i in range(m):
        for j in range(n):
            dist = 0
            for k in range(d):
               s_sum = sigma_sq1[i,k]  + sigma_sq2[j,k]
               dist = dist + (mu1[i,k] - mu2[j,k]) * (mu1[i,k] - mu2[j,k]) / s_sum + log(s_sum)
            result[i,j] = -dist
    return result
