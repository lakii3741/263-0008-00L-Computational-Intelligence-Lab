"""
Taken and adjusted from Surprise library source code
Only common users (or items) are considered. 
"""
import numpy as np
cimport numpy as np
from libc.math cimport sqrt

def compute_cosine_sim(int n_items, dict ur, int min_support):
    """
    :param n_items: Total number of items (n_x)
    :param ur: User ratings dictionary {user_id: [(item_id, rating), ...]}
    :param min_support: Minimum common users for a valid similarity
    """

    cdef double [:, ::1] prods = np.zeros((n_items, n_items), dtype=np.float64)
    cdef long [:, ::1] freq = np.zeros((n_items, n_items), dtype=np.int64)
    cdef double [:, ::1] sqi = np.zeros((n_items, n_items), dtype=np.float64)
    cdef double [:, ::1] sqj = np.zeros((n_items, n_items), dtype=np.float64)
    cdef double [:, ::1] sim = np.zeros((n_items, n_items), dtype=np.float64)

    # Loop variables
    cdef int i, j, a, b, u, n_u_ratings
    cdef double ri, rj, denum
    cdef int min_sprt = min_support

    # Pick user u
    for u, user_ratings in ur.items():
        n_u_ratings = len(user_ratings)
        # Pick item i that user u has rated
        for a in range(n_u_ratings):
            i, ri = user_ratings[a]
            # Pick item j that user u has rated
            for b in range(n_u_ratings):
                j, rj = user_ratings[b]

                # update 
                freq[i, j] += 1
                prods[i, j] += ri * rj
                sqi[i, j] += ri * ri
                sqj[i, j] += rj * rj

    # Filling in similarity matrix
    for i in range(n_items):
        # Each item perfectly similar to itself
        sim[i, i] = 1.0
        for j in range(i + 1, n_items):
            # Minimal support constraint
            if freq[i, j] < min_sprt:
                sim[i, j] = 0
            else:
                denum = sqrt(sqi[i, j] * sqj[i, j])
                sim[i, j] = prods[i, j] / denum
            # Using antisymmetry of similarity matrix
            sim[j, i] = sim[i, j]
    
    return np.asarray(sim)
