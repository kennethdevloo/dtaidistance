import math
import pytest
import numpy as np
from dtaidistance import dtw, dtw_c





def test_keogh_nn():

    X = np.array([[1, 5, 6, 1, 1], [1, 2, 7, 2, 1], [
                 25, 22, 15, 41, 21]], dtype=np.double)
    X2 = np.array([[1, 5, 6, 1, 1], [25, 2, 15, 41, 21], [
                  25, 22, 15, 41, 21], [1, 2, 7, 2, 1]], dtype=np.double)

    L, U = dtw.lb_keogh_enveloppes_fast(X2, 2)
    Y = dtw.nearest_neighbour_lb_keogh_fast(
        X, X2, L, U, distParams={'window': 2, 'psi': 0})

    assert np.array_equal(Y ,np.array([0,3,2], dtype = np.int))

    


def test_distance_matrix_block():
    for parallel in [False, True]:
        for use_c in [False,True]:
            for use_nogil in [False, True]:
                run_distance_matrix_block(parallel=parallel, use_c=use_c, use_nogil=use_nogil)


if __name__ == "__main__":
    test_keogh_nn()
