"""
Test for the plebdisc interface Module
"""
import numpy as np
import plebdisc_interface.datastructures as datastructures
import plebdisc_interface.plebdisc_wrapper as plebdisc_wrapper
import matplotlib.pyplot as plt

def create_random_feats(N, M=4):
    return np.random.randint(255, size=(N, M)).astype('u1')


def test_py_compute_similarity_matrix():
    feats1 = np.zeros((50, 4), dtype=np.uint8)
    feats1[10:20, 0] = 1
    feats1[20:30, 1] = 1
    dist = np.cos(2*np.pi/64)
    T = dist/2 + 0.5
    similarity_matrix = plebdisc_wrapper.py_compute_similarity_matrix(feats1, feats1, 1, 50, 0, T, 1)
    Ndots = similarity_matrix[1][-1]
    print len(similarity_matrix[0])
    print Ndots
    similarity_matrix = datastructures.similarity_matrix2npy(similarity_matrix, 50, 50, diffspeech=1)
    print np.sum(similarity_matrix == 1)
    np.savetxt('t', similarity_matrix, fmt='%.0f')
    expected_res = np.zeros((50, 50))
    expected_res[10:20, 10:20] = 1
    expected_res[20:30, 20:30] = 1
    expected_res[10:20, 20:30] = dist
    expected_res[20:30, 10:20] = dist
    assert Ndots == 200
    assert np.all(similarity_matrix == expected_res)


test_py_compute_similarity_matrix()
