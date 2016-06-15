import numpy as np
import ctypes
import plebdisc_wrapper


compute_similarity_matrix = plebdisc_wrapper.py_compute_similarity_matrix
filter_matrix = plebdisc_wrapper.py_filter_matrix
find_matches = plebdisc_wrapper.py_find_matches
refine_matches = plebdisc_wrapper.py_refine_matches
refine_matches2 = plebdisc_wrapper.py_refine_matches2


def run(file1, file2, S=64, P=4, B=50, D=5, T=0.25, dx=25, dy=3, medthr=0.5, rhothr=0, R=10, castthr=7, trimthr=0.2):
    diffspeech = file1 != file2
    feats1, feats2 = read_feats(file1, file2, S)
    Nmax = max(feats1.shape[0], feats2.shape[0])
    dotlist, cumsum = compute_similarity_matrix(feats1, feats2, P, B, D, T, diffspeech)
    dotlist_mf, cumsum_mf, hough = filter_matrix(dotlist, cumsum, diffspeech, Nmax, dx, dy, medthr)
    matchlist = find_matches(dotlist_mf, cumsum_mf, hough, dx, dy, diffspeech, Nmax, rhothr)
    matchlist2 = np.copy(matchlist)
    matchlist = refine_matches(matchlist, feats1, feats2, R, castthr, trimthr)
    matchlist2 = refine_matches2(matchlist, feats1, feats2, R, castthr, trimthr, 0)


def read_feats(file1, file2, S):
    feats1 = np.fromfile(file1, dtype=ctypes.c_ubyte)
    feats1 = feats1.reshape((-1, S/8))
    feats2 = np.fromfile(file2, dtype=ctypes.c_ubyte)
    feats2 = feats2.reshape((-1, S/8))
    return feats1, feats2
