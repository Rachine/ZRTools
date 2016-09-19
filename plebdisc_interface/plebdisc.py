"""Call to the plebdisc functions.

What are "xp" and "yp" ?
The similarity matrix is rotated by 45 degrees. To visualise it, take the DTW
matrix between 2 words. The "xp" axis is along the DTW path and "yp" is
orthogonal to "xp". The values of "yp" are also shifted so that they are
always positive.

What are Dot, DotXV and cumsum ?
A dot is a point in the similarity matrix positioned by its coordinates
"xp" and "yp"
A dotXV is a point only positionned by its "xp" coordinate. DotXV only exists in
lists sorted by their "yp" coordinate. We need the cumsum to know how many points
share the same "yp" coordinates and retrieve the "yp" value.

dx and dy:
sometimes in internal functions, those parmeters are multiplied of divided by 2
so we need to check for that

Matches:
matches are positionned the their cordinates in the similarity matrix in the
original frame. Matches a tuples (int xA, int xB, int yA, int yB,
float rhoampl, float score). So the 1st element of the pair, in the 1st file,
starts at frame xA and end at frame xB. And the 2nd element of the pair, in
the 2nd file, start at frame yA and end at frame yB.
"""
import numpy as np
import ctypes
import plebdisc_wrapper


compute_similarity_matrix = plebdisc_wrapper.py_compute_similarity_matrix
"""Compute the "similaity matrix" between the 2 inputs features

Parameters:
----------
feats1: array, the features of the 1st file after lsh
feats2: array, the features of the 2nd file after lsh
P: int, number of permutations
B: int, width of the band search
D: int, length of the diagonal search if match
T: float, threshold for match
diffspeech: int, 1 if the files are different, 0 otherwise

Returns:
-------
dotlist: array, array of DotXV, need cumsum to get the original indices
cumsum: array, cumulative sum of the "yp" indices in the dotlist
"""

filter_matrix = plebdisc_wrapper.py_filter_matrix
"""Filter matrix with median filter along the "xp" axis and gaussian filter
along the "yp" axis. The similarity matrix becomes binary.

Parameters:
----------
dotlist: array, array of DotXV, need cumsum to get the original indices
cumsum: array, cumulative sum of the "yp" indices in the dotlist
Nmax: int, maximal number of features between both files
dx: width of the median filter along the "xp" axis
dy: width of the gaussian filter along the "yp" axis
medthr: median filter parameter

Returns:
-------
dotlist_mf: array, array of DotXV, need cumsum to get the original indices
cumsum_mf: array, cumulative sum of the "yp" indices in the dotlist
hough: array, result of the Hough transform
"""

find_matches = plebdisc_wrapper.py_find_matches
"""Find potential matches in the similarity matrix. A match is a succession
of points in the similarity matrix with a value of 1.

Parameters:
----------
dotlist_mf: array, array of DotXV, need cumsum to get the original indices
cumsum_mf: array, cumulative sum of the "yp" indices in the dotlist
dx: int, minimal length of a match.
dy: int, maximal deviation of the path
diffspeech: int, 1 if the files are the same, 0 otherwise
Nmax: int, maximal number of features between both files
rhothr: threshold on the Hough transform. Unnecessary, use 0 for no threshold

Returns:
matchlist: array, matches.
"""

refine_matches = plebdisc_wrapper.py_refine_matches
refine_matches2 = plebdisc_wrapper.py_refine_matches2


def run(file1, file2, S=64, P=4, B=50, D=5, T=0.25, dx=25, dy=3, medthr=0.5, rhothr=0, R=10, castthr=7, trimthr=0.2):
    diffspeech = file1 != file2
    feats1, feats2 = read_feats(file1, file2, S)
    Nmax = max(feats1.shape[0], feats2.shape[0])
    dotlist, cumsum = compute_similarity_matrix(feats1, feats2, P, B, D, T, diffspeech)
    dotlist_mf, cumsum_mf, hough = filter_matrix(dotlist, cumsum, diffspeech, Nmax, dx, dy, medthr)
    matchlist = find_matches(dotlist_mf, cumsum_mf, hough, dx, dy, diffspeech, Nmax, rhothr)
    return matchlist
    # matchlist2 = np.copy(matchlist)
    # matchlist = refine_matches(matchlist, feats1, feats2, R, castthr, trimthr)
    # matchlist2 = refine_matches2(matchlist, feats1, feats2, R, castthr, trimthr, 0)


def read_feats(file1, file2, S):
    feats1 = np.fromfile(file1, dtype=ctypes.c_ubyte)
    feats1 = feats1.reshape((-1, S/8))
    feats2 = np.fromfile(file2, dtype=ctypes.c_ubyte)
    feats2 = feats2.reshape((-1, S/8))
    return feats1, feats2
