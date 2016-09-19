import cython
import numpy as np
cimport numpy as np
from cpython cimport bool
# from libc.stdlib cimport free


np.import_array()

ctypedef struct FullMatch:
    int xA, xB, yA, yB
    float dtw
    float length
    float disto

cdef extern from "../ZRTools/plebdisc/dot.h":
    ctypedef struct Match:
        int xA, xB, yA, yB
        float rhoampl
        float score

# match = np.dtype([('xA', 'i4'), ('xB', 'i4'), ('yA', 'i4'), ('yB', 'i4'), ('rhoampl', 'f4', 'score', 'f4')])

cdef extern from "cosine.h":
    ctypedef unsigned char byte
    float approximate_cosine(byte* x, byte* y, int sig_num_bytes)
    bint signature_is_zeroed(byte* x, int sig_num_bytes)

cdef int SPHW = 100
cdef float epsilon = np.finfo(np.float32).eps
cdef int MIN_LEN = 4


@cython.boundscheck(False)
def secondpass(np.ndarray[Match, ndim=1] matchlist, np.ndarray feats1, np.ndarray feats2, int R, float castthr, float trimthr, int strategy, int nbest=5, bool exact=False):
    """Returns the best matches for each initial point.
    """
    cdef np.ndarray[Match, ndim=2] new_matchlist = np.zeros((matchlist.shape[0], nbest**2,), dtype=[('xA', 'i4'), ('xB', 'i4'), ('yA', 'i4'), ('yB', 'i4'), ('rhoampl', 'f4'), ('score', 'f4')])
    cdef int n_matches = 0
    cdef float ALPHA = 0.5
    cdef int N1 = feats1.shape[0]
    cdef int N2 = feats2.shape[0]
    cdef int xM, yM
    cdef np.ndarray[int, ndim=1] xA, yA, xB, yB
    cdef np.ndarray[int, ndim=2] resA, resB
    cdef int i, j
    for n in range(matchlist.shape[0]):
        xM = int(0.5*(matchlist[n].xA+matchlist[n].xB))
        yM = int(0.5*(matchlist[n].yA+matchlist[n].yB))
      
        resA = sig_find_paths(feats1, feats2, -1, xM, yM, castthr, trimthr, R, strategy, ALPHA, nbest, exact);
        resA = np.append(resA, np.zeros((nbest-resA.shape[0], 2), dtype=np.int32), axis=0)
        xA = resA[:, 0]
        yA = resA[:, 1]
        xA = xM-xA
        yA = yM-yA
      
        resB = sig_find_paths(feats1, feats2, 1, xM, yM, castthr, trimthr, R, strategy, ALPHA, nbest, exact)
        resB = np.append(resB, np.zeros((nbest-resB.shape[0], 2), dtype=np.int32), axis=0)
        xB = resB[:, 0]
        yB = resB[:, 1]
        xB = xM+xB
        yB = yM+yB

        # assert np.all(xA >= 0) and np.all(yA >= 0) and np.all(xB < N1) and np.all(yB < N2)

        for i in range(nbest):
            for j in range(nbest):
                new_matchlist[n,i*nbest + j].xA = max(xA[i], 0)
                new_matchlist[n,i*nbest + j].xB = min(xB[j], N1-1)
                new_matchlist[n,i*nbest + j].yA = max(yA[i], 0)
                new_matchlist[n,i*nbest + j].yB = min(yB[j], N2-1)
    return new_matchlist


@cython.boundscheck(False)
def secondpass_exact_aren(np.ndarray[Match, ndim=1] matchlist, np.ndarray[float, ndim=2] feats1, np.ndarray[float, ndim=2] feats2, int R, float castthr, float trimthr, int strategy, int nbest=5):
    cdef np.ndarray[FullMatch] new_matchlist = np.zeros((matchlist.shape[0],), dtype=[('xA', 'i4'), ('xB', 'i4'), ('yA', 'i4'), ('yB', 'i4'), ('dtw', 'f4'), ('length', 'f4'), ('disto', 'f4')])
    cdef int n_matches = 0
    cdef float ALPHA = 0.5
    cdef int N1 = feats1.shape[0]
    cdef int N2 = feats2.shape[0]
    cdef int xM, yM
    cdef int xA, yA, xB, yB
    cdef np.ndarray[int, ndim=1] resA, resB
    cdef int i, j
    cdef float d, disto, l
    for n in range(matchlist.shape[0]):
        xM = int(0.5*(matchlist[n].xA+matchlist[n].xB))
        yM = int(0.5*(matchlist[n].yA+matchlist[n].yB))

        xA, xB = sig_find_paths_aren_exact(feats1, feats2, -1, xM, yM, castthr, trimthr, R, strategy, ALPHA, nbest);
        xA = xM-xA
        yA = yM-yA
      
        yA, yB = sig_find_paths_aren_exact(feats1, feats2, 1, xM, yM, castthr, trimthr, R, strategy, ALPHA, nbest)
        xB = xM+xB
        yB = yM+yB

        # assert np.all(xA >= 0) and np.all(yA >= 0) and np.all(xB < N1) and np.all(yB < N2)

        new_matchlist[n].xA = max(xA, 0)
        new_matchlist[n].xB = min(xB, N1-1)
        new_matchlist[n].yA = max(yA, 0)
        new_matchlist[n].yB = min(yB, N2-1)

        dist_array = outer_cosine(feats1[new_matchlist[n].xA:new_matchlist[n].xB+1], feats2[new_matchlist[n].yA:new_matchlist[n].yB+1])
        d, disto = dtw(dist_array)
        l = np.sqrt((new_matchlist[n].xA - new_matchlist[n].xB+1)**2 + (new_matchlist[n].yA-new_matchlist[n].yB+1)**2)
        new_matchlist[n].dtw = d
        new_matchlist[n].length = l
        new_matchlist[n].disto = disto
        
    return new_matchlist


@cython.boundscheck(False)
@cython.cdivision(True)
cdef sig_find_paths_aren_exact(np.ndarray[float, ndim=2] feats1, np.ndarray[float, ndim=2] feats2, int direction, int xM, int yM, float castthr, float trimthr, int R, int strategy, float alpha, int nbest):

    cdef float bound = 1e10
    cdef np.ndarray[float, ndim=2] scr = np.ones((SPHW+1, SPHW+1), dtype=np.float32) * bound
    cdef np.ndarray[int, ndim=2] path = np.zeros((SPHW+1, SPHW+1), dtype=np.int32)
    cdef float prev_cost
    cdef np.ndarray[int, ndim=2] path_lengths = np.ones((SPHW+1, SPHW+1), dtype=np.int32) * SPHW

    scr[0, 0] = 0
    path_lengths[0, 0] = 0
    cdef float value
    cdef int xE = 0
    cdef int yE = 0
    cdef int N1 = feats1.shape[0]
    cdef int N2 = feats2.shape[0]
    assert feats1.shape[1] == feats2.shape[1]
    cdef float subst_cost
    cdef bint cont
    cdef int i, j
    cdef int dim = feats1.shape[1]
    cdef int x, y
    cdef np.ndarray[float, ndim=1] featx
    cdef np.ndarray[float, ndim=1] featy

    cdef int n_elt = 1
    cdef np.ndarray[float, ndim=2] elts = np.empty(((SPHW+1)**2, 2), dtype=np.float32)
    cdef np.ndarray[int, ndim=2] elts_idx = np.empty(((SPHW+1)**2, 2), dtype=np.int32)
    elts[0] = 0, 0
    elts_idx[0] = 0, 0

    for i in range(1, SPHW+1):
        cont = False
        for j in range(max(1,i-R), min(SPHW+1,i+R)):

            x = direction*i+xM
            y = direction*j+yM

            if ( x < 0 or x >= N1 or y < 0 or y >= N2 ):
                continue

            subst_cost = 0

            featx = feats1[x]
            featy = feats2[y]
            if np.all(featx == 0) or np.all(featy == 0):
                subst_cost = -1
            else:
                subst_cost = cosine(featx, featy)

            subst_cost = (1-subst_cost)/2

            if scr[i-1, j-1] <= scr[i-1, j] and scr[i-1, j-1] <= scr[i, j-1]:
                prev_cost = scr[i-1, j-1]
                path[i, j] = 1
                if (strategy == 2):
                    path_lengths[i, j] = path_lengths[i-1, j-1] + 1
            elif scr[i-1, j] <= scr[i, j-1]:
                prev_cost = scr[i-1, j]
                path[i, j] = 2
                if (strategy == 2):
                    path_lengths[i, j] = path_lengths[i-1, j] + 1
            else:
                prev_cost = scr[i, j-1]
                path[i, j] = 3
                if (strategy == 2):
                    path_lengths[i, j] = path_lengths[i, j-1] + 1

            if (strategy == 0 or strategy == 2):
                scr[i, j] = prev_cost + subst_cost
            elif (strategy == 1):
                scr[i, j] = alpha * prev_cost + (1 - alpha) * subst_cost

            if strategy == 2:
                value = scr[i, j]  / path_lengths[i, j]
            else:
                value = scr[i, j]
                
            if value > castthr:
                scr[i, j] = bound
            else:
                cont = True
                if (i+1)*(i+1)+(j+1)*(j+1) > (xE)*(xE)+(yE)*(yE):
                    xE = i+1
                    yE = j+1

        if not cont:
            break

    # Trim the loose ends
    cdef float last
    cdef int s
    while( xE > 0 and yE > 0 ):
        last = scr[xE-1, yE-1]
        s = path[xE-1, yE-1]
        xE = xE - (s==2 or s==1)
        yE = yE - (s==3 or s==1)
        if ( s == 0 or last-scr[xE-1, yE-1] < trimthr ):
            break

    return xE, yE


# @cython.wraparound(False)
@cython.boundscheck(False)
@cython.cdivision(True)
cdef np.ndarray[int, ndim=2] sig_find_paths(np.ndarray feats1, np.ndarray feats2, int direction, int xM, int yM, float castthr, float trimthr, int R, int strategy, float alpha, int nbest, bool exact):

    cdef float bound = 1e10
    cdef np.ndarray[float, ndim=2] scr = np.ones((SPHW+1, SPHW+1), dtype=np.float32) * bound
    cdef np.ndarray[int, ndim=2] path = np.zeros((SPHW+1, SPHW+1), dtype=np.int32)
    cdef float prev_cost
    cdef np.ndarray[int, ndim=2] path_lengths = np.ones((SPHW+1, SPHW+1), dtype=np.int32) * SPHW

    scr[0, 0] = 0
    path_lengths[0, 0] = 0
    cdef float value
    cdef int xE = 0
    cdef int yE = 0
    cdef int N1 = feats1.shape[0]
    cdef int N2 = feats2.shape[0]
    assert feats1.shape[1] == feats2.shape[1]
    cdef float subst_cost
    cdef bint cont
    cdef int i, j
    cdef int nbytes = feats1.shape[1]
    cdef int x, y

    cdef np.ndarray[float, ndim=1] featx
    cdef np.ndarray[float, ndim=1] featy

    cdef byte* featx_approx
    cdef byte* featy_approx
    
    cdef int n_elt = 1
    cdef np.ndarray[float, ndim=2] elts = np.empty(((SPHW+1)**2, 2), dtype=np.float32)
    cdef np.ndarray[int, ndim=2] elts_idx = np.empty(((SPHW+1)**2, 2), dtype=np.int32)
    elts[0] = 0, 0
    elts_idx[0] = 0, 0

    for i in range(1, SPHW+1):
        cont = False
        for j in range(max(1,i-R), min(SPHW+1,i+R)):

            x = direction*i+xM
            y = direction*j+yM

            if ( x < 0 or x >= N1 or y < 0 or y >= N2 ):
                continue

            subst_cost = 0

            if exact:
                featx = feats1[x]
                featy = feats2[y]
                if np.all(featx == 0) or np.all(featy == 0):
                    subst_cost = -1
                else:
                    subst_cost = cosine(featx, featy)
            else:
                featx_approx = <byte*> (feats1.data + x*nbytes*sizeof(byte))
                featy_approx = <byte*> (feats2.data + y*nbytes*sizeof(byte))
                if signature_is_zeroed(featx_approx, nbytes) or signature_is_zeroed(featy_approx, nbytes):
                    subst_cost = -1
                else:
                    subst_cost = approximate_cosine(featx_approx, featy_approx, nbytes)

            subst_cost = (1-subst_cost)/2

            if scr[i-1, j-1] <= scr[i-1, j] and scr[i-1, j-1] <= scr[i, j-1]:
                prev_cost = scr[i-1, j-1]
                path[i, j] = 1
                if (strategy == 2):
                    path_lengths[i, j] = path_lengths[i-1, j-1] + 1
            elif scr[i-1, j] <= scr[i, j-1]:
                prev_cost = scr[i-1, j]
                path[i, j] = 2
                if (strategy == 2):
                    path_lengths[i, j] = path_lengths[i-1, j] + 1
            else:
                prev_cost = scr[i, j-1]
                path[i, j] = 3
                if (strategy == 2):
                    path_lengths[i, j] = path_lengths[i, j-1] + 1

            if (strategy == 0 or strategy == 2):
                scr[i, j] = prev_cost + subst_cost
            elif (strategy == 1):
                scr[i, j] = alpha * prev_cost + (1 - alpha) * subst_cost

            if strategy == 2:
                value = scr[i, j]  / path_lengths[i, j]
            else:
                value = scr[i, j]
                
            if value > castthr:
                scr[i, j] = bound
            else:
                cont = True
                elts[n_elt, 0] = i*i+j*j
                elts[n_elt, 1] = value
                elts_idx[n_elt, 0] = i
                elts_idx[n_elt, 1] = j
                n_elt += 1

        if not cont:
            break


    elts = elts[:n_elt]
    elts_idx = elts_idx[:n_elt]
    elts = (elts - np.mean(elts, axis=0)) / (np.std(elts, axis=0) + epsilon)
    cdef np.ndarray[float, ndim=1] sum_elts = elts[:, 0] - elts[:, 1]
    # sort is n*log(n), min is n, do best method according to the length
    # of the array and number of elements returned
    cdef np.ndarray[long, ndim=1] sorted_idx = np.argsort(sum_elts)
    return elts_idx[sorted_idx[n_elt-nbest:]]


def rescore_matchlist(np.ndarray[Match, ndim=2] matchlist,
                      np.ndarray feats1,
                      np.ndarray feats2,
                      bool with_dist=True):
    cdef int i = 0
    cdef int j = 0
    cdef int nbest = matchlist.shape[1]
    cdef int nmatch = matchlist.shape[0]
    cdef int xA, xB, yA, yB
    cdef int ibest = 0
    cdef np.ndarray f1, f2
    # cdef np.ndarray[byte, ndim=2] f1, f2
    cdef np.ndarray[float, ndim=2] dist_array, dist_array_i
    cdef np.ndarray[FullMatch, ndim=1] new_matchlist = np.zeros((matchlist.shape[0],), dtype=[('xA', 'i4'), ('xB', 'i4'), ('yA', 'i4'), ('yB', 'i4'), ('dtw', 'f4'), ('length', 'f4'), ('disto', 'f4')])
    cdef np.ndarray[float, ndim=1] dtws = np.empty((nbest,), dtype=np.float32)
    cdef np.ndarray[float, ndim=1] lengths = np.empty((nbest,), dtype=np.float32)
    cdef np.ndarray[float, ndim=1] distos = np.empty((nbest,), dtype=np.float32)
    cdef np.ndarray[float, ndim=1] scores = np.empty((nbest,), dtype=np.float32)
    cdef np.ndarray[float, ndim=1] stddtws, stdlengths
    cdef Match bestmatch
    cdef int N1 = feats1.shape[0]
    cdef int N2 = feats2.shape[0]
    cdef int xAmin, xBmin, yAmax, yBmax
    for i in range(nmatch):
        xAmin = N1; xBmax = 0; yAmin = N2; yBmax = 0
        for j in range(nbest):
            if matchlist[i, j].xA < xAmin:
                xAmin = matchlist[i, j].xA
            if matchlist[i, j].xB > xBmax:
                xBmax = matchlist[i, j].xB
            if matchlist[i, j].yA < yAmin:
                yAmin = matchlist[i, j].yA
            if matchlist[i, j].yB > yBmax:
                yBmax = matchlist[i, j].yB
        f1 = feats1[xAmin:xBmax+1, :]
        f2 = feats2[yAmin:yBmax+1, :]
        dist_array = outer_cosine(f1, f2)
        # dist_array = outer_approximate_cosine(f1, f2)
    # for i in range(nmatch):
        for j in range(nbest):
            xA = matchlist[i, j].xA
            yA = matchlist[i, j].yA
            xB = matchlist[i, j].xB
            yB = matchlist[i, j].yB
            dist_array_i = dist_array[xA-xAmin:xB-xAmin+1, yA-yAmin:yB-yAmin+1]
            dtws[j], distos[j] = dtw(dist_array_i)
            lengths[j] = np.sqrt(float((xB-xA)**2 + (yB-yA)**2))
        stddtws = (dtws - np.mean(dtws)) / (np.std(dtws) + epsilon)
        stdlengths = (lengths - np.mean(lengths)) / (np.std(lengths) + epsilon)
        # stddistos = (distos - np.mean(distos)) / np.std(distos)
        if with_dist:
            scores = - stddtws + stdlengths - distos
        else:
            scores = -stddtws
        ibest = np.argmax(scores)
        bestmatch = matchlist[i, ibest]
        new_matchlist[i].xA = bestmatch.xA; new_matchlist[i].xB = bestmatch.xB
        new_matchlist[i].yA = bestmatch.yA; new_matchlist[i].yB = bestmatch.yB
        new_matchlist[i].dtw = dtws[ibest]
        new_matchlist[i].length = lengths[ibest]
        new_matchlist[i].disto = distos[ibest]
    return new_matchlist


# Replace by numpy ufunc and outer for efficiency ?
@cython.boundscheck(False)
cdef np.ndarray[float, ndim=2] outer_approximate_cosine(np.ndarray feats1, np.ndarray feats2):
# cdef np.ndarray[float, ndim=2] outer_approximate_cosine(np.ndarray[byte, ndim=2] feats1, np.ndarray[byte, ndim=2] feats2):
    assert feats1.shape[1] == feats2.shape[1]
    cdef int nbytes = feats1.shape[1]
    cdef int N = feats1.shape[0]
    cdef int M = feats2.shape[0]
    cdef int i = 0
    cdef int j = 0
    cdef np.ndarray[float, ndim=2] dist_array = np.empty((N, M), dtype=np.float32)
    cdef byte* featx
    cdef byte* featy
    for i in range(0, N):
        for j in range(0, M):
            featx = <byte*> (feats1.data + i*nbytes*sizeof(byte))
            featy = <byte*> (feats2.data + j*nbytes*sizeof(byte))
            if signature_is_zeroed(featx, nbytes) or signature_is_zeroed(featy, nbytes):
                dist_array[i, j] = 1
            else:
                dist_array[i, j] = (1 - approximate_cosine(featx, featy, nbytes)) / 2
    return dist_array


@cython.boundscheck(False)
cdef float cosine(np.ndarray[float, ndim=1] x, np.ndarray[float, ndim=1] y):
    cdef float x2 = np.sqrt(np.sum(x ** 2))
    cdef float y2 = np.sqrt(np.sum(y ** 2))
    return np.dot(x, y.T) / (x2*y2 + epsilon)


# author: Thomas Schatz, Roland Thiolliere
def outer_cosine(x, y):
    x2 = np.sqrt(np.sum(x ** 2, axis=1))
    y2 = np.sqrt(np.sum(y ** 2, axis=1))
    ix = x2 == 0.
    iy = y2 == 0.
    d = np.dot(x, y.T) / (np.outer(x2, y2) + epsilon)
    d = 1 - d/2
    d[ix, :] = 1.
    d[:, iy] = 1.
    for i in ix:
        d[i, iy] = 0.
    return d.astype(np.float32)


@cython.boundscheck(False)
cdef dtw(np.ndarray[float, ndim=2] dist_array):
    cdef int N = dist_array.shape[0]
    cdef int M = dist_array.shape[1]
    cdef int i, j
    cdef np.ndarray[float, ndim=2] cost = np.empty((N, M), dtype=np.float32)
    cdef np.ndarray[signed char, ndim=2] path_dir = np.empty((N, M), dtype=np.int8)
    cdef np.ndarray[int, ndim=2] path_len = np.empty((N, M), dtype=np.int32)
    # cdef cnp.npy_intp* dims= [N, M]
    # cdef np.ndarray[float, ndim=2] cost = np.PyArray_EMPTY(2, dims, np.NPY_FLOAT32, 0)
    # cdef np.ndarray[signed char, ndim=2] path_dir = np.PyArray_EMPTY(2, dims, np.NPY_INT8, 0)
    # cdef np.ndarray[int, ndim=2] path_len = np.PyArray_EMPTY(2, dims, np.NPY_INT32, 0)
    cdef float final_cost
    # initialization
    cost[0,0] = dist_array[0,0]
    path_len[0, 0] = 1
    for i in range(1,N):
        cost[i,0] = dist_array[i,0] + cost[i-1,0]
        path_len[i, 0] = i+1
    for j in range(1,M):
        cost[0,j] = dist_array[0,j] + cost[0,j-1]
        path_len[0, j] = j+1
    # the dynamic programming loop
    for i in range(1,N):
        for j in range(1,M):
            if cost[i-1, j-1] < cost[i-1, j] and cost[i-1, j-1] < cost[i, j-1]:
                cost[i, j] = dist_array[i, j] + cost[i-1, j-1]
                path_dir[i, j] = 0
                path_len[i, j] = path_len[i-1, j-1] + 1
            elif cost[i-1, j] < cost[i, j-1]:
                cost[i, j] = dist_array[i, j] + cost[i-1, j]
                path_dir[i, j] = -1
                path_len[i, j] = path_len[i-1, j] + 1
            else:
                cost[i, j] = dist_array[i, j] + cost[i, j-1]
                path_dir[i, j] = 1
                path_len[i, j] = path_len[i, j-1] + 1

    final_cost = cost[N-1, M-1] / path_len[N-1, M-1]
    #TODO:
    # cdef float disto = float(np.sum(path_dir != 0 )) / path_len[N-1, M-1]
    cdef float disto = 0
    return final_cost, disto
    # return final_cost, path_len[N-1, M-1], disto
