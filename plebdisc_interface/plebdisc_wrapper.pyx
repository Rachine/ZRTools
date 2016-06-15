import numpy as np
cimport numpy as np
import ctypes

np.import_array()


cdef extern from "numpy/arrayobject.h":
    ctypedef int intp
    ctypedef extern class numpy.ndarray [object PyArrayObject]:
        cdef char *data
        cdef int nd
        cdef intp *dimensions
        cdef intp *strides
        cdef int flags

cdef extern from "stdint.h" nogil:
    ctypedef ssize_t intptr_t
    ctypedef  size_t uintptr_t


# in the test, we need to make sure that here struct == packed struct
cdef extern from "dot.h":
    ctypedef struct DotXV:
        int xp
        float val
    ctypedef struct Match:
        int xA, xB, yA, yB
        float rhoampl
        float score


ctypedef unsigned char byte
ctypedef unsigned long ulong
ctypedef byte* byte_p

# cdef extern from "signature.h":
#     ctypedef struct signature_t:
#         int id
#         byte_p byte_
#         byte query


ptr_t = np.dtype('i{}'.format(sizeof(intptr_t)))
int_t = np.dtype('i{}'.format(sizeof(int)))
dotxv_t = np.dtype(dtype=[('xp', 'i4'), ('val', 'f4')])


cdef extern from "plebdisc_funs.c":

    np.ndarray[DotXV] compute_similarity_matrix (
        np.ndarray[int] cumsum,
        np.ndarray[byte, ndim=2] feats1, np.ndarray[byte, ndim=2] feats2,
        int N1, int N2, int diffspeech, int P, int B, int D, float T)
    np.ndarray[DotXV] filter_matrix (
        np.ndarray[int] cumsum_mf, np.ndarray[float] hough,
        np.ndarray[DotXV] dotlist, np.ndarray[int] cumsum,
        int diffspeech, int Nmax, int dx, int dy, float medthr)
    void find_matches(
        np.ndarray[Match] matchlist,
        np.ndarray[DotXV] dotlist_mf, np.ndarray[int] cumsum_mf,
        np.ndarray[float] hough,
        int dx, int dy, int diffspeech, int Nmax, float rhothr)
    void refine_matches(
        np.ndarray[Match] matchlist,
        np.ndarray[byte, ndim=2] feats1, np.ndarray[byte, ndim=2] feats2,
        int R, float castthr, float trimthr)
    void refine_matches2(
        np.ndarray[Match] matchlist,
        np.ndarray[byte, ndim=2] feats1, np.ndarray[byte, ndim=2] feats2,
        int R, float castthr, float trimthr, int strategy)


cpdef py_compute_similarity_matrix(np.ndarray[unsigned char, ndim=2] feats1, np.ndarray[unsigned char, ndim=2] feats2, int P, int B, int D, float T, int diffspeech):
    cdef int N1 = feats1.shape[0]
    cdef int N2 = feats2.shape[0]
    cdef int Nmax = max(N1, N2)
    cdef np.ndarray[int] cumsum = np.empty(((diffspeech+1)*Nmax+1,), dtype=np.int32)
    cdef np.ndarray[DotXV] dotlist = compute_similarity_matrix(cumsum, feats1, feats2, N1, N2, diffspeech, P, B, D, T)
    return dotlist, cumsum


cpdef py_filter_matrix(dotlist, cumsum,
                       int diffspeech, int Nmax, int dx, int dy, float medthr):
    # cdef np.ndarray[DotXV] dotlist_mf = np.empty((0,), dotxv_t)
    cdef int n_rows = (diffspeech+1)*Nmax+1
    cdef np.ndarray[int] cumsum_mf = np.empty((n_rows,), dtype=np.int32)
    cdef np.ndarray[float] hough = np.empty((n_rows,), dtype=np.float32)
    cdef np.ndarray[DotXV] dotlist_mf = filter_matrix(
        cumsum_mf, hough, dotlist, cumsum,
        diffspeech, Nmax, dx, dy, medthr)
    return dotlist_mf, cumsum_mf, hough


cpdef py_find_matches(np.ndarray[DotXV] dotlist_mf, np.ndarray[int] cumsum_mf, np.ndarray[float] hough, int dx, int dy, int diffspeech, Nmax, rhothr):
    cdef np.ndarray[Match] matchlist = np.empty((0,), dtype=[
        ('xA', 'i4'), ('xB', 'i4'), ('yA', 'i4'), ('yB', 'i4'),
        ('rhoampl', 'f4'), ('score', 'f4')])
    find_matches(matchlist, dotlist_mf, cumsum_mf, hough, dx, dy, diffspeech, Nmax, rhothr)
    return matchlist


cpdef py_refine_matches(np.ndarray[Match] matchlist,
                        np.ndarray[unsigned char, ndim=2] feats1,
                        np.ndarray[unsigned char, ndim=2] feats2,
                        int R, float castthr, float trimthr):
    refine_matches(matchlist, feats1, feats2, R, castthr, trimthr)
    return matchlist

cpdef py_refine_matches2(np.ndarray[Match] matchlist,
                         np.ndarray[unsigned char, ndim=2] feats1,
                         np.ndarray[unsigned char, ndim=2] feats2,
                         int R, float castthr, float trimthr,
                         int strategy):
    refine_matches2(matchlist, feats1, feats2, R, castthr, trimthr, strategy)
    return matchlist

# def py_read_feats(file1, file2):
#     feats1 = np.fromfile(file1, dtype=np.uint64)
#     feats2 = np.fromfile(file2, dtype=np.uint64)
#     return _py_read_feats(feats1, feats2)


# cdef _py_read_feats(np.ndarray[ulong] feats1, np.ndarray[ulong] feats2):

#     cdef py_signature a
#     print sizeof(a)

#     cdef np.ndarray[py_signature] sigs1 = np.empty((feats1.shape[0],), dtype=[('id', 'i8'), ('byte_', ptr_t), ('query', 'u8')])
    
#     cdef intptr_t addr = <intptr_t> &feats1[0]
#     for i in range(feats1.shape[0]):
#         sigs1['byte_'][i] = addr
#         addr += 1

#     cdef np.ndarray[py_signature] sigs2 = np.empty((feats2.shape[0],), dtype=[('id', int_t), ('byte_', ptr_t), ('query', 'u1')])
    
#     addr = <intptr_t> &feats2[0]
#     for i in range(feats2.shape[0]):
#         sigs2['byte_'][i] = addr
#         addr += 1

#     # cdef signature_t *sigs_c = <signature_t *> sigs1.data

#     return sigs1, sigs2
    
