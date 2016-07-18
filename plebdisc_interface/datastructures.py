import numpy as np


def similarity_matrix2npy(similarity_matrix, N=None, M=None, diffspeech=0):
    return dotlist2mat(dotlistXV2dotlist(similarity_matrix[0], similarity_matrix[1], max(N, M), diffspeech), N, M)


def dotlist2mat(dotlist, N=None, M=None):
    xp, yp, val = dotlist['xp'], dotlist['yp'], dotlist['val']
    x = (xp - yp) / 2
    y = (xp + yp) / 2
    # print np.min(x)
    # print np.min(y)
    # print np.max(x) + 1
    # print np.max(y) + 1
    if N == None:
        N = np.max(x) + 1
    if M == None:
        M = np.max(y) + 1
    mat = np.zeros((N, M))
    for i in range(len(dotlist)):
        mat[x[i], y[i]] = val[i]
    return mat


def dotlistXV2dotlist(dotlistXV, cumsum, Nmax, diffspeech=False):
    assert cumsum[0] == 0
    diffspeech=int(diffspeech)
    # counts = cumsum[1:] - cumsum[:-1]
    Ndots = cumsum[-1]
    dotlist = np.empty((Ndots,), dtype=[('xp', 'i4'), ('yp', 'i4'), ('val', 'f4')])
    i = 0
    for yp, S in enumerate(cumsum[1:]):
        while i < S:
            dotlist[i] = dotlistXV[i]['xp'], yp - diffspeech*Nmax, dotlistXV[i]['val']
            i += 1
    assert i == Ndots, i
    return dotlist


def matchlist_getmiddles(matchlist):
    xA, xB, yA, yB, rhoampl = matchlist['xA'], matchlist['xB'], matchlist['yA'], matchlist['yB'], matchlist['rhoampl']
    # xA, xB, yA, yB, rhoampl = matchlist['xA'], matchlist['xB'], matchlist['yA'], matchlist['yB'], matchlist['rhoampl']
    # x1 = (xA - yA) / 2
    # y1 = (xA + yA) / 2
    # x2 = (xB - yB) / 2
    # y2 = (xB + yB) / 2
    x = (xA + xB).astype(float) / 2
    y = (yA + yB).astype(float) / 2
    middles = np.empty((len(matchlist,)), dtype=[('x', 'f4'), ('y', 'f4'), ('rhoampl', 'f4')])
    middles['x'] = x
    middles['y'] = y
    middles['rhoampl'] = rhoampl
    return middles

def matchlist2mat(matchlist, N=None, M=None):
    x1, x2, y1, y2, rhoampl = matchlist['xA'], matchlist['xB'], matchlist['yA'], matchlist['yB'], matchlist['rhoampl']
    # x1 = (xA - yA) / 2
    # y1 = (xA + yA) / 2
    # x2 = (xB - yB) / 2
    # y2 = (xB + yB) / 2
    if N == None:
        N = max(np.max(x1), np.max(x2)) + 1
    if M == None:
        M = max(np.max(y1), np.max(y2)) + 1
    mat = np.zeros((N, M))
    for i in range(len(matchlist)):
        mat[(x2[i] + x1[i])/2, (y2[i] + y1[i])/2] = rhoampl[i]
    return mat
