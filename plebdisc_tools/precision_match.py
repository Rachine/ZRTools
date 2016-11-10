"""Module for loading gold transcription and spoken terms
and annotate them

Gold trans:
file
onset offset phone
onset offset phone
...
(time in sec)

spoken terms:
Class classid
file onset offset
file onset offset
...

Class classid
...
"""
import tde.data.interval as interval
import os, errno
import numpy as np
from recordtype import recordtype
import datastructures as ds
import numba as nb

__all__ = ['silentremove', 'load_gold', 'load_matches', 
           'find_match_triphones', 'count_phone_matches', 'count_matches',
           'minimal_achievable_ned', 'min_levenstein_distance',
           'levenstein_distance', 'nb_levestein_distance',
           'nb_min_levestein_distance' ]

# RESOLUTION PARAMETERS:
FRATE = 100  # number of frames per seconds

SIL_LABEL = 'SIL'
NOISES = set(['SIL', 'SPN'])

# FILE1 = 's0101a'
# FILE2 = FILE1
# FILE2 = 's2001a'

def silentremove(filename):
    try:
        os.remove(filename)
    except OSError as e:
        if e.errno != errno.ENOENT:
            # errno.ENOENT = no such file or directory
            raise

LabelledInterval = recordtype('LabelledInterval' ,'label interval')


def load_gold(goldfile):
    res = {}
    current_file = ''
    prev_end = -1
    with open(goldfile) as fin:
        for line in fin:
            splitted = line.strip().split()
            if not splitted:
                # empty line
                continue
            elif len(splitted) == 1:
                # New file
                current_file = splitted[0]
                res[current_file] = []
                prev_end = 0
            else:
                assert len(splitted) == 3, splitted
                start, end = map(float, splitted[:2])
                if prev_end != start:
                    astart, aend = prev_end, start
                    res[current_file].append(LabelledInterval(
                        SIL_LABEL, interval.Interval(astart, aend)))
                phone = splitted[2]
                res[current_file].append(LabelledInterval(
                    phone, interval.Interval(start, end)))
                prev_end = end
    return res


Match = recordtype('Match', ['f1', 't1', 'f2', 't2'])
LabelledMatch = recordtype('Match', ['f1', 't1', 'triphone1', 'word1_left', 'word1_right', 'f2', 't2', 'triphone2', 'word2_left', 'word2_right'])

def load_matches(matchfile):
    FILE1, FILE2 = os.path.basename(matchfile).split(".")[0].split()
    matchlist = ds.read_matchlist(matchfile)
    dotlist = ds.matchlist_getmiddles(matchlist)
    res = np.empty((len(dotlist),), dtype=[('f1', '|S30'), ('t1', 'f4'), ('f2', '|S30'), ('t2', 'f4')])
    res['f1'] = FILE1
    res['f2'] = FILE2
    res['t1'] = dotlist['x'].astype(float)/100
    res['t2'] = dotlist['y'].astype(float)/100
    res = np.sort(res, order=['t1', 't2'])
    return res


def find_match_triphones(gold_dict, matches):
    """Find matches"""
    phone_index = 0
    res = []
    for match in np.sort(matches, order='t1'):
        triphone = []
        fname = match['f1']
        while phone_index < len(gold_dict[fname]):
            phone_interval = gold_dict[fname][phone_index]
            if ((phone_interval.interval.start <= match['t1'] and
                 phone_interval.interval.end > match['t1'])):
                s = SIL_LABEL
                if phone_index > 0:
                    s = gold_dict[fname][phone_index-1].label
                e = SIL_LABEL
                if phone_index < len(gold_dict[fname])-1:
                    e = gold_dict[fname][phone_index+1].label
                triphone = [s, phone_interval.label, e]
                i_start = max(phone_index-10, 0)
                i_end = min(phone_index+11, len(gold_dict[fname]))
                word_left = [x.label for x in
                    gold_dict[fname][i_start:phone_index+1]]
                word_right = [x.label for x in
                    gold_dict[fname][phone_index:i_end]]
                res.append(LabelledMatch(
                    match['f1'], match['t1'], triphone, word_left, word_right,
                    match['f2'], match['t2'], None, None, None))
                break
            else:
                # transcription not started, first phone not encontered
                # let's increment our phone counter
                phone_index += 1
            if phone_index == len(gold_dict[fname]):
                raise

    phone_index = 0
    indexes = np.argsort(matches, order='t2')
    for i, match in enumerate(matches[indexes]):
        assert res[indexes[i]].t2 == match['t2']
        triphone = []
        fname = match['f2']
        while phone_index < len(gold_dict[fname]):
            phone_interval = gold_dict[fname][phone_index]
            if ((phone_interval.interval.start <= match['t2'] and
                 phone_interval.interval.end > match['t2'])):
                s = SIL_LABEL
                if phone_index > 0:
                    s = gold_dict[fname][phone_index-1].label
                e = SIL_LABEL
                if phone_index < len(gold_dict[fname])-1:
                    e = gold_dict[fname][phone_index+1].label
                triphone = [s, phone_interval.label, e]
                # triphone = [
                #     gold_dict[fname][phone_index-1].label,
                #     phone_interval.label,
                #     gold_dict[fname][phone_index+1].label]
                i_start = max(phone_index-10, 0)
                i_end = min(phone_index+11, len(gold_dict[fname]))
                word_left = [x.label for x in
                    gold_dict[fname][i_start:phone_index+1]]
                word_right = [x.label for x in
                    gold_dict[fname][phone_index:i_end]]
                res[indexes[i]].triphone2 = triphone
                res[indexes[i]].word2_left = word_left
                res[indexes[i]].word2_right = word_right
                break
            else:
                # transcription not started, first phone not encontered
                # let's increment our phone counter
                phone_index += 1
            if phone_index == len(gold_dict[fname]):
                print match
                raise
    return res


def count_phone_matches(labelled_matches):
    # counting matches:
    count = 0
    sils = 0
    for labelledMatch in labelled_matches:
        tri1, tri2 = labelledMatch.triphone1, labelledMatch.triphone2
        if tri1[1] == tri2[1]:  # and tri1[1] != 'SIL':
            if tri1[1] != 'SIL':
                count += 1
            else:
                sils += 1
    # return (count, float(count) / len(labelled_matches),)
    return count, len(labelled_matches) - sils


def count_matches(labelled_matches):
    # counting matches:
    count = 0
    count2 = 0
    for labelledMatch in labelled_matches:
        tri1, tri2 = labelledMatch.triphone1, labelledMatch.triphone2
        # if ((any(tri1[i] == tri2[i] and tri1[i] != 'SIL' for i in range(3)) or
        #      any(tri1[:-1][i] == tri2[1:][i] and tri1[i] != 'SIL' for i in range(2)) or
        #      any(tri1[1:][i] == tri2[:-1][i] and tri1[i] != 'SIL' for i in range(2)))):
        s = set(tri1).intersection(set(tri2)).difference([SIL_LABEL])
        if s:
            count += 1
        if len(s) >= 2:
            count2 += 1
        # ued = nlp.ued(tri1, tri2)
        # if ued <= 2:
        #     count += 1
        # if ued <= 1:
        #     count2 += 1
        #     print tri1, tri2            
    return (count, float(count) / len(labelled_matches),)


def minimal_achievable_ned(labelled_matches):
    man = []
    for match in labelled_matches:
        x1, y1 = match.word1_left[::-1], match.word2_left[::-1]
        x1, y1 = min_levenstein_distance(x1, y1)
        x2, y2 = match.word1_right, match.word2_right
        x2, y2 = min_levenstein_distance(x2, y2)
        assert np.all([x1, x2, y2, y1] > 1)
        word1 = match.word1_left[-x1:] + match.word1_right[1:x2]
        word2 = match.word2_left[-y1:] + match.word2_right[1:y2]
        man.append(float(levenstein_distance(word1, word2)) / max(x1 + x2 + 1, y1 + y2 + 1))
        # print(match.word1_left, match.word1_right)
        # print(match.word2_left, match.word2_right)
        # print(x1, x2)
        # print(y1, y2)
        # print(word1, word2)
        # print(levenstein_distance(word1, word2))
        # raise
    return np.array(man)

MIN_DISTANCE = 1


def min_levenstein_distance(l1, l2):
    symbols = list(set(l1+l2))
    symbol2ix = {v: k for k, v in enumerate(symbols)}
    l1_arr = np.fromiter((symbol2ix[s] for s in l1), dtype=np.uint32)
    l2_arr = np.fromiter((symbol2ix[s] for s in l2), dtype=np.uint32)
    return nb_min_levestein_distance(l1_arr, l2_arr)


def levenstein_distance(l1, l2):
    symbols = list(set(l1+l2))
    symbol2ix = {v: k for k, v in enumerate(symbols)}
    l1_arr = np.fromiter((symbol2ix[s] for s in l1), dtype=np.uint32)
    l2_arr = np.fromiter((symbol2ix[s] for s in l2), dtype=np.uint32)
    return nb_levestein_distance(l1_arr, l2_arr)[len(l1_arr), len(l2_arr)]

@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def nb_levestein_distance(l1, l2):
    n = len(l1)
    m = len(l2)
    print('{} {}'.format(n,m))
    d = np.empty((n+1, m+1), dtype=np.int32)
    d[:, 0] = np.arange(n+1)
    d[0, :] = np.arange(m+1)
    for i in range(1, n+1):
        for j in range(1, m+1):
            if l1[i-1] == l2[j-1]:
                d[i, j] = d[i-1, j-1]
            else:
                d[i, j] = min(d[i, j-1] + 1,
                              d[i-1, j] + 1,
                              d[i-1, j-1] + 1)
    return d

@nb.vectorize([nb.float64(nb.float64, nb.float64)])
def nb_min_levestein_distance(l1, l2):
    n = len(l1)
    m = len(l2)
    d = nb_levestein_distance(l1, l2)
    vi = MIN_DISTANCE
    vj = MIN_DISTANCE
    min_value = float(d[MIN_DISTANCE, MIN_DISTANCE]) / (MIN_DISTANCE)
    for i in range(MIN_DISTANCE, n+1):
        for j in range(MIN_DISTANCE, m+1):
            e = float(d[i, j]) / max(i, j)
            if e <= min_value:
                min_value = e
                vi = i
                vj = j
    return vi, vj


if __name__ == '__main__':
    gold = load_gold('new_english.phn')
    matches = load_matches('matchlist_1stpass_mfcc.raw')
    labelled_matches = find_match_triphones(gold, matches)
    # print count_matches(labelled_matches)
    # man = minimal_achievable_ned(labelled_matches)
    # import matplotlib.pyplot as plt
    # man = np.cumsum(np.sort(man)) / np.arange(1, len(man) +1)
    # plt.plot(np.arange(len(man)), man)
    # plt.show()
