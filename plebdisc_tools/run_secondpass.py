from os.path import join as pjoin
import os
import numpy as np
import py_secondpass
from numpy.lib.recfunctions import append_fields
import plebdisc
import multiprocessing
import ctypes


n_cpus = 12

matches_dir = 'mfcc_split_matches'
result_file = matches_dir
feats_dir = 'mfcc_split'

files_list = os.listdir(matches_dir)
files_dict = {}

for fname in files_list:
    try:
        files_dict[fname[:3]].append(fname)
    except KeyError:
        files_dict[fname[:3]] = [fname]
    # try:
    #     files_dict[fname[:3]].append(pjoin(matches_dir, fname))
    # except KeyError:
    #     files_dict[fname[:3]] = [pjoin(matches_dir, fname)]


def second_pass_worker(args):
    return second_pass(*args)


def second_pass(files, strat, thr, spk):
    with open('{}_{}_strat{}.txt'.format(result_file, spk, strat), 'w') as fout:
        for f in files:
            matchlist = read_matchlist(pjoin(matches_dir, f))
            matchlist = append_fields(matchlist, 'score', np.zeros((matchlist.shape[0],), dtype=np.float32))
            f1, f2 = f.split()
            feats1 = pjoin(feats_dir, f1 + '.fea')
            feats2 = pjoin(feats_dir, f2 + '.fea')
            feats1sig = pjoin(feats_dir, f1 + '.sig')
            feats2sig = pjoin(feats_dir, f2 + '.sig')
            feats1sig, feats2sig = read_feats(feats1sig, feats2sig, S=64)
            feats1, feats2 = map(lambda x: np.fromfile(x, dtype=np.float32).reshape((-1, 39)), [feats1, feats2])
            matchlist = py_secondpass.secondpass(matchlist, feats1sig, feats2sig, 10, thr, 0.25, strat)
            matchlist = py_secondpass.rescore_matchlist(matchlist, feats1, feats2)
            append_matches(matchlist, f1, f2, fout)


def read_feats(file1, file2, S):
    feats1 = np.fromfile(file1, dtype=ctypes.c_ubyte)
    feats1 = feats1.reshape((-1, S/8))
    feats2 = np.fromfile(file2, dtype=ctypes.c_ubyte)
    feats2 = feats2.reshape((-1, S/8))
    return feats1, feats2


def read_matchlist(matchlist_file):
    return np.fromfile(matchlist_file, dtype=[
        ('xA', 'i4'), ('xB', 'i4'),
        ('yA', 'i4'), ('yB', 'i4'),
        ('rhoampl', 'f4')
    ])


def append_matches(matchlist, f1, f2, fout, min_len=7):
    for m in matchlist:
        if ((m['xB'] - m['xA'] >= min_len and
             m['yB'] - m['yA'] >= min_len)):
            fout.write(' '.join(str(x) for x in [
                f1, f2,
                m['xA'], m['xB'], m['yA'], m['yB'],
                m['dtw'], m['length'], 0
                ]))
            fout.write('\n')


args = [[files, 1, 0.3, spk] for spk, files in files_dict.iteritems()]
pool = multiprocessing.Pool(n_cpus)
pool.map(second_pass_worker, args)
# second_pass(files_dict['s01'], 2, 0.3, 's01')
