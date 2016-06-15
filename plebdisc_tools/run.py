from os.path import join as pjoin
import os
import numpy as np
import plebdisc
import multiprocessing
import h5features
import h5py
import plebdisc as p


n_cpus = 12

feats_file = 'mfcc_split.h5f'
vad_file = 'english_split_vad.txt'
feats_dir = os.path.splitext(feats_file)[0]
matches_dir = feats_dir + '_matches'
percentile = 90


all_files = p.launch_lsh(feats_file, feats_dir, 64, with_vad=vad_file)
files_list = []
for spk, files in all_files.iteritems():
    aux = p.fdict()
    aux.stats = {'S': 64, 'D': 39}
    aux[spk] = files
    files_list.append(aux)


def run(files):
    T = p.compute_percentile_param(percentile, files, 'within',
                                   with_vad=vad_file, lsh=True)
    p.launch_plebdisc(files, '/dev/null', within=True,
                      B=50, T=T, castthr=T, dump_matchlist=matches_dir)
    


args = files_list
pool = multiprocessing.Pool(n_cpus)
pool.map(run, args)
