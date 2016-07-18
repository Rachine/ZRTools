from os.path import join as pjoin
import os
import multiprocessing
import sys
import precision_match as pm
import argparse


def run(files, matches_dir):
    matches = []
    for f in files:
        f1, f2 = f.split()
        aux = pm.load_matches(pjoin(matches_dir, f))
        aux['f1'] = f1
        aux['f2'] = f2
        if len(aux) > 0:
	    matches.append(aux)

    corrects = 0
    total = 0
    for m in matches:
        labelled_matches = pm.find_match_triphones(gold, m)
        a1, a2 = pm.count_phone_matches(labelled_matches)
        corrects += a1
        total += a2
    return corrects, total


def run_wrapper(args):
    return run(*args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'matches_dir', metavar='MATCHES_FOLDER', nargs=1,
        help='folder containing the match files')
    parser.add_argument(
        'gold', metavar='GOLD_ANNOTATIONS', nargs=1,
        help='gold annotation in .phn format (see tde)')
    parser.add_argument(
        '-j', metavar='N_CORES', type=int, default=1,
        help='number of cores to use')


if __name__ == "__main__":
    args = parse_args()
    n_cpus = args['j']
    matches_dir = args['matches_dir']

    gold = pm.load_gold(args['gold'])

    n = n_cpus
    files_list = os.listdir(matches_dir)
    n_files = len(files_list)
    args = [(files_list[i:min(i+n, n_files)], matches_dir)
            for i in range(0, n_files, n)]

    # args = list(itertools.product(files_list, percentiles))
    pool = multiprocessing.Pool(n_cpus)
    res = pool.map(run_wrapper, args)

    corrects = sum(r[0] for r in res)
    total = sum(r[1] for r in res)
    print float(corrects) / total, total
