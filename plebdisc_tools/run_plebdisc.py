from os.path import join as pjoin
import os
import multiprocessing
import plebdisc as p
import argparse
import h5py


def prepare(feats_file, feats_dir, vad_file=None):
    D = h5py.File(feats_file)['features']['features'].shape[1]  # TODO: replace by h5features function
    all_files = p.launch_lsh(feats_file, feats_dir, 64, with_vad=vad_file)
    files_list = []
    for spk, files in all_files.iteritems():
        aux = p.fdict()
        aux.stats = {'S': 64, 'D': D}
        aux[spk] = files
        files_list.append(aux)
    return files_list


def run(files, percentile, vad_file, matches_dir):
    T = p.compute_percentile_param(percentile, files, 'within',
                                   with_vad=vad_file, lsh=True)
    # TODO B=10*duration (or maybe 5*duration)
    p.launch_plebdisc(files, '/dev/null', within=True,
                      B=50, T=T, castthr=T, dump_matchlist=matches_dir)    


def run_wrapper(args):
    return run(*args)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        'featsfile', metavar='FEATURES_FILE',
        help='features file in h5features format')
    parser.add_argument(
        'percentile', metavar='PERCENTILE', type=int,
        help='percentile to use for threshold')
    parser.add_argument(
        'matches_dir', metavar='MATCHES_FOLDER', nargs='?',
        help='optional: folder where to store the matches')
    parser.add_argument(
        '--vadfile', metavar='VAD_FILE', nargs='?',
        help='voice activity detection file (see tde for format)')
    parser.add_argument(
        '-j', metavar='N_CORES', type=int, default=1,
        help='number of cores to use')

    return vars(parser.parse_args())


if __name__ == '__main__':
    args = parse_args()
    n_cpus = args['j']
    percentile = args['percentile']

    feats_file = args['featsfile']
    feats_dir = os.path.splitext(feats_file)[0]
    try:
        matches_dir = args['matches_dir']
    except KeyError:
        matches_dir = pjoin(feats_dir, '_matches')
    try:
        vad_file = args['vadfile']
    except KeyError:
        vad_file = None

    args = prepare(feats_file, feats_dir, vad_file=vad_file)
    args = [(a, percentile, vad_file, matches_dir) for a in args]
    pool = multiprocessing.Pool(n_cpus)
    pool.map(run_wrapper, args)
