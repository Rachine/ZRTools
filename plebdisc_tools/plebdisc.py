#!/usr/bin/env python 

'''
plebdisc.py

DESCRIPTION
===========


ENVIRONMENTAL VARIABLES
=======================

ZRPATH : Path to the directory with ZRTOOLS binary directory
        to setset in bash:
            
            >> export ZRPATH={YOUR ZRPATH}

'''

from __future__ import print_function, division

from subprocess import Popen, PIPE
from itertools import izip, product
from collections import defaultdict

import multiprocessing
import numpy as np
import numba as nb
import argparse
import subprocess
import h5features
import tempfile
import os
import h5py
import re

from corpus import get_speaker


__all__ = ['launch_lsh', 'launch_job', 'check_call_stdout', 'get_speaker',
        'launch_plebdisc', 'merge_results', 'do_cosine_similarity', 
        'do_norm_hamming_sim', 'read_sigs_remove_sils', 'estimate_recall',
        'compute_percentile_param', 'estimate_similarities_distribution',
        'read_vad', 'write_vad_files' ] 


# regexs to decode the stdout from plebdisc
_reg_plebdisc = re.compile(r"    Dumping (\d+) matches\n((.*\n)+)", re.MULTILINE)  

# environmental variable thats is set to the installation of ZRTools binary package
try:
    binpath = os.environ['ZRPATH']
except:
    binpath = '../ZRTools/plebdisc'

assert os.path.isdir(binpath), 'ZRPATH not set or {} doesn\'t exit'.format(binpath) 
 

VALID_BITS = [32, 64]

class fdict(dict):
    def __init__(self, *args, **kwargs):
        super(fdict, self).__init__(*args, **kwargs)
        self.stats = {}


def launch_lsh(features_file, featsdir, S=64, files=None, with_vad=None,
               split=False):
    """Launch lsh for specified features, return a dictionnary containing for
    all files the path to their signature, features and vad.

    Parameters:
    ----------
    features_file: h5features file name
    featsdir: output folder where the features will be written
    S: int, number of bits of the lsh signature
    files: list, only launch lsh on the specified files.
        (must be the basename of the file, like in the h5features file)
    with_vad: optional VAD file
    
    """
   
    if S not in VALID_BITS:
        raise ValueError('S={} must be 32 or 64'.format(S))

    def aux(f, feats, S, D, featfile, sigfile, vadfile=None, vad=None):
        with open(featfile, 'wb') as fout:
            fout.write(feats.tobytes())
        
        command_ = '{}/lsh -S {} -D {} -projfile proj_b{}xd{}_seed1 -featfile {} -sigfile {}'
        command_ = command_.format(binpath, S, D, S, D, featfile, sigfile)
        if vadfile:
            with open(vadfile, 'w') as fout:
                for interval in vad[f]:
                    fout.write(' '.join(map(str, interval)) + '\n')
            command_ += ' -vadfile {}'.format(vadfile)
        p_ = Popen(command_, shell=True, stdout=PIPE, stderr=PIPE)
        output_, err_ = p_.communicate() 
        if 'usage' in err_:
            print(err_)
            raise NameError('Cannot run command: {}'.format(command_))

    vad = {}
    if with_vad:
        vad_ = defaultdict(list)
        with open(with_vad) as fin:
            for line in fin:
                fname, start, end = line.strip().split()
                start, end = map(lambda t: int(float(t) * 100), (start, end))
                vad_[fname].append((start, end))
        vad = dict(vad_)

    # generate a file with DxS normal random values [=numpy.random.norm(0.0,1.0,D*S)] 
    D = h5py.File(features_file)['features']['features'].shape[1]
    command_ = '{}/genproj -D {} -S {} -seed 1'.format(binpath, D, S)
    p_ = Popen(command_, shell=True, stdout=PIPE, stderr=PIPE)
    output_, err_ = p_.communicate()
    if 'unknown' in err_:
        print(err_)
        raise NameError('Cannot run command: {}'.format(command_))

    res = fdict()
    res.stats = {'S':S, 'D':D} 
    if not os.path.exists(featsdir):
        os.makedirs(featsdir)

    if files == None:
        files = h5features.read(features_file)[0].keys()
    
    for f in files:

        spk = get_speaker(f)

        if not split:
            sigfile = os.path.join(featsdir, f + ".sig")
            vadfile = os.path.join(featsdir, f + ".vad")
            featfile = os.path.join(featsdir, f + ".fea")
            feats = h5features.read(features_file, from_item=f)[1][f]
            
            if with_vad:
                aux(f, feats, S, D, featfile, sigfile, vadfile, vad)
            else:
                aux(f, feats, S, D, featfile, sigfile)
            
            try :
                res[spk][f] =  {'sig': sigfile, 'fea': featfile}
            except KeyError:
                res[spk] = {f: {'sig': sigfile, 'fea': featfile}}
            
            if with_vad:
                res[spk][f]['vad'] = vadfile
        else:
            intervals = vad[f]
            for i, (start, end) in enumerate(intervals):
                fi = '{}_{}'.format(f, i)
                sigfile = os.path.join(featsdir, fi + ".sig")
                vadfile = os.path.join(featsdir, fi + ".vad")
                featfile = os.path.join(featsdir, fi + ".fea")
                feats = h5features.read(
                    features_file, from_item=f,
                    from_time=float(start)/100, to_time=float(end)/100)[1][f]
                aux(fi, feats, S, D, featfile, sigfile)
                try:
                    res[spk][fi] = {'sig': sigfile, 'fea': featfile}
                except KeyError:
                    res[spk] = {fi: {'sig': sigfile, 'fea': featfile}}
    return res


def launch_job(commands, stdout, n_cpu):
    if n_cpu > 1:
        pool = multiprocessing.Pool(n_cpu)
        # manager = multiprocessing.Manager()
        # lock = manager.Lock()
        args = [(command, stdout) for command in commands]
        pool.map(check_call_stdout, args)


def check_call_stdout(command_stdout):
    command, stdout = command_stdout
    return subprocess.check_call(command, stdout=stdout)


def launch_plebdisc(files, output, within=True, P=4, B=100, T=0.5, D=10, S=64, medthr=0.5, dx=25, dy=3, rhothr=0, castthr=0.5, R=10, onepass=False, rescoring=True, dump_matchlist=None, dump_sparsematrix=None, dump_filteredmatrix=None):
    """Call plebdisc
    """

    if S not in VALID_BITS:                                    
        raise ValueError('S={} must be 32 or 64'.format(S))    

    if not within:
        raise NotImplementedError
    
    tmpfiles = {}
    tmpfiles_rescore = {}
    if dump_matchlist:
        try:
            os.makedirs(dump_matchlist)
        except OSError:
            pass

    try:
        for spk in files:
            n = len(files[spk]) ** 2
            for f1, f2 in product(files[spk], files[spk]):
                
                sigfile1, sigfile2 = files[spk][f1]['sig'], files[spk][f2]['sig']
                fout, matching_fname = tempfile.mkstemp(prefix='pldisc_match_')
                fout_rescore, tmpfile_rescore = tempfile.mkstemp(prefix='pldisc_')
                f1_f2 = '{} {}'.format(f1, f2)
                tmpfiles[f1_f2] = matching_fname
                tmpfiles_rescore[f1_f2] = tmpfile_rescore
                
                command_ = '{}/plebdisc -D {} -S {} -P {} -T {} -B {} -dtwscore 1 ' + \
                          '-dx {} -dy {} -medthr {} -twopass {} -R {} -castthr {} ' + \
                          ' -trimthr {} -rhothr {} -file1 {} -file2 {}' 
                command_ = command_.format(binpath, D, S, P, T, B,
                        dx, dy, medthr, int(not onepass), R, castthr,
                        castthr, rhothr, sigfile1, sigfile2) # TODO: castthr replace trimthr?
                if dump_matchlist:
                    command_ += ' -dump-matchlist {}/"{}"'.format(dump_matchlist, f1_f2)
                if dump_sparsematrix:
                    assert len(dump_sparsematrix) == 2
                    command_ += ' -dump-sparsematrix {} {}'.format(*dump_sparsematrix)
                if dump_filteredmatrix:
                    assert len(dump_filteredmatrix) == 2
                    command_ += ' -dump-filteredmatrix {} {}'.format(*dump_filteredmatrix)
                
                p_ = Popen(command_, shell=True, stdout=PIPE, stderr=PIPE)   
                output_, err_ = p_.communicate()                                         
                if 'usage' in err_ or 'Error' in err_:
                    print(err_)
                    raise NameError('Cannot run command: {}'.format(command_))           

                # matching_fname will contain the decodec output of plebdisc (only the mathes) 
                # information that is an input of restcore_dtw
                res_ = _reg_plebdisc.findall(output_)
                if not res_:
                    raise NameError('not results from plebdisc')
                res_ = list(res_[0])
                n_matches, matches = int(res_[0]), ''.join(res_[1]) # TODO: can it contain more 2 el
                matches = re.sub('\\n+','\\n', matches) # double CR produce errors in rescore_dtw
                assert n_matches == matches.count('\n'), 'could\'t decode plebdisc stdout'
                os.write(fout, matches)

                if rescoring:
                    feafile1, feafile2 = files[spk][f1]['fea'], files[spk][f2]['fea']
                    command_ = '{}/rescore_dtw -D {} -file1 {} -file2 {} -matchlist {}'
                    command_ = command_.format(binpath, files.stats['D'], feafile1, feafile2, 
                            matching_fname)
                    p_ = Popen(command_, shell=True, stdout=fout_rescore, stderr=PIPE)
                    output_, err_ = p_.communicate()
                    if 'usage' in err_ or 'Error' in err_: 
                        print(err_)  
                        raise NameError('Cannot run command: {}'.format(command_))    
                    #subprocess.check_call([command], shell=True, stdout=fout_rescore)

                os.close(fout)
                os.close(fout_rescore)

        if rescoring:
            merge_results(tmpfiles_rescore, output)
        else:
            merge_results(tmpfiles, output)

    finally:
        for f in tmpfiles.itervalues():
            pass
            tryremove(f)
        for f in tmpfiles_rescore.itervalues():
            pass
            tryremove(f)


def merge_results(filedict, output, layout='pairs'):
    """Merge individual file comparisons into one large file
    
    Different format are:


    by_file:
    file1 file2
    f1_start f1_end f2_start f2_end dtw_score aren_score 
    f1_start f1_end f2_start f2_end dtw_score aren_score 
    ...

    file1 file2
    ...


    pairs:
    file1 file2 f1_start f1_end f2_start f2_end dtw_score aren_score 
    file1 file2 f1_start f1_end f2_start f2_end dtw_score aren_score 
    ...
    """
    with open(output, 'w') as fout:
        for fname, pairs in filedict.iteritems():
            fname_bname = os.path.splitext(os.path.basename(fname))[0]
            if layout == 'by_file':
                fout.write('{}\n'.format(fname_bname))
                with open(pairs) as fin:
                    for line in fin:
                        fout.write(line)
                fout.write('\n')
            elif layout == 'pairs':
                with open(pairs) as fin:
                    for line in fin:
                        fout.write('{0} '.format(fname_bname))
                        fout.write(line)
    

def cosine(v1, v2):
    norm = np.linalg.norm
    return np.dot(v1, v2.T) / (norm(v1) * norm(v2))


@nb.autojit
def do_cosine_similarity(X1, X2):
    n_feats = X1.shape[0]
    similarities = np.empty((n_feats,))
    for i in range(n_feats):
        similarities[i] = cosine(X1[i], X2[i])
    return similarities


def do_norm_hamming_sim(X1, X2, S):
    n_feats = X1.shape[0]
    similarities = np.empty((n_feats,))
    vectgmpy = np.vectorize(lambda x: bin(x).count('1'))
    similarities = np.bitwise_xor(X1, X2)
    return np.cos(vectgmpy(similarities).astype(float) / 64 * np.pi)


def read_sigs_remove_sils(files):
    if files.stats['S'] == 64:
        dtype = np.uint64
    elif files.stats['S'] == 32:
        dtype = np.uint32
    else:
        raise NotImplementedError
    features_dict = {}
    for fname, fs in files.iteritems():
        feats = [np.fromfile(paths['sig'], dtype=dtype) for paths in fs.itervalues()]
        feats = np.concatenate(feats, axis=0)
        features_dict[fname] = feats
    # features_dict = {fname: np.fromfile(paths['sig'], dtype=dtype)
    #                  for files in files.itervalues() for fname, paths in files.iteritems()}
    for fname, feats in features_dict.iteritems():
        features_dict[fname] = feats[feats > 0]
    return features_dict


def estimate_recall(width, files, threshold, permuts=1,
                    comparison='within', samples=100000):
    """Estimate the percentage of signatures above a threshold found
    considering a certain width
    """
    if comparison != 'within':
        raise NotImplementedError
    
    features_dict = read_sigs_remove_sils(files)
    nbytes = files.stats['S'] // 8
    samples_per_file = samples // len(features_dict)
    res = np.empty((len(features_dict)))
    for i, feats in enumerate(features_dict.itervalues()):
        samples1 = np.random.choice(feats.shape[0], samples_per_file)
        samples2 = np.random.choice(feats.shape[0], samples_per_file)
        similarities = do_norm_hamming_sim(
            feats[samples1], feats[samples2], files.stats['S'])
        similar_mask = similarities > threshold
        similar_samples1 = samples1[similar_mask]
        similar_samples2 = samples2[similar_mask]

        r = np.empty((permuts, np.sum(similar_mask)), dtype=bool)
        for p in range(permuts):
            sorted_idx = np.argsort(feats)
            reverse_sort = np.argsort(sorted_idx)
            sorted_samples1 = reverse_sort[similar_samples1]
            sorted_samples2 = reverse_sort[similar_samples2]
            r[p, :] = np.abs(sorted_samples1 - sorted_samples2) < width
            new_order = np.random.permutation(np.arange(nbytes))
            splitted_feats = feats.view(dtype='({},)u1'.format(nbytes))
            splitted_feats = splitted_feats[:, new_order].flatten()
            feats = splitted_feats.view(dtype='u{}'.format(nbytes)).reshape((feats.shape[0],))
        r = np.logical_or.reduce(r)
        res[i] = np.mean(r)
    return np.mean(res)


def compute_percentile_param(percentile, features_file, comparison='within',
                             samples=100000, with_vad=False, lsh=False):
    """Compute the threshold parameter from a percentile value

    percentile: int or list of int
    """
    assert comparison in set(['general', 'within', 'across'])
    if comparison != 'within':
        raise NotImplementedError
    if not lsh:
        if not with_vad:
            features_dict = h5features.read(features_file)[1]
        else:
            features_dict = {}
            vad = read_vad(with_vad)
            for f, intervals in vad.iteritems():
                feats = [h5features.read(features_file, from_item=f, from_time=interval[0], to_time=interval[1])[1][f] for interval in intervals]
                features_dict[f] = np.concatenate(feats, axis=0)
    else:
        features_dict = read_sigs_remove_sils(features_file)
        # else:
        #     features_dict = {}
        #     vad = read_vad(with_vad)
        #     for f, intervals in vad.iteritems():
        #         feats = []
        #         for interval in intervals:
        #             interval = (np.array(interval) * 100).astype(int)
        #             feats.append(np.fromfile(features_file[f]['sig'], dtype=dtype)[interval[0]:interval[1]])
        #         features_dict[f] = np.concatenate(feats, axis=0)

    samples_per_file = samples // len(features_dict)

    def estimate_similarities_distribution(feats1, feats2, n_samples, S=False):
        samples1 = np.random.choice(feats1.shape[0], n_samples)
        samples2 = np.random.choice(feats2.shape[0], n_samples)
        sampled_feats1 = feats1[samples1]
        sampled_feats2 = feats2[samples2]
        if not S:
            fun = do_cosine_similarity
        else:
            fun = lambda x, y: do_norm_hamming_sim(x, y, S)
        return fun(sampled_feats1, sampled_feats2)

    similarities = []
    for feats in features_dict.itervalues():
        S = False
        if lsh:
            S = features_file.stats['S']
        similarities.append(estimate_similarities_distribution(
            feats, feats, samples_per_file, S=S))

    similarities = np.concatenate(similarities)
    return np.percentile(similarities, percentile)


def read_vad(vadfile):
    # vad_arr = np.loadtxt(vadfile)
    # files, indexes, counts = np.unique(vad_arr, return_index=True, return_counts=True)
    # vad = {f: vad_arr[i:i+c] for f, i, c in zip(files, indexes, counts)}
    vad = {}
    with open(vadfile) as fin:
        for line in fin:
            splitted = line.strip().split()
            assert len(splitted) == 3
            if splitted[0] not in vad:
                vad[splitted[0]] = []
            interval = map(float, splitted[1:])
            assert interval[1] > interval[0]
            vad[splitted[0]].append(interval)
    return vad


def write_vad_files(vad, files, outputfolder):
    if files == None:
        files = vad.keys()
    for f in files:
        with open(os.path.join(outputfolder, '{}.vad'.format(f)), 'w') as fout:
            for interval in vad[f]:
                fout.write(' '.join(map(
                    lambda x: str(int(x*100+1)), interval)) + '\n')


def tryremove(f):
    try:
        os.remove(f)
    except:
        pass


def _restricted_float(x):
    '''_restricted_float(x) 
     
    Parameters
    ----------
    x: float

    Check if a values is between 0. -> 100.0 raise argparse.ArgumentTypeError if out of limits
    '''
    x = float(x)
    if x < 0.0 or x > 100.0:
        raise argparse.ArgumentTypeError("%r not in range [0.0, 100.0]"%(x,))
    
    return x


def main():
    ''' Example:
    >> python plebdisc.py -q 80 -files 's0101a_0' 's0101a_4' -S 32 -D 39 mfcc_split.old output
    '''

    parser = argparse.ArgumentParser(description='Run plebdisc ')
    parser.add_argument("--verbose", help="increase output verbosity",
                         action="store_true")

    # ...General...
    parser.add_argument('features_file', help='features file in the h5f')
    parser.add_argument('output', help='where the results will be stored')
    parser.add_argument('-files', nargs='+', required=False, help='only compare this files')
    parser.add_argument('-S', default=64, type=int, help='size of the signatures (in bits)')
    parser.add_argument('-maxframes', default=None, type=int)
    parser.add_argument('-within', default='within')
    parser.add_argument('-vad', nargs=1, required=False, help='vad file')

    # ...Similarity matrix...
    parser.add_argument('-P', default=4, type=int, help='number of permutations')
    parser.add_argument('-B', default=100, type=int, help='width of the band')
    parser.add_argument('-q', default=90, type=_restricted_float, 
            help=('percentile threshold, frames above that percentile ',
                  'will be considered similars'))
    parser.add_argument('-D', default=10, type=int, 
            help=('length of diagonal search when point found'))

    # ... Matches search ...
    parser.add_argument('-medthr', default=0.5, type=float, 
            help=('minimal proportion of matches in the smoothing',
                  'window for the point to be considered a match'))
    parser.add_argument('-dx', default=25, type=int, 
            help=('size of the smoothing window along ',
			      'time axis (diagonal), and double of the',
			      ' minimal length of a match'))
    parser.add_argument('-dy', default=3, type=int, 
            help='size of the gaussian window (anti-diagonal)')
    parser.add_argument('-rhothr', default=0, type=int, 
            help=('threshold on the number of points in the diagonal',
                  'for it to be investigated'))
    parser.add_argument('-matchlist', type=str, required=False, 
            help='store the match list in that file')
    parser.add_argument('-onepass', default=False, action='store_true', 
            help=('stop after computing matchlist (do not do S-DTW)'))

    # ...S-DTW...
    parser.add_argument('-R', default=10, type=int, help='dtw band width')
    parser.add_argument('-qdtw', default=90, type=_restricted_float, 
            help=('percentile threshold, frames above that percentile',
                  ' will be considered similars'))
    parser.add_argument('-alpha', default=0.5, type=float, 
            help='exponential smoothing parameter')
    parser.add_argument('-Tscore', default=0.75, type=float, help='TODO')
    
    args = parser.parse_args() # TODO: gives a TypeError if using -h  

    features_file = args.features_file
    
    T, castthr = compute_percentile_param(percentile=(args.q, args.qdtw), 
            features_file=features_file, comparison=args.within)

    try:
        featsdir = tempfile.mkdtemp(prefix='pldisc_')
        with_vad = False
        if args.vad != None:
            with_vad = args.vad[0]
            # write_vad_files(with_vad, files, featsdir)

        files = launch_lsh(features_file, featsdir, S=args.S, files=args.files, 
                           with_vad=with_vad, split=False)
        
        launch_plebdisc(files, args.output, args.within, args.P, args.B, T, args.D, 
			            args.S, args.medthr, args.dx, args.dy, args.rhothr, castthr,
            		    args.R, args.onepass)

    finally:
        tryremove(featsdir)


if __name__ == '__main__':
    main()


