"""Wrapper for bash commands and a few tools for plebdisc"""

from .plebdisc import compute_percentile_param, fdict, launch_lsh, launch_plebdisc

from .precision_match import (launch_lsh, launch_job, launch_plebdisc, 
                             merge_results, do_cosine_similarity, 
                             do_norm_hamming_sim, read_sigs_remove_sils, 
                             estimate_recall, compute_percentile_param, 
                             estimate_similarities_distribution)


