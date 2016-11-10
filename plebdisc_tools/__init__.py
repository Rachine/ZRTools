"""Wrapper for bash commands and a few tools for plebdisc"""

from .plebdisc import ( compute_percentile_param, fdict, 
                        launch_lsh, launch_plebdisc )

from .precision_match import ( silentremove, load_gold,
                               load_matches, find_match_triphones,
                               count_phone_matches, count_matches,
                               minimal_achievable_ned, min_levenstein_distance,
                               levenstein_distance, nb_levestein_distance,
                               nb_min_levestein_distance )





