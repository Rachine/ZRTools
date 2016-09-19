
=====================
Project organisation:
=====================

3 subprojects:

- ZRTools: original repo (with a few modifs). Contains the c code and scripts
- plebdisc_interface: python interface for ZRTools functions. Extension of a few functions
- plebdisc_tools: python scripts (wrapper for main function and several analysis tools)

ZRTools:
--------

Unsupervised spoken term discovery.


plebdisc_interface:
-------------------

Split plebdisc in 4 subfunction and expose those to python.
Numpy interface with those subfunctions.

- compute_similarity_matrix
- filter_matrix
- find_matches
- refine_matches

plebdisc_tools:
---------------

- main:
  - call lsh.main, plebdisc.main (h5features interface)
  - estimate "good" parameters
- io: load binary files in numpy arrays + a few operations on those.
- evaluate:
  - final step (ned) WIP
  - matches (precision, minimal achievable ned)
