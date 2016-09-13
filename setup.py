# from distutils.core import setup
from setuptools import setup, find_packages
from distutils.extension import Extension
from Cython.Build import cythonize
import numpy as np


setup(
    name = 'ZRTools',
    ext_modules = cythonize([
        Extension("plebdisc_interface.plebdisc_wrapper",
                  ["plebdisc_interface/plebdisc_wrapper.pyx", "ZRTools/plebdisc/util.c", "ZRTools/plebdisc/second_pass.c", "ZRTools/plebdisc/score_matches.c", "ZRTools/plebdisc/feat.c", "ZRTools/plebdisc/dot.c", "ZRTools/plebdisc/signature.c"],
                  include_dirs=[".", "plebdisc_interface", "ZRTools/plebdisc/", np.get_include()], extra_compile_args=["-std=c99"]),
        Extension("plebdisc_interface.py_secondpass",
                  ["plebdisc_interface/py_secondpass.pyx"],
                  include_dirs=[np.get_include()])]),
    packages = find_packages()
)
