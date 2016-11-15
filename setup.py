"""Setup script for the ZRTools package"""

import os
from setuptools import setup, find_packages
from setuptools.command.install import install
from distutils.extension import Extension
from subprocess import call

from Cython.Build import cythonize
import numpy as np

VERSION = '0.1'

# build ZRTools binary files
HERE = os.path.dirname(os.path.abspath(__file__))
bin_dir = os.path.join(HERE, 'plebdisc_tools')
zrtools_dir = os.path.join(HERE, 'ZRTools/plebdisc')
cmd_ = 'make DESTDIR={} --directory={} install'.format(bin_dir, zrtools_dir)
call([cmd_], shell=True)

setup(name = 'ZRTools',
      version=VERSION,
      description='Wrapper for ZRTools',
      url='https://github.com/bootphon/ZRTools',
      author='Bootphon Team',
      author_email='zerospeech2017@gmail.com',
      ext_modules = cythonize([
          Extension("plebdisc_interface.plebdisc_wrapper",
                    ["plebdisc_interface/plebdisc_wrapper.pyx",
                        "ZRTools/plebdisc/util.c", 
                        "ZRTools/plebdisc/second_pass.c", 
                        "ZRTools/plebdisc/score_matches.c", 
                        "ZRTools/plebdisc/feat.c", 
                        "ZRTools/plebdisc/dot.c", 
                        "ZRTools/plebdisc/signature.c"],
                    include_dirs=[".", 
                        "plebdisc_interface", 
                        "ZRTools/plebdisc/", 
                        np.get_include()], 
                    extra_compile_args=["-std=c99"]),
          Extension("plebdisc_interface.py_secondpass",
                    ["plebdisc_interface/py_secondpass.pyx"],
                    include_dirs=[np.get_include()])]),
      dependency_links=['https://github.com/bootphon/h5features'],
      install_requires=['numba>=0.29.0']
      include_package_data=True,
      packages = find_packages()
)


