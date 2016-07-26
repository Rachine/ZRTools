from distutils.core import setup
from Cython.Build import cythonize

setup(
    ext_modules = cythonize("plebdisc_interface/py_secondpass.pyx")
)
