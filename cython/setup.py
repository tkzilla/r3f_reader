#setup.py for cython
from distutils.core import setup
from Cython.Build import cythonize

setup(name='r3x_converter', ext_modules=cythonize('r3x_converter.pyx'))

