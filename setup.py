#setup.py for py2exe
from distutils.core import setup
import py2exe
import matplotlib

setup(console=['r3x_reader.py'],
	data_files=matplotlib.get_py2exe_datafiles())