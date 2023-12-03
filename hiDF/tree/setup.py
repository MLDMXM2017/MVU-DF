import numpy
from numpy.distutils.core import setup
from Cython.Build import cythonize

setup(
    name="_tree",
    ext_modules=cythonize("_tree.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    name="_criterion",
    ext_modules=cythonize("_criterion.pyx"),
    include_dirs=[numpy.get_include()]
)
 
setup(
    name="_splitter",
    ext_modules=cythonize("_splitter.pyx"),
    include_dirs=[numpy.get_include()]
)

setup(
    name="_utils",
    ext_modules=cythonize("_utils.pyx"),
    include_dirs=[numpy.get_include()]
)