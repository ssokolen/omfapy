from numpy.distutils.core import Extension, setup
from setuptools import find_packages

ext = Extension(name = 'libomfa',
                sources = ['./fortran/misc/algebra.f90',
                           './fortran/misc/random.f90',
                           './fortran/libomfa/realization.f90'],
                include_dirs = ['./fortran/libomfa'])

setup(
    name = 'omfapy',
    version = '0.1',
    packages = find_packages(),

    install_requires = ['numpy', 'scipy', 'pandas'],

    ext_modules = [ext],

    # Metadata
    author = 'Stanislav Sokolenko',
    author_email = 'stanislav@sokolenko.net',
    description = ('Python package for Metabolic Flux Analysis with a focus '
                   'on model and observation simulation'),
    license = 'Apache',
    keywords = 'MFA metabolic flux analysis',
)
