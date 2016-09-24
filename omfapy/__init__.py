import logging
from logging import handlers
import numpy as np
from os import path
import sys
import warnings

# Local native imports
from model import Model
from fit import PIFit, GLSFit
from analysis import Analysis
import simulation

# Adding fortran libraries to path
sys.path.append(path.join(path.realpath(path.dirname(__file__)),
                          '../fortran/'))

# Local Fortran imports
from libomfa import realization

# Initializing random state generator
prng = np.random.RandomState()

# Default parameters (tuple ranges where values are generated randomly)
default_param = {'import':{'csv_delimiter':',',
                           'quote_delimiter':'"',
                           'reaction_delimiter':'->',
                           'species_delimiter':'+'},
                 'model':{'n_compounds':(10, 200),
                          'n_reactions':(10, 200),
                          'n_pools':(5, 200),
                          'n_branches':(1, 20),
                          'd_branching':(0.3, 0.7),
                          'p_density':(0.3, 0.7),
                          'p_density_increase':(0.3, 0.7),
                          'p_compound_distribution':(0.3, 0.7),
                          'p_reaction_distribution':(0.3, 0.7)},
                'fluxes':{'n_iterations':1000,
                          'n_sets':1,
                          'max_flux':100,
                          'selection_criteria':'abs_variance',
                          'selection_percentile':0.1}}
last_param = {}

# Defining logger generating function that each module can use
logger_param = {'file_path': path.join(path.dirname(path.realpath(__file__)), '../log', 'omfa.log'),
                'file_format':'[%(asctime)s] (%(module)s) - %(levelname)s - %(message)s',
                'console_format':'(%(module)s) - %(levelname)s - %(message)s',
                'file_level':logging.DEBUG,
                'console_level':logging.DEBUG,
                'max_bytes':10240,
                'backup_count':3}

def set_logger(param):

    logger = logging.getLogger('OMFA')
    logger.setLevel(logging.DEBUG)

    # Console feed
    ch = logging.StreamHandler()
    ch.setLevel(param['console_level'])
    ch_format = logging.Formatter(param['console_format'])
    ch.setFormatter(ch_format)
    logger.addHandler(ch)

    # File feed
    fh = handlers.RotatingFileHandler(param['file_path'], 
                                      maxBytes=param['max_bytes'], 
                                      backupCount=param['backup_count'])
    fh.setLevel(param['file_level'])
    fh_format = logging.Formatter(param['file_format'])
    fh.setFormatter(fh_format)
    logger.addHandler(fh)

    return(logger)

logger = set_logger(logger_param)

# Custom exceptions
#--------------------------------------------------------------------------
class ModelError(Exception):
    pass

class AnalysisError(Exception):
    pass

