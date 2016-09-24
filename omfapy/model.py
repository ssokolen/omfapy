"""
Definition of omfa.Model class
"""

import copy
from ctypes import POINTER, c_int, c_float
from cvxopt import solvers, blas, matrix
from cvxopt.modeling import variable, op
import hashlib
import itertools as it
from lxml import etree
import math
import numpy as np
import pandas as pd
import re
from scipy.linalg import qr
from scipy.stats import norm
import string
import warnings

# Local imports
import omfapy as omfa
from omfapy.sampling import chebyshev_center, analytic_center, \
                            feasible_point, check_feasibility, variable_range

# The model class itself
#--------------------------------------------------------------------------
class Model():

    # Attributes
    stoichiometry = None
    _hash = None

    _nullspace = None
    
    #--------------------------------------------------------------------
    def __init__(self, stoichiometry):
        """
        Initializes model directly as a wrapper around a 
        pandas.DataFrame, containing the stoichiometry matrix. 

        Parameters
        ----------
        stoichiometry: pandas.DataFrame
        """
        
        # Performing standard pandas.DataFrame initialization
        self.stoichiometry = stoichiometry 

    #====================================================================>
    # Properties
    #   - functions necessary to keep the model nullspace updated with
    #     changes to stoichiometry
    
    #--------------------------------------------------------------------
    @property
    def nullspace(self):
        """
        Returns nullspace of stoichiometry matrix.

        Details
        -------
        Rather than calculating the nullspace each time, the stoichiometry
        matrix is hashed and the nullspace updated only if needed. This
        may actually impose a performance penalty, but ensures that the
        calculated nullspace remains identical for a given stoichiometry.

        Returns
        -------
        nullspace: pandas.DataFrame
        """

        stoich = self.stoichiometry

        # Checking hash
        new_hash = hashlib.md5(stoich.to_string()).digest()

        if new_hash != self._hash:
            self._hash = new_hash

            # Calculating nullspace
            U, s, V = np.linalg.svd(stoich, full_matrices=True)

            base_singular = stoich.shape[1] - stoich.shape[0]
            extra_singular = len([v for v in s if v < 1e-20])
            n_singular = base_singular + extra_singular

            kernel = V[-n_singular:,:].T
            kernel = pd.DataFrame(kernel, index=stoich.columns)

            names = ['B{0}'.format(i) for i in range(1, kernel.shape[1] + 1)]
            kernel.columns = names 

            self._nullspace = kernel

        return(self._nullspace)

    #==========================================================================>
    # Constructor methods
    #   - initialize the Model class from a number of different inputs
    
    #--------------------------------------------------------------------------
    @classmethod
    def from_generator(cls, run_checks=True, seed=None, **param):
        """
        Generates stoichiometric model suitable for OMFA.

        run_checks (bool): By default, param are tested to ensure
            consistency, with missing values generated. A bit of extra speed can
            be gained by turning all these checks off. It is recommended to do
            a trial run first with the checks turned on.

        seed (int): By default, a random number generator is initialized at
            omfa import. This generator can be re-initialized for the generation
            of a specific model with a specific seed.

        **param: Parameters for model generation. Unspecified parameters
            are generated from limits in omfa.default_param['model']

            n_compounds (int): The number of compounds.
            n_reactions (int): The number of intracellular reactions.
            n_pools (int): The number of transport fluxes.
            n_branches (int): The number of branches from the main biosynthethesis
                pathway trunk (defines the number of independent fluxes that must be
                specified).
            d_branching (real, list): A list of fractions representing the
                probability of finding 1, 2, 3, etc. branches stemming from the
                same node e.g. [0.25, 0.25, 0.50] translates into 25% of branches
                stemming from biosynthesis trunk, 25% of branches stemming from
                a node where two branches stem from the trunk and 50% of branches 
                stemming from a node where three branches stem from trunk.
            p_density (real): The fraction of reaction stoichiometry matrix that is
                non-zero (how connected the metabolites are).
            p_density_increase (real): A fraction specifying the relative number of
                reactions added to single compounds. A higher value quickly
                increases the connectivity of single compounds.
            p_compound_distribution (real): A fraction specifying the distribution
                of compound connectivity. Low values result in the uniform 
                distribution of reactions per compound. High values result in a
                small subset of compounds being involved in many reactions as
                compared to the rest.
            p_reaction_distribution (real): A fraction specifying the distribution
                of reaction connectivity. Low values result in the uniform 
                distribution of compounds across reactions. High values result in a
                small subset of reactions involving more compounds as compared
                to the rest.

        """

        # Setting random number generator
        if seed is None:
            prng = omfa.prng
        else:
            prng = np.random.RandomState(seed)

        # Generating random values for missing parameters
        #------------------------------------------------

        if run_checks:

            if omfa.libomfa is None:
                msg = 'Fortran library required for model generation'
                raise Exception(msg)

            # n_compounds
            if 'n_compounds' not in param:
                minval = [omfa.default_param['model']['n_compounds'][0]]
                maxval = [omfa.default_param['model']['n_compounds'][1]]

                try:
                    n_reactions = param['n_reactions']
                    minval.append(n_reactions + 1)
                except KeyError:
                    pass

                try:
                    n_pools = param['n_pools']
                    n_reactions = param['n_creations']
                    n_branches = param['n_branches']
                    maxval.append(n_pools + n_reactions - n_branches - 1)
                except KeyError:
                    pass

                low = max(minval)
                high = min(maxval)
                param['n_compounds'] = prng.randint(low, high)
            
            # n_reactions
            if 'n_reactions' not in param:
                minval = [omfa.default_param['model']['n_reactions'][0]]
                maxval = [omfa.default_param['model']['n_reactions'][1]]

                n_compounds = param['n_compounds']
                maxval.append(n_compounds - 1)

                try:
                    n_pools = param['n_pools']
                    n_branches = param['n_branches']
                    maxval.append(n_compounds + n_branches - n_pools - 1)
                except KeyError:
                    pass

                try:
                    n_branches = param['n_branches']
                    minval.append(n_branches * 2 + 1)
                except KeyError:
                    pass
                
                low = max(minval)
                high = min(maxval)
                param['n_reactions'] = prng.randint(low, high)

            # n_branches
            if 'n_branches' not in param:
                minval = [omfa.default_param['model']['n_branches'][0]]
                maxval = [omfa.default_param['model']['n_branches'][1]]

                n_reactions = param['n_reactions']
                maxval.append(math.floor((n_reactions - 1)/2))

                try:
                    n_pools = param['n_pools']
                    n_compounds = param['n_compounds']
                    maxval.append(n_pools - n_compounds + n_reactions - 1)
                except KeyError:
                    pass
                
                low = max(minval)
                high = min(maxval)
                if abs(low - high) < 1:
                    param['n_branches'] = 1
                else:
                    param['n_branches'] = prng.randint(low, high)

            # n_pools
            if 'n_pools' not in param:
                minval = [omfa.default_param['model']['n_pools'][0]]
                maxval = [omfa.default_param['model']['n_pools'][1]]

                n_compounds = param['n_compounds']
                n_reactions = param['n_reactions']
                n_branches = param['n_branches']
                minval.append(n_compounds - n_reactions + n_branches + 1)
                maxval.append(n_compounds)

                low = max(minval)
                high = min(maxval)
                param['n_pools'] = prng.randint(low, high)

            # p_density
            if 'p_density' not in param:
                minval = [omfa.default_param['model']['p_density'][0]]
                maxval = [omfa.default_param['model']['p_density'][1]]

                n_compounds = param['n_compounds']
                n_reactions = param['n_reactions']
                minval.append(float(n_reactions + n_compounds - 1)/(n_reactions * n_compounds))
                
                low = max(minval)
                high = min(maxval)
                param['p_density'] = prng.uniform(low, high)
            
            # d_branching
            if 'd_branching' not in param:
                low = omfa.default_param['model']['d_branching'][0]
                high = omfa.default_param['model']['d_branching'][1]
                param['d_branching'] = [prng.uniform(low, high)]
            
            # p_density_increase, p_compound_distribution, p_reaction_distribution
            density_param = ['p_density_increase', 
                                  'p_compound_distribution', 
                                  'p_reaction_distribution']
            for p in density_param:
                if p not in param:
                    low = omfa.default_param['model'][p][0]
                    high = omfa.default_param['model'][p][1]

                    param[p] = prng.uniform(low, high)

            # Wrong options
            valid_param = [p for p in omfa.default_param['model']]
            invalid_param = [p for p in param if p not in valid_param]

            if len(invalid_param) > 0:
                msg = ', '.join(invalid_param) + ' could not be recognized.\n'
                sol = 'The following are valid param: ' + ', '.join(valid_param)
                raise Exception(msg + sol)

            # Performing sanity checks
            #-------------------------

            n_compounds = param['n_compounds']
            n_reactions = param['n_reactions']
            n_pools = param['n_pools']
            n_branches = param['n_branches']
            d_branching = param['d_branching']
            p_density = param['p_density']

            if n_reactions >= n_compounds:
                msg = 'Model generation defined for overdetermined systems only\n'
                sol = 'n_compounds > n_reactions\n'
                raise Exception(msg + sol)

            if n_reactions < (n_branches)*2:
                msg = 'Depending on branching distribution, there may not be enough reactions\n'
                warnings.warn(msg)

            if n_pools <= n_compounds - n_reactions + n_branches:
                msg = 'Not enough transport fluxes\n'
                sol = 'n_pools > n_compounds - n_reactions + n_branches'
                raise Exception(msg + sol)

            if n_pools > n_compounds:
                msg = 'More transport fluxes than compounds\n'
                sol = 'n_pools <= n_compounds'
                raise Exception(msg + sol)

            for p in d_branching:
                if p < 0 or p > 1:
                    msg = 'Elements of d_branching must be fractions\n'
                    sol = '0 > d_branching[i] < 1'
                    raise Exception(msg + sol)

            if sum(d_branching) > 1:
                msg = 'The sum of proabability values in d_branching must be less than or equal to 1'
                raise Exception(msg)
            elif sum(d_branching) < 1:
                d_branching.append(1 - sum(d_branching))
                param['d_branching'] = d_branching
            
            min_density = float(n_reactions + n_compounds - 1)/(n_reactions * n_compounds)
            if p_density < min_density:
                msg = 'Specified density not large enough for minimum connectivity\n'
                sol = '(p_density > ' + str(min_density) + ')'
                raise Exception(msg + sol)

            density_param = ['p_density', 'p_density_increase', 
                                  'p_compound_distribution', 'p_reaction_distribution'] 
            for p in density_param:
                if param[p] < 0 or param[p] > 1:
                    msg = p + 'must be a fraction\n'
                    sol = '0 > ' + p + ' < 1'
                    raise Exception(msg + sol)

            invalid = [p for p in param if p not in omfa.default_param['model']]
            valid = [p for p in omfa.default_param['model']]

            if len(invalid) > 0:
                msg = ', '.join(invalid) + ' are invalid parameters\n'
                sol = 'Only the following param can be set: ' + ', '.join(valid)
                raise Exception(msg + sol)

        # Modifying parameters for use
        #-----------------------------

        n_branches = param['n_branches']
        d_branching = param['d_branching']
        d_branching = prng.multinomial(n_branches, d_branching)
        d_branching = np.array(d_branching, dtype=np.int32)
        nd_branching = d_branching.size

        n_compounds = param['n_compounds']
        n_reactions = param['n_reactions']
        p_density = param['p_density']
        n_density = p_density * n_compounds * n_reactions

        n_pools = param['n_pools']
        n_branches = param['n_branches']
        p_basic = np.array([n_compounds, n_reactions, n_pools, n_branches, n_density], 
                           dtype=np.int32)

        p1 = param['p_density_increase']
        p2 = param['p_compound_distribution']
        p3 = param['p_reaction_distribution']
        p_density = np.array([p1, p2, p3], dtype=np.float32)

        reaction_matrix = np.zeros((n_compounds, n_reactions), dtype=np.int32)
        transport_compounds = np.zeros(n_compounds, dtype=np.int32)
        
        # Generating
        #-----------

        seed = prng.randint(10e8)

        omfa.libomfa.generateModel(p_basic.ctypes.data_as(POINTER(c_int)),
                                   p_density.ctypes.data_as(POINTER(c_float)),
                                   d_branching.ctypes.data_as(POINTER(c_int)),
                                   c_int(nd_branching),
                                   reaction_matrix.ctypes.data_as(POINTER(c_int)),
                                   transport_compounds.ctypes.data_as(POINTER(c_int)),
                                   c_int(seed))

        # Unpacking transport fluxes into matrix form
        shape = (n_compounds, sum(transport_compounds != 0))
        transport_matrix = np.zeros(shape, dtype=np.int32)

        count = 0
        for i in range(n_compounds):
            if transport_compounds[i] != 0:
                transport_matrix[i, count] = transport_compounds[i]
                count += 1

        # Combining
        model = np.hstack((reaction_matrix, transport_matrix))
        
        # Generating compound names
        upper = string.uppercase
        name_pool = [l for l in upper]

        repeat = 1
        while len(name_pool) < n_compounds:
            name_pool = [''.join(l) for l in it.product(upper, repeat=repeat)]
            repeat += 1

        compound_names = name_pool[:n_compounds]

        # Internal reaction names
        reaction_names = ['R' + str(i + 1) for i in range(n_reactions)]

        # External reaction names
        transport_names = ['T' + str(i + 1) for i in range(n_pools)]

        # Converting to DataFrame
        model_df = pd.DataFrame(model, dtype=np.float64, index=compound_names, 
                                columns=reaction_names + transport_names)

        # Storing last parameters for analysis
        omfa.last_param = copy.deepcopy(param)

        return(cls(model_df))

    #---------------------------------------------------------------------------
    @classmethod
    def from_csv(cls, path, **param):
        """
        Reads OMFA from csv.

        path (str): Path to file

        **param: Parameters for how to interpret csv data. Unspecified parameters
            are generated from limits in omfa.default_param['import']

            csv_delimiter (str): csv entry delimiter, typically a comma
            quote_delimiter (str): csv quote delimiter, typically a quotation mark
            reaction_delimiter (str): separates reactants from products, 
                typically a right arrow
            species_delimiter (str): separates the reaction species,
                typically a plus sign

        """

        # Unpacking parameters
        #---------------------

        for p in omfa.default_param['import']:
            if p not in param:
                param[p] = omfa.default_param['import'][p]

        csv_delim = param['csv_delimiter']
        quote_delim = param['quote_delimiter']
        reaction_delim = param['reaction_delimiter']
        species_delim = param['species_delimiter']

        # Reading file
        #-------------
        
        f = open(path)
        
        with f:
            lines = [i.strip() for i in f.read().split('\n')]


        for i in xrange(len(lines) - 1, -1, -1):
            if reaction_delim not in lines[i]:
                # Removing any lines without a reaction delimiter
                lines.pop(i)
            else:
                # Splitting remaining lines into reactants and products
                lines[i] = [j for j in lines[i].split(reaction_delim)]
                lines[i] = [j.strip() for j in lines[i]]

        # Generating temporary names for the reactions
        names = ['F' + str(i) for i in xrange(len(lines))]

        # Initializing stoichiometry dictionary and name index
        stoichiometry = {}

        # Looping through the reaction
        for i in xrange(len(lines)):
            
            reactants = lines[i][0].split(csv_delim)
            reactants = [j.strip(quote_delim) for j in reactants]

            products = lines[i][1].split(csv_delim)
            products = [j.strip(quote_delim) for j in products]

            # Identifying reaction name
            if len(reactants) > 1:
                if (reactants[1] != species_delim) and (reactants[0] != ''):
                    name = reactants.pop(0).strip()
                else:
                    name = names.pop(0)
            else:
                name = names.pop(0)

            stoichiometry[name] = {}

            reactants = [j.strip() for j in reactants if len(j.strip()) > 0]
            products = [j.strip() for j in products if len(j.strip()) > 0]

            # Removing species delimiters
            reactants = [j for j in reactants if j != species_delim]
            products = [j for j in products if j != species_delim]

            # Converting entries into stoichiometry/compound pairs
            match = '(?<=[0-9])[^.^-^\d^\w](?=[^\d^\s])'
            for i in xrange(len(reactants)):
                entry = re.split(match, reactants[i])
                
                if len(entry) == 1:
                    stoichiometry[name][entry[0]] = -1 
                else:
                    stoichiometry[name][entry[1]] = -float(entry[0]) 

            for i in xrange(len(products)):
                entry = re.split(match, products[i])
                
                if len(entry) == 1:
                    stoichiometry[name][entry[0]] = 1 
                else:
                    stoichiometry[name][entry[1]] = float(entry[0]) 

        # Converting to data frame
        stoichiometry = pd.DataFrame(stoichiometry)
        stoichiometry.fillna(0, inplace=True)

        return(cls(stoichiometry))

    #==========================================================================>
    # Validation functions 
    #   - asses Model suitability for MFA

    #---------------------------------------------------------------------------
    def validate(self, transport=[]):
        """
        Assesses the validity of the model for MFA, checking for dead_end
        compounds, redundant relations, and dangling reactions.
        """

        stoich = self.stoichiometry

        omfa.logger.info('Validating model')

        # Finding issues
        dead_end = self.find_dead_end_compounds()
        redundant = self.find_redundant_compounds()
        dangling = self.find_dangling_reactions(transport=transport)

        # Combining
        problem_compounds = list(set(dead_end) | set(redundant))
        problem_reactions = dangling

        return(problem_compounds, problem_reactions)

    #---------------------------------------------------------------------------
    def find_dead_end_compounds(self):
        """
        Removes dead-end compounds i.e. those that are only produced or
        consumed by one reaction
        """   

        stoich = self.stoichiometry

        dead_end = [i for i in stoich.index if sum(stoich.ix[i, stoich.columns] != 0) <= 1]

        msg = str(len(dead_end)) + ' compounds identified as dead-end'
        omfa.logger.info(msg)
        for i in range(len(dead_end)):
            omfa.logger.info('\t' + str(i+1) + ' ' + dead_end[i])

        return(dead_end)

    #---------------------------------------------------------------------------
    def find_redundant_compounds(self):
        """
        Removes redundant compounds i.e. those that add no new net information.
        Typically, these will be compounds that show up paired in every
        reaction, such as ATP/ADP, NAD/NADH.
        """   

        stoich = self.stoichiometry

        # Transposing, to use the qr algorithm
        stoich_t = stoich.T
        n_rows = stoich_t.shape[0]
        n_cols = stoich_t.shape[1]
        
        if n_cols > n_rows:
            zeros = np.zeros((n_cols - n_rows, n_cols))
            padded = np.vstack((stoich_t, zeros))
        else:
            zeros = np.zeros((n_rows, n_rows - n_cols))
            padded = np.hstack((stoich_t, zeros))
        
        Q, R, P = qr(padded, pivoting=True)
        dependent = P[np.abs(R.diagonal()) < 1e-10]

        # Trimming padded values
        redundant = stoich.index[[i for i in dependent if i < n_cols]]

        msg = str(len(redundant)) + ' compounds identified as redundant'
        omfa.logger.info(msg)
        for i in range(len(redundant)):
            omfa.logger.info('\t' + str(i+1) + ' ' + redundant[i])

        return(redundant)

    #---------------------------------------------------------------------------
    def find_dangling_reactions(self, transport=[]):
        """
        Removes non-transport reactions that have either no products or no
        reactants, a situtation that may have occured as a result of matrix 
        trimming.

        transport (list): List of transport reactions are defined as
            one-way and should be ignored in the search.
        
        """

        stoich = self.stoichiometry

        invalid = set(transport) - set(stoich.columns)
        if len(invalid) > 0:
            msg = 'Invalid transport reactions\n'
            invalid = ', '.join(invalid)
            sol = '{0} are not in the stoichiometry matrix'.format(invalid)
            raise ValueError(msg + sol)

        reactions = set(stoich.columns) - set(transport)

        no_reactants = [i for i in reactions if sum(stoich[i] < 0) == 0]
        no_products = [i for i in reactions if sum(stoich[i] > 0) == 0]
        dangling = list(set(no_reactants) | set(no_products))

        msg = str(len(dangling)) + ' reactions identified as dangling'
        omfa.logger.info(msg)

        for i in range(len(dangling)):
            name = dangling[i]
            reactants = stoich.index[stoich[name] < 0]
            reactant_stoich = stoich.ix[stoich[name] < 0, name]

            products = stoich.index[stoich[name] > 0]
            product_stoich = stoich.ix[stoich[name] > 0, name]

            reactant_terms = [str(-reactant_stoich[j]) + ' ' + str(reactants[j]) for j in range(len(reactants))] 
            product_terms = [str(product_stoich[j]) + ' ' + str(products[j]) for j in range(len(products))] 
            stoich_string = ' + '.join(reactant_terms) + ' -> ' + ' - '.join(product_terms)

            omfa.logger.info('\t' + str(i+1) + ' [' +dangling[i] + '] ' + stoich_string)

        return(dangling)

    #=====================================================================>
    # Modification functions 
    #   - change Model data in-place 

    #---------------------------------------------------------------------
    def remove_compounds(self, compounds):
        """
        Shortcut for dropping compounds from the model.

        compounds (list): List of valid compound names.
        """

        # Checking if string to avoid iterating by character
        if isinstance(compounds, basestring):
            compounds = [compounds]
        # Carrying on to check if integer to maintain consistency
        elif isinstance(compounds, (int, long)):
            compounds = [compounds]

        # Identify invalid compounds
        invalid = set(compounds) - set(self.stoichiometry.index)

        if len(invalid) > 0:
            msg = 'The following compounds are invalid: ' + ', '.join(invalid)
            omfa.logger.error(msg)
            raise omfa.ModelError(msg)
        else:
            msg = 'Dropping ' + ', '.join(compounds)
            omfa.logger.info(msg)
            self.stoichiometry = self.stoichiometry.drop(compounds)

    #---------------------------------------------------------------------
    def remove_reactions(self, reactions):
        """
        Shortcut for dropping reactions from the model.

        compounds (list): List of valid reaction identifiers.
        """

        # Checking if string to avoid iterating by character
        if isinstance(reactions, basestring):
            reactions = [reactions]
        # Carrying on to check if integer to maintain consistency
        elif isinstance(reactions, (int, long)):
            reactions = [reactions]

        # Identify invalid reactions
        invalid = set(reactions) - set(self.stoichiometry.columns)

        if len(invalid) > 0:
            msg = 'The following reactions are invalid: ' + ', '.join(invalid)
            omfa.logger.error(msg)
            raise omfa.ModelError(msg)
        else:
            msg = 'Dropping ' + ', '.join(reactions)
            omfa.logger.info(msg)
            self.stoichiometry = self.stoichiometry.drop(reactions, 1)

    #---------------------------------------------------------------------
    def perturb_reaction(self, reaction,
                         abs_bias=None, rel_bias=None, 
                         abs_sd=None, rel_sd=None, seed=None):
        """
        Shortcut for modifying a model reaction for simulation purposes.
        Normally distributes 

        compound (str): Reaction name. 
        abs_bias: pandas.Series
            A set of bias values corresponding to each compound's
            stoichiometry coefficient.
        rel_bias: pandas.Series
            A set of bias values corresponding to each compound's
            stoichiometry coefficient, represented as a fraction of 
            the stoichiometry coefficient.
        abs_sd: pandas.Series
            A set of standard deviation values corresponding to each 
            compound's stoichiometry coefficient.
        rel_sd: pandas.Series
            A set of standard deviation values corresponding to each 
            compound's stoichiometry coefficient, represented as a 
            fraction of the stoichiometry coefficient.
        seed: hashable 
            By default, a random number generator is initialized on 
            package import. A new generator can be initialized with 
            the provided seed to be used for this operation alone.
        """

        # Coding parameters as dictionary
        param = {'abs_bias':abs_bias,
                 'rel_bias':rel_bias,
                 'abs_sd':abs_sd,
                 'rel_sd':rel_sd}

        # Extracting reaction
        try:
            stoichiometry = self.stoichiometry[reaction] 
        except KeyError:
            msg = '{0} is not a valid reaction.'.format(reaction)
            raise ValueError(msg)

        # Checking input
        for parameter in param:
            if not param[parameter] is None:
                if not set(param[parameter].index).issubset(
                           set(stoichiometry.index)):
                    msg = 'Noise parameters must relate to specific compounds'
                    raise ValueError(msg)
            else:
                param[parameter] = pd.Series(0, index=stoichiometry.index)

        for parameter in ['rel_bias', 'rel_sd']:
            if any(param[parameter] > 1) or any(param[parameter] < 0):
                msg = 'Relative parameters must be fractions'
                raise ValueError(msg)

        # Setting random number generator
        if seed is None:
            prng = omfa.prng
        else:
            prng = np.random.RandomState(seed)

        # Generating noise
        noise = stoichiometry.copy()
        noise.iloc[:] = prng.randn(len(stoichiometry.index))
        noise = (noise * param['abs_sd'] + 
                 noise * stoichiometry * param['rel_sd'])

        noise.fillna(0, inplace=True)

        # Adding bias
        updated = (stoichiometry + param['abs_bias'] + 
                   stoichiometry * param['rel_bias'] + noise)

        updated.fillna(0, inplace=True)

        self.stoichiometry[reaction] = updated

    #=====================================================================>
    # Solution space functions 
    #   -  probes and samples constrained solution spaces

    #---------------------------------------------------------------------
    def check_constraints(self, lower, upper):
        """
        Determines whether the specified lower and upper fluxe constraints
        are consistent with each other and the defined stoichiometry.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.

        Returns
        -------
        check_passed: bool
            True if no issues detected, False otherwise.
        """

        stoich = self.stoichiometry

        # Initializing check variable
        check_passed = True

        # Checking that limits correspond to valid fluxes
        if not set(lower.index).issubset(set(self.stoichiometry.columns)):
            msg = 'Limits must correspond to valid model fluxes'
            omfa.logger.warning(msg)
            check_passed = False

        if not set(upper.index).issubset(set(self.stoichiometry.columns)):
            msg = 'Limits must correspond to valid model fluxes'
            omfa.logger.warning(msg)
            check_passed = False

        # Lower limit may not be higher than upper limit
        invalid = lower > upper
        if any(invalid):
            msg = 'Upper and lower constraints must be consistent'
            omfa.logger.warning(msg)
            check_passed = False

        # Upper and lower constraints may not match 
        matching = lower == upper
        if any(matching):
            msg = 'Upper and lower constraints may not match'
            omfa.logger.warning(msg)
            check_passed = False

        # Performing LP problem to test constraints
        G, h = self.generate_basis_constraints(
                   lower, upper, relation='le', normalize=True)

        check_passed = check_passed and check_feasibility(G, h)

        if check_passed:
            msg = 'Lower and upper flux boundaries consistent'
            omfa.logger.info(msg)

        return(check_passed)

    #---------------------------------------------------------------------
    def reduce_constraints(self, lower, upper):
        """
        Eliminates redundant flux constraints. The elimination is performed
        on upper then lower constraints, one flux at a time. It's possible that 
        a lower constraint on one flux may make an upper constraint on another 
        flux redundant or vice versa. The order of elimination is performed 
        according to constraint index and can be manipulated by changing the 
        order of the index.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.

        Returns
        -------
        lower_reduced: pandas.Series
            A non-redundant set of lower limits on fluxes found in the model.
        upper_reduced: pandas.Series
            A non-redundant set of upper limits on fluxes found in the model.
        check_passed: bool
            True if no issues detected, False otherwise.
        """

        # Checking if constraints are valid
        check_passed = self.check_constraints(lower, upper)
    
        if not check_passed:
            msg = 'Lower and upper flux boundaries are not consistent.'
            omfa.logger.error(msg)
            raise omfa.ModelError(msg)

        # Global settings 
        solvers.options['show_progress'] = False

        kernel = self.nullspace.copy()
        m, n = kernel.shape
        x = variable(n)

        Al = np.array(kernel.ix[lower.index], dtype=np.float64)
        Al_opt = matrix(Al)
        lb = np.array(lower, dtype=np.float64)
        lb_opt = matrix(lb)
        i_lower = range(len(lower.index))

        Au = np.array(kernel.ix[upper.index], dtype=np.float64)
        Au_opt = matrix(Au)
        ub = np.array(upper, dtype=np.float64)
        ub_opt = matrix(ub)
        i_upper = range(len(upper.index))

        n_eliminated = 0

        # Eliminating upper constraints
        lower_constraints = [Al_opt[k,:]*x >= lb_opt[k] 
                             for k in i_lower] 
        
        while True: 

            for i in i_upper:
                upper_constraints = [Au_opt[k,:]*x <= ub_opt[k] 
                                     for k in i_upper if k != i]

                objective = -Au_opt[i,:]*x
                extra_constraint = [Au_opt[i,:]*x <= ub_opt[i] + 1]

                model = op(objective, lower_constraints +
                                      upper_constraints +
                                      extra_constraint)
                model.solve()
                status = model.status

                if status == 'optimal':
                    if Au[i,:].dot(np.array(x.value)) <= ub[i]:
                        i_upper.pop(i_upper.index(i))
                        n_eliminated = n_eliminated + 1

                        msg = 'Dropping upper constraint on {}'.format(
                                  upper.index[i])
                        omfa.logger.debug(msg)
                        break

                elif re.search('infeasible', status) is not None:
                    # Re-run the LP to confirm infeasability without constraint
                    model = op(objective, lower_constraints + upper_constraints)
                    model.solve()
                    status = model.status

                    if re.search('infeasible', status) is not None:
                        i_upper.pop(i_upper.index(i))
                        n_eliminated = n_eliminated + 1

                        msg = 'Dropping upper constraint on {}'.format(
                                  upper.index[i])
                        omfa.logger.debug(msg)
                        break

            if len(i_upper) == 0:
                break
            
            if i == i_upper[-1]:
                break

        # Eliminating lower constraints
        upper_constraints = [Au_opt[k,:]*x <= ub_opt[k] 
                             for k in i_upper] 
        
        while True: 

            for i in i_lower:
                lower_constraints = [Al_opt[k,:]*x >= lb_opt[k] 
                                     for k in i_lower if k != i]

                objective = Al_opt[i,:]*x
                extra_constraint = [Al_opt[i,:]*x >= lb_opt[i] - 1]

                model = op(objective, lower_constraints +
                                      upper_constraints +
                                      extra_constraint)
                model.solve()
                status = model.status

                if status == 'optimal':
                    if Al[i,:].dot(np.array(x.value)) >= lb[i]:
                        i_lower.pop(i_lower.index(i))
                        n_eliminated = n_eliminated + 1

                        msg = 'Dropping lower constraint on {}'.format(
                                  lower.index[i])
                        omfa.logger.debug(msg)
                        break

                elif re.search('infeasible', status) is not None:
                    # Re-run the LP to confirm infeasability without constraint
                    model = op(objective, lower_constraints + upper_constraints)
                    model.solve()
                    status = model.status

                    if re.search('infeasible', status) is not None:
                        i_lower.pop(i_lower.index(i))
                        n_eliminated = n_eliminated + 1

                        msg = 'Dropping lower constraint on {}'.format(
                                  lower.index[i])
                        omfa.logger.debug(msg)
                        break
            
            if len(i_lower) == 0:
                break

            if i == i_lower[-1]:
                break

        lower = lower.ix[i_lower]
        upper = upper.ix[i_upper]

        msg = '{} constraints eliminated as redundant'.format(n_eliminated)
        omfa.logger.info(msg)

        return(lower, upper)

    #---------------------------------------------------------------------
    def generate_basis_constraints(self, lower, upper, 
                                   relation='ge', normalize=False):
        """
        Converts lower and upper constraints on model fluxes to 
        strictly lower or upper constraints on basis variables that span the
        model's nullspace. This function uses the stoichiometric 
        relations to reduce the dimensionality of the solution space.

        Given Sx = 0 where lower <= x <= upper, calculate G and h such that
        Gb <= h or Gb >= h where b is a vector of basis variables defined
        from the nullspace of S.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.
        relation: str
            One of either 'ge' for Gb >= h or 'le' for Gb <= h.
        normalize: bool
            True if G[i,:] and h[i] should be normalized by sum(abs(G[i, :])).

        Returns
        -------
        G: pandas.DataFrame
            Matrix G forming the inequality constraints Gb >= h or Gb <= h 
            where b is a vector of basis variables, rather than fluxes.
        h: pandas.Series
            Vector h forming the inequality constraints Gb >= h or Gb <= h
            where b is a vector of basis variables, rather than fluxes.
        """

        # Forming constraints
        kernel = self.nullspace
        basis_names = kernel.columns

        lower_names = ['LC_{0}'.format(i) for i in lower.index]
        upper_names = ['UC_{0}'.format(i) for i in upper.index]
        constraint_names = lower_names + upper_names

        if relation == 'le':
            G = pd.concat([-kernel.ix[lower.index], kernel.ix[upper.index]]) 
            G.index = constraint_names

            h = pd.concat([-lower, upper])
            h.index = constraint_names
        elif relation == 'ge':
            G = pd.concat([kernel.ix[lower.index], -kernel.ix[upper.index]])
            G.index = constraint_names

            h = pd.concat([lower, -upper])
            h.index = constraint_names
        else:
            msg = 'Relation argument must be one of "le" or "ge"'
            raise omfa.ModelError(msg)

        if normalize:
            scale = kernel.abs().sum(axis=1)
            scale = pd.concat([scale.ix[lower.index], scale.ix[upper.index]])
            scale.index = constraint_names

            G = G.div(scale, axis=0)
            h = h.div(scale, axis=0)

        return(G, h)

    #---------------------------------------------------------------------
    def generate_basis_centroid(self, lower, upper,  
                                method='chebyshev', progress=False):
        """
        Calculates approximate centroid of the stoichiometric solution
        space basis.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.
        method: str
            One of either 'chebyshev' or 'analytic'.
        progress: bool
            True if detailed progress text from optimization should be shown.

        Returns
        -------
        basis: pd.Series
            Centroid location in terms of the basis variables.
        """

        # Generating inequalities as a function of basis variables
        G, h = self.generate_basis_constraints(
                   lower, upper, relation='le', normalize=True)

        if method == 'chebyshev':
            basis = chebyshev_center(G, h, progress=progress)
        elif method == 'analytic':
            basis = analytic_center(G, h, progress=progress)
        else:
            msg = ('"{}" is not a valid method. '.format(method),
                   'Use "chebyshev" or "analytic."')
            omfa.logger.error(msg)
            raise ValueError(msg)

        return(basis)

    #---------------------------------------------------------------------
    def generate_flux_centroid(self, lower, upper,
                               method='chebyshev', progress=False):
        """
        Calculates approximate centroid of the stoichiometric solution
        space.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.
        progress: bool
            True if detailed progress text from optimization should be shown.

        Returns
        -------
        fluxes: pd.Series
            Centroid location in terms of the flux variables.
        """

        # Performing the basis calculation and converting to
        # normal flux space
        kernel = self.nullspace
        basis = self.generate_basis_centroid(lower, upper, method, progress)

        fluxes = kernel.dot(basis)

        return(fluxes)

    #---------------------------------------------------------------------
    def generate_basis_ranges(self, lower, upper,  progress=False):
        """
        Calculates the range of values that each basis variable can
        take, using the stoichiometry nullspace.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.
        progress: bool
            True if detailed progress text from optimization should be shown.

        Returns
        -------
        ranges: pd.DataFrame
            A dataframe with columns indicating the lowest and highest
            values each basis variable can take.
        """

        # Converting to basis constraints
        G, h = self.generate_basis_constraints(
                   lower, upper, relation='le', normalize=True)

        ranges = variable_range(G, h, progress=progress)

        return(ranges)

    #---------------------------------------------------------------------
    def generate_flux_ranges(self, lower, upper, progress=False):
        """
        Calculates the range of values that each flux can take.

        Parameters
        ----------
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.
        progress: bool
            True if detailed progress text from optimization should be shown.

        Returns
        -------
        ranges: pd.DataFrame
            A dataframe with columns indicating the lowest and highest
            values each flux can take.
        """

        stoich = self.stoichiometry

        # Reforming constraints in the form of Gx <= h
        lower_names = ['LC_{0}'.format(i) for i in lower.index]
        upper_names = ['UC_{0}'.format(i) for i in upper.index]
        constraint_names = lower_names + upper_names

        G = pd.DataFrame(0, index = constraint_names, columns = stoich.columns)

        for i in lower.index:
            G.ix['LC_{0}'.format(i), i] = -1

        for i in upper.index:
            G.ix['UC_{0}'.format(i), i] = 1

        h = pd.concat([-lower, upper])
        h.index = constraint_names

        # Generating equality constraints
        A = stoich 
        b = pd.Series(0, index=A.index)

        # Calculating ranges
        ranges = variable_range(G, h, A=A, b=b, progress=progress)

        return(ranges)

    #---------------------------------------------------------------------
    def generate_sample(self, *args, **kwargs):
        """
        Wrapper around generate_sample() in the simulation module.
        """

        return(omfa.simulation.generate_sample(self, *args, **kwargs))

    #==========================================================================>
    # Covariance 
    #   - convenience functions for generating and checking covariance matrices 
    #     to be used in MFA fitting

    #---------------------------------------------------------------------------
    def check_covariance(self, covar):
        """
        Determines whether the specified fluxes are sufficient to solve the
        model.

        Parameters
        ----------
        covar: pandas.DataFrame 
            Covariance matrix corresponding to either flux measurement or net
            balance variability.

        Returns
        -------
        check_passed: bool
            True if no issues detected, False otherwise.
        """

        # Initializing check variable
        check_passed = True

        # Index and column names must be the same
        if sorted(covar.index) != sorted(covar.columns):
            msg = 'Covariance matrix must have the same rows and columns'
            omfa.logger.warning(msg)
            check_passed = False

        # If some of the covariance entries are fluxes, they must all be fluxes
        if len(set(covar.index) & set(self.stoichiometry.columns)) > 0:
            if not set(covar.index).issubset(set(self.stoichiometry.columns)):
                msg = 'Some flux entries found'
                sol = '\nCovariance entries must be either all fluxes or all balances'
                omfa.logger.warning(msg + sol)
                check_passed = False

        # If some of the covariance entries are balances, they must all be balances 
        if len(set(covar.index) & set(self.stoichiometry.index)) > 0:
            if not set(covar.index).issubset(set(self.stoichiometry.index)):
                msg = 'Some balance entries found'
                sol = '\nCovariance entries must be either all fluxes or all balances'
                omfa.logger.warning(msg + sol)
                check_passed = False

        # Matrix must be symmetric
        if not np.allclose(covar, covar.T):
            msg = 'Covariance matrix must be symmetric'
            omfa.logger.warning(msg)
            check_passed = False

        # Checking determinant
        if np.linalg.det(covar) == 0:
            msg = 'Covariance matrix must be invertible'
            omfa.logger.warning(msg)
            check_passed = False

        # Testing for positive definite
        try:
            np.linalg.cholesky(covar)
        except np.linalg.LinAlgError:
            msg = 'Covariance matrix must be positive definite'
            sol = '\nThis issue may be caused due to a near-zero eigenvalue appearing negative, which can be artificially corrected'
            omfa.logger.warning(msg + sol)
            check_passed = False

        if check_passed:
            msg = 'Covariance matrix suitable for MFA'
            omfa.logger.info(msg)

        return(check_passed)

    #---------------------------------------------------------------------------
    def generate_covariance(self, observations, rel_measurement=None, abs_measurement=None, abs_balance=None):
        """
        Generates a skeleton covariance model for use in OMFA fitting functions.
        The output can be further modified before fitting.

        Parameters
        ----------
        observations: pandas.Series 
            A pandas 1D array of flux values with names corresponding to model 
            reactions/transport fluxes. Flux names are used to seed the output
            covariance matrix and the flux values are needed to convert relative
            standard deviations (coefficients of varience) into absolute values.
        rel_measurement: float or pandas.Series
            Measurement standard deviation as a fraction of the measured flux.
            If float, value is applied to all observations. Otherwise, Series
            index must be found in observations. Total measurement error is the
            sum of relative and absolute errors.
        abs_measurement: float or pandas.Series
            Measurement standard deviation as an absolute flux value.
            If float, value is applied to all observations. Otherwise, Series
            index must be found in observations. Total measurement error is the
            sum of relative and absolute errors.
        abs_balance: float or pandas.Series
            Standard deviation that correpsonds to a particular balance rather
            than an observed flux, given in absolute terms. If float, value is
            applied to all balances. Total balance error is the sum of relative
            and absolute errors.

        Returns
        -------
        covar: pandas.DataFrame
            A pandas DataFrame that corresponds to the model's covariance. As a
            default, covar shape is generated from observations and corresponds
            to measurement error alone (suitable for PI fit). If balance error
            terms are included, covariance is generated around model balances 
            instead, factoring in the measurement error by stoichiometry
            (suitable for GLS fit).
        """

        balances = self.stoichiometry.index

        # Checking that observations correspond to valid fluxes
        if not set(observations.index).issubset(set(self.stoichiometry.columns)):
            msg= 'Observations must correspond to valid model fluxes'
            raise omfa.ModelError(msg)

        # Applying error to measurements
        if rel_measurement is None:
            rel_measurement = 0
        
        if abs_measurement is None:
            abs_measurement = 0

        tot_measurement = (abs(rel_measurement * observations) + abs_measurement) ** 2

        flux_covar = pd.DataFrame(0, index=observations.index, columns=observations.index)
        
        for i in observations.index:
            flux_covar.ix[i, i] = tot_measurement[i]

        # If no balance errors are provided, the covariance matrix initialized
        # from fluxes is returned. Otherwise, covariance must be calculated as
        # a function of stoichiometry balances
        if abs_balance is None:
            return(flux_covar)

        # Generating observed matrix from stoichiometry
        observed_matrix = self.stoichiometry[observations.index]

        balance_covar = observed_matrix.dot(flux_covar).dot(observed_matrix.T)

        # Adding absolute error to all terms
        for i in balances:
            balance_covar.ix[i, i] = balance_covar.ix[i, i] + abs_balance

        return(balance_covar)

    #==========================================================================>
    # Fit 
    #   - feed into omfa Fit classes

    #---------------------------------------------------------------------------
    def check_fit(self, fluxes):
        """
        Determines whether the specified fluxes are sufficient to solve the
        model.

        Parameters
        ----------
        fluxes: pandas.Series 
            A pandas 1D array with names corresponding to observed 
            reactions/transport fluxes.

        Returns
        -------
        check_passed: bool
            True if no issues detected, False otherwise.
        """

        # Initializing check variable
        check_passed = True

        # Removing measured fluxes from stoichiometry matrix
        fluxes_calc = [f for f in self.stoichiometry.columns if f not in fluxes.index]
        stoich_calc = self.stoichiometry.ix[:,fluxes_calc]

        # Unknown stoichiometry matrix must be overdetermined
        n_rows = stoich_calc.shape[0]
        n_cols = stoich_calc.shape[1]
        rank = np.linalg.matrix_rank(stoich_calc)

        if n_rows <= n_cols:
            msg = 'Not enough flux measurements to validate flux fit'
            omfa.logger.warning(msg)
            check_passed = False
        
        if rank < n_cols:
            msg = 'Stoichiometry columns not of full rank'
            omfa.logger.warning(msg)
            check_passed = False

        # Padding stoichiometry matrix
        n_max = max(n_rows, n_cols)
        padded = np.zeros((n_max, n_max))
        padded[:n_rows, :n_cols] = stoich_calc

        # Performing SVD
        U, s, V = np.linalg.svd(padded, full_matrices=True)
        singular = [i for i in range(len(s)) if s[i] < 1e-10]

        # Trimming singular and V values if columns had to be padded
        if n_cols < n_rows:
            difference = n_rows - n_cols
            V = V[:-difference, :-difference]
            singular = [i for i in singular if i < n_cols]
        
        kernel = V[singular,:].T

        # Identifying non-zero rows
        non_calculable = [i for i in range(kernel.shape[0]) if any(np.abs(kernel[i, :]) > 1e-10)]
        columns = stoich_calc.columns[non_calculable]

        if len(columns) > 0:
            msg = str(len(columns)) + ' non-calculable fluxes: ' + ', '.join(columns)
            omfa.logger.warning(msg)
            check_passed = False

        # Checking condition number
        condition = np.linalg.cond(stoich_calc)
        if condition > 1000:
            msg = 'Condition number is too high (' + str(condition) + ')'
            omfa.logger.warning(msg)
            check_passed = False

        if check_passed:
            msg = 'Stoichiometry matrix suitable for MFA (' + str(n_rows) + ', ' + str(n_cols) + ')'
            omfa.logger.info(msg)

        return(check_passed)

    #---------------------------------------------------------------------------
    def fit_pi(self, *args, **kwargs):
        
        return(omfa.PIFit(self, *args, **kwargs))

    #---------------------------------------------------------------------------
    def fit_gls(self, *args, **kwargs):
        
        return(omfa.GLSFit(self, *args, **kwargs))
