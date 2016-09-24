"""
Definition of omfa.Analysis class.
"""

import copy
import itertools as it
from numpy.random import RandomState
import pandas as pd
import sqlite3
import warnings

# Local imports
import omfapy as omfa
from omfapy.simulation import generate_observations

class Analysis():
    """
    Groups a number of simulation and analysis functions to increase ease
    of use. All analysis is based around a single SQLite database. For
    one-off functions, see the simulation module.
    """

    # Attributes
    model = None
    stoichiometry = None

    # Sample generation
    _xo = None

    # Random number generation
    prng = None

    # Database definitions
    db_name = None

    _db_columns = {'samples':
                       ['sample', 'rank', 'observed', 'flux', 'value'],
                   'observations':
                       ['sample', 'label', 'rank', 
                        'observation', 'flux', 'value'],
                   'gls_overall':
                       ['sample', 'label', 'observation', 'p_chi2'],
                   'gls_calculated':
                       ['sample', 'label', 'rank',
                        'observation', 'flux', 'value',  
                        'p_t', 'ci_low', 'ci_high'],
                   'gls_predicted':
                       ['sample', 'label', 'rank',  
                        'observation', 'flux', 'value',  
                        'pi_low', 'pi_high'],
                   'pi_calculated':
                       ['sample', 'label', 'rank',
                        'observation', 'flux', 'value'],
                   'pi_overall':
                       ['sample', 'label', 'observation', 'p_chi2']}

    _db_types = {'sample': 'INTEGER', 
                 'flux': 'TEXT', 
                 'label': 'INTEGER',
                 'observation': 'INTEGER',
                 'value': 'REAL',
                 'observed': 'INTEGER',
                 'rank': 'INTEGER',
                 'p_t': 'REAL',
                 'p_chi2': 'REAL',
                 'ci_low': 'REAL',
                 'ci_high': 'REAL',
                 'pi_low': 'REAL',
                 'pi_high': 'REAL'}

    #--------------------------------------------------------------------------
    def __init__(self, db_name, model, seed=None, overwrite=False, warn=False):
        """
        Initializes analysis with a database name and model.
        
        Parameters
        ----------
        db_name: str
            Path to the database used for analysis.
        model: omfa.Model
            Model used in the analysis. 
        seed: hashable 
            Seed for stochasctic generation algorithms. Note, final results will
            depend on both the starting seed and order of operations.
        overwrite: bool 
            True if existing databases should be overwritten. 
            False if new analysis should be appended.
        warn: bool
            True if the presence of an existing table with 
            inconsistent column definition should generate a warning, 
            False if it should generate an error.
        
        Details
        -------
        Initializes database with the provided name and creates required 
        tables.
        """
        
        self.db_name = db_name
        self.model = model
        self.stoichiometry = model.stoichiometry

        # Initializing tables
        for table in self._db_columns:
            self._create_db(table, overwrite, warn)

        # Initializing prng
        self.prng = RandomState(seed)

    #==========================================================================>
    # Utility methods
    #   - helper functions that may be used multiple times
    
    #--------------------------------------------------------------------------
    def _create_db(self, table, overwrite=False, warn=False):
        """
        Initializes a pre-set database table if it doesn't exist.
        
        Parameters
        ----------
        table: str 
            One of "samples", "observations", "gls_calculated",
            "gls_observed", "gls_overall", or "pi_overall".
        overwrite: bool 
            True if existing table should be overwritten. 
            False if existing table should be checked.
        warn: bool 
            True if the presence of an existing table with 
            inconsistent column definition should generate a warning, 
            False if it should generate an error.

        Details
        -------
        Creates table in databases if it doesn't exist. If a table exists,
        it's columns are checked to see if they conform to required 
        definitions.
        """

        # Checking if requested table is valid
        if table not in self._db_columns:
            msg = '"{}" table is not defined for the Analysis class.\n'
            sol = 'Table must be one of the following: {}'.format(
                      ', '.join([c for c in self._db_columns]))
            raise omfa.AnalysisException(msg + sol)

        # Connecting to database
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        # Dropping table if needed
        if overwrite:
            cur.execute("DROP TABLE IF EXISTS {};".format(table))
            start_sample = 1

        # Checking if table exists 
        cur.execute("SELECT name FROM sqlite_master WHERE type = 'table';")
        tables = [i[0] for i in cur.fetchall()]
        
        # If table exists, checking columns
        if table in tables:
            cur.execute("PRAGMA table_info({});".format(table))
            table_info = cur.fetchall()

            columns = {}
            for info in table_info:
                columns[info[1]] = info[2]

            missing_columns = []
            wrong_types = []
            problems = 0

            for i, column in enumerate(self._db_columns[table]):
                if column not in columns:
                    missing_columns.append(column)
                    problems = problems + 1  
                elif columns[column] != self._db_types[column]:
                    wrong_types.append(column)
                    problems = problems + 1

            if problems > 0:
                msg = ('Existing table "{}" does not meet requirements.\n'
                            .format(table))
                
                if len(missing_columns) > 0:
                    det = ('\tThe following columns are missing:\n\t\t{}\n'
                               .format(', '.join(missing_columns)))
                    msg = msg + det

                if len(wrong_types) > 0:
                    det = ('\tThe following columns have wrong types:\n\t\t{}\n'
                               .format(', '.join(wrong_types)))
                    msg = msg + det

                if warn:
                    omfa.logger.warning(msg)
                else:
                    omfa.logger.error(msg)
                    raise omfa.AnalysisError(msg)
    
        # Creating table
        columns = copy.deepcopy(self._db_columns[table])
        for i, column in enumerate(columns):
            columns[i] = '{0} {1}'.format(column, self._db_types[column])

        com = 'CREATE TABLE IF NOT EXISTS {0} ({1});'.format(
                  table, ', '.join(columns)) 

        cur.execute(com)
        con.commit()

        con.close()

    #------------------------------------------------------------------------
    # Generate a single column index
    def _gen_single_index(self, table, column_name, unique=False):
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        if unique:
            unique_string = "UNIQUE"
        else:
            unique_string = ""

        try:
            query = 'DROP INDEX {1}_{0}_index;'
            com = query.format(table, column_name)

            con.cursor().execute(com)
            con.commit()
        except sqlite3.OperationalError:
            pass

        query = 'CREATE {2} INDEX IF NOT EXISTS {1}_{0}_index ON {1} ({0});'
        com = query.format(column_name, table, unique_string)

        cur.execute(com)
        con.commit()
        con.close()

    #------------------------------------------------------------------------
    # Generate multiple single column indexes
    def _gen_single_indexes(self, table, columns, unique=False):
        if unique:
            unique_string = "UNIQUE"
        else:
            unique_string = ""

        for column_name in columns:
            self._gen_single_index(table, column_name, unique)

    #------------------------------------------------------------------------
    # Generate a single multi-column index
    def _gen_comp_index(self, table, columns, unique=False):
        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        if unique:
            unique_string = "UNIQUE"
        else:
            unique_string = ""

        column_list = ', '.join(columns)
        column_names = '_'.join(columns)

        try:
            query = 'DROP INDEX {1}_{0}_index;'
            com = query.format(table, column_names)

            con.cursor().execute(com)
            con.commit()
        except sqlite3.OperationalError:
            pass

        query = 'CREATE {3} INDEX {2}_{1}_index ON {1} ({0});'
        com = query.format(column_list, table, column_names, unique_string)
        
        cur.execute(com)
        con.commit()
        con.close()

    #------------------------------------------------------------------------
    # Generate all combinations of multi-column indexes
    def _gen_comp_indexes(self, table, columns, unique=False):
        if unique:
            unique_string = "UNIQUE"
        else:
            unique_string = ""

        for repeat in range(2, len(columns) + 1):
            combinations = it.combinations(columns, repeat)
            for combination in combinations:
                self._gen_comp_index(table, combination, unique)

    #------------------------------------------------------------------------
    # Generate both single and compound indexes
    def _gen_all_indexes(self, table, columns):
        self._gen_single_indexes(table, columns)
        self._gen_comp_indexes(table, columns)

    #==========================================================================>
    # Main methods
    #   - flux generation and calculation
    
    #--------------------------------------------------------------------------
    def generate_samples(self, n_samples, lower, upper, mode='fill',
                         method='ma', fortran=True, **param):
        """
        Generates a random sample from the model solution space defined by
        lower and upper constraints.

        Parameters
        ----------
        model: omfa.Model
        lower: pandas.Series
            A set of lower limits on fluxes found in the model.
        upper: pandas.Series
            A set of upper limits on fluxes found in the model.
        mode: str
            'add' to add n_samples regardless of how many samples have already
            been generated, 'fill' to add as many new samples as needed to
            have a total number of n_samples
        method: str
            One of either 'rda' for random direction algorithm or 'ma' for
            mirror algorithm.
        fortran: bool
            Whether to use a python or fortran implementation. The fortran
            implementation is much faster, but requires the successful
            compilation of the provided fortran source code.

        **param:
            Parameters that depend on the algorithm.

            Generic:

            xo: pandas.Series
                A set of flux values that meets constraint restrictions
                to use as a sampling starting point.
            n_burnin: int
                The number of steps to take before sampling.
            n_iter: int
                The number of steps to take for drawing samples.
            n_out: int
                The number of steps to return from those taken.

            method='rda':

            n_max: int
                The maximum total number of steps that can be missed
                due to a direction being picked with no available steps.

            method='ma':
            
            sd: pandas.Series
                Standard deviations to be used for choosing a random
                direction. The standard deviation values correspond
                to basis variables rather than fluxes.

        Details
        -------
        Adds new samples directly to the samples table in the database.
        """

        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        # Getting number of current samples
        query = ('SELECT MAX(sample) FROM samples')
        cur.execute(query)
        start_sample = cur.fetchall()[0][0]
       
        if start_sample is None:
            start_sample = 1
        else:
            start_sample = start_sample + 1

        # Calculating stop sample based on mode
        if mode == 'add':
            stop_sample = start_sample + n_samples
        elif mode == 'fill':
            stop_sample = n_samples + 1

        if stop_sample <= start_sample:
            return

        # Recalling OMFAModel
        m = self.model

        # If xo not provided, generate one and save it
        if 'xo' not in param:
            xo = m.generate_chebyshev_flux_center(lower, upper)
            param['xo'] = xo

        self._xo = param['xo']

        seed = self.prng.tomaxint()

        # Generating distributions
        samples = m.generate_sample(lower, upper, method=method, fortran=True,
                                    n_out=n_samples, seed=seed, **param) 

        # Melting to generate one column
        samples.columns = range(start_sample, stop_sample)
        melted = samples.stack()
        melted.index.names = ['flux', 'sample']
        melted = melted.swaplevel(0, 1)
        melted = pd.DataFrame(melted, columns=['value'])
        melted.sort_index(inplace=True)
        melted = melted.reset_index()

        # Ranking flux in terms of flux magnitude
        f = lambda x: abs(x).rank(method='first')

        ranks = melted.groupby(['sample'])['value'].transform(f)
        melted['rank'] = ranks
        melted['observed'] = [True if flux in lower.index else False
                              for flux in melted['flux']]

        pd.io.sql.to_sql(melted, 'samples', con, 
                         if_exists='append', index=False, 
                         dtype=self._db_types)

        # Generating indeces
        columns = ['sample', 'flux', 'rank', 'observed']
        self._gen_all_indexes('samples', columns)
        con.close()

    #--------------------------------------------------------------------------
    def generate_observations(self, n_observations, label, mode='fill', 
                              abs_bias=None, rel_bias=None, 
                              abs_sd=None, rel_sd=None):
        """
        Generates n observations per set of flux values by perturbing the
        sampled flux values by normally distributed noise. Any noise parameters
        that are not None must be set for all fluxes. The function does
        not make any assumptions.
        
        Parameters
        ----------
        n_observations: int
            The number of observations to generate from each set of fluxes.
        label: int or str
            A label for the perturbation conditions (for database retrieval).
        mode: str
            'add' to add N_observation regardless of how many observations have 
            already been generated, 'fill' to add as many new observations as 
            needed to have a total number of n_observations. All observations
            added per sample
        abs_bias: pandas.Series
            A set of bias values corresponding to each flux.
        rel_bias: pandas.Series
            A set of bias values corresponding to each flux, represented
            as a fraction of the observed flux.
        abs_sd: pandas.Series
            A set of standard deviation values corresponding to each flux.
        rel_sd: pandas.Series
            A set of standard deviation values corresponding to each flux,
            represented as a fraction of the observed flux.

        Details
        -------
        For every generated sample, adds new observations directly to the
        observations table in the database.
        """

        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        # Getting a list of real samples
        cur.execute('SELECT DISTINCT sample FROM samples')
        samples = [i[0] for i in cur.fetchall()]

        if len(samples) == 0:
            msg = 'No samples found.\n' 
            sol = 'Run generate_samples() before generate_observations().'
            omfa.logger.warn(msg + sol)

        # Looping through samples
        for sample in samples:

            # Getting number of observations tied to perturbation label
            query = ("SELECT MAX(observation) FROM observations "
                     "WHERE sample = {0} and label = '{1}'").format(
                      sample, label)

            cur.execute(query)
            start_observation = cur.fetchall()[0][0]
           
            if start_observation is None:
                start_observation = 1
            else:
                start_observation = start_observation + 1

            # Calculating stop sample based on mode
            if mode == 'add':
                stop_observation = start_observation + n_observations
                n_new_observations = n_observations
            elif mode == 'fill':
                stop_observation = n_observations + 1
                n_new_observations = stop_observation - start_observation

            if stop_observation <= start_observation:
                continue

            com = ('SELECT flux, value FROM samples '
                   'WHERE sample = {} AND observed = 1')
            query = com.format(sample)
           
            real_observed = pd.read_sql(query, con, index_col='flux')['value']

            seed = self.prng.tomaxint()

            # Observations
            observations = generate_observations(
                real_observed, n_new_observations, abs_bias=abs_bias,
                rel_bias=rel_bias, abs_sd=abs_sd, rel_sd=rel_sd, seed=seed)

            observations.columns = range(start_observation, stop_observation)
            melted = observations.stack()
            melted.index.names = ['flux', 'observation']
            melted = melted.swaplevel(0, 1)
            melted = pd.DataFrame(melted, columns=['value'])
            melted.sort_index(inplace=True)
            melted = melted.reset_index()

            # Ranking flux in terms of flux magnitude
            f = lambda x: abs(x).rank(method='first')

            ranks = melted.groupby(['observation'])['value'].transform(f)
            melted['rank'] = ranks
            melted['label'] = label
            melted['sample'] = sample

            columns = ['label', 'sample', 'observation', 
                       'flux', 'value', 'rank']
            melted = melted[columns]

            pd.io.sql.to_sql(melted, 'observations', con, 
                             if_exists='append', index=False, 
                             dtype=self._db_types)

        # Generating indeces
        columns = ['sample', 'label', 'flux', 'rank', 'observation']
        self._gen_all_indexes('observations', columns)
        con.close()

    #--------------------------------------------------------------------------
    def calculate_gls(self, label, rel_measurement=0, abs_measurement=0.01, 
                      abs_balance=0.0001):
        """
        Performs General Least Squares calculations on a given subset of the
        data. Input parameters are used to construct the covariance matrix
        used in validation.

        Parameters
        ----------
        label: int or str
            A label for the perturbation conditions (for database retrieval).
        rel_measurement: float or pandas.Series
            Measurement standard deviation as a fraction of the flux.
            If float, value is applied to all fluxes. Otherwise, Series
            index must be found in observations. Total measurement error is the
            sum of relative and absolute errors.
        abs_measurement: float or pandas.Series
            Measurement standard deviation as an absolute flux value.
            If float, value is applied to all fluxes. Otherwise, Series
            index must be found in observations. Total measurement error is the
            sum of relative and absolute errors. This term is primarily used to 
            prevent variances of 0 from appearing when a flux is 0.
        abs_balance: float or pandas.Series
            Standard deviation, given in absolute terms, that correpsonds to a 
            particular material balance rather than a flux. If float, value is
            applied to all balances. This term is primarily used to prevent
            variances of 0 from appearing in the covariance matrix.

        Details
        -------
        Calculates a GLS fit and performs the correspondsing validation on
        each observation with the given label.
        """

        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        # Getting a list of real samples
        cur.execute('SELECT DISTINCT sample FROM samples')
        samples = [i[0] for i in cur.fetchall()]

        if len(samples) == 0:
            msg = 'No samples found.\n' 
            sol = 'Run generate_samples() and generate_observations().'
            omfa.logger.warn(msg + sol)

        # Checking to make sure that there are entries with the given label
        cur.execute("SELECT * FROM observations "
                    "WHERE label = '{}' LIMIT 1;".format(label))
        results = [i[0] for i in cur.fetchall()]

        if len(results) == 0:
            msg = 'No observations with the label "{}" found.\n'.format(label) 
            sol = 'Check label or run generate_observations().'
            omfa.logger.warn(msg + sol)

        # Recalling OMFAModel
        m = self.model

        # Looping through samples
        for sample in samples:

            # Getting real values
            com = ('SELECT flux, value, observed FROM samples '
                   'WHERE sample = {} AND observed = 1')
            query = com.format(sample)
           
            real = pd.read_sql(query, con, index_col='flux')

            # Fetching observations that haven't been calculated
            com = ("SELECT o.label, o.observation, o.flux, o.value "
                   "FROM observations as o "
                   "LEFT JOIN gls_overall as g "
                   "ON o.label = g.label AND "
                   "   o.sample = g.sample AND "
                   "   o.observation = g.observation "
                   "WHERE o.sample = {0} AND o.label = '{1}'")
            query = com.format(sample, label)

            observations = pd.read_sql(query, con)

            indexes = ['flux']
            columns = ['observation']
            observed = pd.pivot_table(observations, values='value', 
                                      index=indexes, columns=columns)
                                        
            # Fitting GLS
            covar = m.generate_covariance(real['value'], 
                                          rel_measurement=rel_measurement, 
                                          abs_measurement=abs_measurement, 
                                          abs_balance=abs_balance)  
            
            fit_gls = m.fit_gls(observed, covar)
            p_chi2, p_t, ci, pi = fit_gls.validate()
   
            # First, the flux based calculations
            p_t.columns.names = ['observation']
            p_t.index.names = ['flux']
            p_t = p_t.stack()
            p_t = pd.DataFrame({'p_t':p_t})

            ci.index.names = ['flux']
            ci = ci.stack(0)
            ci.columns = ['ci_low', 'value', 'ci_high'] 

            frames = [p_t, ci]
            gls_out = pd.concat(frames, axis=1)
            gls_out = gls_out.reset_index()

            f = lambda x: abs(x).rank(method='first')

            ranks = gls_out.groupby(['observation'])['value'].transform(f)
            gls_out['rank'] = ranks
            gls_out['label'] = label 
            gls_out['sample'] = sample

            gls_out.sort(['label', 'sample', 'observation', 'flux'], 
                         inplace=True)
            
            columns = ['label', 'sample', 'observation', 'flux', 'value', 
                       'rank', 'p_t', 'ci_low', 'ci_high']
            gls_out = gls_out[columns]

            pd.io.sql.to_sql(gls_out, 'gls_calculated', con, 
                             if_exists='append', index=False,
                             dtype=self._db_types)

            # Chi2 results
            p_chi2 = pd.DataFrame({'p_chi2':p_chi2})
            p_chi2['label'] = label 
            p_chi2['sample'] = sample

            p_chi2 = p_chi2.reset_index()
            p_chi2.sort(['label', 'sample', 'observation'], inplace=True)

            columns = ['label', 'sample', 'observation', 'p_chi2']
            p_chi2 = p_chi2[columns]

            pd.io.sql.to_sql(p_chi2, 'gls_overall', con, 
                             if_exists='append', index=False,
                             dtype=self._db_types)

        # Generating indeces
        columns = ['sample', 'label', 'flux', 'rank', 'observation']
        self._gen_all_indexes('gls_calculated', columns)

        columns = ['sample', 'label', 'observation']
        self._gen_all_indexes('gls_overall', columns)
        con.close()

    #--------------------------------------------------------------------------
    def calculate_pi(self, label, rel_measurement=0, abs_measurement=0.01):
        """
        Performs Pseudo Inverse calculations on a given subset of the
        data. Input parameters are used to construct the covariance matrix
        used in validation.

        Parameters
        ----------
        label: int or str
            A label for the perturbation conditions (for database retrieval).
        rel_measurement: float or pandas.Series
            Measurement standard deviation as a fraction of the flux.
            If float, value is applied to all fluxes. Otherwise, Series
            index must be found in observations. Total measurement error is the
            sum of relative and absolute errors.
        abs_measurement: float or pandas.Series
            Measurement standard deviation as an absolute flux value.
            If float, value is applied to all fluxes. Otherwise, Series
            index must be found in observations. Total measurement error is the
            sum of relative and absolute errors. This term is primarily used to 
            prevent variances of 0 from appearing when a flux is 0.

        Details
        -------
        Calculates a PI fit and performs the correspondsing validation on
        each observation with the given label.
        """

        con = sqlite3.connect(self.db_name)
        cur = con.cursor()

        # Getting a list of real samples
        cur.execute('SELECT DISTINCT sample FROM samples')
        samples = [i[0] for i in cur.fetchall()]

        if len(samples) == 0:
            msg = 'No samples found.\n' 
            sol = 'Run generate_samples() and generate_observations().'
            omfa.logger.warn(msg + sol)

        # Checking to make sure that there are entries with the given label
        cur.execute("SELECT * FROM observations "
                    "WHERE label = '{}' LIMIT 1;".format(label))
        results = [i[0] for i in cur.fetchall()]

        if len(results) == 0:
            msg = 'No observations with the label "{}" found.\n'.format(label) 
            sol = 'Check label or run generate_observations().'
            omfa.logger.warn(msg + sol)

        # Recalling OMFAModel
        m = self.model

        # Looping through samples
        for sample in samples:

            # Getting real values
            com = ('SELECT flux, value, observed FROM samples '
                   'WHERE sample = {} AND observed = 1')
            query = com.format(sample)
           
            real = pd.read_sql(query, con, index_col='flux')

            # Fetching observations that haven't been calculated
            com = ("SELECT o.label, o.observation, o.flux, o.value "
                   "FROM observations as o "
                   "LEFT JOIN pi_overall as p "
                   "ON o.label = p.label AND "
                   "   o.sample = p.sample AND "
                   "   o.observation = p.observation "
                   "WHERE o.sample = {0} AND o.label = '{1}'")
            query = com.format(sample, label)

            observations = pd.read_sql(query, con)

            indexes = ['flux']
            columns = ['observation']
            observed = pd.pivot_table(observations, values='value', 
                                      index=indexes, columns=columns)
                                        
            # Fitting PI
            covar = m.generate_covariance(real['value'], 
                                          rel_measurement=rel_measurement, 
                                          abs_measurement=abs_measurement)  
            
            fit_pi = m.fit_pi(observed)
            p_chi2 = fit_pi.validate(covar)

            # Calculated flux values
            calculated = fit_pi.calculated_fluxes
            calculated.columns.names = ['observation']
            calculated.index.names = ['flux']

            calculated = calculated.stack()
            calculated = pd.DataFrame({'value':calculated})
            calculated = calculated.reset_index()

            f = lambda x: abs(x).rank(method='first')

            ranks = calculated.groupby(['observation'])['value'].transform(f)
            calculated['rank'] = ranks
            calculated['label'] = label
            calculated['sample'] = sample
            calculated.sort(['label', 'sample', 'observation', 'flux'], 
                            inplace=True)

            
            columns = ['label', 'sample', 'observation', 
                       'flux', 'value', 'rank']
            calculated = calculated[columns]

            pd.io.sql.to_sql(calculated, 'pi_calculated', con, 
                             if_exists='append', index=False,
                             dtype=self._db_types)

   
            # Chi2 results
            p_chi2 = pd.DataFrame({'p_chi2':p_chi2})
            p_chi2['label'] = label 
            p_chi2['sample'] = sample

            p_chi2 = p_chi2.reset_index()
            p_chi2.sort(['label', 'sample', 'observation'], inplace=True)

            columns = ['label', 'sample', 'observation', 'p_chi2']
            p_chi2 = p_chi2[columns]

            pd.io.sql.to_sql(p_chi2, 'pi_overall', con, 
                             if_exists='append', index=False,
                             dtype=self._db_types)

        # Generating indeces 
        columns = ['sample', 'label', 'flux', 'rank', 'observation']
        self._gen_all_indexes('gls_calculated', columns)
        
        columns = ['sample', 'label', 'observation']
        self._gen_all_indexes('pi_overall', columns)
        con.close()



