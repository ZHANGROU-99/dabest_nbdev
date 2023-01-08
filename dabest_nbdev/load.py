# AUTOGENERATED! DO NOT EDIT! File to edit: ../nbs/API/load.ipynb.

# %% auto 0
__all__ = ['Dabest', 'load']

# %% ../nbs/API/load.ipynb 3
from nbdev.showdoc import *
import nbdev
nbdev.nbdev_export()

# %% ../nbs/API/load.ipynb 4
__version__ = "0.3.1"

# %% ../nbs/API/load.ipynb 5
class Dabest(object):

    """
    Class for estimation statistics and plots.
    """

    def __init__(self, data, idx, x, y, paired, id_col, ci, 
                resamples, random_seed, proportional, delta2, 
                experiment, experiment_label, x1_level, mini_meta):

        """
        Parses and stores pandas DataFrames in preparation for estimation
        statistics. You should not be calling this class directly; instead,
        use `dabest.load()` to parse your DataFrame prior to analysis.
        """

        # Import standard data science libraries.
        import numpy as np
        import pandas as pd
        import seaborn as sns

        self.__delta2       = delta2
        self.__experiment   = experiment
        self.__ci           = ci
        self.__data         = data
        self.__id_col       = id_col
        self.__is_paired    = paired
        self.__resamples    = resamples
        self.__random_seed  = random_seed
        self.__proportional = proportional
        self.__mini_meta    = mini_meta 

        # Make a copy of the data, so we don't make alterations to it.
        data_in = data.copy()
        # data_in.reset_index(inplace=True)
        # data_in_index_name = data_in.index.name


        # Check if it is a valid mini_meta case
        if mini_meta is True:
            if proportional is True:
                err0 = '`proportional` and `mini_meta` cannot be True at the same time.'
                raise ValueError(err0)
            elif delta2 is True:
                err0 = '`delta` and `mini_meta` cannot be True at the same time.'
                raise ValueError(err0)
            
            if all([isinstance(i, str) for i in idx]):
                if len(pd.unique([t for t in idx]).tolist())!=2:
                    err0 = '`mini_meta` is True, but `idx` ({})'.format(idx) 
                    err1 = 'does not contain exactly 2 columns.'
                    raise ValueError(err0 + err1)
            elif all([isinstance(i, (tuple, list)) for i in idx]):
                all_idx_lengths = [len(t) for t in idx]
                if (np.array(all_idx_lengths) != 2).any():
                    err1 = "`mini_meta` is True, but some idx "
                    err2 = "in {} does not consist only of two groups.".format(idx)
                    raise ValueError(err1 + err2)
            


        # Check if this is a 2x2 ANOVA case and x & y are valid columns
        # Create experiment_label and x1_level
        if delta2 is True:
            if proportional is True:
                err0 = '`proportional` and `delta` cannot be True at the same time.'
                raise ValueError(err0)
            # idx should not be specified
            if idx:
                err0 = '`idx` should not be specified when `delta2` is True.'.format(len(x))
                raise ValueError(err0)

            # Check if x is valid
            if len(x) != 2:
                err0 = '`delta2` is True but the number of variables indicated by `x` is {}.'.format(len(x))
                raise ValueError(err0)
            else:
                for i in x:
                    if i not in data_in.columns:
                        err = '{0} is not a column in `data`. Please check.'.format(i)
                        raise IndexError(err)

            # Check if y is valid
            if not y:
                err0 = '`delta2` is True but `y` is not indicated.'
                raise ValueError(err0)
            elif y not in data_in.columns:
                err = '{0} is not a column in `data`. Please check.'.format(y)
                raise IndexError(err)

            # Check if experiment is valid
            if experiment not in data_in.columns:
                err = '{0} is not a column in `data`. Please check.'.format(experiment)
                raise IndexError(err)

            # Check if experiment_label is valid and create experiment when needed
            if experiment_label:
                if len(experiment_label) != 2:
                    err0 = '`experiment_label` does not have a length of 2.'
                    raise ValueError(err0)
                else: 
                    for i in experiment_label:
                        if i not in data_in[experiment].unique():
                            err = '{0} is not an element in the column `{1}` of `data`. Please check.'.format(i, experiment)
                            raise IndexError(err)
            else:
                experiment_label = data_in[experiment].unique()

            # Check if x1_level is valid
            if x1_level:
                if len(x1_level) != 2:
                    err0 = '`x1_level` does not have a length of 2.'
                    raise ValueError(err0)
                else: 
                    for i in x1_level:
                        if i not in data_in[x[0]].unique():
                            err = '{0} is not an element in the column `{1}` of `data`. Please check.'.format(i, experiment)
                            raise IndexError(err)

            else:
                x1_level = data_in[x[0]].unique()    
        self.__experiment_label = experiment_label
        self.__x1_level         = x1_level


        # Check if idx is specified
        if delta2 is False and not idx:
            err = '`idx` is not a column in `data`. Please check.'
            raise IndexError(err)


        # create new x & idx and record the second variable if this is a valid 2x2 ANOVA case
        if delta2 is True:
            # add a new column which is a combination of experiment and the first variable
            new_col_name = experiment+x[0]
            while new_col_name in data_in.columns:
                new_col_name += "_"
            data_in[new_col_name] = data_in[x[0]].astype(str) + " " + data_in[experiment].astype(str)

            #create idx            
            idx = []
            for i in list(map(lambda x: str(x), experiment_label)):
                temp = []
                for j in list(map(lambda x: str(x), x1_level)):
                    temp.append(j + " " + i)
                idx.append(temp)         
            self.__idx = idx
            self.__x1  = x[0]
            self.__x2  = x[1]
            # record the second variable and create idx
            x = new_col_name
        else:
            self.__idx = idx
            self.__x1  = None
            self.__x2  = None

        # Determine the kind of estimation plot we need to produce.
        if all([isinstance(i, str) for i in idx]):
            # flatten out idx.
            all_plot_groups = pd.unique([t for t in idx]).tolist()
            if len(idx) > len(all_plot_groups):
                err0 = '`idx` contains duplicated groups. Please remove any duplicates and try again.'
                raise ValueError(err0)
                
            # We need to re-wrap this idx inside another tuple so as to
            # easily loop thru each pairwise group later on.
            self.__idx = (idx,)

        elif all([isinstance(i, (tuple, list)) for i in idx]):
            all_plot_groups = pd.unique([tt for t in idx for tt in t]).tolist()
            
            actual_groups_given = sum([len(i) for i in idx])
            
            if actual_groups_given > len(all_plot_groups):
                err0 = 'Groups are repeated across tuples,'
                err1 = ' or a tuple has repeated groups in it.'
                err2 = ' Please remove any duplicates and try again.'
                raise ValueError(err0 + err1 + err2)

        else: # mix of string and tuple?
            err = 'There seems to be a problem with the idx you'
            'entered--{}.'.format(idx)
            raise ValueError(err)

        # Having parsed the idx, check if it is a kosher paired plot,
        # if so stated.
        #if paired is True:
        #    all_idx_lengths = [len(t) for t in self.__idx]
        #    if (np.array(all_idx_lengths) != 2).any():
        #        err1 = "`is_paired` is True, but some idx "
        #        err2 = "in {} does not consist only of two groups.".format(idx)
        #        raise ValueError(err1 + err2)

        # Check if there is a typo on paired
        if paired:
            if paired not in ("baseline", "sequential"):
                err = '{} assigned for `paired` is not valid.'.format(paired)
                raise ValueError(err)


        # Determine the type of data: wide or long.
        if x is None and y is not None:
            err = 'You have only specified `y`. Please also specify `x`.'
            raise ValueError(err)

        elif y is None and x is not None:
            err = 'You have only specified `x`. Please also specify `y`.'
            raise ValueError(err)

        # Identify the type of data that was passed in.
        elif x is not None and y is not None:
            # Assume we have a long dataset.
            # check both x and y are column names in data.
            if x not in data_in.columns:
                err = '{0} is not a column in `data`. Please check.'.format(x)
                raise IndexError(err)
            if y not in data_in.columns:
                err = '{0} is not a column in `data`. Please check.'.format(y)
                raise IndexError(err)

            # check y is numeric.
            if not np.issubdtype(data_in[y].dtype, np.number):
                err = '{0} is a column in `data`, but it is not numeric.'.format(y)
                raise ValueError(err)

            # check all the idx can be found in data_in[x]
            for g in all_plot_groups:
                if g not in data_in[x].unique():
                    err0 = '"{0}" is not a group in the column `{1}`.'.format(g, x)
                    err1 = " Please check `idx` and try again."
                    raise IndexError(err0 + err1)

            # Select only rows where the value in the `x` column 
            # is found in `idx`.
            plot_data = data_in[data_in.loc[:, x].isin(all_plot_groups)].copy()
            
            # plot_data.drop("index", inplace=True, axis=1)

            # Assign attributes
            self.__x = x
            self.__y = y
            self.__xvar = x
            self.__yvar = y

        elif x is None and y is None:
            # Assume we have a wide dataset.
            # Assign attributes appropriately.
            self.__x = None
            self.__y = None
            self.__xvar = "group"
            self.__yvar = "value"

            # First, check we have all columns in the dataset.
            for g in all_plot_groups:
                if g not in data_in.columns:
                    err0 = '"{0}" is not a column in `data`.'.format(g)
                    err1 = " Please check `idx` and try again."
                    raise IndexError(err0 + err1)
                    
            set_all_columns     = set(data_in.columns.tolist())
            set_all_plot_groups = set(all_plot_groups)
            id_vars = set_all_columns.difference(set_all_plot_groups)

            plot_data = pd.melt(data_in,
                                id_vars=id_vars,
                                value_vars=all_plot_groups,
                                value_name=self.__yvar,
                                var_name=self.__xvar)
                                
        # Added in v0.2.7.
        # remove any NA rows.
        plot_data.dropna(axis=0, how='any', subset=[self.__yvar], inplace=True)

        
        # Lines 131 to 140 added in v0.2.3.
        # Fixes a bug that jammed up when the xvar column was already 
        # a pandas Categorical. Now we check for this and act appropriately.
        if isinstance(plot_data[self.__xvar].dtype, 
                      pd.CategoricalDtype) is True:
            plot_data[self.__xvar].cat.remove_unused_categories(inplace=True)
            plot_data[self.__xvar].cat.reorder_categories(all_plot_groups, 
                                                          ordered=True, 
                                                          inplace=True)
        else:
            plot_data.loc[:, self.__xvar] = pd.Categorical(plot_data[self.__xvar],
                                               categories=all_plot_groups,
                                               ordered=True)
        
        # # The line below was added in v0.2.4, removed in v0.2.5.
        # plot_data.dropna(inplace=True)
        
        self.__plot_data = plot_data
        
        self.__all_plot_groups = all_plot_groups


        # Sanity check that all idxs are paired, if so desired.
        #if paired is True:
        #    if id_col is None:
        #        err = "`id_col` must be specified if `is_paired` is set to True."
        #        raise IndexError(err)
        #    elif id_col not in plot_data.columns:
        #        err = "{} is not a column in `data`. ".format(id_col)
        #        raise IndexError(err)

        # Check if `id_col` is valid
        if paired:
            if id_col is None:
                err = "`id_col` must be specified if `paired` is assigned with a not NoneType value."
                raise IndexError(err)
            elif id_col not in plot_data.columns:
                err = "{} is not a column in `data`. ".format(id_col)
                raise IndexError(err)

        EffectSizeDataFrame_kwargs = dict(ci=ci, is_paired=paired,
                                           random_seed=random_seed,
                                           resamples=resamples,
                                           proportional=proportional, 
                                           delta2=delta2, 
                                           experiment_label=self.__experiment_label,
                                           x1_level=self.__x1_level,
                                           x2=self.__x2,
                                           mini_meta = mini_meta)

        self.__mean_diff    = EffectSizeDataFrame(self, "mean_diff",
                                                **EffectSizeDataFrame_kwargs)

        self.__median_diff  = EffectSizeDataFrame(self, "median_diff",
                                               **EffectSizeDataFrame_kwargs)

        self.__cohens_d     = EffectSizeDataFrame(self, "cohens_d",
                                                **EffectSizeDataFrame_kwargs)

        self.__cohens_h     = EffectSizeDataFrame(self, "cohens_h",
                                                **EffectSizeDataFrame_kwargs)                                       

        self.__hedges_g     = EffectSizeDataFrame(self, "hedges_g",
                                                **EffectSizeDataFrame_kwargs)

        if not paired:
            self.__cliffs_delta = EffectSizeDataFrame(self, "cliffs_delta",
                                                    **EffectSizeDataFrame_kwargs)
        else:
            self.__cliffs_delta = "The data is paired; Cliff's delta is therefore undefined."


    def __repr__(self):
        # from .__init__ import __version__
        import datetime as dt
        import numpy as np

        # from .misc_tools import print_greeting

        # Removed due to the deprecation of is_paired
        #if self.__is_paired:
        #    es = "Paired e"
        #else:
        #    es = "E"

        greeting_header = print_greeting()

        RM_STATUS = {'baseline'  : 'for repeated measures against baseline \n', 
                     'sequential': 'for the sequential design of repeated-measures experiment \n',
                     'None'      : ''
                    }

        PAIRED_STATUS = {'baseline'   : 'Paired e', 
                         'sequential' : 'Paired e',
                         'None'       : 'E'
        }

        first_line = {"rm_status"    : RM_STATUS[str(self.__is_paired)],
                      "paired_status": PAIRED_STATUS[str(self.__is_paired)]}

        s1 = "{paired_status}ffect size(s) {rm_status}".format(**first_line)
        s2 = "with {}% confidence intervals will be computed for:".format(self.__ci)
        desc_line = s1 + s2

        out = [greeting_header + "\n\n" + desc_line]

        comparisons = []

        if self.__is_paired == 'sequential':
            for j, current_tuple in enumerate(self.__idx):
                for ix, test_name in enumerate(current_tuple[1:]):
                    control_name = current_tuple[ix]
                    comparisons.append("{} minus {}".format(test_name, control_name))
        else:
            for j, current_tuple in enumerate(self.__idx):
                control_name = current_tuple[0]

                for ix, test_name in enumerate(current_tuple[1:]):
                    comparisons.append("{} minus {}".format(test_name, control_name))

        if self.__delta2 is True:
            comparisons.append("{} minus {} (only for mean difference)".format(self.__experiment_label[1], self.__experiment_label[0]))
        
        if self.__mini_meta is True:
            comparisons.append("weighted delta (only for mean difference)")

        for j, g in enumerate(comparisons):
            out.append("{}. {}".format(j+1, g))

        resamples_line1 = "\n{} resamples ".format(self.__resamples)
        resamples_line2 = "will be used to generate the effect size bootstraps."
        out.append(resamples_line1 + resamples_line2)

        return "\n".join(out)


    # def __variable_name(self):
    #     return [k for k,v in locals().items() if v is self]
    #
    # @property
    # def variable_name(self):
    #     return self.__variable_name()
    
    @property
    def mean_diff(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the mean difference, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.
        
        Example
        -------
        >>> from scipy.stats import norm
        >>> import pandas as pd
        >>> import dabest
        >>> control = norm.rvs(loc=0, size=30, random_state=12345)
        >>> test    = norm.rvs(loc=0.5, size=30, random_state=12345)
        >>> my_df   = pd.DataFrame({"control": control,
                                    "test": test})
        >>> my_dabest_object = dabest.load(my_df, idx=("control", "test"))
        >>> my_dabest_object.mean_diff
        
        Notes
        -----
        This is simply the mean of the control group subtracted from
        the mean of the test group.
        
        .. math::
            \\text{Mean difference} = \\overline{x}_{Test} - \\overline{x}_{Control}
            
        where :math:`\\overline{x}` is the mean for the group :math:`x`.
        """
        return self.__mean_diff
        
        
    @property    
    def median_diff(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the median difference, its confidence interval, and relevant statistics, for all comparisons  as indicated via the `idx` and `paired` argument in `dabest.load()`.
        
        Example
        -------
        >>> from scipy.stats import norm
        >>> import pandas as pd
        >>> import dabest
        >>> control = norm.rvs(loc=0, size=30, random_state=12345)
        >>> test    = norm.rvs(loc=0.5, size=30, random_state=12345)
        >>> my_df   = pd.DataFrame({"control": control,
                                    "test": test})
        >>> my_dabest_object = dabest.load(my_df, idx=("control", "test"))
        >>> my_dabest_object.median_diff
        
        Notes
        -----
        This is the median difference between the control group and the test group.
        
        If the comparison(s) are unpaired, median_diff is computed with the following equation:

        .. math::
            \\text{Median difference} = \\widetilde{x}_{Test} - \\widetilde{x}_{Control}
            
        where :math:`\\widetilde{x}` is the median for the group :math:`x`.

        If the comparison(s) are paired, median_diff is computed with the following equation:

        .. math::
            \\text{Median difference} = \\widetilde{x}_{Test - Control}

        """
        return self.__median_diff
        
        
    @property
    def cohens_d(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the standardized mean difference Cohen's `d`, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.
        
        Example
        -------
        >>> from scipy.stats import norm
        >>> import pandas as pd
        >>> import dabest
        >>> control = norm.rvs(loc=0, size=30, random_state=12345)
        >>> test    = norm.rvs(loc=0.5, size=30, random_state=12345)
        >>> my_df   = pd.DataFrame({"control": control,
                                    "test": test})
        >>> my_dabest_object = dabest.load(my_df, idx=("control", "test"))
        >>> my_dabest_object.cohens_d
        
        Notes
        -----
        Cohen's `d` is simply the mean of the control group subtracted from
        the mean of the test group.
        
        If `paired` is None, then the comparison(s) are unpaired; 
        otherwise the comparison(s) are paired.

        If the comparison(s) are unpaired, Cohen's `d` is computed with the following equation:
        
        .. math::
            
            d = \\frac{\\overline{x}_{Test} - \\overline{x}_{Control}} {\\text{pooled standard deviation}}
                
        
        For paired comparisons, Cohen's d is given by
        
        .. math::
            d = \\frac{\\overline{x}_{Test} - \\overline{x}_{Control}} {\\text{average standard deviation}}
            
        where :math:`\\overline{x}` is the mean of the respective group of observations, :math:`{Var}_{x}` denotes the variance of that group,
        
        .. math::
        
            \\text{pooled standard deviation} = \\sqrt{ \\frac{(n_{control} - 1) * {Var}_{control} + (n_{test} - 1) * {Var}_{test} } {n_{control} + n_{test} - 2} }
        
        and
        
        .. math::
        
            \\text{average standard deviation} = \\sqrt{ \\frac{{Var}_{control} + {Var}_{test}} {2}}
            
        The sample variance (and standard deviation) uses N-1 degrees of freedoms.
        This is an application of `Bessel's correction <https://en.wikipedia.org/wiki/Bessel%27s_correction>`_, and yields the unbiased
        sample variance.
        
        References:
            https://en.wikipedia.org/wiki/Effect_size#Cohen's_d
            https://en.wikipedia.org/wiki/Bessel%27s_correction
            https://en.wikipedia.org/wiki/Standard_deviation#Corrected_sample_standard_deviation
        """
        return self.__cohens_d
    
    
    @property
    def cohens_h(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the standardized mean difference Cohen's `h`, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `directional` argument in `dabest.load()`.

        Example
        -------
        >>> from scipy.stats import randint
        >>> import pandas as pd
        >>> import dabest
        >>> control = randint.rvs(0, 2, size=30, random_state=12345)
        >>> test    = randint.rvs(0, 2, size=30, random_state=12345)
        >>> my_df   = pd.DataFrame({"control": control,
                                    "test": test})
        >>> my_dabest_object = dabest.load(my_df, idx=("control", "test")
        >>> my_dabest_object.cohens_h

        Notes
        -----
        Cohen's 'h' uses the information of proportion in the control and test groups to calculate the distance between two proportions.
        It can be used to describe the difference between two proportions as "small", "medium", or "large".
        It can be used to determine if the difference between two proportions is "meaningful".

        A directional Cohen's 'h' is computed with the following equation:

        .. math::
            h = 2 * \\arcsin{\\sqrt{proportion_{Test}}} - 2 * \\arcsin{\\sqrt{proportion_{Control}}}

        For a non-directional Cohen's 'h', the equation is:
        .. math::
            h = \\abs{2 * \\arcsin{\\sqrt{proportion_{Test}}}} - \\abs{2 * \\arcsin{\\sqrt{proportion_{Control}}}}
        
        References:
            https://en.wikipedia.org/wiki/Cohen%27s_h
        """
        return self.__cohens_h


    @property  
    def hedges_g(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for the standardized mean difference Hedges' `g`, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.
        
        
        Example
        -------
        >>> from scipy.stats import norm
        >>> import pandas as pd
        >>> import dabest
        >>> control = norm.rvs(loc=0, size=30, random_state=12345)
        >>> test    = norm.rvs(loc=0.5, size=30, random_state=12345)
        >>> my_df   = pd.DataFrame({"control": control,
                                    "test": test})
        >>> my_dabest_object = dabest.load(my_df, idx=("control", "test"))
        >>> my_dabest_object.hedges_g
        
        Notes
        -----
        
        Hedges' `g` is :py:attr:`cohens_d` corrected for bias via multiplication with the following correction factor:
        
        .. math::
            \\frac{ \\Gamma( \\frac{a} {2} )} {\\sqrt{ \\frac{a} {2} } \\times \\Gamma( \\frac{a - 1} {2} )}
            
        where
        
        .. math::
            a = {n}_{control} + {n}_{test} - 2
            
        and :math:`\\Gamma(x)` is the `Gamma function <https://en.wikipedia.org/wiki/Gamma_function>`_.
            
        
        
        References:
            https://en.wikipedia.org/wiki/Effect_size#Hedges'_g
            https://journals.sagepub.com/doi/10.3102/10769986006002107
        """
        return self.__hedges_g
        
        
    @property    
    def cliffs_delta(self):
        """
        Returns an :py:class:`EffectSizeDataFrame` for Cliff's delta, its confidence interval, and relevant statistics, for all comparisons as indicated via the `idx` and `paired` argument in `dabest.load()`.
        
        
        Example
        -------
        >>> from scipy.stats import norm
        >>> import pandas as pd
        >>> import dabest
        >>> control = norm.rvs(loc=0, size=30, random_state=12345)
        >>> test    = norm.rvs(loc=0.5, size=30, random_state=12345)
        >>> my_df   = pd.DataFrame({"control": control,
                                    "test": test})
        >>> my_dabest_object = dabest.load(my_df, idx=("control", "test"))
        >>> my_dabest_object.cliffs_delta
        
        
        Notes
        -----
        
        Cliff's delta is a measure of ordinal dominance, ie. how often the values from the test sample are larger than values from the control sample.
        
        .. math::
            \\text{Cliff's delta} = \\frac{\\#({x}_{test} > {x}_{control}) - \\#({x}_{test} < {x}_{control})} {{n}_{Test} \\times {n}_{Control}}
            
            
        where :math:`\\#` denotes the number of times a value from the test sample exceeds (or is lesser than) values in the control sample. 
         
        Cliff's delta ranges from -1 to 1; it can also be thought of as a measure of the degree of overlap between the two samples. An attractive aspect of this effect size is that it does not make an assumptions about the underlying distributions that the samples were drawn from. 
        
        References:
            https://en.wikipedia.org/wiki/Effect_size#Effect_size_for_ordinal_data
            https://psycnet.apa.org/record/1994-08169-001
        """
        return self.__cliffs_delta


    @property
    def data(self):
        """
        Returns the pandas DataFrame that was passed to `dabest.load()`.
        """
        return self.__data


    @property
    def idx(self):
        """
        Returns the order of categories that was passed to `dabest.load()`.
        """
        return self.__idx
    

    @property
    def x1(self):
        return self.__x1


    @property
    def x1_level(self):
        return self.__x1_level


    @property
    def x2(self):
        return self.__x2


    @property
    def experiment(self):
        return self.__experiment
    

    @property
    def experiment_label(self):
        return self.__experiment_label


    @property
    def delta2(self):
        return self.__delta2


    @property
    def is_paired(self):
        """
        Returns the type of repeated-measures experiment.
        """
        return self.__is_paired


    @property
    def id_col(self):
        """
        Returns the id column declared to `dabest.load()`.
        """
        return self.__id_col


    @property
    def ci(self):
        """
        The width of the desired confidence interval.
        """
        return self.__ci


    @property
    def resamples(self):
        """
        The number of resamples used to generate the bootstrap.
        """
        return self.__resamples


    @property
    def random_seed(self):
        """
        The number used to initialise the numpy random seed generator, ie.
        `seed_value` from `numpy.random.seed(seed_value)` is returned.
        """
        return self.__random_seed


    @property
    def x(self):
        """
        Returns the x column that was passed to `dabest.load()`, if any.
        """
        return self.__x


    @property
    def y(self):
        """
        Returns the y column that was passed to `dabest.load()`, if any.
        """
        return self.__y


    @property
    def _xvar(self):
        """
        Returns the xvar in dabest.plot_data.
        """
        return self.__xvar


    @property
    def _yvar(self):
        """
        Returns the yvar in dabest.plot_data.
        """
        return self.__yvar


    @property
    def _plot_data(self):
        """
        Returns the pandas DataFrame used to produce the estimation stats/plots.
        """
        return self.__plot_data

    
    @property
    def proportional(self):
        """
        Returns the proportional parameter class.
        """
        return self.__proportional

    
    @property
    def mini_meta(self):
        """
        Returns the mini_meta boolean parameter.
        """
        return self.__mini_meta


    @property
    def _all_plot_groups(self):
        """
        Returns the all plot groups, as indicated via the `idx` keyword.
        """
        return self.__all_plot_groups

# %% ../nbs/API/load.ipynb 6
def load(data, idx=None, x=None, y=None, paired=None, id_col=None,
        ci=95, resamples=5000, random_seed=12345, proportional=False, 
        delta2 = False, experiment = None, experiment_label = None,
        x1_level = None, mini_meta=False):
    '''
    Loads data in preparation for estimation statistics.

    This is designed to work with pandas DataFrames.

    Parameters
    ----------
    data : pandas DataFrame
    idx : tuple
        List of column names (if 'x' is not supplied) or of category names
        (if 'x' is supplied). This can be expressed as a tuple of tuples,
        with each individual tuple producing its own contrast plot
    x : string or list, default None
        Column name(s) of the independent variable. This can be expressed as
        a list of 2 elements if and only if 'delta2' is True; otherwise it 
        can only be a string.
    y : string, default None
        Column names for data to be plotted on the x-axis and y-axis.
    paired : string, default None
        The type of the experiment under which the data are obtained
    id_col : default None.
        Required if `paired` is True.
    ci : integer, default 95
        The confidence interval width. The default of 95 produces 95%
        confidence intervals.
    resamples : integer, default 5000.
        The number of resamples taken to generate the bootstraps which are used
        to generate the confidence intervals.
    random_seed : int, default 12345
        This integer is used to seed the random number generator during
        bootstrap resampling, ensuring that the confidence intervals
        reported are replicable.
    proportional : boolean, default False. 
        TO INCLUDE MORE DESCRIPTION ABOUT DATA FORMAT
    delta2 : boolean, default False
        Indicator of delta-delta experiment
    experiment : String, default None
        The name of the column of the dataframe which contains the label of 
        experiments
    experiment_lab : list, default None
        A list of String to specify the order of subplots for delta-delta plots.
        This can be expressed as a list of 2 elements if and only if 'delta2' 
        is True; otherwise it can only be a string. 
    x1_level : list, default None
        A list of String to specify the order of subplots for delta-delta plots.
        This can be expressed as a list of 2 elements if and only if 'delta2' 
        is True; otherwise it can only be a string. 
    mini_meta : boolean, default False
        Indicator of weighted delta calculation.

    Returns
    -------
    A `Dabest` object.

    Example
    --------
    Load libraries.

    >>> import numpy as np
    >>> import pandas as pd
    >>> import dabest

    Create dummy data for demonstration.

    >>> np.random.seed(88888)
    >>> N = 10
    >>> c1 = sp.stats.norm.rvs(loc=100, scale=5, size=N)
    >>> t1 = sp.stats.norm.rvs(loc=115, scale=5, size=N)
    >>> df = pd.DataFrame({'Control 1' : c1, 'Test 1': t1})

    Load the data.

    >>> my_data = dabest_nbdev.load(df, idx=("Control 1", "Test 1"))

    '''

    return Dabest(data, idx, x, y, paired, id_col, ci, resamples, random_seed, proportional, delta2, experiment, experiment_label, x1_level, mini_meta)
