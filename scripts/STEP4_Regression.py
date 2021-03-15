import pandas as pd
import copy
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
import numpy as np
import seaborn as sb
import matplotlib.pyplot as plt
from sklearn import preprocessing
import statsmodels.api as sm
# https://www.statsmodels.org/stable/api.html
from linearmodels import PooledOLS
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from datetime import datetime
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")

class combine_dataframes():

    def __init__(self,
                 initial_panel,
                 all_panels,
                 consec_panels,
                 appid_imputed_and_deleted_missing_df,
                 dev_index_gecoded_df,
                 appid_text_cluster_labeled_df):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
        self.dfa = appid_imputed_and_deleted_missing_df
        self.dfd = dev_index_gecoded_df
        self.dfl = appid_text_cluster_labeled_df

    def combine_imputed_deleted_missing_with_text_labels(self):
        inter_df = self.dfa.join(self.dfl, how='inner')
        inter_df.index.names = ['appid']
        inter_df.reset_index(inplace=True)
        dfd2 = self.dfd.reset_index()
        dfd2 = dfd2[['developer', 'location', 'longitude', 'latitude']]
        # ATTENTION: here I used the initial panel developer to conduct m:1 merge for appid index to developer index
        inter_df.rename(columns={'developer_'+self.initial_panel: 'developer'}, inplace=True) # here I did not delete apps that changed developer over time
        result_df = inter_df.merge(dfd2, on='developer', how='left', validate='m:1')
        result_df.set_index('appid', inplace=True)
        new_count_cols = ['count_'+i for i in self.consec_panels]
        for i in new_count_cols: # for the purpose of creating dataframes in the groupby count
            result_df[i] = 0
        return result_df

    def convert_to_dev_multiindex(self):
        result_df = self.combine_imputed_deleted_missing_with_text_labels()
        result_df.reset_index(inplace=True)
        num_apps_df = result_df.groupby('developer')['appid'].nunique().rename('num_apps_owned').to_frame()
        result_df = result_df.merge(num_apps_df, on='developer', how='inner')
        result_df.set_index(['developer', 'appid'], inplace=True)
        return result_df

#########################################################################################
#######   REGRESSION   ##################################################################
#########################################################################################

class regression_analysis():
    """by default, regression analysis will either use cross sectional data or panel data with CONSECUTIVE panels,
    this is because we can calculate t-1 easily."""

    def __init__(self,
                 df,
                 initial_panel,
                 consec_panels,
                 dep_var=None,
                 independent_vars=None,
                 panel_long_df=None):
        self.df = df # df is the output of combine_imputed_deleted_missing_with_text_labels
        self.initial_panel = initial_panel
        self.consec_panels = consec_panels
        self.dep_var = dep_var
        self.independent_vars = independent_vars
        self.panel_long_df = panel_long_df

    def select_partial_vars(self, text):
        l1 = []
        for i in self.df.columns:
            if text in i:
                contains_digit = any(map(str.isdigit, i))
                if contains_digit is True: # seperate the panel information from the variable sub-string
                    x = i.split("_", -1)
                    x = x[:-1]
                    y = '_'.join(x)
                    l1.append(y)
                else: # the string column name does not contain panel informaiton (time-invariant variables)
                    l1.append(i)
        return list(set(l1))

    def select_vars(self, the_panel=None, **kwargs):
        if 'var_list' in kwargs.keys():
            variables = kwargs['var_list']
        elif 'partial_var' in kwargs.keys():
            variables = self.select_partial_vars(text=kwargs['partial_var'])
        elif 'single_var' in kwargs.keys():
            variables = [kwargs['single_var']]
        if the_panel is None:
            selected_cols = []
            for p in self.consec_panels:
                cols = [item + '_' + p for item in variables]
                selected_cols.extend(cols)
        else:
            selected_cols = [item + '_' + the_panel for item in variables]
        selected_df = self.df[selected_cols]
        return selected_df

    def select_panel_vars(self, time_invariant_vars, time_variant_vars, dep_vars):
        vars = ['latitude',
                'longitude',
                'developer',
                'location',
                'combined_panels_kmeans_labels',
                'combined_panels_kmeans_labels_count']
        vars.extend(time_invariant_vars)
        vars_2 = []
        for i in time_variant_vars:
            vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        for i in dep_vars:
            vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        vars.extend(vars_2)
        new_df = self.df[vars]
        return new_df

    # -------------------------- independent var of interest ---------------------------------------------
    def convert_df_from_wide_to_long(self, time_variant_vars, time_invariant_vars, dep_vars):
        new_df = self.select_panel_vars(time_invariant_vars, time_variant_vars, dep_vars)
        new_df = new_df.reset_index()
        stub_names = copy.deepcopy(time_variant_vars)
        stub_names.extend(dep_vars)
        new_df = pd.wide_to_long(new_df, stubnames=stub_names, i="index", j="panel", sep='_') # here you can add developer for multiindex output
        new_df = new_df.sort_index()
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   panel_long_df=new_df)
    # -------------------------- Model Selection ---------------------------------------------------------
    """
    https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IVRegressionResults.spec_hausman.html#statsmodels.sandbox.regression.gmm.IVRegressionResults.spec_hausman
    Whether to use fixed effect or random effects, you have to run hausman test. 
    """

    # -------------------------- Regression --------------------------------------------------------------
    """
    # http://www.data-analysis-in-python.org/t_statsmodels.html
    # https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
    # https://bashtage.github.io/linearmodels/doc/panel/models.html
    """
    def correlation_matrix(self, dep_vars, time_variant_vars, time_invariant_vars, the_panel):
        """
        This is for the purpose of checking multicolinearity between independent variables
        """
        vars = []
        vars.extend([i + '_' + the_panel for i in time_variant_vars])
        vars.extend(time_invariant_vars)
        vars.extend([i + '_' + the_panel for i in dep_vars])
        new_df = self.df.copy(deep=True)
        hdf = new_df[vars]
        df_corr = hdf.corr()
        return df_corr

    def regression(self, dep_var, time_variant_vars, time_invariant_vars,
                   cross_section, reg_type, the_panel=None):
        """
        run convert_df_from_wide_to_long first and get self.panel_long_df updated, then run this method
        """
        if cross_section is True:
            x_vars = [i + '_' + the_panel for i in time_variant_vars]
            x_vars.extend(time_invariant_vars)
            new_df = self.df.copy(deep=True)
            independents_df = new_df[x_vars]
            X = sm.add_constant(independents_df)
            dep_var = dep_var + '_' + the_panel
            y = new_df[[dep_var]]
            if reg_type == 'OLS':
                model = sm.OLS(y, X)
                results = model.fit()
                results_robust = results.get_robustcov_results()
                print(results_robust.summary())
                return results_robust
            elif reg_type == 'logit':
                model = sm.Logit(y, X)
                results = model.fit()
                print(results.summary())
                return results
            elif reg_type == 'probit':
                model = sm.Probit(y, X)
                results = model.fit()
                print(results.summary())
                return results
        else: # panel regression models
            x_vars = []
            x_vars.extend(time_variant_vars)
            x_vars.extend(time_invariant_vars)
            new_df = self.panel_long_df.copy(deep=True)
            independents_df = new_df[x_vars]
            X = sm.add_constant(independents_df)
            y = new_df[[dep_var]]
            if reg_type == 'POOLED_OLS':
                model = PooledOLS(y, X)
                # results_robust = results.get_robustcov_results()
                # pooledOLS_res = model.fit(cov_type='clustered', cluster_entity=True)
                # Store values for checking homoskedasticity graphically
                # fittedvals_pooled_OLS = pooledOLS_res.predict().fitted_values
                # residuals_pooled_OLS = pooledOLS_res.resids
            elif reg_type == 'FE': # niche_type will be abosrbed in this model because it is time-invariant
                model = PanelOLS(y, X,
                                 entity_effects=True,
                                 time_effects=True,
                                 drop_absorbed=True)
            elif reg_type == 'RE':
                model = RandomEffects(y, X,)
            results = model.fit()
            print(results)
            return results

    def several_regressions(self, dep_vars, time_variant_vars, time_invariant_vars,
                   cross_section, reg_type, the_panel=None):
        results_dict = dict.fromkeys(dep_vars)
        for dep_var in dep_vars:
            result = self.regression(dep_var, time_variant_vars, time_invariant_vars,
                   cross_section, reg_type, the_panel)
            results_dict[dep_var] = result
        return results_dict
    # ----------------------------------------------------------------------------------------------------
    def print_col_names(self, the_panel):
        vars_in_a_panel = []
        for i in self.df.columns:
            if the_panel in i:
                print(i)
                vars_in_a_panel.append(i)
        return vars_in_a_panel

    def print_unique_value_of_var_panel(self, single_var, the_panel=None):
        if the_panel is not None:
            col_name = single_var + '_' + the_panel
            unique_l = self.df[col_name].unique()
            print(col_name, 'contains', len(unique_l), 'unique values')
            print(unique_l)
            return unique_l
        else:
            col_name = [single_var+'_'+i for i in self.consec_panels]
            d = dict.fromkeys(col_name)
            for j in d.keys():
                unique_l = self.df[j].unique()
                print(j, 'contains', len(unique_l), 'unique values')
                print(unique_l)
                d[j]=unique_l
                print()
            return d

    def cat_var_count(self, cat_var, the_panel=None):
        if the_panel is not None:
            col_name = cat_var + '_' + the_panel
            rd = self.df.groupby(col_name)['count_'+the_panel].count()
            if cat_var == 'minInstalls': # minInstalls should not be sorted by the number of apps in each group, rather by index
                rd = rd.sort_index(ascending=False)
            else:
                rd = rd.sort_values(ascending=False)
            print(rd)
            return rd
        else:
            col_name = [cat_var+'_'+i for i in self.consec_panels]
            df_list = []
            for j in range(len(col_name)):
                rd = self.df.groupby(col_name[j])['count_'+self.consec_panels[j]].count()
                if cat_var == 'minInstalls':
                    rd = rd.sort_index(ascending=False)
                else:
                    rd = rd.sort_values(ascending=False)
                rd = rd.to_frame()
                df_list.append(rd)
            dfn = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
            print(dfn)
            return dfn

    # The all() function returns True if all items in an iterable are true, otherwise it returns False.
    # so if all([False, False, False)] is False, it will return False
    # and if all([False, True, True)] is False, it will return False (INSTEAD of true as expected)
    # all([]) will only return True is all elements are True
    def find_time_variant_rows(self, cat_var):
        df2 = self.select_vars(single_var=cat_var)
        df_time_variant = []
        for index, row in df2.iterrows():
            row_time_variant = []
            for j in range(len(df2.columns) - 1):
                if row[df2.columns[j]] == row[df2.columns[j + 1]]:
                    row_time_variant.append(False)
                else:
                    row_time_variant.append(True)
            if any(row_time_variant) is True:
                df_time_variant.append(True)
            else:
                df_time_variant.append(False)
        time_variant_df = df2[df_time_variant]
        time_variant_appids = time_variant_df.index.tolist()
        return time_variant_df, time_variant_appids

    def change_time_variant_to_invariant(self, cat_var):
        """
        use the last panel as the standard value for time-invariant variables
        """
        time_variant_df, time_variant_appids = self.find_time_variant_rows(cat_var=cat_var)
        col_names = [cat_var + '_' + i for i in self.consec_panels]
        for i in time_variant_appids:
            for j in col_names:
                self.df.at[i, j] = self.df.at[i, cat_var+'_' + self.consec_panels[-1]] # this one intends to change class attributes
        return self.df

    def create_new_dummies_from_cat_var(self, cat_var, time_invariant=False):
        if time_invariant is True:
            self.df = self.change_time_variant_to_invariant(cat_var)
        else:
            pass
        if cat_var == 'genreId':
            df1 = self.select_vars(single_var=cat_var)
            if time_invariant is True:
                df1['genreIdGame'] = df1['genreId_' + self.consec_panels[-1]].apply(lambda x: 1 if 'GAME' in x else 0)
            else:
                for i in self.consec_panels:
                    df1['genreIdGame_'+i] = df1['genreId_'+i].apply(lambda x: 1 if 'GAME' in x else 0)
            dcols = ['genreId_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        elif cat_var == 'contentRating':
            df1 = self.select_vars(single_var=cat_var)
            if time_invariant is True:
                df1['contentRatingEveryone'] = df1['contentRating_' + self.consec_panels[-1]].apply(
                    lambda x: 1 if 'Everyone' in x else 0)
            else:
                for i in self.consec_panels:
                    df1['contentRatingEveryone_'+i] = df1['contentRating_'+i].apply(lambda x: 1 if 'Everyone' in x else 0)
            dcols = ['contentRating_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        elif cat_var == 'minInstalls':
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1['minInstallsTop_'+i] = df1['minInstalls_'+i].apply(lambda x: 1 if x >= 1.000000e+07 else 0)
                df1['minInstallsMiddle_' + i] = df1['minInstalls_' + i].apply(lambda x: 1 if x < 1.000000e+07 and x >= 1.000000e+04 else 0)
                df1['minInstallsBottom_' + i] = df1['minInstalls_' + i].apply(lambda x: 1 if x < 1.000000e+04 else 0)
            dcols = ['minInstalls_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        elif cat_var == 'free':
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1['paidTrue_' + i] = df1['free_' + i].apply(lambda x: 1 if x is False else 0)
            dcols = ['free_' + i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        else:
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1[cat_var + 'True_' + i] = df1[cat_var + '_' + i].apply(lambda x: 1 if x is True else 0)
            dcols = [cat_var + '_' + i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        return regression_analysis(df=self.df, initial_panel=self.initial_panel, consec_panels=self.consec_panels)

    def standardize_continuous_vars(self, con_var, method):
        vars = [con_var + '_' + i for i in self.consec_panels]
        new_df = self.df.copy(deep=True)
        df2 = new_df[vars]
        print('before standardization:')
        for i in df2.columns:
            print(i)
            print(df2[i].describe())
            print()
        if method == 'zscore':
            scaler = preprocessing.StandardScaler()
            df3 = scaler.fit_transform(df2)
            df3 = pd.DataFrame(df3)
            scaled_col_names = ['ZSCORE' + i for i in vars]
            df3.columns = scaled_col_names
            df3.index = df2.index.tolist()
        print('after standardization:')
        for i in df3.columns:
            print(i)
            print(df3[i].describe())
            print()
        self.df = self.df.join(df3, how='inner')
        return regression_analysis(df=self.df, initial_panel=self.initial_panel, consec_panels=self.consec_panels)

    def peek_at_missing(self, **kwargs):
        df1 = self.select_vars(**kwargs)
        null_data = df1[df1.isnull().any(axis=1)]
        return null_data

    def replace_literal_true(self, cat_var): # after checking unique value of cat_var, some cat_var has 'True' instead of True
        cols = [cat_var+'_'+i for i in self.consec_panels]
        for j in cols:
            self.df.loc[self.df[j] == 'True', j] = True
        return self.df

    def check_whether_cat_dummies_are_mutually_exclusive(self, the_panel, **kwargs):
        df1 = self.select_vars(the_panel=the_panel, **kwargs)
        df1['sum_dummies'] = df1.sum(axis=1)
        l = df1.sum_dummies.unique()
        print('The unique value of all dummy columns summation should only be 1 if those dummies are mutually exclusive')
        print(l)
        if len(l) > 1:
            df2 = df1.loc[df1['sum_dummies']!=1]
            return df2
        else:
            return df1

    # ------------------------------------------------------------------------------------------
    def unlist_a_col_containing_list_of_strings(x):
        if x is not None and re.search(r'\[\]+', x):
            s = eval(x)
            if isinstance(s, list):
                s2 = ', '.join(str(ele) for ele in s)
                return s2
        else:
            return x

    def select_vars_for_reg_df(self, cross_section=True):
        pass




