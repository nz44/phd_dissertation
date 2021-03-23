import pandas as pd
import copy
from pathlib import Path
import pickle
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
from linearmodels.panel import compare
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
    reg_table_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/reg_results_tables')
    descriptive_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats')

    def __init__(self,
                 df,
                 initial_panel,
                 consec_panels,
                 dep_var=None,
                 independent_vars=None,
                 panel_long_df=None,
                 individual_dummies_df=None,
                 descriptive_stats_tables=None,
                 continuous_vars=None,
                 dummy_vars=None,
                 categorical_vars=None,
                 time_invariant_dummies=None,
                 several_reg_results_pandas=None):
        self.df = df # df is the output of combine_imputed_deleted_missing_with_text_labels
        self.initial_panel = initial_panel
        self.consec_panels = consec_panels
        self.dep_var = dep_var
        self.independent_vars = independent_vars
        self.panel_long_df = panel_long_df
        self.i_dummies_df = individual_dummies_df
        self.descriptive_stats_tables = descriptive_stats_tables
        self.con_vars = continuous_vars
        self.dum_vars = dummy_vars
        self.cat_vars = categorical_vars
        self.tiv_dums = time_invariant_dummies
        self.several_reg_results = several_reg_results_pandas

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
        new_df = self.df.copy(deep=True)
        selected_df = new_df[selected_cols]
        return selected_df

    def select_panel_vars(self, time_invariant_vars, time_variant_vars, dep_vars, demaned_time_variant_vars=None):
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
        if demaned_time_variant_vars is not None:
            for i in demaned_time_variant_vars:
                vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        for i in dep_vars:
            vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        vars.extend(vars_2)
        new_df = self.df.copy(deep=True)
        new_df = new_df[vars]
        return new_df

    # -------------------------- independent var of interest ---------------------------------------------
    def convert_df_from_wide_to_long(self, time_variant_vars, time_invariant_vars, dep_vars, demaned_time_variant_vars=None):
        new_df = self.select_panel_vars(time_invariant_vars, time_variant_vars, dep_vars, demaned_time_variant_vars)
        new_df = new_df.reset_index()
        stub_names = copy.deepcopy(time_variant_vars)
        stub_names.extend(dep_vars)
        if demaned_time_variant_vars is not None:
            stub_names.extend(demaned_time_variant_vars)
        new_df = pd.wide_to_long(new_df, stubnames=stub_names, i="index", j="panel", sep='_') # here you can add developer for multiindex output
        new_df = new_df.sort_index()
        self.panel_long_df = new_df
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   panel_long_df=self.panel_long_df,
                                   individual_dummies_df=self.i_dummies_df)

    def combine_individual_dummies_to_long_panel(self):
        """
        run after self.create_individual_app_dummies()
        """
        dfl = self.panel_long_df.copy(deep=True)
        dfr = self.i_dummies_df.copy(deep=True)
        df = dfl.join(dfr, how='left')
        self.panel_long_df = df
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   panel_long_df=self.panel_long_df,
                                   individual_dummies_df=self.i_dummies_df)
    # ===================== check correlation ===========================================================
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

    # ----------------------------------------------------------------------------------------------------
    """
    Mar 21, 2021
    I decide to find the correlation between a varibale with its lags, since I cannot use FE due to time-invariant niche-app,
    (even if I demean other variable, I cannot ignore the fact that if you do not demean niche-app, how can you subtract out the 
    time invariant effect). 
    https://www.statsmodels.org/stable/generated/statsmodels.sandbox.regression.gmm.IVRegressionResults.spec_hausman.html#statsmodels.sandbox.regression.gmm.IVRegressionResults.spec_hausman
    Whether to use fixed effect or random effects, you have to run hausman test. 
    So I might include interval variables in a regular OLS.
    """
    def time_series_corr_single_var(self, var):
        vars = [var + '_' + i for i in self.consec_panels]


    # ========================= descriptive statistics ==================================================
    def add_sum_row(self, df):
        sum_row = df.sum(axis=0)
        sum_row = sum_row.to_frame().T
        sum_row.index = ['sum']
        df = pd.concat([df, sum_row], join="inner")
        return df

    def add_sum_col(self, df):
        sum_row = df.sum(axis=1)
        df['sum'] = sum_row
        return df

    def descriptive_stats_of_reg_vars_for_single_panel(self,
                                                       continuous_vars,
                                                       dummy_vars,
                                                       cat_vars,
                                                       time_invar_dum,
                                                       the_panel,
                                                       add_sum_row_col=True):
        """
        This is must be run after self.create_new_dummies_from_cat_var to get updated self.df
        """
        con_vars = [i + '_' + the_panel for i in continuous_vars]
        dum_vars = [i + '_' + the_panel for i in dummy_vars]
        cat_vars = [i + '_' + the_panel for i in cat_vars]

        dum_vars.extend(time_invar_dum)
        new_df = self.df.copy(deep=True)
        con_vars_df = new_df[con_vars]
        dum_vars_df = new_df[dum_vars]
        cat_vars_df = new_df[cat_vars]
        # ----- Continuous Variables Summary Stats ---------------------------------------
        con_vars_sum_stats = con_vars_df.agg(['mean', 'std', 'min', 'median', 'max', 'count'], axis=0)
        # ----- Dummy Variables Count ----------------------------------------------------
        dum_stats_dfs = []
        for i in dum_vars:
            dum_vars_df['Count'+i] = 0
            df = dum_vars_df[[i, 'Count'+i]].groupby(i).count()
            dum_stats_dfs.append(df)
        dum_vars_sum_stats = functools.reduce(lambda a, b: a.join(b, how='inner'), dum_stats_dfs)
        if add_sum_row_col is True:
            dum_vars_sum_stats = self.add_sum_row(dum_vars_sum_stats)
        # ---- Categorical Variables Count -----------------------------------------------
        # ----- Dummy by Dummy and Dummy by Category --------------------------------------
        cat_stats_dict = dict.fromkeys(cat_vars)
        for i in cat_vars:
            cat_vars_df['Count'+i] = 0
            df = cat_vars_df[[i, 'Count'+i]].groupby(i).count()
            if 'minInstalls' in i:
                df.sort_index(inplace=True)
            else:
                df.sort_values(by='Count'+i, ascending=False, inplace=True)
            if add_sum_row_col is True:
                df = self.add_sum_row(df)
            cat_stats_dict[i] = df
        # -------------------------------------------
        dummy_cat_dfs = []
        for i in dum_vars:
            sub_dummy_cat_dfs = []
            for j in cat_vars:
                df = new_df[[i, j]]
                df2 = pd.crosstab(df[i], df[j])
                df2.columns = [str(c) + '_' + the_panel for c in df2.columns]
                sub_dummy_cat_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), sub_dummy_cat_dfs)
            df3.index = [i + '_' + str(j) for j in df3.index]
            dummy_cat_dfs.append(df3)
        dummy_cat_cocat_df = functools.reduce(lambda a, b: pd.concat([a,b], join='inner'), dummy_cat_dfs)
        if add_sum_row_col is True:
            dummy_cat_cocat_df = self.add_sum_col(dummy_cat_cocat_df)
        # -------------------------------------------
        # first two categorical variables cross tab
        i, j = cat_vars[0], cat_vars[1]
        df = new_df[[i, j]]
        cc_df = pd.crosstab(df[i], df[j])
        cc_df.columns = [str(c) + '_' + the_panel for c in cc_df.columns]
        cc_df.index = [str(c) + '_' + the_panel for c in cc_df.index]
        if add_sum_row_col is True:
            cc_df = self.add_sum_row(cc_df)
            cc_df = self.add_sum_col(cc_df)
        # ----- Continuous Variables by Dummy or Category --------------------------------
        # -------------------------------------
        continuous_by_dummies = dict.fromkeys(con_vars)
        for i in con_vars:
            sub_groupby_dfs = []
            for j in dum_vars:
                df = new_df[[i, j]]
                agg_func_math = {i:['mean', 'std', 'min', 'median', 'max', 'count']}
                df2 = df.groupby([j]).agg(agg_func_math, axis=0)
                df2.columns = ['mean', 'std', 'min', 'median', 'max', 'count']
                sub_groupby_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: pd.concat([a, b], join='inner'), sub_groupby_dfs)
            df3.index = [j + '_' + str(z) for z in df3.index]
            continuous_by_dummies[i] = df3
        # --------------------------------------
        groupby_cat_dfs = dict.fromkeys(cat_vars)
        for i in cat_vars:
            sub_groupby_dfs = []
            for j in con_vars:
                df = new_df[[i, j]]
                agg_func_math = {j:['mean', 'std', 'min', 'median', 'max', 'count']}
                df2 = df.groupby([i]).agg(agg_func_math, axis=0)
                sub_groupby_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), sub_groupby_dfs)
            df3.index = [str(c) + '_' + the_panel for c in df3.index]
            groupby_cat_dfs[i] = df3
        # ----- Update Instance Attributes ------------------------------------------------
        self.descriptive_stats_tables = {'continuous_vars_stats': con_vars_sum_stats,
                                         'dummy_vars_stats': dum_vars_sum_stats,
                                         'categorical_vars_count': cat_stats_dict,
                                         'crosstab_dummy_categorical_vars': dummy_cat_cocat_df,
                                         'crosstab_two_categorical_vars': cc_df,
                                         'continuous_vars_by_dummies': continuous_by_dummies,
                                         'continuous_vars_by_categorical': groupby_cat_dfs}
        return regression_analysis(df=self.df,
                                   panel_long_df=self.panel_long_df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   descriptive_stats_tables=self.descriptive_stats_tables,
                                   continuous_vars=con_vars,
                                   dummy_vars=dum_vars,
                                   categorical_vars=cat_vars)

    def export_single_descriptive_table_to_latex(self, df, colname_map, key, k2=None):
        # ------------ change column and row names ------------------
        df2 = df.copy(deep=True)
        for varname in colname_map.keys():
            for col in df2.columns:
                if col == 'niche_app':
                    col = 'nicheApp'
                elif col == 'Countniche_app':
                    col = 'CountnicheApp'
                else:
                    col2 = col.split('_')
                    col2 = col2[0]
                if varname == col2:
                    df2.rename(columns={col: colname_map[varname]}, inplace=True)
                elif varname in col2:
                    df2.rename(columns={col: colname_map[varname]}, inplace=True)
        if k2 is None:
            filename = key + '_latex.tex'
        else:
            filename = key + '_' + k2 + '_latex.tex'
        df2.to_latex(
            buf=regression_analysis.descriptive_stats_path / filename,
            columns=None,
            col_space=None,
            header=True,
            index=True,
            na_rep='NaN',
            formatters=None,
            float_format="%.2f",
            sparsify=None,
            index_names=True,
            bold_rows=False,
            column_format=None,
            longtable=None,
            escape=None,
            encoding=None,
            decimal='.',
            multicolumn=None,
            multicolumn_format=None,
            multirow=None,
            caption=None,
            label=None,
            position=None)
        return df2

    def export_descriptive_stats_to_latex(self, colname_map):
        stats_dict = copy.deepcopy(self.descriptive_stats_tables)
        for key, item in stats_dict.items():
            if isinstance(item, pd.DataFrame):
                self.export_single_descriptive_table_to_latex(df=item,
                                                              colname_map=colname_map,
                                                              key=key)
            else:
                for k2, item2 in item.items():
                    self.export_single_descriptive_table_to_latex(df=item2,
                                                                  colname_map=colname_map,
                                                                  key=key,
                                                                  k2=k2)

    # ###############################################################################################
    # -------------------------- Regression --------------------------------------------------------------
    """
    # http://www.data-analysis-in-python.org/t_statsmodels.html
    # https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
    # https://bashtage.github.io/linearmodels/doc/panel/models.html
    """

    # ###############################################################################################
    def regression(self, dep_var, time_variant_vars, time_invariant_vars,
                   cross_section, reg_type, the_panel=None):
        """
        run convert_df_from_wide_to_long first and get self.panel_long_df updated, then run this method
        https://bashtage.github.io/linearmodels/doc/panel/models.html
        I have observed that using demeaned time variant independent variables in POOLED_OLS generats the same
        coefficient estimates (for time variant) as the ones from FE model with un-demeaned data.
        The dependent variable is better to interpret as staying un-demeaned.
        The coefficient of time invariant variables are then interpreted as when all other time variant variables are
        set to their group specific means.
        The weird thing, why does niche_app still get appear in FE model？
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
                results = results.get_robustcov_results()
            elif reg_type == 'logit':
                model = sm.Logit(y, X)
                results = model.fit()
            elif reg_type == 'probit':
                model = sm.Probit(y, X)
                results = model.fit()
        else: # panel regression models
            x_vars = []
            x_vars.extend(time_variant_vars)
            x_vars.extend(time_invariant_vars)
            if reg_type == 'PPOLED_OLS_with_individual_dummies':
                x_vars.extend(self.i_dummies_df.columns)
            new_df = self.panel_long_df.copy(deep=True)
            independents_df = new_df[x_vars]
            X = sm.add_constant(independents_df)
            y = new_df[[dep_var]]
            # https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
            if reg_type == 'POOLED_OLS':
                model = PooledOLS(y, X)
                results = model.fit(cov_type='clustered', cluster_entity=True)
            elif reg_type == 'PPOLED_OLS_with_individual_dummies':
                model = PooledOLS(y, X)
                results = model.fit(cov_type='clustered', cluster_entity=True)
            elif reg_type == 'FE': # niche_type will be abosrbed in this model because it is time-invariant
                model = PanelOLS(y, X,
                                 drop_absorbed=True,
                                 entity_effects=True,
                                 time_effects=True)
                results = model.fit(cov_type = 'clustered', cluster_entity = True)
            elif reg_type == 'RE':
                model = RandomEffects(y, X,)
                results = model.fit(cov_type = 'clustered', cluster_entity = True)
        return results

    def several_regressions(self, dep_vars, time_variant_vars, time_invariant_vars, cross_section, reg_types, the_panel=None):
        results_dict_k_reg_type = dict.fromkeys(reg_types, dict.fromkeys(dep_vars))
        results_dict_k_dep_var = dict.fromkeys(dep_vars, dict.fromkeys(reg_types))
        for reg_type in reg_types:
            for dep_var in dep_vars:
                result = self.regression(
                    dep_var,
                    time_variant_vars,
                    time_invariant_vars,
                    cross_section,
                    reg_type,
                    the_panel)
                results_dict_k_reg_type[reg_type][dep_var] = result
                results_dict_k_dep_var[dep_var][reg_type] = result
        return results_dict_k_reg_type, results_dict_k_dep_var

    def compare_several_panel_reg_results(self, results_dict):
        """
        https://bashtage.github.io/linearmodels/panel/examples/examples.html
        results_dict is the output of several_regressions
        """
        panels_compare = {}
        for dep_var, results in results_dict[1].items():
            panels_regs = {}
            for reg_type in results.keys():
                if reg_type in ['FE', 'RE', 'POOLED_OLS', 'PPOLED_OLS_with_individual_dummies']:
                    panels_regs[reg_type] = results[reg_type]
            compared_results = compare(panels_regs)
            print(compared_results)
            panels_compare[dep_var] = compared_results
        return panels_compare

    def compile_several_reg_results_into_pandas(self, panels_compare):
        """
        panel_compare is the output of self.compare_several_panel_reg_results(self, results_dict)
        """
        def change_index_name(df, end_fix):
            dfn = df.copy(deep=True)
            for i in dfn.index:
                dfn.rename(index={i: i+'_'+end_fix}, inplace=True)
            return dfn
        df5 = []
        for dep_var, compare in panels_compare.items():
            df_params = change_index_name(compare.params, end_fix='coef')
            df_tstats = change_index_name(compare.tstats, end_fix='tstats')
            df_pvalues = change_index_name(compare.pvalues, end_fix='pvalues')
            df1 = [df_params, df_tstats, df_pvalues]
            df2 = functools.reduce(lambda a, b: pd.concat([a, b], axis=0), df1)
            df3 = [df2,
                   compare.f_statistic.T,
                   compare.rsquared.to_frame().T,
                   compare.nobs.to_frame().T,
                   compare.cov_estimator.to_frame().T]
            df4 = functools.reduce(lambda a, b: pd.concat([a, b], axis=0), df3)
            for i in df4.columns:
                df4.rename(columns={i: dep_var+'_'+i}, inplace=True)
            df5.append(df4)
        df6 = functools.reduce(lambda a, b: a.join(b, how='inner'), df5)
        return df6

    def compile_several_reg_pandas_to_latex(self, cross_section, the_panel):
        pass
        # if cross_section is True:
        #     filename = self.initial_panel + '_' + the_panel + '_' + reg_type + '_coeffs.pickle'
        # else:
        #     filename = self.initial_panel + '_panel_' + reg_type + '_coeffs.pickle'
        # q = regression_analysis.reg_table_path / filename
        # pickle.dump(coef_df, open(q, 'wb'))

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
                df1['contentRatingAdult'] = df1['contentRating_' + self.consec_panels[-1]].apply(
                    lambda x: 0 if 'Everyone' in x else 1)
            else:
                for i in self.consec_panels:
                    df1['contentRatingAdult_'+i] = df1['contentRating_'+i].apply(lambda x: 0 if 'Everyone' in x else 1)
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
        return regression_analysis(df=self.df,
                                   panel_long_df=self.panel_long_df,
                                   individual_dummies_df=self.i_dummies_df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels)

    def create_individual_app_dummies(self):
        df = self.df.copy(deep=True)
        df.reset_index(inplace=True)
        df['appId'] = df['index']
        df.set_index('index', inplace=True)
        dummies = df[['appId']]
        dummies = pd.get_dummies(dummies, columns=['appId'], drop_first=True)
        self.i_dummies_df = dummies
        return regression_analysis(df=self.df,
                                   panel_long_df=self.panel_long_df,
                                   individual_dummies_df=self.i_dummies_df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels)

    def create_demean_time_variant_vars(self, time_variant_vars):
        """
        Because individual dummies regression takes too much time, I decide use this for FE, so that I could also include time invariant variables.
        """
        df = self.df.copy(deep=True)
        dfs = []
        for i in time_variant_vars:
            ts_i = [i + '_' + p for p in self.consec_panels]
            sub_df = df[ts_i]
            sub_df['PanelMean'+i] = sub_df.mean(axis=1)
            for p in self.consec_panels:
                sub_df['DeMeaned'+i+'_'+p] = sub_df[i+'_'+p] - sub_df['PanelMean'+i]
            ts_idm = ['DeMeaned' + i + '_' + p for p in self.consec_panels]
            dfs.append(sub_df[ts_idm])
        df_new = functools.reduce(lambda a, b: a.join(b, how='inner'), dfs)
        self.df = self.df.join(df_new, how='inner')
        return regression_analysis(df=self.df,
                                   panel_long_df=self.panel_long_df,
                                   individual_dummies_df=self.i_dummies_df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels)

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
        return regression_analysis(df=self.df,
                                   panel_long_df=self.panel_long_df,
                                   individual_dummies_df=self.i_dummies_df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels)

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




