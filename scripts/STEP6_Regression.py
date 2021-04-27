import pandas as pd
import copy
from pathlib import Path
import pickle
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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


class regression():
    panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__')
    reg_table_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/reg_results_tables')
    descriptive_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/tables')
    descriptive_stats_graphs = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/graphs')

    def __init__(self,
                 initial_panel,
                 all_panels,
                 dep_vars,
                 independent_vars,
                 subsample_names=None,
                 reg_dict=None,
                 subsample_reg_results=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.dep_vars = dep_vars
        self.independent_vars = independent_vars
        self.ssnames = subsample_names
        self.reg_dict = reg_dict
        self.subsample_reg_results = subsample_reg_results

    def open_long_df_dict(self):
        f_name = self.initial_panel + '_converted_long_table.pickle'
        q = regression.panel_path / 'converted_long_tables' / f_name
        with open(q, 'rb') as f:
            self.reg_dict = pickle.load(f)
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           subsample_reg_results=self.subsample_reg_results)

    def add_subsample_names(self):
        self.ssnames = dict.fromkeys(self.reg_dict.keys())
        for key, content in self.reg_dict.items():
            self.ssnames[key] = list(content.keys())
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           subsample_reg_results=self.subsample_reg_results)

    def regression_for_each_subsample(self, n_niche_scales):
        self.subsample_reg_results = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.reg_dict.items():
            self.subsample_reg_results[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                self.subsample_reg_results[name1][name2] = dict.fromkeys(self.dep_vars)
                x_vars = self.independent_vars
                niche_dummy_col = name1 + '_' + name2 + '_NicheDummy'
                x_vars.append(niche_dummy_col)
                niche_post_col = 'PostX' + niche_dummy_col
                x_vars.append(niche_post_col)
                if name2 == 'full':
                    niche_scale_cols = ['full_full_NicheScaleDummy_' + str(i) for i in range(n_niche_scales)]
                    x_vars.extend(niche_scale_cols)
                    niche_scale_post_cols = ['PostX' + i for i in niche_scale_cols]
                    x_vars.extend(niche_scale_post_cols)
                for y_var in self.dep_vars:
                    reg_types = ['POOLED_OLS', 'FE', 'RE']
                    self.subsample_reg_results[name1][name2][y_var] = dict.fromkeys(reg_types)
                    for reg_type in reg_types:
                        self.subsample_reg_results[name1][name2][y_var][reg_type] = self.regression(
                            y_var=y_var,
                            x_vars=x_vars,
                            df=df,
                            reg_type=reg_type)
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           subsample_reg_results=self.subsample_reg_results)

    def regression(self,
                   y_var,
                   x_vars,
                   df,
                   reg_type):
        """
        Internal function
        """
        independents_df = df[x_vars]
        X = sm.add_constant(independents_df)
        y = df[[y_var]]
        # https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
        if reg_type == 'POOLED_OLS':
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


##############################################################################################
    def several_regressions(self, dep_vars, time_variant_vars, time_invariant_vars, cross_section, reg_types, the_panel=None):
        results_dict = dict.fromkeys(reg_types, dict.fromkeys(dep_vars))
        for reg_type in reg_types:
            for dep_var in dep_vars:
                result = self.regression(
                    dep_var,
                    time_variant_vars,
                    time_invariant_vars,
                    cross_section,
                    reg_type,
                    the_panel)
                results_dict[reg_type][dep_var] = result
        return results_dict

    def compare_several_panel_reg_results(self, results_dict, dep_vars):
        """
        https://bashtage.github.io/linearmodels/panel/examples/examples.html
        results_dict is the output of several_regressions
        """
        panels_results = dict.fromkeys(dep_vars, {})
        panels_compare = dict.fromkeys(dep_vars)
        for reg_type, results in results_dict.items():
            if reg_type in ['FE', 'RE', 'POOLED_OLS', 'PPOLED_OLS_with_individual_dummies']:
                for dep_var in results.keys():
                    panels_results[dep_var][reg_type] = results[dep_var]
                    compared_results = compare(panels_results[dep_var])
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
        # round the entire dataframe
        df7 = df6.T
        for i in df7.columns:
            if i not in ['_cov_type', 'nobs']:
                df7[i] = df7[i].astype(float).round(decimals=3)
            elif i == 'nobs':
                df7[i] = df7[i].astype(int)
        df8 = df7.T
        return df8

    def customize_reg_results_pandas_before_output_latex(self,
                                             df,
                                             selective_dep_vars,
                                             selective_reg_types,
                                             p_value_as_asterisk=False):
        """
        :param df6: the output of self.compile_several_reg_results_into_pandas(panels_compare)
        :return: df8: the customized version of pandas that will ultimately transformed into latex
        """
        cols_to_keep = [i + '_' + j for j in selective_reg_types for i in selective_dep_vars]
        df7 = df.copy(deep=True)
        df8 = df7[cols_to_keep]
        if p_value_as_asterisk is True:
            df9 = df8.T
        for j in df9.columns:
            if '_pvalues' in j:
                df9[j + '<0.01'] = df9[j].apply(lambda x: '***' if x < 0.01 else 'not sig at 1%')
                df9[j + '<0.05'] = df9[j].apply(lambda x: '**' if x < 0.05 else 'not sig at 5%')
                df9[j + '<0.1'] = df9[j].apply(lambda x: '*' if x < 0.1 else 'not sig at 10%')
        ind_vars_sig_at_1_percent = []
        ind_vars_sig_at_5_percent = []
        ind_vars_sig_at_10_percent = []
        for j in df9.columns:
            if '***' in df9[j].values:
                ind_var = j.rstrip('pvalues<0.01').rstrip('_')
                ind_vars_sig_at_1_percent.append(ind_var)
            elif '**' in df9[j].values:
                ind_var = j.rstrip('pvalues<0.05').rstrip('_')
                ind_vars_sig_at_5_percent.append(ind_var)
            elif '*' in df9[j].values:
                ind_var = j.rstrip('pvalues<0.1').rstrip('_')
                ind_vars_sig_at_10_percent.append(ind_var)
        for i in ind_vars_sig_at_1_percent:
            ind_vars_sig_at_5_percent.remove(i)
            ind_vars_sig_at_10_percent.remove(i)
            df9[i + '_coef'] = df9.apply(
                        lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.01'] if row[i + '_pvalues<0.01'] != 'not sig at 1%' else str(row[i + '_coef']),
                         axis=1)
        for i in ind_vars_sig_at_5_percent:
            ind_vars_sig_at_10_percent.remove(i)
            df9[i + '_coef'] = df9.apply(
                        lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.05'] if row[i + '_pvalues<0.05'] != 'not sig at 5%' else str(row[i + '_coef']),
                        axis=1)
        for i in ind_vars_sig_at_10_percent:
            df9[i + '_coef'] = df9.apply(
                        lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.1'] if row[i + '_pvalues<0.1'] != 'not sig at 10%' else str(row[i + '_coef']),
                        axis=1)
        cols_to_keep = [i for i in df9.columns if '_coef' in i]
        cols_to_keep.extend(['F stat', 'P-value', 'rsquared', 'nobs', '_cov_type'])
        df10 = df9[cols_to_keep]
        df10 = df10.T
        return df10

    def output_reg_results_pandas_to_latex(self, df, the_reg_type):
        """
        :param df: the output of self.customize_pandas_before_output_latex()
        :return: df10:
        """
        df2 = df.copy(deep=True)
        # ------------ rename columns ---------------------------------------------
        for i in df2.columns:
            z = i.rstrip(the_reg_type).rstrip('_')
            for j in regression_analysis.var_latex_map.keys():
                if j == z:
                    df2.rename(columns={i: regression_analysis.var_latex_map[j]}, inplace=True)
        # ------------ prepare column for multiindex rows --------------------------
        df2 = df2.reset_index()
        def set_row_level_0(x):
            if x in ['niche_app_coef']:
                return '\makecell[l]{Niche \\\ Indicators}'
            elif x in ['const_coef', 'genreIdGame_coef', 'contentRatingAdult_coef', 'DaysSinceReleased_coef']:
                return '\makecell[l]{Time-invariant \\\ Variables}'
            elif x in ['DeMeanedscore_coef', 'DeMeanedZSCOREreviews_coef', 'DeMeanedminInstallsTop_coef', 'DeMeanedminInstallsMiddle_coef']:
                return '\makecell[l]{Time-variant \\\ Variables}'
            else:
                return '\makecell[l]{Regression \\\ Statistics}'
        df2['Variable Groups'] = df2['index'].apply(lambda x: set_row_level_0(x))
        # ------------ rename rows -------------------------------------------------
        def set_row_level_1(x):
            if '_coef' in x:
                x = x.rstrip('coef').rstrip('_')
            for i in regression_analysis.var_latex_map.keys():
                if i == x:
                    return regression_analysis.var_latex_map[i]
        df2['Independent Variables'] = df2['index'].apply(lambda x: set_row_level_1(x))
        # manually set the order of rows you want to present in final latex table
        def set_row_order_level_0(x):
            if x == '\makecell[l]{Niche \\\ Indicators}':
                return 0
            elif x == '\makecell[l]{Time-invariant \\\ Variables}':
                return 1
            elif x == '\makecell[l]{Time-variant \\\ Variables}':
                return 2
            elif x == '\makecell[l]{Regression \\\ Statistics}':
                return 3
        df2['row_order_level_0'] = df2['Variable Groups'].apply(lambda x: set_row_order_level_0(x))
        df2.sort_values(by='row_order_level_0', inplace=True)
        df2.set_index(['Variable Groups', 'Independent Variables'], inplace=True)
        df2.drop(['index', 'row_order_level_0'], axis=1, inplace=True)
        df2.columns = pd.MultiIndex.from_product([['pooled OLS with demeaned time-variant variables'],
                                                  df2.columns.tolist()])
        # -------------------- save and output to latex -------------------------------
        filename = self.initial_panel + '_POOLED_OLS_demeaned.tex'
        df3 = df2.to_latex(buf=regression_analysis.reg_table_path / filename,
                           multirow=True,
                           multicolumn=True,
                           caption=('Regression Results'),
                           position='h!',
                           label='table:2',
                           escape=False)
        return df2



