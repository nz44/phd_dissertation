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
                 reg_dict_xy=None,
                 single_panel_df=None,
                 subsample_op_results=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.dep_vars = dep_vars
        self.independent_vars = independent_vars
        self.ssnames = subsample_names
        self.reg_dict = reg_dict
        self.reg_dict_xy = reg_dict_xy
        self.single_panel_df = single_panel_df
        self.subsample_op_results = subsample_op_results

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
                           reg_dict_xy=self.reg_dict_xy,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

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
                           reg_dict_xy=self.reg_dict_xy,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def select_x_y_for_subsample(self, n_niche_scales):
        self.reg_dict_xy = dict.fromkeys(self.reg_dict.keys())
        for name1, content1 in self.reg_dict.items():
            self.reg_dict_xy[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                x_vars = copy.deepcopy(self.independent_vars)
                if name2 == 'full':
                    self.reg_dict_xy['full']['full'] = dict.fromkeys(['NicheDummy', 'NicheScaleDummies'])
                    # here is the full_full_NicheScaleDummy_0 is dropped as baseline group
                    # you cannot run regression with both nichedummy and nichescaledummies together
                    # -------------------------------------------------------------------------------
                    x_vars_full_nd = []
                    x_vars_full_nd.extend(x_vars)
                    x_vars_full_nd.extend(['full_full_NicheDummy', 'PostXfull_full_NicheDummy'])
                    x_vars_full_nd.extend(self.dep_vars)
                    self.reg_dict_xy['full']['full']['NicheDummy'] = df[x_vars_full_nd]
                    # -------------------------------------------------------------------------------
                    niche_scale_cols = ['full_full_NicheScaleDummy_' + str(i) for i in range(1, n_niche_scales)]
                    x_vars_full_sd = ['PostX' + i for i in niche_scale_cols]
                    x_vars_full_sd.extend(niche_scale_cols)
                    x_vars_full_sd.extend(x_vars)
                    x_vars_full_sd.extend(self.dep_vars)
                    self.reg_dict_xy['full']['full']['NicheScaleDummies'] = df[x_vars_full_sd]
                # you can delete this elif block once you've run the may 2021 procedure because sub-sample and minInstallstop dummies are the same
                elif 'minInstalls' in name2:
                    self.reg_dict_xy['minInstalls'][name2] = dict.fromkeys(['NicheDummy'])
                    x_vars_mininstalls = [name1 + '_' + name2 + '_NicheDummy',
                                          'PostX' + name1 + '_' + name2 + '_NicheDummy']
                    x_vars_mininstalls.extend(x_vars)
                    x_vars_mininstalls.extend(self.dep_vars)
                    # since this minInstall is subsample sliced according to
                    remove_minInstalls_dummies = [i for i in x_vars_mininstalls if 'DeMeanedminInstalls' not in i]
                    self.reg_dict_xy['minInstalls'][name2]['NicheDummy'] = df[remove_minInstalls_dummies]
                else:
                    self.reg_dict_xy[name1][name2] = dict.fromkeys(['NicheDummy'])
                    x_vars.extend([name1 + '_' + name2 + '_NicheDummy',
                                  'PostX' + name1 + '_' + name2 + '_NicheDummy'])
                    x_vars.extend(self.dep_vars)
                    unchecked_df = df[x_vars]
                    checked_df = unchecked_df.copy(deep=True)
                    # for some genreId, some column variables has no variation (the same value),
                    # print out them and delete them (if they are not dep vars)
                    for i in unchecked_df.columns:
                        unique_v = unchecked_df[i].unique()
                        if len(unique_v) == 1:
                            print(name1, name2, i, ' contains only 1 unique value ')
                            checked_df.drop(i, axis=1, inplace=True)
                            print('dropped ', i)
                    # additionally, you need to check whether two columns are perfectly correlated, and delete one of them.
                    # this especially could happen to DeMeanedminInstallsTop and DeMeanedminInstallsMiddle
                    if all(x in checked_df.columns for x in ['DeMeanedminInstallsTop', 'DeMeanedminInstallsMiddle']):
                        correlation = checked_df['DeMeanedminInstallsTop'].corr(checked_df['DeMeanedminInstallsMiddle'])
                        if 0.99 <= abs(correlation) <= 1:
                            print(name1, name2, ' DeMeanedminInstallsTop and DeMeanedminInstallsMiddle are perfectly correlated ')
                            checked_df.drop('DeMeanedminInstallsMiddle', axis=1, inplace=True)
                            print(name1, name2,
                                  ' DROPPED DeMeanedminInstallsMiddle to avoid exog full rank error in regressions ')
                    self.reg_dict_xy[name1][name2]['NicheDummy'] = checked_df
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           reg_dict_xy=self.reg_dict_xy,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def slice_single_panel(self, the_panel):
        self.single_panel_df = dict.fromkeys(self.reg_dict_xy.keys())
        for name1, content1 in self.reg_dict_xy.items():
            self.single_panel_df[name1] = dict.fromkeys(content1.keys())
            for name2, content2 in content1.items():
                self.single_panel_df[name1][name2] = dict.fromkeys(content2.keys())
                for name3, df in content2.items():
                    print(name1, name2, name3)
                    print(df.shape)
                    v = self._slice_a_panel_from_long_df(the_panel, df)
                    print(v.shape)
                    self.single_panel_df[name1][name2][name3] = v
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           reg_dict_xy=self.reg_dict_xy,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def _slice_a_panel_from_long_df(self, the_panel, df):
        """
        The function should be run after select_x_y_for_subsample, otherwise you would have too many variables.
        df has multiindex structure, we are slicing the secondary index with the_panel.
        """
        df2 = df.copy(deep=True)
        df3 = df2.reset_index()
        df3 = df3.loc[df3['panel'] == int(the_panel)]
        df3 = df3.set_index('index')
        # delete panel and PostDummy because for a single panel the value are all the same
        cols_drop = []
        for i in df3.columns:
            if 'PostX' in i:
                cols_drop.append(i)
        cols_drop.extend(['panel', 'PostDummy'])
        df3 = df3.drop(cols_drop, axis=1)
        return df3

    def correlation_for_single_panel(self):
        for name1, content1 in self.single_panel_df.items():
            for name2, content2 in content1.items():
                for name3, df in content2.items():
                    dfcorr = df.corr(method='pearson')
                    f_name = name1 + '_' + name2 +  '_' + name3 + '.csv'
                    q = regression.panel_path / 'correlations' / f_name
                    dfcorr.to_csv(q)
                    print(name1, name2, name3, ' correlation matrix exported to csv ')
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           reg_dict_xy=self.reg_dict_xy,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def all_regressions(self, reg_func, xy_df):
        """
        This is run after self.slice_single_panel for cross section
        reg_func is self._cross_section_regression abd xy_df is self.single_panel_df for cross section
        reg_func is self._panel_regression and xy_df is self.reg_dict_xy
        """
        self.subsample_op_results = dict.fromkeys(xy_df.keys())
        for name1, content1 in xy_df.items():
            self.subsample_op_results[name1] = dict.fromkeys(content1.keys())
            for name2, content2 in content1.items():
                self.subsample_op_results[name1][name2] = dict.fromkeys(content2.keys())
                for name3, df in content2.items():
                    self.subsample_op_results[name1][name2][name3] = dict.fromkeys(self.dep_vars)
                    x_vars = [elem for elem in df.columns if elem not in self.dep_vars]
                    for y in self.dep_vars:
                        print('start regressing on ', name1, name2, name3, y)
                        self.subsample_op_results[name1][name2][name3][y] = reg_func(
                                      y_var=y,
                                      x_vars=x_vars,
                                      df=df)
                        print('finished regressing on ', name1, name2, name3, y)
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def _cross_section_regression(self,
                                  y_var,
                                  x_vars,
                                  df):
        """
        https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
        #https://www.statsmodels.org/stable/rlm.html
        https://stackoverflow.com/questions/30553838/getting-statsmodels-to-use-heteroskedasticity-corrected-standard-errors-in-coeff
        source code for HC0, HC1, HC2, and HC3, white and Mackinnon
        https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html
        """
        independents_df = df[x_vars]
        X = sm.add_constant(independents_df)
        y = df[[y_var]]
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC3')
        return results

    def _panel_regression(self,
                   y_var,
                   x_vars,
                   df):
        """
        Internal function
        return a dictionary containing all different type of panel reg results
        I will not run fixed effects model here because they will drop time-invariant variables.
        In addition, I just wanted to check whether for the time variant variables, the demeaned time variant variables
        will have the same coefficient in POOLED OLS as the time variant variables in FE.
        """
        independents_df = df[x_vars]
        X = sm.add_constant(independents_df)
        y = df[[y_var]]
        # https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
        result_dict = dict.fromkeys(['POOLED_OLS', 'RE'])
        for key in result_dict.keys():
            if key == 'POOLED_OLS':
                print('start Pooled_ols regression')
                model = PooledOLS(y, X)
                result_dict['POOLED_OLS'] = model.fit(cov_type='clustered', cluster_entity=True)
            # elif key == 'FE': # niche_type will be abosrbed in this model because it is time-invariant
            #     model = PanelOLS(y, X,
            #                      drop_absorbed=True,
            #                      entity_effects=True,
            #                      time_effects=True)
            #     result_dict['FE'] = model.fit(cov_type = 'clustered', cluster_entity = True)
            else:
                print('start RE regression')
                model = RandomEffects(y, X,)
                result_dict['RE'] = model.fit(cov_type = 'clustered', cluster_entity = True)
        return result_dict

    def put_reg_results_into_pandas(self, reg_folder_name):
        """
        This is a general pandas, not the one put into latex.
        You can trim the pandas later to decide how to put it into latex.
        reg_folder_name is one of cross_section, panel_FE, panel_RE, panel_pooled_OLS
        """
        combined_niche_dummies_dfs = []
        df = pd.DataFrame()
        reg_level_dfs = []
        for name1, content1 in self.subsample_op_results.items():
            for name2, content2 in content1.items():
                for name3, content3 in content2.items():
                    # for OLS results ====================================================================================
                    if reg_folder_name == 'cross_section':
                        reg_level_stats_df = pd.DataFrame(columns=['index', 'nobs', 'rsquared', 'rsquared_adj'])
                        combined_niche_dummies = pd.DataFrame(columns=['index'])
                        for y, result in content3.items():
                            cols_to_add = [y, y + '_pvalue']
                            combined_niche_dummies.loc[:, cols_to_add] = None
                            df, reg_level_stats_df, combined_niche_dummies = self._put_ols_results_into_pandas(
                                name1, name2, name3, y, result,
                                df, reg_level_stats_df, combined_niche_dummies)
                    # for PANEL results ==================================================================================
                    else:
                        reg_level_stats_df = pd.DataFrame(columns=['index'])
                        combined_niche_dummies = pd.DataFrame(columns=['index'])
                        for y, result in content3.items():
                            for panel_reg, panel_res in result.items():
                                cols_to_add = [y + '_' + panel_reg, y + '_' + panel_reg + '_pvalue']
                                combined_niche_dummies.loc[:, cols_to_add] = None
                                col_reg_stats = [i + '_' + panel_reg for i in ['nobs', 'rsquared']]
                                reg_level_stats_df.loc[:, col_reg_stats] = None
                                df, reg_level_stats_df, combined_niche_dummies = self._put_panel_results_into_pandas(
                                    name1, name2, name3, y, panel_reg, panel_res,
                                    df, reg_level_stats_df, combined_niche_dummies)
                    # ------------------------------------------------------------------------------------------
                    f_name = name1 + '_' + name2 + '_' + name3 + '_coefficients' + '.csv'
                    q = regression.panel_path / 'reg_results' / reg_folder_name / f_name
                    df = df.round(3)
                    df.to_csv(q)
                    # ------------------------------------------------------------------------------------------
                    reg_level_dfs.append(reg_level_stats_df)
                    combined_niche_dummies_dfs.append(combined_niche_dummies)
                # After appending for each level 3 name dataframes ================================================
                combined_df = functools.reduce(lambda a, b: pd.concat([a, b]), reg_level_dfs)
                combined_df.set_index('index', inplace=True)
                for i in combined_df.columns:
                    if 'rsquared' in i:
                        combined_df = combined_df.astype(float).round({i: 3})
                f_name = 'regression_level_stats.csv'
                q = regression.panel_path / 'reg_results' / reg_folder_name / f_name
                combined_df.to_csv(q)
                # After appending for each level 3 name dataframes ================================================
                niche_df = functools.reduce(lambda a, b: pd.concat([a, b]), combined_niche_dummies_dfs)
                niche_df.set_index('index', inplace=True)
                niche_df = niche_df.astype(float).round(3)
                f_name = 'NicheDummy_combined.csv'
                q = regression.panel_path / 'reg_results' / reg_folder_name / f_name
                niche_df.to_csv(q)
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def _put_ols_results_into_pandas(self, name1, name2, name3, y, result,
                                     df, reg_level_stats_df, combined_niche_dummies):
        # single regression coefficients and p-values ---------------------------------
        df[y + '_params'] = result.params
        df[y + '_pvalue'] = result.pvalues
        # regression level stats ------------------------------------------------------
        reg_level_stats_df.at[0, 'index'] = name1 + '_' + name2 + '_' + name3 + '_' + y
        reg_level_stats_df.at[0, 'nobs'] = result.nobs
        reg_level_stats_df.at[0, 'rsquared'] = result.rsquared
        reg_level_stats_df.at[0, 'rsquared_adj'] = result.rsquared_adj
        # put all niche dummies into a single table -----------------------------------
        index_list = []
        for i in result.params.index.values:
            if 'Niche' in i:
                index_list.append(i)
        for i in range(len(index_list)):
            combined_niche_dummies.at[i, 'index'] = index_list[i]
            combined_niche_dummies.at[i, y] = result.params.loc[index_list[i]]
            combined_niche_dummies.at[i, y + '_pvalue'] = result.pvalues.loc[index_list[i]]
        return df, reg_level_stats_df, combined_niche_dummies

    def _put_panel_results_into_pandas(self, name1, name2, name3, y, panel_reg, panel_res,
                                       df, reg_level_stats_df, combined_niche_dummies):
        # single panel regression coefficients and p-values -------------------------
        df[y + '_' + panel_reg + '_params'] = panel_res.params
        df[y + '_' + panel_reg + '_pvalue'] = panel_res.pvalues
        # panel regression level stats ------------------------------------------------------
        reg_level_stats_df.at[0, 'index'] = name1 + '_' + name2 + '_' + name3 + '_' + y + '_' + panel_reg
        reg_level_stats_df.at[0, 'nobs' + '_' + panel_reg] = panel_res.nobs
        reg_level_stats_df.at[0, 'rsquared' + '_' + panel_reg] = panel_res.rsquared
        # no adjusted rsquared for panel regressions
        # put all niche dummies into a single table -----------------------------------
        index_list = []
        for i in panel_res.params.index.values:
            if 'Niche' in i:
                index_list.append(i)
        for i in range(len(index_list)):
            combined_niche_dummies.at[i, 'index'] = index_list[i]
            combined_niche_dummies.at[i, y + '_' + panel_reg] = panel_res.params.loc[index_list[i]]
            combined_niche_dummies.at[i, y + '_' + panel_reg + '_pvalue'] = panel_res.pvalues.loc[index_list[i]]
        return df, reg_level_stats_df, combined_niche_dummies

    def convert_csv_to_latex_result_PostXNicheDummy(self, result_type, table_type):
        """
        table_type == table_1:
        include full, minInstalls and developer sub-samples, pooled OLS results with
        post * niche dummy (PostXNicheDummy) for now.
        table_type == table_2:
        include genreId sub-samples, pooled OLS results with
        post * niche dummy (PostXNicheDummy) for now.
        table_type == table_3:
        include full sample, pooled OLS results with
        post * niche scale dummies 1-19 (PostXNicheDummy) for now.
        """
        if result_type == 'panel':
            q = regression.panel_path / 'reg_results' / 'panel' / 'NicheDummy_combined.csv'
        else:
            q = regression.panel_path / 'reg_results' / 'cross_section' / 'NicheDummy_combined.csv'
        df = pd.read_csv(q)
        df.set_index('index', inplace=True)
        # -------------------------------------------------------------------------------
        if result_type == 'panel':
            selected_col_names = []
            substrings = ['POOLED_OLS', 'POOLED_OLS_pvalue']
            for i in df.columns:
                if any([substring in i for substring in substrings]):
                    selected_col_names.append(i)
        else:
            selected_col_names = df.columns.tolist()
        # -------------------------------------------------------------------------------
        selected_row_indices = []
        if table_type == 'table_1':
            if result_type == 'panel':
                substrings = ['PostX', 'NicheDummy']
            else:
                substrings = ['NicheDummy']
            for i in df.index.values:
                if all([substring in i for substring in substrings]) and 'genreId' not in i:
                    selected_row_indices.append(i)
        elif table_type == 'table_2':
            if result_type == 'panel':
                substrings = ['PostX', 'NicheDummy', 'genreId']
            else:
                substrings = ['NicheDummy', 'genreId']
            for i in df.index.values:
                if all([substring in i for substring in substrings]):
                    selected_row_indices.append(i)
        else:
            if result_type == 'panel':
                substrings = ['PostX', 'NicheScaleDummy']
            else:
                substrings = ['NicheScaleDummy']
            for i in df.index.values:
                if all([substring in i for substring in substrings]):
                    selected_row_indices.append(i)
        # -------------------------------------------------------------------------------
        df2 = df.loc[selected_row_indices, selected_col_names]
        return df2

    def convert_csv_to_latex_panel_result_table_2(self):
        pass


########################################################################################################################
    # pooled OLS with individual dummies will be another completely different function.

    # extra code
    def extra_code(self):
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
                lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.01'] if row[
                                                                                    i + '_pvalues<0.01'] != 'not sig at 1%' else str(
                    row[i + '_coef']),
                axis=1)
        for i in ind_vars_sig_at_5_percent:
            ind_vars_sig_at_10_percent.remove(i)
            df9[i + '_coef'] = df9.apply(
                lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.05'] if row[
                                                                                    i + '_pvalues<0.05'] != 'not sig at 5%' else str(
                    row[i + '_coef']),
                axis=1)
        for i in ind_vars_sig_at_10_percent:
            df9[i + '_coef'] = df9.apply(
                lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.1'] if row[
                                                                                   i + '_pvalues<0.1'] != 'not sig at 10%' else str(
                    row[i + '_coef']),
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



