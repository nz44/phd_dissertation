import pandas as pd
import copy
from pathlib import Path
import pickle
pd.set_option('display.max_colwidth', 100)
pd.options.display.max_rows = 999
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
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
import re
today = datetime.today()
yearmonth = today.strftime("%Y%m")

class cross_section_reg():
    """
    2021/06/26:
    I will start a new map, use one class for cross section only and another class for panel regression
    The parallel trend (beta) graph will be embedded in the cross section class.
    """
    panel_essay_2_3_common_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_3_common_panels___')
    panel_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_panels___')
    panel_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_3_panels___')
    reg_table_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/reg_results_tables')
    reg_table_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/reg_results_tables')
    des_stats_tables_essay_2 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/descriptive_stats/tables')
    des_stats_tables_essay_3 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/descriptive_stats/tables')
    des_stats_graphs_essay_2 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/descriptive_stats/graphs')
    des_stats_graphs_essay_3 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/descriptive_stats/graphs')
    graph_ylabel_dict = {'containsAdsTrue': 'ContainsAds',
                         'offersIAPTrue': 'OffersIAP',
                         'paidTrue': 'Paid',
                         'Imputedprice': 'Price'}
    graph_subsample_title_dict = {'Leaders full': 'Market Leaders Full Sample',
                                    'Leaders category_GAME': 'Market Leaders Game Apps',
                                    'Leaders category_BUSINESS': 'Market Leaders Business Apps',
                                    'Leaders category_SOCIAL': 'Market Leaders Social Apps',
                                    'Leaders category_LIFESTYLE': 'Market Leaders Lifestyle Apps',
                                    'Leaders category_MEDICAL': 'Market Leaders Medical Apps',
                                  'Non-leaders full': 'Market Followers Full Sample',
                                  'Non-leaders category_GAME': 'Market Followers Game Apps',
                                  'Non-leaders category_BUSINESS': 'Market Followers Business Apps',
                                  'Non-leaders category_SOCIAL': 'Market Followers Social Apps',
                                  'Non-leaders category_LIFESTYLE': 'Market Followers Lifestyle Apps',
                                  'Non-leaders category_MEDICAL': 'Market Followers Medical Apps'}
    def __init__(self,
                 initial_panel,
                 all_panels,
                 df=None,
                 subsample_names=None,
                 descriptive_stats_tables=None,
                 several_reg_results_pandas=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.ssnames = subsample_names
        self.descriptive_stats_tables = descriptive_stats_tables
        self.several_reg_results = several_reg_results_pandas

    def open_cross_section_reg_df(self):
        f_name = self.initial_panel + '_cross_section_df.pickle'
        q = cross_section_reg.panel_essay_2_3_common_path / 'cross_section_dfs' / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return cross_section_reg(
                            initial_panel=self.initial_panel,
                            all_panels=self.all_panels,
                            df=self.df,
                            subsample_names=self.ssnames)

    def _open_predicted_labels_dict(self):
        f_name = self.initial_panel + '_predicted_labels_dict.pickle'
        q = cross_section_reg.panel_essay_2_3_common_path / 'predicted_text_labels' / f_name
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    def create_subsample_name_dict(self):
        d = self._open_predicted_labels_dict()
        self.ssnames = dict.fromkeys(d.keys())
        for name1, content in d.items():
            self.ssnames[name1] = list(content.keys())
        return cross_section_reg(
                            initial_panel=self.initial_panel,
                            all_panels=self.all_panels,
                            df=self.df,
                            subsample_names=self.ssnames)
# ===================== graph parallel trend ========================================================
# 1. Use minInstalls (as a continous var) to graph parallel trend, because covid demand shock affect minInstalls,
# then in turn it affects pricing variables
    def df_parallel_trend_minInstalls(self):

    def graph_parallel_trend(self):
        data = self.reg_dict_xy
        ls = []
        mdf = []
        for name1, content1 in data.items():
            for name2, content2 in content1.items():
                for nichetype, df in content2.items():
                    df2 = df.copy(deep=True)
                    if nichetype != 'NicheScaleDummies':
                        NicheDummy = name1 + '_' + name2 + '_' + nichetype
                        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8.5))
                        fig.tight_layout(pad=4)
                        for i in range(len(self.dep_vars)):
                            df3 = df2[[self.dep_vars[i], NicheDummy]].reset_index()
                            df3['panel'] = pd.to_datetime(df3['panel'], format='%Y%m')
                            ls.append(df3)
                            # groupby nichedummy and return the mean of the dependent variable
                            df4 = df3.groupby(['panel', NicheDummy])[self.dep_vars[i]].mean() \
                                .unstack(level=-1) \
                                .rename(columns={0: 'Broad', 1: 'Niche'})
                            mdf.append(df4)
                            ax.flat[i].plot(df4.index.values, df4['Broad'], marker='o', label='Broad')
                            ax.flat[i].plot(df4.index.values, df4['Niche'], marker='o', label='Niche')
                            ax.flat[i].legend(loc='lower right',
                                              title=None)
                            ax.flat[i].set_ylabel(regression.graph_ylabel_dict[self.dep_vars[i]])
                            ax.flat[i].set_xlabel('Time')
                            ax.flat[i].grid(axis='y')
                            # add vertical line for DiD (stay-at-home order, most states issued in mid March, some in early April.)
                            # https://www.nytimes.com/interactive/2020/us/coronavirus-stay-at-home-order.html
                            # plt.axvline(datetime(2020, 3, 15))
                            ax.flat[i].axvline(x=datetime(2020, 3, 15), linewidth=2, color='red')
                            # ax.flat[i].text(datetime(2020, 3, 15), -.05, 'stay \nhome \norder', color='red',
                            #                 transform=ax.flat[i].get_xaxis_transform(),
                            #                 ha='center', va='top')
                            # You cannot use ax rorate ticks because the time will become random numbers
                            # ax.flat[i].set_xticklabels(ax.flat[i].get_xticks(), rotation=30)
                            plt.setp(ax.flat[i].get_xticklabels(), rotation=30)
                        if name1 != 'genreId':
                            subsample_name = name1 + ' ' + name2
                            title = self.initial_panel + ' Dataset -- Panel ' \
                                    + regression.graph_subsample_title_dict[subsample_name] \
                                    + '\n Mean Pricing Variables for Niche VS. Broad \nType Apps Before and After 2020 March'
                        else:
                            title = self.initial_panel + ' Dataset -- Panel ' \
                                    + name1 + ' ' + name2 \
                                    + '\n Mean Pricing Variables for Niche VS. Broad \nType Apps Before and After 2020 March'
                            title = title.replace("genreId", "Category")
                            title = title.replace("_", " ")
                            title = title.lower()
                        title = title.title()
                        fig.suptitle(title, fontsize=14)
                        plt.subplots_adjust(top=0.85)
                        filename = self.initial_panel + '_' + name1 + '_' + name2 + '_parallel_trend.png'
                        fig.savefig(regression.descriptive_stats_graphs / 'parallel_trend' / filename,
                                    facecolor='white',
                                    dpi=300)
        return ls, mdf
        # return regression(initial_panel=self.initial_panel,
        #                    all_panels=self.all_panels,
        #                    dep_vars=self.dep_vars,
        #                    independent_vars=self.independent_vars,
        #                    subsample_names=self.ssnames,
        #                    reg_dict=self.reg_dict,
        #                    reg_dict_xy=self.reg_dict_xy,
        #                    single_panel_df=self.single_panel_df,
        #                    subsample_op_results=self.subsample_op_results)


    def select_x_y_for_subsample(self, n_niche_scales, individual_dummies):
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
                    x_vars_full_nd = x_vars \
                                     + ['full_full_NicheDummy', 'PostXfull_full_NicheDummy'] \
                                     + self.dep_vars
                    self.reg_dict_xy['full']['full']['NicheDummy'] = df[x_vars_full_nd]
                    # -------------------------------------------------------------------------------
                    niche_scale_cols = ['full_full_NicheScaleDummy_' + str(i) for i in range(1, n_niche_scales)]
                    x_vars_full_sd = niche_scale_cols \
                                     + ['PostX' + i for i in niche_scale_cols] \
                                     + x_vars + self.dep_vars
                    if individual_dummies is False:
                        self.reg_dict_xy['full']['full']['NicheScaleDummies'] = df[x_vars_full_sd]
                    else:
                        self.reg_dict_xy = df
                # you can delete this elif block once you've run the may 2021 procedure because sub-sample and minInstallstop dummies are the same
                elif 'minInstalls' in name2:
                    self.reg_dict_xy['minInstalls'][name2] = dict.fromkeys(['NicheDummy'])
                    x_vars_mininstalls = [name1 + '_' + name2 + '_NicheDummy',
                                          'PostX' + name1 + '_' + name2 + '_NicheDummy'] \
                                        + x_vars + self.dep_vars
                    # since this minInstall is subsample sliced according to
                    remove_minInstalls_dummies = [i for i in x_vars_mininstalls if 'DeMeanedminInstalls' not in i]
                    self.reg_dict_xy['minInstalls'][name2]['NicheDummy'] = df[remove_minInstalls_dummies]
                elif 'top' in name2:
                    self.reg_dict_xy[name1][name2] = dict.fromkeys(['NicheDummy'])
                    x_vars_developer = [name1 + '_' + name2 + '_NicheDummy',
                                        'PostX' + name1 + '_' + name2 + '_NicheDummy'] \
                                        + x_vars + self.dep_vars
                    # since this top and non-top is subsample sliced according to top_digital_firms
                    remove_top_digital_firms = [i for i in x_vars_developer if i != 'top_digital_firms']
                    self.reg_dict_xy['developer'][name2]['NicheDummy'] = df[remove_top_digital_firms]
                else:
                    self.reg_dict_xy[name1][name2] = dict.fromkeys(['NicheDummy'])
                    genreid_x_vars = [name1 + '_' + name2 + '_NicheDummy',
                                      'PostX' + name1 + '_' + name2 + '_NicheDummy']\
                                      + x_vars + self.dep_vars
                    unchecked_df = df[genreid_x_vars]
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
        for i in cols_drop:
            if i in df3.columns:
                df3 = df3.drop(i, axis=1)
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

    def graph_parallel_trend(self):
        data = self.reg_dict_xy
        ls = []
        mdf = []
        for name1, content1 in data.items():
            for name2, content2 in content1.items():
                for nichetype, df in content2.items():
                    df2 = df.copy(deep=True)
                    if nichetype != 'NicheScaleDummies':
                        NicheDummy = name1 + '_' + name2 + '_' + nichetype
                        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8.5))
                        fig.tight_layout(pad=4)
                        for i in range(len(self.dep_vars)):
                            df3 = df2[[self.dep_vars[i], NicheDummy]].reset_index()
                            df3['panel'] = pd.to_datetime(df3['panel'], format='%Y%m')
                            ls.append(df3)
                            # groupby nichedummy and return the mean of the dependent variable
                            df4 = df3.groupby(['panel', NicheDummy])[self.dep_vars[i]].mean()\
                                    .unstack(level=-1) \
                                    .rename(columns={0: 'Broad', 1: 'Niche'})
                            mdf.append(df4)
                            ax.flat[i].plot(df4.index.values, df4['Broad'], marker='o', label = 'Broad')
                            ax.flat[i].plot(df4.index.values, df4['Niche'], marker='o', label = 'Niche')
                            ax.flat[i].legend(loc='lower right',
                                              title=None)
                            ax.flat[i].set_ylabel(regression.graph_ylabel_dict[self.dep_vars[i]])
                            ax.flat[i].set_xlabel('Time')
                            ax.flat[i].grid(axis='y')
                            # add vertical line for DiD (stay-at-home order, most states issued in mid March, some in early April.)
                            # https://www.nytimes.com/interactive/2020/us/coronavirus-stay-at-home-order.html
                            # plt.axvline(datetime(2020, 3, 15))
                            ax.flat[i].axvline(x=datetime(2020, 3, 15), linewidth=2, color='red')
                            # ax.flat[i].text(datetime(2020, 3, 15), -.05, 'stay \nhome \norder', color='red',
                            #                 transform=ax.flat[i].get_xaxis_transform(),
                            #                 ha='center', va='top')
                            # You cannot use ax rorate ticks because the time will become random numbers
                            # ax.flat[i].set_xticklabels(ax.flat[i].get_xticks(), rotation=30)
                            plt.setp(ax.flat[i].get_xticklabels(), rotation=30)
                        if name1 != 'genreId':
                            subsample_name = name1 + ' ' + name2
                            title = self.initial_panel + ' Dataset -- Panel ' \
                                + regression.graph_subsample_title_dict[subsample_name] \
                                + '\n Mean Pricing Variables for Niche VS. Broad \nType Apps Before and After 2020 March'
                        else:
                            title = self.initial_panel + ' Dataset -- Panel ' \
                                + name1 + ' ' + name2 \
                                + '\n Mean Pricing Variables for Niche VS. Broad \nType Apps Before and After 2020 March'
                            title = title.replace("genreId", "Category")
                            title = title.replace("_", " ")
                            title = title.lower()
                        title = title.title()
                        fig.suptitle(title, fontsize=14)
                        plt.subplots_adjust(top=0.85)
                        filename = self.initial_panel + '_' + name1 + '_' + name2 + '_parallel_trend.png'
                        fig.savefig(regression.descriptive_stats_graphs / 'parallel_trend' / filename,
                                    facecolor='white',
                                    dpi=300)
        return ls, mdf
        # return regression(initial_panel=self.initial_panel,
        #                    all_panels=self.all_panels,
        #                    dep_vars=self.dep_vars,
        #                    independent_vars=self.independent_vars,
        #                    subsample_names=self.ssnames,
        #                    reg_dict=self.reg_dict,
        #                    reg_dict_xy=self.reg_dict_xy,
        #                    single_panel_df=self.single_panel_df,
        #                    subsample_op_results=self.subsample_op_results)

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
        reg_level_dfs = []
        for name1, content1 in self.subsample_op_results.items():
            for name2, content2 in content1.items():
                for name3, content3 in content2.items():
                    # for OLS results ====================================================================================
                    if reg_folder_name == 'cross_section':
                        reg_level_stats_df = pd.DataFrame(columns=['index', 'nobs', 'rsquared', 'rsquared_adj'])
                        combined_niche_dummies = pd.DataFrame(columns=['index'])
                        df = pd.DataFrame(columns=['index'])
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
                        df = pd.DataFrame(columns=['index'])
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
        df['index'] = result.params.index
        df.set_index('index', inplace=True)
        coef = result.params.rename(y + '_params').to_frame()
        pv = result.pvalues.rename(y + '_pvalue').to_frame()
        df = df.join(coef, how='inner')
        df = df.join(pv, how='inner')
        # regression level stats ------------------------------------------------------
        reg_level_stats_df.at[0, 'index'] = name1 + '_' + name2 + '_' + name3 + '_' + y
        reg_level_stats_df.at[0, 'nobs'] = result.nobs
        reg_level_stats_df.at[0, 'rsquared'] = result.rsquared
        reg_level_stats_df.at[0, 'rsquared_adj'] = result.rsquared_adj
        # put all niche dummies into a single table -----------------------------------
        index_list = []
        for i in result.params.index.values:
            if 'Niche' in i or i == 'const':
                index_list.append(i)
        for i in range(len(index_list)):
            combined_niche_dummies.at[i, 'index'] = index_list[i]
            combined_niche_dummies.at[i, y] = result.params.loc[index_list[i]]
            combined_niche_dummies.at[i, y + '_pvalue'] = result.pvalues.loc[index_list[i]]
        combined_niche_dummies.at[
            combined_niche_dummies['index'] == 'const', 'index'] = name1 + '_' + name2 + '_' + name3 + '_Constant'
        return df, reg_level_stats_df, combined_niche_dummies

    def _put_panel_results_into_pandas(self, name1, name2, name3, y, panel_reg, panel_res,
                                       df, reg_level_stats_df, combined_niche_dummies):
        # single panel regression coefficients and p-values ---------------------------------
        df['index'] = panel_res.params.index
        df.set_index('index', inplace=True)
        coef = panel_res.params.rename(y + panel_reg + '_params').to_frame()
        pv = panel_res.pvalues.rename(y + panel_reg + '_pvalue').to_frame()
        df = df.join(coef, how='inner')
        df = df.join(pv, how='inner')
        # panel regression level stats ------------------------------------------------------
        reg_level_stats_df.at[0, 'index'] = name1 + '_' + name2 + '_' + name3 + '_' + y + '_' + panel_reg
        reg_level_stats_df.at[0, 'nobs_' + panel_reg] = panel_res.nobs
        reg_level_stats_df.at[0, 'rsquared_' + panel_reg] = panel_res.rsquared
        # no adjusted rsquared for panel regressions
        # put all niche dummies into a single table -----------------------------------------
        index_list = []
        for i in panel_res.params.index.values:
            if 'Niche' in i or i == 'PostDummy' or i == 'const': # PostDummy is the treatment dummy in DiD
                index_list.append(i)
        for i in range(len(index_list)):
            combined_niche_dummies.at[i, 'index'] = index_list[i]
            combined_niche_dummies.at[i, y + '_' + panel_reg] = panel_res.params.loc[index_list[i]]
            combined_niche_dummies.at[i, y + '_' + panel_reg + '_pvalue'] = panel_res.pvalues.loc[index_list[i]]
        # change the PostDummy and const index to incorporate names for the purpose of future concatenating frames
        combined_niche_dummies.at[combined_niche_dummies['index'] == 'PostDummy', 'index'] = name1 + '_' + name2 + '_' + name3 + '_PostDummy'
        combined_niche_dummies.at[
            combined_niche_dummies['index'] == 'const', 'index'] = name1 + '_' + name2 + '_' + name3 + '_Constant'
        return df, reg_level_stats_df, combined_niche_dummies

    def select_cols_and_rows_result_PostXNicheDummy(self, result_type, table_type):
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
                contain = ['PostX', 'NicheDummy', 'PostDummy', 'Constant']
            else:
                contain = ['NicheDummy', 'Constant']
            not_contain = ['genreId', 'NicheScaleDummy', 'NicheScaleDummies']
            for i in df.index.values:
                if any([substring in i for substring in contain]) and not any([substring in i for substring in not_contain]):
                    selected_row_indices.append(i)
        elif table_type == 'table_2':
            if result_type == 'panel':
                substrings = ['PostX', 'NicheDummy', 'PostDummy', 'Constant']
            else:
                substrings = ['NicheDummy', 'Constant']
            for i in df.index.values:
                if any([substring in i for substring in substrings]) and 'genreId' in i:
                    selected_row_indices.append(i)
        else:
            if result_type == 'panel':
                substrings = ['PostXfull_full_NicheScaleDummy',
                              'full_full_NicheScaleDummy',
                              'NicheScaleDummies_PostDummy',
                              'NicheScaleDummies_Constant']
            else:
                substrings = ['NicheScaleDummy',
                              'NicheScaleDummies_Constant']
            for i in df.index.values:
                if any([substring in i for substring in substrings]):
                    selected_row_indices.append(i)
        # -------------------------------------------------------------------------------
        df2 = df.loc[selected_row_indices, selected_col_names]
        return df2

    def add_pvalue_asterisk_to_results(self, df):
        """
        This should be run after self.select_cols_and_rows_result_PostXNicheDummy(self, result_type, table_type).
        df is the output of self.select_cols_and_rows_result_PostXNicheDummy
        """
        # convert p_value columns to asterisks
        df2 = df.copy(deep=True)
        pvaluecols = [i for i in df2.columns if 'pvalue' in i]
        def f(x):
            if 0.05 < float(x) <= 0.1:
                return '*'
            elif 0.01 < float(x) <= 0.05:
                return '**'
            elif float(x) <= 0.01:
                return '***'
            else:
                return ''
        for i in pvaluecols:
            df2[i] = df2[i].apply(f)
        # concat pvalue cols to coefficient cols
        for i in df2.columns:
            if i not in pvaluecols:
                df2[i] = df2[i].astype(str) + df2[i+'_pvalue']
        df2.drop(pvaluecols, axis=1, inplace=True)
        return df2

    def set_row_and_column_groups(self, df, result_type, table_type):
        """
        The input df is the output of self.add_pvalue_asterisk_to_results(df)
        """
        df2 = df.copy(deep=True)
        # group columns ---------------------------------------------------------
        for i in df2.columns:
            new_i = i.replace('_POOLED_OLS', '')
            df2.rename(columns={i: new_i}, inplace=True)
        df2.rename(columns={'offersIAPTrue': 'OffersIAP',
                            'containsAdsTrue': 'ContainsAds',
                            'paidTrue': 'Paid',
                            'Imputedprice': 'Price'},
                   inplace=True)
        df2.columns = pd.MultiIndex.from_product([['Dependant Variables'],
                                                  df2.columns.tolist()])
        df2 = df2.reset_index()
        def reformat_index(x):
            x = x.replace('_', ' ')
            x = x.lower()
            return x
        df2['index'] = df2['index'].apply(reformat_index)
        df2.rename(columns={'index': 'Independent Vars'}, inplace=True)
        df2['Samples'] = None
        def replace_all(text, dic):
            for i, j in dic.items():
                text = text.replace(i, j)
            return text
        def set_sample(x):
            if 'full' in x:
                x = 'full'
            elif 'genreid' in x:
                rep = {' nichedummy': '',
                       'postxgenreid ': '',
                        'genreid ': '',
                       ' postdummy': '',
                       ' constant': ''}
                x = replace_all(x, rep)
            elif 'mininstalls imputedmininstalls ' in x:
                rep = {' nichedummy': '',
                       'postxmininstalls imputedmininstalls ': '',
                        'mininstalls imputedmininstalls ': '',
                       ' postdummy': '',
                       ' constant': ''}
                x = replace_all(x, rep)
            elif 'developer' in x:
                rep = {' nichedummy': '',
                       'postxdeveloper ': '',
                        'developer ': '',
                       ' postdummy': '',
                       ' constant': ''}
                x = replace_all(x, rep)
            else:
                x = x
            return x
        df2['Samples'] = df2['Independent Vars'].apply(set_sample)
        df2['Samples'] = df2['Samples'].apply(lambda x: x.capitalize())
        # ----------------------------------------------------------------------------------
        def set_row_indep_var_panel(x):
            if 'constant' in x:
                return 'Constant'
            elif 'postdummy' in x:
                return 'Post'
            elif 'nichedummy' in x and 'postx' not in x:
                return 'Niche'
            elif 'nichedummy' in x and 'postx' in x:
                return 'PostNiche'
            elif 'postx' in x and 'nichescaledummy' in x:
                rep = {'postxfull full ': 'Post',
                       'nichescaledummy': 'NicheScale'}
                x = replace_all(x, rep)
                return x
            elif 'full full nichescaledummy' in x:
                rep = {'full full nichescaledummy': 'NicheScale'}
                x = replace_all(x, rep)
                return x
            else:
                return x
        def set_row_indep_var_ols(x):
            if 'constant' in x:
                return 'Constant'
            elif 'nichedummy' in x:
                return 'Niche'
            elif 'nichescaledummy' in x:
                return 'NicheScale'
            else:
                return x
        # ----------------------------------------------------------------------------------
        if result_type == 'panel':
            df2['Independent Vars'] = df2['Independent Vars'].apply(set_row_indep_var_panel)
        else:
            df2['Independent Vars'] = df2['Independent Vars'].apply(set_row_indep_var_ols)
        # sort independent vars in the order Post Niche and Postniche
        df2['ind_var_order'] = None
        df2.at[df2['Independent Vars'] == 'Constant', 'ind_var_order'] = 0
        df2.at[df2['Independent Vars'] == 'Niche', 'ind_var_order'] = 1
        df2.at[df2['Independent Vars'] == 'Post', 'ind_var_order'] = 2
        df2.at[df2['Independent Vars'] == 'PostNiche', 'ind_var_order'] = 3
        for i in range(1, 20):
            df2.at[df2['Independent Vars'] == 'NicheScale ' + str(i), 'ind_var_order'] = i+2
            df2.at[df2['Independent Vars'] == 'PostNicheScale ' + str(i), 'ind_var_order'] = i+2
        df2 = df2.groupby(['Samples'], sort=False) \
            .apply(lambda x: x.sort_values(['ind_var_order'], ascending=True)) \
            .reset_index(drop=True).drop(['ind_var_order'], axis=1)
        if table_type == 'table_3':
            df2.drop(['Samples'], axis=1, inplace=True)
            df2.set_index('Independent Vars', inplace=True)
        else:
            df2.set_index(['Samples', 'Independent Vars'], inplace=True)
        return df2

    def convert_to_latex(self, df, result_type, table_type):
        """
        df is the output of self.set_row_and_column_groups(self, df, result_type)
        """
        f_name = result_type + '_' + table_type + '_NicheDummy_Combined.tex'
        df.to_latex(buf=regression.reg_table_path / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        return df


    # ========================================================================================================
    # ========================================================================================================
    def create_var_definition_latex(self):
        df = pd.DataFrame()
        # the variable name here are the actual one shown in the paper, not the ones embedded in the code
        # MAPPING -------------------------------------------------
        var_definitions = {
                    'containsAdsTrue': '\makecell[l]{Dummy variable equals 1 if an app \\\ contains advertisement, 0 otherwise.}',
                    'offersIAPTrue': '\makecell[l]{Dummy variable equals 1 if an app \\\ offers in-app-purchase, 0 otherwise.}',
                    'paidTrue': '\makecell[l]{Dummy variable equals 1 if an app charges a positive \\\ price upfront, 0 otherwise.}',
                    'Imputedprice': '\makecell[l]{App price in USD.}',
                    'PostDummy': """ \makecell[l]{Dummy variable equals 1 if the observation \\\ 
                                    is after Mar 2020 (inclusive), and 0 if the \\\ 
                                    observation is before Mar 2020} """,
                    'PostNiche': """ \makecell[l]{Interaction variable between Post and Niche,\\\  
                                    which equals 1 if and only if the observation \\\ 
                                    equals to 1 for both Post and Niche.} """,
                    'PostNicheScale': """ \makecell[l]{Interaction variable between Post and NicheScale dummies,\\\  
                                    which equals 1 if and only if the observation \\\ 
                                    equals to 1 for both Post and NicheScale.} """,
                    'Imputedscore': '\makecell[l]{Average score, between 1 and 5, of an app in a period. }',
                    'DeMeanedImputedscore': """ \makecell[l]{Demeaning the score across all panels \\\ 
                                            within each app.} """,
                    'minInstallsTop': """ \makecell[l]{Dummy variable equals 1 if the app \\\ 
                                            has minimum installs above 10,000,000 \\\ 
                                            (inclusive) in a panel, 0 otherwise.} """,
                    'DeMeanedminInstallsTop': '\makecell[l]{Demeaning Tier1 across all panels \\\ within each app.}',
                    'minInstallsMiddle': """ \makecell[l]{Dummy variable equals 1 if the app \\\ 
                                            has minimum installs below 10,000,000 and \\\ 
                                            above 100,000 (inclusive), 0 otherwise.} """,
                    'DeMeanedminInstallsMiddle': '\makecell[l]{Demeaning Tier2 across all panels \\\ within each app.}',
                    'Imputedreviews': '\makecell[l]{Number of reviews of an app in a panel.}',
                    'ZScoreDeMeanedImputedreviews': """ \makecell[l]{Demeaning Z-Score standardized \\\ 
                                                        number of reviews across all panels within each app.} """,
                    'contentRatingAdult': '\makecell[l]{Dummy variable equals 1 if the app has \\\ adult content, 0 otherwise.}',
                    'size': '\makecell[l]{Size (MB)}',
                    'DaysSinceReleased': '\makecell[l]{Number of days since the app was \\\ launched on Google Play Store.}',
                    'top_digital_firms': '\makecell[l]{Dummy variable equals 1 if the app is \\\ owned by a Top Digital Firm.}',
                    'NicheDummy': """ \makecell[l]{Dummy variable equals 1 if an app \\\ 
                                        is a niche-type app, 0 if an app is broad-type app.}""",
                    'NicheScaleDummies': """ \makecell[l]{20 dummy variables representing the degree of niche property of an app. \\\ 
                                            NicheScale_{0} equals to 1 represents that the app is the most broad type, \\\ 
                                            while NicheScale_{19} equals to 1 represents that the app is the most niche type.} """}

        time_variancy = {
            'Time Variant': ['containsAdsTrue',
                                     'offersIAPTrue',
                                     'paidTrue',
                                     'Imputedprice',
                                     'PostDummy',
                                     'PostNiche',
                                     'PostNicheScale',
                                     'Imputedscore',
                                     'DeMeanedImputedscore',
                                     'minInstallsTop',
                                     'DeMeanedminInstallsTop',
                                     'minInstallsMiddle',
                                     'DeMeanedminInstallsMiddle',
                                     'Imputedreviews',
                                     'ZScoreDeMeanedImputedreviews'],
            'Time Invariant': ['contentRatingAdult',
                                       'size',
                                       'DaysSinceReleased',
                                       'top_digital_firms',
                                       'NicheDummy',
                                       'NicheScaleDummies']}

        df['reference'] = ['containsAdsTrue',
                                            'offersIAPTrue',
                                            'paidTrue',
                                            'Imputedprice',
                                            'PostDummy',
                                            'PostNiche',
                                            'PostNicheScale',
                                            'Imputedscore',
                                            'DeMeanedImputedscore',
                                            'minInstallsTop',
                                            'DeMeanedminInstallsTop',
                                            'minInstallsMiddle',
                                            'DeMeanedminInstallsMiddle',
                                            'Imputedreviews',
                                            'ZScoreDeMeanedImputedreviews',
                                            'contentRatingAdult',
                                            'size',
                                            'DaysSinceReleased',
                                            'top_digital_firms',
                                            'NicheDummy',
                                            'NicheScaleDummies']

        df['Variables'] = [regression.var_names[i] for i in df['reference']]

        df['Definitions'] = None
        for var, definition in var_definitions.items():
            df.at[df['reference'] == var, 'Definitions'] = definition

        df['Type'] = None
        for type, var_list in time_variancy.items():
            for j in var_list:
                for i in df['reference']:
                    if i == j:
                        df.at[df['reference'] == i, 'Type'] = type

        df.drop('reference', axis=1, inplace=True)
        df.set_index(['Type', 'Variables'], inplace=True)

        f_name = 'variable_definition.tex'
        df.to_latex(buf=regression.descriptive_stats_tables / f_name,
                    multirow=True,
                    multicolumn=True,
                    longtable=True,
                    position='h!',
                    escape=False)
        return df

    # ========================================================================================================
    # ========================================================================================================

    def slice_single_panel_descriptive_stats(self, var_list, the_panel):
        self.reg_dict_xy = dict.fromkeys(self.reg_dict.keys())
        for name1, content1 in self.reg_dict.items():
            self.reg_dict_xy[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                NicheDummy = name1 + '_' + name2 + '_NicheDummy'
                variables = copy.deepcopy(var_list)
                variables = variables + [NicheDummy]
                vdf = df[variables]
                for i in vdf.columns:
                    if 'NicheDummy' in i:
                        vdf.rename(columns={i: 'NicheDummy'}, inplace=True)
                self.reg_dict_xy[name1][name2] = vdf
        self.single_panel_df = dict.fromkeys(self.reg_dict_xy.keys())
        for name1, content1 in self.reg_dict_xy.items():
            self.single_panel_df[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                self.single_panel_df[name1][name2] = self._slice_a_panel_from_long_df(
                    the_panel=the_panel, df=df)
        return regression(initial_panel=self.initial_panel,
                          all_panels=self.all_panels,
                          dep_vars=self.dep_vars,
                          independent_vars=self.independent_vars,
                          subsample_names=self.ssnames,
                          reg_dict=self.reg_dict,
                          reg_dict_xy=self.reg_dict_xy,
                          single_panel_df=self.single_panel_df,
                          subsample_op_results=self.subsample_op_results)

    def single_panel_descriptive_stats(self, var_list, the_panel, type):
        """
        run self.slice_single_panel_descriptive_stats before running this,
        DO NOT run self.select_x_y_for_subsample() and self.slice_single_panel() before this, because
        regression need different dataframes from descriptive stats.
        """
        for name1, content1 in self.single_panel_df.items():
            for name2, df in content1.items():
                df2 = df.copy(deep=True)
                if type == 'dummy':
                    df_list = []
                    for i in var_list:
                    # some dummy var has been deleted because of they are perfectly correlated with other dummy var
                        if i in df2.columns:
                            if df2[i].nunique() == 2:
                                df3 = df2[i].value_counts()
                                df3 = df3.rename(i).to_frame().T.rename(columns={0: 'False', 1: 'True'})
                            else: # sometimes the entire column is 1 or 0
                                if df2[i].unique() == 1:
                                    df3 = df2[i].value_counts()
                                    df3 = df3.rename(i).to_frame().T.rename(columns={1: 'True'})
                                    df3['False'] = 0
                                else:
                                    df3 = df2[i].value_counts()
                                    df3 = df3.rename(i).to_frame().T.rename(columns={0: 'False'})
                                    df3['True'] = 0
                            df3['count'] = df3['False'] + df3['True']
                            df_list.append(df3)
                    f_name = self.initial_panel + '_PANEL_' + the_panel + '_' + name1 + '_' + name2  + '_Dummy_Stats.csv'
                    q = regression.descriptive_stats_tables / 'dummy_stats' / f_name
                elif type == 'continuous':
                    df_list = []
                    for i in var_list:
                        if i in df2.columns:
                            df3 = df2[i].describe(include=[np.number])
                            df3 = df3.rename(i).to_frame().T
                            df_list.append(df3)
                    f_name = self.initial_panel + '_PANEL_' + the_panel + '_' + name1 + '_' + name2  + '_Continuous_Stats.csv'
                    q = regression.descriptive_stats_tables / 'continuous_stats' / f_name
                else: # for DeMeaned Variables Only (They are continuous)
                    df_list = []
                    for i in var_list:
                        if i in df2.columns:
                            df3 = df2[i].describe(include=[np.number])
                            df3 = df3.rename(i).to_frame().T
                            df_list.append(df3)
                    f_name = self.initial_panel + '_PANEL_' + the_panel + '_' + name1 + '_' + name2 + '_DeMeaned_Stats.csv'
                    q = regression.descriptive_stats_tables / 'demeaned_stats' / f_name
                stats_df = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
                stats_df.to_csv(q)
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)

    def export_descriptive_stats_to_latex(self, the_panel):
        sample_name = {'_full_full': 'Full',
                     '_minInstalls_ImputedminInstalls_tier1': 'Tier1',
                     '_minInstalls_ImputedminInstalls_tier2': 'Tier2',
                     '_minInstalls_ImputedminInstalls_tier3': 'Tier3',
                     '_developer_top': 'Top',
                     '_developer_non-top': 'Non-top'}
        # =============== DUMMY TABLE ==============================================================
        df_list = []
        for name, mapped_name in sample_name.items():
            f_name = self.initial_panel + '_PANEL_' + the_panel + name + '_Dummy_Stats.csv'
            q = regression.descriptive_stats_tables / 'dummy_stats' / f_name
            df = pd.read_csv(q)
            df['Sample'] = mapped_name
            df_list.append(df)
        df2 = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
        df2['Variables'] = [regression.var_names[i] for i in df2.iloc[:,0]]
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2 = df2[['Sample', 'Variables', 'True', 'False', 'count']]
        df2.set_index(['Sample', 'Variables'], inplace=True)
        f_name = self.initial_panel + '_PANEL_' + the_panel + '_dummy_vars_descriptive_stats.tex'
        df2.to_latex(buf=regression.descriptive_stats_tables / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        # =============== CONTINUOUS TABLE =========================================================
        df_list = []
        for name, mapped_name in sample_name.items():
            f_name = self.initial_panel + '_PANEL_' + the_panel + name + '_Continuous_Stats.csv'
            q = regression.descriptive_stats_tables / 'continuous_stats' / f_name
            df = pd.read_csv(q)
            df['Sample'] = mapped_name
            df_list.append(df)
        df2 = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
        df2['Variables'] = [regression.var_names[i] for i in df2.iloc[:,0]]
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2.rename(columns={'50%': 'median'}, inplace=True)
        df2 = df2[['Sample', 'Variables', 'mean', 'std', 'min', 'median', 'max', 'count']]
        df2.set_index(['Sample', 'Variables'], inplace=True)
        # --- convert data type and round ----
        df2['count'] = df2['count'].astype(int)
        for i in ['mean', 'std', 'min', 'median', 'max']:
            df2[i] = df2[i].apply(lambda x: round(x, 2))
        f_name = self.initial_panel + '_PANEL_' + the_panel + '_continuous_vars_descriptive_stats.tex'
        df2.to_latex(buf=regression.descriptive_stats_tables / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        # =============== DEMEANED TABLE =========================================================
        df_list = []
        for name, mapped_name in sample_name.items():
            f_name = self.initial_panel + '_PANEL_' + the_panel + name + '_DeMeaned_Stats.csv'
            q = regression.descriptive_stats_tables / 'demeaned_stats' / f_name
            df = pd.read_csv(q)
            df['Sample'] = mapped_name
            df_list.append(df)
        df2 = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
        df2['Variables'] = [regression.var_names[i] for i in df2.iloc[:,0]]
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2.rename(columns={'50%': 'median'}, inplace=True)
        df2 = df2[['Sample', 'Variables', 'mean', 'std', 'min', 'median', 'max', 'count']]
        df2.set_index(['Sample', 'Variables'], inplace=True)
        # --- convert data type and round ----
        df2['count'] = df2['count'].astype(int)
        for i in ['mean', 'std', 'min', 'median', 'max']:
            df2[i] = df2[i].apply(lambda x: round(x, 2))
        f_name = self.initial_panel + '_PANEL_' + the_panel + '_demeaned_vars_descriptive_stats.tex'
        df2.to_latex(buf=regression.descriptive_stats_tables / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        return regression(initial_panel=self.initial_panel,
                           all_panels=self.all_panels,
                           dep_vars=self.dep_vars,
                           independent_vars=self.independent_vars,
                           subsample_names=self.ssnames,
                           reg_dict=self.reg_dict,
                           single_panel_df=self.single_panel_df,
                           subsample_op_results=self.subsample_op_results)






