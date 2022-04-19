import pandas as pd
from pathlib import Path
import pickle
import copy
import math

from statsmodels.compat import lzip

pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
from scipy.stats import boxcox
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat.python import lzip
from linearmodels import PooledOLS
from linearmodels import RandomEffects
from datetime import datetime
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")


class stats_and_regs:
    """2021 July 18
    This is the new version written based on the STEP10_ESSAY_2_3_Long_Table_Prep.py
    2022 Mar 26
    Combine market leaders and followers regression and statistics into the same class.
    Run robustness checks which include regressions with deleted missings (without imputing the missing) and with imputing the missing
    and they ways to validate the regression coefficients of the same variables (loosely speaking niche and post niche) are different
    for different sub-samples by pooled regression with sample dummies.
    """

    full_sample_panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/___full_sample___')
    nlp_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/nlp_stats')
    des_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/TABLES')
    des_stats_graphs = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/GRAPHS')
    ols_results = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____OLS_RESULTS____')
    panel_results = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____PANEL_RESULTS____')

    # ----------------------------- slicing variables --------------------------------------------------------------------
    sub_sample_key_level1 = ['FULL', 'ML', 'MF']
    FULL_sample_key_level2 = ['FULL', 'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                              'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']
    ML_sample_key_level2 = ['ML', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL']
    MF_sample_key_level2 = ['MF', 'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL']

    sub_sample_categorical_vars = ['MF_CAT', 'ML_CAT', 'FULL_CAT', 'FULL_TIER', 'FULL_FIRM']
    sub_sample_graph_cat_vars_d = {'FULL':['FULL_CAT', 'FULL_TIER', 'FULL_FIRM'],
                                   'ML':['ML_CAT'],
                                   'MF':['MF_CAT']}

    regplot_color_palette = {'FULL':{'FULL_CAT': sns.color_palette("hls", 5),
                                     'FULL_TIER': sns.color_palette("hls", 3),
                                     'FULL_FIRM': sns.color_palette("hls", 2)},
                             'ML':{'ML_CAT': sns.color_palette("hls", 5)},
                             'MF':{'MF_CAT': sns.color_palette("hls", 5)}}

    sub_sample_d = { 'FULL': dict.fromkeys(FULL_sample_key_level2),
                     'ML': dict.fromkeys(ML_sample_key_level2),
                     'MF': dict.fromkeys(MF_sample_key_level2)}

    sub_sample_l = FULL_sample_key_level2 + ML_sample_key_level2 + MF_sample_key_level2

    graph_layout_categorical = plt.subplots(3, 2)
    graph_layout_full_firm = plt.subplots(2, 1)
    graph_layout_full_tiers = plt.subplots(3, 1)

    core_dummy_y_vars_d = {'original': ['containsAdsdummy', 'offersIAPdummy', 'noisy_death', 'T_TO_TIER1_minInstalls', 'MA'],
                           'imputed':  ['imputed_containsAdsdummy', 'imputed_offersIAPdummy', 'noisy_death', 'T_TO_TIER1_minInstalls', 'MA']}
    core_continuous_y_vars_d = {'original': ['nlog_price', 'nlog_minInstalls'],
                                'imputed':  ['nlog_imputed_price', 'nlog_imputed_minInstalls']}


    # ---------------- variables below has original version and imputed versions ----------------
    # all y variables are time variant
    # they have both imputed and original variables
    balanced_y_vars = ['price', 'minInstalls', 'containsAdsdummy', 'offersIAPdummy']
    scaled_balanced_y_vars = ['nlog_price', 'nlog_minInstalls', 'containsAdsdummy', 'offersIAPdummy']
    imputed_balanced_y_vars = ['imputed_price', 'imputed_minInstalls', 'imputed_containsAdsdummy', 'imputed_offersIAPdummy']
    scaled_imputed_balanced_y_vars = ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'imputed_containsAdsdummy', 'imputed_offersIAPdummy']
    # all the unbalanced variables are created based on the top_firm, which is based on deverloperClean, and which is based on imputed_developer, and imputed_minInstalls
    # noisy_death is based on left merge indicator, so it's hard to say it is based on original or imputed variables
    unbalanced_only_y_vars = ['noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA']
    # all control variables are time variant for the easier implementation of regression
    control_vars = ['score', 'reviews', 'adultcontent']
    scaled_control_vars = ['score', 'nlog_reviews', 'adultcontent']
    imputed_control_vars = ['imputed_score', 'imputed_reviews', 'imputed_adultcontent']
    scaled_imputed_control_vars = ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent']
    # niche variables are calculated from kmeans on self.tcn + 'Clean' (which is in turn based on 'imputed_'+self.tcn)
    # essentially all niche variables are imputed variables (based on imputed app descriptions)
    niche_vars = ['continuous_niche', 'dummy_niche']
    # -------------------------------------------------------------------------------------------
    scale_var_dict = {
        'nlog_plus_one': ['reviews', 'minInstalls', 'price']
    }
    # -------------------- test names -----------------------------------------------------------
    # https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html
    jb_test_names = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
    bp_test_names = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]

    ##################################################################################################################
    def __init__(self,
                 initial_panel,
                 all_panels,
                 df=None,
                 original_dict=None,
                 imputed_dict=None,
                 reg_results=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.original_dict = original_dict
        self.imputed_dict = imputed_dict
        self.reg_results = reg_results

    # ====================== The set of functions below are regularly used common functions in pre_processing class =============================
    def _open_df(self, balanced, keyword):
        """
        :param balanced:
        :param keyword: could be any of 'merged', 'imputed', 'nlp' or 'reg_ready'
        :return:
        """
        print('------------------------ open_df ' + keyword + ' ---------------------------')
        if balanced is True:
            f_name = self.initial_panel + '_balanced_' + keyword + '.pickle'
        else:
            f_name = self.initial_panel + '_unbalanced_' + keyword + '.pickle'
        q = self.full_sample_panel_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    def _save_df(self, DF, balanced, keyword):
        """
        I will save the df at the very end, which include imputed variables, sub-sample dummies and NLP niche variables
        :param balanced: Whether to include appids that are missing in each subsequent month as compared to the original month
        :param keyword: could be any of 'merged', 'imputed', 'nlp' or 'reg_ready'
        :return:
        """
        print('------------------------ _save_df ' + keyword + ' ---------------------------')
        if balanced is True:
            f_name = self.initial_panel + '_balanced_' + keyword + '.pickle'
        else:
            f_name = self.initial_panel + '_unbalanced_' + keyword + '.pickle'
        q = self.full_sample_panel_path / f_name
        pickle.dump(DF, open(q, 'wb'))
        return None

    def _x_and_y_vars(self, balanced, imputed, scaled, niche_vars, months, panel):
        """
        :param df: the dataframe that has all the variables
        :param balanced: there are some y variables that only exist in the unbalanced dataset
        :param imputed: whether the variables are imputed, note that y variables that only exist in unbalanced dataset are created from other imputed variables
        :param scaled: natural log transformation of certain variables
        :param months: a list of months, such as ['202001'] or self.all_panels
        :return: a dictionary with keys 'x', 'y' or 'both'
        """
        # ----------------- unbalanced variables -------------------------------
        y_vars = self.balanced_y_vars + self.unbalanced_only_y_vars
        x_vars = niche_vars + self.control_vars
        if scaled is True:
            y_vars = self.scaled_balanced_y_vars + self.unbalanced_only_y_vars
            x_vars = niche_vars + self.scaled_control_vars
        if imputed is True:
            y_vars = self.imputed_balanced_y_vars + self.unbalanced_only_y_vars
            x_vars = niche_vars + self.imputed_control_vars
            if scaled is True:
                y_vars = self.scaled_imputed_balanced_y_vars + self.unbalanced_only_y_vars
                x_vars = niche_vars + self.scaled_imputed_control_vars
        # -------- balanced variables which only differs from unbalanced variables in terms of y variables --------
        if balanced is True:
            y_vars = self.balanced_y_vars
            if scaled is True:
                y_vars = self.scaled_balanced_y_vars
            if imputed is True:
                y_vars = self.imputed_balanced_y_vars
                if scaled is True:
                    y_vars = self.scaled_imputed_balanced_y_vars
        if panel is True:
            x_cols = [i + '_' + m for m in months for i in x_vars]
            y_cols = [i + '_' + m for m in months for i in y_vars]
            both_cols = x_cols + y_cols
        else:
            x_cols = {}
            y_cols = {}
            both_cols = {}
            xy = x_vars + y_vars
            for m in months:
                x_cols[m] = [i + '_' + m for i in x_vars]
                y_cols[m] = [i + '_' + m for i in y_vars]
                both_cols[m] = [i + '_' + m for i in xy]
        var_dict = {
            'x': x_cols,
            'y': y_cols,
            'both': both_cols
        }
        return var_dict

    def _scale_var_printout_descriptive_stats(self, df, k, ss, imputed):
        """
        :param df: A subsetted dataframe has no missing in the original variable columns
                   The existing columns in this dataframe should all be numeric
        :param scale_var_dict: key is the method and the value is a list of core variable names
        :return:
        """
        folder = 'original'
        for method, vars in self.scale_var_dict.items():
            for var in vars:
                vs = [var + '_' + m for m in self.all_panels]
                if imputed is True:
                    folder = 'imputed'
                    vs = ['imputed_' + var + '_' + m for m in self.all_panels]
                for v in vs:
                    if method == 'nlog_plus_one':
                        df['nlog_' + v] = df.apply(lambda row: np.log(row[v] + 1), axis=1)
        num_cols = [x for x in list(df.columns) if x not in self.sub_sample_categorical_vars]
        for i in num_cols:
            df[i] = pd.to_numeric(df[i])
        f_name = k + '_' + ss + '.csv'
        df.describe(include=[np.number]).to_csv(self.ols_results / 'var_statistics' / folder / f_name)
        return df

    def _create_categorical_sub_sample_vars(self, df):
        """
        :param df: should be self._open_df(balanced=balanced, keyword='imputed')
        :return:
        """
        print('------------------------ _create_categorical_sub_sample_vars ----------------------')
        # print(list(df.columns))
        # --------- create categorical sub-sample slicing variables for future graphing --------
        for v in ['ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL',
                  'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL',
                  'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                  'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']:
            # print(df[v].value_counts(dropna=False))
            df[v + '_cat'] = df.apply(lambda row: v if row[v] == 1 else '', axis=1)
            # print(df[v + '_cat'].value_counts(dropna=False))
        df['ML_CAT'] = df['ML_GAME_cat'] + df['ML_BUSINESS_cat'] + df['ML_SOCIAL_cat'] + \
                       df['ML_LIFESTYLE_cat'] + df['ML_MEDICAL_cat']
        df['MF_CAT'] = df['MF_GAME_cat'] + df['MF_BUSINESS_cat'] + df['MF_SOCIAL_cat'] + \
                       df['MF_LIFESTYLE_cat'] + df['MF_MEDICAL_cat']
        df['FULL_TIER'] = df['Tier1_cat'] + df['Tier2_cat'] + df['Tier3_cat']
        df['FULL_FIRM'] = df['top_firm_cat'] + df['non_top_firm_cat']
        df['FULL_CAT'] = df['FULL_GAME_cat'] + df['FULL_BUSINESS_cat'] + df['FULL_SOCIAL_cat'] + \
                         df['FULL_LIFESTYLE_cat'] + df['FULL_MEDICAL_cat']
        for v in ['ML_CAT', 'MF_CAT', 'FULL_TIER', 'FULL_FIRM', 'FULL_CAT']:
            df[v] = df[v].astype("category")
        print(df[['ML_CAT', 'MF_CAT', 'FULL_TIER', 'FULL_FIRM', 'FULL_CAT']].dtypes)
        return df

    def create_subsample_dict_and_merge_in_niche_vars_and_scale_vars(self, balanced):
        """
        :param balanced:
        :return: The slicing dummies are based on imputed variables such as imputed_minInstalls, top firms (imputed_developer_ and developerClean_), and imputed_genreId
                please refer to STEP2_pre_processing create_sub_sample_dummies.
        """
        print('------------------------ create_subsample_dict_and_merge_in_niche_vars ---------------------------')
        # ----------- open imputed dataframe ------------------------------------------
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        df2 = self.df.copy()
        df2 = self._create_categorical_sub_sample_vars(df=df2)
        print(df2.shape)
        # print(list(df2.columns))
        # -------- create empty dictionary placeholder for dictionary of dataframes
        res_original = copy.deepcopy(self.sub_sample_d)
        res_imputed = copy.deepcopy(self.sub_sample_d)
        # ----------- open nlp k means cluster labels ------------------------------------------
        filename = self.initial_panel + '_merged_niche_vars_with_appid.pickle'
        q = self.nlp_stats_path / filename
        with open(q, 'rb') as f:
            niche_dict = pickle.load(f)
        for k, s in res_original.items():
            # ----------- slicing into sub-samples and merge in the nlp labels -----------------
            for ss in s.keys():
                if ss == 'FULL':
                    df3 = df2.copy()
                else:
                    df3 = df2.loc[df2[ss]==1]
                # ------------------------------------------------------------------------------
                print(k + '---' + ss + ' before merging in niche variables')
                print(df3.shape)
                df3 = df3.merge(niche_dict[k][ss], how='outer', left_index=True, right_index=True)
                df3 = df3.fillna(value=np.nan)
                # make the time-invariant niche variables a set of time-variant variables
                for m in self.all_panels:
                    df3['continuous_niche_' + m] = df3['continuous_niche']
                    df3['dummy_niche_' + m] = df3['dummy_niche']
                df3.drop(['continuous_niche', 'dummy_niche'], axis=1, inplace=True)
                print(k + '---' + ss + ' after merging in niche variables')
                print(df3.shape)
                # --------------------------------------------------------
                print(k + '---' + ss + '---delete missing in the original variables')
                v_dict = self._x_and_y_vars(balanced=balanced, imputed=False, scaled=False,
                                            niche_vars=self.niche_vars,
                                            months=self.all_panels, panel=True)
                df4 = df3.dropna(axis=0, how='any', subset=v_dict['both'])
                df4 = df4.loc[:, v_dict['both']+self.sub_sample_categorical_vars]
                print(df4.shape)
                # print(list(df4.columns))
                # --------------------------------------------------------
                df4 = self._scale_var_printout_descriptive_stats(df=df4, k=k, ss=ss, imputed=False)
                # --------------------------------------------------------
                res_original[k][ss] = df4
                # --------------------------------------------------------
                print(k + '---' + ss + '---delete missing in the imputed variables')
                v_dict = self._x_and_y_vars(balanced=balanced, imputed=True, scaled=False,
                                            niche_vars=self.niche_vars,
                                            months=self.all_panels, panel=True)
                df5 = df3.dropna(axis=0, how='any', subset=v_dict['both'])
                df5 = df5.loc[:, v_dict['both']+self.sub_sample_categorical_vars]
                print(df5.shape)
                # print(list(df5.columns))
                # --------------------------------------------------------
                df5 = self._scale_var_printout_descriptive_stats(df=df5, k=k, ss=ss, imputed=True)
                # --------------------------------------------------------
                res_imputed[k][ss] = df5
        self.original_dict = res_original
        self.imputed_dict = res_imputed
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        original_dict=self.original_dict,
                        imputed_dict=self.imputed_dict)

    def table_cat_y_variables_against_niche_dummy(self, balanced):
        """
        you must run create_subsample_dict_and_merge_in_niche_vars_and_scale_vars before running this
        :param balanced:
        :return:
        """
        print('------------------------ table_cat_y_variables_against_niche_dummy ---------------------------')
        if balanced is True:
            b = 'balanced'
        else:
            b = 'unbalanced'
        original_vars = self._x_and_y_vars(balanced=balanced, imputed=False, scaled=True,
                                           niche_vars=['dummy_niche'],
                                           months=self.all_panels, panel=False)
        imputed_vars = self._x_and_y_vars(balanced=balanced, imputed=True, scaled=True,
                                          niche_vars=['dummy_niche'],
                                          months=self.all_panels, panel=False)
        for m in ['202104', '202105', '202106', '202107']:
            x_var = 'dummy_niche_' + m
            data_d = {'original': {'data': self.original_dict,
                                   'y_vars': original_vars['y'][m]},
                      'imputed': {'data': self.imputed_dict,
                                  'y_vars': imputed_vars['y'][m]}}
            for im, content in data_d.items():
                dummy_ys = []
                dummy_y_names = []
                for y_var in content['y_vars']:
                    if any([i in y_var for i in ['containsAds', 'offersIAP', 'MA', 'noisy_death', 'T_TO_TIER1']]):
                        dummy_y_names.append(y_var)
                        y_var_true = y_var.replace('_'+m, '') + '_true'
                        dummy_ys.append(y_var_true)
                        y_var_false = y_var.replace('_'+m, '') + '_false'
                        dummy_ys.append(y_var_false)
                # combine the value counts into a single dataframe and graph them in a single horizontal bar graph
                INDEX1 = []
                for i in self.sub_sample_l:
                    INDEX1 = INDEX1 + [i] * 2
                INDEX2 = ['Niche', 'Broad'] * len(self.sub_sample_l)
                dfg_dummy = pd.DataFrame(columns=dummy_ys,
                                         index=[INDEX1, INDEX2])
                print(dfg_dummy.head())
                for y_var in dummy_y_names:
                    y_var_true = y_var.replace('_' + m, '') + '_true'
                    y_var_false = y_var.replace('_' + m, '') + '_false'
                    for k, s in content['data'].items():
                        for ss in s.keys():
                            df = content['data'][k][ss].copy()
                            df2 = df.value_counts(subset=[x_var, y_var]).to_frame().reset_index()
                            df2.rename(columns={df2.columns[2]: 'n'}, inplace=True)
                            print(ss)
                            print(df2)
                            # some y variables in some months have only 0s in either or both niche broad apps.
                            if 1 in df2.loc[df2[x_var] == 1, y_var].unique():
                                v11 = df2.loc[(df2[x_var] == 1) & (df2[y_var] == 1), 'n'].squeeze()
                            else:
                                v11 = 0
                            if 1 in df2.loc[df2[x_var] == 0, y_var].unique():
                                v01 = df2.loc[(df2[x_var] == 0) & (df2[y_var] == 1), 'n'].squeeze()
                            else:
                                v01 = 0
                            dfg_dummy.at[(ss, 'Niche'), y_var_true] = v11
                            dfg_dummy.at[(ss, 'Broad'), y_var_true] = v01
                            # some y variables in some months have only 1s in either or both niche broad apps.
                            if 0 in df2.loc[df2[x_var] == 1, y_var].unique():
                                v10 = df2.loc[(df2[x_var] == 1) & (df2[y_var] == 0), 'n'].squeeze()
                            else:
                                v10 = 0
                            if 0 in df2.loc[df2[x_var] == 0, y_var].unique():
                                v00 = df2.loc[(df2[x_var] == 0) & (df2[y_var] == 0), 'n'].squeeze()
                            else:
                                v00 = 0
                            dfg_dummy.at[(ss, 'Niche'), y_var_false] = v10
                            dfg_dummy.at[(ss, 'Broad'), y_var_false] = v00
                f_name = 'cat_y_vars_counts_against_niche.csv'
                dfg_dummy.to_csv(self.des_stats_tables / b / im / m / 'dummy_niche' / f_name)
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            original_dict=self.original_dict,
            imputed_dict=self.imputed_dict,
            reg_results=self.reg_results)

    def graph_y_variables_against_niche(self, balanced):
        """
        you must run create_subsample_dict_and_merge_in_niche_vars_and_scale_vars before running this
        :param balanced:
        :return:
        """
        print('------------------------ graph_y_variables_against_niche ---------------------------')
        fig_params = {'x_axis_ss': {'FULL':  {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['FULL']),
                                              'figsize': (18, 7),
                                              'gridspec_kw': {'width_ratios': [3.7, 1.5, 1]}},
                                      'ML':  {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['ML']),
                                              'figsize': (14, 7),
                                              'gridspec_kw': None},
                                      'MF':  {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['MF']),
                                              'figsize': (14, 7),
                                              'gridspec_kw': None}},
                      'y_axis_ss': {'FULL':   {'nrows': len(self.sub_sample_graph_cat_vars_d['FULL']),
                                              'ncols': 1,
                                              'figsize': (7, 15),
                                              'gridspec_kw': {'height_ratios': [2.8, 1.4, 1]}},
                                      'ML':  {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['ML']),
                                              'figsize': (14, 7),
                                              'gridspec_kw': None},
                                      'MF':  {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['MF']),
                                              'figsize': (14, 7),
                                              'gridspec_kw': None}},
                      'hue_ss': {'FULL':     {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['FULL']),
                                              'figsize': (18, 7),
                                              'gridspec_kw': {'width_ratios': None}},
                                    'ML':    {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['ML']),
                                              'figsize': (14, 7),
                                              'gridspec_kw': None},
                                    'MF':    {'nrows': 1,
                                              'ncols': len(self.sub_sample_graph_cat_vars_d['MF']),
                                              'figsize': (14, 7),
                                              'gridspec_kw': None}}
                      }
        if balanced is True:
            b = 'balanced'
        else:
            b = 'unbalanced'
        for x_var in ['continuous_niche', 'dummy_niche']:
            original_vars = self._x_and_y_vars(balanced=balanced, imputed=False, scaled=True,
                                               niche_vars=[x_var],
                                               months=self.all_panels, panel=False)
            imputed_vars = self._x_and_y_vars(balanced=balanced, imputed=True, scaled=True,
                                              niche_vars=[x_var],
                                              months=self.all_panels, panel=False)
            for m in self.all_panels:
                x_var_m = x_var + '_' + m
                data_d = {'original': {'data': self.original_dict,
                                       'y_vars': original_vars['y'][m]},
                          'imputed': {'data': self.imputed_dict,
                                      'y_vars': imputed_vars['y'][m]}}
                for im, content in data_d.items():
                    for k, s in content['data'].items():
                        for ss in s.keys():
                            if ss in ['FULL', 'MF', 'ML']:
                                df = content['data'][k][ss].copy()
                                print(ss)
                                for y_var in self.core_dummy_y_vars_d[im] + self.core_continuous_y_vars_d[im]:
                                    print('*************************** prepare graph dataframe *****************************')
                                    y_var_m = y_var + '_' + m
                                    var_cols = [x_var_m, y_var_m] + self.sub_sample_graph_cat_vars_d[ss]
                                    df2 = df[var_cols]
                                    print(list(df2.columns))
                                    print(y_var + ' VS ' + x_var)
                                    print('*************************** start graphing dummy y and dummy niche ***************************')
                                    if y_var in self.core_dummy_y_vars_d[im] and x_var == 'dummy_niche':
                                        fig, axes = plt.subplots(nrows=fig_params['x_axis_ss'][ss]['nrows'],
                                                                 ncols=fig_params['x_axis_ss'][ss]['ncols'],
                                                                 figsize=fig_params['x_axis_ss'][ss]['figsize'],
                                                                 gridspec_kw=fig_params['x_axis_ss'][ss]['gridspec_kw'],
                                                                 sharey='row', sharex='col')
                                        sns.set_style("whitegrid")
                                        for i in range(len(self.sub_sample_graph_cat_vars_d[ss])):
                                            if len(self.sub_sample_graph_cat_vars_d[ss]) > 1:
                                                this_ax = axes[i]
                                            else:
                                                this_ax = axes  # because a single nrows=1, ncols=1 axes is not a numpy array thus not subscriptable
                                            # the bar with total niche and broad apps, with a lighter color
                                            sns.countplot(x=self.sub_sample_graph_cat_vars_d[ss][i],
                                                          data=df2, hue=x_var_m, ax=this_ax,
                                                          hue_order=[1, 0], # important for legend labels
                                                          palette={1: 'pink', 0: 'lightsteelblue'})
                                            # on top of the previous total bar, the bar with y_var is True niche and broad apps, with a darker color
                                            df5 = df2.loc[df2[y_var_m]==1]
                                            if df5.shape[0] > 0:
                                                sns.countplot(x=self.sub_sample_graph_cat_vars_d[ss][i],
                                                              data=df5, hue=x_var_m, ax=this_ax,
                                                              hue_order=[1, 0], # important for legend labels
                                                              palette={1: 'crimson', 0: 'steelblue'})
                                                handles, labels = this_ax.get_legend_handles_labels()
                                                labels = ['Niche', 'Broad', 'Niche and ' + y_var + ' True',
                                                          'Broad and ' + y_var + ' True']
                                            else:
                                                # some y variables such as noisy death has 0 ==1 in some months
                                                handles, labels = this_ax.get_legend_handles_labels()
                                                labels = ['Niche', 'Broad']
                                            this_ax.set(xlabel=None)
                                            this_ax.get_legend().remove()
                                        fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(0.5, -0.04),
                                                   ncol=len(labels))
                                    print('*************************** start graphing continuous y and dummy niche *************************** ')
                                    if y_var in self.core_continuous_y_vars_d[im] and x_var == 'dummy_niche':
                                        fig, axes = plt.subplots(nrows=fig_params['x_axis_ss'][ss]['nrows'],
                                                                 ncols=fig_params['x_axis_ss'][ss]['ncols'],
                                                                 figsize=fig_params['x_axis_ss'][ss]['figsize'],
                                                                 gridspec_kw=fig_params['x_axis_ss'][ss]['gridspec_kw'],
                                                                 sharey='row', sharex='col')
                                        sns.set_style("whitegrid")
                                        for i in range(len(self.sub_sample_graph_cat_vars_d[ss])):
                                            if len(self.sub_sample_graph_cat_vars_d[ss]) > 1:
                                                this_ax = axes[i]
                                            else:
                                                this_ax = axes  # because a single nrows=1, ncols=1 axes is not a numpy array thus not subscriptable
                                            sns.boxplot(x=self.sub_sample_graph_cat_vars_d[ss][i],
                                                        hue_order=[1, 0],
                                                        y=y_var_m, hue=x_var_m, data=df2, ax=this_ax)
                                            handles, labels = this_ax.get_legend_handles_labels()
                                            labels = ['Niche', 'Broad']
                                            this_ax.set(xlabel=None)
                                            this_ax.get_legend().remove()
                                        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.13, 0.87), ncol=1)
                                    print('*************************** start graphing dummy y and continuous niche ***************************')
                                    if y_var in self.core_dummy_y_vars_d[im] and x_var == 'continuous_niche':
                                        fig, axes = plt.subplots(nrows=fig_params['y_axis_ss'][ss]['nrows'],
                                                                 ncols=fig_params['y_axis_ss'][ss]['ncols'],
                                                                 figsize=fig_params['y_axis_ss'][ss]['figsize'],
                                                                 gridspec_kw=fig_params['y_axis_ss'][ss]['gridspec_kw'],
                                                                 sharey='row', sharex='col')
                                        sns.set_style("whitegrid")
                                        for i in range(len(self.sub_sample_graph_cat_vars_d[ss])):
                                            if len(self.sub_sample_graph_cat_vars_d[ss]) > 1:
                                                this_ax = axes[i]
                                            else:
                                                this_ax = axes  # because a single nrows=1, ncols=1 axes is not a numpy array thus not subscriptable
                                            sns.boxplot(x=x_var_m,
                                                        hue_order=[1, 0], orient='h',
                                                        y=self.sub_sample_graph_cat_vars_d[ss][i], hue=y_var_m,
                                                        data=df2, ax=this_ax)
                                            handles, labels = this_ax.get_legend_handles_labels()
                                            labels = ['True', 'False']
                                            this_ax.set(ylabel=None)
                                            this_ax.get_legend().remove()
                                        fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.13, 0.87), ncol=1, title=y_var)
                                    print('*************************** start graphing continuous y and continuous niche ************************')
                                    if y_var in self.core_continuous_y_vars_d[im] and x_var == 'continuous_niche':
                                        print(df2.dtypes)
                                        fig, axes = plt.subplots(nrows=fig_params['hue_ss'][ss]['nrows'],
                                                                 ncols=fig_params['hue_ss'][ss]['ncols'],
                                                                 figsize=fig_params['hue_ss'][ss]['figsize'],
                                                                 gridspec_kw=fig_params['hue_ss'][ss]['gridspec_kw'],
                                                                 sharey='row', sharex='col')
                                        sns.set_style("whitegrid")
                                        for i in range(len(self.sub_sample_graph_cat_vars_d[ss])):
                                            if len(self.sub_sample_graph_cat_vars_d[ss]) > 1:
                                                this_ax = axes[i]
                                            else:
                                                this_ax = axes  # because a single nrows=1, ncols=1 axes is not a numpy array thus not subscriptable
                                            ss_cats = df2[self.sub_sample_graph_cat_vars_d[ss][i]].unique().tolist()
                                            print(ss_cats)
                                            for j in range(len(ss_cats)):
                                                cat = ss_cats[j]
                                                the_color = self.regplot_color_palette[ss][self.sub_sample_graph_cat_vars_d[ss][i]][j]
                                                print(cat)
                                                df3 = df2.loc[df2[self.sub_sample_graph_cat_vars_d[ss][i]]==cat]
                                                sns.regplot(x=x_var_m, y=y_var_m,
                                                            truncate=False,
                                                            color=the_color,
                                                            data=df3, ax=this_ax,
                                                            label=cat)
                                            handles, labels = this_ax.get_legend_handles_labels()
                                            this_ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.2, 1), ncol=1)
                                    # ----------------- save ---------------------------------------
                                    f_name = ss + '_' + y_var + '.png'
                                    q = self.des_stats_graphs / b / im / m / x_var / f_name
                                    fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            original_dict=self.original_dict,
            imputed_dict=self.imputed_dict,
            reg_results=self.reg_results)

    def _check_cross_sectional_ols_assumptions(self, balanced, imputed, niche_v, k, ss, y, sms_results):
        """
        :param sms_results: statsmodel results object
        :return:
        """
        if balanced is True:
            b = 'balanced'
        else:
            b = 'unbalanced'
        if imputed is True:
            im = 'imputed'
        else:
            im = 'original'
        # normality of residual --------------------------------------------------------------
        test = sms.jarque_bera(sms_results.resid)
        test = lzip(self.jb_test_names, test) # this is a list of tuples
        test_df = pd.DataFrame(test, columns =['test_statistics', 'value'])
        f_name = k + '_' + ss + '_' + y + '_jb_test' + '.csv'
        save_f_name = self.ols_results / 'ols_assumptions_check' / b / im / niche_v / 'residual_normality' / f_name
        test_df.to_csv(save_f_name)
        # multi-collinearity -----------------------------------------------------------------
        test = np.linalg.cond(sms_results.model.exog)
        f_name = k + '_' + ss + '_' + y + '_multicollinearity.txt'
        save_f_name = self.ols_results / 'ols_assumptions_check' / b / im / niche_v / 'multicollinearity' / f_name
        with open(save_f_name, 'w') as f:
            f.writelines(str(test))
        # heteroskedasticity Breush-Pagan test -------------------------------------------------
        test = sms.het_breuschpagan(sms_results.resid, sms_results.model.exog)
        test = lzip(self.bp_test_names, test)
        test_df = pd.DataFrame(test, columns =['test_statistics', 'value'])
        f_name = k + '_' + ss + '_' + y + '_bp_test.csv'
        save_f_name = self.ols_results / 'ols_assumptions_check' / b / im / niche_v / 'heteroskedasticity' / f_name
        test_df.to_csv(save_f_name)
        # linearity Harvey-Collier -------------------------------------------------------------
        # this test not seem to work with my dataset because it raises singular matrix error message
        # I guess for the dummy_niche regressor, the relationship is not a linear one
        # I will visually plot the y variables against x variables to check linearity
        return None

    def cross_section_regression(self, balanced):
        """
        https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
        #https://www.statsmodels.org/stable/rlm.html
        https://stackoverflow.com/questions/30553838/getting-statsmodels-to-use-heteroskedasticity-corrected-standard-errors-in-coeff
        source code for HC0, HC1, HC2, and HC3, white and Mackinnon
        https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html
        https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/
        do cross sectional regression with both imputed and original variables for robustness checks
        # ---- ols regression assumptions check -----------------------
        # https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html
        """
        print('----------------------------- cross_section_regression ---------------------------------')
        if balanced is True:
            b = 'balanced'
        else:
            b = 'unbalanced'
        # one set of regression use continuous niche as independent var, the other set uses dummy niche
        for niche_v in self.niche_vars:
            original_vars = self._x_and_y_vars(balanced=balanced, imputed=False, scaled=True,
                                               niche_vars=[niche_v],
                                               months=self.all_panels, panel=False)
            imputed_vars = self._x_and_y_vars(balanced=balanced, imputed=True, scaled=True,
                                              niche_vars=[niche_v],
                                              months=self.all_panels, panel=False)
            for m in self.all_panels:
                for k, s in self.sub_sample_d.items():
                    for ss in s.keys():
                        dfo = self.original_dict[k][ss].copy()
                        dfi = self.imputed_dict[k][ss].copy()
                        # ================================ original variable ols ======================================
                        for y in original_vars['y'][m]:
                            o_x_str = ' + '.join(original_vars['x'][m])
                            formula = y + ' ~ ' + o_x_str
                            print(k + '--' + ss + '-- formula original --')
                            print(formula)
                            original_results = smf.ols(formula, data=dfo).fit()
                            table = original_results.summary().tables[1].as_csv()
                            f_name = k + '_' + ss + '_' + y + '_OLS_RESULTS.csv'
                            save_f_name = self.ols_results / 'raw_results' / b / 'original' / niche_v / f_name
                            with open(save_f_name, 'w') as fh:
                                fh.write(table)
                            self._check_cross_sectional_ols_assumptions(balanced=balanced,
                                                                        imputed=False,
                                                                        niche_v=niche_v,
                                                                        k=k, ss=ss, y=y,
                                                                        sms_results=original_results)
                        # ================================ imputed variable ols =======================================
                        for y in imputed_vars['y'][m]:
                            i_x_str = ' + '.join(imputed_vars['x'][m])
                            formula = y + ' ~ ' + i_x_str
                            print(k + '--' + ss + '-- formula imputed --')
                            print(formula)
                            imputed_results = smf.ols(formula, data=dfi).fit()
                            table = imputed_results.summary().tables[1].as_csv()
                            f_name = k + '_' + ss + '_' + y + '_OLS_RESULTS.csv'
                            save_f_name = self.ols_results / 'raw_results' / b / 'imputed' / niche_v / f_name
                            with open(save_f_name, 'w') as fh:
                                fh.write(table)
                            self._check_cross_sectional_ols_assumptions(balanced=balanced,
                                                                        imputed=True,
                                                                        niche_v=niche_v,
                                                                        k=k, ss=ss, y=y,
                                                                        sms_results=imputed_results)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        original_dict=self.original_dict,
                        imputed_dict=self.imputed_dict,
                        reg_results = self.reg_results)

    def summarize_ols_results(self, balanced):
        """
        :param balanced:
        :return:
        """
        print('----------------------------- summarize_ols_results ---------------------------------')
        if balanced is True:
            b = 'balanced'
        else:
            b = 'unbalanced'
        for niche_v in self.niche_vars:
            for im in ['original', 'imputed']:
                all_vars = self._x_and_y_vars(balanced=balanced, imputed=False, scaled=True,
                                              niche_vars=[niche_v],
                                              months=self.all_panels, panel=False)
                if im == 'imputed':
                    all_vars = self._x_and_y_vars(balanced=balanced, imputed=True, scaled=True,
                                                  niche_vars=[niche_v],
                                                  months=self.all_panels, panel=False)
                # ------------------- create empty dataframe to hold the statistics ------------------------------
                all_ss = self.FULL_sample_key_level2 + self.MF_sample_key_level2 + self.ML_sample_key_level2
                index1 = []
                for i in self.all_panels:
                    index1 = index1 + [i] * len(all_ss)
                index2 = all_ss * len(self.all_panels)
                y_core_ls = [i.replace('_'+self.initial_panel, '') for i in all_vars['y'][self.initial_panel]]
                # print(y_core_ls)
                res_df = pd.DataFrame(columns=y_core_ls, index=[index1, index2])
                # print(res_df.head())
                for m in self.all_panels:
                    for y in all_vars['y'][m]:
                        # print(y)
                        y_core = y.replace('_'+m, '')
                        # print(y_core)
                        for k, s in self.sub_sample_d.items():
                            for ss in s:
                                f_name = k + '_' + ss + '_' + y + '_OLS_RESULTS.csv'
                                df = pd.read_csv(self.ols_results / 'raw_results' / b / im / niche_v / f_name)
                                # remove whitespaces in column names and the first column (which will be set as index)
                                df[df.columns[0]] = df[df.columns[0]].str.strip()
                                df.set_index(df.columns[0], inplace=True)
                                df.columns = df.columns.str.strip()
                                df['P>|t|']=df['P>|t|'].astype(np.float64)
                                pvalue = df.loc[niche_v+'_'+m, 'P>|t|']
                                coef = df.loc[niche_v+'_'+m, 'coef']
                                if pvalue <= 0.01:
                                    asterisk = '***'
                                elif pvalue <= 0.05 and pvalue > 0.01:
                                    asterisk = '**'
                                elif pvalue <= 0.1 and pvalue > 0.05:
                                    asterisk = '*'
                                else:
                                    asterisk = ''
                                res_df.at[(m, ss), y_core] = str(round(coef, 2)) + asterisk
                f_name = b + '_' + niche_v + '_' + im  + '_ols_cross_sectional_results.csv'
                save_f_name = self.ols_results / 'summary_results' / f_name
                res_df.to_csv(save_f_name)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        original_dict=self.original_dict,
                        imputed_dict=self.imputed_dict,
                        reg_results = self.reg_results)



    def _create_cross_section_reg_results_df_for_parallel_trend_beta_graph(self, alpha):
        """
        possible input for reg_type are: 'cross_section_ols', uses self._cross_section_regression()
        alpha = 0.05 for 95% CI of coefficients
        """
        # all dependant variables in one dictionary
        res_results = dict.fromkeys(self.all_y_reg_vars)
        # all subsamples are hue in the same graph
        for y_var in self.all_y_reg_vars:
            res_results[y_var] = self.reg_results[y_var]
        #  since every reg result is one row in dataframe
        res_df = dict.fromkeys(self.all_y_reg_vars)
        for y_var, panels in res_results.items():
            # order in lists are persistent (unlike sets or dictionaries)
            panel_content = []
            sub_samples_content = []
            beta_nichedummy_content = []
            ci_lower = []
            ci_upper = []
            for panel, subsamples in panels.items():
                for name1, content1 in subsamples.items():
                    for name2, reg_result in content1.items():
                        panel_content.append(panel)
                        sub_samples_content.append(name1 + '_' + name2)
                        nichedummy = name1 + '_' + name2 + '_NicheDummy'
                        beta_nichedummy_content.append(reg_result.params[nichedummy])
                        ci_lower.append(reg_result.conf_int(alpha=alpha).loc[nichedummy, 0])
                        ci_upper.append(reg_result.conf_int(alpha=alpha).loc[nichedummy, 1])
            d = {'panel': panel_content,
                 'sub_samples': sub_samples_content,
                 'beta_nichedummy': beta_nichedummy_content,
                 'ci_lower': ci_lower,
                 'ci_upper': ci_upper}
            df = pd.DataFrame(data=d)
            # create error bars (positive distance away from beta) for easier ax.errorbar graphing
            df['lower_error'] = df['beta_nichedummy'] - df['ci_lower']
            df['upper_error'] = df['ci_upper'] - df['beta_nichedummy']
            # sort by panels
            df["panel"] = pd.to_datetime(df["panel"], format='%Y%m')
            df["panel"] = df["panel"].dt.strftime('%Y-%m')
            df = df.sort_values(by=["panel"])
            res_df[y_var] = df
        return res_df

    def _put_reg_results_into_pandas_for_single_y_var(self, reg_type, y_var, the_panel=None):
        """
        :param result: is the output of self._reg_for_all_subsamples(
            reg_type='panel_pooled_ols',
            y_var=any one of ['LogWNImputedprice', 'LogImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue'])
            the documentation of the PanelResult class (which result is)
        :return:
        """
        # ============= 1. extract results info and put them into dicts ==================
        params_pvalues_dict = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            params_pvalues_dict[name1] = dict.fromkeys(content1)
            for name2 in content1:
                # ---------- specify the rows to extract ---------------
                index_to_extract = {
                    'cross_section_ols': ['const', name1 + '_' + name2 + '_NicheDummy'],
                    'panel_pooled_ols': [
                        'const',
                        name1 + '_' + name2 + '_NicheDummy',
                        'PostDummy',
                        name1 + '_' + name2 + '_PostXNicheDummy']
                }
                # ---------- get the coefficients ----------------------
                if reg_type == 'cross_section_ols':
                    x = self.reg_results[y_var][the_panel][name1][name2].params
                else:
                    x = self.reg_results[y_var][name1][name2].params
                x = x.to_frame()
                x.columns = ['parameter']
                y = x.loc[index_to_extract[reg_type]]
                # ---------- get the pvalues ---------------------------
                if reg_type == 'cross_section_ols':
                    z1 = self.reg_results[y_var][the_panel][name1][name2].pvalues
                else:
                    z1 = self.reg_results[y_var][name1][name2].pvalues
                z1 = z1.to_frame()
                z1.columns = ['pvalue']
                z2 = z1.loc[index_to_extract[reg_type]]
                def _assign_asterisk(v):
                    if 0.05 < v <= 0.1:
                        return '*'
                    elif 0.01 < v <= 0.05:
                        return '**'
                    elif v <= 0.01:
                        return '***'
                    else:
                        return ''
                z2['asterisk'] = z2['pvalue'].apply(lambda x: _assign_asterisk(x))
                y2 = y.join(z2, how='inner')
                y2['parameter'] = y2['parameter'].round(3).astype(str)
                y2['parameter'] = y2['parameter'] + y2['asterisk']
                y2.rename(index={'const': 'Constant',
                                name1 + '_' + name2 + '_NicheDummy': 'Niche',
                                'PostDummy': 'Post',
                                name1 + '_' + name2 + '_PostXNicheDummy': 'PostNiche'},
                         inplace=True)
                y2 = y2.reset_index()
                y2.drop(columns=['pvalue', 'asterisk'], inplace=True)
                y2.insert(0, 'Samples', [name1 + '_' + name2] * len(y2.index))
                y2['Samples'] = y2['Samples'].apply(lambda x: self.combo_name12_reg_table_names[x] if x in self.combo_name12_reg_table_names.keys() else 'None')
                y2.rename(columns={'index': 'Independent Vars',
                                   'parameter': self.dep_vars_reg_table_names[y_var]},
                          inplace=True)
                params_pvalues_dict[name1][name2] = y2
        # ========= concatenate dataframes into a single dataframe for each combo ==========
        params_pvalues_combo_dict = self._rearrange_combo_df_dict(d = params_pvalues_dict)
        res = dict.fromkeys(params_pvalues_combo_dict.keys())
        for combo, content1 in params_pvalues_combo_dict.items():
            df_list = []
            for name12, df in content1.items():
                df_list.append(df)
            adf = functools.reduce(lambda a, b: a.append(b), df_list)
            res[combo] = adf
        return res

    def put_reg_results_into_pandas_for_all_y_var(self, reg_type, the_panel=None):
        res1 = dict.fromkeys(self.all_y_reg_vars)
        if reg_type == 'cross_section_ols':
            for y in self.all_y_reg_vars:
                res1[y] = self._put_reg_results_into_pandas_for_single_y_var(reg_type=reg_type,
                                                                             y_var=y,
                                                                             the_panel=the_panel)
        else:
            for y in self.all_y_reg_vars:
                res1[y] = self._put_reg_results_into_pandas_for_single_y_var(reg_type=reg_type, y_var=y)
        res2 = dict.fromkeys(['combo1', 'combo2', 'combo3', 'combo4'])
        for combo in res2.keys():
            df_list = []
            for y in self.all_y_reg_vars:
                df_list.append(res1[y][combo])
            adf = functools.reduce(lambda a, b: a.merge(b, how='inner',
                                                        on=['Samples', 'Independent Vars']),
                                   df_list)
            print(adf)
            filename = combo + '_' + reg_type + '_reg_results.csv'
            adf.to_csv(self.main_path / 'reg_tables_ready_for_latex' / filename)
            res2[combo] = adf
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

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

    def combine_app_level_text_cluster_stats_with_df(self):
        df = self._open_imputed_deleted_divided_df()
        d = self._open_app_level_text_cluster_stats()
        list_of_dfs = [d['full']['full']]
        for name1, content1 in d.items():
            for name2, stats_df in content1.items():
                if name2 != 'full':
                    list_of_dfs.append(stats_df)
        combined_stats_df = functools.reduce(lambda a, b: a.join(b, how='left'), list_of_dfs)
        self.cdf = df.join(combined_stats_df, how='inner')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def check_text_label_contents(self):
        df2 = self.cdf.copy(deep=True)
        d = essay_1_stats_and_regs_201907._open_predicted_labels_dict()
        for name1, content in d.items():
            for name2, text_label_col in content.items():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                unique_labels = df2[label_col_name].unique().tolist()
                unique_labels = [x for x in unique_labels if math.isnan(x) is False]
                print(name1, name2, ' -- unique text labels are --')
                print(unique_labels)
                print()
                for label_num in unique_labels:
                    df3 = df2.loc[df2[label_col_name]==label_num, [self.tcn + 'ModeClean']]
                    if len(df3.index) >= 10:
                        df3 = df3.sample(n=10)
                    f_name = self.initial_panel + '_' + name1 + '_' + name2 + '_' + 'TL_' + str(label_num) + '_' + self.tcn + '_sample.csv'
                    q = essay_1_stats_and_regs_201907.panel_essay_1_path / 'check_predicted_label_text_cols' / f_name
                    df3.to_csv(q)
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _numApps_per_cluster(self):
        d2 = self._open_predicted_labels_dict()
        d = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            d[name1] = dict.fromkeys(content1)
            for name2 in d[name1].keys():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                s2 = d2[name1][name2].groupby(
                    [label_col_name]).size(
                    ).sort_values(
                    ascending=False)
                d[name1][name2] = s2.rename('Apps Count').to_frame()
        return d




    def _prepare_pricing_vars_for_graph_group_by_var(self,
                                                     group_by_var,
                                                     the_panel=None):
        """
        group_by_var could by either "NicheDummy" or "cluster_size_bin"
        the dataframe (self.cdf) is after the function combine_app_level_text_cluster_stats_with_df
        """
        key_vars = ['Imputedprice',
                    'LogImputedprice',
                    # use this for regression and descriptive stats because it added uniform white noise to avoid 0 price
                    'LogWNImputedprice',
                    'ImputedminInstalls',
                    'LogImputedminInstalls',
                    'offersIAPTrue',
                    'containsAdsTrue']
        if the_panel is not None:
            selected_vars = [i + '_' + the_panel for i in key_vars]
        else:
            selected_vars = [i + '_' + j for j in self.all_panels for i in key_vars]
        d = self._slice_subsamples_dict()
        res12 = dict.fromkeys(self.ssnames.keys())
        res34 = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in d.items():
            res12[name1] = dict.fromkeys(content1.keys())
            res34[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                # ---- prepare regular df with log transformed imputedprice and imputed mininstalls --------
                text_label_var = name1 + '_' + name2 + '_kmeans_labels'
                numApps_in_cluster = name1 + '_' + name2 + '_numApps_in_cluster'
                group_by_var_name = name1 + '_' + name2 + '_' + group_by_var
                # ------------------------------------------------------------------------------------------
                svars = selected_vars + [text_label_var,
                                         group_by_var_name,
                                         numApps_in_cluster]
                df2 = df[svars]
                # change niche 0 1 to Broad and Niche for clearer table and graphing
                if group_by_var == 'NicheDummy':
                    df2.loc[df2[group_by_var_name] == 1, group_by_var_name] = 'Niche'
                    df2.loc[df2[group_by_var_name] == 0, group_by_var_name] = 'Broad'
                if the_panel is not None:
                    res12[name1][name2] = df2
                else:
                    # ---------- when no panel is specified, you will need the long form ----------------------
                    df2 = df2.reset_index()
                    ldf = pd.wide_to_long(
                        df2,
                        stubnames=key_vars,
                        i=['index'],
                        j="panel",
                        sep='_').reset_index()
                    ldf["panel"] = pd.to_datetime(ldf["panel"], format='%Y%m')
                    ldf["panel"] = ldf["panel"].dt.strftime('%Y-%m')
                    ldf = ldf.sort_values(by=["index", "panel"]).set_index('index')
                    res12[name1][name2] = ldf
                # ------ prepare df consisting of percentage True in each text cluster size bin for offersIAP and containsAds ------
                if the_panel is not None:
                    panel_var_list = ['offersIAPTrue_' + the_panel, 'containsAdsTrue_' + the_panel]
                    panel_value_var_list = ['TRUE_offersIAPTrue_' + the_panel, 'TRUE_containsAdsTrue_' + the_panel]
                else:
                    panel_var_list = ['offersIAPTrue_' + i for i in self.all_panels] + \
                                     ['containsAdsTrue_' + i for i in self.all_panels]
                    panel_value_var_list = ['TRUE_offersIAPTrue_' + i for i in self.all_panels] + \
                                           ['TRUE_containsAdsTrue_' + i for i in self.all_panels]
                # calculate the percentage True
                df_list = []
                for var in panel_var_list:
                    df3 = pd.crosstab(  index=df2[group_by_var_name],
                                        columns=[df2[var]],
                                        margins=True)
                    # for cases where only column 1 or column 0 exist for a sub text cluster or niche dummy group
                    if 1 not in df3.columns:
                        print(name1, name2, the_panel, var, 'column 1 does not exist.')
                        df3[1] = 0
                        print('created column 1 with zeros. ')
                    if 0 not in df3.columns:
                        print(name1, name2, the_panel, var, 'column 0 does not exist.')
                        df3[0] = 0
                        print('created column 0 with zeros. ')
                    df3['TRUE_' + var] = df3[1] / df3['All'] * 100
                    df3['FALSE_' + var] = df3[0] / df3['All'] * 100
                    df3['TOTAL_' + var] = df3['TRUE_' + var] + df3['FALSE_' + var]
                    df_list.append(df3[['TRUE_' + var]])
                df4 = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
                df4['TOTAL'] = 100 # because the text cluster group that do not exist are not in the rows, so TOTAL is 100
                df4.drop(index='All', inplace=True)
                total = df2.groupby(group_by_var_name)[var].count().to_frame()
                total.rename(columns={var: 'Total_Count'}, inplace=True)
                df5 = total.join(df4, how='left').fillna(0)
                df5.drop(columns='Total_Count', inplace=True)
                df5.reset_index(inplace=True)
                if the_panel is not None:
                    # ------- reshape to have seaborn hues (only for cross section descriptive stats) --------------------
                    # conver to long to have hue for different dependant variables
                    df6 = pd.melt(df5,
                                  id_vars=[group_by_var_name, "TOTAL"],
                                  value_vars=panel_value_var_list)
                    df6.rename(columns={'value': 'TRUE', 'variable': 'dep_var'}, inplace=True)
                    df6['dep_var'] = df6['dep_var'].str.replace('TRUE_', '', regex=False)
                    res34[name1][name2] = df6
                else:
                    # convert to long to have hue for different niche or non-niche dummies
                    ldf = pd.wide_to_long(
                        df5,
                        stubnames=['TRUE_offersIAPTrue', 'TRUE_containsAdsTrue'],
                        i=[group_by_var_name],
                        j="panel",
                        sep='_').reset_index()
                    ldf["panel"] = pd.to_datetime(ldf["panel"], format='%Y%m')
                    ldf["panel"] = ldf["panel"].dt.strftime('%Y-%m')
                    ldf = ldf.sort_values(by=["panel"])
                    res34[name1][name2] = ldf
        return res12, res34

    def _rearrange_combo_df_dict(self, d):
        """
        :param d: is any prepared/graph-ready dataframes organized in the dictionary tree in the default structure
        :return:
        """
        res = dict.fromkeys(self.graph_combo_ssnames.keys())
        for combo, name1_2 in self.graph_combo_ssnames.items():
            res[combo] = dict.fromkeys(name1_2)
        for combo in res.keys():
            for name1, content1 in self.ssnames.items():
                for name2 in content1:
                    the_name = name1 + '_' + name2
                    if the_name in res[combo].keys():
                        res[combo][the_name] = d[name1][name2]
        return res

    def graph_histogram_pricing_vars_by_niche(self, combo, the_panel):
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
            group_by_var='NicheDummy',
            the_panel=the_panel)
        res12 = self._rearrange_combo_df_dict(d=res12)
        key_vars = ['LogImputedprice', 'Imputedprice', 'LogWNImputedprice',
                    'LogImputedminInstalls', 'ImputedminInstalls']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo]['figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = self.graph_combo_ssnames[combo]  # for df names and column names name1 + name2
            for j in range(len(name1_2_l)):
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                sns.histplot(data=res12[combo][name1_2_l[j]],
                             x=key_vars[i] + "_" + the_panel,
                             hue=name1_2_l[j] + '_NicheDummy',
                             ax=ax.flat[j])
                sns.despine(right=True, top=True)
                graph_title = self.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_ylabel(self.graph_dep_vars_ylabels[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                ax.flat[j].legend().set_visible(False)
            fig.legend(labels=['Niche App : Yes', 'Niche App : No'],
                       loc='lower right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            file_keywords=key_vars[i] + '_' + combo + '_histogram_' + the_panel,
                                            # graph_title=self.multi_graph_combo_suptitle[combo] + \
                                            #             ' Cross Section Histogram \n' + \
                                            #             self.graph_dep_vars_titles[key_vars[i]] + the_panel,
                                            relevant_folder_name='pricing_vars_stats')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)


    def table_descriptive_stats_pricing_vars(self, the_panel):
        """
        The table basic is the data version of graph_descriptive_stats_pricing_vars, but putting
        all combos into a single table for each panel.
        """
        for groupby_var in ['cluster_size_bin', 'NicheDummy']:
            res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
                                        group_by_var=groupby_var,
                                        the_panel=the_panel)
            res12 = self._rearrange_combo_df_dict(d=res12)
            total_combo_df = []
            total_combo_keys = []
            for name1, value1 in res12.items():
                ldf = []
                keys_ldf = []
                for name2, value2 in value1.items():
                    groupby_var2 = name2 + '_' + groupby_var
                    df = value2.copy()
                    # --------- cluster size depand on whether you used option combine_tex_tcluster --------------------
                    df2 = df[['LogWNImputedprice_'+ the_panel,
                              'LogImputedminInstalls_'+ the_panel,
                              'offersIAPTrue_'+ the_panel,
                              'containsAdsTrue_'+ the_panel,
                              groupby_var2]].groupby(groupby_var2).describe()
                    ldf.append(df2)
                    keys_ldf.append(name2)
                df4 = pd.concat(ldf, keys=keys_ldf)
                df4= df4.round(2)
                f_name = name1 + '_pricing_vars_' + groupby_var + '_stats_panel_' + the_panel + '.csv'
                q = self.des_stats_tables_essay_1 / f_name
                df4.to_csv(q)
                if name1 != 'combo4':
                    total_combo_df.append(df4)
                    total_combo_keys.append(name1)
            df5 = pd.concat(total_combo_df, keys=total_combo_keys)
            # slicing means of all pricing vars and std and median for log price and log minimum installs
            df6 = df5.swaplevel(axis=1)
            df7 = df6.loc[:, ['mean', '50%', 'std']]
            # print(df7.columns)
            # print(df7.head())
            df8 = df7.drop(('std', 'offersIAPTrue_' + the_panel), axis=1)
            df8 = df8.drop(('std', 'containsAdsTrue_' + the_panel), axis=1)
            df8 = df8.drop(('50%', 'offersIAPTrue_' + the_panel), axis=1)
            df8 = df8.drop(('50%', 'containsAdsTrue_' + the_panel), axis=1)
            df8 = df8.drop(('combo2', 'full_full'), axis=0)
            df8 = df8.drop(('combo3', 'full_full'), axis=0)
            df8 = df8.swaplevel(axis=1)
            df8 = df8.reindex(['LogWNImputedprice_' + the_panel,
                               'LogImputedminInstalls_' + the_panel,
                               'offersIAPTrue_' + the_panel,
                               'containsAdsTrue_' + the_panel], axis=1, level=0)
            df8 = df8.reindex(['Niche', 'Broad'], axis=0, level=2)
            # print(df8.columns)
            f_name = 'ALL_SUBSAMPLES_pricing_vars_' + groupby_var + '_stats_panel_' + the_panel + '.csv'
            q = self.des_stats_tables_essay_1 / f_name
            df8.to_csv(q)
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)


    def graph_descriptive_stats_pricing_vars(self, combo, the_panel, group_by_var, graph_type):
        """
        For the containsAdsTrue and offersIAPTrue I will put them into 1 graph with different hues
        :param key_vars: 'Imputedprice','ImputedminInstalls','both_IAP_and_ADS'
        :param the_panel: '202106'
        :return:
        """
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
                                    group_by_var=group_by_var,
                                    the_panel=the_panel)
        res12 = self._rearrange_combo_df_dict(d=res12)
        res34 = self._rearrange_combo_df_dict(d=res34)
        key_vars = ['LogWNImputedprice', 'LogImputedminInstalls', 'both_IAP_and_ADS']
        group_by_var_x_order = {'NicheDummy': ['Niche', 'Broad'], 'cluster_size_bin': None}
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo]['figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = list(res12[combo].keys())
            for j in range(len(name1_2_l)):
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                if key_vars[i] in ['LogWNImputedprice', 'LogImputedminInstalls']:
                    if graph_type == 'violin':
                        sns.violinplot(
                            x= name1_2_l[j] + '_' + group_by_var,
                            order=group_by_var_x_order[group_by_var],
                            y= key_vars[i] + "_" + the_panel,
                            data=res12[combo][name1_2_l[j]],
                            color=".8",
                            inner=None,  # because you are overlaying stripplot
                            cut=0,
                            ax=ax.flat[j])
                        # overlay swamp plot with violin plot
                        sns.stripplot(
                            x= name1_2_l[j] + '_' + group_by_var,
                            order=group_by_var_x_order[group_by_var],
                            y= key_vars[i] + "_" + the_panel,
                            data=res12[combo][name1_2_l[j]],
                            jitter=True,
                            ax=ax.flat[j])
                    elif graph_type == 'box':
                        sns.boxplot(
                            x = name1_2_l[j] + '_' + group_by_var,
                            order = group_by_var_x_order[group_by_var],
                            y = key_vars[i] + "_" + the_panel,
                            data=res12[combo][name1_2_l[j]], palette="Set3",
                            ax=ax.flat[j])
                else:
                    total_palette = {"containsAdsTrue_" + the_panel: 'paleturquoise',
                                     "offersIAPTrue_"+ the_panel: 'paleturquoise'}
                    sns.barplot(x= name1_2_l[j] + '_' + group_by_var,
                                order=group_by_var_x_order[group_by_var],
                                y='TOTAL', # total does not matter since if the subsample does not have any apps in a text cluster, the total will always be 0
                                data=res34[combo][name1_2_l[j]],
                                hue="dep_var",
                                palette=total_palette,
                                ax=ax.flat[j])
                    # bar chart 2 -> bottom bars that overlap with the backdrop of bar chart 1,
                    # chart 2 represents the contains ads True group, thus the remaining backdrop chart 1 represents the False group
                    true_palette = {"containsAdsTrue_" + the_panel: 'darkturquoise',
                                    "offersIAPTrue_" + the_panel: 'teal'}
                    sns.barplot(x= name1_2_l[j] + '_' + group_by_var,
                                order=group_by_var_x_order[group_by_var],
                                y='TRUE',
                                data=res34[combo][name1_2_l[j]],
                                hue="dep_var",
                                palette=true_palette,
                                ax=ax.flat[j])
                    # add legend
                    sns.despine(right=True, top=True)
                graph_title = self.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_ylim(bottom=0)
                ax.flat[j].set_ylabel(self.graph_dep_vars_ylabels[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                if group_by_var != 'NicheDummy':
                    ax.flat[j].set_xlabel(self.group_by_var_x_label[group_by_var])
                    for tick in ax.flat[j].get_xticklabels():
                        tick.set_rotation(45)
                else:
                    ax.flat[j].set(xlabel=None)
                ax.flat[j].legend().set_visible(False)
                if key_vars[i] == 'both_IAP_and_ADS':
                    top_bar = mpatches.Patch(color='paleturquoise',
                                             label='Total (100%)')
                    middle_bar = mpatches.Patch(color='darkturquoise',
                                                label='Contains Ads (%)')
                    bottom_bar = mpatches.Patch(color='teal',
                                                label='Offers IAP (%)')
                    fig.legend(handles=[top_bar, middle_bar, bottom_bar],
                               labels=['Total (100%)', 'Contains Ads (%)', 'Offers IAP (%)'],
                               loc='upper right',
                               ncol=1,  frameon=False)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            graph_type = graph_type,
                                            file_keywords=key_vars[i] + '_' + combo + '__' + the_panel,
                                            # graph_title=self.multi_graph_combo_suptitle[combo] + \
                                            #             ' Cross Section Descriptive Statistics of \n' + \
                                            #             self.graph_dep_vars_titles[key_vars[i]] + the_panel,
                                            relevant_folder_name='pricing_vars_stats')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)


    def graph_corr_heatmap_among_dep_vars(self, the_panel):
        dep_vars = ['LogImputedprice', 'LogImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue']
        selected_vars = [i + '_' + the_panel for i in dep_vars]
        df = self.cdf.copy(deep=True)
        dep_var_df = df[selected_vars]
        correlation_matrix = dep_var_df.corr()
        f_name = the_panel + '_full_sample_dep_vars_corr_matrix.csv'
        q = self.des_stats_tables_essay_1 / f_name
        correlation_matrix.to_csv(q)
        # ------------------------------------------------
        plt.figure(figsize=(9, 9))
        labels = ['Log \nPrice', 'Log \nminInstalls', 'IAP', 'Ads']
        heatmap = sns.heatmap(correlation_matrix,
                              xticklabels=labels, yticklabels=labels,
                              vmin=-1, vmax=1, annot=True)
        filename = 'Full Sample Dependent Variables Correlation Heatmap'
        # heatmap.set_title(filename, fontdict={'fontsize': 12}, pad=12)
        plt.savefig(self.des_stats_graphs_essay_1 / 'correlation_heatmaps' / filename,
                    facecolor='white',
                    dpi=300)
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_group_mean_subsamples_parallel_trends(self, combo):
        """
        https://github.com/mwaskom/seaborn/blob/master/seaborn/relational.py
        seaborn line plot create a band for measuring central tendancy (95% CI),
        I am curious to know how they calculated the band.
        """
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
            group_by_var='NicheDummy')
        res12 = self._rearrange_combo_df_dict(d=res12)
        res34 = self._rearrange_combo_df_dict(d=res34)
        key_vars = ['LogImputedminInstalls', 'LogWNImputedprice', 'TRUE_offersIAPTrue', 'TRUE_containsAdsTrue']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo][
                                       'figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = list(res12[combo].keys())
            for j in range(len(name1_2_l)):
                nichedummy = name1_2_l[j] + "_NicheDummy"
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                hue_order = [1, 0]
                if key_vars[i] in ['LogImputedminInstalls', 'LogWNImputedprice']:
                    sns.lineplot(
                        data=res12[combo][name1_2_l[j]],
                        x="panel",
                        y= key_vars[i],
                        hue=nichedummy,
                        hue_order=hue_order,
                        markers=True,
                        style=nichedummy,
                        dashes=False,
                        ax = ax.flat[j])
                    ylimits = {'LogImputedminInstalls':{'bottom':0, 'top':25},
                               'LogWNImputedprice':{'bottom':0, 'top':2}}
                    ax.flat[j].set_ylim(bottom=ylimits[key_vars[i]]['bottom'],
                                        top=ylimits[key_vars[i]]['top'])
                else:
                    sns.lineplot(
                        data=res34[combo][name1_2_l[j]],
                        x="panel",
                        y= key_vars[i],
                        hue=nichedummy,
                        hue_order=hue_order,
                        markers=True,
                        style=nichedummy,
                        dashes=False,
                        ax = ax.flat[j])
                    ax.flat[j].set_ylim(bottom=0, top=100)
                graph_title = essay_1_stats_and_regs_201907.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                ax.flat[j].set_xlabel("Time")
                ax.flat[j].set_ylabel(self.graph_dep_vars_ylabels[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                ax.flat[j].legend().set_visible(False)
            fig.legend(labels=['Niche App : Yes', 'Niche App : No'],
                       loc='lower right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            file_keywords=key_vars[i] + '_' + combo + '_group_mean_parallel_trends',
                                            # graph_title=self.multi_graph_combo_suptitle[combo] + ' ' + \
                                            #             self.graph_dep_vars_titles[key_vars[i]] + \
                                            #             " Group Mean Parallel Trends",
                                            relevant_folder_name='parallel_trend_group_mean')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_beta_parallel_trends(self, combo, alpha):
        """
        :return: six graphs per page (each graph is 1 sub-sample), 1 page has 1 dep var, hues are leaders and non-leaders
        """
        res = self._create_cross_section_reg_results_df_for_parallel_trend_beta_graph(alpha)
        for dep_var in self.all_y_reg_vars:
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo]['figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = self.graph_combo_ssnames[combo]
            for j in range(len(name1_2_l)):
                df = res[dep_var].copy(deep=True)
                df_subsample = df.loc[df['sub_samples']==name1_2_l[j]]
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                beta_error = [df_subsample['lower_error'], df_subsample['upper_error']]
                ax.flat[j].errorbar(df_subsample['panel'],
                                    df_subsample['beta_nichedummy'],
                                    color='cadetblue',
                                    yerr=beta_error,
                                    fmt='o-', # dot with line
                                    capsize=3)
                ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                graph_title = self.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_xlabel("Time")
                ax.flat[j].set_ylabel('Niche Dummy Coefficient')
                ax.flat[j].grid(True)
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                handles, labels = ax.flat[j].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            file_keywords=dep_var + '_' + combo + '_beta_nichedummy_parallel_trends',
                                            # graph_title=essay_1_stats_and_regs_201907.multi_graph_combo_suptitle[combo] + \
                                            #       ' ' + self.graph_dep_vars_titles[dep_var] + \
                                            #       " \nRegress on Niche Dummy Coefficient Parallel Trends",
                                            relevant_folder_name='parallel_trend_nichedummy')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def cat_var_count(self, cat_var, the_panel=None):
        if the_panel is not None:
            col_name = cat_var + '_' + the_panel
            rd = self.cdf.groupby(col_name)['count_' + the_panel].count()
            if cat_var == 'minInstalls':  # minInstalls should not be sorted by the number of apps in each group, rather by index
                rd = rd.sort_index(ascending=False)
            else:
                rd = rd.sort_values(ascending=False)
            print(rd)
            return rd
        else:
            col_name = [cat_var + '_' + i for i in self.all_panels]
            df_list = []
            for j in range(len(col_name)):
                rd = self.df.groupby(col_name[j])['count_' + self.all_panels[j]].count()
                if cat_var == 'minInstalls':
                    rd = rd.sort_index(ascending=False)
                else:
                    rd = rd.sort_values(ascending=False)
                rd = rd.to_frame()
                df_list.append(rd)
            dfn = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
            print(dfn)
            return dfn

    def impute_missingSize_as_zero(self):
        """
        size is time invariant, use the mode size as the time invariant variable.
        If the size is not missing, it must not be zero. It is equivalent as having a dummies, where missing is 0 and non-missing is 1,
        and the interaction of the dummy with the original variable is imputing the original's missing as zeros.
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['size'])
        df1['size'] = df1.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        df1['size'] = df1['size'].fillna(0)
        dcols = ['size_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_contentRating_dummy(self):
        """
        contentRating dummy is time invariant, using the mode (this mode is different from previous imputation mode
        because there is no missings (all imputed).
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['ImputedcontentRating'])
        df1['contentRatingMode'] = df1.mode(axis=1, numeric_only=False, dropna=False).iloc[:, 0]
        df1['contentRatingAdult'] = df1['contentRatingMode'].apply(
            lambda x: 0 if 'Everyone' in x else 1)
        dcols = ['ImputedcontentRating_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def count_number_of_days_since_released(self):
        """
        :param var: time invariant independent variables, could either be released or updated
        :return: a new variable which is the number of days between today() and the datetime
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedreleased'])
        df1['releasedMode'] = df1.mode(axis=1, numeric_only=False, dropna=False).iloc[:, 0]
        df1['DaysSinceReleased'] = pd.Timestamp.now().normalize() - df1['releasedMode']
        df1['DaysSinceReleased'] = df1['DaysSinceReleased'].apply(lambda x: int(x.days))
        dcols = ['Imputedreleased_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_paid_dummies(self):
        """
        paid dummies are time variant
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedfree'])
        for i in self.all_panels:
            df1['paidTrue_' + i] = df1['Imputedfree_' + i].apply(lambda x: 1 if x is False else 0)
        dcols = ['Imputedfree_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def add_white_noise_to_Imputedprice(self):
        """
        Adding white noise Yt = Yt + Nt is to make sure that price fluctuation around a fixed level, so that in Leaders Medical
        sub-sample where all prices are zeros the regression continue to run.
        The results will nevertheless show that nichedummy is neither economically or statistically significant, but all we need to
        do is to show that.
        https://numpy.org/doc/stable/reference/random/index.html
        I think you should add noise before log or other transformation.
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedprice'])
        for i in self.all_panels:
            df1['WNImputedprice_' + i] = df1['Imputedprice_' + i] + np.random.uniform(low=0.01, high=0.1, size=len(df1.index))
            print(i)
            print(df1['Imputedprice_' + i].describe().round(3))
            print(df1['WNImputedprice_' + i].describe().round(3))
            print()
        dcols = ['Imputedprice_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def log_transform_pricing_vars(self):
        df1 = self._select_vars(df=self.cdf,
                                time_variant_vars_list=['Imputedprice', 'WNImputedprice', 'ImputedminInstalls'])
        for i in self.all_panels:
            df1['LogImputedprice_' + i] = np.log(df1['Imputedprice_' + i] + 1)
            df1['LogWNImputedprice_' + i] = np.log(df1['WNImputedprice_' + i] + 1)
            df1['LogImputedminInstalls_' + i] = np.log(df1['ImputedminInstalls_' + i] + 1)
            print(i)
            print(df1['LogImputedprice_' + i].describe().round(3))
            print(df1['LogWNImputedprice_' + i].describe().round(3))
            print(df1['ImputedminInstalls_' + i].describe().round(3))
            print(df1['LogImputedminInstalls_' + i].describe().round(3))
            print()
        dcols = ['Imputedprice_' + i for i in self.all_panels] \
                + ['ImputedminInstalls_' + i for i in self.all_panels] \
                + ['WNImputedprice_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return stats_and_regs(
            tcn=self.tcn,
            df=self.cdf)

    def box_cox_transform_pricing_vars(self):
        """
        I use box_cox instead of log transformation is too weak for Imputedprice and not making it any normal.
        https://towardsdatascience.com/types-of-transformations-for-better-normal-distribution-61c22668d3b9
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedprice', 'WNImputedprice', 'ImputedminInstalls'])
        for i in self.all_panels:
            df1['BCImputedprice_' + i],  lam_Imputedprice = boxcox(df1['Imputedprice_' + i] + 1)
            df1['BCWNImputedprice_' + i], lam_WNImputedprice = boxcox(df1['WNImputedprice_' + i] + 1)
            df1['BCImputedminInstalls_' + i],  lam_ImputedminInstalls = boxcox(df1['ImputedminInstalls_' + i] + 1)
        dcols = ['Imputedprice_' + i for i in self.all_panels] \
                + ['ImputedminInstalls_' + i for i in self.all_panels] \
                + ['WNImputedprice_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def create_generic_true_false_dummies(self, cat_var):
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputed' + cat_var])
        for i in self.all_panels:
            df1[cat_var + 'True_' + i] = df1['Imputed' + cat_var + '_' + i].apply(lambda x: 1 if x is True else 0)
        dcols = ['Imputed' + cat_var + '_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def create_NicheDummy(self):
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                niche_col_name = name1 + '_' + name2 + '_NicheDummy'
                self.cdf[niche_col_name] = self.cdf[label_col_name].apply(
                    lambda x: 0 if x in self.broadDummy_labels[name1][name2] else 1)
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def create_PostDummy(self):
        start_covid_us = datetime.strptime('202003', "%Y%m")
        POST_dummies = []
        for i in self.all_panels:
            panel = datetime.strptime(i, "%Y%m")
            if panel >= start_covid_us:
                self.cdf['PostDummy_' + i] = 1
                POST_dummies.append('PostDummy_' + i)
            else:
                self.cdf['PostDummy_' + i] = 0
                POST_dummies.append('PostDummy_' + i)
        print('CREATED the following post dummies:')
        print(POST_dummies)
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def create_PostXNiche_interactions(self):
        PostXNiche_dummies = []
        for i in self.all_panels:
            for name1 in self.ssnames.keys():
                for name2 in self.ssnames[name1]:
                    postdummy = 'PostDummy_' + i
                    nichedummy = name1 + '_' + name2 + '_NicheDummy'
                    postxnichedummy = name1 + '_' + name2 + '_PostXNicheDummy_' + i
                    self.cdf[postxnichedummy] = self.cdf[nichedummy] * self.cdf[postdummy]
                    PostXNiche_dummies.append(postxnichedummy)
        print('CREATED the following post niche interaction dummies:')
        print(PostXNiche_dummies)
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    def standardize_continuous_vars(self, con_var, method):
        """
        :param con_var:
        :param method: note that preprocessing sklearn transforms each feature (column)
        for example, min max transformation uses the max and min of each column, not of the entire dataframe
        :return:
        """
        df2 = self._select_vars(df=self.cdf, time_variant_vars_list=[con_var])
        print('before standardization:')
        for i in df2.columns:
            print(i)
            print(df2[i].describe().round(3))
            print()
        if method == 'zscore':
            scaler = preprocessing.StandardScaler()
            df3 = scaler.fit_transform(df2)
            df3 = pd.DataFrame(df3)
            df3.columns = ['ZScore' + i for i in df2.columns]
            df3.index = df2.index.tolist()
        print('after standardization:')
        for i in df3.columns:
            print(i)
            print(df3[i].describe().round(3))
            print()
        self.cdf = self.cdf.join(df3, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

    # I will first standardize variables and then demean them
    def create_demean_time_variant_vars(self, time_variant_vars):
        """
        Because individual dummies regression takes too much time, I decide use this for FE, so that I could also include time invariant variables.
        """
        dfs = []
        for i in time_variant_vars:
            sub_df = self._select_vars(df=self.cdf, time_variant_vars_list=[i])
            sub_df['PanelMean' + i] = sub_df.mean(axis=1)
            for p in self.all_panels:
                sub_df['DeMeaned' + i + '_' + p] = sub_df[i + '_' + p] - sub_df['PanelMean' + i]
            ts_idm = ['DeMeaned' + i + '_' + p for p in self.all_panels]
            for z in sub_df[ts_idm].columns:
                print(z)
                print(sub_df[ts_idm][z].describe().round(3))
                print()
            dfs.append(sub_df[ts_idm])
        df_new = functools.reduce(lambda a, b: a.join(b, how='inner'), dfs)
        self.cdf = self.cdf.join(df_new, how='inner')
        return stats_and_regs(
                           tcn=self.tcn,
                           df=self.cdf)

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



    # ==================================================================================================
    # COMPARE the coefficient from different samples
    # ==================================================================================================
    """
    Apr 2, 2022
    https://www.theanalysisfactor.com/compare-regression-coefficients/
    Simply include an interaction term between Sex (male/female) and any predictor whose coefficient you want to compare.  
    If you want to compare all of them because you believe that all predictors have different effects for men and women, 
    then include an interaction term between sex and each predictor.  If you have 6 predictors, that means 6 interaction terms.
    In such a model, if Sex is a dummy variable (and it should be), two things happen:
    1.the coefficient for each predictor becomes the coefficient for that variable ONLY for the reference group.
    2. the interaction term between sex and each predictor represents the DIFFERENCE in the coefficients between 
    the reference group and the comparison group.  If you want to know the coefficient for the comparison group, 
    you have to add the coefficients for the predictor alone and that predictor’s interaction with Sex.
    The beauty of this approach is that the p-value for each interaction term gives you a significance 
    test for the difference in those coefficients.
    """



