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
from matplotlib import dates
import matplotlib.patches as mpatches
from sklearn import preprocessing
from scipy.stats import boxcox
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat.python import lzip
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from linearmodels.panel import PanelOLS
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
# linux paths
#     full_sample_panel_path = Path(
#         '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/___full_sample___')
#     nlp_stats_path = Path(
#         '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/nlp_stats')
#     des_stats_tables = Path(
#         '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/TABLES')
#     des_stats_graphs = Path(
#         '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/GRAPHS')
#     ols_results = Path(
#         '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____OLS_RESULTS____')
#     panel_results = Path(
#         '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____PANEL_RESULTS____')
# win paths
    full_sample_panel_path = Path(
        "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//__PANELS__//___full_sample___")
    nlp_stats_path = Path(
        "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___/__PANELS__//nlp_stats")
    des_stats_tables = Path(
        "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____DESCRIPTIVE_STATS____//TABLES")
    des_stats_graphs = Path(
        "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____DESCRIPTIVE_STATS____//GRAPHS")
    ols_results = Path(
        "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____OLS_RESULTS____")
    panel_results = Path(
        "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____PANEL_RESULTS____")

    # ----------------------------- slicing variables --------------------------------------------------------------------
    sub_sample_key_level1 = ['FULL', 'ML', 'MF']
    FULL_sample_key_level2 = ['FULL', 'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                              'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']
    ML_sample_key_level2 = ['ML', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL']
    MF_sample_key_level2 = ['MF', 'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL']

    sub_sample_categorical_vars = ['MF_CAT', 'ML_CAT', 'FULL', 'ML_MF', 'FULL_CAT', 'FULL_TIER', 'FULL_FIRM']
    sub_sample_graph_cat_vars_d = {'FULL': ['FULL', 'ML_MF', 'FULL_CAT', 'FULL_TIER', 'FULL_FIRM'],
                                   'ML':   ['ML_CAT'],
                                   'MF':   ['MF_CAT']}

    regplot_color_palette = {'FULL':{'FULL_CAT': sns.color_palette("hls", 5),
                                     'FULL_TIER': sns.color_palette("hls", 3),
                                     'FULL_FIRM': sns.color_palette("hls", 2)},
                             'ML':{'ML_CAT': sns.color_palette("hls", 5)},
                             'MF':{'MF_CAT': sns.color_palette("hls", 5)}}

    sub_sample_d =  dict.fromkeys(['FULL', 'MF', 'ML'])

    sub_sample_l = FULL_sample_key_level2 + ML_sample_key_level2 + MF_sample_key_level2

    graph_layout_categorical = plt.subplots(3, 2)
    graph_layout_full_firm = plt.subplots(2, 1)
    graph_layout_full_tiers = plt.subplots(3, 1)

    core_dummy_y_vars_d = {'original': ['containsAdsdummy', 'offersIAPdummy', 'noisy_death',
                                        'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA'],
                           'imputed':  ['imputed_containsAdsdummy', 'imputed_offersIAPdummy',
                                        'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA']}
    core_scaled_continuous_y_vars_d = {'original': ['nlog_price', 'nlog_minInstalls'],
                                       'imputed':  ['nlog_imputed_price', 'nlog_imputed_minInstalls']}
    core_unscaled_continuous_y_vars_d = {'original': ['price', 'minInstalls'],
                                         'imputed':  ['imputed_price', 'imputed_minInstalls']}

    core_scaled_control_vars = {'original': ['score', 'nlog_reviews', 'adultcontent', 'daysreleased', 'size'],
                                'imputed': ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                            'imputed_daysreleased', 'imputed_size']}

    core_unscaled_control_vars = {'original': ['score', 'reviews', 'adultcontent', 'daysreleased', 'size'],
                                  'imputed': ['imputed_score', 'imputed_reviews', 'imputed_adultcontent',
                                              'imputed_daysreleased', 'imputed_size']}

    # niche variables are calculated from kmeans on self.tcn + 'Clean' (which is in turn based on 'imputed_'+self.tcn)
    # essentially all niche variables are imputed variables (based on imputed app descriptions)
    niche_vars = ['continuous_niche']
    # time dummies only exist in long form dataframe
    time_dummies = ['period_0', 'period_1', 'period_2', 'period_3']
    niche_time_interactions = ['period_0_continuous_niche', 'period_1_continuous_niche',
                               'period_2_continuous_niche', 'period_3_continuous_niche']
    # For the purpose of descriptive statistics, all variables are scaled and WITHOUT adding whitenoise (so that dummy stays dummy)
    des_stats_all_vars = {
        'original': {
            'continuous': ['nlog_price', 'nlog_minInstalls',
                           'score', 'nlog_reviews', 'daysreleased', 'size',
                           'continuous_niche', 'period_0_continuous_niche', 'period_1_continuous_niche',
                           'period_2_continuous_niche', 'period_3_continuous_niche'],
            'dummy': ['containsAdsdummy', 'offersIAPdummy', 'noisy_death',
                      'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA',
                      'adultcontent', 'period_0', 'period_1', 'period_2', 'period_3']
        },
        'imputed': {
            'continuous': ['nlog_imputed_price', 'nlog_imputed_minInstalls',
                           'imputed_score', 'nlog_imputed_reviews',
                           'imputed_daysreleased', 'imputed_size',
                           'continuous_niche', 'period_0_continuous_niche', 'period_1_continuous_niche',
                           'period_2_continuous_niche', 'period_3_continuous_niche'],
            'dummy': ['imputed_containsAdsdummy', 'imputed_offersIAPdummy',
                      'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA',
                      'imputed_adultcontent', 'period_0', 'period_1', 'period_2', 'period_3']
        }
    }
    # no need to specify original and imputed in scale_var_dict because every var in here has imputed form
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
                 ss_data_dict=None,
                 reg_results=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.ss_data_dict = ss_data_dict
        self.reg_results = reg_results

    # ====================== The set of functions below are regularly used common functions in pre_processing class =============================
    def _open_df(self, balanced, keyword):
        """
        :param balanced:
        :param keyword: could be any of 'merged', 'imputed', 'nlp' or 'reg_ready'
        :return:
        """
        print('------------------------ open_df ' + keyword + ' ---------------------------')
        f_name = self.initial_panel + '_' + balanced + '_' + keyword + '.pickle'
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
        f_name = self.initial_panel + '_' + balanced + '_' + keyword + '.pickle'
        q = self.full_sample_panel_path / f_name
        pickle.dump(DF, open(q, 'wb'))
        return None

    def _add_white_noise_to_all_ys(self, df, imputed):
        """
        May 11: hold off from adding white noise because we decided to run stacked OLS in only FULL, MF or ML
                with categorical dummies.
        :param
        :return:
        """
        print('------------------------ _add_white_noise_to_all_ys ----------------------')
        for var in self.core_dummy_y_vars_d[imputed] + self.core_unscaled_continuous_y_vars_d[imputed]:
            vs = [var + '_' + m for m in self.all_panels]
            for v in vs:
                print(v + ' BEFORE uniform 0 0.05')
                print(df[v].describe())
                df[v] = df[v] + np.random.uniform(low=0.0, high=0.05, size=df[v].shape)
                print(v + ' AFTER uniform 0 0.05')
                print(df[v].describe())
        return df

    def _scale_var_printout_descriptive_stats(self, df, imputed):
        """
        :param imputed should either be 'original' or 'imputed'
        :param df: A subsetted dataframe has no missing in the original variable columns
                   The existing columns in this dataframe should all be numeric
        :param scale_var_dict: key is the method and the value is a list of core variable names
        :return:
        """
        print('------------------------ _scale_var_printout_descriptive_stats ----------------------')
        for method, vars in self.scale_var_dict.items():
            for var in vars:
                if imputed == 'imputed':
                    vs = ['imputed_' + var + '_' + m for m in self.all_panels]
                else:
                    vs = [var + '_' + m for m in self.all_panels]
                for v in vs:
                    if method == 'nlog_plus_one':
                        print(v + ' BEFORE nlog_plus_one')
                        print(df[v].describe())
                        df['nlog_' + v] = df.apply(lambda row: np.log(row[v] + 1), axis=1)
                        print(v + ' AFTER nlog_plus_one')
                        print(df['nlog_' + v].describe())
        # --- convert everything to numeric before regression or graphing ---------------
        num_cols = [x for x in list(df.columns) if x not in self.sub_sample_categorical_vars]
        for i in num_cols:
            df[i] = pd.to_numeric(df[i])
        return df

    def _create_categorical_sub_sample_vars(self, df):
        """
        :param df: should be self._open_df(balanced=balanced, keyword='imputed')
        :return:
        """
        print('------------------------ _create_categorical_sub_sample_vars ----------------------')
        # print(list(df.columns))
        # --------- create categorical sub-sample slicing variables for future graphing --------
        for v in ['ML', 'MF', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL',
                  'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL',
                  'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                  'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']:
            # print(df[v].value_counts(dropna=False))
            df[v + '_cat'] = df.apply(lambda row: v if row[v] == 1 else '', axis=1)
            # print(df[v + '_cat'].value_counts(dropna=False))
        df['ML_MF'] = df['ML_cat'] + df['MF_cat']
        df['ML_CAT'] = df['ML_GAME_cat'] + df['ML_BUSINESS_cat'] + df['ML_SOCIAL_cat'] + \
                       df['ML_LIFESTYLE_cat'] + df['ML_MEDICAL_cat']
        df['MF_CAT'] = df['MF_GAME_cat'] + df['MF_BUSINESS_cat'] + df['MF_SOCIAL_cat'] + \
                       df['MF_LIFESTYLE_cat'] + df['MF_MEDICAL_cat']
        df['FULL_TIER'] = df['Tier1_cat'] + df['Tier2_cat'] + df['Tier3_cat']
        df['FULL_FIRM'] = df['top_firm_cat'] + df['non_top_firm_cat']
        df['FULL_CAT'] = df['FULL_GAME_cat'] + df['FULL_BUSINESS_cat'] + df['FULL_SOCIAL_cat'] + \
                         df['FULL_LIFESTYLE_cat'] + df['FULL_MEDICAL_cat']
        # easier for the purpose of groupby describe
        df['FULL'] = 'FULL'
        for v in ['FULL', 'ML_MF', 'ML_CAT', 'MF_CAT', 'FULL_TIER', 'FULL_FIRM', 'FULL_CAT']:
            df[v] = df[v].astype("category")
        print(df[['FULL', 'ML_MF', 'ML_CAT', 'MF_CAT', 'FULL_TIER', 'FULL_FIRM', 'FULL_CAT']].dtypes)
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
        ss_data_d = {'original': res_original, 'imputed': res_imputed}
        # ----------- open nlp k means cluster labels ------------------------------------------
        filename = self.initial_panel + '_merged_niche_vars_with_appid.pickle'
        q = self.nlp_stats_path / filename
        with open(q, 'rb') as f:
            niche_dict = pickle.load(f)
        for k, s in self.sub_sample_d.items():
            # ----------- slicing into sub-samples and merge in the nlp labels -----------------
            if k == 'FULL':
                df3 = df2.copy()
            else:
                df3 = df2.loc[df2[k]==1]
            # ------------------------------------------------------------------------------
            print(k + '--- before merging in niche variables')
            print(df3.shape)
            df3 = df3.merge(niche_dict[k][k], how='outer', left_index=True, right_index=True)
            df3 = df3.fillna(value=np.nan)
            # make the time-invariant niche variables a set of time-variant variables
            for m in self.all_panels:
                df3['continuous_niche_' + m] = df3['continuous_niche']
                df3['dummy_niche_' + m] = df3['dummy_niche']
            df3.drop(['continuous_niche', 'dummy_niche'], axis=1, inplace=True)
            print(k + '--- after merging in niche variables')
            print(df3.shape)
            for im in ['original', 'imputed']:
                # --------------------------------------------------------
                print(k + '--- delete missing in the ' + im + ' unscaled variables')
                cols = self.core_unscaled_continuous_y_vars_d[im] + self.core_dummy_y_vars_d[im]\
                       + self.core_unscaled_control_vars[im] + self.niche_vars
                cols_m = [i + '_' + m for m in self.all_panels for i in cols]
                df4 = df3.dropna(axis=0, how='any', subset=cols_m)
                df4 = df4.loc[:, cols_m + self.sub_sample_categorical_vars]
                print(df4.shape)
                # print(list(df4.columns))
                # --------------------------------------------------------
                # df4 = self._add_white_noise_to_all_ys(df=df4, imputed=im)
                df4 = self._scale_var_printout_descriptive_stats(df=df4, imputed=im)
                # --------------------------------------------------------
                ss_data_d[im][k] = df4
        self.ss_data_dict = ss_data_d
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict)

    def des_stats_continuous_niche(self, balanced):
        """
        The purpose of this is to justify your creation of dummy_niche and continuous niche.
        Note that continuous niche is 1 - app's current cluster size / largest cluster size in the sub-sample.
        This graph only applies to k-means clustering within the FULL, ML and MF samples.
        May 3, I am not going to run categorical sub-sample regressions, all regressions will include categories as dummy variables.
        So the descriptive stats and histograms will groupby categoreis (social, game, business).
        :param balanced:
        :return:
        """
        print('------------------------ des_stats_continuous_niche ---------------------------')
        for im in ['original', 'imputed']:
            dfss = []
            for k, s in self.sub_sample_d.items():
                df = self.ss_data_dict[im][k].copy()
                for m in self.all_panels:
                    for cat in self.sub_sample_graph_cat_vars_d[k]:
                        cols = ['continuous_niche_' + m, cat]
                        df_niche = df[cols]
                        # table -------------------------------------------
                        table = df_niche.groupby(cat).describe()
                        table.columns = table.columns.droplevel(0) # because the column has multiindex
                        table.reset_index(inplace=True)
                        table.rename(columns={cat: 'Sub-Samples'}, inplace=True)
                        table['month'] = m
                        table.insert(0, 'month', table.pop('month'))
                        print(table.head())
                        dfss.append(table)
                        # graph -------------------------------------------
                        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10),
                                                 sharey='row', sharex='col')
                        sns.set_style("whitegrid")
                        sns.histplot(data=df_niche,
                                     x='continuous_niche_' + m,
                                     hue=cat,
                                     multiple="dodge",
                                     shrink=0.9,
                                     bins=10,
                                     ax=axes)
                        axes.set_label(k + ' bins for continuous niche')
                        axes.set_ylabel('apps count')
                        f_name = m + '_' + k + '_' + cat + '_continuous_niche_histogram.png'
                        q = self.des_stats_graphs / balanced / 'continuous_niche' / f_name
                        fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
            # save combined table ---------------------------------------------
            final_table = pd.concat(dfss, axis=0)
            f_name = 'continuous_niche_by_sub_samples.csv'
            final_table.to_csv(self.des_stats_tables / balanced / 'continuous_niche' / f_name)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict)

    def des_stats_as_in_dissertation_format(self, balanced):
        """
        The tables generated by this function follows the format in overleaf data section
        https://www.overleaf.com/project/623deb3069c58d0c7d2ac114
        that compares both the imputed and the original variables FULL, MF and ML sub-samples for a particular month
        The continuous variables are dummy variables are summarized in two different tables.
        :return:
        """
        print('------------------------ des_stats_as_in_dissertation_format ---------------------------')
        df_list_store = {'POOLED': {'continuous': [], 'dummy': []},
                         'MONTH':  {'continuous': [], 'dummy': []}}
        for df_name in ['POOLED', 'MONTH']:
            for im in ['original', 'imputed']:
                vscon = self.des_stats_all_vars[im]['continuous']
                vsdum = self.des_stats_all_vars[im]['dummy']
                for k, s in self.sub_sample_d.items():
                    df = self.ss_data_dict[im][k].copy()
                    print(k + ' before converting to long')
                    print(df.shape)
                    dfl = self._convert_to_long_df(df=df, im=im)
                    dfl = self._create_time_dummies_and_interactions(df=dfl)
                    print(k + ' after converting to long')
                    print(dfl.shape)
                    print(list(dfl.columns))
                    # ------------------- statistics ----------------------------------------------------
                    if df_name == 'POOLED':
                        dfv = dfl[vscon]
                        tablec = dfv.describe()
                        dfv = dfl[vsdum]
                        value_counts_dfs = []
                        for v in vsdum:
                            tabled = dfv[v].value_counts(dropna=False).to_frame()
                            value_counts_dfs.append(tabled)
                    else:
                        dfv = dfl[vscon]
                        dfv.reset_index(inplace=True)
                        dfv.drop(['app_id'], axis=1, inplace=True)
                        tablec = dfv.groupby('month').describe()
                        dfv = dfl[vsdum]
                        dfv.reset_index(inplace=True)
                        dfv.drop(['app_id'], axis=1, inplace=True)
                        value_counts_dfs = []
                        for v in vsdum:
                            tabled = dfv[['month', v]].groupby('month').value_counts(dropna=False)
                            tabled.name = v
                            tabled = tabled.to_frame()
                            value_counts_dfs.append(tabled)
                    tabledum = pd.concat(value_counts_dfs, axis=1)
                    table_dict = {'continuous': tablec, 'dummy': tabledum}
                    for vtype, table in table_dict.items():
                        table.reset_index(inplace=True)
                        table['sample'] = k
                        table['im'] = im
                        table.insert(0, 'im', table.pop('im'))
                        table.insert(1, 'sample', table.pop('sample'))
                        print(table.head())
                        df_list_store[df_name][vtype].append(table)
            # save pooled continuous variables table ---------------------------------------------
            final_table = pd.concat(df_list_store[df_name]['continuous'], axis=0)
            f_name = df_name + '_CON_VAR_STATS.csv'
            final_table.to_csv(self.des_stats_tables / balanced / 'table_in_dissertation' / f_name)
            # save pooled dummy variables table --------------------------------------------------
            final_table = pd.concat(df_list_store[df_name]['dummy'], axis=0)
            f_name = df_name + '_DUM_VAR_STATS.csv'
            final_table.to_csv(self.des_stats_tables / balanced / 'table_in_dissertation' / f_name)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict)

    def table_cat_y_variables_against_niche_dummy(self, balanced):
        """
        you must run create_subsample_dict_and_merge_in_niche_vars_and_scale_vars before running this
        :param balanced:
        :return:
        """
        print('------------------------ table_cat_y_variables_against_niche_dummy ---------------------------')
        for m in self.all_panels:
            x_var = 'dummy_niche_' + m
            data_d = {'original': {'data': self.ss_data_dict['original'],
                                   'dummy_y_vars': self.core_dummy_y_vars_d['original'],
                                   'continuous_y_vars': self.core_scaled_continuous_y_vars_d['original']},
                      'imputed': {'data': self.ss_data_dict['imputed'],
                                  'dummy_y_vars': self.core_dummy_y_vars_d['imputed'],
                                  'continuous_y_vars': self.core_scaled_continuous_y_vars_d['imputed']}}
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
                dfg_dummy.to_csv(self.des_stats_tables / balanced / im / m / 'dummy_niche' / f_name)
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            ss_data_dict=self.ss_data_dict,
            reg_results=self.reg_results)

    def graph_y_variables_against_niche(self, balanced):
        """
        you must run create_subsample_dict_and_merge_in_niche_vars_and_scale_vars before running this
        :param x_var could be either 'continuous_niche' or 'dummy_niche'
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
        for im in ['original', 'imputed']:
            for k, s in self.sub_sample_d.items():
                df = self.ss_data_dict[im][k].copy()
                print(k)
                for m in self.all_panels:
                    for y_var in self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]:
                        for x_var in ['dummy_niche','continuous_niche']:
                            x_var_m = x_var + '_' + m
                            y_var_m = y_var + '_' + m
                            print('*************************** start graphing ' + y_var_m + ' against ' + x_var_m + ' *****************************')
                            var_cols = [x_var_m, y_var_m] + self.sub_sample_graph_cat_vars_d[k]
                            df2 = df[var_cols]
                            print(list(df2.columns))
                            fig, axes = plt.subplots(nrows=fig_params['x_axis_ss'][ss]['nrows'],
                                                     ncols=fig_params['x_axis_ss'][ss]['ncols'],
                                                     figsize=fig_params['x_axis_ss'][ss]['figsize'],
                                                     gridspec_kw=fig_params['x_axis_ss'][ss]['gridspec_kw'],
                                                     sharey='row', sharex='col')
                            sns.set_style("whitegrid")
                            for i in range(len(self.sub_sample_graph_cat_vars_d[k])):
                                if len(self.sub_sample_graph_cat_vars_d[k]) > 1:
                                    this_ax = axes[i]
                                else:
                                    this_ax = axes  # because a single nrows=1, ncols=1 axes is not a numpy array thus not subscriptable
                                if y_var in self.core_dummy_y_vars_d[im] and x_var == 'dummy_niche':
                                    # the bar with total niche and broad apps, with a lighter color
                                    sns.countplot(x=self.sub_sample_graph_cat_vars_d[k][i],
                                                  data=df2, hue=x_var_m, ax=this_ax,
                                                  hue_order=[1, 0], # important for legend labels
                                                  palette={1: 'pink', 0: 'lightsteelblue'})
                                    # on top of the previous total bar, the bar with y_var is True niche and broad apps, with a darker color
                                    df5 = df2.loc[df2[y_var_m]==1]
                                    if df5.shape[0] > 0:
                                        sns.countplot(x=self.sub_sample_graph_cat_vars_d[k][i],
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
                                elif y_var in self.core_dummy_y_vars_d[im] and x_var == 'continuous_niche':
                                    sns.boxplot(x=x_var_m,
                                                hue_order=[1, 0], orient='h',
                                                y=self.sub_sample_graph_cat_vars_d[k][i], hue=y_var_m,
                                                data=df2, ax=this_ax)
                                    handles, labels = this_ax.get_legend_handles_labels()
                                    labels = ['True', 'False']
                                    this_ax.set(ylabel=None)
                                    this_ax.get_legend().remove()
                                    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.15, 0.85), ncol=1,
                                           title=y_var)
                                elif y_var in self.core_scaled_continuous_y_vars_d[im] and x_var == 'dummy_niche':
                                    sns.boxplot(x=self.sub_sample_graph_cat_vars_d[k][i],
                                                hue_order=[1, 0],
                                                y=y_var_m, hue=x_var_m, data=df2, ax=this_ax)
                                    handles, labels = this_ax.get_legend_handles_labels()
                                    labels = ['Niche', 'Broad']
                                    this_ax.set(xlabel=None)
                                    this_ax.get_legend().remove()
                                    fig.legend(handles, labels, loc='upper left', bbox_to_anchor=(0.15, 0.85), ncol=1)
                                else:
                                    ss_cats = df2[self.sub_sample_graph_cat_vars_d[k][i]].unique().tolist()
                                    print(ss_cats)
                                    for j in range(len(ss_cats)):
                                        cat = ss_cats[j]
                                        the_color = self.regplot_color_palette[k][self.sub_sample_graph_cat_vars_d[ss][i]][j]
                                        print(cat)
                                        df3 = df2.loc[df2[self.sub_sample_graph_cat_vars_d[k][i]]==cat]
                                        sns.regplot(x=x_var_m, y=y_var_m,
                                                    truncate=False,
                                                    color=the_color,
                                                    data=df3, ax=this_ax,
                                                    label=cat)
                                    handles, labels = this_ax.get_legend_handles_labels()
                                    this_ax.legend(handles=handles, labels=labels, loc='upper left', bbox_to_anchor=(-0.2, 1), ncol=1)
                            # ----------------- save ---------------------------------------
                            f_name = k + '_' + y_var + '.png'
                            q = self.des_stats_graphs / balanced / im / m / x_var / f_name
                            fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            ss_data_dict=self.ss_data_dict,
            reg_results=self.reg_results)

    def heatmap_among_variables(self, balanced):
        print('*************************** heatmap_among_variables *************************** ')
        for m in self.all_panels:
            for im in ['original', 'imputed']:
                for k, s in self.sub_sample_d.items():
                    print(k)
                    df = self.ss_data_dict[im][k].copy()
                    all_ys = self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]
                    all_ys_m = [i + '_' + m for i in all_ys]
                    all_xs = self.core_scaled_control_vars[im] + self.niche_vars
                    all_xs_m = [i + '_' + m for i in all_xs]
                    var_d = {'y_vars': all_ys_m, 'x_vars': all_xs_m}
                    for name, var_ls in var_d.items():
                        df2 = df[var_ls]
                        df3 = df2.corr()
                        print(list(df2.columns))
                        print(df3)
                        mask = np.triu(np.ones_like(df3.corr(), dtype=np.bool))
                        print(mask)
                        fig, axes = plt.subplots(nrows=1,
                                                 ncols=1,
                                                 figsize=(10, 10))
                        sns.set_style("whitegrid")
                        sns.heatmap(data=df3,
                                    vmin=-1, vmax=1, annot=True,
                                    mask=mask,
                                    cmap='BrBG',
                                    ax=axes)
                        f_name = k + '_' + name + '_heatmap.png'
                        q = self.des_stats_graphs / balanced / im / m / f_name
                        fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            ss_data_dict=self.ss_data_dict,
            reg_results=self.reg_results)

    def _check_cross_sectional_ols_assumptions(self, balanced, im, niche_v, k, y, sms_results):
        """
        :param sms_results: statsmodel results object
        :return:
        """
        # normality of residual --------------------------------------------------------------
        test = sms.jarque_bera(sms_results.resid)
        test = lzip(self.jb_test_names, test) # this is a list of tuples
        test_df = pd.DataFrame(test, columns =['test_statistics', 'value'])
        f_name = k + '_' + y + '_jb_test.csv'
        save_f_name = self.ols_results / 'ols_assumptions_check' / balanced / im / niche_v / 'residual_normality' / f_name
        test_df.to_csv(save_f_name)
        # multi-collinearity -----------------------------------------------------------------
        test = np.linalg.cond(sms_results.model.exog)
        f_name = k + '_' + y + '_multicollinearity.txt'
        save_f_name = self.ols_results / 'ols_assumptions_check' / balanced / im / niche_v / 'multicollinearity' / f_name
        with open(save_f_name, 'w') as f:
            f.writelines(str(test))
        # heteroskedasticity Breush-Pagan test -------------------------------------------------
        test = sms.het_breuschpagan(sms_results.resid, sms_results.model.exog)
        test = lzip(self.bp_test_names, test)
        test_df = pd.DataFrame(test, columns =['test_statistics', 'value'])
        f_name = k + '_' + y + '_bp_test.csv'
        save_f_name = self.ols_results / 'ols_assumptions_check' / balanced / im / niche_v / 'heteroskedasticity' / f_name
        test_df.to_csv(save_f_name)
        # linearity Harvey-Collier -------------------------------------------------------------
        # this test not seem to work with my dataset because it raises singular matrix error message
        # I guess for the dummy_niche regressor, the relationship is not a linear one
        # I will visually plot the y variables against x variables to check linearity
        return None

    def price_elasticity_regression(self, balanced):
        print('----------------------------- price_elasticity_regression ---------------------------------')
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

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
        # one set of regression use continuous niche as independent var, the other set uses dummy niche
        for im in ['original', 'imputed']:
            for niche_v in self.niche_vars:
                xs = self.core_scaled_control_vars[im] + [niche_v]
                ys = self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]
                for m in self.all_panels:
                    xs_m = [i + '_' + m for i in xs]
                    ys_m = [i + '_' + m for i in ys]
                    for k, s in self.sub_sample_d.items():
                        df = self.ss_data_dict[im][k].copy()
                        for y in ys_m:
                            o_x_str = ' + '.join(xs_m)
                            formula = y + ' ~ ' + o_x_str
                            print(im + ' -- ' + k + '-- formula --')
                            print(formula)
                            results = smf.ols(formula, data=df).fit()
                            table = results.summary().tables[1].as_csv()
                            f_name = k + '_' + y + '_OLS_RESULTS.csv'
                            save_f_name = self.ols_results / 'ols_raw_results' / balanced / im / niche_v / f_name
                            with open(save_f_name, 'w') as fh:
                                fh.write(table)
                            self._check_cross_sectional_ols_assumptions(balanced=balanced,
                                                                        im=im,
                                                                        niche_v=niche_v,
                                                                        k=k, y=y,
                                                                        sms_results=results)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def stacked_cross_section_regression(self, balanced):
        """
        purpose of this function is to check whether the coefficient on dummy_niche or continuous_niche is indeed same in sub-sample regressions
        For the stacked regression, you need to have dummy for sub-samples, the interaction between sub-sample dummies and the niche variables.
        For the simplicity, we will use the single niche variable determined in the larger sample. Because the sub-sample niche variables will not
        have valid values (np.nan) for the rows that do not belong to that sub-sample.
        :param balanced:
        :return:
        """
        print('----------------------------- stacked_cross_section_regression ---------------------------------')
        # These dataframes do not contain slicing dummies, rather, they contain self.sub_sample_categorical_vars
        sub_sample_dummies = {'FULL': {'FULL_TIER': ['Tier1', 'Tier2'],
                                       'FULL_FIRM': ['top_firm'],
                                       'FULL_CAT': ['FULL_GAME', 'FULL_SOCIAL', 'FULL_BUSINESS', 'FULL_MEDICAL']},
                              'ML': {'ML_CAT': ['ML_GAME', 'ML_SOCIAL', 'ML_BUSINESS', 'ML_MEDICAL']},
                              'MF': {'MF_CAT': ['MF_GAME', 'MF_SOCIAL', 'MF_BUSINESS', 'MF_MEDICAL']}}
        for im in ['original', 'imputed']:
            for niche_v in self.niche_vars:
                xs = self.core_scaled_control_vars[im] + [niche_v]
                ys = self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]
                for m in self.all_panels:
                    ys_m = [i + '_' + m for i in ys]
                    xs_m = [i + '_' + m for i in xs]
                    for k, s in sub_sample_dummies.items():
                        x_niche_ss_dummies = []
                        df = self.ss_data_dict[im][k].copy() # selecting FULL, ML and MF sub-samples and the niche_v is in those samples rather than their sub-samples
                        for cat_v, values in s.items():
                            for value in values:
                                df[value] = df.apply(lambda row: 1 if row[cat_v] == value else 0, axis=1)
                                df[value + '_' + niche_v] = df[value] * df[niche_v + '_' + m]
                                x_niche_ss_dummies.append(value)
                                x_niche_ss_dummies.append(value + '_' + niche_v)
                        all_xs_m = xs_m + x_niche_ss_dummies
                        print(im + ' -- ' + k + ' -- ' + niche_v + ' x variables ')
                        print(all_xs_m)
                        for y in ys_m:
                            o_x_str = ' + '.join(all_xs_m)
                            formula = y + ' ~ ' + o_x_str
                            # print(formula)
                            results = smf.ols(formula, data=df).fit()
                            table = results.summary().tables[1].as_csv()
                            f_name = k + '_' + y + '_STACKED_OLS_RESULTS.csv'
                            save_f_name = self.ols_results / 'stacked_ols_raw_results' / balanced / im / niche_v / f_name
                            with open(save_f_name, 'w') as fh:
                                fh.write(table)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def _convert_to_long_df(self, df, im):
        """
        :param df:
        :param im: is either 'original' or 'imputed'
        :return:
        """
        print('----------------------------- _convert_to_long_df ---------------------------------')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'app_id'}, inplace=True)
        # print(list(df2.columns))
        core_vars = self.core_dummy_y_vars_d[im]\
                    + self.core_scaled_continuous_y_vars_d[im]\
                    + self.core_scaled_control_vars[im]\
                    + self.niche_vars
        # print('core_vars')
        # print(core_vars)
        df2 = df.loc[:, [i + '_' + m for m in self.all_panels for i in core_vars]+['app_id']]
        print(df2.shape)
        df2 = pd.wide_to_long(df2, core_vars, sep='_', i='app_id', j='month')
        # get month as a column for the purpose of creating time dummies
        df2.reset_index(inplace=True)
        # panel regression needs entity and time indices
        df2.set_index(['app_id', 'month'], drop=False, inplace=True)
        df2['month'] = df2['month'].astype(str)
        print(df2.shape)
        print(list(df2.columns))
        print(df2['month'].value_counts(dropna=False))
        print(df2.head(1))
        return df2

    def _create_time_dummies_and_interactions(self, df):
        """
        https://www.littler.com/publication-press/publication/stay-top-stay-home-list-statewide
        https://www.usatoday.com/storytelling/coronavirus-reopening-america-map/
        :param df:
        :return:
        The event study time dummy is 1 only for that period after stay-at-home order in Mar 2020.
        What I'm suggesting is turning your diff-in-diff into an event study.
        Leah's email 2022 Apr 28.
        You were estimating
        outcome_i,t = beta_0 + beta_1 treatment_i + beta_2 after_t + beta_3 treatment_i * after_t + e_i,t  (1)

        Assume there are two post-treatment periods after1 and after2 and that you have indicator variables for each period with the same name. You could instead estimate
        outcome_i,t = beta_0 + beta_1 treatment_i + beta_2 after1_t + beta_3 after2_t + beta_4 treatment_i * after1_t + beta_5 treatment_i * after2_t + e_i,t (2)

        This breaks up the beta_3 term from (1) into 2 parts in (2).  beta_3 in (1) is the weighted average of beta_4 and beta_5.

        You could be even more flexible and include indicators for each period, even before treatment.
        This allows you to assess whether there are pre-treatment changes in the treated group relative to the control.

        Since too many time dummies affect the interpretation, I am going to divide time dummies into 4-periods intervals, and the pre-covid periods are not represented by time dummies to prevent multi-collinearity
        """
        print('----------------------------- _create_time_dummies_and_interactions ---------------------------------')
        period_0 = ['202003', '202004']
        period_1 = ['202009', '202010', '202011', '202012']
        period_2 = ['202101', '202102', '202103', '202104']
        period_3 = ['202105', '202106', '202107']
        periods_after_covid = {'period_0': period_0,
                               'period_1': period_1,
                               'period_2': period_2,
                               'period_3': period_3}
        # print(periods_after_events)
        for time_dummy_name, periods in periods_after_covid.items():
            df[time_dummy_name] = df.apply(lambda row: 1 if row['month'] in periods else 0, axis=1)
            for niche_v in self.niche_vars:
                interaction = time_dummy_name + '_' + niche_v
                df[interaction] = df[time_dummy_name] * df[niche_v]
                print(df[[time_dummy_name, interaction]].describe())
        return df

    def panel_regression(self, balanced, type):
        # for continuous niche there is not treat variable
        # for dummy niche the treated is dummy_niche == 1
        cap_type = type.upper()
        folder_name = type + '_raw_results'
        print('------------------------ panel_regression ' + type + ' -----------------------------')
        for im in ['original', 'imputed']:
            for k, s in self.sub_sample_d.items():
                df = self.ss_data_dict[im][k].copy()
                # convert into long form --------------------------------------------------------------------------------------
                df_long = self._convert_to_long_df(df=df, im=im)
                # create after covid-stay-at-home time dummies and interaction with niche variable ------------------------
                df_long = self._create_time_dummies_and_interactions(df=df_long)
                time_dummies_and_interactions = [i for i in list(df_long.columns) if 'period_' in i]
                time_dummies = [i for i in time_dummies_and_interactions if 'niche' not in i]
                dummy_niche_interactions = [i for i in time_dummies_and_interactions if 'dummy_niche' in i]
                continuous_niche_interactions = [i for i in time_dummies_and_interactions if 'continuous_niche' in i]
                niche_interactions = {'dummy_niche': dummy_niche_interactions, 'continuous_niche': continuous_niche_interactions}
                # ------------ export descriptive stats of both x and y variables before regression -----------------------
                # The dummy ys are included in continuous y variables descriptive statistics because I have added white noise to it.
                all_continuous_vars = self.core_scaled_continuous_y_vars_d[im] + self.core_dummy_y_vars_d[im] + \
                           self.core_scaled_control_vars[im] + ['continuous_niche'] + continuous_niche_interactions
                continuous_stats = df_long[all_continuous_vars].describe()
                f_name = k + '_CON_REG_VARS_STATS.csv'
                continuous_stats.to_csv(self.des_stats_tables / balanced / im / 'POOLED' / f_name)
                all_dummy_vars =  dummy_niche_interactions + time_dummies + ['dummy_niche']
                dummy_stats = df_long[all_dummy_vars].value_counts()
                f_name = k + '_DUMMY_REG_VARS_STATS.csv'
                dummy_stats.to_csv(self.des_stats_tables / balanced / im / 'POOLED' / f_name)
                for niche_v in self.niche_vars:
                    xs = [niche_v] + niche_interactions[niche_v] + \
                         self.core_scaled_control_vars[im] + time_dummies
                    xsdf = sm.add_constant(df_long[xs])
                    for y in self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]:
                        ydf = df_long[[y]]
                        print(k + '---- fit model with ' + y + ' ~ ' + ' + '.join(xs))
                        if type == 'pooled_ols':
                            mod = PooledOLS(ydf, xsdf)
                        if type == 'panel_fe':
                            mod = PanelOLS(ydf, xsdf, entity_effects=True)
                        res = mod.fit(cov_type='unadjusted')
                        res_table = res.summary.as_csv()
                        f_name = k + '_' + y + '_' + cap_type + '_RESULTS.csv'
                        save_f_name = self.panel_results / folder_name / balanced / im / niche_v / f_name
                        with open(save_f_name, 'w') as fh:
                            fh.write(res_table)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def _format_pvalue_to_asterisk(self, pvalue):
        if pvalue <= 0.01:
            asterisk = '***'
        elif pvalue <= 0.05 and pvalue > 0.01:
            asterisk = '**'
        elif pvalue <= 0.1 and pvalue > 0.05:
            asterisk = '*'
        else:
            asterisk = ''
        return asterisk

    def summarize_ols_results(self, balanced, stacked):
        """
        :param balanced:
        :return:
        """
        print('----------------------------- summarize_ols_results ---------------------------------')
        if stacked is True:
            name = 'stacked_ols'
            cap_name = 'STACKED_OLS'
        else:
            name = 'ols'
            cap_name = 'OLS'
        for im in ['original', 'imputed']:
            for niche_v in self.niche_vars:
                # ------------------- create empty dataframe to hold the statistics ------------------------------
                # create different indices:
                if stacked is True:
                    sub_sample_dummies = {'FULL': ['Tier1', 'Tier2', 'top_firm', 'FULL_GAME', 'FULL_SOCIAL', 'FULL_BUSINESS', 'FULL_MEDICAL'],
                                          'ML': ['ML_GAME', 'ML_SOCIAL', 'ML_BUSINESS', 'ML_MEDICAL'],
                                          'MF': ['MF_GAME', 'MF_SOCIAL', 'MF_BUSINESS', 'MF_MEDICAL']}
                    interaction = []
                    for i in ['FULL', 'ML', 'MF']:
                        interaction = interaction + [j + '_' + niche_v for j in sub_sample_dummies[i]]
                    index2_core = interaction + [i + '_' + niche_v for i in ['FULL', 'ML', 'MF']]
                else:
                    index2_core = self.FULL_sample_key_level2 + self.MF_sample_key_level2 + self.ML_sample_key_level2
                index1 = []
                for i in self.all_panels:
                    index1 = index1 + [i] * len(index2_core)
                index2 = index2_core * len(self.all_panels)
                y_core_ls = self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]
                # print(y_core_ls)
                res_df = pd.DataFrame(columns=y_core_ls, index=[index1, index2])
                # print(res_df.head())
                for y_core in y_core_ls:
                    for m in self.all_panels:
                        # print(y)
                        y = y_core + '_' + m
                        # print(y_core)
                        for k, s in self.sub_sample_d.items():
                            f_name = k + '_' + y + '_' + cap_name + '_RESULTS.csv'
                            folder = name + '_raw_results'
                            df = pd.read_csv(self.ols_results / folder / balanced / im / niche_v / f_name)
                            # remove whitespaces in column names and the first column (which will be set as index)
                            df[df.columns[0]] = df[df.columns[0]].str.strip()
                            df.set_index(df.columns[0], inplace=True)
                            df.columns = df.columns.str.strip()
                            df['P>|t|']=df['P>|t|'].astype(np.float64)
                            # -------------------------------------------------
                            # the results has a full sample niche variables and the interaction between the full sample niche variable and sub-sample dummies
                            if stacked is True:
                                fullsample_pvalue = df.loc[niche_v+'_'+m, 'P>|t|']
                                fullsample_coef = df.loc[niche_v+'_'+m, 'coef']
                                res_df.at[(m, k + '_' + niche_v), y_core] = str(
                                    round(fullsample_coef, 2)) + self._format_pvalue_to_asterisk(fullsample_pvalue)
                                interactions = [j + '_' + niche_v for j in sub_sample_dummies[k]]
                                for v in interactions:
                                    p = df.loc[v, 'P>|t|']
                                    c = df.loc[v, 'coef']
                                    res_df.at[(m, v), y_core] = str(
                                        round(c, 2)) + self._format_pvalue_to_asterisk(p)
                            # the entire results only has one niche variable
                            else:
                                pvalue = df.loc[niche_v+'_'+m, 'P>|t|']
                                coef = df.loc[niche_v+'_'+m, 'coef']
                                res_df.at[(m, ss), y_core] = str(round(coef, 2)) + self._format_pvalue_to_asterisk(
                                    pvalue)
                                # -------------------------------------------------
                f_name = balanced + '_' + niche_v + '_' + im + '_' + name + '_cross_sectional_results.csv'
                save_f_name = self.ols_results / 'summary_results' / name / f_name
                res_df.to_csv(save_f_name)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def summarize_ols_results_in_graph(self, balanced):
        """
        should first run self.summarize_ols_results
        :param balanced:
        :return:
        """
        print('----------------------------- summarize_ols_results_in_graph ---------------------------------')
        for niche_v in self.niche_vars:
            for im in ['original', 'imputed']:
                imputed_y_core_vars = self.core_scaled_continuous_y_vars_d[im] + self.core_dummy_y_vars_d[im]
                f_name = b + '_' + niche_v + '_' + im + '_ols_cross_sectional_results.csv'
                df = pd.read_csv(self.ols_results / 'summary_results' / f_name)
                df.reset_index(inplace=True)
                df.rename(columns={
                    'Unnamed: 0': 'Month',
                    'Unnamed: 1': 'Sub-sample'}, inplace=True)
                for k, s in self.sub_sample_d.items():
                    ss_list = list(s.keys())
                    fig, axes = plt.subplots(nrows=len(imputed_y_core_vars),
                                             ncols=1,
                                             figsize=(10, 24),
                                             sharey='row', sharex='col')
                    sns.set_style("whitegrid")
                    for i in range(len(imputed_y_core_vars)):
                        y = imputed_y_core_vars[i]
                        df2 = df.loc[df['Sub-sample'].isin(ss_list), ['Month', 'Sub-sample', y]].copy()
                        df2[y] = df2[y].apply(lambda x: x.replace('*', '') if isinstance(x, str) else x)
                        df2[y] = pd.to_numeric(df2[y])
                        df2['Month'] = pd.to_datetime(df2['Month'], format='%Y%m')
                        df2 = df2.sort_values(by=['Month'])
                        sns.lineplot(data=df2, x='Month', y=y, hue='Sub-sample', ax=axes[i])
                        axes[i].set(xticks=df2.Month.values)
                        axes[i].xaxis.set_major_formatter(dates.DateFormatter("%Y%m"))
                        axes[i].tick_params(axis='x', labelrotation=90)
                        handles, labels = axes[i].get_legend_handles_labels()
                        axes[i].get_legend().remove()
                    fig.legend(handles, labels, loc='lower left', ncol=1)
                    f_name = k + '_beta_time_trend.png'
                    q = self.ols_results / 'ols_beta_graph' / balanced / im / niche_v / f_name
                    fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)





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



