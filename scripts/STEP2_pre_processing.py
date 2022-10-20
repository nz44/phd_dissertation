from pathlib import Path
import pickle
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import copy
import math
import random
from datetime import datetime
from datetime import date
import collections
import operator
import functools
import itertools
import re
import os
from tqdm.notebook import tqdm
tqdm.pandas()
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from scipy.spatial.distance import cdist
from sklearn.pipeline import Pipeline
from sklearn import metrics
vectorizer = TfidfVectorizer()
import spacy
import spacy_fastlang
nlp = spacy.load("en_core_web_sm")
nlp.add_pipe("language_detector")
import nltk
from nltk.corpus import words
import matplotlib
from matplotlib.ticker import (MultipleLocator, AutoMinorLocator)
import seaborn
import copy
#################################################################################################################
# the input dataframe are the output of merge_panels_into_single_df() method of app_detail_dicts class
class pre_processing():
    full_sample_panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/___full_sample___')
    missing_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/full_sample_missing_stats')
    imputation_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/full_sample_imputation_stats')
    check_app_descriptions  = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/full_sample_check_app_descriptions')
    nlp_graph_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/nlp_graphs')
    nlp_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/nlp_stats')
    niche_text_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/niche_check_app_descriptions')
    des_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/TABLES')
    des_stats_graphs = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/GRAPHS')

    # https://www.forbes.com/top-digital-companies/list/3/#tab:rank
    # https://companiesmarketcap.com/tech/largest-tech-companies-by-market-cap/
    # https://www.gamedesigning.org/gaming/mobile-companies/
    top_digital_firms_substring = [
         'apple inc',
         'microsoft',
         'samsung',
         'google',
         'at&t',
         'amazon',
         'verizon',
         'china mobile',
         'disney',
         'facebook',
         'alibaba',
         'intel corporation',
         'softbank',
         'ibm',
         'tencent',
         'nippon telegraph & tel',
         'cisco',
         'oracle',
         'deutsche telekom',
         'taiwan semiconductor',
         'kddi',
         'sap se',
         'telefonica',
         'america movil',
         'hon hai',
         'dell inc',
         'orange, s.a.',
         'china telecom',
         'sk hynix',
         'accenture',
         'broadcom',
         'micron',
         'qualcomm',
         'paypal',
         'china unicom',
         'hp inc',
         'bce',
         'tata',
         'automatic data processing',
         'bt group',
         'mitsubishi',
         'canon inc',
         'booking',
         'saudi telecom',
         'jd.com',
         'texas instruments',
         'netflix',
         'philips',
         'etisalat',
         'baidu',
         'asml',
         'salesforce',
         'applied materials',
         'recruit holdings',
         'singtel',
         'adobe',
         'xiaomi',
         'telstra',
         'vmware',
         'te connectivity',
         'sk holdings',
         'murata manufacturing',
         'cognizant',
         'nvidia',
         'ebay',
         'telenor',
         'vodafone',
         'sk telecom',
         'vivendi',
         'naspers',
         'infosys',
         'china tower corp',
         'swisscom',
         'corning',
         'fidelity',
         'rogers',
         'nintendo',
         'kyocera',
         'nxp semiconductors',
         'dish network',
         'rakuten',
         'altice europe',
         'telus',
         'capgemini',
         'activision blizzard',
         'analog devices',
         'lam research',
         'dxc technology',
         'legend holding',
         'lenovo',
         'netease',
         'tokyo electron',
         'keyence',
         'telkom indonesia',
         'nokia',
         'fortive',
         'ericsson',
         'fiserv',
         'fujitsu',
         'hewlett packard enterprise',
    # ------- switch to companiesmarketcap.com ---------------
         'instagram',
         'linkedin',
         'huawei',
         'tesla, inc',
         'shopify',
         'beijing kwai', # alias for kuaishou
         'kuaishou',
         'sony',
         'square, inc',
         'uber technologies',
         'zoom.us',
         'snap inc',
         'amd',
         'snowflake',
         'atlassian',
         'nxp semiconductors',
         'infineon',
         'mediatek',
         'naver',
         'crowdstrike',
         'palantir',
         'palo alto networks',
         'fortinet',
         'skyworks',
         'xilinx',
         'teladoc',
         'ringcentral',
         'unity',
         'zebra',
         'lg electronics',
         'zscaler',
         'fujifilm',
         'keysight',
         'smic',
         'slack',
         'arista networks',
         'cloudflare',
         'united microelectronics',
         'cerner',
         'qorvo',
         'yandex',
         'enphase',
         'lyft',
         'renesas',
         'coupa',
         'seagate',
         'on semiconductor',
         'citrix',
         'ase technology',
         'akamai',
         'wix',
         'qualtrics',
         'netapp',
         'entegris',
         'dynatrace',
         'asm international',
         'godaddy',
         'disco corp',
         'line corporation',
         'line games',
         'five9',
         'sina', # alias for weibo
         'mcafee',
         'dropbox',
         'rohm',
         'advantech',
         'amec',
         'teamviewer',
         'kingsoft',
         'realtek',
         'fiverr',
         'genpact',
         'fastly',
         'be semiconductor',
         'avast',
         'samanage', # alias for solarwinds
         'solarwinds',
         'descartes',
         'stitch fix',
         'riot blockchain',
         'power integrations',
         'nordic semiconductor',
         'ambarella',
        # ---------- switch to games ----------------------------------
         'blizzard entertainment',
         'electronic arts',
         'niantic',
         'bandai namco',
         'ubisoft',
         'warner bros',
         'square enix',
         'konami',
         'zynga',
         'nexon',
         'jam city',
         'gameloft',
         'supercell',
         'machine zone',
         'mixi',
         'gungho',
         'netmarble',
         'kabam games',
         'ncsoft',
         'com2us',
         'super evil megacorp',
         'disruptor beam',
         'playrix',
         'next games',
         'socialpoint',
         'dena co',
         'scopely',
         'ourpalm',
         'cyberagent',
         'pocket gems',
         'rovio entertainment',
         'space ape',
         'flaregames',
         'playdemic',
         'funplus',
         'ustwo games',
         'colopl',
         'igg.com',
         'miniclip']

    top_digital_firms_exactly_match = ['king', 'glu', 'peak', 'lumen']

    sub_sample_d = { 'FULL': dict.fromkeys(['FULL', 'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                                            'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']),
                     'ML': dict.fromkeys(['ML', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL']),
                     'MF': dict.fromkeys(['MF', 'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL'])}

    def __init__(self,
                 initial_panel,
                 all_panels,
                 tcn,
                 df=None,
                 ss_text_cols=None,
                 optimal_tf_idf_para=None,
                 tf_idf_matrices=None,
                 optimal_svd_df=None,
                 svd_matrices=None,
                 pca_matrices=None,
                 optimal_k_cluster_dict=None,
                 output_labels=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.tcn = tcn
        self.df = df
        self.ss_text_cols = ss_text_cols
        self.optimal_tf_idf_para = optimal_tf_idf_para
        self.tf_idf_matrices = tf_idf_matrices
        self.optimal_svd_df = optimal_svd_df
        self.svd_matrices = svd_matrices
        self.pca_matrices = pca_matrices
        self.optimal_k_cluster_dict = optimal_k_cluster_dict
        self.output_labels = output_labels
    # 20220922 note: I deleted slice_text_cols_for_sub_samples because I only do NLP for the full sample and I will delete non-english text descriptions,
    # and use frequent based term to do clustering (Beil, Ester, Xu 2002)
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

    def _select_dfs_per_var(self, var_list, select_one_panel=None):
        """
        return a list of dataframes that each df consists of the same variable across all months/or one month
        """
        print('----------- _select_dfs_per_var ----------------')
        dfs = []
        for var in var_list:
            if select_one_panel is None:
                col_list = [var + '_' + i for i in self.all_panels]
            else:
                col_list = [var + '_' + select_one_panel]
            df2 = self.df.copy()
            df2 = df2[col_list]
            print('selected the following columns for the df ', df2.shape)
            print(col_list)
            dfs.append(df2)
        return dfs

    # this will create a list of dataframes that each df contains only var_list to that month
    def _select_dfs_per_month(self, var_list):
        """
        return a dictionary of dataframes that each df consists a list of variables in the same month, the key is the month
        should only include time-variant variables in the var_list, should not include 'merge_'
        """
        print('----------------------- _select_dfs_per_month ------------------------')
        dfs = {}
        print(var_list)
        for i in self.all_panels:
            col_list = [var + '_' + i for var in var_list]
            df2 = self.df.copy()
            print(col_list)
            df2 = df2[col_list]
            print('selected the following columns for the df ', df2.shape)
            dfs[i] = df2
        return dfs

    def _create_mode_column(self, var):
        """
        By default, the mode is created using self.all_panels of the var.
        If the mode of a variables over months are np.nan, we will make sure it does nto get into the mode column
        """
        print('------------------------ _create_mode_column ---------------------------')
        self.df = self.df.fillna(value=np.nan)
        df = self.df.copy(deep=True)
        col_list = [var + '_' + i for i in self.all_panels]
        df2 = df[col_list]
        self.df['mode_' + var] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        print('the unqiue values in the mode of ' + var)
        print(self.df['mode_' + var].value_counts(dropna=False))
        return self.df

    def _create_mean_column(self, var):
        """
        By default, the mean is created using self.all_panels of the var
        """
        print('-------------------- _create_mean_column ----------------------')
        self.df = self.df.fillna(value=np.nan)
        df = self.df.copy(deep=True)
        col_list = [var + '_' + i for i in self.all_panels]
        df2 = df[col_list]
        self.df['mean_' + var] = df2.mean(axis=1, skipna=True)
        print('the summary statistics of the month average of ' + var)
        print(self.df['mean_' + var].describe())
        return self.df

    def _checking_consecutive_values(self, df, var, check_nan, check_key_value, new_col):
        """
        this is a helper function where the input is a dataframe containing only the variable across all months in the panel
        var is the variable name
        if check_nan is true, we are checking after a period the variable is all nans till the end of the panel
        check_key_value is a string or a number that we want to know from which period the row has this key value till the end of the panel
        we are checking this by iterate through the first month to the last month
        for example, if app id 1234 is completely missing in period 6 (merge_6 == left), and has some data in period 7 (merge_7 = both),
        and completely missing in period 8, 9, 10 (merge_8,9,10 == left) till the last period
        then we measure app id 1234 as death starting from period 8. However if an app is only missing for one period and has data till the end,
        it does not count towards dead apps.
        after check the condition for this backward period, let us check the total number so far,
        if the total figure equals to the number of periods we have checked starting from the last period
        that means the apps has been dead since this point onward till the end of the panel
        Then we could update the new_col with this period's datetime string.
        Then the loop will go one period earlier in time, if that period satisfied, then new_col will update to that period's datetime string.
        however if that period does not satisfy, then the total figure will be less than the number of periods we've checked,
        we will keep the new_col unchaged (it is either in the period it satisfied or 0 initial condition)
        Even if any of the earlier period satisfy again, the total figure will not equal to the number of periods we've checked,
        so the new_col will not be updated.
        """
        print('---------------------- _checking_consecutive_values -----------------------------')
        # sort the all panels from the most recent to the oldest
        # I do not use self.all_panels because not all variables exist for each panel (merge_month) does not exist for the first month
        var_month = list(df.columns)
        substr = var + '_'
        all_months = [x.replace(substr, '') for x in var_month]
        new_panel = [datetime.strptime(i, "%Y%m") for i in all_months]
        forward_month = sorted(new_panel)
        backward_month = sorted(new_panel, reverse=True)
        forward_month = [datetime.strftime(i, "%Y%m") for i in forward_month]
        backward_month = [datetime.strftime(i, "%Y%m") for i in backward_month]
        print('forward month')
        print(forward_month)
        print('backward month')
        print(backward_month)
        df[new_col] = 0
        for i in range(len(backward_month)):
            v = var + '_' + backward_month[i]
            if i > 0:
                if check_nan is True:
                    pass
                else:
                    # axis = 1 means v refers to column names, if axis =0, v must refer to row index
                    df['conditions_satisfied'] = df.apply(lambda row: row['conditions_satisfied']+1 if row[v] == check_key_value else row['conditions_satisfied']+0, axis=1)
            else:
                if check_nan is True:
                    pass
                else:
                    df['conditions_satisfied'] = df.apply(lambda row: 1 if row[v] == check_key_value else 0, axis=1)
            df[new_col] = df.apply(lambda row: backward_month[i] if row['conditions_satisfied'] == i+1 else row[new_col], axis=1)
        print('the unqiue values in conditions_satisfied')
        print(df['conditions_satisfied'].value_counts(dropna=False))
        print('the unqiue values in ' + new_col)
        print(df[new_col].value_counts(dropna=False))
        return df

    def _impute_missing_by_func(self, func, preffix, var, sequence, m):
        """
        :param func: functional input, usually mode or mean
        :param preffix: 'mean' or 'mode'
        :param var:
        :param sequence: 0 means it is the first imputation method, that convert var + '_' + p to 'imputed_' + var + '_' + p
                          other numbers means the input variable already 'imputed_' in var name.
        :return:
        """
        print('---------------------- _impute_missing_by_func -----------------------------')
        if sequence == 0:
            input_var = var
        else:
            input_var = 'imputed_' + var
        imputed_var = 'imputed_' + var
        self.df = func(var=input_var)
        self.df[imputed_var + '_' + m] = self.df.apply(
            lambda row: row[preffix + '_' + input_var] \
                if pd.isnull(row[input_var + '_' + m])\
                else row[input_var + '_' + m], axis=1)
        return self.df

    def _impute_missing_by_mode_if_all_months_are_the_same(self, var, sequence, m):
        """
        :param m: the missing month, one of self.all_panels
        :param var:
        :param sequence: 0 means it is the first imputation method, that convert var + '_' + p to 'imputed_' + var + '_' + p
                          other numbers means the input variable already 'imputed_' in var name.
        :return:
        """
        print('---------------------- _impute_missing_by_mode_if_all_months_are_the_same -----------------------------')
        if sequence == 0:
            input_var = var
            cols = [var + '_' + p for p in self.all_panels]
        else:
            input_var = 'imputed_' + var
            cols = ['imputed_' + var + '_' + p for p in self.all_panels]
        imputed_var = 'imputed_' + var
        self.df = self._create_mode_column(var=input_var)
        def _check_unique_value_per_row(row):
            ls = []
            for i in cols:
                if pd.notnull(row[i]):
                    ls.append(row[i])
            n = len(set(ls))
            if n == 1:
                return True
            else:
                return False
        self.df['non-missings-are-same'] = self.df.apply(lambda row:
                                                         1 if _check_unique_value_per_row(row) else 0, axis=1)
        self.df[imputed_var + '_' + m] = self.df.apply(lambda row:
                                                         row['mode_' + input_var] if row['non-missings-are-same'] == 1\
                                                             else row[input_var + '_' + m], axis=1)
        return self.df

    def _impute_missing_with_zero_in_initial_month(self, var, sequence):
        """
        :param sequence: 0 means it is the first imputation method, that convert var + '_' + p to 'imputed_' + var + '_' + p
                         other numbers means the input variable already 'imputed_' in var name.
        :return:
        """
        print('---------------------- _impute_missing_with_zero_in_initial_month -----------------------------')
        if sequence == 0:
            input_var = var
        else:
            input_var = 'imputed_' + var
        self.df['imputed_' + var + '_' + self.initial_panel] = self.df.apply(
            lambda row: 0 if pd.isnull(row[input_var + '_' + self.initial_panel])\
                else row[input_var + '_' + self.initial_panel], axis=1)
        return self.df

    def _impute_missing_by_previous_nonmissing_column(self, var, sequence, m):
        """
        This is used in imputing pricing variables that suppose to change over time and may follow a trend.
        For the_missing_month, we will seek its closest non-missing month back in time (NOT forward in time).
        the_missing_month should be one of self.all_panels, eg. '202107'
        suffix is the newly created column's suffix
        :param sequence: 0 means it is the first imputation method, that convert var + '_' + p to 'imputed_' + var + '_' + p
                          other numbers means the input variable already 'imputed_' in var name.
        """
        print('----------- _impute_missing_by_previous_nonmissing_column ----------------')
        self.df = self.df.fillna(value=np.nan)
        if sequence == 0:
            v_original_m = var + '_' + m
            v_original = var
        else:
            v_original_m = 'imputed_' + var + '_' + m
            v_original = 'imputed_' + var
        v_imputed_m = 'imputed_' + var + '_' + m
        print('v_original : ' + v_original_m + ' ' + 'v_imputed : ' + v_imputed_m)
        # ------ get all the previous months and sort it from the closest (to today) to the furthest (to today) --------------
        # you cannot impute using the previous month for the data in the first month
        # and if the missing month is not the initial month, you sill need to start somewhere, so first assign the
        # imputed month to its original month
        self.df[v_imputed_m] = self.df[v_original_m]
        # ------------------------ for the missing month which is not the initial month --------------------------------------
        if m != self.initial_panel:
            the_missing_month_dt = datetime.strptime(m, "%Y%m")
            all_the_previous_months = []
            for p in self.all_panels:
                month = datetime.strptime(p, "%Y%m")
                if month < the_missing_month_dt:
                    all_the_previous_months.append(month)
            backward_previous_months = sorted(all_the_previous_months, reverse=True)
            backward_previous_months = [datetime.strftime(i, "%Y%m") for i in backward_previous_months]
            print('impute using the following previous months: ')
            print(backward_previous_months)
            # ----- iterate month data and stop until you find a non-missing value -----------
            for i in backward_previous_months:
                self.df[v_imputed_m] = self.df.apply(
                    lambda row: row[v_original + '_' + i]\
                        if (pd.isnull(row[v_imputed_m]) and pd.notnull(row[v_original + '_' + i]))\
                        else row[v_imputed_m], axis=1)
        # print(df[v_imputed].isna().sum())
        return self.df

    def _impute_missing_by_any_other_nonmissing_column(self, var, sequence):
        """
        :param var: only for imputing time-INvariant variables
        :param sequence: 0 means it is the first imputation method, that convert var + '_' + p to 'imputed_' + var + '_' + p
                          other numbers means the input variable already 'imputed_' in var name.
        :return:
        """
        print('----------- _impute_missing_by_any_other_nonmissing_column ----------------')
        self.df = self.df.fillna(value=np.nan)
        all_the_other_months = []
        for month in self.all_panels:
            if month != self.initial_panel:
                all_the_other_months.append(month)
        print('delete the initial panel for all the other month')
        print(all_the_other_months)
        # since the order does not matter, we do not need to sort all the other months from the earliest to the most recent
        if sequence == 0:
            input_var_name = var
        else:
            input_var_name = 'imputed_' + var
        self.df['imputed_' + var] = self.df[input_var_name + '_' + self.initial_panel]
        for i in all_the_other_months:
            self.df['imputed_' + var] = self.df.apply(
                lambda row: row[input_var_name + '_' + i] \
                    if (pd.isnull(row['imputed_' + var]) and pd.notnull(row[input_var_name + '_' + i]))\
                    else row['imputed_' + var], axis=1)
        # check the number of rows that have missing in app descriptions in all months (which cannot be imputed)
        return self.df

    # count the rows that are all nan in each month
    def _count_missing(self,
                      time_variant_var_list,
                      before_imputation,
                      balanced,
                      time_invariant_var_list=None):
        print('----------------------- _count_missing -------------------------')
        if balanced is True:
            b = 'balanced'
        else:
            b = 'unbalanced'
        if before_imputation is True:
            im = 'before_imputation'
        else:
            im = 'after_imputation'
        self.df = self.df.fillna(value=np.nan)
        if before_imputation is True:
            # time_invariant var usually comes in with month in its names because of the way the data has been scraped in every month
            dfs = self._select_dfs_per_month(var_list=time_variant_var_list + time_invariant_var_list)
            all_vars = time_variant_var_list + time_invariant_var_list
        else:
            imputed_time_variant_var_list = ['imputed_' + i for i in time_variant_var_list]
            imputed_time_invariant_var_list = ['imputed_' + i for i in time_invariant_var_list]
            dfs = self._select_dfs_per_month(var_list=imputed_time_variant_var_list)
            all_vars = imputed_time_variant_var_list + imputed_time_invariant_var_list
        d_keys = copy.deepcopy(all_vars)
        d_keys = ['rows_missing_in_' + i for i in d_keys]
        d_keys.insert(0, 'rows_original')  # insert 10 at 4th index
        d_keys.insert(1, 'rows_missing_in_any')
        d_keys.insert(2, 'rows_missing_in_all')
        summary_df = pd.DataFrame(
            columns=d_keys,
            index=list(dfs.keys()) + ['time_invariant'])
        for month, df in dfs.items():
            print(month, ' before deleting nans : ', df.shape)
            rows_original = df.shape[0]
            summary_df.at[month, 'rows_original'] = rows_original
            cols = df.columns.values.tolist()
            df2 = df.dropna(axis=0, how='any', subset=cols)
            summary_df.at[month, 'rows_missing_in_any'] = rows_original - df2.shape[0]
            df2 = df.dropna(axis=0, how='all', subset=cols)
            summary_df.at[month, 'rows_missing_in_all'] = rows_original - df2.shape[0]
            for v in cols:
                df2 = df.dropna(axis=0, how='all', subset=[v])
                v_col = v.replace('_' + month, '')
                v_col = 'rows_missing_in_' + v_col
                summary_df.at[month, v_col] = rows_original - df2.shape[0]
            if before_imputation is False:
                for v in imputed_time_invariant_var_list:
                    dfc = self.df.copy()
                    df3 = dfc.dropna(axis=0, how='all', subset=[v])
                    v_col = 'rows_missing_in_' + v
                    summary_df.at['time_invariant', v_col] = rows_original - df3.shape[0]
        # -------------------- save the summary dataframe -----------------------------
        filename = self.initial_panel + '_count_missing_by_month_and_in_each_var.csv'
        q = self.missing_stats_path / im / b / filename
        summary_df.to_csv(q)
        return None

    def impute_missing(self, imputation_var_methods_dict, balanced):
        """
        :param imputation_var_methods_dict: A dictionary with variable name as key and imputation methods as value ('mode', 'mean', 'any_other' and 'previous')
        :return: it will update self.df with the imputed columns
        """
        print('------------------------- impute_missing -------------------------------------')
        print('Open merge dataframe')
        self.df = self._open_df(balanced=balanced, keyword='merge')
        # print(list(self.df.columns))
        tvls = list(imputation_var_methods_dict['time_variant'].keys())
        tils = list(imputation_var_methods_dict['time_invariant'].keys())
        self._count_missing(   time_variant_var_list=tvls,
                               before_imputation=True,
                               balanced=balanced,
                               time_invariant_var_list=tils)
        for var_type, content in imputation_var_methods_dict.items():
            if var_type == 'time_invariant':
                print('--------------- impute_time_invariant_missing ------------------------')
                for var, method_ls in content.items():
                    for method in method_ls: # time invariant imputation has not chained operations
                        print('BEFORE IMPUTATION total missing ' + var + '----' + self.initial_panel)
                        print(self.df[var + '_' + self.initial_panel].isna().sum())
                        print('TIME INVARIANT Start imputing ' + var + ' with ' + method)
                        if method == 'any_other':
                            self.df = self._impute_missing_by_any_other_nonmissing_column(var=var, sequence=0)
                        if method == 'mode': # mode is generally used to impute for time-invariant variables
                            self.df = self._create_mode_column(var=var)
                            self.df['imputed_' + var] = self.df.apply(
                                lambda row: row['mode_' + var] \
                                    if pd.isnull(row[var + '_' + self.initial_panel]) else row[var + '_' + self.initial_panel], axis=1)
                        print('AFTER IMPUTATION USING ' + method + ' METHOD total missing ' + var)
                        print(self.df['imputed_' + var].isna().sum())
            if var_type == 'time_variant':
                print('--------------- impute_time_variant_missing ------------------------')
                for var, method_ls in content.items():
                    for p in self.all_panels:
                        print('BEFORE IMPUTATION total missing ' + var + '----' + p)
                        print(self.df[var + '_' + p].isna().sum())
                        print('TIME VARIANT Start imputing ' + var + ' with ' + ' -- '.join(method_ls))
                        for m in range(len(method_ls)):
                            if method_ls[m] == 'mode':
                                self.df = self._impute_missing_by_func(var=var, func=self._create_mode_column,
                                                                       preffix='mode', sequence=m, m=p)
                            if method_ls[m] == 'mean':
                                self.df = self._impute_missing_by_func(var=var, func=self._create_mean_column,
                                                                       preffix='mean', sequence=m, m=p)
                            if method_ls[m] == 'mode if all months are the same':
                                self.df = self._impute_missing_by_mode_if_all_months_are_the_same(var=var, sequence=m, m=p)
                            if method_ls[m] == 'zero for missing in the first month':
                                self.df = self._impute_missing_with_zero_in_initial_month(var=var, sequence=m)
                            if method_ls[m] == 'previous':
                                self.df = self._impute_missing_by_previous_nonmissing_column(
                                    var=var, sequence=m, m=p)
                        print('AFTER IMPUTATION USING ' + ' -- '.join(method_ls) + ' METHOD total missing ' + var + '----' + p)
                        print(self.df['imputed_' + var + '_' + p].isna().sum())
        print('Finished imputing the dataframe')
        # print(self.df.columns)
        self._count_missing(   time_variant_var_list=tvls,
                               before_imputation=False,
                               balanced=balanced,
                               time_invariant_var_list=tils)
        print('saving the imputed dataframe')
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def create_adultcontent_var(self, balanced):
        print('-------------------- create_adultcontent_var -------------------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        # ---------------------------------------------------
        def _adult_dummy(x):
            if pd.isnull(x):
                return np.nan
            elif isinstance(x, list):
                x = " ".join(x)
            if any([i in x for i in ['Nudity', 'Sexual', 'Mature 17+', 'Adults only 18+']]):
                return 1
            else:
                return 0
        # ---------------------------------------------------
        for m in self.all_panels:
            print(self.df['contentRating_' + m].value_counts(dropna=False))
            self.df['adultcontent_' + m] = self.df.apply(
                lambda row: _adult_dummy(row['contentRating_' + m]), axis=1)
            print(self.df['adultcontent_' + m].value_counts(dropna=False))
            print(self.df['imputed_contentRating_' + m].value_counts(dropna=False))
            self.df['imputed_adultcontent_' + m] = self.df.apply(
                lambda row: _adult_dummy(row['imputed_contentRating_' + m]), axis=1)
            print(self.df['imputed_adultcontent_' + m].value_counts(dropna=False))
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def create_daysreleased_var(self, balanced):
        print('-------------------- create_daysreleased_var -------------------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        print(list(self.df.columns))
        self.df = self.df.fillna(value=np.nan)
        # ----------------------------------------------------------------------
        # This column is already in datetime format
        # create imputed daysreleased based on imputed_released and daysreleased based on released
        v_names = {'released': 'daysreleased', 'imputed_released': 'imputed_daysreleased'}
        for m in self.all_panels:
            for r_v, days_v in v_names.items():
                print(self.df[r_v+'_'+m].dtype)
                print(self.df[r_v+'_'+m].describe())
                month_dt = datetime.strptime(m, "%Y%m")
                self.df[days_v+'_'+m] = self.df.apply(lambda row: month_dt - row[r_v+'_'+m], axis=1)
                self.df[days_v+'_'+m] = self.df[days_v+'_'+m].dt.days
                # since released is imputed by mode (faster than previous, and this variable is not worth the time to use previous method)
                # we are going to change the negative datetime to 0
                self.df[days_v+'_'+m] = self.df.apply(lambda row: 0\
                    if row[days_v+'_'+m] < 0 else row[days_v+'_'+m], axis=1)
                print(self.df[days_v+'_'+m].dtype)
                print(self.df[days_v+'_'+m].describe())
        print(list(self.df.columns))
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def convert_true_false_to_1_0_dummies(self, balanced, vl):
        print('-------------------- convert_offersIAP_and_containAds_to_numeric -------------------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        def _convert_1_0(x):
            if pd.isnull(x):
                return np.nan
            elif x is True:
                return 1
            elif x is False:
                return 0
            elif isinstance(x, str) and x.strip() == 'True':
                return 1
            elif isinstance(x, str) and x.strip() == 'False':
                return 0
            elif isinstance(x, int):
                return x # part of the imputation 'zero for missing in the first month'
        for v in vl:
            for m in self.all_panels:
                print(self.df[v + '_' + m].value_counts(dropna=False))
                self.df[v + 'dummy_' + m] = self.df.apply(
                    lambda row: _convert_1_0(row[v + '_' + m]), axis=1)
                print(self.df[v + 'dummy_' + m].value_counts(dropna=False))
                print(self.df['imputed_' + v + '_' + m].value_counts(dropna=False))
                self.df['imputed_' + v + 'dummy_' + m] = self.df.apply(
                    lambda row: _convert_1_0(row['imputed_' + v + '_' + m]), axis=1)
                print(self.df['imputed_' + v + 'dummy_' + m].value_counts(dropna=False))
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    # =========== The set of functions below create time variant variables that exist only in the unbalanced panel ============
    # ---UNBALANCED PANEL--- replace merged_month columns with noisy_death -----------------------------------------------------------------
    def unbalanced_panel_create_noisy_death_dummy(self):
        print('----------- unbalanced_panel_create_noisy_death_dummy ----------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=False, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        # print(df2.head())
        df = self.df.copy()
        merged_list = []
        for i in df.columns:
            if 'merge_' in i:
                merged_list.append(i)
        df2 = df[merged_list]
        df3 = self._checking_consecutive_values(df=df2,
                                                var='merge',
                                                check_nan=False,
                                                check_key_value='left_only',
                                                new_col='app_death_month')
        print('before merging in app_death_month ' + str(self.df.shape[0]) + '--' + str(self.df.shape[1]))
        # if you opened an imputed dataframe that with the app_death_month already created, None means the already created column name will stay as it is
        self.df = self.df.merge(df3[['app_death_month']],
                                how='inner',
                                left_index=True, right_index=True,
                                suffixes = [None, '_y'])
        print('after merging in app_death_month ' + str(self.df.shape[0]) + '--' + str(self.df.shape[1]))
        # create noisy death dummy
        for i in range(len(self.all_panels)):
            if i == 0:
                # the initial month, there is no merge_ variable, and so no app dies in the first month
                self.df['noisy_death_' + self.all_panels[i]] = 0
            else:
                self.df['noisy_death_' + self.all_panels[i]] = self.df.apply(
                    lambda row: 1 if row['app_death_month'] == self.all_panels[i] else row['noisy_death_' + self.all_panels[i-1]], axis=1)
            print(self.df['noisy_death_' + self.all_panels[i]].value_counts(dropna=False))
        # ls = ['noisy_death_' + i for i in self.all_panels]
        # print(self.df[ls].head())
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=False, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    # ---UNBALANCED PANEL--- create dummies indicating the app has transitioned from market follower to market leader ---------------------------
    # according to our definition of market leaders and followers, they are either switched to a top firm or minInstalls increase over a threshhold of
    def unbalanced_panel_create_TRANSITION_tier1_minInstalls(self):
        # according to the definition of tier1 minInstalls sample (>= 1.000000e+07), we will create variable  assign the app 1 if in
        print('----------- unbalanced_panel_create_TRANSITION_dummy ----------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=False, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        df = self.df.copy()
        # the new variables is a panel variables transition to tier 1 minInstalls
        for i in range(len(self.all_panels)):
            if i == 0:
                df['T_TO_TIER1_minInstalls_' + self.all_panels[i]] = 0
            else:
                df['T_TO_TIER1_minInstalls_' + self.all_panels[i]] = df.apply(
                    lambda row: 1\
                        if (row['imputed_minInstalls_' + self.all_panels[i-1]] < 1.000000e+07\
                            and row['imputed_minInstalls_' + self.all_panels[i]] >= 1.000000e+07)\
                        else row['T_TO_TIER1_minInstalls_' + self.all_panels[i-1]], axis=1)
            print(df['T_TO_TIER1_minInstalls_' + self.all_panels[i]].value_counts(dropna=False))
        self.df = df
        # ls = ['T_TO_TIER1_minInstalls_' + i for i in self.all_panels]
        # print(self.df[ls].head())
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=False, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def _format_text_for_developer(self, text):
        if text is not None:
            result_text = ''.join(c.lower() for c in text if not c.isspace())  # remove spaces
            punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~+'''  # remove functuations
            for ele in result_text:
                if ele in punc:
                    result_text = result_text.replace(ele, "")
            extra1 = re.compile(
                r'(corporated$)|(corporation$)|(corp$)|(company$)|(limited$)|(games$)|(game$)|(studios$)|(studio$)|(mobile$)')
            extra2 = re.compile(
                r'(technologies$)|(technology$)|(tech$)|(solutions$)|(solution$)|(com$)|(llc$)|(inc$)|(ltd$)|(apps$)|(app$)|(org$)|(gmbh$)')
            res1 = re.sub(extra1, '', result_text)
            res2 = re.sub(extra2, '', res1)
        else:
            res2 = np.nan
        return res2

    # ----- for both balanced and unbalanced panel -----------------------------------------------------------------------------------------------
    def create_top_firm_dummy(self, balanced):
        """
        https://www.forbes.com/top-digital-companies/list/#tab:rank
        The default calculate only the time-invariant top firm dummy, if you set add_time_variant == True,
        the function will also calculate the time-variant top firm dummy
        developerClean is based on imputed_developer
        :return:
        """
        print('------------------------------ create_top_firm_dummy ----------------------------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        print('------------------------------ create_time_invariant_top_firm_dummy ----------------------------------')
        for p in self.all_panels:
            self.df['developerClean_' + p] = self.df.apply(
                lambda row: self._format_text_for_developer(row['imputed_developer_' + p])\
                    if pd.notnull(row['imputed_developer_' + p]) else "NO_DEVELOPER", axis=1)
        self.df = self._create_mode_column(var='developerClean')
        self.df['top_firm'] = self.df.apply(
            lambda row: 1 \
                if (any([i in row['mode_developerClean'] for i in self.top_digital_firms_substring]) or row[
                'mode_developerClean'] in self.top_digital_firms_exactly_match) \
                else 0, axis=1)
        self.df['top_firm'] = self.df.apply(
            lambda row: np.nan if row['mode_developerClean'] == "NO_DEVELOPER" else row['top_firm'], axis=1)
        print(self.df['top_firm'].value_counts(dropna=False))
        if balanced is False:
            print('------------------------------ create_time_variant_top_firm_dummy ----------------------------------')
            for p in self.all_panels:
                self.df['top_firm_'+p] = self.df.apply(
                    lambda row: 1 \
                        if (any([i in row['developerClean_' + p] for i in self.top_digital_firms_substring])\
                            or row['developerClean_' + p] in self.top_digital_firms_exactly_match)\
                        else 0, axis=1)
                self.df['top_firm_' + p] = self.df.apply(
                    lambda row: np.nan if row['developerClean_' + p] == "NO_DEVELOPER" else row['top_firm_' + p], axis=1)
                print(self.df['top_firm_' + p].value_counts(dropna=False))
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def unbalanced_panel_create_TRANSITION_top_firm_and_MA_dummy(self):
        """
        merger and acquisition dummy is 1 in a period if the app's developerClean is different from the previous one
        developerClean is based on imputed_developer
        """
        print('------------------ unbalanced_panel_create_TRANSITION_top_firm ----------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=False, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        df = self.df.copy()
        # the new variables is a panel variables transition to tier 1 minInstalls
        for i in range(len(self.all_panels)):
            if i == 0:
                df['T_TO_top_firm_' + self.all_panels[i]] = 0
                df['MA_' + self.all_panels[i]] = 0
            else:
                df['T_TO_top_firm_' + self.all_panels[i]] = df.apply(
                    lambda row: 1\
                        if (row['top_firm_' + self.all_panels[i-1]] == 0 and row[
                        'top_firm_' + self.all_panels[i]] == 1)\
                        else row['T_TO_top_firm_' + self.all_panels[i-1]], axis=1)
                df['MA_' + self.all_panels[i]] = df.apply(
                    lambda row: 1\
                        if row['developerClean_' + self.all_panels[i-1]] != row['developerClean_' + self.all_panels[i]]\
                        else row['MA_' + self.all_panels[i-1]], axis=1)
            print(df['T_TO_top_firm_' + self.all_panels[i]].value_counts(dropna=False))
            print(df['MA_' + self.all_panels[i]].value_counts(dropna=False))
        # ----------------- assign and show the results -----------------------------
        self.df = df
        # ls = ['T_TO_top_firm_' + i for i in self.all_panels]
        # print(self.df[ls].head())
        # ls = ['MA_' + i for i in self.all_panels]
        # print(self.df[ls].head())
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=False, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def create_sub_sample_dummies(self, balanced):
        """
        This way you do not need to seperately save data as sub-samples
        """
        print('----------------------------- create_sub_sample_dummies ----------------------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        self.df = self.df.fillna(value=np.nan)
        # ------------------------- minInstalls sub-samples -----------------------------------------
        self.df['Tier1'] = 0
        self.df['Tier1'].loc[self.df['imputed_minInstalls_'+self.all_panels[-1]] >= 1.000000e+07] = 1
        self.df['Tier2'] = 0
        self.df['Tier2'].loc[(self.df['imputed_minInstalls_' + self.all_panels[-1]] < 1.000000e+07) &
                             (self.df['imputed_minInstalls_' + self.all_panels[-1]] >= 1.000000e+05)] = 1
        self.df['Tier3'] = 0
        self.df['Tier3'].loc[self.df['imputed_minInstalls_' + self.all_panels[-1]] < 1.000000e+05] = 1
        # ------------------------- top firm sub-samples --------------------------------------------
        # this has been done in self.create_time_invariant_top_firm_dummy, var_name is top_firm
        # I will create non_top_firm just for the convenience of subsetting
        self.df['non_top_firm'] = self.df.apply(lambda row: 1 if row['top_firm'] == 0 else 0, axis=1)
        # ------------------------- create market_leader --------------------------------------------
        self.df['ML'] = self.df.apply(lambda row: 1 if (row['Tier1'] == 1 or row['top_firm'] == 1) else 0, axis=1)
        self.df['MF'] = self.df.apply(lambda row: 1 if row['ML'] == 0 else 0, axis=1)
        # imputed_genreId is time-invariant and based on the mode_genreId
        print(self.df['imputed_genreId'].value_counts(dropna=False))
        print(self.df.shape)
        # replace NaN with some string
        app_categories = self.df['imputed_genreId'].unique().tolist()
        print(len(app_categories), app_categories)
        def _determine_broad_category(x):
            if pd.isnull(x):
                return np.nan
            elif 'GAME' in x:
                return 'FULL_GAME'
            elif x in [  'FINANCE',
                         'EDUCATION',
                         'NEWS_AND_MAGAZINES',
                         'BUSINESS',
                         'PRODUCTIVITY',
                         'TOOLS',
                         'BOOKS_AND_REFERENCE',
                         'LIBRARIES_AND_DEMO']:
                return 'FULL_BUSINESS'
            elif x in [  'COMMUNICATION',
                         'FOOD_AND_DRINK',
                         'SOCIAL',
                         'SHOPPING',
                         'DATING',
                         'EVENTS',
                         'WEATHER',
                         'MAPS_AND_NAVIGATION',
                         'AUTO_AND_VEHICLES']:
                return 'FULL_SOCIAL'
            elif x in [  'PERSONALIZATION',
                         'SPORTS',
                         'MUSIC_AND_AUDIO',
                         'ENTERTAINMENT',
                         'TRAVEL_AND_LOCAL',
                         'LIFESTYLE',
                         'PHOTOGRAPHY',
                         'VIDEO_PLAYERS',
                         'PARENTING',
                         'COMICS',
                         'ART_AND_DESIGN',
                         'BEAUTY',
                         'HOUSE_AND_HOME']:
                return 'FULL_LIFESTYLE'
            elif x in [  'HEALTH_AND_FITNESS',
                         'MEDICAL']:
                return 'FULL_MEDICAL'
        self.df['FULL_GENRE'] = self.df.apply(lambda row: _determine_broad_category(row['imputed_genreId']), axis=1)
        print(self.df['FULL_GENRE'].value_counts(dropna=False))
        print(self.df.shape)
        # pandas get_dummies will treat np.nan as 0
        dfdummies = pd.get_dummies(self.df['FULL_GENRE'])
        # print(dfdummies.head())
        # print(dfdummies.shape)
        print('before merging the dummmies df to overall df')
        print(self.df.shape)
        # if you opened an imputed dataframe that with the app_death_month already created, None means the already created column name will stay as it is
        self.df = self.df.merge(dfdummies,
                                how='inner',
                                left_index=True, right_index=True,
                                suffixes = [None, '_y'])
        print('after merging the dummmies df to overall df')
        print(self.df.shape)
        # ------------------------- create nested market_leader\follower category sub-samples -------------------------
        self.df['ML_GAME'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_GAME']==1 and row['ML']==1) else 0, axis=1)
        self.df['ML_BUSINESS'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_BUSINESS']==1 and row['ML']==1) else 0, axis=1)
        self.df['ML_SOCIAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_SOCIAL']==1 and row['ML']==1) else 0, axis=1)
        self.df['ML_LIFESTYLE'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_LIFESTYLE']==1 and row['ML']==1) else 0, axis=1)
        self.df['ML_MEDICAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_MEDICAL']==1 and row['ML']==1) else 0, axis=1)
        self.df['MF_GAME'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_GAME']==1 and row['MF']==1) else 0, axis=1)
        self.df['MF_BUSINESS'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_BUSINESS']==1 and row['MF']==1) else 0, axis=1)
        self.df['MF_SOCIAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_SOCIAL']==1 and row['MF']==1) else 0, axis=1)
        self.df['MF_LIFESTYLE'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_LIFESTYLE']==1 and row['MF']==1) else 0, axis=1)
        self.df['MF_MEDICAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_MEDICAL']==1 and row['MF']==1) else 0, axis=1)
        # --------------------- count and check -----------------------------------------------------
        vs = [  'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm', 'FULL_GENRE',
                'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL',
                'ML', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL',
                'MF', 'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL']
        for v in vs:
            print('-------------------------- ' + v + ' --------------------------------------')
            print(self.df[v].value_counts(dropna=False))
        print('saving to the imputed dataframe')
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    ##################################################################################################################
    ### NLP for each sub-samples
    #################################################################################################################
    def clean_and_prepare_app_description(self, balanced, keep_only_unique_words):
        """
        # _________________ process text __________________________________________________
        # Adding ^ in []  excludes any character in
        # the set. Here, [^ab5] it matches characters that are
        # not a, b, or 5.
        """
        print('----------- clean_and_prepare_app_description ----------------')
        print('Open imputed dataframe')
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        print(self.df.columns)
        self.df = self.df.fillna(value=np.nan)
        print('1. removing the np.nan rows')
        print('before removing np.nan in imputed description')
        print(self.df.shape)
        self.df = self.df.dropna(subset=['imputed_'+self.tcn])
        print('after removing np.nan in imputed description')
        print(self.df.shape)
        print('2. create a column of SpaCy NLP object')
        self.df[self.tcn + '_NLP'] = self.df.progress_apply(
                lambda row: nlp(row['imputed_'+self.tcn].lower()), axis=1)
        print('3. remove non-english app description')
        self.df[self.tcn + 'Clean'] = self.df.progress_apply(
                lambda row: 1 if row[self.tcn + '_NLP']._.language == 'en' else 0, axis=1)
        print(self.df[self.tcn + '_english'].describe())
        print('before removing non-english in imputed description')
        print(self.df.shape)
        self.df = self.df.loc[self.df[self.tcn + '_english']==1]
        print('after removing non-english in imputed description')
        print(self.df.shape)
        print('4. remove stop words, keep only alphabetic characters, remove digits, remove single character word')
        self.df[self.tcn + 'Clean'] = self.df.progress_apply(
                lambda row: [w for w in row[self.tcn + '_NLP'] if (w.is_alpha) and (not w.is_stop) and (not w.is_digit) and (len(w.shape_)>1)], axis=1)
        print('5. get the roots of the words')
        self.df[self.tcn + 'Clean'] = self.df[self.tcn + 'Clean'].progress_apply(lambda x: [w.lemma_ for w in x])
        # this step is taking too long, and you will truncate svd so ignoring the irreleven non-english words that appear less frequently
        if keep_only_unique_words is True:
            print('6. remove non-english vocabulary from text descriptions and keep only unique words')
            self.df[self.tcn + 'Clean'] = self.df[self.tcn + 'Clean'].progress_apply(lambda x: list(set(x).intersection(set(words.words()))))
        print('7. get the number of words in app descriptions')
        self.df['length_token_app_description'] = self.df.apply(
                lambda row: len(row[self.tcn + 'Clean']), axis=1)
        print(self.df[['length_token_app_description']].head())
        print(self.df['length_token_app_description'].describe())
        print('7. Join the list of text to string in cleaned app description')
        self.df[self.tcn + 'Clean'] = self.df[self.tcn + 'Clean'].progress_apply(lambda x: " ".join(x))
        # ---------- save a sample of app description text, their mode and their cleaned mode -------------------------
        print('8. saving the imputed dataframe with new self.tcn + Clean column (based on imputed + self.tcn) and a sample of that column')
        df2 = self.df.copy(deep=True)
        df3 = df2[[self.tcn + 'Clean']]
        df3 = df3.sample(n=20)
        filename = self.initial_panel + '_cleaned_time_invariant_' + self.tcn + '.xlsx'
        q = self.check_app_descriptions / filename
        df3.to_excel(q)
        self._save_df(DF=self.df, balanced=balanced, keyword='imputed')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn)

    def _graph_histogram_token(self, df, bins, before_or_after):
        df['length_token_app_description_cut'] = pd.cut(df['length_token_app_description'],
                                                             bins=bins, precision=0)
        fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10),
                                 sharey='row', sharex='col')
        sns.set_style("darkgrid")
        axes = sns.countplot(data=df, x='length_token_app_description_cut', color='royalblue')
        axes.grid(True)
        axes.set_xlabel('Number of Words in An App\' Description')
        axes.set_ylabel('Apps Count')
        axes.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
        f_name = before_or_after + '_removing_too_short_too_long_histogram_of_num_words_in_app_description.png'
        q = self.nlp_graph_path / f_name
        fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return None

    def remove_text_descriptions_with_too_few_or_too_many_tokens(self, balanced,
                                          LB_token_length_to_remove=None,
                                          UP_token_length_to_remove=None):
        print('-------------------- remove_text_descriptions_with_too_few_or_too_many_tokens ---------------------')
        # after looking at the description of 'length_token_app_description' and the histogram and remove app descriptions that
        # are either too short or too long, use the lower bound and upper bound percentage points
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        print(self.df.shape)
        self._graph_histogram_token(df=self.df, bins=20, before_or_after='before')
        if LB_token_length_to_remove is not None and UP_token_length_to_remove is not None:
            print('before removing rows that token length below ' + str(LB_token_length_to_remove) + ' and above ' + str(UP_token_length_to_remove))
            print(self.df.shape)
            df2 = self.df.loc[(self.df['length_token_app_description']>=LB_token_length_to_remove) & (self.df['length_token_app_description']<=UP_token_length_to_remove)]
            print('after removing token length too short and too long')
            print(df2.shape)
            self._graph_histogram_token(df=df2, bins=20, before_or_after='after')
            # select the text columns and save
            self.ss_text_cols = df2[self.tcn + 'Clean']
            print(self.ss_text_cols.shape)
            f_name = self.initial_panel + '_full_clean_app_descriptions.pickle'
            q = self.nlp_stats_path / f_name
            pickle.dump(self.ss_text_cols, open(q, 'wb'))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols)

    def find_optimal_tfidf_parameters(self, binary, trim=True):
        print('----------------------------- transform_several_tf_idf_matrices_with_different_parameters ----------------------------------')
        """
        After ignoring terms that appear in more than 50% of the documents (as set by max_df=0.5) [ignoring too frequent words]
        and ignore terms that appear in less than 5% of the documents (as set by min_df+0.05) [ignoring too infrequent words]
        https://scikit-learn.org/stable/auto_examples/text/plot_document_clustering.html
        create matrices that loop through 0.5-0.99 max_df and 0.01-0.49 min_df
        the criteria is the percentage of non-zero entries in tf-idf matrix. choose the combination with the smallest non-zero percentage
        """
        filename = self.initial_panel + '_full_clean_app_descriptions.pickle'
        q = self.nlp_stats_path / filename
        self.ss_text_cols = pickle.load(open(q, 'rb'))
        print(self.ss_text_cols.shape)
        # ---------------------------------------------------------------------------------------
        if trim is True:
            parameters = {'coarse': {'max_df': [0.5, 0.6, 0.7, 0.8, 0.9],
                                     'min_df': [0.01, 0.02, 0.03]},
                          'fine': {'max_df': [0.5, 0.6, 0.7, 0.8, 0.9],
                                   'min_df': [0.001, 0.002, 0.003, 0.004, 0.005, 0.006, 0.007, 0.008, 0.009, 0.01]}}
            df = pd.DataFrame(columns = ['type', 'max_df', 'min_df', 'percentage_nonzero', 'feature_number'])
            for n in ['coarse', 'fine']:
                for i in range(len(parameters[n]['max_df'])):
                    for j in range(len(parameters[n]['min_df'])):
                        maxdf = parameters[n]['max_df'][i]
                        mindf = parameters[n]['min_df'][j]
                        print('max_df--', str(maxdf), 'min_df--', str(mindf))
                        vectorizer = TfidfVectorizer(stop_words='english',
                                                     strip_accents='unicode',
                                                     analyzer='word',
                                                     # binary should be true when you are using only unique words each app description
                                                     binary=binary,
                                                     max_df=maxdf,
                                                     min_df=mindf)
                        matrix = vectorizer.fit_transform(self.ss_text_cols)
                        # ------------------- get the table with words and app id ----------------------
                        matrix_df = pd.DataFrame(matrix.toarray(),
                                                 columns=vectorizer.get_feature_names())
                        matrix_df['app_ids'] = self.ss_text_cols.index.tolist()
                        matrix_df.set_index('app_ids', inplace=True)
                        print(matrix_df.shape)
                        df = df.append({'type':n,
                                        'max_df': maxdf,
                                        'min_df': mindf,
                                        'percentage_nonzero': matrix.nnz / np.prod(matrix.shape),
                                        'feature_number': matrix.shape[1]},
                                        ignore_index=True)
            # ------------------- save parameters ---------------------------------------------------------
            print('parameter_grid')
            print(df)
            filename = self.initial_panel + '_tf_idf_optimal_parameters.xlsx'
            q = self.nlp_stats_path / filename
            df.to_excel(q)
        else:
            vectorizer = TfidfVectorizer(stop_words='english',
                                         strip_accents='unicode',
                                         analyzer='word',
                                         # binary should be true when you are using only unique words each app description
                                         binary=binary)
            matrix = vectorizer.fit_transform(self.ss_text_cols)
            matrix_df = pd.DataFrame(matrix.toarray(),
                                     columns=vectorizer.get_feature_names())
            matrix_df['app_ids'] = self.ss_text_cols.index.tolist()
            matrix_df.set_index('app_ids', inplace=True)
            print('the tfidf matrix dimension without setting maxdf and mindf')
            print(matrix_df.shape)
            # print(list(matrix_df.columns))
        # ---------------- no need to make decision now ----------------------------------
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices)
    def graph_optimal_tfidf_param(self, picked_fine_maxdf=None, picked_fine_mindf=None):
        print('------------------------------ graph_optimal_tfidf_param --------------------------------')
        filename = self.initial_panel + '_tf_idf_optimal_parameters.xlsx'
        q = self.nlp_stats_path / filename
        df = pd.read_excel(q)
        # graph the min_df on the x-axis, and the percentage of nonzero cells on the y-axis, hue is max_df
        for n in ['coarse', 'fine']:
            df2 = df.loc[df['type']==n]
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10),
                                     sharey='row', sharex='col')
            sns.set_style("darkgrid")
            axes = sns.lineplot(data=df2, x='min_df', y='feature_number', hue='max_df')
            axes.grid(True)
            axes.set_xlabel('threshold_min')
            axes.set_ylabel('Column Number (Unique Words)')
            axes.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            if picked_fine_maxdf is not None and picked_fine_mindf is not None and n == 'fine':
                cols = df2.loc[(df2.max_df==picked_fine_maxdf) & (df2.min_df==picked_fine_mindf), 'feature_number'].squeeze()
                max_cols = df2.feature_number.max()
                axes.axvline(x=picked_fine_mindf, ymin=0, ymax=cols/max_cols, color='red', linestyle='--')
                axes.axhline(y=cols, xmin=0, xmax=picked_fine_mindf/0.01, color='red', linestyle='--')
            axes.legend(title="threshold_max",
                        bbox_to_anchor=(1.02, 0.3), loc='upper left', borderaxespad=0.1)
            print('ticks')
            tick_locations = df2['min_df'].loc[df2['max_df']==0.5].tolist()
            print(tick_locations)
            axes.set_xticks(ticks=tick_locations, minor=False)
            f_name = self.initial_panel + '_' + n + '_TFIDF_MIN_MAX_DF.png'
            q = self.nlp_graph_path / f_name
            fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices)
    def transform_tfidf_matrix(self, binary, picked_maxdf=None, picked_mindf=None):
        """
        20220926: The reason I chose maxdf = 0.9 and mindf = 0.02 because it scrafices the smallest amount of word variaty but reduces the largest amount of word columns
        See the graph.
        """
        print('----------------------------- transform_tfidf_matrix ---------------------------')
        filename = self.initial_panel + '_full_clean_app_descriptions.pickle'
        q = self.nlp_stats_path / filename
        self.ss_text_cols = pickle.load(open(q, 'rb'))
        print(self.ss_text_cols.shape)
        if picked_mindf is not None and picked_maxdf is not None:
            vectorizer = TfidfVectorizer(stop_words='english',
                                         strip_accents='unicode',
                                         analyzer='word',
                                         # binary should be true when you are using only unique words within each app description
                                         binary=binary,
                                         max_df=picked_maxdf,
                                         min_df=picked_mindf)
        else:
            vectorizer = TfidfVectorizer(stop_words='english',
                                         strip_accents='unicode',
                                         analyzer='word',
                                         binary=binary)
        matrix = vectorizer.fit_transform(self.ss_text_cols)
        # ------------------- get the table with words and app id ----------------------
        matrix_df = pd.DataFrame(matrix.toarray(),
                                 columns=vectorizer.get_feature_names())
        matrix_df['app_ids'] = self.ss_text_cols.index.tolist()
        matrix_df.set_index('app_ids', inplace=True)
        print(matrix_df.shape)
        print(matrix_df.head())
        # ---------------- save tf-idf matrices dictionary -------------------------------------------
        if picked_mindf is not None and picked_maxdf is not None:
            filename = self.initial_panel + '_' + str(picked_maxdf) + '_' + str(picked_mindf) + '_FULL_TFIDF_MATRIX.pickle'
        else:
            filename = self.initial_panel + '_FULL_TFIDF_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        pickle.dump(matrix_df, open(q, 'wb'))
        self.tf_idf_matrices = matrix_df
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices)

    def find_optimal_svd_component(self, picked_tfidf_param):
        """
        https://medium.com/swlh/truncated-singular-value-decomposition-svd-using-amazon-food-reviews-891d97af5d8d
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        I picked 0.6_0.01 is because the flatter part of the curve is more obvious above 95% and more words is better.
        """
        print('----------------------------- find_optimal_svd_component ----------------------------')
        # ---------------- open tf-idf matrices dictionary -------------------------------------------
        filename = self.initial_panel + '_' + picked_tfidf_param + '_FULL_TFIDF_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        self.tf_idf_matrices = pickle.load(open(q, 'rb'))
        print(self.tf_idf_matrices.shape)
        self.optimal_svd_df = pd.DataFrame(columns=['n_component', 'explained_ratio'])
        n_comp = np.round(np.arange(start=1, stop=self.tf_idf_matrices.shape[1]-1, step=100))
        n_comp = n_comp.astype(int)
        print(n_comp)
        explained = []
        for x in tqdm(n_comp):
            svd = TruncatedSVD(n_components=x)
            svd.fit(self.tf_idf_matrices)
            explained.append(svd.explained_variance_ratio_.sum())
            print("Number of components = %r and explained variance = %r" % (x, svd.explained_variance_ratio_.sum()))
            self.optimal_svd_df = self.optimal_svd_df.append({
                              'n_component': x,
                              'explained_ratio': svd.explained_variance_ratio_.sum()},
                              ignore_index=True)
        # ----------------- save optimal svd dictionary --------------------------------------------------------------
        filename = self.initial_panel + '_FULL_optimal_svd_comp_' + picked_tfidf_param + '.xlsx'
        q = self.nlp_stats_path / filename
        self.optimal_svd_df.to_excel(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df)

    def graph_svd_optimal_components(self, picked_tfidf_param, picked_cutoff=None):
        print('------------------------------- graph_svd_optimal_components ---------------------------------')
        filename = self.initial_panel + '_FULL_optimal_svd_comp_' + picked_tfidf_param + '.xlsx'
        q = self.nlp_stats_path / filename
        self.optimal_svd_df = pd.read_excel(q)
        n_comp = self.optimal_svd_df['n_component'].tolist()
        explained = self.optimal_svd_df['explained_ratio'].tolist()
        # ---------------------- plot and save graph ------------------------------------------------------------
        fig, ax = plt.subplots()
        sns.set_style("darkgrid")
        ax.plot(n_comp, explained)
        ax.grid(True)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_yticks(ticks=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1], minor=False)
        if picked_cutoff is not None:
            df2 = self.optimal_svd_df.loc[self.optimal_svd_df.explained_ratio>=picked_cutoff]
            explained_ratio = df2.explained_ratio.min()
            cols = df2.loc[df2.explained_ratio.idxmin(), 'n_component'].squeeze()
            max_cols = self.optimal_svd_df.loc[self.optimal_svd_df.explained_ratio.idxmax(), 'n_component'].squeeze()
            max_ratio = self.optimal_svd_df.explained_ratio.max()
            ax.axvline(x=cols, ymin=0, ymax=explained_ratio/max_ratio, color='red', linestyle='--')
            ax.axhline(y=explained_ratio, xmin=0, xmax=cols/max_cols, color='red', linestyle='--')
        plt.xlabel('Number of Columns (unique words)')
        plt.ylabel("Explained Variance")
        filename = self.initial_panel + '_' + picked_tfidf_param + '_FULL_optimal_svd_graph.png'
        fig.savefig(self.nlp_graph_path / 'optimal_svd_comp' / filename, facecolor='white', dpi=300)
        plt.show()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df)

    def truncate_svd(self, picked_tfidf_param, expalined_ratio_cutoff, random_state):
        print('----------------------------- truncate_svd ----------------------------')
        filename = self.initial_panel + '_FULL_TFIDF_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        self.tf_idf_matrices = pickle.load(open(q, 'rb'))
        filename = self.initial_panel + '_FULL_optimal_svd_comp_' + picked_tfidf_param + '.xlsx'
        q = self.nlp_stats_path / filename
        self.optimal_svd_df = pd.read_excel(q)
        svddf = self.optimal_svd_df.loc[self.optimal_svd_df['explained_ratio'] >= expalined_ratio_cutoff].sort_values(by='explained_ratio')
        # pick the n_comp that generate explained ratio just above the cutoff
        n_comp = svddf['n_component'].iloc[0]
        print(picked_tfidf_param)
        print(svddf)
        print(n_comp)
        svd = TruncatedSVD(n_components=n_comp,
                           random_state=random_state)
        matrix_transformed = svd.fit_transform(self.tf_idf_matrices)
        matrix_transformed_df = pd.DataFrame(
            matrix_transformed)  # do not need to assign column names because those do not correspond to each topic words (they have been transformed)
        matrix_transformed_df['app_ids'] = self.tf_idf_matrices.index.tolist()
        matrix_transformed_df.set_index('app_ids', inplace=True)
        print(matrix_transformed_df.shape)
        print(matrix_transformed_df.head())
        self.svd_matrices = matrix_transformed_df
        # ----------------- save -------------------------------------------
        filename = self.initial_panel + '_' + picked_tfidf_param + '_cutoff_' + str(expalined_ratio_cutoff) + '_FULL_SVD_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        pickle.dump(self.svd_matrices, open(q, 'wb'))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df,
                              svd_matrices=self.svd_matrices)

    def optimal_k_table(self, method, picked_tfidf_param, expalined_ratio_cutoff, num_intervals_between_cluster):
        """
        https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
        https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
        I think it is just better off using sihouette score because global maximum is better to find out than the elbow point.
        The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
        and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b).
        To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of.
        Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
        """
        print('-------------------------------- optimal_k_table ----------------------------------')
        filename = self.initial_panel + '_' + picked_tfidf_param + '_cutoff_' + str(expalined_ratio_cutoff) + '_FULL_SVD_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        self.svd_matrices = pickle.load(open(q, 'rb'))
        print(self.svd_matrices.shape)
        n_comp = self.svd_matrices.shape[1]
        sdf = pd.DataFrame(columns=['k_cluster', method+'_score'])
        n_cluster_list = np.round(np.linspace(2, self.svd_matrices.shape[0]-1, num_intervals_between_cluster))
        n_cluster_list = n_cluster_list.astype(int)
        print('find the ', method, ' scores of the following k clusters')
        print(n_cluster_list)
        scores = []
        print(method, ' -- ', self.initial_panel, ' _TF-IDF-MAXDF-MINDF_ ', picked_tfidf_param, ' _SVD-NCOMP_ ', n_comp)
        for x in tqdm(n_cluster_list):
            km = KMeans(n_clusters=x)
            km = km.fit(self.svd_matrices)
            labels = km.labels_
            if method == 'silhouette':
                s_score = silhouette_score(self.svd_matrices, labels, metric='cosine')
            elif method == 'elbow':
                s_score = km.inertia_
            scores.append(s_score)
            sdf = sdf.append({'k_cluster': x,
                              method+'_score': s_score},
                              ignore_index=True)
        # ----------------- save -----------------------------------------------
        f_name = self.initial_panel + '_FULL_' + picked_tfidf_param + '_cutoff_' + str(expalined_ratio_cutoff) + '_svd_cols_' + str(n_comp) + '_' + method + '_score_df.xlsx'
        q = self.nlp_stats_path / f_name
        sdf.to_excel(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df,
                              svd_matrices=self.svd_matrices)

    def graph_optimal_k(self, method, picked_tfidf_param,
                        picked_svd_comp, expalined_ratio_cutoff, picked_optimal_k=None):
        print('--------------------------- graph_optimal_k ----------------------------------')
        f_name = self.initial_panel + '_FULL_' + picked_tfidf_param + '_cutoff_' + str(expalined_ratio_cutoff) + '_svd_cols_' + str(picked_svd_comp) + '_' + method + '_score_df.xlsx'
        q = self.nlp_stats_path / f_name
        df = pd.read_excel(q)
        n_cluster_list = df['k_cluster'].tolist()
        scores = df[method + '_score'].tolist()
        fig, ax = plt.subplots()
        sns.set_style("darkgrid")
        ax.grid(True)
        ax.plot(n_cluster_list, scores, 'bx-')
        if picked_optimal_k is not None:
            score_of_picked_k = df.loc[df['k_cluster']==picked_optimal_k, method + '_score'].squeeze()
            max_score = df.loc[df.k_cluster.idxmin(), method + '_score'].squeeze()
            max_cluster = df.k_cluster.max()
            ax.axvline(x=picked_optimal_k, ymin=0, ymax=score_of_picked_k/max_score, color='red', linestyle='--')
            ax.axhline(y=score_of_picked_k, xmin=0, xmax=picked_optimal_k/max_cluster, color='red', linestyle='--')
        plt.xlabel('k')
        if method == 'silhouette':
            plt.ylabel('silhouette scores (cosine distance)')
        elif method == 'elbow':
            plt.ylabel('inertia scores')
        ax.get_xaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        ax.set_xticks(ticks=n_cluster_list, minor=False)
        filename = self.initial_panel + '_FULL_' + picked_tfidf_param + '_cutoff_' + str(expalined_ratio_cutoff) + '_svd_cols_' + str(picked_svd_comp) + '_' + method + '_graph.png'
        fig.savefig(self.nlp_graph_path / 'optimal_clusters' / filename, facecolor='white', dpi=300)
        plt.show()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df,
                              svd_matrices=self.svd_matrices)

    def refine_optimal_k_cluster(self, picked_tfidf_param, expalined_ratio_cutoff, picked_svd_comp,
                                 picked_LB_num_cluster, picked_UP_num_cluster, intervals):
        """
        :return:
        """
        print('----------------------------- refine_optimal_k_cluster ----------------------------')
        filename = self.initial_panel + '_' + picked_tfidf_param + '_cutoff_' + str(expalined_ratio_cutoff) + '_FULL_SVD_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        self.svd_matrices = pickle.load(open(q, 'rb'))
        n_cluster_list = np.round(np.linspace(picked_LB_num_cluster, picked_UP_num_cluster, intervals))
        n_cluster_list = n_cluster_list.astype(int)
        # draw both silhouette and inertia score graphs
        silhouette_scores = []
        inertia_scores = []
        print('SILHOUETTE SCORE -- ', self.initial_panel, ' _TF-IDF-MAXDF-MINDF_ ', picked_tfidf_param, ' _SVD-NCOMP_ ',
              str(picked_svd_comp))
        sdf = pd.DataFrame(columns=['tf_idf_maxdf_mindf', 'svd_comp', 'k_cluster', 'silhouette_score', 'inertia_score'])
        for x in tqdm(n_cluster_list):
            km = KMeans(n_clusters=x)
            km = km.fit(self.svd_matrices)
            labels = km.labels_
            s_score = silhouette_score(self.svd_matrices, labels, metric='cosine')
            silhouette_scores.append(s_score)
            e_score = km.inertia_
            inertia_scores.append(e_score)
            sdf = sdf.append({'tf_idf_maxdf_mindf': picked_tfidf_param,
                              'svd_comp': int(picked_svd_comp),
                              'k_cluster': x,
                              'silhouette_score': s_score,
                              'inertia_score': e_score},
                               ignore_index=True)
        # ----------------- save -----------------------------------------------
        f_name = self.initial_panel + '_FULL_' + picked_tfidf_param + '_' + str(expalined_ratio_cutoff) + '_SILHOUETTE_AND_INERTIA.xlsx'
        q = self.nlp_stats_path / f_name
        sdf.to_excel(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df,
                              svd_matrices=self.svd_matrices,
                              optimal_k_cluster_dict=self.optimal_k_cluster_dict)

    def graph_refined_optimal_k_silhouette_and_inertia(self, picked_tfidf_param, expalined_ratio_cutoff,
                                                       picked_optimal_k=None):
        print('--------------------------- graph_refined_optimal_k_silhouette_and_inertia -------------------------')
        f_name = self.initial_panel + '_FULL_' + picked_tfidf_param + '_' + str(
            expalined_ratio_cutoff) + '_SILHOUETTE_AND_INERTIA.xlsx'
        q = self.nlp_stats_path / f_name
        df = pd.read_excel(q)
        n_cluster_list = df['k_cluster'].tolist()
        silhouette_scores = df['silhouette_score'].tolist()
        inertia_scores = df['inertia_score'].tolist()
        # --------------- graph ------------------------------------------------
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.set_style("darkgrid")
        ax.grid(True)
        ax.plot(n_cluster_list, silhouette_scores, color='red')
        ax.tick_params(axis='y', labelcolor='red')
        ax.set_ylabel('silhouette score')
        ax.set_xticks(ticks=n_cluster_list, minor=False)
        ax.get_xaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if picked_optimal_k is not None:
            sil = df.loc[df.k_cluster == picked_optimal_k, 'silhouette_score'].squeeze()
            max_sil = df.silhouette_score.max()
            max_k = df.k_cluster.max()
            ax.axhline(y=sil, xmin=0, xmax=picked_optimal_k / max_k, color='blue', linestyle='--')
            ax.axvline(x=picked_optimal_k, ymin=0, ymax=sil/max_sil, color='blue', linestyle='--')
        plt.xlabel('k')
        plt.xticks(rotation=90)
        # Generate a new Axes instance, on the twin-X axes (same position)
        ax2 = ax.twinx()
        # Plot exponential sequence, set scale to logarithmic and change tick color
        ax2.plot(n_cluster_list, inertia_scores, color='green')
        ax2.tick_params(axis='y', labelcolor='green')
        ax2.get_yaxis().set_major_formatter(
            matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
        if picked_optimal_k is not None:
            ine = df.loc[df.k_cluster == picked_optimal_k, 'inertia_score'].squeeze()
            max_ine = df.inertia_score.max()
            max_k = df.k_cluster.max()
            ax2.axhline(y=ine, xmin=picked_optimal_k / max_k, xmax=1, color='blue', linestyle='--')
            ax2.axvline(x=picked_optimal_k, ymin=0, ymax=ine / max_ine, color='blue', linestyle='--')
        ax2.set_ylabel('inertia score')
        plt.tight_layout()
        filename = self.initial_panel + '_FULL_' + picked_tfidf_param + '_' + str(expalined_ratio_cutoff) + '_SILHOUETTE_AND_INERTIA.png'
        fig.savefig(self.nlp_graph_path / 'optimal_clusters' / filename, facecolor='white', dpi=300)
        plt.show()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df,
                              svd_matrices=self.svd_matrices,
                              optimal_k_cluster_dict=self.optimal_k_cluster_dict)

    def kmeans_cluster(self, random_state, cluster_num_for_robustness_check, picked_tfidf_param, expalined_ratio_cutoff):
        print('----------------------------- kmeans_cluster ----------------------------')
        filename = self.initial_panel + '_' + picked_tfidf_param + '_cutoff_' + str(
            expalined_ratio_cutoff) + '_FULL_SVD_MATRIX.pickle'
        q = self.nlp_stats_path / filename
        self.svd_matrices = pickle.load(open(q, 'rb'))
        self.output_labels = {}
        matrix = self.svd_matrices.copy()
        for i in cluster_num_for_robustness_check:
            print('optimal k clusters---' + str(i))
            print('input matrix shape')
            print(matrix.shape)
            y_kmeans = KMeans(
                n_clusters=int(i),
                random_state=random_state
            ).fit_predict(
                matrix
            )  # it is equivalent as using fit then .label_.
            self.svd_matrices['cluster_num_' + str(i) + '_kmeans_labels'] = y_kmeans
            self.output_labels['cluster_num_' + str(i) + '_kmeans_labels'] = self.svd_matrices[['cluster_num_' + str(i) + '_kmeans_labels']]
        # --------------------------- save -------------------------------------------------
        # for this one, you do not need to run text cluster label every month when you scraped new data, because they would more or less stay the same
        filename = self.initial_panel + '_' + picked_tfidf_param + '_kmeans_predicted_labels_dict_with_various_cluster_nums.pickle'
        q = self.nlp_stats_path / filename
        pickle.dump(self.output_labels, open(q, 'wb'))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              tf_idf_matrices=self.tf_idf_matrices,
                              optimal_svd_df=self.optimal_svd_df,
                              svd_matrices=self.svd_matrices,
                              optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                              output_labels=self.output_labels)

    ##################################################################################################################
    ### create INDEPENDENT VARIABLES (Niche_sub_samples, dummy and continuous), post and post niche interactions
    # ALL niche vairables (the continuous_niche and niche are created by the NLP label generated using self.tcn + 'Clean',
    # which is in turn based on 'imputed_'+self.tcn
    ##################################################################################################################
    def _numApps_per_cluster(self, k_name):
        print('----------------------------- _numApps_per_cluster ----------------------------')
        s2 = self.output_labels[k_name].groupby(
            [k_name]).size().sort_values(ascending=False)
        s3 = s2.rename('num_apps_in_cluster').to_frame()
        return s3

    def create_continuous_niche(self, picked_tfidf_param):
        print('----------------------------- create_continuous_niche ----------------------------')
        filename = self.initial_panel + '_' + picked_tfidf_param + '_kmeans_predicted_labels_dict_with_various_cluster_nums.pickle'
        q = self.nlp_stats_path / filename
        self.output_labels = pickle.load(open(q, 'rb'))
        for k_name, labels in self.output_labels.items():
            print(k_name + ' --- Number apps per cluster')
            dfs = self._numApps_per_cluster(k_name=k_name)
            dfs['cluster_labels'] = np.arange(dfs.shape[0]) + 1
            dfs.reset_index(inplace=True)
            dfs = dfs[[k_name, 'cluster_labels', 'num_apps_in_cluster']]
            print(dfs.shape)
            print(dfs.head())
            largest_num = dfs.at[0, 'num_apps_in_cluster']
            dfs['percentage_as_of_the_largest_cluster'] = dfs['num_apps_in_cluster'] / largest_num
            dfs = dfs.round(2)
            dfs['continuous_niche'] = 1 - dfs['percentage_as_of_the_largest_cluster']
            # print(dfs.shape)
            # print(dfs.head())
            print(k_name + ' merge in with text labels to assign each niche dummy to app ids')
            dfl = labels.copy()
            dfl.reset_index(inplace=True)
            print(dfl.shape)
            print(dfl.head())
            dff = dfl.merge(dfs, how='left', on=k_name)
            dff = dff.set_index('app_ids')
            print(dff.shape)
            print(dff.head())
            f_name = self.initial_panel + '_' + picked_tfidf_param + '_' + k_name + '_with_appids.pickle'
            q = self.nlp_stats_path / f_name
            pickle.dump(dff, open(q, 'wb'))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              output_labels=self.output_labels)

    def full_histogram_continuous_niche(self, picked_tfidf_param, cluster_num_for_robustness_check):
        """
        compare whether the continuous niche is similarly distributed according to different cluster number
        """
        print('------------------------ des_stats_continuous_niche ---------------------------')
        # continuous niche is based on imputed text descriptions, so it is imputed.
        final_table = pd.DataFrame(columns=['cluster_number', 'niche_cut', 'count'])
        for i in cluster_num_for_robustness_check:
            k_name = 'cluster_num_' + str(i) + '_kmeans_labels'
            f_name = self.initial_panel + '_' + picked_tfidf_param + '_' + k_name + '_with_appids.pickle'
            q = self.nlp_stats_path / f_name
            df_niche = pickle.load(open(q, 'rb'))
            df_niche['continuous_niche_cut'] = pd.cut(df_niche['continuous_niche'],
                                                      bins=[0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
                                                      right=False)
            print(df_niche.continuous_niche_cut.unique())
            fig, axes = plt.subplots(nrows=1, ncols=1, figsize=(10, 10),
                                     sharey='row', sharex='col')
            sns.set_style("darkgrid")
            axes = sns.countplot(data=df_niche, x="continuous_niche_cut", color='royalblue')
            axes.grid(True)
            axes.set_xlabel('Niche Index')
            axes.set_ylabel('Apps Count')
            axes.get_yaxis().set_major_formatter(
                matplotlib.ticker.FuncFormatter(lambda x, p: format(int(x), ',')))
            axes.yaxis.set_minor_locator(MultipleLocator(200))
            axes.set_xticklabels(axes.get_xticklabels(), rotation=90)
            f_name = picked_tfidf_param + '_' + k_name + '_continuous_niche_histogram.png'
            q = self.des_stats_graphs / 'unbalanced' / 'continuous_niche' / f_name
            fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
            table = df_niche.groupby(['continuous_niche_cut']).size().rename("count").reset_index()
            table['cluster_number'] = i
            print(table.head())
            final_table = final_table.append(table)
        # save combined table ---------------------------------------------
        f_name = 'count_by_continuous_niche_cut_various_cluster_number.xlsx'
        final_table.to_excel(self.des_stats_tables / 'unbalanced' / 'continuous_niche' / f_name)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              output_labels=self.output_labels)

    def eyeball_niche_text(self, picked_tfidf_param, cluster_num_for_robustness_check, fraction):
        """
        The function randomly take 10 apps each from each cluster.
        :return:
        """
        filename = self.initial_panel + '_full_clean_app_descriptions.pickle'
        q = self.nlp_stats_path / filename
        self.ss_text_cols = pickle.load(open(q, 'rb'))
        for i in cluster_num_for_robustness_check:
            k_name = 'cluster_num_' + str(i) + '_kmeans_labels'
            f_name = self.initial_panel + '_' + picked_tfidf_param + '_' + k_name + '_with_appids.pickle'
            q = self.nlp_stats_path / f_name
            df_niche = pickle.load(open(q, 'rb'))
            print(df_niche.shape)
            # print(df1.head())
            df3 = df_niche.join(self.ss_text_cols, how='inner')
            print(k_name + ' --- joined text col with niche variables ---')
            print(df3.shape)
            print(df3.head())
            df3 = df3.sort_values(by='num_apps_in_cluster', ascending=False)
            filename = self.initial_panel + '_' + picked_tfidf_param + '_' + k_name + '_with_app_descriptions.pickle'
            q = self.niche_text_path / filename
            df3.to_pickle(q)
            df4 = df3.groupby('num_apps_in_cluster').apply(lambda x: x.sample(frac=fraction)).reset_index(drop=True)
            df4 = df4.sort_values(by='cluster_labels', ascending=False).groupby('num_apps_in_cluster').apply(lambda x: x)
            # lambda x is needed to convert DataFrameGroupBy object to DataFrame
            filename = self.initial_panel + '_' + picked_tfidf_param + '_' + k_name + '_niche_text_five_percent_sample.xlsx'
            q = self.niche_text_path / filename
            df4.to_excel(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              ss_text_cols=self.ss_text_cols,
                              output_labels=self.output_labels)