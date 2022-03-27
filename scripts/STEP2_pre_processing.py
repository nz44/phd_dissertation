from pathlib import Path
import pickle
import pandas as pd
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings('ignore')
import matplotlib.pyplot as plt
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
from tqdm import tqdm
tqdm.pandas()
import spacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import matplotlib
import seaborn
import copy
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
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

    def __init__(self,
                 initial_panel,
                 all_panels,
                 tcn,
                 df=None,
                 appids_to_remove=None,
                 appids_with_changing_developers=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.tcn = tcn
        self.df = df
        self.appids_to_remove = appids_to_remove
        self.appids_with_changing_developers=appids_with_changing_developers

    # ====================== The set of functions below are regularly used common functions in pre_processing class =============================
    def open_merged_df(self, balanced):
        print('----------- open_merged_df ----------------')
        if balanced is True:
            f_name = self.initial_panel + '_balanced_MERGED.pickle'
        else:
            f_name = self.initial_panel + '_unbalanced_MERGED.pickle'
        q = self.full_sample_panel_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        self.df = df
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def save_df(self, balanced):
        """
        I will save the df at the very end, which include imputed variables, sub-sample dummies and NLP niche variables
        :param balanced: Whether to include appids that are missing in each subsequent month as compared to the original month
        :return:
        """
        print('----------- save_df ----------------')
        if balanced is True:
            f_name = self.initial_panel + '_pre_processed_balanced.pickle'
        else:
            f_name = self.initial_panel + '_pre_processed_unbalanced.pickle'
        q = self.full_sample_panel_path / f_name
        pickle.dump(self.df, open(q, 'wb'))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

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
        """
        print('----------- _select_dfs_per_month ----------------')
        dfs = {}
        print(var_list)
        for i in self.all_panels:
            if i != self.initial_panel:
                col_list = [var + '_' + i for var in var_list]
            else:
                col_list = [var + '_' + i for var in var_list]
                col_list.remove('merge_' + i)
            df2 = self.df.copy()
            print(col_list)
            df2 = df2[col_list]
            print('selected the following columns for the df ', df2.shape)
            dfs[i] = df2
        return dfs

    def _create_mode_column(self, var):
        """
        By default, the mode is created using self.all_panels of the var
        """
        print('--------------------- _create_mode_column -------------------')
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

    def _impute_missing_by_previous_nonmissing_column(self, var, the_missing_month):
        """
        This is used in imputing pricing variables that suppose to change over time and may follow a trend.
        For the_missing_month, we will seek its closest non-missing month back in time (NOT forward in time).
        the_missing_month should be one of self.all_panels, eg. '202107'
        suffix is the newly created column's suffix
        """
        print('----------- _impute_missing_by_previous_nonmissing_column ----------------')
        self.df = self.df.fillna(value=np.nan)
        # ------ get all the previous months and sort it from the closest (to today) to the furthest (to today) --------------
        the_missing_month = datetime.strptime(the_missing_month, "%Y%m")
        all_the_previous_months = []
        for p in self.all_panels:
            month = datetime.strptime(p, "%Y%m")
            if month < the_missing_month:
                all_the_previous_months.append(month)
        backward_previous_months = sorted(all_the_previous_months, reverse=True)
        backward_previous_months = [datetime.strftime(i, "%Y%m") for i in backward_previous_months]
        # ----- iterate month data and stop until you find a non-missing value -----------
        for i in range(len(backward_previous_months)):
            if i == 0:
                self.df['imputed_' + var + '_' + the_missing_month] = self.df.apply(
                    lambda row: row[var + '_' + backward_previous_months[i]]\
                        if row[var + '_' + backward_previous_months[i]].notnull() else np.nan, axis=1)
            else:
                self.df['imputed_' + var + '_' + the_missing_month] = self.df.apply(
                    lambda row: row[var + '_' + backward_previous_months[i]]\
                        if (row[var + '_' + backward_previous_months[i]].notnull() and row['imputed_' + var + '_' + the_missing_month].isnull())\
                        else row['imputed_' + var + '_' + the_missing_month], axis=1)
        return self.df

    def _remove_stopwords(self, text):
        text = text.lower()
        tokens_without_sw = [word for word in text.split() if word not in stopwords]
        filtered_sentence = (" ").join(tokens_without_sw)
        return filtered_sentence

    # count the rows that are all nan in each month
    def count_missing(self,
                      time_variant_var_list,
                      before_imputation,
                      balanced,
                      time_invariant_var_list=None):
        print('----------------------- count_missing -------------------------')
        print('Time Variant Variables: ')
        print(time_variant_var_list)
        self.df = self.df.fillna(value=np.nan)
        dfs = self._select_dfs_per_month(var_list=time_variant_var_list)
        # Initialize data to Dicts of series.
        d_keys = copy.deepcopy(time_variant_var_list)
        if time_invariant_var_list is not None:
            d_keys = d_keys + time_invariant_var_list
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
            if time_invariant_var_list is not None:
                for v in time_invariant_var_list:
                    dfc = self.df.copy()
                    df3 = dfc.dropna(axis=0, how='all', subset=[v])
                    v_col = 'rows_missing_in_' + v
                    summary_df.at['time_invariant', v_col] = rows_original - df3.shape[0]
        # -------------------- save the summary dataframe -----------------------------
        filename = self.initial_panel + '_count_missing_by_month_and_in_each_var.csv'
        if before_imputation is True:
            if balanced is True:
                q = self.missing_stats_path / 'before_imputation' / 'balanced'/ filename
            else:
                q = self.missing_stats_path / 'before_imputation' / 'unbalanced' / filename
        else:
            if balanced is True:
                q = self.missing_stats_path / 'after_imputation' / 'balanced' / filename
            else:
                q = self.missing_stats_path / 'after_imputation' / 'unbalanced' / filename
        summary_df.to_csv(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

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
        print('----------- _checking_consecutive_values ----------------')
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

    # ====================== The function below cleans and prepares the app description columns for NLP =============================
    def clean_and_prepare_app_description(self):
        """
        # _________________ process text __________________________________________________
        # Adding ^ in []  excludes any character in
        # the set. Here, [^ab5] it matches characters that are
        # not a, b, or 5.
        """
        print('----------- clean_and_prepare_app_description ----------------')
        self.df = self.df.fillna(value=np.nan)
        self.df = self._create_mode_column(var=self.tcn)
        self.df[self.tcn + 'ModeClean'] = self.df[self.tcn + 'Mode']
        self.df[self.tcn + 'ModeClean'] = self.df[self.tcn + 'ModeClean'].apply(
            lambda x: re.sub(r'[^\w\s]', '', x)).apply(
            lambda x: re.sub(r'[0-9]', '', x)).apply(
            lambda x: self._remove_stopwords(x))
        print(self.initial_panel, ' finished cleaning ', self.tcn + 'ModeClean')
        print()
        # ---------- save a sample of app description text, their mode and their cleaned mode -------------------------
        cols = [self.tcn + '_' + i for i in self.all_panels]
        cols.extend([self.tcn + 'Mode', self.tcn + 'ModeClean'])
        df2 = self.df.copy(deep=True)
        df3 = df2[cols]
        df3 = df3.sample(n=20)
        filename = self.initial_panel + '_cleaned_mode_' + self.tcn + '.csv'
        q = self.check_app_descriptions / filename
        df3.to_csv(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # ================= The function below impute the list of variables by their pre-defined imputation methods =======================
    def impute_missing(self, imputation_var_methods_dict):
        """
        :param imputation_var_methods_dict: A dictionary with variable name as key and imputation methods as value ('mode', 'mean' and 'previous')
        :return: it will update self.df with the imputed columns
        """
        print('------------------------- impute_missing -------------------------------------')
        self.df = self.df.fillna(value=np.nan)
        for var, method in imputation_var_methods_dict.items():
            if method == 'mean':
                self.df = self._create_mean_column(var=var)
            if method == 'mode':
                self.df = self._create_mode_column(var=var)
            for p in self.all_panels:
                print('BEFORE IMPUTATION total missing ' + var + '----' + p)
                print(self.df[var + '_' + p].isna().sum())
                print('Start imputing ' + var + ' with ' + method)
                if method == 'previous':
                    self.df = self._impute_missing_by_previous_nonmissing_column(
                        var=var, the_missing_month=p)
                if method == 'mean':
                    self.df['imputed_' + var + '_' + p] = self.df.apply(
                        lambda row: row['mean_'+var]\
                            if row[var + '_' + p].isnull() else row[var + '_' + p])
                if method == 'mode':
                    self.df['imputed_' + var + '_' + p] = self.df.apply(
                        lambda row: row['mode_'+var]\
                            if row[var + '_' + p].isnull() else row[var + '_' + p])
                print('AFTER IMPUTATION USING ' + method + ' METHOD total missing ' + var + '----' + p)
                print(self.df['imputed_' + var + '_' + p].isna().sum())
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # =========== The set of functions below create time variant variables that exist only in the unbalanced panel ============
    # ---UNBALANCED PANEL--- replace merged_month columns with noisy_death -----------------------------------------------------------------
    def unbalanced_panel_create_noisy_death_dummy(self):
        print('----------- unbalanced_panel_create_noisy_death_dummy ----------------')
        self.df = self.df.fillna(value=np.nan)
        df = self.df.copy()
        merged_list = []
        for i in df.columns:
            if 'merge_' in i:
                merged_list.append(i)
        df2 = df[merged_list]
        # print(df2.head())
        df3 = self._checking_consecutive_values(df=df2,
                                                var='merge',
                                                check_nan=False,
                                                check_key_value='left_only',
                                                new_col='app_death_month')
        # print(df3.head())
        # create noisy death dummy
        for i in range(len(self.all_panels)):
            if i == 0:
                df3['noisy_death_' + self.all_panels[i]] = df3.apply(
                    lambda row: 1 if row['app_death_month'] == self.all_panels[i] else 0, axis=1)
            else:
                df3['noisy_death_' + self.all_panels[i]] = df3.apply(
                    lambda row: 1 if row['app_death_month'] == self.all_panels[i] else row['noisy_death_' + self.all_panels[i-1]], axis=1)
            print(df3['noisy_death_' + self.all_panels[i]].value_counts(dropna=False))
        self.df = df3
        ls = ['noisy_death_' + i for i in self.all_panels]
        print(self.df[ls].head())
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # ---UNBALANCED PANEL--- create dummies indicating the app has transitioned from market follower to market leader ---------------------------
    # according to our definition of market leaders and followers, they are either switched to a top firm or minInstalls increase over a threshhold of
    def unbalanced_panel_create_TRANSITION_tier1_minInstalls(self):
        # according to the definition of tier1 minInstalls sample (>= 1.000000e+07), we will create variable  assign the app 1 if in
        print('----------- unbalanced_panel_create_TRANSITION_dummy ----------------')
        self.df = self.df.fillna(value=np.nan)
        df = self.df.copy()
        # the new variables is a panel variables transition to tier 1 minInstalls
        for i in range(len(self.all_panels)):
            if i == 0:
                df['T_TO_TIER1_minInstalls_' + self.all_panels[i]] = 0
            else:
                df['T_TO_TIER1_minInstalls_' + self.all_panels[i]] = df.apply(
                    lambda row: 1\
                        if (row['minInstalls_' + self.all_panels[i-1]] < 1.000000e+07 and row['minInstalls_' + self.all_panels[i]] >= 1.000000e+07)\
                        else row['T_TO_TIER1_minInstalls_' + self.all_panels[i-1]], axis=1)
            print(df['T_TO_TIER1_minInstalls_' + self.all_panels[i]].value_counts(dropna=False))
        self.df = df
        ls = ['T_TO_TIER1_minInstalls_' + i for i in self.all_panels]
        print(self.df[ls].head())
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

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
    def create_time_variant_top_firm_dummy(self):
        # this is a time variant variable, because we want to see which apps transitioned from non-top firm to top firm
        # or which app has gone through merger and acquisition
        # in the balanced panel, we will use the mode of time_variant_top_firm_dummy to create time_invariant_top_firm_dummy
        """
        https://www.forbes.com/top-digital-companies/list/#tab:rank
        :return:
        """
        print('----------- create_time_variant_top_firm_dummy ----------------')
        self.df = self.df.fillna(value=np.nan)
        # clean the developer text column
        for j in self.all_panels:
            self.df['developerClean_' + j] = self.df['developer_' + j].apply(
                lambda x: self._format_text_for_developer(x))
        # create top firm dummy based on 'developerClean_'
        for p in self.all_panels:
            self.df['top_firm_'+p] = 0
            for i in tqdm(range(len(self.top_digital_firms_substring))):
                for j, row in self.df.iterrows():
                    if self.top_digital_firms_substring[i] in row['developerClean_'+p]:
                        self.df.at[j, 'top_firm_'+p] = 1
            for i in tqdm(range(len(self.top_digital_firms_exactly_match))):
                for j, row in self.df.iterrows():
                    if self.top_digital_firms_exactly_match[i] == row['developerClean_'+p]:
                        self.df.at[j, 'top_firm_'+p] = 1
        # ------------------ print check -----------------------------------------
        for p in self.all_panels:
            print(self.df['top_firm_'+p].value_counts(dropna=False))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def create_time_invariant_top_firm_dummy(self):
        print('---------------- create_time_invariant_top_firm_dummy ---------------------')
        self.df = self.df.fillna(value=np.nan)
        # clean the developer text column
        for j in self.all_panels:
            self.df['developerClean_' + j] = self.df['developer_' + j].apply(
                lambda x: self._format_text_for_developer(x))
        self.df = self._create_mode_column(var='developerClean')
        # create top firm dummy based on 'developerMode'
        self.df['top_firm'] = self.df.apply(
            lambda row: 1\
                if (process.extractOne(row['mode_developerClean'], self.top_digital_firms_substring, score_cutoff=80)\
                    or row['mode_developerClean'] in self.top_digital_firms_exactly_match)\
                else 0)
        print(self.df['top_firm'].value_counts(dropna=False))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def unbalanced_panel_create_TRANSITION_top_firm_and_MA_dummy(self):
        """
        merger and acquisition dummy is 1 in a period if the app's developerClean is different from the previous one
        """
        print('------------------ unbalanced_panel_create_TRANSITION_top_firm ----------------')
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
        ls = ['T_TO_top_firm_' + i for i in self.all_panels]
        print(self.df[ls].head())
        ls = ['MA_' + i for i in self.all_panels]
        print(self.df[ls].head())
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def create_sub_sample_dummies(self):
        """
        This way you do not need to seperately save data as sub-samples
        """
        print('----------------------------- create_sub_sample_dummies ----------------------------')
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
        # ------------------------- create market_leader --------------------------------------------
        self.df['market_leader'] = self.df.apply(lambda row:1\
                                                    if (row['Tier1']==1 or row['top_firm']==1)\
                                                    else 0, axis=1)
        # create categorical sub-samples, for nested market-leader categorical sub-samples, use two dummies to filter
        self.df = self._create_mode_column(var='imputed_genreId')
        app_categories = self.df['mode_imputed_genreId'].unique().tolist()
        print(len(app_categories), app_categories)
        self.df['FULL_GAME'] = self.df.apply(lambda row: 1 if 'GAME' in row['mode_imputed_genreId'] else 0, axis=1)
        self.df['FULL_BUSINESS'] = self.df.apply(lambda row: 1\
                                                        if row['mode_imputed_genreId'] in [  'FINANCE',
                                                                                             'EDUCATION',
                                                                                             'NEWS_AND_MAGAZINES',
                                                                                             'BUSINESS',
                                                                                             'PRODUCTIVITY',
                                                                                             'TOOLS',
                                                                                             'BOOKS_AND_REFERENCE',
                                                                                             'LIBRARIES_AND_DEMO'] else 0, axis=1)
        self.df['FULL_SOCIAL'] = self.df.apply(lambda row: 1\
                                                    if row['mode_imputed_genreId'] in [  'COMMUNICATION',
                                                                                         'FOOD_AND_DRINK',
                                                                                         'SOCIAL',
                                                                                         'SHOPPING',
                                                                                         'DATING',
                                                                                         'EVENTS',
                                                                                         'WEATHER',
                                                                                         'MAPS_AND_NAVIGATION',
                                                                                         'AUTO_AND_VEHICLES'] else 0, axis=1)
        self.df['FULL_LIFESTYLE'] = self.df.apply(lambda row: 1\
                                                        if row['mode_imputed_genreId'] in [  'PERSONALIZATION',
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
                                                                                             'HOUSE_AND_HOME'] else 0, axis=1)
        self.df['FULL_MEDICAL'] = self.df.apply(lambda row: 1\
                                                        if row['mode_imputed_genreId'] in [  'HEALTH_AND_FITNESS',
                                                                                             'MEDICAL'] else 0, axis=1)
        # ------------------------- create nested market_leader\follower category sub-samples -------------------------
        self.df['ML_GAME'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_GAME']==1 and row['market_leader']==1) else 0, axis=1)
        self.df['ML_BUSINESS'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_BUSINESS']==1 and row['market_leader']==1) else 0, axis=1)
        self.df['ML_SOCIAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_SOCIAL']==1 and row['market_leader']==1) else 0, axis=1)
        self.df['ML_LIFESTYLE'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_LIFESTYLE']==1 and row['market_leader']==1) else 0, axis=1)
        self.df['ML_MEDICAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_MEDICAL']==1 and row['market_leader']==1) else 0, axis=1)
        self.df['MF_GAME'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_GAME']==1 and row['market_leader']==0) else 0, axis=1)
        self.df['MF_BUSINESS'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_BUSINESS']==1 and row['market_leader']==0) else 0, axis=1)
        self.df['MF_SOCIAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_SOCIAL']==1 and row['market_leader']==0) else 0, axis=1)
        self.df['MF_LIFESTYLE'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_LIFESTYLE']==1 and row['market_leader']==0) else 0, axis=1)
        self.df['MF_MEDICAL'] = self.df.apply(lambda row: 1\
                                            if(row['FULL_MEDICAL']==1 and row['market_leader']==0) else 0, axis=1)
        # --------------------- count and check -----------------------------------------------------
        vs = [  'Tier1', 'Tier2', 'Tier3', 'market_leader',
                'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL',
                'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL',
                'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL']
        for v in vs:
            print('-------------------------- ' + v + ' --------------------------------------')
            print(self.df[v].value_counts(dropna=False))
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)
