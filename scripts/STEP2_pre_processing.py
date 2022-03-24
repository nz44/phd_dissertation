from pathlib import Path
import pickle
import pandas as pd
pd.set_option('display.max_rows', 500)
import warnings
warnings.filterwarnings('ignore')
from tqdm import tqdm
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
import nltk
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import matplotlib
import seaborn
import copy
#################################################################################################################
# the input dataframe are the output of merge_panels_into_single_df() method of app_detail_dicts class
class pre_processing():
    stoplist = nltk.corpus.stopwords.words('english')
    panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/___essay_1___')
    missing_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/missing_stats')

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

    def open_merged_df(self, balanced):
        if balanced is True:
            f_name = self.initial_panel + '_balanced_MERGED.pickle'
        else:
            f_name = self.initial_panel + '_unbalanced_MERGED.pickle'
        q = pre_processing.panel_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        self.df = df
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    ############################# ONLY FOR ESSAY THREE UNBALANCED PANEL REGS ###############################################################
    # this is a helper function where the input is a dataframe containing only the variable across all months in the panel
    # var is the variable name
    # if check_nan is true, we are checking after a period the variable is all nans till the end of the panel
    # check_key_value is a string or a number that we want to know from which period the row has this key value till the end of the panel
    # we are checking this by iterate through the first month to the last month
    def _checking_consecutive_values(self, df, var, check_nan, check_key_value, new_col):
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
            # for example, if app id 1234 is completely missing in period 6 (merge_6 == left), and has some data in period 7 (merge_7 = both),
            # and completely missing in period 8, 9, 10 (merge_8,9,10 == left) till the last period
            # then we measure app id 1234 as death starting from period 8. However if an app is only missing for one period and has data till the end,
            # it does not count towards dead apps.
            # after check the condition for this backward period, let us check the total number so far,
            # if the total figure equals to the number of periods we have checked starting from the last period
            # that means the apps has been dead since this point onward till the end of the panel
            # Then we could update the new_col with this period's datetime string.
            # Then the loop will go one period earlier in time, if that period satisfied, then new_col will update to that period's datetime string.
            # however if that period does not satisfy, then the total figure will be less than the number of periods we've checked,
            # we will keep the new_col unchaged (it is either in the period it satisfied or 0 initial condition)
            # Even if any of the earlier period satisfy again, the total figure will not equal to the number of periods we've checked,
            # so the new_col will not be updated.
            df[new_col] = df.apply(lambda row: backward_month[i] if row['conditions_satisfied'] == i+1 else row[new_col], axis=1)
        print('the unqiue values in conditions_satisfied')
        print(df['conditions_satisfied'].value_counts(dropna=False))
        print('the unqiue values in ' + new_col)
        print(df[new_col].value_counts(dropna=False))
        return df

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
        print(self.df.head())
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # ---UNBALANCED PANEL--- create dummies indicating the app has transitioned from market follower to market leader ---------------------------
    # according to our definition of market leaders and followers, they are either switched to a top firm or minInstalls increase over a threshhold of
    def unbalanced_panel_create_TRANSITION_dummy(self):
        print('----------- unbalanced_panel_create_TRANSITION_dummy ----------------')
        print(var_list)
        self.df = self.df.fillna(value=np.nan)
        dfs = self.select_dfs_per_month(var_list=var_list)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # ---UNBALANCED PANEL--- create dummies indicating the app has changed firms (Merger and Acquisition) ----------------------------------
    def unbalanced_panel_create_MA_dummy(self):
        print('----------- unbalanced_panel_create_MA_dummy ----------------')
        print(var_list)
        self.df = self.df.fillna(value=np.nan)
        dfs = self.select_dfs_per_month(var_list=var_list)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # #######################################################################################################################################

    def open_imputed_missing_df(self):
        f_name = self.initial_panel + '_imputed_missing.pickle'
        q = pre_processing.panel_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        self.df = df
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    # create a list of dataframes that each df consists of the same variable across all months/or one month
    def select_dfs_per_var(self, var_list, select_one_panel=None):
        print('----------- select_dfs_per_var ----------------')
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
    def select_dfs_per_month(self, var_list):
        print('----------- select_dfs_per_month ----------------')
        dfs = {}
        print(var_list)
        for i in self.all_panels:
            if i != self.initial_panel:
                col_list = [var + '_' + i for var in var_list]
            else:
                col_list = [var + '_' + i for var in var_list]
                col_list.remove('merge_'+i)
            df2 = self.df.copy()
            print(col_list)
            df2 = df2[col_list]
            print('selected the following columns for the df ', df2.shape)
            dfs[i] = df2
        return dfs

    # count the rows that are all nan in each month
    def count_missing_per_month(self, var_list):
        print('----------- count_missing_per_month ----------------')
        print(var_list)
        self.df = self.df.fillna(value=np.nan)
        dfs = self.select_dfs_per_month(var_list=var_list)
        # Initialize data to Dicts of series.
        d_keys = copy.deepcopy(var_list)
        d_keys = ['rows_missing_in_' + i for i in d_keys]
        d_keys.insert(0, 'rows_original') # insert 10 at 4th index
        d_keys.insert(1, 'rows_missing_in_any')
        d_keys.insert(2, 'rows_missing_in_all')
        summary_df = pd.DataFrame(
                    columns = d_keys,
                    index = list(dfs.keys()))
        for month, df in dfs.items():
            print(month, ' before deleting nans : ', df.shape)
            rows_original = df.shape[0]
            summary_df.at[month, 'rows_original'] = rows_original
            if month == self.initial_panel:
                cols = df.columns.values.tolist()  # initial month does not have the merge variable
            else:
                cols = df.columns.values.tolist()
                cols.remove('merge_' + month) # remove will not return a list
            df2 = df.dropna(axis=0, how='any', subset=cols)
            summary_df.at[month, 'rows_missing_in_any'] = rows_original - df2.shape[0]
            df2 = df.dropna(axis=0, how='all', subset=cols)
            summary_df.at[month, 'rows_missing_in_all'] = rows_original - df2.shape[0]
            for v in cols:
                df2 = df.dropna(axis=0, how='all', subset=[v])
                v_col = v.replace('_'+month, '')
                v_col = 'rows_missing_in_' + v_col
                summary_df.at[month, v_col] = rows_original - df2.shape[0]
        filename = self.initial_panel + '_count_missing_by_month_and_in_each_var.csv'
        q = self.missing_stats_path / 'before_imputation' / filename
        summary_df.to_csv(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def remove_rows_with_missing_in_all_text_cols(self):
        """
        do this before impute_text_col
        """
        df2 = self.df.copy(deep=True)
        cols = [self.tcn + '_' + item for item in self.all_panels]
        text_df = df2[cols]
        appids_to_remove = self.find_rows_contain_all_missing(df=text_df)
        print(self.initial_panel, ' before removing rows with all missing in ', self.tcn, ' has shape : ', self.df.shape)
        self.df = self.df.drop(appids_to_remove, axis=0)
        print(self.initial_panel, 'after removing rows with all missing in ', self.tcn, ' has shape : ', self.df.shape)
        print()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def mode_text_col(self):
        """
        impute the missing panels using its non-missing panels
        """
        cols = [self.tcn + '_' + item for item in self.all_panels]
        for j in cols:
            self.df[j] = self.df[j].fillna('')
        df2 = self.df.copy(deep=True)
        df3 = df2[cols]
        df3[self.tcn + 'Mode'] = df3.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        text_col = df3[self.tcn + 'Mode']
        null_data = text_col[text_col.isnull()]
        print(self.initial_panel, ' IMPUTED ', self.tcn, ' using Mode. ')
        if len(null_data) == 0:
            print('NO MISSING remaining in mode text col.')
        self.df = self.df.join(df3[self.tcn + 'Mode'], how='inner')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def remove_stopwords(self, text):
        text = text.lower()
        tokens_without_sw = [word for word in text.split() if word not in pre_processing.stoplist]
        filtered_sentence = (" ").join(tokens_without_sw)
        return filtered_sentence

    def clean_text_col(self):  # use take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
        """
        # _________________ process text __________________________________________________
        # Adding ^ in []  excludes any character in
        # the set. Here, [^ab5] it matches characters that are
        # not a, b, or 5.
        """
        self.df[self.tcn + 'ModeClean'] = self.df[self.tcn + 'Mode']
        self.df[self.tcn + 'ModeClean'] = self.df[self.tcn + 'ModeClean'].apply(
            lambda x: re.sub(r'[^\w\s]', '', x)).apply(
            lambda x: re.sub(r'[0-9]', '', x)).apply(
            lambda x: self.remove_stopwords(x))
        print(self.initial_panel, ' finished cleaning ', self.tcn + 'ModeClean')
        print()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def check_cleaned_mode_text_col(self):
        cols = [self.tcn + '_' + item for item in self.all_panels]
        cols.extend([self.tcn + 'Mode', self.tcn + 'ModeClean'])
        df2 = self.df.copy(deep=True)
        df3 = df2[cols]
        df3 = df3.sample(n=50)
        # ---------- save ------------------------------------------------------
        filename = self.initial_panel + '_cleaned_mode_' + self.tcn + '.csv'
        q = pre_processing.panel_path / 'check_text_cols' / filename
        df3.to_csv(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def format_text_for_developer(self, text):
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

    def create_time_invariant_developer_col(self):
        for j in self.all_panels:
            self.df['developerClean_' + j] = self.df['developer_' + j].apply(lambda x: self.format_text_for_developer(x))
        df2 = self.select_the_var(var='developerClean')
        appids_with_time_invar_developer = self.find_rows_contain_identical_value_for_nonmissings(df=df2)
        self.df['developerTimeInvar'] = False
        # use the mode developer for those apps that have identical developer value throughout all panels
        df2 = self.df.copy(deep=True)
        devs = ['developer_' + j for j in self.all_panels]
        df2 = df2[devs]
        self.df['developerMode'] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        self.df['developerTimeInvar'].loc[appids_with_time_invar_developer] = self.df['developerMode']
        # ----------- stats --------------------------------------------------------------------
        df2 = self.df.copy(deep=True)
        print(self.initial_panel, ' shape : ', df2.shape)
        df3 = df2.loc[df2['developerTimeInvar'] != False]
        print(self.initial_panel, ' appids with time-invariant developer for ALL (none missing) panels : ',
              len(df3.index))
        df3 = df2.loc[df2['developerTimeInvar'] == False]
        print(self.initial_panel, ' appids changed developer throughout ALL (none missing) panels : ',
              len(df3.index))
        self.appids_with_changing_developers = df3.index.tolist()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def compare_original_and_imputed_developer(self):
        df2 = self.df.copy(deep=True)
        developer = ['developer_' + j for j in self.all_panels]
        devclean = ['developerClean_' + j for j in self.all_panels]
        developer.extend(devclean)
        developer.extend(['developerTimeInvar', 'developerMode'])
        df3 = df2[developer]
        # ---------- save ------------------------------------------------------
        filename = self.initial_panel + '_developer_names_compare.csv'
        q = pre_processing.panel_path / 'check_imputed' / filename
        df3.to_csv(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def select_imputation_panels(self, df, method, current_panel):
        """
        :param df: input dataframe is a deep copy of self.select_the_var(var=var)
        :param method: all panels before, all panels before and after
        :param current_panel: the time where you decide which panel are before and after in self.all_panels
        :return: the dataframe after selecting the useful panels
        """
        df2 = df.copy(deep=True)
        the_panel = datetime.strptime(current_panel, "%Y%m")
        if method == 'all panels before and after':
            return df2
        elif method == 'all panels before':
            before_panels = []
            for i in self.all_panels:
                panel = datetime.strptime(i, "%Y%m")
                if panel <= the_panel:
                    before_panels.append(i)
            cols_to_keep = []
            for i in df2.columns:
                z = i.split('_')
                if z[1] in before_panels:
                    cols_to_keep.append(i)
            return df2[cols_to_keep]

    def find_rows_contain_identical_value_for_nonmissings(self, df):
        """
        :return: appids that have identical values for all its non-missing columns
        """
        df2 = df.copy(deep=True)
        # ---------- strategy for rows contains no missing values --------------------------
        appids_no_missing = self.find_rows_contain_no_missing(df=df2)
        df3 = df2.loc[appids_no_missing]
        # you need to select the True ones, df3.eq(df3.iloc[:, 0], axis=0).all(1) will return a pandas series with T and F
        df3 = df3.loc[df3.eq(df3.iloc[:, 0], axis=0).all(1)]
        appids_identical = df3.index.tolist()
        # ---------- strategy for rows contains at least one missing value -----------------
        appids_any_missing = self.find_rows_contain_any_missing(df=df2)
        df4 = df2.loc[appids_any_missing]
        df5 = df4.T
        appids_part2 = []
        # now each column name is appid
        for i in df5.columns:
            df6 = df5[i].dropna()
            # if empty it means the entire row contains ALL missings
            if not df6.empty:
                if len(set(df6.tolist())) == 1:
                    appids_part2.append(i)
        appids_identical.extend(appids_part2)
        return appids_identical

    def implement_imputation_methods(self, df, method):
        """
        :param df: The dataframe is the output of self.select_imputation_panels()
        :return: imputed column (a pandas series), you need to use this column to fill missing in the original columns
        """
        df2 = df.copy(deep=True)
        if method == 'mean':
            df2[method] = df2.mean(axis=1, skipna=True)
        elif method == 'mode':
            df2[method] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        elif method == 'mode if none-missing are all the same':
            appids = self.find_rows_contain_identical_value_for_nonmissings(df=df2)
            df2['mode'] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
            df2[method] = None
            # assign 'mode if none-missing are all the same' to 'mode' when 'nonmissings_are_same' is True
            df2[method].loc[appids] = df2['mode']
        elif method == 'previous':
            # because you are imputing the last column in 'all panels before' using the second last column
            reversed_columns = df2.columns[::-1]
            df2[method] = df2[reversed_columns[0]]
            # iterratively fill na in df[method] with previous panels, starting from the most recent previous
            for i in range(1, len(reversed_columns)):
                df2[method] = df2[method].fillna(df2[reversed_columns[i]])
        return df2[method]

    def impute_missing(self, var):
        df3 = self.select_the_var(var=var)
        cols_to_drop_later = df3.columns.tolist()
        if var in ['minInstalls', 'price', 'updated', 'free']:
            for i in self.all_panels:
                df4 = self.select_imputation_panels(df=df3,
                                                    method='all panels before',
                                                    current_panel=i)
                if var in ['minInstalls', 'price', 'updated']:
                    the_imputed_panel = self.implement_imputation_methods(df=df4,
                                                                          method='previous')
                else:
                    the_imputed_panel = self.implement_imputation_methods(df=df4,
                                                                          method='mode if none-missing are all the same')
                df3['Imputed' + var + '_' + i] = df3[var + '_' + i]
                df3['Imputed' + var + '_' + i] = df3['Imputed' + var + '_' + i].fillna(the_imputed_panel)
        elif var in ['score', 'reviews', 'ratings', 'released', 'contentRating', 'genreId', 'size', 'containsAds', 'offersIAP']:
            for i in self.all_panels:
                df4 = self.select_imputation_panels(df=df3,
                                                    method='all panels before and after',
                                                    current_panel=i)
                if var in ['score', 'reviews', 'ratings']:
                    the_imputed_panel = self.implement_imputation_methods(df=df4,
                                                                          method='mean')
                elif var in ['released', 'contentRating', 'genreId', 'size']:
                    the_imputed_panel = self.implement_imputation_methods(df=df4,
                                                                          method='mode')
                else:
                    the_imputed_panel = self.implement_imputation_methods(df=df4,
                                                                          method='mode if none-missing are all the same')
                df3['Imputed' + var + '_' + i] = df3[var + '_' + i]
                df3['Imputed' + var + '_' + i] = df3['Imputed' + var + '_' + i].fillna(the_imputed_panel)
        df3.drop(cols_to_drop_later, axis=1, inplace=True)
        print('finished imputing missing for variable : ', var)
        return df3

    def impute_list_of_vars(self, balanced, list_of_vars):
        """
        Since I have save in this funciton, make sure to run text column imputation and developer column
        before this step
        """
        imputed_dfs = []
        for i in list_of_vars:
            df3 = self.impute_missing(var=i)
            imputed_dfs.append(df3)
        combined_imputed = functools.reduce(lambda a, b: a.join(b, how='inner'), imputed_dfs)
        self.df = self.df.join(combined_imputed, how='inner')
        # -------------------- save -----------------------------------
        if balanced is True:
            filename = self.initial_panel + '_balanced_imputed_missing.pickle'
        else:
            filename = self.initial_panel + '_unbalanced_imputed_missing.pickle'
        q = pre_processing.panel_path / filename
        pickle.dump(self.df, open(q, 'wb'))
        print('finished imputing missing for', self.initial_panel, 'dataset')
        print()
        print()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def compare_original_and_imputed_var(self, var_list):
        # ----------- open imputed dataframe ----------------------------------
        f_name = self.initial_panel + '_imputed_missing.pickle'
        q = pre_processing.panel_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        df2 = df.copy(deep=True)
        # ----------- take out rows with missing for easy comparison -----------
        for var in var_list:
            varcols = [var + '_' + i for i in self.all_panels]
            df3 = df2[varcols]
            df3 = df3[df3.isna().any(axis=1)]
            appids = df3.index.tolist()
            imputed_var = ['Imputed' + var + '_' + i for i in self.all_panels]
            varcols.extend(imputed_var)
            df4 = df2[varcols]
            df4 = df4.loc[appids]
            # ---------- save ------------------------------------------------------
            filename = self.initial_panel + '_' + var + '_compare_missing_with_imputed.csv'
            q = pre_processing.panel_path / 'check_imputed' / filename
            df4.to_csv(q)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def find_rows_contain_no_missing(self, df):
        """
        True means contains at least one (or more) missing
        False in contains_missing means this row has no missing panels
        """
        df2 = df.copy(deep=True)
        df3 = df2.loc[df2.isnull().any(axis=1)]
        df2['contains_missing'] = False
        df2['contains_missing'].loc[df3.index] = True
        df4 = df2.loc[df2['contains_missing'] == False]
        return df4.index.tolist()

    def find_rows_contain_any_missing(self, df):
        """
        True means contains at least one (or more) missing
        False in contains_missing means this row has no missing panels
        """
        df2 = df.copy(deep=True)
        df3 = df2.loc[df2.isnull().any(axis=1)]
        df2['contains_missing'] = False
        df2['contains_missing'].loc[df3.index] = True
        return df3.index.tolist()

    def find_rows_contain_all_missing(self, df):
        """
        True means contains ALL missings in every single panel (no non-missing)
        False in all_missing means this row contains at least one non-missing value
        """
        df2 = df.copy(deep=True)
        df3 = df2.loc[df2.isnull().all(axis=1)]
        df2['all_missing'] = False
        df2['all_missing'].loc[df3.index] = True
        return df3.index.tolist()

    def find_rows_to_remove_after_imputation(self, var_list):
        """
        The var_list contains only imputed variables, remove rows contain any missing
        """
        appids_to_remove = []
        for i in var_list:
            df = self.select_the_var(var=i)
            appids = self.find_rows_contain_any_missing(df=df)
            print(self.initial_panel, ' has : ', len(appids),
                  ' apps contains at least 1 missing in ', i)
            appids_to_remove.extend(appids)
        unique_appids = list(set(appids_to_remove))
        print(self.initial_panel, ' has in TOTAL : ', len(unique_appids),
              ' unique apps contains at least 1 missing in any of the imputed variables')
        print()
        self.appids_to_remove = unique_appids
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)

    def drop_rows(self):
        """
        This function is used for deleting variables contain at least one missing after imputation
        """
        print(self.initial_panel, ' before remove rows : ', self.df.shape)
        print('removing ', len(self.appids_to_remove), ' for all imputed variables.')
        appids = self.appids_to_remove
        print('removing ', len(self.appids_with_changing_developers), ' for appids that changed developers over time.')
        appids.extend(self.appids_with_changing_developers)
        unique_appids = list(set(appids))
        print('removing in total ', len(unique_appids), ' unique appids.')
        self.df = self.df.drop(index=unique_appids)
        print(self.initial_panel, ' after remove rows : ', self.df.shape)
        print()
        # -------------------- save -----------------------------------
        filename = self.initial_panel + '_imputed_and_deleted_missing.pickle'
        q = pre_processing.panel_path / filename
        pickle.dump(self.df, open(q, 'wb'))
        print('finished deleting missing for', self.initial_panel, 'dataset')
        print()
        print()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              tcn=self.tcn,
                              appids_to_remove=self.appids_to_remove,
                              appids_with_changing_developers=self.appids_with_changing_developers)


