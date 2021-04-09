from pathlib import Path
import pickle
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
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
import geopy
from geopy.geocoders import AzureMaps
from tqdm import tqdm
tqdm.pandas()
import matplotlib
import seaborn
import copy
#################################################################################################################
# the input dataframe are the output of merge_panels_into_single_df() method of app_detail_dicts class
class pre_processing():

    missing_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/missing_counts')

    imputed_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__')

    def __init__(self,
                 df,
                 initial_panel=None,
                 all_panels=None,
                 df_developer_index=None,
                 df_developer_index_geocoded=None,
                 df_multiindex=None):
        self.df = df
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df_di = df_developer_index
        self.df_dig = df_developer_index_geocoded
        self.df_mi = df_multiindex

    # BASIC FUNCTIONS
    ###############################################################################################################################
    def print_col_names(self, text=None):
        cols = []
        if text is not None:
            for col in self.df.columns:
                if text in col:
                    print(col)
                    cols.append(col)
        elif text is None:
            for col in self.df.columns:
                print(col)
                cols = self.df.columns
        return cols

    def select_the_var(self, var, select_one_panel=None):
        col_list = [var + '_' + i for i in self.all_panels]
        if select_one_panel is not None:
            col_list = [var + '_' + select_one_panel]
        df2 = self.df.copy(deep=True)
        df2 = df2[col_list]
        return df2

    def select_dfs_from_var_list(self, var_list, select_one_panel=None):
        dfs = []
        for var in var_list:
            df2 = self.select_the_var(var, select_one_panel)
            dfs.append(df2)
        return dfs

    def drop_rows(self, list_of_row_labels):
        self.df = self.df.drop(index=list_of_row_labels)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)

    def drop_cols(self, list_of_col_names):
        self.df.drop(columns=list_of_col_names, inplace=True)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)

    # STANDARD SINGLE VAR STATS
    ###############################################################################################################################
    def mean_of_var_panels(self, var):
        new_df = self.select_the_var(var=var)
        new_df_mean = new_df.mean(axis=1).to_frame(name=var+'_stats')
        new_df = new_df.join(new_df_mean, how='inner')
        new_df.sort_values(by=var+'_stats', axis=0, ascending=False, inplace=True)
        return new_df

    def standard_deviation_of_var_panels(self, var):
        new_df = self.select_the_var(var=var)
        new_df_std = new_df.std(axis=1).to_frame(name=var+'_stats')
        new_df = new_df.join(new_df_std, how='inner')
        new_df.sort_values(by=var+'_stats', axis=0, ascending=False, inplace=True)
        return new_df

    # REVERT INDEX (FROM APPID to DEVELOPER)
    ###############################################################################################################################
    def lat_and_long_columns(self, multiindex=False):
        dfd = self.convert_appid_to_developer_index(multiindex=multiindex)
        # ------------ start geociding ------------------------------------------------------------
        geopy.geocoders.options.default_timeout = 7
        geolocator = AzureMaps(subscription_key='zLTKWFX7Ng5foT0nxB-CD-vgriqXUiNlk4IMhfD-PTQ')
        dfd['location'] = dfd['developerAddress_'+self.all_panels[0]].progress_apply(lambda loc: geolocator.geocode(loc) if loc else None)
        dfd['longitude'] = dfd['location'].apply(lambda loc: loc.longitude if loc else None)
        dfd['latitude'] = dfd['location'].apply(lambda loc: loc.latitude if loc else None)
        self.df_dig = dfd
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)

    def convert_appid_to_developer_index(self, multiindex):
        time_invariant_df, time_invariant_appids = self.check_whether_var_varies_across_panels(var='developer')
        df2 = self.df.loc[time_invariant_appids]
        if multiindex is True:
            df2 = df2.reset_index().set_index(['developer_'+self.initial_panel, 'index'])
            df2.index.rename(['developer', 'appId'], inplace=True)
            # remove developers b/c we have only kept time-invariant developer information
            for j in df2.columns:
                if 'developer_' in j:
                    df2.drop(j, axis=1, inplace=True)
            # add number of apps variable to each row
            df3 = df2.reset_index().groupby('developer')['appId'].nunique().rename('num_apps_owned').to_frame()

            df2 = df2.reset_index().merge(df3, on='developer', how='left')
            df2.set_index(['developer', 'appId'], inplace=True)
            self.df_mi = df2
            return df2
        elif multiindex is False:
            dev_level_vars = ['developer', 'developerId', 'developerEmail', 'developerWebsite', 'developerAddress']
            cols = []
            for v in dev_level_vars:
                new_df = self.select_the_var(var=v)
                cols.extend(new_df.columns.tolist())
            df2 = df2[cols]
            df2 = df2.reset_index(drop=True).set_index('developer_'+self.initial_panel)
            df2.index.rename('developer', inplace=True)
            # drop duplicate index rows
            index = df2.index
            is_duplicate = index.duplicated(keep="first")
            not_duplicate = ~is_duplicate
            df2 = df2[not_duplicate]
            # remove developers b/c we have only kept time-invariant developer information
            for j in df2.columns:
                if 'developer_' in j:
                    df2.drop(j, axis=1, inplace=True)
            self.df_di = df2
            return pre_processing(df=self.df,
                                  initial_panel=self.initial_panel,
                                  all_panels=self.all_panels,
                                  df_developer_index_geocoded=self.df_dig,
                                  df_developer_index=self.df_di,
                                  df_multiindex=self.df_mi)

    # TIME INVARIANT VARIABLE
    ###############################################################################################################################
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

    # DELETE OUTLIERS
    ###############################################################################################################################
    def peek_at_outliers(self, var, method, quantiles, q_inter, **kwargs):
        # method determines which var you are using histogram over, if none, use it over the var itself
        # if average, or standard deviation, first calculate the average and std of the var over all panels, then draw the histogram on that average or std
        if method == 'std':
            df_with_stats = self.standard_deviation_of_var_panels(var=var)
        elif method == 'mean':
            df_with_stats = self.mean_of_var_panels(var=var)
        # -------------------------------------------------------------
        s_q = df_with_stats[[var + '_stats']].quantile(q=quantiles, axis=0, interpolation=q_inter)
        if 'ind' in kwargs.keys():
            ax = df_with_stats[var + '_stats'].plot.kde(ind=kwargs['ind'])
        else:
            ax = df_with_stats[var+'_stats'].plot.kde()
        return s_q, ax

    def define_outlier_appids(self, var, method, cutoff_q, q_inter): # first peek_at_outliers, then decide at which quantile to truncate the data
        if method == 'std':
            df_with_stats = self.standard_deviation_of_var_panels(var=var)
            s_q = df_with_stats[[var + '_stats']].quantile(q=cutoff_q, axis=0, interpolation=q_inter)
        # -------------------------------------------------------------
        cutoff_value = s_q.iat[0] # this is pandas series
        print('The cutoff value for', var, 'at', cutoff_q, '_th quantile is', cutoff_value)
        df_outliers = df_with_stats.loc[(df_with_stats[var+'_stats'] >= cutoff_value)]
        print('number of outliers are', len(df_outliers.index), 'out of', len(df_with_stats.index), 'total apps.')
        outlier_appids = df_outliers.index.tolist()
        return df_outliers, outlier_appids

    # IMPUTE MISSING and DELETING MISSING that cannot be imputed
    ###############################################################################################################################
    # need to delete those apps with missing in developer before finding out which apps have changed developers over time
    def count_missing(self, var_list, name, select_one_panel=None):
        dfs = self.select_dfs_from_var_list(var_list=var_list,
                                            select_one_panel=select_one_panel)
        summary_dfs = []
        for df in dfs:
            df3 = df.isnull().sum().rename('count missing').to_frame()
            summary_dfs.append(df3)
        combined_df = functools.reduce(lambda a, b: pd.concat([a, b]), summary_dfs)
        filename = self.initial_panel + '_' + name + '.csv'
        q = pre_processing.missing_path / filename
        combined_df.to_csv(q)
        print(combined_df)
        print()
        print()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)

    def cols_missing_ratio(self):
        num_of_cols_above_missing_threshold = 0
        missing_cols_and_missing_ratios = []
        missing_cols = []
        for col in self.df.columns:
            null_data = self.df[[col]][self.df[col].isnull()]
            r = len(null_data.index) / len(self.df.index)
            if r >= self.missing_ratio:
                num_of_cols_above_missing_threshold += 1
                missing_cols_and_missing_ratios.append((col, r))
                missing_cols.append(col)
        print('total number of columns contain missing value above', self.missing_ratio, 'is', num_of_cols_above_missing_threshold)
        print('out of total number of columns', len(self.df.columns))
        print(missing_cols_and_missing_ratios)
        return missing_cols_and_missing_ratios, missing_cols

    def rows_missing_ratio(self):
        df_t = self.df.T
        num_of_cols_above_missing_threshold = 0
        missing_cols_and_missing_ratios = []
        missing_cols = []
        for col in df_t.columns:
            null_data = df_t[[col]][df_t[col].isnull()]
            r = len(null_data.index) / len(df_t.index)
            if r >= self.missing_ratio:
                num_of_cols_above_missing_threshold += 1
                missing_cols_and_missing_ratios.append((col, r))
                missing_cols.append(col)
        print('total number of apps contain missing attributes above', self.missing_ratio, 'is',
              num_of_cols_above_missing_threshold)
        print('out of total number of apps', len(df_t.columns))
        print(missing_cols_and_missing_ratios)
        return missing_cols_and_missing_ratios, missing_cols

    def check_appids_with_at_least_one_missing(self, var):
        df2 = self.select_the_var(var=var)
        null_data = df2[df2.isnull().any(axis=1)]
        return list(null_data.index.values), null_data

    def check_apps_with_consecutive_missing_panels(self, var, number_consec_panels_missing):
        df2 = self.select_the_var(var=var)
        null_data = df2[df2.isnull().any(axis=1)]
        null_data_t = null_data.T
        appids_with_consec_missing_panels = []
        for appid in null_data_t.columns:
            app_panels = null_data_t[[appid]]
            # https://stackoverflow.com/questions/29007830/identifying-consecutive-nans-with-pandas
            # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
            # nan row (panel) will be 0 and none-missing rows (panel) wil be 1
            # and each row is the sum of all the rows preceding it
            # so when two rows have the same value means 1 missing row occured. Three rows have the same value, meaning two consecutive missing rows occured.
            cumlative_none_missing_df = app_panels.notnull().astype(int).cumsum()
            # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
            missing_count = cumlative_none_missing_df.groupby(cumlative_none_missing_df.columns.tolist(), as_index=False).size()
            consec_missing_count = missing_count['size'].max()
            threshold = number_consec_panels_missing+1
            if consec_missing_count >= threshold: # == 2 means only 1 panel is missing, == 3 means 2 consecutive panels are missing, note greater or equal than the threshold
                appids_with_consec_missing_panels.append(appid)
        appids_intend_to_drop = null_data_t[appids_with_consec_missing_panels]
        print('number of apps with at least', number_consec_panels_missing, 'consecutive missing panels for', var, 'are', len(appids_with_consec_missing_panels))
        print('out of', len(df2.index), 'apps.')
        return appids_intend_to_drop, appids_with_consec_missing_panels

    def print_unique_values(self, cat_var):
        cols = [cat_var + '_' + i for i in self.all_panels]
        for j in cols:
            unique_l = self.df[j].unique()
            print(j, 'contains', len(unique_l), 'unique values')
            print(unique_l)
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)

    def check_if_col_has_identical_value_except_for_missing(self, var):
        df2 = self.select_the_var(var=var)
        null_data = df2[df2.isnull().any(axis=1)]
        null_data_t = null_data.T
        appids_have_same_value_except_missing = []
        for j in null_data_t.columns:
            l1 = null_data_t[j]
            l2 = l1.dropna()
            l3 = list(set(l2.tolist()))
            if len(l3) == 1:
                appids_have_same_value_except_missing.append(j)
        dfr = null_data.loc[appids_have_same_value_except_missing]
        return dfr, appids_have_same_value_except_missing

    def select_imputation_panels(self, df, method, current_panel):
        """
        :param df: input dataframe is a deep copy of self.select_the_var(var=var)
        :param method: all panels before, all panels before and after
        :param current_panel: the time where you decide which panel are before and after in self.all_panels
        :return: the dataframe after selecting the useful panels
        """
        the_panel = datetime.strptime(current_panel, "%Y%m")
        if method == 'all panels before and after':
            return df
        elif method == 'all panels before':
            before_panels = []
            for i in self.all_panels:
                panel = datetime.strptime(i, "%Y%m")
                if panel <= the_panel:
                    before_panels.append(i)
            cols_to_keep = []
            for i in df.columns:
                z = i.split('_')
                if z[1] in before_panels:
                    cols_to_keep.append(i)
            return df[cols_to_keep]

    def check_if_all_non_missing_are_same_in_a_row(self, df):
        # https://stackoverflow.com/questions/22701799/pandas-dataframe-find-rows-where-all-columns-equal
        # https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.eq.html
        # first need to remove none from each ( comparing None to other value will return False)
        df2 = df.copy(deep=True)
        ready_to_concat = []
        for i in range(len(df.index)):
            df3 = df2.iloc[i, :].to_frame().T
            df3 = df3.dropna(axis='columns')
            if df3.empty: # means all columns in this row are None
                df3['nonmissings_are_same'] = False
            else:
                df3['nonmissings_are_same'] = df3.eq(df3.iloc[:, 0], axis=0).all(1)
            ready_to_concat.append(df3['nonmissings_are_same'])
        df4 = functools.reduce(lambda a, b: pd.concat([a, b]), ready_to_concat)
        return df4

    def check_whether_var_varies_across_panels(self, var):
        """without dropping missings first, just use this single function to find out time variant rows and delete them"""
        df2 = self.select_the_var(var=var)
        if var == 'developer':
            for j in df2.columns:
                df2[j] = df2[j].apply(lambda x: self.format_text_for_developer(x))
        nonmissings_are_same = self.check_if_all_non_missing_are_same_in_a_row(df=df2)
        time_invariant_df = df2.loc[nonmissings_are_same]
        time_invariant_appids = time_invariant_df.index.tolist()
        return time_invariant_df, time_invariant_appids

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
            nonmissings_are_same = self.check_if_all_non_missing_are_same_in_a_row(df=df2)
            df2['mode'] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
            df2[method] = None
            # assign 'mode if none-missing are all the same' to 'mode' when 'nonmissings_are_same' is True
            df2[method].loc[nonmissings_are_same] = df2['mode']
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
                elif var in ['free']:
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
                elif var in ['containsAds', 'offersIAP']:
                    the_imputed_panel = self.implement_imputation_methods(df=df4,
                                                                          method='mode if none-missing are all the same')
                df3['Imputed' + var + '_' + i] = df3[var + '_' + i]
                df3['Imputed' + var + '_' + i] = df3['Imputed' + var + '_' + i].fillna(the_imputed_panel)
        df3.drop(cols_to_drop_later, axis=1, inplace=True)
        print('finished imputing missing for variable : ', var)
        return df3

    def impute_list_of_vars(self, list_of_vars):
        imputed_dfs = []
        for i in list_of_vars:
            df3 = self.impute_missing(var=i)
            imputed_dfs.append(df3)
        combined_imputed = functools.reduce(lambda a, b: a.join(b, how='inner'), imputed_dfs)
        self.df = self.df.join(combined_imputed, how='inner')
        # -------------------- save -----------------------------------
        filename = self.initial_panel + '_imputed_missing.pickle'
        q = pre_processing.imputed_path / filename
        pickle.dump(self.df, open(q, 'wb'))
        print('finished imputing missing for', self.initial_panel, 'dataset')
        print()
        print()
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)


