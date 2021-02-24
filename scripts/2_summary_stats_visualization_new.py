import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from datetime import datetime as dt
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

    def __init__(self,
                 df,
                 df_index,
                 df_developer_index=None,
                 df_developer_index_geocoded=None,
                 df_multiindex=None,
                 df_multiindex_geocoded=None,
                 initial_panel=None,
                 all_panels=None,
                 consec_panels=None):
        self.df = df
        self.df_index = df_index
        self.df_di = df_developer_index
        self.df_dig = df_developer_index_geocoded
        self.df_mi = df_multiindex
        self.df_mig = df_multiindex_geocoded
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels

    # SELECTION FUNCTIONS
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

    def select_the_panel(self, panel):
        col_list = [i for i in self.df.columns if panel in i]
        return col_list

    def select_the_var(self, var, consecutive=False, select_one_panel=None):
        if consecutive is True:  # only panels scraped in 202009 and onwards are monthly panels
            col_list = [var + '_' + i for i in self.consec_panels]
            panel_list = self.consec_panels
        elif consecutive is False:
            col_list = [var + '_' + i for i in self.all_panels]
            panel_list = self.all_panels
        elif select_one_panel is not None:
            col_list = [var + '_' + select_one_panel]
            panel_list = select_one_panel
        return col_list, panel_list

    def select_var_df(self, var_list, consecutive=False, select_one_panel=None):
        dfs = []
        for var in var_list:
            col_list, panel_list = self.select_the_var(var, consecutive, select_one_panel)
            df2 = self.keep_cols(list_of_col_names=col_list)
            dfs.append(df2)
        return dfs

    def peek_at_df_conditional_on_var_value(self, var, var_value, consecutive=False, select_one_panel=None, full_df=False):
        """you can also use this to peek_at_missing, just set var_value=None"""
        col_list, panel_list = self.select_the_var(var=var, consecutive=consecutive, select_one_panel=select_one_panel)
        if full_df is False:
            df2 = self.keep_cols(list_of_col_names=col_list)
        elif full_df is True:
            df2 = self.df
        for j in df2.columns:
            if full_df is False:
                if var_value is not None:
                    pass

    def peek_at_appid_and_var(self, appid, var):
        col, panels = self.select_the_var(var=var)
        new_df = self.keep_cols(list_of_col_names=col)
        new_df = new_df.loc[[appid]]
        return new_df

    def peek_at_sample_var_panels(self, var, sample):
        col, panels = self.select_the_var(var=var)
        new_df = self.keep_cols(list_of_col_names=col)
        new_df = new_df.sample(n=sample)
        return new_df

    def keep_rows(self, list_of_row_labels):
        if self.df_index == 'appid':
            new_df = self.df.loc[list_of_row_labels]
            return new_df
        elif self.df_index == 'dev_multiindex_geocoded':
            new_df = self.df[self.df.index.get_level_values('appId').isin(list_of_row_labels)]
            return new_df

    def keep_cols(self, list_of_col_names):
        new_df = self.df[list_of_col_names]
        return new_df

    def keep_both_cols_and_rows(self, list_of_row_labels, list_of_col_names):
        new_df = self.keep_rows(list_of_row_labels=list_of_row_labels)
        new_df2 = new_df[list_of_col_names]
        return new_df2

    def drop_rows(self, list_of_row_labels):
        if self.df_index == 'appid':
            new_df = self.df.drop(index=list_of_row_labels)
            return new_df
        elif self.df_index == 'dev_multiindex_geocoded':
            rows_index_to_drop = self.df.index.get_level_values('appId').isin(list_of_row_labels)
            rows_index_to_keep = ~rows_index_to_drop
            new_df = self.df[rows_index_to_keep]
            return new_df

    def drop_cols(self, list_of_col_names):
        new_df = self.df.drop(columns=list_of_col_names)
        return new_df

    def drop_both_cols_and_rows(self, list_of_row_labels, list_of_col_names):
        new_df = self.df.drop(index=list_of_row_labels)
        new_df2 = new_df.drop(columns=list_of_col_names)
        return new_df2

    def replace_cols_list(self, list_new_cols):
        combined_cols = functools.reduce(lambda a, b: a.join(b, how='inner'), list_new_cols)
        col_names = combined_cols.columns.tolist()
        new_df = self.drop_cols(list_of_col_names=col_names)
        new_df = new_df.join(combined_cols, how='inner')
        return new_df

    def replace_cols(self, new_cols):
        col_names = new_cols.columns.tolist()
        new_df = self.drop_cols(list_of_col_names=col_names)
        new_df = new_df.join(new_cols, how='inner')
        return new_df

    def drop_col_row_and_replace_cols(self, list_of_row_labels, list_of_col_names, new_cols):
        new_df = self.drop_both_cols_and_rows(list_of_row_labels=list_of_row_labels, list_of_col_names=list_of_col_names)
        col_names = new_cols.columns.tolist()
        new_df2 = new_df.drop(columns=col_names)
        if self.df_index == 'appid':
            new_df2 = new_df2.join(new_cols, how='inner')
            return new_df2
        elif self.df_index == 'dev_multiindex_geocoded':
            new_df2 = new_df2.join(new_cols, on=['developer', 'appId'], how='inner')
            return new_df2

    # STANDARD SINGLE VAR STATS
    ###############################################################################################################################
    def mean_of_var_panels(self, var):
        col, panels = self.select_the_var(var=var)
        new_df = self.keep_cols(list_of_col_names=col)
        new_df_mean = new_df.mean(axis=1).to_frame(name=var+'_stats')
        new_df = new_df.join(new_df_mean, how='inner')
        new_df.sort_values(by=var+'_stats', axis=0, ascending=False, inplace=True)
        return new_df

    def standard_deviation_of_var_panels(self, var):
        col, panels = self.select_the_var(var=var)
        new_df = self.keep_cols(list_of_col_names=col)
        new_df_std = new_df.std(axis=1).to_frame(name=var+'_stats')
        new_df = new_df.join(new_df_std, how='inner')
        new_df.sort_values(by=var+'_stats', axis=0, ascending=False, inplace=True)
        return new_df

    # REVERT INDEX (FROM APPID to DEVELOPER)
    ###############################################################################################################################
    def lat_and_long_columns(self, multiindex=False, consecutive=False):
        dfd = self.convert_appid_to_developer_index(multiindex=multiindex, consecutive=consecutive)
        # ------------ start geociding ------------------------------------------------------------
        geopy.geocoders.options.default_timeout = 7
        geolocator = AzureMaps(subscription_key='zLTKWFX7Ng5foT0nxB-CD-vgriqXUiNlk4IMhfD-PTQ')
        dfd['location'] = dfd['developerAddress_'+self.all_panels[0]].progress_apply(lambda loc: geolocator.geocode(loc) if loc else None)
        dfd['longitude'] = dfd['location'].apply(lambda loc: loc.longitude if loc else None)
        dfd['latitude'] = dfd['location'].apply(lambda loc: loc.latitude if loc else None)
        self.df_dig = dfd
        return dfd

    def convert_appid_to_developer_index(self, multiindex, consecutive=False):
        time_invariant_df, time_invariant_appids = self.check_whether_var_varies_across_panels(var='developer', consecutive=consecutive)
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
                col_list, panel_list = self.select_the_var(var=v, consecutive=consecutive)
                cols.extend(col_list)
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
            return df2

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

    def check_whether_var_varies_across_panels(self, var, consecutive=False):
        """without dropping missings first, just use this single function to find out time variant rows and delete them"""
        col_list, panel_list = self.select_the_var(var=var, consecutive=consecutive)
        df2 = self.df[col_list]
        if var == 'developer':
            for j in df2.columns:
                df2[j] = df2[j].apply(lambda x: self.format_text_for_developer(x))
        df_time_invariant_indicators = [] # true, time invariant; false, NOT time invariant
        for index, row in df2.iterrows():
            row_time_invariant_indicators = []
            for j in range(len(df2.columns)-1):
                if row[df2.columns[j]] == row[df2.columns[j+1]]:
                    row_time_invariant_indicators.append(True)
                else:
                    row_time_invariant_indicators.append(False)
            if all(row_time_invariant_indicators) is True:
                df_time_invariant_indicators.append(True)
            else:
                df_time_invariant_indicators.append(False)
        time_invariant_df = df2[df_time_invariant_indicators]
        time_invariant_appids = time_invariant_df.index.tolist()
        return time_invariant_df, time_invariant_appids

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
    def count_missing(self, var_list, consecutive=False, select_one_panel=None, group_by=None):
        dfs = self.select_var_df(var_list=var_list, consecutive=consecutive, select_one_panel=select_one_panel)
        summary_dfs = []
        for df in dfs:
            df3 = df.isnull().sum().rename('count missing').to_frame()
            summary_dfs.append(df3)
        combined_df = functools.reduce(lambda a, b: pd.concat([a, b]), summary_dfs)
        return combined_df

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

    def check_apps_with_consecutive_missing_panels(self, var, number_consec_panels_missing):
        col, panels = self.select_the_var(var=var)
        df2 = self.keep_cols(list_of_col_names=col)
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

    def impute_missing_using_adj_panels(self, var, adj_panels, method):
        col, panels = self.select_the_var(var=var)
        df2 = self.keep_cols(list_of_col_names=col)
        df_list = []
        for j in range(len(df2.columns)):
            if j <= adj_panels // 2 or j in [0, 1]:
                df = df2.iloc[:, 0:adj_panels + 1]
            elif j >= len(df2.columns) - adj_panels // 2 - 1:
                df = df2.iloc[:, len(df2.columns) - adj_panels - 1:len(df2.columns)]
            else:
                if adj_panels == 1:
                    df = df2.iloc[:, j - 1:j + 1]
                else:
                    df = df2.iloc[:, j - adj_panels // 2:j + adj_panels // 2 + 1]
            if method == 'mean':
                df[method] = df.mean(axis=1, skipna=True)
            elif method == 'mode':
                df[method] = df.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
            elif method == 'previous':
                df[method] = df.iloc[:, 0]
            else:
                df[method] = 0
            dfd = copy.deepcopy(df)
            for col in dfd.columns:
                dfd.loc[dfd[col].isnull(), col] = dfd[method]
            dfd = dfd[[df2.columns[j]]]
            df_list.append(dfd)
        imputed_df = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
        return imputed_df

#################################################################################################################
class summary_stats_tables(pre_processing):
    def __init__(self, missing_ratio, **kwargs):
        super().__init__(**kwargs)
        self.missing_ratio = missing_ratio

    # Output summary stats tables (app level features are to be performed on app-level dataframe)
    ###############################################################################################################################
    def stats_table_numeric(self, var_list, consecutive=False, select_one_panel=None, group_by=None):
        dfs = self.select_var_df(var_list=var_list, consecutive=consecutive, select_one_panel=select_one_panel)
        summary_dfs = []
        for df in dfs:
            df3 = df.agg(func=['count', 'min', 'max', 'mean', 'std'],
                         axis=0)
            if isinstance(df3, pd.Series):
                df3 = df3.to_frame()
            summary_dfs.append(df3)
        combined_df = functools.reduce(lambda a, b: a.join(b, how='inner'), summary_dfs)
        combined_df = combined_df.T
        return combined_df

    # there are no categorical variables, all the cat variables have been converted to dummy in 1_scraping_cleaning_merge.py
    # it shows that almost ALL dummies are NOT time invariant
    def stats_table_dummy(self, var_list, consecutive=False, select_one_panel=None, group_by=None):
        dfs = self.select_var_df(var_list=var_list, consecutive=consecutive, select_one_panel=select_one_panel)
        summary_dfs = []
        for df in dfs:
            for col in df.columns:
                df_col = df[[col]]
                df_col[col + '_count'] = 'count'
                tab = df_col.groupby(by=col, dropna=False).count()
                summary_dfs.append(tab)
        combined_df = functools.reduce(lambda a, b: a.join(b, how='inner'), summary_dfs)
        combined_df = combined_df.T
        combined_df['count'] = combined_df[0] + combined_df[1]
        return combined_df

    def stats_table_datetime(self, var_list, consecutive=False, select_one_panel=None, group_by=None):
        dfs = self.select_var_df(var_list=var_list, consecutive=consecutive, select_one_panel=select_one_panel)
        summary_dfs = []
        for df in dfs:
            df3 = df.agg(func=['count', 'min', 'max'],
                         axis=0)
            summary_dfs.append(df3)
        combined_df = functools.reduce(lambda a, b: a.join(b, how='inner'), summary_dfs)
        combined_df = combined_df.T
        return combined_df

    def stats_table_address(self, var='developerAddress', consecutive=False, select_one_panel=None, group_by=None):
        pass

    def combine_stats_tables(self, table_func_1, table_func_2, table_func_3=None):
        pass

    def convert_stats_tables(self, style):
        pass

#################################################################################################################
class single_variable_stats(pre_processing):
    def __init__(self, v1, var_type, **kwargs):
        super().__init__(**kwargs)
        self.v1 = v1
        self.var_type = var_type

    def tabulate(self):
        col, panels = self.select_the_var(var=self.v1)
        new_df = self.keep_cols(list_of_col_names=col)
        tab_list = []
        if self.var_type in ['category', 'dummy']:
            for i in self.all_panels:
                new_df[i+'_count'] = 'count'
                tab = new_df[[self.v1+'_'+i, i+'_count']].groupby(by=self.v1+'_'+i, dropna=False).count()
                tab_list.append(tab)
            final_df = functools.reduce(lambda a, b : a.join(b, how='inner'), tab_list)
        return final_df

    def transform_the_var(self, log=False, standardize=False, min_max=False):
    # https://towardsdatascience.com/catalog-of-variable-transformations-to-make-your-model-works-better-7b506bf80b97
        col, panels = self.select_the_var(var=self.v1)
        new_df = self.keep_cols(list_of_col_names=col)
        if log is True:
            cols_to_keep = [i+'_log' for i in new_df.columns]
            for i in new_df.columns:
                new_df[i+'_log'] = new_df[i].apply(lambda x: math.log(x+1))
        new_df = new_df[cols_to_keep]
        return new_df

    def binning_the_var(self, num_bins, pre_transformation=None):
        if pre_transformation == 'log':
            new_df = self.transform_the_var(log=True)
        else: # when no pre-transformation is performed
            col, panels = self.select_the_var(var=self.v1)
            new_df = self.keep_cols(list_of_col_names=col)
        for i in new_df.columns:
            new_df[i + '_bin'] = pd.cut(new_df[i], bins=num_bins, labels=False)
        return new_df

    def bin_boundary_and_count(self, num_bins, pre_transformation=None, view=None):
        new_df = self.binning_the_var(num_bins, pre_transformation)
        result_df_list = []
        if pre_transformation is None:
            for i in self.all_panels:
                couple_df = new_df[[self.v1+'_'+i, self.v1+'_'+i+'_bin']]
                result_df = couple_df.groupby(by=self.v1+'_'+i+'_bin', dropna=False).agg(['mean', 'max', 'min', 'count'])
                result_df_list.append(result_df)
        elif pre_transformation == 'log':
            for i in self.all_panels:
                couple_df = new_df[[self.v1+'_'+i+'_log', self.v1+'_'+i+'_log_bin']]
                result_df = couple_df.groupby(by=self.v1+'_'+i+'_log_bin', dropna=False).agg(['mean', 'max', 'min', 'count'])
                result_df_list.append(result_df)
        if view is not None:
            df_v_list = []
            for df in result_df_list:
                col_list = [i for i in df.columns if view in i]
                df_v = df[col_list]
                df_v_list.append(df_v)
            final_df = functools.reduce(lambda a, b: a.join(b, how='inner'), df_v_list)
        else:
            final_df = functools.reduce(lambda a, b: a.join(b, how='inner'), result_df_list)
        return final_df

    def change_in_var_over_interval(self, int_start, int_end):
        col, panels = self.select_the_var(var=self.v1)
        new_df = self.keep_cols(list_of_col_names=col)
        star_var = self.v1 + '_' + int_start
        end_var = self.v1 + '_' + int_end
        if self.var_type == 'continuous':
            new_df['change_in_'+self.v1] = new_df[end_var] - new_df[star_var]
            new_df.sort_values(by='change_in_'+self.v1, axis=0, ascending=False, inplace=True, na_position='first')
        return new_df

    def delta_one_period_change_in_var(self):
        col, panels = self.select_the_var(var=self.v1, consecutive_panels=True)
        new_df = self.keep_cols(list_of_col_names=col)
        for i in range(len(self.consec_panels)):
            if i != 0:
                new_col_name = self.v1+'_'+self.consec_panels[i]+'_'+self.consec_panels[i-1]
                new_df[new_col_name] = new_df[self.v1+'_'+self.consec_panels[i]] - new_df[self.v1+'_'+self.consec_panels[i-1]]
        return new_df

    def density_distribution(self):
        col, panels = self.select_the_var(var=self.v1)
        new_df = self.keep_cols(list_of_col_names=col)
        # ----------------------------------------------------------------------------------
        fig, axs = plt.subplots(nrows=len(self.all_panels), sharex=True, sharey=True, figsize=(10, 80), dpi=500, facecolor='white')
        fig.suptitle(self.v1+' Density Distributions')
        if self.var_type == 'continuous':
            for i in range(len(new_df.columns)):
                new_df[new_df.columns[i]].plot.kde(ax=axs[i])
                axs[i].set_title(new_df.columns[i] +' kde plot')
        for ax in axs:
            ax.label_outer()
        return fig

    def histogram(self, num_bins, pre_transformation=None):
        if pre_transformation=='log':
            new_df = self.transform_the_var(log=True)
        elif pre_transformation is None:
            col, panels = self.select_the_var(var=self.v1)
            new_df = self.keep_cols(list_of_col_names=col)
        # ----------------------------------------------------------------------------------
        fig, axs = plt.subplots(nrows=len(self.all_panels), sharex=True, sharey=True, figsize=(10, 80), dpi=500, facecolor='white')
        fig.suptitle(self.v1+' Histogram')
        if self.var_type == 'continuous':
            for i in range(len(new_df.columns)):
                new_df[new_df.columns[i]].hist(ax=axs[i], bins=num_bins)
                axs[i].set_title(new_df.columns[i] +' histogram')
        for ax in axs:
            ax.label_outer()
        return fig


#################################################################################################################
class two_variables_stats(pre_processing):
    def __init__(self, v1, v1_type, v2, v2_type, **kwargs):
        super().__init__(**kwargs)
        self.v1 = v1
        self.v1_type = v1_type
        self.v2 = v2
        self.v2_type = v2_type

    def cross_tab(self):
        pass
