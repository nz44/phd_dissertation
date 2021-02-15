import pandas as pd
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from datetime import datetime as dt
from datetime import date
import functools
import collections
import operator
import functools
import re
import matplotlib
import seaborn
#################################################################################################################
# the input dataframe are the output of merge_panels_into_single_df() method of app_detail_dicts class
class summary_statistics():

    def __init__(self, merged_df, initial_panel=None, all_panels=None, consec_panels=None,
                 numeric_vars=None, dummy_vars=None, text_vars=None, datetime_vars=None,
                 address_vars=None, misc_vars=None, time_variant_vars=None, time_invariant_vars=None):
        self.df = merged_df
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
        self.numeric_vars = numeric_vars
        self.dummy_vars = dummy_vars
        self.text_vars = text_vars
        self.datetime_vars = datetime_vars
        self.misc_vars = misc_vars
        self.address_vars = address_vars
        self.time_variant_vars = time_variant_vars
        self.time_invariant_vars = time_invariant_vars

    def print_col_names(self):
        for col in self.df.columns:
            print(col)

    def print_num_rows(self):
        print(len(self.df.index))

    def search_col_contains(self, text, consecutive_panels=False):
        if consecutive_panels is True: # only panels scraped in 202009 and onwards are monthly panels
            col_list = [text + '_' + i for i in self.consec_panels]
        else:
            col_list = [text + '_' + i for i in self.all_panels]
        # print(col_list)
        return col_list

    def drop_cols(self, list_of_col_names):
        new_df = self.df.drop(list_of_col_names, axis=1)
        # print('Before dropping we have', len(self.df.columns), 'columns')
        # print('After dropping we have', len(new_df.columns), 'columns')
        return new_df

    def keep_cols(self, list_of_col_names):
        new_df = self.df[list_of_col_names]
        # print('Before keeping we have', len(self.df.columns), 'columns')
        # print('After keeping we have', len(new_df.columns), 'columns')
        return new_df

    def replace_cols(self, new_cols):
        col_names = new_cols.columns.tolist()
        new_df = self.drop_cols(list_of_col_names=col_names)
        new_df = new_df.join(new_cols, how='inner')
        if len(new_df.index) == len(self.df.index):
            print('successfully replaced the old cols with replacement cols:')
            print(col_names)
        return new_df

    def drop_rows(self, list_of_row_labels):
        new_df = self.df.drop(list_of_row_labels, axis=0)
        print('Before dropping we have', len(self.df.index), 'rows')
        print('After dropping we have', len(new_df.index), 'rows')
        return new_df

    def peek_at_appid_and_var(self, appid, var):
        l1 = self.search_col_contains(text=var)
        new_df = self.keep_cols(list_of_col_names=l1)
        new_df = new_df.loc[[appid]]
        return new_df

    def peek_at_sample_var_panels(self, var, sample):
        l1 = self.search_col_contains(text=var)
        new_df = self.keep_cols(list_of_col_names=l1)
        new_df = new_df.sample(n=sample)
        return new_df

    def mean_of_var_panels(self, var):
        l1 = self.search_col_contains(text=var)
        new_df = self.keep_cols(list_of_col_names=l1)
        new_df_mean = new_df.mean(axis=1).to_frame(name=var+'_stats')
        new_df = new_df.join(new_df_mean, how='inner')
        new_df.sort_values(by=var+'_stats', axis=0, ascending=False, inplace=True)
        return new_df

    def standard_deviation_of_var_panels(self, var):
        l1 = self.search_col_contains(text=var)
        new_df = self.keep_cols(list_of_col_names=l1)
        new_df_std = new_df.std(axis=1).to_frame(name=var+'_stats')
        new_df = new_df.join(new_df_std, how='inner')
        new_df.sort_values(by=var+'_stats', axis=0, ascending=False, inplace=True)
        return new_df

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


#################################################################################################################
class impute_missing(summary_statistics):
    def __init__(self, missing_ratio, **kwargs):
        super().__init__(**kwargs)
        self.missing_ratio = missing_ratio

    def cols_missing_ratio(self):
        num_of_cols_above_missing_threshold = 0
        missing_cols_and_missing_ratios = []
        missing_cols = []
        for col in self.df.columns:
            null_data = self.df[[col]][self.df[[col]].isnull().any(axis=1)]
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
            null_data = df_t[[col]][df_t[[col]].isnull().any(axis=1)]
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

    def peek_at_missing(self, var, sample):
        for col in self.df.columns:
            if var in col: # var is a substring of the full column name
                null_df = self.df[[col]][self.df[[col]].isnull().any(axis=1)]
                print(col, 'contains', len(null_df.index), 'of missing values')
                if len(null_df.index) >= sample:
                    null_sample = null_df.sample(sample)
                    print('randomly select', sample, 'rows from', col)
                    print(null_sample)
                    print()

    # STRATEGY 1: ------------------------------------------------------------------
    ### VARS: minInstalls
    # if the missing panel(s) are in between none-missing ones, take the average of before and after panel values and fill in the missing
    # if the minInstalls are missing for consecutively three panels or more, delete that row (because this is an important variable).
    def check_apps_with_consecutive_missing_panels(self, var, number_consec_panels_missing):
        l1 = self.search_col_contains(text=var)
        df2 = self.keep_cols(list_of_col_names=l1)
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

    def impute_the_missing_panel_according_to_adjacent_panel(self, var): # the self.df here should be the newly passed df that has deleted all rows and cols that will not be imputed
        l1 = self.search_col_contains(text=var)
        df2 = self.keep_cols(list_of_col_names=l1)
        for i in range(len(df2.columns)):
            if i == 0: # the first panel is missing, impute with the next panel
                df2[df2.columns[i]] = df2.apply(
                    lambda row: row[df2.columns[i+1]] if np.isnan(row[df2.columns[i]]) else row[df2.columns[i]],
                    axis=1
                )
            else: # all other panels impute with previous panels
                df2[df2.columns[i]] = df2.apply(
                    lambda row: row[df2.columns[i-1]] if np.isnan(row[df2.columns[i]]) else row[df2.columns[i]],
                    axis=1
                )
        return df2


#################################################################################################################
class single_variable_stats(summary_statistics):
    def __init__(self, v1, var_type, **kwargs):
        super().__init__(**kwargs)
        self.v1 = v1
        self.var_type = var_type

    def tabulate(self):
        l1 = self.search_col_contains(text=self.v1)
        new_df = self.keep_cols(list_of_col_names=l1)
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
        l1 = self.search_col_contains(text=self.v1)
        new_df = self.keep_cols(list_of_col_names=l1)
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
            l1 = self.search_col_contains(text=self.v1)
            new_df = self.keep_cols(list_of_col_names=l1)
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
        l1 = self.search_col_contains(text=self.v1)
        new_df = self.keep_cols(list_of_col_names=l1)
        star_var = self.v1 + '_' + int_start
        end_var = self.v1 + '_' + int_end
        if self.var_type == 'continuous':
            new_df['change_in_'+self.v1] = new_df[end_var] - new_df[star_var]
            new_df.sort_values(by='change_in_'+self.v1, axis=0, ascending=False, inplace=True, na_position='first')
        return new_df

    def delta_one_period_change_in_var(self):
        l1 = self.search_col_contains(text=self.v1, consecutive_panels=True)
        new_df = self.keep_cols(list_of_col_names=l1)
        for i in range(len(self.consec_panels)):
            if i != 0:
                new_col_name = self.v1+'_'+self.consec_panels[i]+'_'+self.consec_panels[i-1]
                new_df[new_col_name] = new_df[self.v1+'_'+self.consec_panels[i]] - new_df[self.v1+'_'+self.consec_panels[i-1]]
        return new_df

    def density_distribution(self):
        l1 = self.search_col_contains(text=self.v1)
        new_df = self.keep_cols(list_of_col_names=l1)
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