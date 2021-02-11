import pandas as pd
from tqdm import tqdm
import numpy as np
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

    def __init__(self, merged_df, initial_date=None, all_panels=None,
                 numeric_vars=None, dummy_vars=None, text_vars=None, datetime_vars=None,
                 address_vars=None, misc_vars=None, time_variant_vars=None, time_invariant_vars=None):
        self.df = merged_df
        self.initial_date = initial_date
        self.all_panels = all_panels
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

    def search_col_contains(self, text):
        col_list = []
        for col in self.df.columns:
            if text in col:
                col_list.append(col)
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
    def __init__(self, merged_df, initial_date=None, all_panels=None,
                 numeric_vars=None, dummy_vars=None, text_vars=None, datetime_vars=None,
                 address_vars=None, misc_vars=None, missing_ratio=None):
        super().__init__(merged_df, initial_date, all_panels, numeric_vars, dummy_vars,
                         text_vars, datetime_vars, address_vars, misc_vars)
        self.missing_ratio = missing_ratio # below the missing ratio, do not need to impute, just delete

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
class select_vars_for_graphs(impute_missing):

    pass



#################################################################################################################
class line_graphs(select_vars_for_graphs):
    # inside init meaning the default value could be None for child class
    def __init__(self, merged_df, initial_date=None, all_panels=None,
                 numeric_vars=None, dummy_vars=None, text_vars=None, datetime_vars=None,
                 address_vars=None, misc_vars=None, interval_start=None, interval_end=None):
        # but inside super __init__ calling the parent properties, should not set None.
        super().__init__(merged_df, initial_date, all_panels, numeric_vars, dummy_vars, text_vars, datetime_vars, address_vars, misc_vars)
        # new child properties are added
        self.interval_start = interval_start
        self.interval_end = interval_end


    def change_in_var_over_interval(self):
        start_date_var = self.v1 + '_' + self.int_start
        end_date_var = self.v1 + '_' + self.int_end
        the_df = self.impute_missing()
        the_df.loc[:, 'change_over_interval'] = the_df.loc[:, end_date_var] - the_df.loc[:, start_date_var]
        the_df = the_df.sort_values(by='change_over_interval', ascending=False)
        n_rows = int(len(the_df.axes[0]))
        x = getattr(self, 'br', None)
        if x is not None:
            top_performers = the_df.loc[the_df['change_over_interval'] > x]
            top_performers = top_performers.index.tolist()
        elif n_rows >= 10:
            top_performers = the_df.head(10)
            top_performers = top_performers.index.tolist()
        else:
            top_performers = the_df
            top_performers = top_performers.index.tolist()
        # the_df.drop('change_over_interval', inplace=True, axis=1)
        # the_df = the_df.T
        return the_df


    def select_vars_for_graph(self, sample=None, var_type=None):
        if sample is not None:
            df2 = self.df.sample(sample)
        else:
            df2 = self.df
        # if var_type is not None:
        #     if var_type == 'numeric':
        #         for var_name in self.numeric_vars:
        #             col_names = list(map(lambda x: var_names + str(x), self.panels))
        #             df3 = df2[col_names]
        #             for i in col_names:
        #                 df3[i] = df3[i].astype(float)
        #     elif var_type == 'dummy':
        #         for var_name in self.dummy_vars:
        #     elif var_type == 'text':
        #         for var_name in self.text_vars:
        #     elif var_type == 'geography':
        #         for var_name in self.address_vars:
        #     elif var_type == 'date':
        #         for var_name in self.datetime_vars:
        return df2




#####################################################################################
        # figure out which apps have increase in the variable over the entire period

        # create a column for graphing xticks (since you already modified panels, so you can skip adding the initial date)
        # time_axis = []
        # for j in panels:
        #     time_axis.append(datetime.datetime.strptime(j, '%Y%m').strftime('%Y %b'))
        # C['x'] = time_axis
        #
        # # take the title of app as the value of a dictionary for annotation in plotting
        # top_performers_dict = dict.fromkeys(top_performers)
        # title_column = 'title_' + initial_date
        # for app_id in top_performers_dict.keys():
        #     top_performers_dict[app_id] = B.at[app_id, title_column]
        #
        # return(C, top_performers_dict)
