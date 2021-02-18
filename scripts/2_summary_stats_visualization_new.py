import pandas as pd
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
import matplotlib
import seaborn
#################################################################################################################
# the input dataframe are the output of merge_panels_into_single_df() method of app_detail_dicts class
class summary_statistics():
    def __init__(self, df, df_developer_index=None, df_multiindex=None, initial_panel=None,
                 all_panels=None, consec_panels=None):
        self.df = df
        self.df_di = df_developer_index
        self.df_mi = df_multiindex
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
    def check_duplicate_indices(self, df_type): # df_type could be appid, or it could be developer, or dev_multi
        if df_type == 'appid':
            dup_index_list = self.df.index[self.df.index.duplicated()].tolist()
        return dup_index_list
    # ******************************************************************************
    # Output summary stats tables
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
                df_col[col+'_count'] = 'count'
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

    def stats_count_missing(self, var_list, consecutive=False, select_one_panel=None, group_by=None):
        dfs = self.select_var_df(var_list=var_list, consecutive=consecutive, select_one_panel=select_one_panel)
        summary_dfs = []
        for df in dfs:
            df3 = df.isnull().sum().rename('count missing').to_frame()
            summary_dfs.append(df3)
        combined_df = functools.reduce(lambda a, b: pd.concat([a, b]), summary_dfs)
        return combined_df

    def peek_at_missing(self, var, select_one_panel, full_df=True, additional_vars=None):
        col_list, panel_list = self.select_the_var(var=var, select_one_panel=select_one_panel)
        for v1 in col_list:
            if full_df is False:
                df = self.df[self.df[v1].isnull()]
                ccols = self.print_col_names(text=var)
                if additional_vars is not None:
                    for j in additional_vars:
                        cols = self.print_col_names(text=j)
                        ccols.extend(cols)
                df2 = df[ccols]
            elif full_df is True:
                df2 = self.df[self.df[v1].isnull()]
        return df2

    def stats_table_address(self, var='developerAddress', consecutive=False, select_one_panel=None, group_by=None):
        pass

    def combine_stats_tables(self, table_func_1, table_func_2, table_func_3=None):
        pass

    def convert_stats_tables(self, style):
        pass

    # ******************************************************************************
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

    def print_num_rows(self):
        print(len(self.df.index))

    def select_the_var(self, var, consecutive=False, select_one_panel=None):
        if consecutive is True: # only panels scraped in 202009 and onwards are monthly panels
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

    def change_index_from_appid_to_developerid(self):
        pass

    def select_the_panel(self, panel):
        col_list = [i for i in self.df.columns if panel in i]
        return col_list

    def keep_both_cols_and_rows(self, list_of_row_labels, list_of_col_names):
        new_df = self.keep_rows(list_of_row_labels=list_of_row_labels)
        new_df2 = new_df[list_of_col_names]
        return new_df2

    def keep_rows(self, list_of_row_labels):
        new_df = self.df.loc[list_of_row_labels]
        return new_df

    def keep_cols(self, list_of_col_names):
        new_df = self.df[list_of_col_names]
        return new_df

    # need to delete those apps with missing in developer before finding out which apps have changed developers over time
    def appids_that_have_missing_in_any_panels(self, var, consecutive=False):
        col_list, panel_list = self.select_the_var(var=var, consecutive=consecutive)
        df2 = self.keep_cols(list_of_col_names=col_list)
        data = df2[df2.isnull().any(axis=1)]
        appids = data.index.tolist()
        return appids, data

    def appids_have_time_variant_var(self, var, consecutive=False, format_text=False):
        df_list, diff_dfs = self.check_whether_var_is_time_invariant(var=var, consecutive=consecutive, format_text=format_text)
        combined_appids = []
        for df in diff_dfs:
            if len(df.index) != 0:
                # df.dropna(axis=1, how='all', inplace=True)
                appid_list = df.index.tolist()
                combined_appids.extend(appid_list)
        unique_index = list(set(combined_appids))
        return diff_dfs, unique_index

    def format_text_for_developer(self, text):
        if text is not None:
            result_text = ''.join(c.lower() for c in text if not c.isspace()) # remove spaces
            punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~+''' # remove functuations
            for ele in result_text:
                if ele in punc:
                    result_text = result_text.replace(ele, "")
            extra1 = re.compile(r'(corporated$)|(corporation$)|(corp$)|(company$)|(limited$)|(games$)|(game$)|(studios$)|(studio$)|(mobile$)')
            extra2 = re.compile(r'(technologies$)|(technology$)|(tech$)|(solutions$)|(solution$)|(com$)|(llc$)|(inc$)|(ltd$)|(apps$)|(app$)|(org$)|(gmbh$)')
            res1 = re.sub(extra1, '', result_text)
            res2 = re.sub(extra2, '', res1)
        else:
            res2 = np.nan
        return res2

    def check_whether_var_is_time_invariant(self, var, consecutive=False, format_text=False): # note the var cannot be appId
        col_list, panel_list = self.select_the_var(var=var, consecutive=consecutive)
        df_list = []
        for j in range(len(col_list)):
            new_df = self.keep_cols(list_of_col_names=col_list[j]).to_frame()
            new_df.rename(columns={col_list[j]: var}, inplace=True)
            if format_text is True:
                new_df[var]=new_df[var].apply(lambda x: self.format_text_for_developer(x))
            df_list.append(new_df)
        diff_dfs = []
        for i in range(len(df_list)-1):
            diff_df = self.dataframe_difference(df_list[i], df_list[i+1], var=var)
            if len(diff_df.index) != 0:
                print(var, 'is NOT time invariant due to conflicts between', col_list[i], 'and', col_list[i+1])
                diff_dfs.extend([diff_df])
            else:
                print(var, 'is time invariant variable for rows are exactly same between', col_list[i], 'and', col_list[i+1])
        return df_list, diff_dfs

    def dataframe_difference(self, df1, df2, var): # ALL YOU need is left only, because you are compare df1 to df2, df2 to df3...
        """Find rows which are different between two DataFrames.
        https://hackersandslackers.com/compare-rows-pandas-dataframes/"""
        comparison_df = df1.reset_index().merge(
            df2,
            on = var,
            indicator = True,
            how = 'left'
        ).set_index('index')
        diff_df = comparison_df[comparison_df['_merge'] == 'left_only']
        return diff_df

    def find_difference_in_var_among_panels(self, var, consecutive=False):
        pass

    def drop_col_row_and_replace_cols(self, list_of_row_labels, list_of_col_names, new_cols):
        new_df = self.drop_both_cols_and_rows(list_of_row_labels=list_of_row_labels, list_of_col_names=list_of_col_names)
        col_names = new_cols.columns.tolist()
        new_df2 = new_df.drop(columns=col_names)
        new_df2 = new_df2.join(new_cols, how='inner')
        return new_df2

    def replace_cols(self, new_cols):
        col_names = new_cols.columns.tolist()
        new_df = self.drop_cols(list_of_col_names=col_names)
        new_df = new_df.join(new_cols, how='inner')
        return new_df

    def drop_both_cols_and_rows(self, list_of_row_labels, list_of_col_names):
        new_df = self.df.drop(index=list_of_row_labels)
        new_df2 = new_df.drop(columns=list_of_col_names)
        return new_df2

    def drop_rows(self, list_of_row_labels):
        new_df = self.df.drop(index=list_of_row_labels)
        return new_df

    def drop_cols(self, list_of_col_names):
        new_df = self.df.drop(columns=list_of_col_names)
        return new_df

    def convert_from_appid_to_developerid_index(self,
        developer_level_vars,
        consecutive=False,
        select_one_panel=None):
        if consecutive is True:
            col_list = []
            for var in developer_level_vars:
                col, panels = self.select_the_var(var, consecutive_panels=True)
                col_list.extend(col)
        if consecutive is False:
            col_list = []
            for var in developer_level_vars:
                col, panels = self.select_the_var(var)
                col_list.extend(col)
        elif select_one_panel is not None:
            col_list = []
            for var in developer_level_vars:
                col, panels = self.select_the_var(var=var, select_one_panel=select_one_panel)
                col_list.append(col)
        new_df = self.keep_cols(list_of_col_names=col_list)
        new_df.reset_index(drop=True, inplace=True)
        for i in new_df.columns:
            if 'developer' in i:
                new_df.set_index(i, inplace=True)
                break
        return new_df

    def convert_from_developerid_to_appid_index(self): # add new developer level variables and then transform it back to appid level
        pass

    def convert_from_appid_to_multiindex_df(self):
        pass
    # ------------------------------------------------------------------------------

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

    # STRATEGY 1: ------------------------------------------------------------------
    ### VARS: minInstalls
    # if the missing panel(s) are in between none-missing ones, take the average of before and after panel values and fill in the missing
    # if the minInstalls are missing for consecutively three panels or more, delete that row (because this is an important variable).
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

    def impute_the_missing_panel_according_to_adjacent_panel(self, var): # the self.df here should be the newly passed df that has deleted all rows and cols that will not be imputed
        col, panels = self.select_the_var(var=var)
        df2 = self.keep_cols(list_of_col_names=col)
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

    def impute_none_with_value_from_adj_panel(self, var):
        pass


#################################################################################################################
class single_variable_stats(summary_statistics):
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
class two_variables_stats(summary_statistics):
    def __init__(self, v1, v1_type, v2, v2_type, **kwargs):
        super().__init__(**kwargs)
        self.v1 = v1
        self.v1_type = v1_type
        self.v2 = v2
        self.v2_type = v2_type

    def cross_tab(self):
        pass
