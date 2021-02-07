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
                 address_vars=None, misc_vars=None):
        self.df = merged_df
        self.initial_date = initial_date
        self.all_panels = all_panels
        self.numeric_vars = numeric_vars
        self.dummy_vars = dummy_vars
        self.text_vars = text_vars
        self.datetime_vars = datetime_vars
        self.misc_vars = misc_vars
        self.address_vars = address_vars

    def print_col_names(self):
        for col in self.df.columns:
            print(col)

    def print_num_rows(self):
        print(len(self.df.index))




#################################################################################################################
class impute_missing(summary_statistics):
    # if you do not want to override parent properties or add new properties, you do not need to initialize in child class
    # it automatically inherits everything from the parent class
    pass




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
