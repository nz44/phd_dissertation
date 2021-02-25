# Dec 23, 2020
# find the products in a niche market (apps belonging to a very narrowly defined category, which is figured out by
# analyzing descriptions and names and figure out if they serve the same function or purpose)
import warnings
warnings.filterwarnings('ignore')
import pickle
import re
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = 500
pd.options.display.max_colwidth = 500
pd.options.display.max_seq_items = 500
from collections import Counter
import random

# -----------------------------------------------------------------------------
# class Column_Name prepare the column names you would like to subset in order to conduct analysis
# -----------------------------------------------------------------------------

class Column_Names:

    def __init__(self, panels): # a list of panel strings that you'd like to subset
        self.panels = panels
        self.text_cols = ('title', 'description', 'summary', 'comments')
        self.price_cols = ('price', 'free_True', 'offersIAP_True', 'IAP_low', 'IAP_high', 'inAppProductPrice', 'adSupported_True',
                           'containsAds_True')
        self.consumer_feedback_cols = ('installs', 'minInstalls', 'score', 'score_5', 'score_4', 'score_3', 'score_2', 'score_1',
                                       'ratings', 'reviews', 'comments')
        self.developer_info_cols = ('developer', 'developerEmail', 'developerWebsite', 'developerAddress')
        self.app_info_cols = ('appId', 'version', 'contentRating', 'genre', 'genreId', 'GAME', 'UTILITIES', 'SOCIAL_MEDIA_LEISURE',
                              'check_all_genre_covered', 'size', 'released_datetime', 'privacyPolicy')
        self.cluster_cols = ('k-means', 'fuzzy-c-means') # time-invariant, do not attach panel at the end

    def get_text_cols_names(self):
        text_cols_names = []
        for i in self.panels:
            for j in self.text_cols:
                c = j + '_' + i
                text_cols_names.append(c)
        return text_cols_names

    def get_price_cols_names(self):
        price_cols_names = []
        for i in self.panels:
            for j in self.price_cols:
                c = j + '_' + i
                price_cols_names.append(c)
        return price_cols_names

    def get_consumer_feedback_cols_names(self):
        consumer_feedback_cols_names = []
        for i in self.panels:
            for j in self.consumer_feedback_cols:
                c = j + '_' + i
                consumer_feedback_cols_names.append(c)
        return consumer_feedback_cols_names

    def get_developer_info_cols_names(self):
        developer_info_cols_names = []
        for i in self.panels:
            for j in self.developer_info_cols:
                c = j + '_' + i
                developer_info_cols_names.append(c)
        return developer_info_cols_names

    def get_app_info_cols_names(self):
        app_info_cols_names = []
        for i in self.panels:
            for j in self.app_info_cols:
                c = j + '_' + i
                app_info_cols_names.append(c)
        return app_info_cols_names


# -----------------------------------------------------------------------------
# function merge_dataframe aim to merge the many dataframes containing different
# targeted variables (such as k-means, geo-labels), and subset the relevant cols from
# the main dataframe, which is opened by open_or_save_files's method open_merged_df.
# df_all_cols is obtained through Column_Names' get_xx_cols_names method.
# Since different panels may contain different cols, one need to check their existence first
# -----------------------------------------------------------------------------
def merge_dataframe(df_all, df_all_cols, df_t, df_t_cols):
    real_cols_all = []
    for i in df_all_cols:
        if i in df_all.columns:
            real_cols_all.append(i)
    df_a = df_all[real_cols_all]
    df_tt = df_t[df_t_cols]
    df = df_a.merge(df_tt, left_index=True, right_index=True)
    return df

# -----------------------------------------------------------------------------
# class dataframe_analysis look at possibly almost anything you'd like to know (by group)
# input dataframe is usually the output of merge_dataframe function
# -----------------------------------------------------------------------------
class Dataframe_Analysis:

    def __init__(self, df, **kwargs):
        self.df = df
        self.group1 = kwargs.get('group1')
        self.group2 = kwargs.get('group2')
        self.group3 = kwargs.get('group3')
        self.group4 = kwargs.get('group4')
        self.group5 = kwargs.get('group5')
        self.group6 = kwargs.get('group6')

    def cross_tab(self):
        if self.group1 is not None and self.group2 is not None:
            x = pd.crosstab(self.df[self.group1], self.df[self.group2], dropna=False,
                            margins=True, margins_name="Total")
            x = x.sort_values(by='Total', axis=1)
            return x

