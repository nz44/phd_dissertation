from pathlib import Path
import pickle
import pandas as pd
pd.set_option('display.max_colwidth', 100)
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
import time
from tqdm import tqdm
tqdm.pandas()
import matplotlib
import seaborn
import copy

"""
2021 July 19 
The new version will also divide into 5 categories (the same divide rule as in essay 2 and essay 3), while keeping the original 49 categories.

"""
class divide_essay_1():
    panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_1_panels___')

    descriptive_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_1___/descriptive_stats/tables')

    def __init__(self,
                 initial_panel,
                 all_panels,
                 df=None,
                 sub_sample_vars_dict=None,
                 sub_sample_counts=None,
                 division_rules=None,
                 subsamples_count_table=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.ssvard = sub_sample_vars_dict
        self.sscounts = sub_sample_counts
        self.division_rules = division_rules
        self.subsamples_count_table = subsamples_count_table

    def open_imputed_and_deleted_missing_df(self):
        f_name = self.initial_panel + '_imputed_and_deleted_missing.pickle'
        q = divide_essay_1.panel_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return divide_essay_1(initial_panel=self.initial_panel,
                                all_panels=self.all_panels,
                                df=self.df,
                                sub_sample_vars_dict=self.ssvard,
                                sub_sample_counts=self.sscounts,
                                division_rules=self.division_rules,
                                subsamples_count_table=self.subsamples_count_table)

    def open_imputed_and_deleted_top_firm_df(self):
        f_name = self.initial_panel + '_imputed_deleted_top_firm.pickle'
        q = divide_essay_1.panel_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return divide_essay_1(initial_panel=self.initial_panel,
                                all_panels=self.all_panels,
                                df=self.df,
                                sub_sample_vars_dict=self.ssvard,
                                sub_sample_counts=self.sscounts,
                                division_rules=self.division_rules,
                                subsamples_count_table=self.subsamples_count_table)

    def count_apps_in_each_category(self):
        """
        The aim of this function is to see which group has too few apps, and that may not be a suitable choice of a sub-sample
        (degree of freedom issue when running regression)
        """
        print(self.initial_panel, ' count number of apps in each group (potential sub-samples) ')
        df2 = self.df.copy(deep=True)
        self.sscounts = {'minInstalls': [],
                         'genreId': [],
                         'starDeveloper': []}
        for category_name, cat_vars in self.ssvard.items():
            for i in cat_vars:
                s1 = df2.groupby([i], dropna=False).size().to_frame()
                s1.columns = [i]
                self.sscounts[category_name].append(s1)
            concated = functools.reduce(lambda a, b: a.join(b, how='inner'), self.sscounts[category_name])
            concated.sort_index(ascending=False, inplace=True)
            self.sscounts[category_name] = concated
        return divide_essay_1(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)
    # this part has been moved to STEP2_pre_processing
    # def create_star_developer_var(self):
    #     """
    #     https://www.forbes.com/top-digital-companies/list/#tab:rank
    #     :return:
    #     """
    #     self.df['developerTimeInvar_formatted'] = self.df['developerTimeInvar'].apply(lambda x: x.lower())
    #     self.df['top_digital_firms'] = 0
    #     for i in tqdm(range(len(divide_essay_1.top_digital_firms_substring))):
    #         for j, row in self.df.iterrows():
    #             if divide_essay_1.top_digital_firms_substring[i] in row['developerTimeInvar_formatted']:
    #                 self.df.at[j, 'top_digital_firms'] = 1
    #     for i in tqdm(range(len(divide_essay_1.top_digital_firms_exactly_match))):
    #         for j, row in self.df.iterrows():
    #             if divide_essay_1.top_digital_firms_exactly_match[i] == row['developerTimeInvar_formatted']:
    #                 self.df.at[j, 'top_digital_firms'] = 1
    #     self.df['non-top_digital_firms'] = 0
    #     self.df.at[self.df['top_digital_firms']==0, 'non-top_digital_firms'] = 1
    #     # ------------------ print check -----------------------------------------
    #     c = self.df.groupby(['top_digital_firms'], dropna=False).size()
    #     print(self.initial_panel, ' : top digital firms.')
    #     print(c)
    #     c = self.df.groupby(['non-top_digital_firms'], dropna=False).size()
    #     print(self.initial_panel, ' : non-top digital firms.')
    #     print(c)
    #     # ------------------ save bc this function takes too long ----------------
    #     filename = self.initial_panel + '_imputed_deleted_top_firm.pickle'
    #     q = divide_essay_1.panel_path / filename
    #     pickle.dump(self.df, open(q, 'wb'))
    #     return divide_essay_1(initial_panel=self.initial_panel,
    #                   all_panels=self.all_panels,
    #                   df=self.df,
    #                   sub_sample_vars_dict=self.ssvard,
    #                   sub_sample_counts=self.sscounts,
    #                   division_rules=self.division_rules,
    #                   subsamples_count_table=self.subsamples_count_table)

    def _create_minInstalls_subsamples(self, df):
        """
        :param df: is the output of self.open_imputed_and_deleted_top_firm_df()
        :return:
        """
        df2 = df.copy(deep=True)
        df2['Tier1'] = 0
        df2['Tier1'].loc[df2['ImputedminInstalls_'+self.all_panels[-1]] >= 1.000000e+07] = 1
        df2['Tier2'] = 0
        df2['Tier2'].loc[(df2['ImputedminInstalls_' + self.all_panels[-1]] < 1.000000e+07) &
                            (df2['ImputedminInstalls_' + self.all_panels[-1]] >= 1.000000e+05)] = 1
        df2['Tier3'] = 0
        df2['Tier3'].loc[df2['ImputedminInstalls_' + self.all_panels[-1]] < 1.000000e+05] = 1
        return df2

    def _create_categories_subsamples(self, df):
        """
        :param df2: is the output of self.open_imputed_and_deleted_top_firm_df()
        :return:
        """
        df2 = df.copy(deep=True)
        genreIds = ['ImputedgenreId_' + j for j in self.all_panels]
        df2 = df2[genreIds]
        df2['ImputedgenreId_Mode'] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        dummies_df = pd.get_dummies(df2['ImputedgenreId_Mode'])
        df = df.join(df2['ImputedgenreId_Mode'].to_frame(), how='inner')
        df = df.join(dummies_df, how='inner')
        # ------------- now create 5 categories just like in essay 2 and 3 ----------------------
        c = df.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose GAME (all games)
        df['category_GAME'] = None
        df['category_GAME'] = df['ImputedgenreId_Mode'].apply(lambda x: 1 if 'GAME' in x else 0)
        c = df.groupby(['category_GAME'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # -------- Check category count excluding games -----------------------------
        df2 = df.copy(deep=True)
        df3 = df2.loc[df2['category_GAME'] == 0]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose BUSINESS (work related, things that improve people's productivity)
        df['category_BUSINESS'] = None
        df['category_BUSINESS'] = df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['FINANCE',
                                 'EDUCATION',
                                 'NEWS_AND_MAGAZINES',
                                 'BUSINESS',
                                 'PRODUCTIVITY',
                                 'TOOLS',
                                 'BOOKS_AND_REFERENCE',
                                 'LIBRARIES_AND_DEMO'] else 0)
        c = df.groupby(['category_BUSINESS'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # -------- Check category count excluding games and business -----------------
        df2 = df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME'] == 0) & (df2['category_BUSINESS'] == 0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose SOCIAL (dating, shopping, eating and drinking)
        # put map and transportation here because one would need them to social
        df['category_SOCIAL'] = None
        df['category_SOCIAL'] = df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['COMMUNICATION',
                                 'FOOD_AND_DRINK',
                                 'SOCIAL',
                                 'SHOPPING',
                                 'DATING',
                                 'EVENTS',
                                 'WEATHER',
                                 'MAPS_AND_NAVIGATION',
                                 'AUTO_AND_VEHICLES'] else 0)
        c = df.groupby(['category_SOCIAL'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # -------- Check category count excluding games and business and social ------
        df2 = df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME'] == 0) & (df2['category_BUSINESS'] == 0) & (df2['category_SOCIAL'] == 0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose LIFESTYLE (all the leisure family activity that does not have a strong social aspect)
        df['category_LIFESTYLE'] = None
        df['category_LIFESTYLE'] = df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['PERSONALIZATION',
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
                                 'HOUSE_AND_HOME'] else 0)
        c = df.groupby(['category_LIFESTYLE'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # ---- Check category count excluding games and business and social and lifestyle ------
        df2 = df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME'] == 0) & (
                df2['category_BUSINESS'] == 0) & (
                              df2['category_SOCIAL'] == 0) & (
                              df2['category_LIFESTYLE'] == 0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose MEDICAL (this is a extremely small group, but seems more important with covid-19 shock)
        df['category_MEDICAL'] = None
        df['category_MEDICAL'] = df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['HEALTH_AND_FITNESS',
                                 'MEDICAL'] else 0)
        c = df.groupby(['category_MEDICAL'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # ---- Check category count excluding games and business and social and lifestyle and medical ------
        df2 = df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME'] == 0) & (
                       df2['category_BUSINESS'] == 0) & (
                       df2['category_SOCIAL'] == 0) & (
                       df2['category_LIFESTYLE'] == 0) & (
                       df2['category_MEDICAL'] == 0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        return df

    def create_and_save_df_with_subsample_dummies(self):
        df1 = self.open_imputed_and_deleted_top_firm_df()
        df2 = self._create_minInstalls_subsamples(df=df1.df)
        df3 = self._create_categories_subsamples(df=df2)
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        self.ssvard = {'minInstalls': ['Tier1', 'Tier2', 'Tier3'],
                       'genreId': app_categories,
                       'categories': ['category_GAME', 'category_BUSINESS', 'category_SOCIAL', 'category_LIFESTYLE', 'category_MEDICAL'],
                       'starDeveloper': ['top_digital_firms', 'non-top_digital_firms']}
        # -------------- save -----------------------------------------------------
        filename = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = divide_essay_1.panel_path / filename
        pickle.dump(df3, open(q, 'wb'))
        print(self.initial_panel, ' finished creating division rules for sub samples and saved dataframe. ')
        return divide_essay_1(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=df3,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)


