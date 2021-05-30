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
import geopy
from geopy.geocoders import AzureMaps
import time
from tqdm import tqdm
tqdm.pandas()
import matplotlib
import seaborn
import copy

"""
In this class, i will create time-invariant dummies indicating sub-smples, in the regression_analysis class, 
I will run both regression for sub samples and regression for the full sample with sub-sample dummies (and interactions).
The niche text label will be generated within each subsample. 
"""
class divide():
    panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_1_panels___')

    descriptive_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/tables')

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

    def open_imputed_and_deleted_missing_df(self, name):
        f_name = self.initial_panel + '_' + name + '.pickle'
        q = divide.panel_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)

    def create_star_developer_var(self):
        """
        https://www.forbes.com/top-digital-companies/list/#tab:rank
        :return:
        """
        self.df['developerTimeInvar_formatted'] = self.df['developerTimeInvar'].apply(lambda x: x.lower())
        self.df['top_digital_firms'] = 0
        for i in tqdm(range(len(divide.top_digital_firms_substring))):
            for j, row in self.df.iterrows():
                if divide.top_digital_firms_substring[i] in row['developerTimeInvar_formatted']:
                    self.df.at[j, 'top_digital_firms'] = 1
        for i in tqdm(range(len(divide.top_digital_firms_exactly_match))):
            for j, row in self.df.iterrows():
                if divide.top_digital_firms_exactly_match[i] == row['developerTimeInvar_formatted']:
                    self.df.at[j, 'top_digital_firms'] = 1
        self.df['non-top_digital_firms'] = 0
        self.df.at[self.df['top_digital_firms']==0, 'non-top_digital_firms'] = 1
        # ------------------ print check -----------------------------------------
        c = self.df.groupby(['top_digital_firms'], dropna=False).size()
        print(self.initial_panel, ' : top digital firms.')
        print(c)
        c = self.df.groupby(['non-top_digital_firms'], dropna=False).size()
        print(self.initial_panel, ' : non-top digital firms.')
        print(c)
        # ------------------ save bc this function takes too long ----------------
        filename = self.initial_panel + '_imputed_deleted_top_firm.pickle'
        q = divide.panel_path / filename
        pickle.dump(self.df, open(q, 'wb'))
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)

    def create_subsample_var_dict(self):
        """
        We are going to divide sub samples by minInstalls, genreIds and star companies
        :return: a dictionary with variables to examine in above mentioned 3 areas.
        """
        self.ssvard = {'minInstalls': ['ImputedminInstalls_' + i for i in self.all_panels],
                       'genreId': ['ImputedgenreId_' + i for i in self.all_panels],
                       'starDeveloper': ['top_digital_firms',
                                         'non-top_digital_firms']}
        return divide(initial_panel=self.initial_panel,
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
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)

    def create_division_rules(self):
        # use the most recent panel of imputedminInstalls as the bar for dividing sub samples
        self.df['ImputedminInstalls_tier1'] = 0
        self.df['ImputedminInstalls_tier1'].loc[self.df['ImputedminInstalls_'+self.all_panels[-1]] >= 1.000000e+07] = 1
        self.df['ImputedminInstalls_tier2'] = 0
        self.df['ImputedminInstalls_tier2'].loc[(self.df['ImputedminInstalls_' + self.all_panels[-1]] < 1.000000e+07) &
                                                (self.df['ImputedminInstalls_' + self.all_panels[-1]] >= 1.000000e+05)] = 1
        self.df['ImputedminInstalls_tier3'] = 0
        self.df['ImputedminInstalls_tier3'].loc[self.df['ImputedminInstalls_' + self.all_panels[-1]] < 1.000000e+05] = 1

        # use the mode of imputedGenreId as the bar for dividing sub samples
        genreIds = ['ImputedgenreId_' + j for j in self.all_panels]
        df2 = self.df.copy(deep=True)
        df2 = df2[genreIds]
        df2['ImputedgenreId_Mode'] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        app_categories = df2['ImputedgenreId_Mode'].unique().tolist()
        dummies_df = pd.get_dummies(df2['ImputedgenreId_Mode'])
        self.df = self.df.join(df2['ImputedgenreId_Mode'].to_frame(), how='inner')
        self.df = self.df.join(dummies_df, how='inner')
        self.division_rules = {'minInstalls': ['ImputedminInstalls_tier1',
                                               'ImputedminInstalls_tier2',
                                               'ImputedminInstalls_tier3'],
                               'genreId': app_categories,
                               'starDeveloper': ['top_digital_firms',
                                                 'non-top_digital_firms']}
        # -------------- save -----------------------------------------------------
        filename = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = divide.panel_path / filename
        pickle.dump(self.df, open(q, 'wb'))
        print(self.initial_panel, ' finished creating division rules for sub samples and saved dataframe. ')
        # -------------- save -----------------------------------------------------
        filename = self.initial_panel + '_subsample_division_rules.pickle'
        q = divide.panel_path / 'subsample_division_rule' / filename
        pickle.dump(self.division_rules, open(q, 'wb'))
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)

    def open_subsamples_df_and_division_rules(self):
        # ----------------------------------------------------------------
        f_name = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = divide.panel_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        # ----------------------------------------------------------------
        f_name = self.initial_panel + '_subsample_division_rules.pickle'
        q = divide.panel_path / 'subsample_division_rule' / f_name
        with open(q, 'rb') as f:
            self.division_rules = pickle.load(f)
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)

    def subsamples_count_pandas(self):
        def set_segments(x):
            x = x.replace('_', ' ')
            x = x.lower()
            if 'imputedmininstalls' in x:
                x = x.replace('imputedmininstalls ', '')
            elif 'top' in x:
                x = x.replace(' digital firms', '')
            else:
                x = x
            return x
        df_list = []
        for name, dummy_vars in self.division_rules.items():
            df = pd.DataFrame(columns=['Segments', 'Size'])
            df['Segments'] = dummy_vars
            for d_var in dummy_vars:
                c = self.df.groupby([d_var], dropna=False).size()
                df.at[df['Segments'] == d_var, 'Size'] = c[1]
            total = df.sum()
            total.name = 'Total'
            df = df.append(total.transpose())
            df.at['Total', 'Segments'] = 'Total'
            df['Segments'] = df['Segments'].apply(set_segments)
            df['Segments'] = df['Segments'].apply(lambda x: x.capitalize())
            df.set_index('Segments', inplace=True)
            # ---------- convert to latex -------------------------------------------------------
            f_name = self.initial_panel + '_' + name + '_subsamples_counts.tex'
            df.to_latex(buf=divide.descriptive_stats_tables / f_name,
                        multirow=True,
                        multicolumn=True,
                        longtable=True,
                        position='h!',
                        escape=False)
            df_list.append(df)
        self.subsamples_count_table = df_list
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df,
                      sub_sample_vars_dict=self.ssvard,
                      sub_sample_counts=self.sscounts,
                      division_rules=self.division_rules,
                      subsamples_count_table=self.subsamples_count_table)

