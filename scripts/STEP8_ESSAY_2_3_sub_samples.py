import pandas as pd
import copy
from pathlib import Path
import pickle
pd.set_option('display.max_colwidth', 100)
pd.options.display.max_rows = 999
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
from sklearn import preprocessing
import statsmodels.api as sm
# https://www.statsmodels.org/stable/api.html
from linearmodels import PooledOLS
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import compare
from datetime import datetime
import functools
import re
today = datetime.today()
yearmonth = today.strftime("%Y%m")

class divide_essay_2_3():
    """
    Based on result of STEP3_sub_samples.py
    """
    panel_path_essay_1 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_1_panels___')

    panel_path_essay_2 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_panels___')

    panel_path_eesay_3 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_3_panels___')

    panel_path_essay_2_3_common = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_3_common_panels___'
    )

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

    def open_imputed_and_deleted_missing_df(self):
        f_name = self.initial_panel + '_imputed_and_deleted_missing.pickle'
        # use panel 1 path because the previous step, STEP2_pre_processing.py,
        # saved imputed and deleted missing to panel 1 path
        q = divide_essay_2_3.panel_path_essay_1 / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return divide_essay_2_3(initial_panel=self.initial_panel,
                                  all_panels=self.all_panels,
                                  df=self.df,
                                  sub_sample_vars_dict=self.ssvard,
                                  sub_sample_counts=self.sscounts,
                                  division_rules=self.division_rules,
                                  subsamples_count_table=self.subsamples_count_table)

    def open_imputed_deleted_top_firm_df(self):
        f_name = self.initial_panel + '_imputed_deleted_top_firm.pickle'
        # use panel 1 path because the previous step, STEP2_pre_processing.py,
        # saved imputed and deleted missing to panel 1 path
        q = divide_essay_2_3.panel_path_essay_2_3_common / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return divide_essay_2_3(initial_panel=self.initial_panel,
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
        for i in tqdm(range(len(divide_essay_2_3.top_digital_firms_substring))):
            for j, row in self.df.iterrows():
                if divide_essay_2_3.top_digital_firms_substring[i] in row['developerTimeInvar_formatted']:
                    self.df.at[j, 'top_digital_firms'] = 1
        for i in tqdm(range(len(divide_essay_2_3.top_digital_firms_exactly_match))):
            for j, row in self.df.iterrows():
                if divide_essay_2_3.top_digital_firms_exactly_match[i] == row['developerTimeInvar_formatted']:
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
        q = divide_essay_2_3.panel_path_essay_2_3_common / filename
        pickle.dump(self.df, open(q, 'wb'))
        return divide_essay_2_3(initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df=self.df,
                              sub_sample_vars_dict=self.ssvard,
                              sub_sample_counts=self.sscounts,
                              division_rules=self.division_rules,
                              subsamples_count_table=self.subsamples_count_table)

    def create_tier1_var(self):
        # use the most recent panel of imputedminInstalls as the bar for dividing sub samples
        self.df['Tier1'] = 0
        self.df['Tier1'].loc[
            self.df['ImputedminInstalls_' + self.all_panels[-1]] >= 1.000000e+07] = 1
        return divide_essay_2_3(initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df=self.df,
                              sub_sample_vars_dict=self.ssvard,
                              sub_sample_counts=self.sscounts,
                              division_rules=self.division_rules,
                              subsamples_count_table=self.subsamples_count_table)

    def create_leader_subsamples(self):
        """
        Based on essay 1 results, we found that niche dummy is not statistically significant for all four pricing variables in Tier1,
        and Niche dummy is not statistically significant for all but containsAds in Top (but the sign is negative for containsAds, which is
        opposite to sign in Tier2 and Tier3).
        So I think since Niche does not affect those 2 categories' pricing strategies, I will analyze them together as market leaders.
        """
        # while creating this variable, I will also check the overlap between Tier1 and Top
        df_tier1 = self.df.loc[self.df['Tier1'] == 1]
        print('Tier 1 number of apps : ', df_tier1.shape[0])
        df_top = self.df.loc[self.df['top_digital_firms'] == 1]
        print('Top firms number of apps : ', df_top.shape[0])
        df3 = self.df.loc[(self.df['Tier1'] == 1) & (self.df['top_digital_firms'] == 1)]
        print('Tier 1 and Top firms (overlap) : ', df3.shape[0])
        self.df['Leaders'] = 0
        self.df.at[self.df['Tier1'] == 1, 'Leaders'] = 1
        self.df.at[self.df['top_digital_firms'] == 1, 'Leaders'] = 1
        self.df['Non-leaders'] = 0
        self.df.at[self.df['Leaders'] == 0, 'Non-leaders'] = 1
        # ------------------ print check -----------------------------------------
        c = self.df.groupby(['Leaders'], dropna=False).size()
        print(self.initial_panel, ' : market leaders apps')
        print(c)
        c = self.df.groupby(['Non-leaders'], dropna=False).size()
        print(self.initial_panel, ' : non-market leaders apps')
        print(c)
        print()
        return divide_essay_2_3(initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df=self.df,
                              sub_sample_vars_dict=self.ssvard,
                              sub_sample_counts=self.sscounts,
                              division_rules=self.division_rules,
                              subsamples_count_table=self.subsamples_count_table)

    def create_category_subsamples(self):
        """
        within leader and non-leader subsamples, we are going to divide the apps into 5-6 categories based on functions:
        business, productivity, games
        """
        # use the mode of imputedGenreId as the bar for dividing sub samples
        genreIds = ['ImputedgenreId_' + j for j in self.all_panels]
        df2 = self.df.copy(deep=True)
        df2 = df2[genreIds]
        df2['ImputedgenreId_Mode'] = df2.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        app_categories = df2['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        self.df = self.df.join(df2['ImputedgenreId_Mode'], how='inner')
        c = self.df.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose GAME (all games)
        self.df['category_GAME'] = None
        self.df['category_GAME'] = self.df['ImputedgenreId_Mode'].apply(lambda x: 1 if 'GAME' in x else 0)
        c = self.df.groupby(['category_GAME'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # -------- Check category count excluding games -----------------------------
        df2 = self.df.copy(deep=True)
        df3 = df2.loc[df2['category_GAME']==0]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose BUSINESS (work related, things that improve people's productivity)
        self.df['category_BUSINESS'] = None
        self.df['category_BUSINESS'] = self.df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['FINANCE',
                                 'EDUCATION',
                                 'NEWS_AND_MAGAZINES',
                                 'BUSINESS',
                                 'PRODUCTIVITY',
                                 'TOOLS',
                                 'BOOKS_AND_REFERENCE',
                                 'LIBRARIES_AND_DEMO'] else 0)
        c = self.df.groupby(['category_BUSINESS'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # -------- Check category count excluding games and business -----------------
        df2 = self.df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME']==0) & (df2['category_BUSINESS']==0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose SOCIAL (dating, shopping, eating and drinking)
        # put map and transportation here because one would need them to social
        self.df['category_SOCIAL'] = None
        self.df['category_SOCIAL'] = self.df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['COMMUNICATION',
                                 'FOOD_AND_DRINK',
                                 'SOCIAL',
                                 'SHOPPING',
                                 'DATING',
                                 'EVENTS',
                                 'WEATHER',
                                 'MAPS_AND_NAVIGATION',
                                 'AUTO_AND_VEHICLES'] else 0)
        c = self.df.groupby(['category_SOCIAL'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # -------- Check category count excluding games and business and social ------
        df2 = self.df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME']==0) & (df2['category_BUSINESS']==0) & (df2['category_SOCIAL']==0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose LIFESTYLE (all the leisure family activity that does not have a strong social aspect)
        self.df['category_LIFESTYLE'] = None
        self.df['category_LIFESTYLE'] = self.df['ImputedgenreId_Mode'].apply(
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
        c = self.df.groupby(['category_LIFESTYLE'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # ---- Check category count excluding games and business and social and lifestyle ------
        df2 = self.df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME']==0) & (
                       df2['category_BUSINESS']==0) & (
                       df2['category_SOCIAL']==0)& (
                       df2['category_LIFESTYLE']==0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # --------------------------------------------------------------------------
        # propose MEDICAL (this is a extremely small group, but seems more important with covid-19 shock)
        self.df['category_MEDICAL'] = None
        self.df['category_MEDICAL'] = self.df['ImputedgenreId_Mode'].apply(
            lambda x: 1 if x in ['HEALTH_AND_FITNESS',
                                 'MEDICAL'] else 0)
        c = self.df.groupby(['category_MEDICAL'], dropna=False).size()
        print(c, ' total ', c.sum())
        print()
        # ---- Check category count excluding games and business and social and lifestyle and medical ------
        df2 = self.df.copy(deep=True)
        df3 = df2.loc[(df2['category_GAME']==0) & (
                       df2['category_BUSINESS']==0) & (
                       df2['category_SOCIAL']==0)& (
                       df2['category_LIFESTYLE']==0)& (
                       df2['category_MEDICAL']==0)]
        app_categories = df3['ImputedgenreId_Mode'].unique().tolist()
        print(len(app_categories), app_categories)
        c = df3.groupby(['ImputedgenreId_Mode'], dropna=False).size().sort_values(ascending=False)
        print(c, ' total ', c.sum())
        print()
        # ------------------- SAVE --------------------------------------------------------------------------
        filename = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = divide_essay_2_3.panel_path_essay_2_3_common / filename
        pickle.dump(self.df, open(q, 'wb'))
        return divide_essay_2_3(initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df=self.df,
                              sub_sample_vars_dict=self.ssvard,
                              sub_sample_counts=self.sscounts,
                              division_rules=self.division_rules,
                              subsamples_count_table=self.subsamples_count_table)
