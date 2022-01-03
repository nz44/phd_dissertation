import pandas as pd
# import copy
from pathlib import Path
import pickle
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
pd.options.mode.chained_assignment = None
import numpy as np
import math
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
from scipy.stats import boxcox
import statsmodels.api as sm
# https://www.statsmodels.org/stable/api.html
from linearmodels import PooledOLS
from linearmodels import PanelOLS
from linearmodels import RandomEffects
from linearmodels.panel import compare
from datetime import datetime
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")


class essay_23_stats_and_regs_201907():
    """Aug 10, 2021
    The main change in this version is that I split the graph of leaders and non-leaders because they belong to essay 2 and essay 3
    respectively, and they will be presented separately in my dissertation.
    """
    initial_panel = '201907'
    all_panels = ['201907',
                  '201908',
                  '201909',
                  '201912',
                  '202001',
                  '202003',
                  '202004',
                  '202009',
                  '202010',
                  '202011',
                  '202012',
                  '202101',
                  '202102',
                  '202103',
                  '202104',
                  '202105',
                  '202106']

    panel_root = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__')
    des_stats_root = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY')
    common_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_3_common___')

    name1_path_keywords = {'Non-leaders': '___essay_2___',
                           'Leaders': '___essay_3___'}

    graph_name1_titles = {
        'Leaders': 'Market Leaders and 5 Main Functional App Categories',
        'Non-leaders': 'Market Followers and 5 Main Functional App Categories'
    }

    name12_graph_title_dict = {'Leaders_full': 'Market Leaders Full Sample',
                                  'Leaders_category_GAME': 'Market Leaders Gaming Apps',
                                  'Leaders_category_BUSINESS': 'Market Leaders Business Apps',
                                  'Leaders_category_SOCIAL': 'Market Leaders Social Apps',
                                  'Leaders_category_LIFESTYLE': 'Market Leaders Lifestyle Apps',
                                  'Leaders_category_MEDICAL': 'Market Leaders Medical Apps',
                                  'Non-leaders_full': 'Market Followers Full Sample',
                                  'Non-leaders_category_GAME': 'Market Followers Gaming Apps',
                                  'Non-leaders_category_BUSINESS': 'Market Followers Business Apps',
                                  'Non-leaders_category_SOCIAL': 'Market Followers Social Apps',
                                  'Non-leaders_category_LIFESTYLE': 'Market Followers Lifestyle Apps',
                                  'Non-leaders_category_MEDICAL': 'Market Followers Medical Apps'}

    name12_reg_table_names = {'Leaders_full': 'Leaders \nFull',
                                  'Leaders_category_GAME': 'Leaders \nGaming Apps',
                                  'Leaders_category_BUSINESS': 'Leaders \nBusiness Apps',
                                  'Leaders_category_SOCIAL': 'Leaders \nSocial Apps',
                                  'Leaders_category_LIFESTYLE': 'Leaders \nLifestyle Apps',
                                  'Leaders_category_MEDICAL': 'Leaders \nMedical Apps',
                                  'Non-leaders_full': 'Followers \nFull',
                                  'Non-leaders_category_GAME': 'Followers \nGaming Apps',
                                  'Non-leaders_category_BUSINESS': 'Followers \nBusiness Apps',
                                  'Non-leaders_category_SOCIAL': 'Followers \nSocial Apps',
                                  'Non-leaders_category_LIFESTYLE': 'Followers \nLifestyle Apps',
                                  'Non-leaders_category_MEDICAL': 'Followers \nMedical Apps'}
    graph_dep_vars_ylabels = {
        'Imputedprice': 'Price',
        'LogImputedprice': 'Log Price',
        'LogWNImputedprice': 'Log Price Adjusted \nWith White Noise',
        'ImputedminInstalls': 'Minimum Installs',
        'LogImputedminInstalls': 'Log Minimum Installs',
        'both_IAP_and_ADS': 'Percentage Points',
        'TRUE%_offersIAPTrue': 'Percentage of Apps Offers IAP',
        'TRUE%_containsAdsTrue': 'Percentage of Apps Contains Ads',
        'offersIAPTrue': 'Percentage of Apps Offers IAP',
        'containsAdsTrue': 'Percentage of Apps Contains Ads'
    }
    graph_dep_vars_titles = {
        'Imputedprice': 'Price',
        'LogImputedprice': 'Log Price',
        'LogWNImputedprice': 'Log Price Adjusted With White Noise',
        'ImputedminInstalls': 'Minimum Installs',
        'LogImputedminInstalls': 'Log Minimum Installs',
        'both_IAP_and_ADS': 'Percentage of Apps that Offers IAP and Contains Ads',
        'TRUE%_offersIAPTrue': 'Percentage of Apps Offers IAP',
        'TRUE%_containsAdsTrue': 'Percentage of Apps Contains Ads',
        'offersIAPTrue': 'Percentage of Apps Offers IAP',
        'containsAdsTrue': 'Percentage of Apps Contains Ads'
    }
    dep_vars_reg_table_names = {
        'Imputedprice' : 'Price',
        'LogImputedprice': 'Log Price',
        'LogWNImputedprice': 'Log Price Adjusted \nWith White Noise',
        'ImputedminInstalls': 'Minimum Installs',
        'LogImputedminInstalls': 'Log Minimum Installs',
        'containsAdsTrue': 'Contains Ads',
        'offersIAPTrue': 'Offers IAP'
    }
    text_cluster_size_bins = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1500]
    text_cluster_size_labels = ['[0, 1]', '(1, 2]', '(2, 3]', '(3, 5]',
                                '(5, 10]', '(10, 20]', '(20, 30]', '(30, 50]',
                                '(50, 100]', '(100, 200]', '(200, 500]', '(500, 1500]']
    all_y_reg_vars = ['LogWNImputedprice',
                      'LogImputedminInstalls',
                      'offersIAPTrue',
                      'containsAdsTrue']

    @property
    def ssnames(self):
        d = self._open_predicted_labels_dict()
        res = dict.fromkeys(d.keys())
        for name1, content1 in d.items():
            res[name1] = list(content1.keys())
        return res

    @property
    def graph_name1_ssnames(self):
        res = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            l = []
            for name2 in content1:
                l.append(name1 + '_' + name2)
            res[name1] = l
        return res

    @classmethod
    def _select_vars(cls, df,
                     time_variant_vars_list=None,
                     time_invariant_vars_list=None):
        df2 = df.copy(deep=True)
        tv_var_list = []
        if time_variant_vars_list is not None:
            for i in time_variant_vars_list:
                vs = [i + '_' + j for j in cls.all_panels]
                tv_var_list = tv_var_list + vs
        ti_var_list = []
        if time_invariant_vars_list is not None:
            for i in time_invariant_vars_list:
                ti_var_list.append(i)
        total_vars = tv_var_list + ti_var_list
        df2 = df2[total_vars]
        return df2

    @classmethod
    def _open_imputed_deleted_divided_df(cls):
        f_name = cls.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = cls.common_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    @classmethod
    def _open_predicted_labels_dict(cls):
        f_name = cls.initial_panel + '_predicted_labels_dict.pickle'
        q = cls.common_path / 'predicted_text_labels' / f_name
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    @classmethod
    def _open_app_level_text_cluster_stats(cls):
        filename = cls.initial_panel + '_dict_app_level_text_cluster_stats.pickle'
        q = cls.common_path / 'app_level_text_cluster_stats' / filename
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    @classmethod
    def _set_title_and_save_graphs(cls, fig,
                                   file_keywords,
                                   relevant_folder_name,
                                   graph_title='',
                                   name1='',
                                   name2=''):
        """
        generic internal function to save graphs according to essay 2 (non-leaders) and essay 3 (leaders).
        name1 and name2 are the key names of essay_1_stats_and_regs_201907.ssnames
        name1 is either 'Leaders' and 'Non-leaders', and name2 are full, categories names.
        graph_title is what is the graph is.
        """
        # ------------ set title -------------------------------------------------------------------------
        if graph_title != '':
            if name1 != '' and name2 != '':
                title = cls.initial_panel + ' ' + cls.name12_graph_title_dict[
                    name1 + '_' + name2] + ' \n' + graph_title
            else:
                title = cls.initial_panel + ' ' + graph_title
            title = title.title()
            fig.suptitle(title, fontsize='medium')
        # ------------ save ------------------------------------------------------------------------------
        filename = cls.initial_panel + '_' + name1 + '_' + name2 + '_' + file_keywords + '.png'
        fig.savefig(cls.des_stats_root / cls.name1_path_keywords[name1] / 'descriptive_stats' / 'graphs' / relevant_folder_name / filename,
                    facecolor='white',
                    dpi=300)

    def __init__(self,
                 tcn,
                 combined_df=None,
                 broad_niche_cutoff=None,
                 broadDummy_labels=None,
                 reg_results=None):
        self.tcn = tcn
        self.cdf = combined_df
        self.broad_niche_cutoff = broad_niche_cutoff
        self.broadDummy_labels = broadDummy_labels
        self.reg_results = reg_results

    def open_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = self.common_path / 'cross_section_dfs' / filename
        with open(q, 'rb') as f:
            self.cdf = pickle.load(f)
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _numApps_per_cluster(self):
        d2 = self._open_predicted_labels_dict()
        d = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            d[name1] = dict.fromkeys(content1)
            for name2 in d[name1].keys():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                s2 = d2[name1][name2].groupby(
                    [label_col_name]).size(
                    ).sort_values(
                    ascending=False)
                d[name1][name2] = s2.rename('Apps Count').to_frame()
        return d

    def _numClusters_per_cluster_size_bin(self):
        d = self._numApps_per_cluster()
        res = dict.fromkeys(d.keys())
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                df2 = df.copy(deep=True)
                # since the min number of apps in a cluster is 1, not 0, so the smallest range (0, 1] is OK.
                # there is an option include_loweest == True, however, it will return float, but I want integer bins, so I will leave it
                # cannot set retbins == True because it will override the labels
                df3 = df2.groupby(pd.cut(x=df2.iloc[:, 0],
                                         bins=self.text_cluster_size_bins,
                                         include_lowest=True,
                                         labels=self.text_cluster_size_labels)
                                  ).count()
                df3.rename(columns={'Apps Count': 'Clusters Count'}, inplace=True)
                res[k1][k2] = df3
        return res

    def _numApps_per_cluster_size_bin(self):
        d1 = self._numApps_per_cluster()
        d3 = self._open_predicted_labels_dict()
        res = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            res[name1] = dict.fromkeys(content1)
            for name2 in content1:
                df = d3[name1][name2].copy(deep=True)
                # create a new column indicating the number of apps in the particular cluster for that app
                predicted_label_col = name1 + '_' + name2 + '_kmeans_labels'
                df['numApps_in_cluster'] = df[predicted_label_col].apply(
                    lambda x: d1[name1][name2].loc[x])
                # create a new column indicating the size bin the text cluster belongs to
                df['cluster_size_bin'] = pd.cut(
                                   x=df['numApps_in_cluster'],
                                   bins=self.text_cluster_size_bins,
                                   include_lowest=True,
                                   labels=self.text_cluster_size_labels)
                # create a new column indicating grouped sum of numApps_in_cluster for each cluster_size
                df2 = df.groupby('cluster_size_bin').count()
                df3 = df2.iloc[:, 0].to_frame()
                df3.columns = ['numApps_in_cluster_size_bin']
                res[name1][name2] = df3
        return res

    def determine_niche_broad_cutoff(self):
        d = self._numApps_per_cluster()
        self.broad_niche_cutoff = dict.fromkeys(self.ssnames.keys())
        self.broadDummy_labels = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            self.broad_niche_cutoff[name1] = dict.fromkeys(content1)
            self.broadDummy_labels[name1] = dict.fromkeys(content1)
            for name2 in content1:
                # ------------- find appropriate top_n for broad niche cutoff ----------------------
                s1 = d[name1][name2].to_numpy()
                s_multiples = np.array([])
                for i in range(len(s1) - 1):
                    multiple = s1[i] / s1[i + 1]
                    s_multiples = np.append(s_multiples, multiple)
                # top_n equals to the first n numbers that are 2
                top_n = 0
                if len(s_multiples) > 2:
                    for i in range(len(s_multiples) - 2):
                        if s_multiples[i] >= 2 and top_n == i:
                            top_n += 1
                        elif s_multiples[i + 1] >= 1.5 and top_n == 0:
                            top_n += 2
                        elif s_multiples[i + 2] >= 1.5 and top_n == 0:
                            top_n += 3
                        elif s_multiples[0] <= 1.1 and top_n == 0:
                            top_n += 2
                        else:
                            if top_n == 0:
                                top_n = 1
                else:
                    top_n = 1
                self.broad_niche_cutoff[name1][name2] = top_n
                self.broadDummy_labels[name1][name2] = d[name1][name2][:top_n].index.tolist()
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def text_cluster_stats_at_app_level(self):
        d1 = self._open_predicted_labels_dict()
        d2 = self._numApps_per_cluster()
        d3 = self._numClusters_per_cluster_size_bin()
        d4 = self._numApps_per_cluster_size_bin()
        res = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            res[name1] = dict.fromkeys(content1)
            for name2 in content1:
                df = d1[name1][name2].copy(deep=True)
                # set column names with name1 and name2 for future joining
                predicted_label = name1 + '_' + name2 + '_kmeans_labels'
                numApps_in_cluster = name1 + '_' + name2 + '_numApps_in_cluster'
                cluster_size_bin = name1 + '_' + name2 + '_cluster_size_bin'
                numClusters_in_cluster_size_bin = name1 + '_' + name2 + '_numClusters_in_cluster_size_bin'
                numApps_in_cluster_size_bin = name1 + '_' + name2 + '_numApps_in_cluster_size_bin'
                # create a new column indicating the number of apps in the particular cluster for that app
                # (do not forget to use .squeeze() here because .loc will return a pandas series)
                df[numApps_in_cluster] = df[predicted_label].apply(
                    lambda x: d2[name1][name2].loc[x].squeeze())
                # create a new column indicating the size bin the text cluster belongs to
                df[cluster_size_bin] = pd.cut(
                                   x=df[numApps_in_cluster],
                                   bins=self.text_cluster_size_bins,
                                   include_lowest=True,
                                   labels=self.text_cluster_size_labels)
                # create a new column indicating number of cluster for each cluster size bin
                df[numClusters_in_cluster_size_bin] = df[cluster_size_bin].apply(
                    lambda x: d3[name1][name2].loc[x].squeeze())
                # create a new column indicating grouped sum of numApps_in_cluster for each cluster_size
                df[numApps_in_cluster_size_bin] = df[cluster_size_bin].apply(
                    lambda x: d4[name1][name2].loc[x].squeeze())
                res[name1][name2] = df
        filename = self.initial_panel + '_dict_app_level_text_cluster_stats.pickle'
        q = self.common_path / 'app_level_text_cluster_stats' / filename
        pickle.dump(res, open(q, 'wb'))
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def combine_app_level_text_cluster_stats_with_df(self):
        df = self._open_imputed_deleted_divided_df()
        d = self._open_app_level_text_cluster_stats()
        x1 = d['Leaders']['full'].copy(deep=True)
        x2 = d['Non-leaders']['full'].copy(deep=True)
        x3 = x1.join(x2, how='outer')
        list_of_dfs = [x3]
        for name1, content1 in d.items():
            for name2, stats_df in content1.items():
                if name2 != 'full':
                    list_of_dfs.append(stats_df)
        combined_stats_df = functools.reduce(lambda a, b: a.join(b, how='left'), list_of_dfs)
        self.cdf = df.join(combined_stats_df, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def check_text_label_contents(self):
        df2 = self.cdf.copy(deep=True)
        d = self._open_predicted_labels_dict()
        for name1, content in d.items():
            for name2, text_label_col in content.items():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                unique_labels = df2[label_col_name].unique().tolist()
                unique_labels = [x for x in unique_labels if math.isnan(x) is False]
                print(name1, name2, ' -- unique text labels are --')
                print(unique_labels)
                print()
                for label_num in unique_labels:
                    df3 = df2.loc[df2[label_col_name]==label_num, [self.tcn + 'ModeClean']]
                    if len(df3.index) >= 10:
                        df3 = df3.sample(n=10)
                    f_name = self.initial_panel + '_' + name1 + '_' + name2 + '_' + 'TL_' + str(label_num) + '_' + self.tcn + '_sample.csv'
                    q = self.common_path / 'check_predicted_label_text_cols' / f_name
                    df3.to_csv(q)
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _text_cluster_group_count(self):
        df2 = self.cdf.copy(deep=True)
        d = dict.fromkeys(self.ssnames.keys())
        self.broad_niche_cutoff = dict.fromkeys(self.ssnames.keys())
        self.nicheDummy_labels = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            d[name1] = dict.fromkeys(content1)
            self.broad_niche_cutoff[name1] = dict.fromkeys(content1)
            self.nicheDummy_labels[name1] = dict.fromkeys(content1)
            for name2 in d[name1].keys():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                # ------------- find appropriate top_n for broad niche cutoff ----------------------
                s1 = df2.groupby([label_col_name]).size().sort_values(ascending=False).to_numpy()
                s_multiples = np.array([])
                for i in range(len(s1)-1):
                    multiple = s1[i]/s1[i+1]
                    s_multiples = np.append(s_multiples, multiple)
                # top_n equals to the first n numbers that are 2
                top_n = 0
                for i in range(len(s_multiples)-2):
                    if s_multiples[i] >= 2 and top_n == i:
                        top_n += 1
                    elif s_multiples[i+1] >= 1.5 and top_n == 0:
                        top_n += 2
                    elif s_multiples[i+2] >= 1.5 and top_n == 0:
                        top_n += 3
                    elif s_multiples[0] <= 1.1 and top_n == 0:
                        top_n += 2
                    else:
                        if top_n == 0:
                            top_n = 1
                self.broad_niche_cutoff[name1][name2] = top_n
                s2 = df2.groupby([label_col_name]).size().sort_values(ascending=False)
                s3 = s2.iloc[:self.broad_niche_cutoff[name1][name2], ]
                self.nicheDummy_labels[name1][name2] = s3.index.tolist()
                # ------------- convert to frame ---------------------------------------------------
                d[name1][name2] = df2.groupby([label_col_name]).size(
                    ).sort_values(ascending=False).rename(name1 + '_' + name2 + '_Apps_Count').to_frame()
        return d

    def _get_xy_var_list(self, name1, name2, y_var, the_panel=None):
        """
        :param name1: leaders non-leaders
        :param name2: all categories
        :param y_var: 'Imputedprice','ImputedminInstalls','offersIAPTrue','containsAdsTrue'
        :param log_y: for price and mininstalls, log = True
        :return:
        """
        time_invar_controls = ['size', 'DaysSinceReleased']
        x_var = [name1 + '_' + name2 + '_NicheDummy']
        if the_panel is None:
            time_var_controls = ['Imputedscore_' + i for i in self.all_panels] + \
                                ['Imputedreviews_' + i for i in self.all_panels]
            y_var = [y_var + '_' + i for i in self.all_panels]
        else:
            time_var_controls = ['Imputedscore_' + the_panel, 'Imputedreviews_' + the_panel]
            y_var = [y_var + '_' + the_panel]
        all_vars = y_var + x_var + time_invar_controls + time_var_controls
        return all_vars

    def _slice_xy_df_for_subsamples(self, y_var, the_panel=None, log_y=False):
        d = self._slice_subsamples_dict()
        res = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in d.items():
            res[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                var_list = self._get_xy_var_list(name1=name1, name2=name2, y_var=y_var, the_panel=the_panel)
                if log_y is False:
                    res[name1][name2] = df[var_list]
                else:
                    df2 = df[var_list]
                    if the_panel is None:
                        for i in self.all_panels:
                            df2['Log' + y_var + '_' + i] = np.log2(df2[y_var + '_' + i] + 1)
                            df2.drop([y_var + '_' + i], axis=1, inplace=True)
                    else:
                        df2['Log' + y_var + '_' + the_panel] = np.log2(df2[y_var + '_' + the_panel] + 1)
                        df2.drop([y_var + '_' + the_panel], axis=1, inplace=True)
                    res[name1][name2] = df2
        return res

    def _slice_subsamples_dict(self):
        """
        :param vars: a list of variables you want to subset
        :return:
        """
        df = self.cdf.copy(deep=True)
        d = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            d[name1] = dict.fromkeys(content1)
            df2 = df.loc[df[name1]==1]
            for name2 in content1:
                if name2 == 'full':
                    d[name1][name2] = df2
                else:
                    d[name1][name2] = df2.loc[df2[name2]==1]
        return d

    def _cross_section_reg_get_xy_var_list(self, name1, name2, y_var, the_panel):
        """
        :param y_var: 'LogWNImputedprice','LogImputedminInstalls','offersIAPTrue','containsAdsTrue'
        :return:
        """
        time_invar_controls = ['size', 'DaysSinceReleased', 'contentRatingAdult']
        x_var = [name1 + '_' + name2 + '_NicheDummy']
        time_var_controls = ['Imputedscore_' + the_panel,
                             'ZScoreImputedreviews_' + the_panel]
        y_var = [y_var + '_' + the_panel]
        all_vars = y_var + x_var + time_invar_controls + time_var_controls
        print(name1, name2, the_panel)
        print('cross section reg x and y variables are :')
        print(all_vars)
        return all_vars

    def _panel_reg_get_xy_var_list(self, name1, name2, y_var):
        time_invar_controls = ['size', 'DaysSinceReleased', 'contentRatingAdult']
        x_var = [name1 + '_' + name2 + '_NicheDummy']
        time_var_x_vars = [name1 + '_' + name2 + '_PostXNicheDummy_' + i for i in self.all_panels] + \
                          ['PostDummy_' + i for i in self.all_panels]
        time_var_controls = ['DeMeanedImputedscore_' + i for i in self.all_panels] + \
                            ['DeMeanedZScoreImputedreviews_' + i for i in self.all_panels]
        y_var = [y_var + '_' + i for i in self.all_panels]
        all_vars = y_var + x_var + time_var_x_vars + time_invar_controls + time_var_controls
        print(name1, name2)
        print('panel reg x and y variables are :')
        print(all_vars)
        return all_vars

    def _cross_section_regression(self, y_var, df, the_panel):
        """
        https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
        #https://www.statsmodels.org/stable/rlm.html
        https://stackoverflow.com/questions/30553838/getting-statsmodels-to-use-heteroskedasticity-corrected-standard-errors-in-coeff
        source code for HC0, HC1, HC2, and HC3, white and Mackinnon
        https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html
        https://timeseriesreasoning.com/contents/zero-inflated-poisson-regression-model/
        """
        # check the correlation among variables
        # dfcorr = df.corr(method='pearson').round(2)
        # print('The correlation table of the cross section regression dataframe is:')
        # print(dfcorr)
        # print()
        all_vars = df.columns.values.tolist()
        # y_var is a string without panel substring
        for i in all_vars:
            if y_var in i:
                all_vars.remove(i)
        independents_df = df[all_vars]
        X = sm.add_constant(independents_df)
        y = df[[y_var + '_' + the_panel]]
        num_dep_var_unique_values = y.nunique().squeeze()
        print(y_var, 'contains', str(num_dep_var_unique_values), 'unqiue values.')
        # I found for leaders medical category group that there is only zeros in y, so OLS does not apply
        # generally, price is pre-dominantly zeros, so use zero inflated regression instead
        if y_var == 'LogImputedprice':
            print(y_var, ' -- The dependant variable has no variation in it, skip this PANEL regression -- ')
            model = sm.ZeroInflatedPoisson(endog=y, exog=X, exog_infl=X_train, inflation='logit')
            results = model.fit()
        else:
            model = sm.OLS(y, X)
            results = model.fit(cov_type='HC3')
        return results

    def _panel_reg_pooled_ols(self,
                   y_var, df):
        """
        Internal function
        return a dictionary containing all different type of panel reg results
        I will not run fixed effects model here because they will drop time-invariant variables.
        In addition, I just wanted to check whether for the time variant variables, the demeaned time variant variables
        will have the same coefficient in POOLED OLS as the time variant variables in FE.
        """
        all_vars = df.columns.values.tolist()
        # y_var is a string without panel substring
        for i in all_vars:
            if y_var in i:
                all_vars.remove(i)
        independents_df = df[all_vars]
        X = sm.add_constant(independents_df)
        y = df[[y_var]]
        # check if there is any variability in Y variable
        # for example, leaders category Medical LogImputedprice has zeros in all its columns
        num_dep_var_unique_values = y.nunique().squeeze()
        if num_dep_var_unique_values == 1:
            print(y_var, ' -- The dependant variable has no variation in it, skip this PANEL regression -- ')
            return None
        else:
            # https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
            print('start Pooled_ols regression')
            model = PooledOLS(y, X)
            result = model.fit(cov_type='clustered', cluster_entity=True)
            return result

    def _reg_for_all_subsamples_for_single_y_var(self, reg_type, y_var):
        data = self._slice_subsamples_dict()
        if reg_type == 'cross_section_ols':
            reg_results = dict.fromkeys(self.all_panels)
            for i in self.all_panels:
                reg_results[i] = dict.fromkeys(self.ssnames.keys())
                for name1, content1 in self.ssnames.items():
                    reg_results[i][name1] = dict.fromkeys(content1)
                    for name2 in content1:
                        allvars = self._cross_section_reg_get_xy_var_list(
                                              name1=name1,
                                              name2=name2,
                                              y_var=y_var,
                                              the_panel=i)
                        df = data[name1][name2][allvars]
                        print(name1, name2, 'Cross Section Regression -- First Check Correlations')
                        reg_results[i][name1][name2] = self._cross_section_regression(
                                              y_var=y_var,
                                              df=df,
                                              the_panel=i)
            for i in self.all_panels:
                self._extract_and_save_reg_results(result=reg_results,
                                                   reg_type=reg_type,
                                                   y_var=y_var,
                                                   the_panel=i)
        elif reg_type == 'panel_pooled_ols':
            reg_results = dict.fromkeys(self.ssnames.keys())
            for name1, content1 in self.ssnames.items():
                reg_results[name1] = dict.fromkeys(content1)
                for name2 in content1:
                    allvars = self._panel_reg_get_xy_var_list(
                                              name1=name1,
                                              name2=name2,
                                              y_var=y_var)
                    # ---------- convert to long for panel regression --------------------
                    df = data[name1][name2][allvars]
                    stubnames = [name1 + '_' + name2 + '_PostXNicheDummy', 'PostDummy',
                                 y_var, 'DeMeanedImputedscore', 'DeMeanedZScoreImputedreviews']
                    df = df.reset_index()
                    ldf = pd.wide_to_long(
                        df,
                        stubnames=stubnames,
                        i=['index'],
                        j="panel",
                        sep='_').reset_index()
                    ldf["panel"] = pd.to_datetime(ldf["panel"], format='%Y%m')
                    ldf = ldf.sort_values(by=["index", "panel"]).set_index('index')
                    ldf = ldf.reset_index().set_index(['index', 'panel'])
                    reg_results[name1][name2] = self._panel_reg_pooled_ols(y_var=y_var, df=ldf)
            self._extract_and_save_reg_results(result=reg_results,
                                               reg_type=reg_type,
                                               y_var=y_var)
        else:
            reg_results = {}
        return reg_results

    def reg_for_all_subsamples_for_all_y_vars(self, reg_type):
        res = dict.fromkeys(self.all_y_reg_vars)
        for y in self.all_y_reg_vars:
            res[y] = self._reg_for_all_subsamples_for_single_y_var(reg_type=reg_type, y_var=y)
        self.reg_results = res
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _extract_and_save_reg_results(self, result, reg_type, y_var, the_panel=None):
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                # ---------- specify the rows to extract ---------------
                index_to_extract = {
                    'cross_section_ols': ['const', name1 + '_' + name2 + '_NicheDummy'],
                    'panel_pooled_ols': [
                                            'const',
                                             name1 + '_' + name2 + '_NicheDummy',
                                            'PostDummy',
                                             name1 + '_' + name2 + '_PostXNicheDummy']
                }
                # ---------- get the coefficients ----------------------
                if reg_type == 'cross_section_ols':
                    x = result[the_panel][name1][name2].params
                else:
                    x = result[name1][name2].params
                x = x.to_frame()
                x.columns = ['parameter']
                y = x.loc[index_to_extract[reg_type]]
                # ---------- get the pvalues ---------------------------
                if reg_type == 'cross_section_ols':
                    z1 = result[the_panel][name1][name2].pvalues
                else:
                    z1 = result[name1][name2].pvalues
                z1 = z1.to_frame()
                z1.columns = ['pvalue']
                z2 = z1.loc[index_to_extract[reg_type]]
                y2 = y.join(z2, how='inner')
                y2 = y2.round(3)
                if the_panel is None:
                    filename = y_var + '_' + name1 + '_' + name2 + '_' + reg_type + '.csv'
                else:
                    filename = y_var + '_' + name1 + '_' + name2 + '_' + reg_type + '_' + the_panel + '.csv'
                y2.to_csv(self.des_stats_root / self.name1_path_keywords[name1] / 'reg_results_tables' / filename)
                print(name1, name2, 'Reg results are saved in the reg_results_tables folder')

    def _create_cross_section_reg_results_df_for_parallel_trend_beta_graph(self, alpha):
        """
        possible input for reg_type are: 'cross_section_ols', uses self._cross_section_regression()
        alpha = 0.05 for 95% CI of coefficients
        """
        # all dependant variables in one dictionary
        res_results = dict.fromkeys(self.all_y_reg_vars)
        # all subsamples are hue in the same graph
        for y_var in self.all_y_reg_vars:
            res_results[y_var] = self.reg_results[y_var]
        #  since every reg result is one row in dataframe
        res_df = dict.fromkeys(self.all_y_reg_vars)
        for y_var, panels in res_results.items():
            # order in lists are persistent (unlike sets or dictionaries)
            panel_content = []
            sub_samples_content = []
            beta_nichedummy_content = []
            ci_lower = []
            ci_upper = []
            for panel, subsamples in panels.items():
                for name1, content1 in subsamples.items():
                    for name2, reg_result in content1.items():
                        panel_content.append(panel)
                        sub_samples_content.append(name1 + '_' + name2)
                        nichedummy = name1 + '_' + name2 + '_NicheDummy'
                        beta_nichedummy_content.append(reg_result.params[nichedummy])
                        ci_lower.append(reg_result.conf_int(alpha=alpha).loc[nichedummy, 0])
                        ci_upper.append(reg_result.conf_int(alpha=alpha).loc[nichedummy, 1])
            d = {'panel': panel_content,
                 'sub_samples': sub_samples_content,
                 'beta_nichedummy': beta_nichedummy_content,
                 'ci_lower': ci_lower,
                 'ci_upper': ci_upper}
            df = pd.DataFrame(data=d)
            # create error bars (positive distance away from beta) for easier ax.errorbar graphing
            df['lower_error'] = df['beta_nichedummy'] - df['ci_lower']
            df['upper_error'] = df['ci_upper'] - df['beta_nichedummy']
            # sort by panels
            df["panel"] = pd.to_datetime(df["panel"], format='%Y%m')
            df["panel"] = df["panel"].dt.strftime('%Y-%m')
            df = df.sort_values(by=["panel"])
            res_df[y_var] = df
        return res_df

    def _put_reg_results_into_pandas_for_single_y_var(self, reg_type, y_var, the_panel=None):
        """
        :param result: is the output of self._reg_for_all_subsamples(
            reg_type='panel_pooled_ols',
            y_var=any one of ['LogWNImputedprice', 'LogImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue'])
            the documentation of the PanelResult class (which result is)
        :return:
        """
        # ============= 1. extract results info and put them into dicts ==================
        params_pvalues_dict = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            params_pvalues_dict[name1] = dict.fromkeys(content1)
            for name2 in content1:
                # ---------- specify the rows to extract ---------------
                index_to_extract = {
                    'cross_section_ols': ['const', name1 + '_' + name2 + '_NicheDummy'],
                    'panel_pooled_ols': [
                        'const',
                        name1 + '_' + name2 + '_NicheDummy',
                        'PostDummy',
                        name1 + '_' + name2 + '_PostXNicheDummy']
                }
                # ---------- get the coefficients ----------------------
                if reg_type == 'cross_section_ols':
                    x = self.reg_results[y_var][the_panel][name1][name2].params
                else:
                    x = self.reg_results[y_var][name1][name2].params
                x = x.to_frame()
                x.columns = ['parameter']
                y = x.loc[index_to_extract[reg_type]]
                # ---------- get the pvalues ---------------------------
                if reg_type == 'cross_section_ols':
                    z1 = self.reg_results[y_var][the_panel][name1][name2].pvalues
                else:
                    z1 = self.reg_results[y_var][name1][name2].pvalues
                z1 = z1.to_frame()
                z1.columns = ['pvalue']
                z2 = z1.loc[index_to_extract[reg_type]]
                def _assign_asterisk(v):
                    if 0.05 < v <= 0.1:
                        return '*'
                    elif 0.01 < v <= 0.05:
                        return '**'
                    elif v <= 0.01:
                        return '***'
                    else:
                        return ''
                z2['asterisk'] = z2['pvalue'].apply(lambda x: _assign_asterisk(x))
                y2 = y.join(z2, how='inner')
                y2['parameter'] = y2['parameter'].round(3).astype(str)
                y2['parameter'] = y2['parameter'] + y2['asterisk']
                y2.rename(index={'const': 'Constant',
                                name1 + '_' + name2 + '_NicheDummy': 'Niche',
                                'PostDummy': 'Post',
                                name1 + '_' + name2 + '_PostXNicheDummy': 'PostNiche'},
                         inplace=True)
                y2 = y2.reset_index()
                y2.drop(columns=['pvalue', 'asterisk'], inplace=True)
                y2.insert(0, 'Samples', [name1 + '_' + name2] * len(y2.index))
                y2['Samples'] = y2['Samples'].apply(lambda x: self.name12_reg_table_names[x] if x in self.name12_reg_table_names.keys() else 'None')
                y2.rename(columns={'index': 'Independent Vars',
                                   'parameter': self.dep_vars_reg_table_names[y_var]},
                          inplace=True)
                params_pvalues_dict[name1][name2] = y2
        # ========= concatenate dataframes into a single dataframe for each name1 ==========
        res = dict.fromkeys(params_pvalues_dict.keys())
        for name1, content1 in params_pvalues_dict.items():
            df_list = []
            for name12, df in content1.items():
                df_list.append(df)
            adf = functools.reduce(lambda a, b: a.append(b), df_list)
            res[name1] = adf
        return res

    def put_reg_results_into_pandas_for_all_y_var(self, reg_type, the_panel=None):
        res1 = dict.fromkeys(self.all_y_reg_vars)
        if reg_type == 'cross_section_ols':
            for y in self.all_y_reg_vars:
                res1[y] = self._put_reg_results_into_pandas_for_single_y_var(reg_type=reg_type,
                                                                             y_var=y,
                                                                             the_panel=the_panel)
        else:
            for y in self.all_y_reg_vars:
                res1[y] = self._put_reg_results_into_pandas_for_single_y_var(reg_type=reg_type, y_var=y)
        res2 = dict.fromkeys(self.ssnames.keys())
        for name1 in res2.keys():
            df_list = []
            for y in self.all_y_reg_vars:
                df_list.append(res1[y][name1])
            adf = functools.reduce(lambda a, b: a.merge(b, how='inner',
                                                        on=['Samples', 'Independent Vars']),
                                   df_list)
            print(adf)
            filename = name1 + '_' + reg_type + '_reg_results.csv'
            adf.to_csv(self.des_stats_root / self.name1_path_keywords[name1] / 'reg_tables_ready_for_latex' / filename)
            res2[name1] = adf
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_numApps_per_text_cluster(self):
        """
        This graph has x-axis as the order rank of text clusters, (for example we have 250 text clusters, we order them from 0 to 249, where
        0th text cluster contains the largest number of apps, as the order rank increases, the number of apps contained in each cluster
        decreases, the y-axis is the number of apps inside each cluster).
        Second meeting with Leah discussed that we will abandon this graph because the number of clusters are too many and they
        are right next to each other to further right of the graph.
        """
        d = self._numApps_per_cluster()
        for name1, content1 in d.items():
            for name2, content2 in content1.items():
                df3 = content2.reset_index()
                df3.columns = ['cluster_labels', 'Apps Count']
                # -------------- plot ----------------------------------------------------------------
                fig, ax = plt.subplots()
                # color the top_n bars
                # after sort descending, the first n ranked clusters (the number in broad_niche_cutoff) is broad
                color = ['red'] * self.broad_niche_cutoff[name1][name2]
                # and the rest of all clusters are niche
                rest = len(df3.index) - self.broad_niche_cutoff[name1][name2]
                color.extend(['blue'] * rest)
                df3.plot.bar( x='cluster_labels',
                              xlabel='Text Clusters',
                              y='Apps Count',
                              ylabel='Apps Count',
                              ax=ax,
                              color=color)
                # customize legend
                BRA = mpatches.Patch(color='red', label='broad apps')
                NIA = mpatches.Patch(color='blue', label='niche apps')
                ax.legend(handles=[BRA, NIA], loc='upper right')
                ax.axes.xaxis.set_ticks([])
                ax.yaxis.set_ticks_position('right')
                ax.spines['left'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.grid(True)
                # label the top n clusters
                df4 = df3.iloc[:self.broad_niche_cutoff[name1][name2], ]
                for index, row in df4.iterrows():
                    value = round(row['Apps Count'])
                    ax.annotate(value,
                                (index, value),
                                xytext=(0, 0.1), # 2 points to the right and 15 points to the top of the point I annotate
                                textcoords='offset points')
                plt.xlabel("Text Clusters")
                plt.ylabel('Apps Count')
                # ------------ set title and save ----------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                file_keywords='numApps_count',
                                                name1=name1,
                                                name2=name2,
                                                graph_title='Histogram of Apps Count In Each Text Cluster',
                                                relevant_folder_name = 'numApps_per_text_cluster')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_numClusters_per_cluster_size_bin(self):
        res = self._numClusters_per_cluster_size_bin()
        for name1, content1 in res.items():
            for name2, dfres in content1.items():
                dfres.reset_index(inplace=True)
                dfres.columns = ['cluster_size_bin', 'Clusters Count']
                fig, ax = plt.subplots()
                fig.subplots_adjust(bottom=0.3)
                dfres.plot.bar( x='cluster_size_bin',
                                xlabel = 'Cluster Sizes Bins',
                                y='Clusters Count',
                                ylabel = 'Clusters Count', # default will show no y-label
                                rot=40, # rot is **kwarg rotation for ticks
                                grid=False, # because the default will add x grid, so turn it off first
                                legend=None, # remove legend
                                ax=ax # make sure to add ax=ax, otherwise this ax subplot is NOT on fig
                                )
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.grid() # since pandas parameter grid = False or True, no options, so I will modify here
                # ------------ set title and save ----------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                file_keywords='numClusters_count',
                                                name1=name1,
                                                name2=name2,
                                                graph_title='Histogram of Clusters In Each Cluster Size Bin',
                                                relevant_folder_name='numClusters_per_cluster_size_bin')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_numApps_per_cluster_size_bin(self):
        res = self._numApps_per_cluster_size_bin()
        for name1, content1 in res.items():
            for name2, dfres in content1.items():
                dfres.reset_index(inplace=True)
                dfres.columns = ['cluster_size_bin', 'numApps_in_cluster_size_bin']
                fig, ax = plt.subplots()
                fig.subplots_adjust(bottom=0.3)
                dfres.plot.bar( x='cluster_size_bin',
                                xlabel = 'Cluster Size Bins',
                                y='numApps_in_cluster_size_bin',
                                ylabel = 'Apps Count', # default will show no y-label
                                rot=40, # rot is **kwarg rotation for ticks
                                grid=False, # because the default will add x grid, so turn it off first
                                legend=None, # remove legend
                                ax=ax # make sure to add ax=ax, otherwise this ax subplot is NOT on fig
                                )
                ax.spines['right'].set_visible(False)
                ax.spines['top'].set_visible(False)
                ax.yaxis.grid() # since pandas parameter grid = False or True, no options, so I will modify here
                # ------------ set title and save ----------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                file_keywords='numApps_per_cluster_size_bin',
                                                name1=name1,
                                                name2=name2,
                                                graph_title='Histogram of Apps Count In Each Cluster Size Bin',
                                                relevant_folder_name='numApps_per_cluster_size_bin')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _groupby_subsample_dfs_by_nichedummy(self):
        d = self._slice_subsamples_dict()
        res = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in d.items():
            res[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                niche_dummy = name1 + '_' + name2 + '_NicheDummy'
                df2 = df.groupby([niche_dummy]).size().to_frame()
                df2.rename(columns={0: name1 + '_' + name2}, index={0: 'Broad Apps', 1: 'Niche Apps'}, inplace=True)
                res[name1][name2] = df2
        return res

    def _combine_name2s_into_single_df(self, name12_list, d):
        """
        :param name2_list: such as ['full_full', 'minInstalls_Tier1', 'minInstalls_Tier2', 'minInstalls_Tier3']
        :param d: the dictionary of single subsample df containing stats
        :return:
        """
        df_list = []
        for name1, content1 in d.items():
            for name2, df in content1.items():
                name12 = name1 + '_' + name2
                if name12 in name12_list:
                    df_list.append(df)
        df2 = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
        l = df2.columns.tolist()
        str_to_replace = {'Non-leaders': '',
                          'Leaders': '',
                          'category': '',
                          '_': ' '}
        for col in l:
            new_col = col
            for k, v in str_to_replace.items():
                new_col = new_col.replace(k, v)
            new_col = new_col.title()
            df2.rename(columns={col: new_col}, inplace=True)
        df2.loc["Total"] = df2.sum(axis=0)
        df2 = df2.sort_values(by='Total', axis=1, ascending=False)
        df2 = df2.drop(labels='Total')
        df2 = df2.T
        return df2

    def niche_by_subsamples_bar_graph(self, name1=None):
        # each sub-sample is a horizontal bar in a single graph
        fig, ax = plt.subplots(figsize=(8, 5))
        fig.subplots_adjust(left=0.2)
        # -------------------------------------------------------------------------
        res = self._groupby_subsample_dfs_by_nichedummy()
        df = self._combine_name2s_into_single_df(name12_list=self.graph_name1_ssnames[name1],
                                                 d=res)
        f_name = name1 + '_niche_by_subsamples_bar_graph.csv'
        q = self.des_stats_root / f_name
        df.to_csv(q)
        # -------------------------------------------------------------------------
        df.plot.barh(stacked=True,
                     color={"Broad Apps": "orangered",
                            "Niche Apps": "lightsalmon"},
                     ax=ax)
        ax.set_ylabel('Sub-samples')
        ax.set_yticklabels(ax.get_yticklabels())
        ax.set_xlabel('Apps Count')
        ax.xaxis.grid()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        graph_title = self.initial_panel + ' ' + self.graph_name1_titles[name1] + \
                      '\n Apps Count by Niche and Broad Types'
        ax.set_title(graph_title)
        ax.legend()
        # ------------------ save file -----------------------------------------------------------------
        self._set_title_and_save_graphs(fig=fig,
                                        name1=name1,
                                        file_keywords=self.graph_name1_titles[name1].lower().replace(' ', '_'),
                                        relevant_folder_name='nichedummy_count_by_subgroup')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _prepare_pricing_vars_for_graph_group_by_var(self,
                                                     group_by_var,
                                                     the_panel=None):
        """
        group_by_var could by either "NicheDummy" or "cluster_size_bin"
        the dataframe (self.cdf) is after the function combine_app_level_text_cluster_stats_with_df
        """
        key_vars = ['Imputedprice',
                    'LogImputedprice',
                    # use this for regression and descriptive stats because it added uniform white noise to avoid 0 price
                    'LogWNImputedprice',
                    'ImputedminInstalls',
                    'LogImputedminInstalls',
                    'offersIAPTrue',
                    'containsAdsTrue']
        if the_panel is not None:
            selected_vars = [i + '_' + the_panel for i in key_vars]
        else:
            selected_vars = [i + '_' + j for j in self.all_panels for i in key_vars]
        d = self._slice_subsamples_dict()
        res12 = dict.fromkeys(self.ssnames.keys())
        res34 = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in d.items():
            res12[name1] = dict.fromkeys(content1.keys())
            res34[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                # ---- prepare regular df with log transformed imputedprice and imputed mininstalls --------
                text_label_var = name1 + '_' + name2 + '_kmeans_labels'
                numApps_in_cluster = name1 + '_' + name2 + '_numApps_in_cluster'
                group_by_var_name = name1 + '_' + name2 + '_' + group_by_var
                # ------------------------------------------------------------------------------------------
                svars = selected_vars + [text_label_var,
                                         group_by_var_name,
                                         numApps_in_cluster]
                df2 = df[svars]
                if the_panel is not None:
                    res12[name1][name2] = df2
                else:
                    # ---------- when no panel is specified, you will need the long form ----------------------
                    df2 = df2.reset_index()
                    ldf = pd.wide_to_long(
                        df2,
                        stubnames=key_vars,
                        i=['index'],
                        j="panel",
                        sep='_').reset_index()
                    ldf["panel"] = pd.to_datetime(ldf["panel"], format='%Y%m')
                    ldf["panel"] = ldf["panel"].dt.strftime('%Y-%m')
                    ldf = ldf.sort_values(by=["index", "panel"]).set_index('index')
                    res12[name1][name2] = ldf
                # ------ prepare df consisting of percentage True in each text cluster size bin for offersIAP and containsAds ------
                if the_panel is not None:
                    panel_var_list = ['offersIAPTrue_' + the_panel, 'containsAdsTrue_' + the_panel]
                    panel_value_var_list = ['TRUE%_offersIAPTrue_' + the_panel, 'TRUE%_containsAdsTrue_' + the_panel]
                else:
                    panel_var_list = ['offersIAPTrue_' + i for i in self.all_panels] + \
                                     ['containsAdsTrue_' + i for i in self.all_panels]
                    panel_value_var_list = ['TRUE%_offersIAPTrue_' + i for i in self.all_panels] + \
                                           ['TRUE%_containsAdsTrue_' + i for i in self.all_panels]
                # calculate the percentage True
                df_list = []
                for var in panel_var_list:
                    df3 = pd.crosstab(  index=df2[group_by_var_name],
                                        columns=[df2[var]],
                                        margins=True)
                    # for cases where only column 1 or column 0 exist for a sub text cluster or niche dummy group
                    if 1 not in df3.columns:
                        print(name1, name2, the_panel, var, 'column 1 does not exist.')
                        df3[1] = 0
                        print('created column 1 with zeros. ')
                    if 0 not in df3.columns:
                        print(name1, name2, the_panel, var, 'column 0 does not exist.')
                        df3[0] = 0
                        print('created column 0 with zeros. ')
                    df3['TRUE%_' + var] = df3[1] / df3['All'] * 100
                    df3['FALSE%_' + var] = df3[0] / df3['All'] * 100
                    df3['TOTAL%_' + var] = df3['TRUE%_' + var] + df3['FALSE%_' + var]
                    df_list.append(df3[['TRUE%_' + var]])
                df4 = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
                df4['TOTAL%'] = 100 # because the text cluster group that do not exist are not in the rows, so TOTAL% is 100
                df4.drop(index='All', inplace=True)
                total = df2.groupby(group_by_var_name)[var].count().to_frame()
                total.rename(columns={var: 'Total_Count'}, inplace=True)
                df5 = total.join(df4, how='left').fillna(0)
                df5.drop(columns='Total_Count', inplace=True)
                df5.reset_index(inplace=True)
                if the_panel is not None:
                    # ------- reshape to have seaborn hues (only for cross section descriptive stats) --------------------
                    # conver to long to have hue for different dependant variables
                    df6 = pd.melt(df5,
                                  id_vars=[group_by_var_name, "TOTAL%"],
                                  value_vars=panel_value_var_list)
                    df6.rename(columns={'value': 'TRUE%', 'variable': 'dep_var'}, inplace=True)
                    df6['dep_var'] = df6['dep_var'].str.replace('TRUE%_', '', regex=False)
                    res34[name1][name2] = df6
                else:
                    # convert to long to have hue for different niche or non-niche dummies
                    ldf = pd.wide_to_long(
                        df5,
                        stubnames=['TRUE%_offersIAPTrue', 'TRUE%_containsAdsTrue'],
                        i=[group_by_var_name],
                        j="panel",
                        sep='_').reset_index()
                    ldf["panel"] = pd.to_datetime(ldf["panel"], format='%Y%m')
                    ldf["panel"] = ldf["panel"].dt.strftime('%Y-%m')
                    ldf = ldf.sort_values(by=["panel"])
                    res34[name1][name2] = ldf
        return res12, res34

    def graph_histogram_pricing_vars_by_niche(self, name1, the_panel):
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
            group_by_var='NicheDummy',
            the_panel=the_panel)
        key_vars = ['LogImputedprice', 'Imputedprice', 'LogWNImputedprice',
                    'LogImputedminInstalls', 'ImputedminInstalls']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=2,
                                   ncols=3,
                                   figsize=(15, 10),
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name2_l = self.ssnames[name1]  # for df names name2 only
            name12_l = self.graph_name1_ssnames[name1]  # for column names name1 + name2
            for j in range(len(name2_l)):
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                sns.histplot(data=res12[name1][name2_l[j]],
                             x=key_vars[i] + "_" + the_panel,
                             hue=name12_l[j] + '_NicheDummy',
                             ax=ax.flat[j])
                sns.despine(right=True, top=True)
                graph_title = self.name12_graph_title_dict[name12_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_ylabel(self.graph_dep_vars_ylabels[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                ax.flat[j].legend().set_visible(False)
            fig.legend(labels=['Niche App : Yes', 'Niche App : No'],
                       loc='lower right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            name1 = name1,
                                            file_keywords=key_vars[i] + '_' + name1 + '_histogram_' + the_panel,
                                            graph_title=self.graph_name1_titles[name1] + \
                                                        ' Cross Section Histogram of \n' + \
                                                        self.graph_dep_vars_titles[key_vars[i]] + the_panel,
                                            relevant_folder_name='pricing_vars_stats')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_descriptive_stats_pricing_vars(self, name1, the_panel):
        """
        For the containsAdsTrue and offersIAPTrue I will put them into 1 graph with different hues
        :param key_vars: 'Imputedprice','ImputedminInstalls','both_IAP_and_ADS'
        :param the_panel: '202106'
        :return:
        """
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
                                    group_by_var='cluster_size_bin',
                                    the_panel=the_panel)
        key_vars = ['LogWNImputedprice', 'LogImputedminInstalls', 'both_IAP_and_ADS']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=2,
                                   ncols=3,
                                   figsize=(15, 10),
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name2_l = self.ssnames[name1] # for df names name2 only
            name12_l = self.graph_name1_ssnames[name1] # for column names name1 + name2
            for j in range(len(name2_l)):
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                if key_vars[i] in ['LogWNImputedprice', 'LogImputedminInstalls']:
                    sns.violinplot(
                        x= name12_l[j] + '_cluster_size_bin',
                        y= key_vars[i] + "_" + the_panel,
                        data=res12[name1][name2_l[j]],
                        color=".8",
                        inner=None,  # because you are overlaying stripplot
                        cut=0,
                        ax=ax.flat[j])
                    # overlay swamp plot with violin plot
                    sns.stripplot(
                        x= name12_l[j] + '_cluster_size_bin',
                        y= key_vars[i] + "_" + the_panel,
                        data=res12[name1][name2_l[j]],
                        jitter=True,
                        ax=ax.flat[j])
                else:
                    total_palette = {"containsAdsTrue_" + the_panel: 'paleturquoise',
                                     "offersIAPTrue_"+ the_panel: 'paleturquoise'}
                    sns.barplot(x= name12_l[j] + '_cluster_size_bin',
                                y='TOTAL%', # total does not matter since if the subsample does not have any apps in a text cluster, the total will always be 0
                                data=res34[name1][name2_l[j]],
                                hue="dep_var",
                                palette=total_palette,
                                ax=ax.flat[j])
                    # bar chart 2 -> bottom bars that overlap with the backdrop of bar chart 1,
                    # chart 2 represents the contains ads True group, thus the remaining backdrop chart 1 represents the False group
                    true_palette = {"containsAdsTrue_" + the_panel: 'darkturquoise',
                                    "offersIAPTrue_" + the_panel: 'teal'}
                    sns.barplot(x= name12_l[j] + '_cluster_size_bin',
                                y='TRUE%',
                                data=res34[name1][name2_l[j]],
                                hue="dep_var",
                                palette=true_palette,
                                ax=ax.flat[j])
                    # add legend
                    sns.despine(right=True, top=True)
                graph_title = self.name12_graph_title_dict[name12_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_ylim(bottom=0)
                ax.flat[j].set_xlabel('Text Cluster Sizes Bins')
                ax.flat[j].set_ylabel(self.graph_dep_vars_ylabels[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                ax.flat[j].legend().set_visible(False)
                if key_vars[i] == 'both_IAP_and_ADS':
                    top_bar = mpatches.Patch(color='paleturquoise',
                                             label='Total (100%)')
                    middle_bar = mpatches.Patch(color='darkturquoise',
                                                label='Contains Ads (%)')
                    bottom_bar = mpatches.Patch(color='teal',
                                                label='Offers IAP (%)')
                    fig.legend(handles=[top_bar, middle_bar, bottom_bar],
                               labels=['Total (100%)', 'Contains Ads (%)', 'Offers IAP (%)'],
                               loc='upper right',
                               ncol=1,  frameon=False)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            name1 = name1, 
                                            file_keywords=key_vars[i] + '_' + name1 + '__' + the_panel,
                                            graph_title=self.graph_name1_titles[name1] + \
                                                        ' Cross Section Descriptive Statistics of \n' + \
                                                        self.graph_dep_vars_titles[key_vars[i]] + the_panel,
                                            relevant_folder_name='pricing_vars_stats')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_corr_heatmap_among_dep_vars(self, name1, the_panel):
        dep_vars = ['LogImputedprice', 'LogImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue']
        selected_vars = [i + '_' + the_panel for i in dep_vars]
        df = self.cdf.copy(deep=True)
        dep_var_df = df[selected_vars]
        correlation_matrix = dep_var_df.corr()
        f_name = the_panel + '_' + name1 + '_dep_vars_corr_matrix.csv'
        q = self.des_stats_root / f_name
        correlation_matrix.to_csv(q)
        # ------------------------------------------------
        plt.figure(figsize=(9, 9))
        labels = ['Log \nPrice', 'Log \nminInstalls', 'IAP', 'Ads']
        heatmap = sns.heatmap(correlation_matrix,
                              xticklabels=labels, yticklabels=labels,
                              vmin=-1, vmax=1, annot=True)
        filename = name1 + ' Dependent Variables Correlation Heatmap'
        heatmap.set_title(filename, fontdict={'fontsize': 12}, pad=12)
        plt.savefig(self.des_stats_root / self.name1_path_keywords[name1] / 'descriptive_stats' / 'graphs' / 'correlation_heatmaps' / filename,
                    facecolor='white',
                    dpi=300)
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_group_mean_subsamples_parallel_trends(self, name1):
        """
        https://github.com/mwaskom/seaborn/blob/master/seaborn/relational.py
        seaborn line plot create a band for measuring central tendancy (95% CI),
        I am curious to know how they calculated the band.
        """
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
            group_by_var='NicheDummy')
        key_vars = ['LogImputedminInstalls', 'LogWNImputedprice', 'TRUE%_offersIAPTrue', 'TRUE%_containsAdsTrue']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=2,
                                   ncols=3,
                                   figsize=(15, 10),
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name2_l = self.ssnames[name1]
            name12_l = self.graph_name1_ssnames[name1]
            for j in range(len(name2_l)):
                nichedummy = name12_l[j] + "_NicheDummy"
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                hue_order = [1, 0]
                if key_vars[i] in ['LogImputedminInstalls', 'LogWNImputedprice']:
                    sns.lineplot(
                        data=res12[name1][name2_l[j]],
                        x="panel",
                        y= key_vars[i],
                        hue=nichedummy,
                        hue_order=hue_order,
                        markers=True,
                        style=nichedummy,
                        dashes=False,
                        ax = ax.flat[j])
                    ylimits = {'LogImputedminInstalls':{'bottom':0, 'top':25},
                               'LogWNImputedprice':{'bottom':0, 'top':2}}
                    ax.flat[j].set_ylim(bottom=ylimits[key_vars[i]]['bottom'],
                                        top=ylimits[key_vars[i]]['top'])
                else:
                    sns.lineplot(
                        data=res34[name1][name2_l[j]],
                        x="panel",
                        y= key_vars[i],
                        hue=nichedummy,
                        hue_order=hue_order,
                        markers=True,
                        style=nichedummy,
                        dashes=False,
                        ax = ax.flat[j])
                    ax.flat[j].set_ylim(bottom=0, top=100)
                graph_title = self.name12_graph_title_dict[name12_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                ax.flat[j].set_xlabel("Time")
                ax.flat[j].set_ylabel(self.graph_dep_vars_ylabels[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                ax.flat[j].legend().set_visible(False)
            fig.legend(labels=['Niche App : Yes', 'Niche App : No'],
                       loc='lower right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            name1=name1,
                                            file_keywords=key_vars[i] + '_' + name1 + '_group_mean_parallel_trends',
                                            graph_title=self.graph_name1_titles[name1] + ' \n' +\
                                                        self.graph_dep_vars_titles[key_vars[i]] +\
                                                        " Group Mean Parallel Trends",
                                            relevant_folder_name='parallel_trend_group_mean')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_beta_parallel_trends(self, name1, alpha):
        """
        :return: six graphs per page (each graph is 1 sub-sample), 1 page has 1 dep var, hues are leaders and non-leaders
        """
        res = self._create_cross_section_reg_results_df_for_parallel_trend_beta_graph(alpha)
        for dep_var in self.all_y_reg_vars:
            fig, ax = plt.subplots(nrows=2,
                                   ncols=3,
                                   figsize=(15, 10),
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = self.graph_name1_ssnames[name1]
            for j in range(len(name1_2_l)):
                df = res[dep_var].copy(deep=True)
                df_subsample = df.loc[df['sub_samples']==name1_2_l[j]]
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                beta_error = [df_subsample['lower_error'], df_subsample['upper_error']]
                ax.flat[j].errorbar(df_subsample['panel'],
                                    df_subsample['beta_nichedummy'],
                                    color='cadetblue',
                                    yerr=beta_error,
                                    fmt='o-', # dot with line
                                    capsize=3)
                ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                graph_title = self.name12_graph_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_xlabel("Time")
                ax.flat[j].set_ylabel('Niche Dummy Coefficient')
                ax.flat[j].grid(True)
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                handles, labels = ax.flat[j].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            name1=name1,
                                            file_keywords=dep_var + '_' + name1 + '_beta_nichedummy_parallel_trends',
                                            graph_title=self.graph_name1_titles[name1] + \
                                                  ' ' + self.graph_dep_vars_titles[dep_var] + \
                                                  " \nRegress on Niche Dummy Coefficient Parallel Trends",
                                            relevant_folder_name='parallel_trend_nichedummy')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    ############## functions for strip violin plots for price and minInstalls graphs #############################
    def _create_log_dep_var_groupby_text_cluster_df(self, the_panel, dep_var):
        """
        :param the_panel: '202106'
        :param dep_var: either 'Imputedprice' or 'ImputedminInstalls'
        :return:
        """
        d = self._select_df_for_key_vars_against_text_clusters(key_vars=[dep_var],
                                                               the_panel=the_panel,
                                                               includeNicheDummy=False,
                                                               convert_to_long=False)
        res = self._df_text_cluster_sizes(d=d)
        for k1, content1 in res.items():
            for k2, df in content1.items():
                df['Log' + dep_var + '_' + the_panel] = np.log2(df[dep_var + '_' + the_panel]+1)
        return res

    ####################### functions for offerIAP, containsAds, paidTrue ####################################################

    def _percentage_of_true_false_groupby_text_cluster(self, the_panel):
        """
        On the y axis, we are going to graph percentage instead of number of apps (still stacked bar graph)
        On the x axis, we are going to use categorical text clusters by the number of apps they contain.
        var is one of paidTrue, offerIAPTrue, containsAdsTrue
        """
        d = self._select_df_for_key_vars_against_text_clusters(key_vars=['offersIAPTrue', 'containsAdsTrue'],
                                                               the_panel=the_panel,
                                                               includeNicheDummy=False,
                                                               convert_to_long=False)
        res = self._df_text_cluster_sizes(d=d)
        res2 = self._groupby_text_size_bins_count_percentage(d=res, the_panel=the_panel,
                                                             count_by_var_list=['offersIAPTrue', 'containsAdsTrue'])
        for k1, content1 in res2.items():
            for k2, df in content1.items():
                # remove one total percentage column because they are the same
                df.drop(columns=['TOTAL%_offersIAPTrue'], axis=1, inplace=True)
                df.rename(columns={'TOTAL%_containsAdsTrue': 'TOTAL%'}, inplace=True)
                # conver to long to have hue in seaborn plotting
                df2 = pd.melt(df,
                              id_vars=['Text Cluster Sizes', "TOTAL%"],
                              value_vars=['TRUE%_containsAdsTrue', 'TRUE%_offersIAPTrue'])
                df2.rename(columns={'value': 'TRUE%', 'variable': 'dep_var'}, inplace=True)
                df2['dep_var'] = df2['dep_var'].str.replace('TRUE%_', '', regex=False)
                res[k1][k2] = df2
        return res

    def cat_var_count(self, cat_var, the_panel=None):
        if the_panel is not None:
            col_name = cat_var + '_' + the_panel
            rd = self.df.groupby(col_name)['count_' + the_panel].count()
            if cat_var == 'minInstalls':  # minInstalls should not be sorted by the number of apps in each group, rather by index
                rd = rd.sort_index(ascending=False)
            else:
                rd = rd.sort_values(ascending=False)
            print(rd)
            return rd
        else:
            col_name = [cat_var + '_' + i for i in self.all_panels]
            df_list = []
            for j in range(len(col_name)):
                rd = self.df.groupby(col_name[j])['count_' + self.all_panels[j]].count()
                if cat_var == 'minInstalls':
                    rd = rd.sort_index(ascending=False)
                else:
                    rd = rd.sort_values(ascending=False)
                rd = rd.to_frame()
                df_list.append(rd)
            dfn = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
            print(dfn)
            return dfn

    def impute_missingSize_as_zero(self):
        """
        size is time invariant, use the mode size as the time invariant variable.
        If the size is not missing, it must not be zero. It is equivalent as having a dummies, where missing is 0 and non-missing is 1,
        and the interaction of the dummy with the original variable is imputing the original's missing as zeros.
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['size'])
        df1['size'] = df1.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
        df1['size'] = df1['size'].fillna(0)
        dcols = ['size_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_contentRating_dummy(self):
        """
        contentRating dummy is time invariant, using the mode (this mode is different from previous imputation mode
        because there is no missings (all imputed).
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['ImputedcontentRating'])
        df1['contentRatingMode'] = df1.mode(axis=1, numeric_only=False, dropna=False).iloc[:, 0]
        df1['contentRatingAdult'] = df1['contentRatingMode'].apply(
            lambda x: 0 if 'Everyone' in x else 1)
        dcols = ['ImputedcontentRating_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def count_number_of_days_since_released(self):
        """
        :param var: time invariant independent variables, could either be released or updated
        :return: a new variable which is the number of days between today() and the datetime
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedreleased'])
        df1['releasedMode'] = df1.mode(axis=1, numeric_only=False, dropna=False).iloc[:, 0]
        df1['DaysSinceReleased'] = pd.Timestamp.now().normalize() - df1['releasedMode']
        df1['DaysSinceReleased'] = df1['DaysSinceReleased'].apply(lambda x: int(x.days))
        dcols = ['Imputedreleased_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_paid_dummies(self):
        """
        paid dummies are time variant
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedfree'])
        for i in self.all_panels:
            df1['paidTrue_' + i] = df1['Imputedfree_' + i].apply(lambda x: 1 if x is False else 0)
        dcols = ['Imputedfree_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def add_white_noise_to_Imputedprice(self):
        """
        Adding white noise Yt = Yt + Nt is to make sure that price fluctuation around a fixed level, so that in Leaders Medical
        sub-sample where all prices are zeros the regression continue to run.
        The results will nevertheless show that nichedummy is neither economically or statistically significant, but all we need to
        do is to show that.
        https://numpy.org/doc/stable/reference/random/index.html
        I think you should add noise before log or other transformation.
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedprice'])
        for i in self.all_panels:
            df1['WNImputedprice_' + i] = df1['Imputedprice_' + i] + np.random.uniform(low=0.01, high=0.1, size=len(df1.index))
            print(i)
            print(df1['Imputedprice_' + i].describe().round(3))
            print(df1['WNImputedprice_' + i].describe().round(3))
            print()
        dcols = ['Imputedprice_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def log_transform_pricing_vars(self):
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedprice', 'WNImputedprice', 'ImputedminInstalls'])
        for i in self.all_panels:
            df1['LogImputedprice_' + i] = np.log(df1['Imputedprice_' + i] + 1)
            df1['LogWNImputedprice_' + i] = np.log(df1['WNImputedprice_' + i] + 1)
            df1['LogImputedminInstalls_' + i] = np.log(df1['ImputedminInstalls_' + i] + 1)
            print(i)
            print(df1['LogImputedprice_' + i].describe().round(3))
            print(df1['LogWNImputedprice_' + i].describe().round(3))
            print(df1['ImputedminInstalls_' + i].describe().round(3))
            print(df1['LogImputedminInstalls_' + i].describe().round(3))
            print()
        dcols = ['Imputedprice_' + i for i in self.all_panels] \
                + ['ImputedminInstalls_' + i for i in self.all_panels] \
                + ['WNImputedprice_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def box_cox_transform_pricing_vars(self):
        """
        I use box_cox instead of log transformation is too weak for Imputedprice and not making it any normal.
        https://towardsdatascience.com/types-of-transformations-for-better-normal-distribution-61c22668d3b9
        """
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedprice', 'WNImputedprice', 'ImputedminInstalls'])
        for i in self.all_panels:
            df1['BCImputedprice_' + i],  lam_Imputedprice = boxcox(df1['Imputedprice_' + i] + 1)
            df1['BCWNImputedprice_' + i], lam_WNImputedprice = boxcox(df1['WNImputedprice_' + i] + 1)
            df1['BCImputedminInstalls_' + i],  lam_ImputedminInstalls = boxcox(df1['ImputedminInstalls_' + i] + 1)
        dcols = ['Imputedprice_' + i for i in self.all_panels] \
                + ['ImputedminInstalls_' + i for i in self.all_panels] \
                + ['WNImputedprice_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_generic_true_false_dummies(self, cat_var):
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputed' + cat_var])
        for i in self.all_panels:
            df1[cat_var + 'True_' + i] = df1['Imputed' + cat_var + '_' + i].apply(lambda x: 1 if x is True else 0)
        dcols = ['Imputed' + cat_var + '_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_NicheDummy(self):
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                niche_col_name = name1 + '_' + name2 + '_NicheDummy'
                self.cdf[niche_col_name] = self.cdf[label_col_name].apply(
                    lambda x: 0 if x in self.broadDummy_labels[name1][name2] else 1)
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_PostDummy(self):
        start_covid_us = datetime.strptime('202003', "%Y%m")
        POST_dummies = []
        for i in self.all_panels:
            panel = datetime.strptime(i, "%Y%m")
            if panel >= start_covid_us:
                self.cdf['PostDummy_' + i] = 1
                POST_dummies.append('PostDummy_' + i)
            else:
                self.cdf['PostDummy_' + i] = 0
                POST_dummies.append('PostDummy_' + i)
        print('CREATED the following post dummies:')
        print(POST_dummies)
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_PostXNiche_interactions(self):
        PostXNiche_dummies = []
        for i in self.all_panels:
            for name1 in self.ssnames.keys():
                for name2 in self.ssnames[name1]:
                    postdummy = 'PostDummy_' + i
                    nichedummy = name1 + '_' + name2 + '_NicheDummy'
                    postxnichedummy = name1 + '_' + name2 + '_PostXNicheDummy_' + i
                    self.cdf[postxnichedummy] = self.cdf[nichedummy] * self.cdf[postdummy]
                    PostXNiche_dummies.append(postxnichedummy)
        print('CREATED the following post niche interaction dummies:')
        print(PostXNiche_dummies)
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def standardize_continuous_vars(self, con_var, method):
        """
        :param con_var:
        :param method: note that preprocessing sklearn transforms each feature (column)
        for example, min max transformation uses the max and min of each column, not of the entire dataframe
        :return:
        """
        df2 = self._select_vars(df=self.cdf, time_variant_vars_list=[con_var])
        print('before standardization:')
        for i in df2.columns:
            print(i)
            print(df2[i].describe().round(3))
            print()
        if method == 'zscore':
            scaler = preprocessing.StandardScaler()
            df3 = scaler.fit_transform(df2)
            df3 = pd.DataFrame(df3)
            df3.columns = ['ZScore' + i for i in df2.columns]
            df3.index = df2.index.tolist()
        print('after standardization:')
        for i in df3.columns:
            print(i)
            print(df3[i].describe().round(3))
            print()
        self.cdf = self.cdf.join(df3, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_demean_time_variant_vars(self, time_variant_vars):
        """
        Because individual dummies regression takes too much time, I decide use this for FE, so that I could also include time invariant variables.
        """
        dfs = []
        for i in time_variant_vars:
            sub_df = self._select_vars(df=self.cdf, time_variant_vars_list=[i])
            sub_df['PanelMean' + i] = sub_df.mean(axis=1)
            for p in self.all_panels:
                sub_df['DeMeaned' + i + '_' + p] = sub_df[i + '_' + p] - sub_df['PanelMean' + i]
            ts_idm = ['DeMeaned' + i + '_' + p for p in self.all_panels]
            for z in sub_df[ts_idm].columns:
                print(z)
                print(sub_df[ts_idm][z].describe().round(3))
                print()
            dfs.append(sub_df[ts_idm])
        df_new = functools.reduce(lambda a, b: a.join(b, how='inner'), dfs)
        self.cdf = self.cdf.join(df_new, how='inner')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    # ----------------------- after creating above variables, save the dataframe as cross section -----------------
    def save_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = self.common_path / 'cross_section_dfs' / filename
        pickle.dump(self.cdf, open(q, 'wb'))
        print(self.initial_panel, ' SAVED CROSS SECTION DFS. ')
        return essay_23_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)
