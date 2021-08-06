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


class reg_preparation_essay_2_3():
    """by default, regression analysis will either use cross sectional data or panel data with CONSECUTIVE panels,
    this is because we can calculate t-1 easily.
    Two major changes:
    1. I deleted the old scatter plots of niche text cluster vs. key variables.
    2. Deleted everything related to nichescale dummies because this does not reveal important information.
    """
    panel_essay_2_3_common_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_3_common_panels___')
    panel_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_panels___')
    panel_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_3_panels___')
    reg_table_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/reg_results_tables')
    reg_table_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/reg_results_tables')
    des_stats_tables_essay_2 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/descriptive_stats/tables')
    des_stats_tables_essay_3 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/descriptive_stats/tables')
    des_stats_graphs_essay_2 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/descriptive_stats/graphs')
    des_stats_graphs_essay_3 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/descriptive_stats/graphs')
    des_stats_graphs_overall = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/overall_graphs')
    graph_subsample_title_dict = {'Leaders full': 'Market Leaders Full Sample',
                                    'Leaders category_GAME': 'Market Leaders Game Apps',
                                    'Leaders category_BUSINESS': 'Market Leaders Business Apps',
                                    'Leaders category_SOCIAL': 'Market Leaders Social Apps',
                                    'Leaders category_LIFESTYLE': 'Market Leaders Lifestyle Apps',
                                    'Leaders category_MEDICAL': 'Market Leaders Medical Apps',
                                  'Non-leaders full': 'Market Followers Full Sample',
                                  'Non-leaders category_GAME': 'Market Followers Game Apps',
                                  'Non-leaders category_BUSINESS': 'Market Followers Business Apps',
                                  'Non-leaders category_SOCIAL': 'Market Followers Social Apps',
                                  'Non-leaders category_LIFESTYLE': 'Market Followers Lifestyle Apps',
                                  'Non-leaders category_MEDICAL': 'Market Followers Medical Apps'}
    var_title_dict = {'ImputedminInstalls': 'Log Minimum Installs',
                      'Imputedprice': 'Log Price',
                      'offersIAPTrue': 'Percentage of Apps Offers IAP',
                      'containsAdsTrue': 'Percentage of Apps Contains Ads',
                      'both_IAP_and_ADS': 'Percentage of Apps Contains Ads and Offers IAP'}
    def __init__(self,
                 initial_panel,
                 all_panels,
                 tcn,
                 subsample_names=None,
                 combined_df=None,
                 broad_niche_cutoff=None,
                 nicheDummy_labels=None,
                 long_cdf=None,
                 individual_dummies_df=None,
                 descriptive_stats_tables=None,
                 several_reg_results_pandas=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.tcn = tcn
        self.ssnames = subsample_names
        self.cdf = combined_df
        self.broad_niche_cutoff = broad_niche_cutoff
        self.nicheDummy_labels = nicheDummy_labels
        self.long_cdf = long_cdf
        self.i_dummies_df = individual_dummies_df
        self.descriptive_stats_tables = descriptive_stats_tables
        self.several_reg_results = several_reg_results_pandas

    ###########################################################################################################
    # Open and Combine Dataframes
    ###########################################################################################################
    def _select_vars(self, df, time_variant_vars_list=None, time_invariant_vars_list=None):
        df2 = df.copy(deep=True)
        tv_var_list = []
        if time_variant_vars_list is not None:
            for i in time_variant_vars_list:
                vs = [i + '_' + j for j in self.all_panels]
                tv_var_list = tv_var_list + vs
        ti_var_list = []
        if time_invariant_vars_list is not None:
            for i in time_invariant_vars_list:
                ti_var_list.append(i)
        total_vars = tv_var_list + ti_var_list
        df2 = df2[total_vars]
        return df2

    def _open_imputed_deleted_divided_df(self):
        f_name = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    def _open_predicted_labels_dict(self):
        f_name = self.initial_panel + '_predicted_labels_dict.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / 'predicted_text_labels' / f_name
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    def open_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / 'cross_section_dfs' / filename
        with open(q, 'rb') as f:
            self.cdf = pickle.load(f)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def combine_text_labels_with_df(self):
        df = self._open_imputed_deleted_divided_df()
        d = self._open_predicted_labels_dict()
        full_text_label_col = pd.concat([d['Leaders']['full'], d['Non-leaders']['full']], axis=0)
        list_of_text_label_cols = [full_text_label_col]
        for name1, content1 in d.items():
            for name2, text_label_col in content1.items():
                if name2 != 'full':
                    list_of_text_label_cols.append(text_label_col)
        combined_label_df = functools.reduce(lambda a, b: a.join(b, how='left'), list_of_text_label_cols)
        self.cdf = df.join(combined_label_df, how='inner')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def check_text_label_contents(self):
        df2 = self.cdf.copy(deep=True)
        d = self._open_predicted_labels_dict()
        for name1, content in d.items():
            for name2, text_label_col in content.items():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                unique_labels = df2[label_col_name].unique()
                for label_num in unique_labels:
                    df3 = df2.loc[df2[label_col_name]==label_num, [self.tcn + 'ModeClean']]
                    if len(df3.index) >= 10:
                        df3 = df3.sample(n=10)
                    f_name = self.initial_panel + '_' + name1 + '_' + name2 + '_' + 'TL_' + str(label_num) + '_' + self.tcn + '_sample.csv'
                    q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / 'check_predicted_label_text_cols' / f_name
                    df3.to_csv(q)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    # Count and Graph NLP Label for all sub samples ###########################################################

    def create_subsample_name_dict(self):
        d = self._open_predicted_labels_dict()
        self.ssnames = dict.fromkeys(d.keys())
        for name1, content in d.items():
            self.ssnames[name1] = list(content.keys())
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,                                  
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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

    ##############################################################################################################
    ###################  GRAPHING ################################################################################
    ##############################################################################################################

    ############## General functions you use in graphing #########################################################
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

    def _select_df_for_key_vars_against_text_clusters(self, key_vars,
                                                      the_panel=None,
                                                      includeNicheDummy=False,
                                                      convert_to_long=False,
                                                      percentage_true_df=False):
        """
        Internal function returns the dataframe for plotting relationship graphs between key_variables and text clusters (by number of apps)
        This is for graphs in essay 2 and essay 3, and for new graphs incorporating Leah's suggestion on June 4 2021
        """
        if the_panel is not None:
            selected_vars = [i + '_' + the_panel for i in key_vars]
        else:
            selected_vars = [i + '_' + j for j in self.all_panels for i in key_vars]
        d = self._slice_subsamples_dict()
        res = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in d.items():
            res[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                text_label_var = name1 + '_' + name2 + '_kmeans_labels'
                niche_dummy = name1 + '_' + name2 + '_NicheDummy'
                if includeNicheDummy is True:
                    svars = selected_vars + [niche_dummy]
                else:
                    svars = selected_vars + [text_label_var]
                if convert_to_long is False:
                    res[name1][name2] = df[svars]
                else:
                    if percentage_true_df is False:
                        res[name1][name2] = self._convert_to_long_form(df=df[svars], var=key_vars)
                    else:
                        df2 = self._convert_to_long_form(df=df[svars], var=key_vars)
                        res[name1][name2] = self._create_percentage_true_df_from_longdf(var=key_vars,
                                                                                        ldf=df2,
                                                                                        nichedummy=niche_dummy)
        return res

    def _set_title_and_save_graphs(self, fig, name1, graph_title, relevant_folder_name,
                                   name2=None,
                                   subsample_one_graph=False,
                                   essay_2_and_3_overall=False):
        """
        generic internal function to save graphs according to essay 2 (non-leaders) and essay 3 (leaders).
        name1 and name2 are the key names of self.ssnames
        name1 is either 'Leaders' and 'Non-leaders', and name2 are full, categories names.
        graph_title is what is the graph is.
        """
        if essay_2_and_3_overall is True:
            title = self.initial_panel + ' ' + graph_title
            title = title.title()
            fig.suptitle(title, fontsize='medium')
            file_title = graph_title.lower().replace(" ", "_")
            filename = self.initial_panel + '_' + file_title + '.png'
            fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_overall / filename,
                        facecolor='white',
                        dpi=300)
        else:
            if subsample_one_graph is False:
                # ------------ set title -------------------------------------------------------------------------
                subsample_name = name1 + ' ' + name2
                title = self.initial_panel + ' ' \
                        + reg_preparation_essay_2_3.graph_subsample_title_dict[subsample_name] + ' ' \
                        + graph_title
                title = title.title()
                fig.suptitle(title, fontsize='medium')
                # ------------------ save file with name (tolower and replace whitespace with underscores) ------
                file_title = graph_title.lower().replace(" ", "_")
                filename = self.initial_panel + '_' + name1 + '_' + name2 + '_' + file_title + '.png'
            else:
                # ------------ set title -------------------------------------------------------------------------
                title = self.initial_panel + ' ' + name1 \
                        + ' ' \
                        + graph_title + ' In All Subsamples'
                title = title.title()
                fig.suptitle(title, fontsize='medium')
                # ------------------ save file with name (tolower and replace whitespace with underscores) ------
                file_title = graph_title.lower().replace(" ", "_")
                filename = self.initial_panel + '_' + name1 + '_' + file_title + '.png'
            if name1 == 'Leaders':
                fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_essay_3 / relevant_folder_name / filename,
                            facecolor='white',
                            dpi=300)
            else:
                fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_essay_2 / relevant_folder_name / filename,
                            facecolor='white',
                            dpi=300)

    ########## count number of text clusters in each cluster size bin ####################################
    def text_cluster_bar_graph_old(self):
        """
        This graph has x-axis as the order rank of text clusters, (for example we have 250 text clusters, we order them from 0 to 249, where
        0th text cluster contains the largest number of apps, as the order rank increases, the number of apps contained in each cluster
        decreases, the y-axis is the number of apps inside each cluster).
        Second meeting with Leah discussed that we will abandon this graph because the number of clusters are too many and they
        are right next to each other to further right of the graph.
        """
        d = self._text_cluster_group_count()
        for name1, content1 in d.items():
            for name2, content2 in content1.items():
                df3 = content2.reset_index()
                df3.columns = ['text clusters', 'number of apps']
                # -------------- plot ----------------------------------------------------------------
                fig, ax = plt.subplots()
                # color the top_n bars
                # after sort descending, the first n ranked clusters (the number in broad_niche_cutoff) is broad
                color = ['red'] * self.broad_niche_cutoff[name1][name2]
                # and the rest of all clusters are niche
                rest = len(df3.index) - self.broad_niche_cutoff[name1][name2]
                color.extend(['blue'] * rest)
                ax = df3.plot.bar(x='text clusters',
                                  y='number of apps',
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
                    value = round(row['number of apps'])
                    ax.annotate(value,
                                (index, value),
                                xytext=(0, 0.1), # 2 points to the right and 15 points to the top of the point I annotate
                                textcoords='offset points')
                plt.xlabel("Text Clusters")
                plt.ylabel("Number of Apps")
                # ------------ set title and save ----------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                name1=name1, name2=name2,
                                                graph_title='Text Cluster Bar Graph',
                                                relevant_folder_name = 'text_cluster_old')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def _num_clusters_in_each_numApp_range(self):
        d = self._open_predicted_labels_dict()
        res = dict.fromkeys(d.keys())
        ranges = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                df2 = df.copy(deep=True)
                # since the min number of apps in a cluster is 1, not 0, so the smallest range (0, 1] is OK.
                # there is an option include_loweest == True, however, it will return float, but I want integer bins, so I will leave it
                df3 = df2.groupby(pd.cut(df2.iloc[:, 0], ranges)).count()
                df3.columns = ['Number of Clusters']
                df3.reset_index(inplace=True)
                df3.rename(columns={ df3.columns[0]: 'Text Cluster Sizes'}, inplace = True)
                res[k1][k2] = df3
        return res

    def text_cluster_bar_graph_new(self):
        """
        Create a ranked categorical variable, number of apps interval, and number of clusters in each app number interval
        """
        res = self._num_clusters_in_each_numApp_range()
        for name1, content1 in res.items():
            for name2, dfres in content1.items():
                fig, ax = plt.subplots()
                fig.subplots_adjust(bottom=0.3)
                # DO NOT DO ax = df.plot.bar(ax=ax), the real thing worked to assign this pandas plot to ax and
                # return an ax object is ax=ax in the parameters,
                # not the ax = df.plot at the front
                dfres.plot.bar( x='Text Cluster Sizes',
                                xlabel = 'Text Cluster Sizes Bins',
                                y='Number of Clusters',
                                ylabel = 'Number of Clusters', # default will show no y-label
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
                                                name1=name1, name2=name2,
                                                graph_title='Text Cluster Sizes',
                                                relevant_folder_name='text_cluster_new')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ############ Count number of apps in each text cluster sized bin, group_by leaders and non-leaders ##############
    def _df_text_cluster_sizes(self, d):
        """
        :param d: a nested dictionary with name1, and name 2 where name 2 contains dfs, either obtained from
        self._slice_subsamples_dict for from self._select_df_for_key_vars_against_text_clusters
        :return: it will return three more new columns that show each row (an app) belong to a text cluster with its size bin,
        whether it is a large text cluster (500, 1000], a broad app, or a small text cluster (1,2], a niche app.
        """
        res = dict.fromkeys(d.keys())
        ranges = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                df2 = df.copy(deep=True)
                def _number_apps_in_each_text_cluster(df2):
                    s1 = df2.groupby([k1 + '_' + k2 + '_kmeans_labels']).size().sort_values(ascending=False)
                    return s1
                s1 = _number_apps_in_each_text_cluster(df2)
                # assign that number to dataframe
                df2['appnum_in_text_cluster'] = df2[k1 + '_' + k2 + '_kmeans_labels'].apply(lambda x: s1.loc[x])
                # create categorical variable indicating how many number of apps are there in a text cluster
                df2['Text Cluster Sizes'] = pd.cut(df2['appnum_in_text_cluster'], bins=ranges)
                res[k1][k2] = df2
        return res

    def _groupby_text_size_bins_count(self, d):
        """
        :param d: is the output of self._df_text_cluster_sizes(d)
        :return:
        """
        res = dict.fromkeys(d.keys())
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                count_df = df.groupby('Text Cluster Sizes').count().reset_index()
                count_df = count_df.iloc[:, 0:2]
                count_df.rename(columns={count_df.columns[1]: "Apps Count in Text Clusters with the Same Size"}, inplace = True)
                res[k1][k2] = count_df
        return res

    def _place_leader_vs_nonleader_in_same_df(self, d):
        """
        This function aims to graph bar chart of apps count in text cluster with same sizes with leader as a bar and non-leader as a bar
        d is the output of self._groupby_text_size_bins_count(d)
        :return:
        """
        res = dict.fromkeys(d['Leaders'].keys())
        for name2 in res.keys():
            df_list = []
            for name1, content1 in d.items():
                df = content1[name2]
                df.set_index('Text Cluster Sizes', inplace=True)
                for i in df.columns:
                    df.rename(columns={i: i + '_' + name1}, inplace=True)
                df_list.append(df)
            df2 = functools.reduce(lambda a, b:  a.join(b, how='inner'), df_list)
            df2.reset_index(inplace=True)
            # conver to long to have hue in seaborn plotting
            df3 = pd.melt(df2,
                          id_vars=['Text Cluster Sizes'],
                          value_vars=["Apps Count in Text Clusters with the Same Size_Leaders",
                                      "Apps Count in Text Clusters with the Same Size_Non-leaders"])
            df3.rename(columns={'value': 'Apps Count in Text Clusters with the Same Size',
                                'variable': 'sub_samples'}, inplace=True)
            df3['sub_samples'] = df3['sub_samples'].str.replace('Apps Count in Text Clusters with the Same Size_', '', regex=False)
            res[name2] = df3
        return res

    def _groupby_text_size_bins_count_percentage(self, d, the_panel, count_by_var_list):
        """
        :param d: is the output of self._df_text_cluster_sizes(d)
        :param count_by_var: the variable to count by, generally == ['offersIAPTrue', 'containsAdsTrue']
        :return:
        """
        res = dict.fromkeys(d.keys())
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                # create percentage by group, ca is containsAds, of is offersIAP
                df_list = []
                for var in count_by_var_list:
                    total = df.groupby('Text Cluster Sizes')[var + '_' + the_panel].count().reset_index()
                    # this is completely WRONG, because you are calculating percentage of text cluster sizes out of all apps where containsAds is True
                    # you actually wanted to calculate the percentage of containsAds is True out of total apps in one cluster.
                    yes = df.loc[df[var + '_' + the_panel] == 1].groupby('Text Cluster Sizes')[
                        var + '_' + the_panel].count().reset_index()
                    # if you only have if statement, you can write it at the very end,[ for i, j in zip() if ...]
                    # but if you have both if and else statements, you must write them before for, [ X if .. else Y for i, j in zip() ]
                    yes['TRUE%_' + var] = [i / j * 100 if j != 0 else 0 for i, j in
                                           zip(yes[var + '_' + the_panel], total[var + '_' + the_panel])]
                    total['TOTAL%_' + var] = [i / j * 100 if j != 0 else 0 for i, j in
                                              zip(total[var + '_' + the_panel], total[var + '_' + the_panel])]
                    yes.drop(columns=[var + '_' + the_panel], axis=1, inplace=True)
                    yes.set_index('Text Cluster Sizes', inplace=True)
                    total.drop(columns=[var + '_' + the_panel], axis=1, inplace=True)
                    total.set_index('Text Cluster Sizes', inplace=True)
                    per_df = yes.join(total, how='inner')
                    df_list.append(per_df)
                df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
                df3.reset_index(inplace=True)
                res[k1][k2] = df3
        return res

    def graph_number_of_apps_leaders_vs_nonleaders_in_text_cluster_size_bin(self):
        """
        This is based on the dataframe self.open_cross_section_reg_df
        :return:
        """
        d = self._slice_subsamples_dict()
        d2 = self._df_text_cluster_sizes(d=d)
        d3 = self._groupby_text_size_bins_count(d=d2)
        d4 = self._place_leader_vs_nonleader_in_same_df(d=d3)
        fig, ax = plt.subplots(nrows=2, ncols=3,
                               figsize=(16, 8.5),
                               sharex='col',
                               sharey='row')
        fig.subplots_adjust(bottom=0.2)
        sub_sample_names = list(d4.keys())
        for i in range(len(sub_sample_names)):
            sns.set(style="whitegrid")
            sns.despine(right=True, top=True)
            sub_sample_palette = {'Leaders': 'royalblue', 'Non-leaders': 'orchid'}
            sns.barplot(x='Text Cluster Sizes',
                        y='Apps Count in Text Clusters with the Same Size',
                        data=d4[sub_sample_names[i]],
                        hue="sub_samples",
                        palette=sub_sample_palette,
                        ax=ax.flat[i])
            title_name = sub_sample_names[i].replace("category_", "").lower().title()
            ax.flat[i].set_title(title_name)
            ax.flat[i].set_ylabel("Apps Count")
            ax.flat[i].set_ylim(bottom=0)
            ax.flat[i].set_xlabel('Text Cluster Sizes Bins')
            ax.flat[i].yaxis.grid(True)
            ax.flat[i].xaxis.set_visible(True)
            for tick in ax.flat[i].get_xticklabels():
                tick.set_rotation(45)
            ax.flat[i].legend().set_visible(False)
            top_bar = mpatches.Patch(color='royalblue',
                                     label='Leaders')
            bottom_bar = mpatches.Patch(color='orchid',
                                        label='Non-leaders')
            fig.legend(handles=[top_bar, bottom_bar], loc='upper right', ncol=1)
        # ------------ set title and save ---------------------------------------------
        self._set_title_and_save_graphs(fig=fig,
                                        name1=None,
                                        graph_title="Leaders and Non-leaders Apps Count By Text Cluster Size Bins",
                                        relevant_folder_name=None,
                                        essay_2_and_3_overall=True)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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

    def _combine_groupbyed_category_subdfs_into_leader_df(self, d):
        res = dict.fromkeys(d.keys())
        for name1, content1 in d.items():
            df_list = []
            for name2, df in content1.items():
                if name2 != 'full': # exclude the full sample because each column represent a category
                    df_list.append(df)
            res[name1] = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
        return res

    def _combine_groupbyed_leaders_subdfs_into_overall_df(self, d):
        df_list = []
        for name1, content1 in d.items():
            for name2, df in content1.items():
                if name2 == 'full': # exclude the full sample because each column represent a category
                    df_list.append(df)
        res = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
        res.rename(columns={'Leaders_full':'Leaders', 'Non-leaders_full':'Non-leaders'}, inplace=True)
        return res

    def _sort_df_descending_left_to_right(self, df):
        df = df.append(df.sum(numeric_only=True), ignore_index=True)
        df = df.sort_values(by=2, axis=1)
        df = df.drop([df.index[2]])
        df.rename(index={0: 'Broad Apps', 1: 'Niche Apps'}, inplace=True)
        return df

    def niche_by_subsamples_bar_graph(self):
        """
        Create a single graph is the histogram by group
        """
        res = self._groupby_subsample_dfs_by_nichedummy()
        res2 = self._combine_groupbyed_category_subdfs_into_leader_df(d=res)
        res3 = self._combine_groupbyed_leaders_subdfs_into_overall_df(d=res)
        fig, ax = plt.subplots(nrows=3, ncols=1,
                               figsize=(7.5, 13),
                               sharex='col')
        fig.subplots_adjust(left=0.2)
        name1_dict = {'Leaders': 'Market Leaders', 'Non-leaders': 'Market Followers'}
        name1_list = list(res.keys())
        for i in range(len(name1_list)+1):
            if i in range(len(name1_list)):
                ytick_labels = []
                df = self._sort_df_descending_left_to_right(df=res2[name1_list[i]])
                for col_name in df.columns:
                    str_to_remove = name1_list[i] + '_category_'
                    label_text = col_name.replace(str_to_remove, '').lower().title()
                    ytick_labels.append(label_text)
                    ax.flat[i].barh(col_name, df.loc['Niche Apps', col_name],
                                    label=label_text + ' Niche Apps',
                                    color='lightsalmon')
                    ax.flat[i].barh(col_name, df.loc['Broad Apps', col_name],
                                    left=df.loc['Niche Apps', col_name],
                                    label=label_text + ' Broad Apps',
                                    color='orangered')
                    ax.flat[i].set_yticks(np.arange(len(ytick_labels)))
                    ax.flat[i].set_yticklabels(ytick_labels)
                    ax.flat[i].set_title(name1_dict[name1_list[i]])
            else:
                ytick_labels = []
                df = self._sort_df_descending_left_to_right(df=res3)
                for col_name in df.columns:
                    ytick_labels.append(name1_dict[col_name])
                    ax.flat[i].barh(col_name, df.loc['Niche Apps', col_name],
                                    label=col_name + ' Niche Apps',
                                    color='lightsalmon')
                    ax.flat[i].barh(col_name, df.loc['Broad Apps', col_name],
                                    left=df.loc['Niche Apps', col_name],
                                    label=col_name + ' Broad Apps',
                                    color='orangered')
                    ax.flat[i].set_yticks(np.arange(len(ytick_labels)))
                    ax.flat[i].set_yticklabels(ytick_labels)
                    ax.flat[i].set_title('Entire Sample')
            ax.flat[i].spines['top'].set_visible(False)
            ax.flat[i].spines['right'].set_visible(False)
            ax.flat[i].xaxis.grid(True)
            ax.flat[i].set_xlabel('Count')
        left_bar = mpatches.Patch(color='lightsalmon', label='Niche Apps')
        right_bar = mpatches.Patch(color='orangered', label='Broad Apps')
        fig.legend(handles=[left_bar, right_bar], loc='lower center', ncol=2)
        # ------------------ save file with name (tolower and replace whitespace with underscores) ------
        self._set_title_and_save_graphs(fig=fig,
                                        name1=None,
                                        graph_title="Niche and Broad Apps Count By Sub-samples",
                                        relevant_folder_name=None,
                                        essay_2_and_3_overall=True)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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

    def put_6_subsamples_dep_vars_in_1_graph(self, key_vars, the_panel):
        """
        For the containsAdsTrue and offersIAPTrue I will put them into 1 graph with different hues
        :param key_vars: 'Imputedprice','ImputedminInstalls','both_IAP_and_ADS'
        :param the_panel: '202106'
        :return:
        """
        for i in range(len(key_vars)):
            for name1, content1 in self.ssnames.items():
                fig, ax = plt.subplots(nrows=2, ncols=3,
                                       figsize=(16, 8.5),
                                       sharex='col',
                                       sharey='row')
                fig.subplots_adjust(bottom=0.2)
                for j in range(len(content1)):
                    sns.set(style="whitegrid")
                    sns.despine(right=True, top=True)
                    if key_vars[i] in ['Imputedprice', 'ImputedminInstalls']:
                        res12 = self._create_log_dep_var_groupby_text_cluster_df(the_panel, dep_var=key_vars[i])
                        sns.violinplot(
                            x='Text Cluster Sizes',
                            y="Log" + key_vars[i] + "_" + the_panel,
                            data=res12[name1][content1[j]],
                            color=".8",
                            inner=None,  # because you are overlaying stripplot
                            cut=0,
                            ax=ax.flat[j])
                        # overlay swamp plot with box plot
                        sns.stripplot(
                            x='Text Cluster Sizes',
                            y="Log" + key_vars[i] + "_" + the_panel,
                            data=res12[name1][content1[j]],
                            jitter=True,
                            ax=ax.flat[j])
                    else:
                        res34 = self._percentage_of_true_false_groupby_text_cluster(the_panel)
                        # bar chart 1 -> is 1 because this is total value
                        total_palette = {"containsAdsTrue": 'paleturquoise', "offersIAPTrue": 'paleturquoise'}
                        sns.barplot(x='Text Cluster Sizes',
                                    y='TOTAL%', # total does not matter since if the subsample does not have any apps in a text cluster, the total will always be 0
                                    data=res34[name1][content1[j]],
                                    hue="dep_var",
                                    palette=total_palette,
                                    ax=ax.flat[j])
                        # bar chart 2 -> bottom bars that overlap with the backdrop of bar chart 1,
                        # chart 2 represents the contains ads True group, thus the remaining backdrop chart 1 represents the False group
                        true_palette = {"containsAdsTrue": 'darkturquoise', "offersIAPTrue": 'teal'}
                        sns.barplot(x='Text Cluster Sizes',
                                    y='TRUE%',
                                    data=res34[name1][content1[j]],
                                    hue="dep_var",
                                    palette=true_palette,
                                    ax=ax.flat[j])
                        ax.flat[j].set_ylabel("Percentage Points")
                        # add legend
                        sns.despine(right=True, top=True)
                    if content1[j] == 'full':
                        ax.flat[j].set_title('Full Sample')
                    else:
                        sample_name = content1[j].replace("category_", "").lower().title()
                        ax.flat[j].set_title(sample_name)
                    ax.flat[j].set_ylim(bottom=0)
                    ax.flat[j].set_xlabel('Text Cluster Sizes Bins')
                    y_label_dict = {'ImputedminInstalls': 'Log Minimum Installs',
                                    'Imputedprice': 'Log Price',
                                    'both_IAP_and_ADS': 'Percentage Points'}
                    ax.flat[j].set_ylabel(y_label_dict[key_vars[i]])
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
                    fig.legend(handles=[top_bar, middle_bar, bottom_bar], loc='upper right', ncol=1)
                # ------------ set title and save ---------------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                name1=name1,
                                                subsample_one_graph=True,
                                                graph_title=reg_preparation_essay_2_3.var_title_dict[key_vars[i]] + " In All Sub-samples",
                                                relevant_folder_name='subgroups_in_one_graph')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ######### graph group means parallel trends for log minInstalls, log pric, offers IAP and contains ADs ####################
    def _convert_to_long_form(self, df, var):
        """
        The function is used inside self._select_df_for_key_vars_against_text_clusters(
        key_vars=['offersIAPTrue'],
        the_panel=None,
        includeNicheDummy=True,
        convert_to_long=True,
        percentage_true_df=True)
        where you have to turn convert_to_long to True.
        :param df: wide form dataframe
        :param var: should be inside a list
        :return:
        """
        df = df.reset_index()
        ldf = pd.wide_to_long(df,
                              stubnames=var,
                              i=['index'],
                              j="panel",
                              sep='_').reset_index()
        ldf["panel"] = pd.to_datetime(ldf["panel"], format='%Y%m')
        ldf["panel"] = ldf["panel"].dt.strftime('%Y-%m')
        ldf = ldf.sort_values(by=["index", "panel"]).set_index('index')
        return ldf

    def _create_percentage_true_df_from_longdf(self, var, ldf, nichedummy):
        """
        The function is used inside self._select_df_for_key_vars_against_text_clusters(
            key_vars=['offersIAPTrue'],
            the_panel=None,
            includeNicheDummy=True,
            convert_to_long=True,
            percentage_true_df=True)
            where you have to turn both convert_to_long and percentage_true to True.
        :param var should be inside a list, just like its outer function var=['containsAds'] or ['offersIAP']
        :param ldf: it is the output of self._convert_to_long_form(var=['containsAds'] or ['offersIAP'])
        :param nichedummy is string name1 + "_" + name2 + "_NicheDummy"
        :return: df ready for graphing parallel trends for offers IAP and contains ads
        """
        # here we assume var list contains only one string variable
        for i in var:
            str_var = i
        res_true = ldf.groupby(by=["panel", nichedummy]).sum().reset_index().rename(columns={str_var: "TRUE"})
        res_total = ldf.groupby(by=["panel", nichedummy]).count().reset_index().rename(columns={str_var: "TOTAL"})
        res_merge = pd.merge(
                        res_true,
                        res_total,
                        how="inner",
                        on=['panel', nichedummy])
        res_merge['percentage_true'] = res_merge['TRUE']/res_merge['TOTAL'] * 100
        res_merge = res_merge.round({'percentage_true': 0})
        # convert back to wide form in order to pass into sns.lineplot
        res_merge = res_merge.drop(['TRUE', 'TOTAL'], axis=1)
        res_merge = res_merge.pivot_table(index=['panel'],
                                          columns=nichedummy,
                                          values='percentage_true')
        return res_merge

    def put_6_subsamples_parallel_trends_in_1_graph(self, key_vars):
        for i in range(len(key_vars)):
            if key_vars[i] in ['ImputedminInstalls', 'Imputedprice']:
                res = self._select_df_for_key_vars_against_text_clusters(
                    key_vars=[key_vars[i]],
                    the_panel=None,
                    includeNicheDummy=True,
                    convert_to_long=True)
            else:
                res = self._select_df_for_key_vars_against_text_clusters(
                    key_vars=[key_vars[i]],
                    the_panel=None,
                    includeNicheDummy=True,
                    convert_to_long=True,
                    percentage_true_df=True)
            for name1, content1 in self.ssnames.items():
                fig, ax = plt.subplots(nrows=2, ncols=3,
                                       figsize=(16, 8.5),
                                       sharex='col',
                                       sharey='row')
                fig.subplots_adjust(bottom=0.2)
                for j in range(len(content1)):
                    nichedummy = name1 + "_" + content1[j] + "_NicheDummy"
                    sns.set(style="whitegrid")
                    sns.despine(right=True, top=True)
                    hue_order = [1, 0]
                    if key_vars[i] in ['ImputedminInstalls', 'Imputedprice']:
                        res[name1][content1[j]]['Log' + key_vars[i]] = np.log2(res[name1][content1[j]][key_vars[i]] + 1)
                        sns.lineplot(
                            data=res[name1][content1[j]],
                            x="panel",
                            y="Log" + key_vars[i],
                            hue=nichedummy,
                            hue_order=hue_order,
                            style=nichedummy,
                            markers=True,
                            dashes=False,
                            ax = ax.flat[j])
                        y_bottom_lim = {'ImputedminInstalls': {'Leaders': 20, 'Non-leaders': 10},
                                        'Imputedprice': {'Leaders': 0, 'Non-leaders': 0}}
                        y_top_lim = {'ImputedminInstalls': {'Leaders': 28, 'Non-leaders': 20},
                                     'Imputedprice': {'Leaders': 0.3, 'Non-leaders': 1.4}}
                        ax.flat[j].set_ylim(bottom=y_bottom_lim[key_vars[i]][name1],
                                            top=y_top_lim[key_vars[i]][name1])
                    else:
                        sns.lineplot(
                            data=res[name1][content1[j]],
                            hue_order=hue_order,
                            markers=True,
                            dashes=False,
                            ax = ax.flat[j])
                        ax.flat[j].set_ylim(bottom=0, top=100)
                    if content1[j] == 'full':
                        ax.flat[j].set_title('Full Sample')
                    else:
                        sample_name = content1[j].replace("category_", "").lower().title()
                        ax.flat[j].set_title(sample_name)
                    ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                    ax.flat[j].set_xlabel("Time")
                    y_label_dict = {'ImputedminInstalls': 'Log Minimum Installs',
                                    'Imputedprice': 'Log Price',
                                    'offersIAPTrue': 'Percentage of Apps Offers IAP',
                                    'containsAdsTrue': 'Percentage of Apps Contains Ads'}
                    ax.flat[j].set_ylabel(y_label_dict[key_vars[i]])
                    ax.flat[j].xaxis.set_visible(True)
                    for tick in ax.flat[j].get_xticklabels():
                        tick.set_rotation(45)
                    ax.flat[j].legend().set_visible(False)
                fig.legend(labels=['Niche App : Yes', 'Niche App : No'],
                           loc='upper right', ncol=2)
                # ------------ set title and save ---------------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                name1=name1,
                                                subsample_one_graph=True,
                                                graph_title=key_vars[i] + " Parallel Trends",
                                                relevant_folder_name='subgroups_in_one_graph')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ######### graph group means parallel trends for log minInstalls, log pric, offers IAP and contains ADs ####################
    def _cross_section_regression(self, y_var, df, the_panel, log_y):
        """
        https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.html#statsmodels.regression.linear_model.RegressionResults
        #https://www.statsmodels.org/stable/rlm.html
        https://stackoverflow.com/questions/30553838/getting-statsmodels-to-use-heteroskedasticity-corrected-standard-errors-in-coeff
        source code for HC0, HC1, HC2, and HC3, white and Mackinnon
        https://www.statsmodels.org/dev/_modules/statsmodels/regression/linear_model.html
        """
        all_vars = df.columns.values.tolist()
        # y_var is a string without panel substring
        for i in all_vars:
            if y_var in i:
                all_vars.remove(i)
        independents_df = df[all_vars]
        X = sm.add_constant(independents_df)
        if log_y is True:
            y = df[['Log' + y_var + '_' + the_panel]]
        else:
            y = df[[y_var + '_' + the_panel]]
        model = sm.OLS(y, X)
        results = model.fit(cov_type='HC3')
        return results

    def _reg_for_all_subsamples_and_all_panels(self, reg_type, y_var, log_y=None):
        """
        possible input for reg_type are: 'cross_section_ols', uses self._cross_section_regression()
        possible y_var 'Imputedprice', 'ImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue'
        log_y True for 'Imputedprice', 'ImputedminInstalls'
        """
        reg_results = dict.fromkeys(self.all_panels)
        for i in self.all_panels:
            d = self._slice_xy_df_for_subsamples(y_var, the_panel=i, log_y=log_y)
            reg_results[i] = dict.fromkeys(self.ssnames.keys())
            for name1, content1 in d.items():
                reg_results[i][name1] = dict.fromkeys(content1.keys())
                for name2, df in content1.items():
                    if reg_type == 'cross_section_ols':
                        reg_results[i][name1][name2] = self._cross_section_regression(y_var=y_var, df=df, the_panel=i, log_y=log_y)
        return reg_results

    def _create_cross_section_reg_results_df_for_all_dep_vars(self, reg_type, alpha):
        """
        possible input for reg_type are: 'cross_section_ols', uses self._cross_section_regression()
        alpha = 0.05 for 95% CI of coefficients
        """
        # all dependant variables in one dictionary
        res_results = dict.fromkeys(['Imputedprice', 'ImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue'])
        dep_y_log_y = {'Imputedprice': True,
                       'ImputedminInstalls': True,
                       'offersIAPTrue': False,
                       'containsAdsTrue': False}
        # all subsamples are hue in the same graph
        for y_var in res_results.keys():
            res_results[y_var] = self._reg_for_all_subsamples_and_all_panels(reg_type=reg_type,
                                                                             y_var=y_var,
                                                                             log_y=dep_y_log_y[y_var])
        #  since every reg result is one row in dataframe
        res_df = dict.fromkeys(['Imputedprice', 'ImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue'])
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

    def graph_leader_vs_non_leaders_beta_parallel_trends(self, reg_type, alpha):
        """
        :return: six graphs per page (each graph is 1 sub-sample), 1 page has 1 dep var, hues are leaders and non-leaders
        """
        res = self._create_cross_section_reg_results_df_for_all_dep_vars(reg_type, alpha)
        for dep_var in ['Imputedprice', 'ImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue']:
            fig, ax = plt.subplots(nrows=2, ncols=3,
                                   figsize=(18, 8.5),
                                   sharex='col',
                                   sharey='row')
            fig.subplots_adjust(bottom=0.2)
            sub_samples = self.ssnames['Leaders']
            for j in range(len(sub_samples)):
                leaders = 'Leaders_' + sub_samples[j]
                followers = 'Non-leaders_' + sub_samples[j]
                df = res[dep_var].copy(deep=True)
                df_leaders = df.loc[df['sub_samples']==leaders]
                df_followers = df.loc[df['sub_samples']==followers]
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                leaders_beta_error = [df_leaders['lower_error'], df_leaders['upper_error']]
                ax.flat[j].errorbar(df_leaders['panel'],
                                    df_leaders['beta_nichedummy'],
                                    color='cadetblue',
                                    yerr=leaders_beta_error,
                                    fmt='o-', # dot with line
                                    capsize=3,
                                    label='Market Leaders')
                followers_beta_error = [df_followers['lower_error'], df_followers['upper_error']]
                ax.flat[j].errorbar(df_followers['panel'],
                                    df_followers['beta_nichedummy'],
                                    color='palevioletred',
                                    yerr=followers_beta_error,
                                    fmt='o-', # dot with line
                                    capsize=3,
                                    label='Market Followers')
                ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                if sub_samples[j] == 'full':
                    sample_name = "Full Sample"
                else:
                    sample_name = sub_samples[j].replace("category_", "").lower().title()
                ax.flat[j].set_title(sample_name)
                ax.flat[j].set_xlabel("Time")
                ax.flat[j].set_ylabel('Niche Dummy Coefficient')
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                handles, labels = ax.flat[j].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', ncol=2)
            dep_var_dict = {'ImputedminInstalls': 'Log Minimum Installs',
                            'Imputedprice': 'Log Price',
                            'offersIAPTrue': 'Percentage of Apps Offers IAP',
                            'containsAdsTrue': 'Percentage of Apps Contains Ads'}
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            name1=None,
                                            subsample_one_graph=True,
                                            graph_title= dep_var_dict[dep_var] +
                                                         " Niche Dummy Coefficient and Its 95% Confidence Interval " +
                                                         "\n Before and After Covid-19 Stay-At-Home Orders",
                                            relevant_folder_name=None,
                                            essay_2_and_3_overall=True)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ###########################################################################################################
    # Create Variables for Regression / Descriptive Stats
    ###########################################################################################################

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
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_generic_true_false_dummies(self, cat_var):
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputed' + cat_var])
        for i in self.all_panels:
            df1[cat_var + 'True_' + i] = df1['Imputed' + cat_var + '_' + i].apply(lambda x: 1 if x is True else 0)
        dcols = ['Imputed' + cat_var + '_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_NicheDummy(self):
        d = self._text_cluster_group_count()
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                self.cdf[name1 + '_' + name2 + '_NicheDummy'] = self.cdf[label_col_name].apply(
                    lambda x: 0 if x in self.nicheDummy_labels[name1][name2] else 1)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_individual_app_dummies(self):
        df = self.cdf.copy(deep=True)
        df.reset_index(inplace=True)
        df['appId'] = df['index']
        df.set_index('index', inplace=True)
        dummies = df[['appId']]
        dummies = pd.get_dummies(dummies, columns=['appId'], drop_first=True)
        self.i_dummies_df = dummies
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_post_dummy_and_interactions(self):
        start_covid_us = datetime.strptime('202003', "%Y%m")
        nichedummies = self.gather_nicheDummies_into_list()
        created_POST_dummies = []
        for i in self.all_panels:
            panel = datetime.strptime(i, "%Y%m")
            if panel >= start_covid_us:
                self.cdf['PostDummy_' + i] = 1
                created_POST_dummies.append('PostDummy_' + i)
                for j in nichedummies:
                    self.cdf['PostX' + j + '_' + i] = self.cdf[j]
                    created_POST_dummies.append('PostX' + j + '_' + i)
            else:
                self.cdf['PostDummy_' + i] = 0
                created_POST_dummies.append('PostDummy_' + i)
                for j in nichedummies:
                    self.cdf['PostX' + j + '_' + i] = 0
                    created_POST_dummies.append('PostX' + j + '_' + i)
        print('CREATED the following post and niche interactions:')
        print(created_POST_dummies)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    # ----------------------- after creating above variables, save the dataframe as cross section -----------------
    def save_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / 'cross_section_dfs' / filename
        pickle.dump(self.cdf, open(q, 'wb'))
        print(self.initial_panel, ' SAVED CROSS SECTION DFS. ')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
            dfs.append(sub_df[ts_idm])
        df_new = functools.reduce(lambda a, b: a.join(b, how='inner'), dfs)
        self.cdf = self.cdf.join(df_new, how='inner')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
            print(df2[i].describe())
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
            print(df3[i].describe())
            print()
        self.cdf = self.cdf.join(df3, how='inner')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ###########################################################################################################
    # the three functions below are the three step you need to do before converting from wide to long
    # make sure finishing creating all the new vars and demeaning, standardizing before convert to long
    ###########################################################################################################

    def gather_nicheDummies_into_list(self):
        """
        :param n: The number of scale niche dummies, should match to create scale niche dummy function
        """
        gathered_list = []
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                niche_dummy_col = name1 + '_' + name2 +'_NicheDummy'
                gathered_list.append(niche_dummy_col)
        print('Gathered niche variables list (including interactions if exist): ', len(gathered_list))
        print(gathered_list)
        return gathered_list

    def gather_PostNicheDummies_into_list(self):
        gathered_list = []
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                niche_dummy_cols = ['PostX' + name1 + '_' + name2 +'_NicheDummy_' + i for i in self.all_panels]
                gathered_list.extend(niche_dummy_cols)
        print('Gathered niche variables list (including interactions if exist): ', len(gathered_list))
        print(gathered_list)
        return gathered_list

    def gather_slice_dummies_into_list(self):
        gathered_list = []
        for name1, content1 in self.ssnames.items():
            gathered_list.append(name1)
            for i in content1:
                if i != 'full':
                    gathered_list.append(i)
        gathered_list = list(set(gathered_list))
        print("SLICING DUMMIES : ", gathered_list)
        return gathered_list

    def select_all_vars_before_slice_subsamples(self, time_variant_vars, time_invariant_vars):
        niche_dummies_cols = self.gather_nicheDummies_into_list()
        post_niche_dummies_cols = self.gather_PostNicheDummies_into_list()
        allvars= niche_dummies_cols + post_niche_dummies_cols + time_invariant_vars
        for var in time_variant_vars:
            varss = [var + '_' + i for i in self.all_panels]
            allvars.extend(varss)
        slice_dummies_cols = self.gather_slice_dummies_into_list()
        allvars.extend(slice_dummies_cols)
        print('SELECTED VARs : ', len(allvars))
        print(allvars)
        df2 = self.cdf.copy(deep=True)
        df3 = df2[allvars]
        print('SELECTING vars before slicing rows: ')
        print(len(allvars), 'have been selected')
        return df3

    def slice_subsample_dataframes(self, time_variant_vars, time_invariant_vars):
        """
        Internal function that will be called by graph functions
        """
        df2 = self.select_all_vars_before_slice_subsamples(time_variant_vars, time_invariant_vars)
        df_dict = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            df_dict[name1] = dict.fromkeys(content1)
            for name2 in df_dict[name1].keys():
                if name2 == 'full':
                    df_dict[name1][name2] = df2.loc[df2[name1] == 1]
                    print(name1, ' ', name2, ' sliced ', df_dict[name1][name2].shape)
                else:
                    df_dict[name1][name2] = df2.loc[(df2[name1] == 1) & (df2[name2] == 1)]
                    print(name1, ' ', name2, ' sliced ', df_dict[name1][name2].shape)
        print(self.initial_panel, ' FINISHED Slicing Sub Samples')
        return df_dict

    def convert_df_from_wide_to_long(self, time_variant_vars, time_invariant_vars):
        """
        :param time_variant_vars: includes demeaned and original (imputed), and dependant variables
        :param time_invariant_vars:
        :return:
        """
        # since nichescale dummies are also time-variant after interact with Post dummy, I need to expand
        # the time variant var list to include those interaction nichescale dummies.
        stub_names = copy.deepcopy(time_variant_vars)
        for name1, content in self.ssnames.items():
            for name2 in content:
                stub_post = 'PostX' + name1 + '_' + name2 + '_' + 'NicheDummy'
                stub_names.append(stub_post)
        print('CREATED stubnames for conversion from wide to long:')
        print(stub_names)
        df_dict = self.slice_subsample_dataframes(time_variant_vars, time_invariant_vars)
        self.long_cdf = dict.fromkeys(df_dict.keys())
        for name1, content1 in df_dict.items():
            self.long_cdf[name1] = dict.fromkeys(content1.keys())
            for name2, df in content1.items():
                new_df = df.reset_index()
                new_df = pd.wide_to_long(new_df,
                                         stubnames=stub_names,
                                         i="index", j="panel", sep='_') # here you can add developer for multiindex output
                new_df = new_df.sort_index()
                print(name1, ' ', name2, ' BEFORE MERGING individual dummies Long table has shape : ', new_df.shape)
                dfr = self.i_dummies_df.copy(deep=True)
                new_df = new_df.join(dfr, how='left')
                print(name1, ' ', name2, ' AFTER MERGING individual dummies Long table has shape : ', new_df.shape)
                self.long_cdf[name1][name2] = new_df
        print('FINISHED converting from wide to long')
        # --------------------------- save -------------------------------------------------
        filename = self.initial_panel + '_converted_long_table.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / 'converted_long_tables' / filename
        pickle.dump(self.long_cdf, open(q, 'wb'))
        print(self.initial_panel, ' SAVED LONG TABLES. ')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

