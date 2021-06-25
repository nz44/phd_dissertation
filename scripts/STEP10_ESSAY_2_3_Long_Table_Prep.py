import pandas as pd
import copy
from pathlib import Path
import pickle
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
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
    graph_ylabel_dict = {'containsAdsTrue': 'ContainsAds',
                         'offersIAPTrue': 'OffersIAP',
                         'paidTrue': 'Paid',
                         'Imputedprice': 'Price'}
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

    def select_vars(self, the_panel=None, **kwargs):
        """
        one input can have both time variant variables list and time invariants vars list, not if else relationship
        """
        selected_cols = []
        if 'time_variant_vars_list' in kwargs.keys():
            time_variables = kwargs['time_variant_vars_list']
            if the_panel is None:
                for p in self.all_panels:
                    selected_cols.extend([item + '_' + p for item in time_variables])
            else:
                selected_cols.extend([item + '_' + the_panel for item in time_variables])
        if 'time_invariant_vars_list' in kwargs.keys():
            selected_cols.extend(kwargs['time_invariant_vars_list'])
        new_df = self.cdf.copy(deep=True)
        selected_df = new_df[selected_cols]
        return selected_df

    ###########################################################################################################
    # Count and Graph NLP Label for all sub samples
    ###########################################################################################################

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

    def _set_title_and_save_graphs(self, fig, name1, name2, graph_title, relevant_folder_name):
        """
        generic internal function to save graphs according to essay 2 (non-leaders) and essay 3 (leaders).
        name1 and name2 are the key names of self.ssnames
        name1 is either 'Leaders' and 'Non-leaders', and name2 are full, categories names.
        graph_title is what is the graph is.
        """
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
        if name1 == 'Leaders':
            fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_essay_3 / relevant_folder_name / filename,
                        facecolor='white',
                        dpi=300)
        else:
            fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_essay_2 / relevant_folder_name / filename,
                        facecolor='white',
                        dpi=300)

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
                df3.rename(columns={ df3.columns[0]: 'Apps Contained in One Cluster'}, inplace = True)
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
                dfres.plot.bar( x='Apps Contained in One Cluster',
                                xlabel = 'Number of Apps Contained in One Cluster',
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

    def _select_df_for_key_vars_against_text_clusters(self, key_vars, the_panel):
        """
        Internal function returns the dataframe for plotting relationship graphs between key_variables and text clusters (by number of apps)
        This is for graphs in essay 2 and essay 3, and for new graphs incorporating Leah's suggestion on June 4 2021
        """
        selected_vars = [i + '_' + the_panel for i in key_vars]
        df2 = self.cdf.copy(deep=True)
        d = {}
        for name1, content1 in self.ssnames.items():
            d[name1] = dict.fromkeys(content1)
            for name2 in content1:
                text_label_var = name1 + '_' + name2 + '_kmeans_labels'
                svars = selected_vars + [text_label_var]
                if name2 == 'full':
                    df3 = df2.loc[df2[name1]==1]
                    d[name1][name2] = df3[svars]
                else:
                    df3 = df2.loc[(df2[name2] == 1) & (df2[name1]==1)]
                    d[name1][name2] = df3[svars]
        return d
    ########################### functions for price graphs #############################################################
    def _create_log_price_groupby_text_cluster_df(self, key_vars, the_panel):
        d = self._select_df_for_key_vars_against_text_clusters(key_vars, the_panel)
        res = dict.fromkeys(d.keys())
        ranges = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                df2 = df.copy(deep=True)
                # create price log (since many prices are zeroes, or below 1, so add 1)
                # so that the log price cannot go below 0
                df2['LogImputedprice_' + the_panel] = np.log2(df2['Imputedprice_' + the_panel]+1)
                # count number of apps in each text cluster
                def _number_apps_in_each_text_cluster(df2):
                    s1 = df2.groupby([k1 + '_' + k2 + '_kmeans_labels']).size().sort_values(ascending=False)
                    return s1
                s1 = _number_apps_in_each_text_cluster(df2)
                # assign that number to dataframe
                df2['appnum_in_text_cluster'] = df2[k1 + '_' + k2 + '_kmeans_labels'].apply(lambda x: s1.loc[x])
                # create categorical variable indicating how many number of apps are there in a text cluster
                df2['Apps Contained in One Cluster'] = pd.cut(df2['appnum_in_text_cluster'], bins=ranges)
                res[k1][k2] = df2
        return res

    def scatter_log_price_text_cluster_categorical_plot(self, key_vars, the_panel):
        res = self._create_log_price_groupby_text_cluster_df(key_vars, the_panel)
        for name1, content1 in res.items():
            for name2, dfres in content1.items():
                # seborn catplot is plt level plot, not axes subplot object
                # https://drawingfromdata.com/pandas/seaborn/matplotlib/visualization/setting-figure-size-matplotlib-seaborn.html
                # sns.catplot returns a facetgrid object, not axis subplot object
                sns.set(style="whitegrid")
                g = sns.catplot(x="Apps Contained in One Cluster",
                                y="LogImputedprice_202106",
                                data=res[name1][name2],
                                height=6,  # make the plot 6 units high
                                aspect=2   # width should be two times height
                                )
                plt.xticks(rotation=45)
                plt.ylim(bottom=0) # natural log of above 1 positive price (ImputedPrice + 1) cannot be negative
                g.set_axis_labels("Number of Apps Contained in One Cluster", "Log Price")
                g.fig.subplots_adjust(left=0.06, bottom = 0.2)
                # ------------ set title and save ----------------------------------------
                self._set_title_and_save_graphs(fig=g.fig, # fig is an attribute of FacetGrid class that return matplotlib fig, so it can uses .suptitle methods
                                                name1=name1, name2=name2,
                                                graph_title="Log Prices in Niche or Broad App Clusters",
                                                relevant_folder_name='niche_scale_scatter_new')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ####################### functions for offerIAP, containsAds, paidTrue ###########################################
    def _percentage_of_true_false_groupby_text_cluster(self, key_vars, the_panel, var):
        """
        On the y axis, we are going to graph percentage instead of number of apps (still stacked bar graph)
        On the x axis, we are going to use categorical text clusters by the number of apps they contain.
        var is one of paidTrue, offerIAPTrue, containsAdsTrue
        """
        d = self._select_df_for_key_vars_against_text_clusters(key_vars, the_panel)
        res = dict.fromkeys(d.keys())
        ranges = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1000]
        for k1, content1 in d.items():
            res[k1] = dict.fromkeys(content1.keys())
            for k2, df in content1.items():
                df2 = df.copy(deep=True)
                # count number of apps in each text cluster
                def _number_apps_in_each_text_cluster(df2):
                    s1 = df2.groupby([k1 + '_' + k2 + '_kmeans_labels']).size().sort_values(ascending=False)
                    return s1
                s1 = _number_apps_in_each_text_cluster(df2)
                # assign that number to dataframe
                df2['appnum_in_text_cluster'] = df2[k1 + '_' + k2 + '_kmeans_labels'].apply(lambda x: s1.loc[x])
                # create categorical variable indicating how many number of apps are there in a text cluster
                df2['Apps Contained in One Cluster'] = pd.cut(df2['appnum_in_text_cluster'], bins=ranges)
                # create percentage by group
                total = df2.groupby('Apps Contained in One Cluster')[var + '_' + the_panel].count().reset_index()
                yes = df2.loc[df2[var + '_' + the_panel] == 1].groupby('Apps Contained in One Cluster')[var + '_' + the_panel].count().reset_index()
                # if you only have if statement, you can write it at the very end,[ for i, j in zip() if ...]
                # but if you have both if and else statements, you must write them before for, [ X if .. else Y for i, j in zip() ]
                yes[var + '_true_percentage'] = [i / j * 100 if j != 0 else 0 for i, j in zip(yes[var + '_' + the_panel], total[var + '_' + the_panel])]
                total[var + '_total_percentage'] = [i / j * 100 if j != 0 else 0 for i, j in zip(total[var + '_' + the_panel], total[var + '_' + the_panel])]
                res[k1][k2] = df2, total, yes
        return res

    def stacked_percentage_bar_graph_groupby_text_cluster(self, key_vars, the_panel, var):
        res = self._percentage_of_true_false_groupby_text_cluster(key_vars, the_panel, var)
        for name1, content1 in res.items():
            for name2, dfres in content1.items():
                # barplot returns ax subplots, but we are overlapping them
                fig = plt.figure(figsize=(12, 6))
                sns.set(style="whitegrid")
                fig.subplots_adjust(bottom=0.2)
                # bar chart 1 -> is 1 because this is total value
                bar1 = sns.barplot(x='Apps Contained in One Cluster', y=var + '_total_percentage', data=dfres[1], color='paleturquoise')
                # bar chart 2 -> bottom bars that overlap with the backdrop of bar chart 1,
                # chart 2 represents the True group, thus the remaining backdrop chart 1 represents the False group
                bar2 = sns.barplot(x='Apps Contained in One Cluster', y=var + '_true_percentage', data=dfres[2], color='darkturquoise')
                # add legend
                sns.despine(right=True, top=True)
                top_bar = mpatches.Patch(color='paleturquoise',  label = var.replace("True", "") + ' : No')
                bottom_bar = mpatches.Patch(color='darkturquoise', label = var.replace("True", "") + ' : Yes')
                plt.legend(handles=[top_bar, bottom_bar])
                # set title and save
                plt.xticks(rotation=45)
                # ------------------
                plt.xlabel("Number of Apps Contained in One Cluster")
                var_title = {'offersIAPTrue': 'Percentage of Apps Offer IAP',
                             'containsAdsTrue': 'Percentage of Apps Contain Ads',
                             'paidTrue': 'Percentage of Apps Charge Upfront'}
                plt.ylabel('Percentage Points')
                # ------------ set title and save ----------------------------------------
                self._set_title_and_save_graphs(fig=fig,
                                                name1=name1, name2=name2,
                                                graph_title= var_title[var] + " in Niche or Broad App Clusters",
                                                relevant_folder_name='niche_scale_scatter_new')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def put_4_dep_vars_graphs_into_single_graph(self, key_vars, the_panel):
        res1 = self._create_log_price_groupby_text_cluster_df(key_vars, the_panel)
        # --------------------------- putting into figure ------------------------------------
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                fig, ax = plt.subplots(nrows=2, ncols=2,
                                       figsize=(11, 8.5),
                                       sharex='col')
                for i in range(len(key_vars)):
                    if key_vars[i] == 'Imputedprice':
                        sns.set(style="whitegrid")
                        sns.stripplot(  x="Apps Contained in One Cluster",
                                        y="LogImputedprice_202106",
                                        data=res1[name1][name2],
                                        ax = ax.flat[i]) # you cannot do ax.flat[i] = sns.histplot, because it will not draw on axes.
                        ax.flat[i].set_xlabel("Number of Apps Contained in One Cluster")
                        ax.flat[i].set_ylabel("Log Price")
                        ax.flat[i].xaxis.set_visible(True)
                        for tick in ax.flat[i].get_xticklabels():
                            tick.set_rotation(45)
                    else:
                        res234 = self._percentage_of_true_false_groupby_text_cluster(key_vars, the_panel, key_vars[i])
                        # bar chart 1 -> is 1 because this is total value
                        sns.set(style="whitegrid")
                        sns.barplot(x='Apps Contained in One Cluster',
                                    y=key_vars[i] + '_total_percentage',
                                    data=res234[name1][name2][1],
                                    color='paleturquoise',
                                    ax = ax.flat[i])
                        # bar chart 2 -> bottom bars that overlap with the backdrop of bar chart 1,
                        # chart 2 represents the True group, thus the remaining backdrop chart 1 represents the False group
                        sns.barplot(x='Apps Contained in One Cluster',
                                    y=key_vars[i] + '_true_percentage',
                                    data=res234[name1][name2][2],
                                    color='darkturquoise',
                                    ax = ax.flat[i])
                        ax.flat[i].set_xlabel("Number of Apps Contained in One Cluster")
                        ax.flat[i].set_ylabel("Percentage Points")
                        # add legend
                        sns.despine(right=True, top=True)
                        top_bar = mpatches.Patch(color='paleturquoise', label=key_vars[i].replace("True", "") + ' : No')
                        bottom_bar = mpatches.Patch(color='darkturquoise', label=key_vars[i].replace("True", "") + ' : Yes')
                        ax.flat[i].legend(handles=[top_bar, bottom_bar], ncol=2)
                        ax.flat[i].xaxis.set_visible(True)
                        for tick in ax.flat[i].get_xticklabels():
                            tick.set_rotation(45)
                # ------------ set title and save ---------------------------------------------
                fig.subplots_adjust(bottom=0.15)
                self._set_title_and_save_graphs(fig=fig,
                                                name1=name1, name2=name2,
                                                graph_title= "Pricing Variables in Niche or Broad App Clusters",
                                                relevant_folder_name='four_dep_vars_in_one_graph')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    ################# functions to graph parallel trend with beta on the y-axis ###############################


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
        df1 = self.select_vars(time_variant_vars_list=['size'])
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
        df1 = self.select_vars(time_variant_vars_list=['ImputedcontentRating'])
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
        df1 = self.select_vars(time_variant_vars_list=['Imputedreleased'])
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
        df1 = self.select_vars(time_variant_vars_list=['Imputedfree'])
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
        df1 = self.select_vars(time_variant_vars_list=['Imputed' + cat_var])
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
        """
        make sure to run this after self.text_cluster_group_count()
        """
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

    def create_demean_time_variant_vars(self, time_variant_vars):
        """
        Because individual dummies regression takes too much time, I decide use this for FE, so that I could also include time invariant variables.
        """
        dfs = []
        for i in time_variant_vars:
            sub_df = self.select_vars(time_variant_vars_list=[i])
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
        df2 = self.select_vars(time_variant_vars_list=[con_var])
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

