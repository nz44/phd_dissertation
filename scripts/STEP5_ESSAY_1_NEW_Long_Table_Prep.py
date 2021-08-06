import pandas as pd
from pathlib import Path
import pickle
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
import statsmodels.api as sm
from datetime import datetime
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")


class reg_preparation_essay_1():
    """2021 July 18
    This is the new version written based on the STEP10_ESSAY_2_3_Long_Table_Prep.py
    """
    panel_essay_1_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_1_panels___')
    reg_table_essay_1_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_1___/reg_results_tables')
    des_stats_tables_essay_1 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_1___/descriptive_stats/tables')
    des_stats_graphs_essay_1 = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_1___/descriptive_stats/graphs')
    des_stats_graphs_overall = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/overall_graphs')
    graph_subsample_title_dict = {  'full_full': 'Full Sample',
                                    'minInstalls_Tier1' : 'Minimum Installs Tier 1',
                                    'minInstalls_Tier2': 'Minimum Installs Tier 2',
                                    'minInstalls_Tier3': 'Minimum Installs Tier 3',
                                    'categories_category_GAME': 'Gaming Apps',
                                    'categories_category_SOCIAL': 'Social Apps',
                                    'categories_category_LIFESTYLE': 'Lifestyle Apps',
                                    'categories_category_MEDICAL': 'Medical Apps',
                                    'categories_category_BUSINESS': 'Business Apps',
                                    'genreId_ART_AND_DESIGN': 'Default Genre Art And Design',
                                    'genreId_COMICS': 'Default Genre Comics',
                                    'genreId_PERSONALIZATION': 'Default Genre Personalization',
                                    'genreId_PHOTOGRAPHY': 'Default Genre Photography',
                                    'genreId_AUTO_AND_VEHICLES': 'Default Genre Auto And Vehicles',
                                    'genreId_GAME_ROLE_PLAYING': 'Default Genre Game Role Playing',
                                    'genreId_GAME_ACTION': 'Default Genre Game Action',
                                    'genreId_GAME_RACING': 'Default Genre Game Racing',
                                    'genreId_TRAVEL_AND_LOCAL': 'Default Genre Travel And Local',
                                    'genreId_GAME_ADVENTURE': 'Default Genre Game Adventure',
                                    'genreId_SOCIAL': 'Default Genre Social',
                                    'genreId_GAME_SIMULATION': 'Default Genre Game Simulation',
                                    'genreId_LIFESTYLE': 'Default Genre Lifestyle',
                                    'genreId_EDUCATION': 'Default Genre Education',
                                    'genreId_BEAUTY': 'Default Genre Beauty',
                                    'genreId_GAME_CASUAL': 'Default Genre Game Casual',
                                    'genreId_BOOKS_AND_REFERENCE': 'Default Genre Books And Reference',
                                    'genreId_BUSINESS': 'Default Genre Business',
                                    'genreId_FINANCE': 'Default Genre Finance',
                                    'genreId_GAME_STRATEGY': 'Default Genre Game Strategy',
                                    'genreId_SPORTS': 'Default Genre Sports',
                                    'genreId_COMMUNICATION': 'Default Genre Communication',
                                    'genreId_DATING': 'Default Genre Dating',
                                    'genreId_ENTERTAINMENT': 'Default Genre Entertainment',
                                    'genreId_GAME_BOARD': 'Default Genre Game Board',
                                    'genreId_EVENTS': 'Default Genre Events',
                                    'genreId_SHOPPING': 'Default Genre Shopping',
                                    'genreId_FOOD_AND_DRINK': 'Default Genre Food And Drink',
                                    'genreId_HEALTH_AND_FITNESS': 'Default Genre Health And Fitness',
                                    'genreId_HOUSE_AND_HOME': 'Default Genre House And Home',
                                    'genreId_TOOLS': 'Default Genre Tools',
                                    'genreId_LIBRARIES_AND_DEMO': 'Default Genre Libraries And Demo',
                                    'genreId_MAPS_AND_NAVIGATION': 'Default Genre Maps And Navigation',
                                    'genreId_MEDICAL': 'Default Genre Medical',
                                    'genreId_MUSIC_AND_AUDIO': 'Default Genre Music And Audio',
                                    'genreId_NEWS_AND_MAGAZINES': 'Default Genre News And Magazines',
                                    'genreId_PARENTING': 'Default Genre Parenting',
                                    'genreId_GAME_PUZZLE': 'Default Genre Game Puzzle',
                                    'genreId_VIDEO_PLAYERS': 'Default Genre Video Players',
                                    'genreId_PRODUCTIVITY': 'Default Genre Productivity',
                                    'genreId_WEATHER': 'Default Genre Weather',
                                    'genreId_GAME_ARCADE': 'Default Genre Game Arcade',
                                    'genreId_GAME_CASINO': 'Default Genre Game Casino',
                                    'genreId_GAME_CARD': 'Default Genre Game Card',
                                    'genreId_GAME_EDUCATIONAL': 'Default Genre Game Educational',
                                    'genreId_GAME_MUSIC': 'Default Genre Game Music',
                                    'genreId_GAME_SPORTS': 'Default Genre Game Sports',
                                    'genreId_GAME_TRIVIA': 'Default Genre Game Trivia',
                                    'genreId_GAME_WORD': 'Default Genre Game Word',
                                    'starDeveloper_top_digital_firms': 'Apps from Top Tier Firms',
                                    'starDeveloper_non-top_digital_firms': 'Apps from Non-top Tier Firms'}
    var_title_dict = {'ImputedminInstalls': 'Log Minimum Installs',
                      'Imputedprice': 'Log Price',
                      'offersIAPTrue': 'Percentage of Apps Offers IAP',
                      'containsAdsTrue': 'Percentage of Apps Contains Ads',
                      'both_IAP_and_ADS': 'Percentage of Apps Contains Ads and Offers IAP'}
    text_cluster_size_bins = [0, 1, 2, 3, 5, 10, 20, 30, 50, 100, 200, 500, 1500]
    text_cluster_size_labels = ['[0, 1]', '(1, 2]', '(2, 3]', '(3, 5]',
                                '(5, 10]', '(10, 20]', '(20, 30]', '(30, 50]',
                                '(50, 100]', '(100, 200]', '(200, 500]', '(500, 1500]']
    graph_combo_name1_list = {
        'combo1': ['full', 'minInstalls'],
        'combo2': ['full', 'starDeveloper'],
        'combo3': ['full', 'categories'],
        'combo4': ['genreId']
    }
    multi_graph_combo_suptitle = {
        'combo1': 'Full Sample and Minimum Install Sub-samples',
        'combo2': 'Full Sample and Apps from Top and Non-top Tier Firms',
        'combo3': 'Full Sample and Five Main Categorical Sub-samples',
        'combo4': 'Default Categorical Sub-samples'
    }
    multi_graph_combo_fig_subplot_layout = {
        'combo1': {'nrows': 2, 'ncols': 2,
                   'figsize': (11, 8.5)},
        'combo2': {'nrows': 1, 'ncols': 3,
                   'figsize': (18, 6)},
        'combo3': {'nrows': 2, 'ncols': 3,
                   'figsize': (16.5, 8.5)},
        'combo4': {'nrows': 7, 'ncols': 7,
                   'figsize': (36, 36)}}

    combo_barh_figsize = {
        'combo1': (8, 5),
        'combo2': (8, 5),
        'combo3': (8, 5),
        'combo4': (8, 16)
    }

    combo_barh_yticklabel_fontsize = {
        'combo1': 10,
        'combo2': 10,
        'combo3': 10,
        'combo4': 6
    }

    combo_graph_titles = {
        'combo1': 'Full Sample and Minimum Installs Sub-samples',
        'combo2': 'Full Sample and Sub-samples Developed by Top and Non-top Firms',
        'combo3': 'Full Sample and Functional Category Sub-samples',
        'combo4': 'Default App Genre Sub-sample'
    }
    # --------------------------------------------------------------------------------------------------------
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
    # Select Vars
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
            for name2 in content1:
                if name2 == 'full':
                    d[name1][name2] = df
                else:
                    d[name1][name2] = df.loc[df[name2]==1]
        return d

    ###########################################################################################################
    # Open, Combine Dataframes check variables and save graph functions
    ###########################################################################################################

    def _open_imputed_deleted_divided_df(self):
        f_name = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = reg_preparation_essay_1.panel_essay_1_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    def _open_predicted_labels_dict(self):
        f_name = self.initial_panel + '_predicted_labels_dict.pickle'
        q = reg_preparation_essay_1.panel_essay_1_path / 'predicted_text_labels' / f_name
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    def _open_app_level_text_cluster_stats(self):
        filename = self.initial_panel + '_dict_app_level_text_cluster_stats.pickle'
        q = reg_preparation_essay_1.panel_essay_1_path / 'app_level_text_cluster_stats' / filename
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    def open_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = reg_preparation_essay_1.panel_essay_1_path / 'cross_section_dfs' / filename
        with open(q, 'rb') as f:
            self.cdf = pickle.load(f)
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def save_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = reg_preparation_essay_1.panel_essay_1_path / 'cross_section_dfs' / filename
        pickle.dump(self.cdf, open(q, 'wb'))
        print(self.initial_panel, ' SAVED CROSS SECTION DFS. ')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def combine_app_level_text_cluster_stats_with_df(self):
        df = self._open_imputed_deleted_divided_df()
        d = self._open_app_level_text_cluster_stats()
        list_of_dfs = [d['full']['full']]
        for name1, content1 in d.items():
            for name2, stats_df in content1.items():
                if name2 != 'full':
                    list_of_dfs.append(stats_df)
        combined_stats_df = functools.reduce(lambda a, b: a.join(b, how='left'), list_of_dfs)
        self.cdf = df.join(combined_stats_df, how='inner')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
                    q = reg_preparation_essay_1.panel_essay_1_path / 'check_predicted_label_text_cols' / f_name
                    df3.to_csv(q)
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_subsample_name_dict(self):
        d = self._open_predicted_labels_dict()
        self.ssnames = dict.fromkeys(d.keys())
        for name1, content in d.items():
            self.ssnames[name1] = list(content.keys())
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def _set_title_and_save_graphs(self, fig,
                                   file_keywords,
                                   relevant_folder_name,
                                   graph_title='',
                                   name1='',
                                   name2=''):
        """
        generic internal function to save graphs according to essay 2 (non-leaders) and essay 3 (leaders).
        name1 and name2 are the key names of self.ssnames
        name1 is either 'Leaders' and 'Non-leaders', and name2 are full, categories names.
        graph_title is what is the graph is.
        """
        # ------------ set title -------------------------------------------------------------------------
        if graph_title != '':
            title = self.initial_panel + ' ' + graph_title
            title = title.title()
            fig.suptitle(title, fontsize='medium')
        # ------------ save ------------------------------------------------------------------------------
        filename = self.initial_panel + '_' + name1 + '_' + name2 + '_' + file_keywords + '.png'
        fig.savefig(reg_preparation_essay_1.des_stats_graphs_essay_1 / relevant_folder_name / filename,
                    facecolor='white',
                    dpi=300)

    ###########################################################################################################
    # Count and Graph NLP Label for all sub samples ###########################################################
    ###########################################################################################################

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

    def determine_niche_broad_cutoff(self):
        d = self._numApps_per_cluster()
        self.broad_niche_cutoff = dict.fromkeys(self.ssnames.keys())
        self.nicheDummy_labels = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            self.broad_niche_cutoff[name1] = dict.fromkeys(content1)
            self.nicheDummy_labels[name1] = dict.fromkeys(content1)
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
                self.nicheDummy_labels[name1][name2] = d[name1][name2][:top_n].index.tolist()
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                       all_panels=self.all_panels,
                                       tcn=self.tcn,
                                       subsample_names=self.ssnames,
                                       combined_df=self.cdf,
                                       broad_niche_cutoff=self.broad_niche_cutoff,
                                       nicheDummy_labels=self.nicheDummy_labels,
                                       long_cdf=self.long_cdf,
                                       individual_dummies_df=self.i_dummies_df)

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
                                                graph_title='Text Cluster Bar Graph',
                                                relevant_folder_name = 'numApps_per_text_cluster')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
                                         bins=reg_preparation_essay_1.text_cluster_size_bins,
                                         include_lowest=True,
                                         labels=reg_preparation_essay_1.text_cluster_size_labels)
                                  ).count()
                df3.rename(columns={'Apps Count': 'Clusters Count'}, inplace=True)
                res[k1][k2] = df3
        return res

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
                                                graph_title='Clusters Count in Certain Size',
                                                relevant_folder_name='numClusters_per_cluster_size_bin')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
                                   bins=reg_preparation_essay_1.text_cluster_size_bins,
                                   include_lowest=True,
                                   labels=reg_preparation_essay_1.text_cluster_size_labels)
                # create a new column indicating grouped sum of numApps_in_cluster for each cluster_size
                df2 = df.groupby('cluster_size_bin').count()
                df3 = df2.iloc[:, 0].to_frame()
                df3.columns = ['numApps_in_cluster_size_bin']
                res[name1][name2] = df3
        return res

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
                                                relevant_folder_name='numApps_per_cluster_size_bin')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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
                                   bins=reg_preparation_essay_1.text_cluster_size_bins,
                                   include_lowest=True,
                                   labels=reg_preparation_essay_1.text_cluster_size_labels)
                # create a new column indicating number of cluster for each cluster size bin
                df[numClusters_in_cluster_size_bin] = df[cluster_size_bin].apply(
                    lambda x: d3[name1][name2].loc[x].squeeze())
                # create a new column indicating grouped sum of numApps_in_cluster for each cluster_size
                df[numApps_in_cluster_size_bin] = df[cluster_size_bin].apply(
                    lambda x: d4[name1][name2].loc[x].squeeze())
                res[name1][name2] = df
        filename = self.initial_panel + '_dict_app_level_text_cluster_stats.pickle'
        q = reg_preparation_essay_1.panel_essay_1_path / 'app_level_text_cluster_stats' / filename
        pickle.dump(res, open(q, 'wb'))
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

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

    def _combine_name2s_into_single_df(self, name2_list, d):
        """
        :param name2_list: such as ['full', 'Tier1', 'Tier2', 'Tier3']
        :param d: the dictionary of single subsample df containing stats
        :return:
        """
        df_list = []
        for name1, content1 in d.items():
            for name2, df in content1.items():
                if name2 in name2_list:
                    df_list.append(df)
        df2 = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
        l = df2.columns.tolist()
        str_to_replace = {'categories_category_': '',
                          'genreId_': '',
                          'minInstalls_': '',
                          'full_': '',
                          'starDeveloper_': '',
                          '_digital_firms': '',
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

    def niche_by_subsamples_bar_graph(self, combo=None):
        # each sub-sample is a horizontal bar in a single graph
        fig, ax = plt.subplots(figsize=reg_preparation_essay_1.combo_barh_figsize[combo])
        fig.subplots_adjust(left=0.2)
        # -------------------------------------------------------------------------
        res = self._groupby_subsample_dfs_by_nichedummy()
        name2_list = []
        for name1 in reg_preparation_essay_1.graph_combo_name1_list[combo]:
            name2_list = name2_list + self.ssnames[name1]
        df = self._combine_name2s_into_single_df(name2_list=name2_list, d=res)
        df.plot.barh(stacked=True,
                     color={"Broad Apps": "orangered",
                            "Niche Apps": "lightsalmon"},
                     ax=ax)
        ax.set_ylabel('Samples')
        ax.set_yticklabels(ax.get_yticklabels(),
                           fontsize=reg_preparation_essay_1.combo_barh_yticklabel_fontsize[combo])
        ax.set_xlabel('Apps Count')
        ax.xaxis.grid()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        graph_title = self.initial_panel + ' ' + reg_preparation_essay_1.combo_graph_titles[combo] + \
                      '\n Apps Count by Niche and Broad Types'
        ax.set_title(graph_title)
        ax.legend()
        # ------------------ save file -----------------------------------------------------------------
        self._set_title_and_save_graphs(fig=fig,
                                        file_keywords=reg_preparation_essay_1.combo_graph_titles[combo].lower().replace(' ', '_'),
                                        relevant_folder_name='nichedummy_count_by_subgroup')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def _prepare_pricing_vars_for_graph_group_by_var(self,
                                                     group_by_var,
                                                     the_panel=None):
        """
        group_by_var could by either "NicheDummy" or "cluster_size_bin"
        the dataframe (self.cdf) is after the function combine_app_level_text_cluster_stats_with_df
        """
        key_vars = ['Imputedprice',
                    'ImputedminInstalls',
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
                    df2['LogImputedprice_' + the_panel] = np.log(df2['Imputedprice_' + the_panel] + 1)
                    df2['LogImputedminInstalls_' + the_panel] = np.log(df2['ImputedminInstalls_' + the_panel] + 1)
                    res12[name1][name2] = df2
                else:
                    for i in self.all_panels:
                        df2['LogImputedprice_' + i] = np.log(df2['Imputedprice_' + i] + 1)
                        df2['LogImputedminInstalls_' + i] = np.log(df2['ImputedminInstalls_' + i] + 1)
                    # ---------- when no panel is specified, you will need the long form ----------------------
                    df2 = df2.reset_index()
                    ldf = pd.wide_to_long(
                        df2,
                        stubnames=['Imputedprice', 'ImputedminInstalls',
                                   'LogImputedprice', 'LogImputedminInstalls',
                                   'offersIAPTrue', 'containsAdsTrue'],
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

    def _rearrange_combo_df_dict(self, d):
        """
        :param d: is any prepared/graph-ready dataframes organized in the dictionary tree in the default structure
        :return:
        """
        res = dict.fromkeys(reg_preparation_essay_1.graph_combo_name1_list.keys())
        for combo in res.keys():
            res[combo] = {}
            for name1 in reg_preparation_essay_1.graph_combo_name1_list[combo]:
                for name2 in self.ssnames[name1]:
                    res[combo][name1 + '_' + name2] = d[name1][name2]
        return res

    def graph_descriptive_stats_pricing_vars(self, combo, key_vars, the_panel):
        """
        For the containsAdsTrue and offersIAPTrue I will put them into 1 graph with different hues
        :param key_vars: 'Imputedprice','ImputedminInstalls','both_IAP_and_ADS'
        :param the_panel: '202106'
        :return:
        """
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
                                    group_by_var='cluster_size_bin',
                                    the_panel=the_panel)
        res12 = self._rearrange_combo_df_dict(d=res12)
        res34 = self._rearrange_combo_df_dict(d=res34)
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=reg_preparation_essay_1.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=reg_preparation_essay_1.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=reg_preparation_essay_1.multi_graph_combo_fig_subplot_layout[combo]['figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = list(res12[combo].keys())
            for j in range(len(name1_2_l)):
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                if key_vars[i] in ['Imputedprice', 'ImputedminInstalls']:
                    sns.violinplot(
                        x= name1_2_l[j] + '_cluster_size_bin',
                        y= "Log" + key_vars[i] + "_" + the_panel,
                        data=res12[combo][name1_2_l[j]],
                        color=".8",
                        inner=None,  # because you are overlaying stripplot
                        cut=0,
                        ax=ax.flat[j])
                    # overlay swamp plot with violin plot
                    sns.stripplot(
                        x= name1_2_l[j] + '_cluster_size_bin',
                        y="Log" + key_vars[i] + "_" + the_panel,
                        data=res12[combo][name1_2_l[j]],
                        jitter=True,
                        ax=ax.flat[j])
                else:
                    total_palette = {"containsAdsTrue_" + the_panel: 'paleturquoise',
                                     "offersIAPTrue_"+ the_panel: 'paleturquoise'}
                    sns.barplot(x= name1_2_l[j] + '_cluster_size_bin',
                                y='TOTAL%', # total does not matter since if the subsample does not have any apps in a text cluster, the total will always be 0
                                data=res34[combo][name1_2_l[j]],
                                hue="dep_var",
                                palette=total_palette,
                                ax=ax.flat[j])
                    # bar chart 2 -> bottom bars that overlap with the backdrop of bar chart 1,
                    # chart 2 represents the contains ads True group, thus the remaining backdrop chart 1 represents the False group
                    true_palette = {"containsAdsTrue_" + the_panel: 'darkturquoise',
                                    "offersIAPTrue_" + the_panel: 'teal'}
                    sns.barplot(x= name1_2_l[j] + '_cluster_size_bin',
                                y='TRUE%',
                                data=res34[combo][name1_2_l[j]],
                                hue="dep_var",
                                palette=true_palette,
                                ax=ax.flat[j])
                    ax.flat[j].set_ylabel("Percentage Points")
                    # add legend
                    sns.despine(right=True, top=True)
                graph_title = reg_preparation_essay_1.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
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
                                            file_keywords=key_vars[i] + '_' + combo + '__' + the_panel,
                                            graph_title=reg_preparation_essay_1.multi_graph_combo_suptitle[combo] + \
                                                        " Cross Section Descriptive Statistics of Pricing Variables for Panel " + \
                                                        the_panel,
                                            relevant_folder_name='pricing_vars_stats')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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

    def _prepare_pricing_vars_for_graph_group_mean_parallel_trends(self):
        res12, res34 = self._prepare_pricing_vars_for_graph_with_text_clusters_bins()
        res12 = self._rearrange_combo_df_dict(d=res12)
        res34 = self._rearrange_combo_df_dict(d=res34)

    def graph_group_mean_subsamples_parallel_trends(self):
        key_vars = ['LogImputedminInstalls', 'LogImputedprice', 'containsAdsTrue', 'offersIAPTrue']
        for i in range(len(key_vars)):
            if key_vars[i] in ['LogImputedminInstalls', 'LogImputedprice']:
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_NicheDummy(self):
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                niche_col_name = name1 + '_' + name2 + '_NicheDummy'
                self.cdf[niche_col_name] = self.cdf[label_col_name].apply(
                    lambda x: 0 if x in self.nicheDummy_labels[name1][name2] else 1)
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
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
        q = reg_preparation_essay_1.panel_essay_1_path / 'converted_long_tables' / filename
        pickle.dump(self.long_cdf, open(q, 'wb'))
        print(self.initial_panel, ' SAVED LONG TABLES. ')
        return reg_preparation_essay_1(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

