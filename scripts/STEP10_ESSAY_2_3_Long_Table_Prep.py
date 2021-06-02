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
    this is because we can calculate t-1 easily."""
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
                                    'Leaders category_GAME': 'Market Leaders Game Sub-sample',
                                    'Leaders category_BUSINESS': 'Market Leaders Business Sub-sample',
                                    'Leaders category_SOCIAL': 'Market Leaders Social Sub-sample',
                                    'Leaders category_LIFESTYLE': 'Market Leaders Lifestyle Sub-sample',
                                    'Leaders category_MEDICAL': 'Market Leaders Medical Sub-sample',
                                  'Non-leaders full': 'Market Followers Full Sample',
                                  'Non-leaders category_GAME': 'Market Followers Game Sub-sample',
                                  'Non-leaders category_BUSINESS': 'Market Followers Business Sub-sample',
                                  'Non-leaders category_SOCIAL': 'Market Followers Social Sub-sample',
                                  'Non-leaders category_LIFESTYLE': 'Market Followers Lifestyle Sub-sample',
                                  'Non-leaders category_MEDICAL': 'Market Followers Medical Sub-sample'}
    def __init__(self,
                 initial_panel,
                 all_panels,
                 tcn,
                 niche_keyvar_dfs=None,
                 subsample_names=None,
                 df=None,
                 text_label_df=None,
                 combined_df=None,
                 text_label_count_df=None,
                 broad_niche_cutoff=None,
                 nicheDummy_labels=None,
                 long_cdf=None,
                 individual_dummies_df=None,
                 descriptive_stats_tables=None,
                 several_reg_results_pandas=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.tcn = tcn
        self.niche_kv_dfs = niche_keyvar_dfs
        self.ssnames = subsample_names
        self.df = df # df is the output of combine_imputed_deleted_missing_with_text_labels
        self.text_label_df = text_label_df
        self.cdf = combined_df
        self.tlc_df = text_label_count_df
        self.broad_niche_cutoff = broad_niche_cutoff
        self.nicheDummy_labels = nicheDummy_labels
        self.long_cdf = long_cdf
        self.i_dummies_df = individual_dummies_df
        self.descriptive_stats_tables = descriptive_stats_tables
        self.several_reg_results = several_reg_results_pandas

    ###########################################################################################################
    # Open and Combine Dataframes
    ###########################################################################################################

    def open_imputed_deleted_divided_df(self):
        f_name = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def open_predicted_labels_dict(self):
        f_name = self.initial_panel + '_predicted_labels_dict.pickle'
        q = reg_preparation_essay_2_3.panel_essay_2_3_common_path / 'predicted_text_labels' / f_name
        with open(q, 'rb') as f:
            self.text_label_df = pickle.load(f)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def combine_text_labels_with_df(self):
        full_text_label_col = pd.concat([self.text_label_df['Leaders']['full'], self.text_label_df['Non-leaders']['full']], axis=0)
        list_of_text_label_cols = [full_text_label_col]
        for name1, content1 in self.text_label_df.items():
            for name2, text_label_col in content1.items():
                if name2 != 'full':
                    list_of_text_label_cols.append(text_label_col)
        combined_label_df = functools.reduce(lambda a, b: a.join(b, how='left'), list_of_text_label_cols)
        self.cdf = self.df.join(combined_label_df, how='inner')
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def check_text_label_contents(self):
        df2 = self.cdf.copy(deep=True)
        for name1, content in self.text_label_df.items():
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
        self.ssnames = dict.fromkeys(self.text_label_df.keys())
        for name1, content in self.text_label_df.items():
            self.ssnames[name1] = list(content.keys())
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def text_cluster_group_count(self):
        df2 = self.cdf.copy(deep=True)
        self.tlc_df = dict.fromkeys(self.ssnames.keys())
        self.broad_niche_cutoff = dict.fromkeys(self.ssnames.keys())
        self.nicheDummy_labels = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            self.tlc_df[name1] = dict.fromkeys(content1)
            self.broad_niche_cutoff[name1] = dict.fromkeys(content1)
            self.nicheDummy_labels[name1] = dict.fromkeys(content1)
            for name2 in self.tlc_df[name1].keys():
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
                self.tlc_df[name1][name2] = df2.groupby([label_col_name]).size(
                    ).sort_values(ascending=False).rename(name1 + '_' + name2 + '_Apps_Count').to_frame()
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def text_cluster_bar_chart(self):
        lcc = copy.deepcopy(self.tlc_df)
        for name1, content1 in lcc.items():
            for name2, content2 in content1.items():
                df3 = content2.reset_index()
                df3.columns = ['text clusters', 'number of apps']
                # -------------- plot ----------------------------------------------------------------
                fig, ax = plt.subplots()
                # color the top_n bars
                color = ['red'] * self.broad_niche_cutoff[name1][name2]
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
                ax.set_xlabel("Text Clusters")
                ax.set_ylabel("Number of Apps")
                # ------------ set title ----------------------------------------
                subsample_name = name1 + ' ' + name2
                title = self.initial_panel + ' ' \
                            + reg_preparation_essay_2_3.graph_subsample_title_dict[subsample_name] \
                            + ' Text Cluster Bar Graph'
                title = title.title()
                ax.set_title(title)
                filename = self.initial_panel + '_' + name1 + '_' + name2 + '_text_cluster_bar.png'
                if name1 == 'Leaders':
                    fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_essay_3 / 'text_cluster_bar' / filename,
                            facecolor='white',
                            dpi=300)
                else:
                    fig.savefig(reg_preparation_essay_2_3.des_stats_graphs_essay_2 / 'text_cluster_bar' / filename,
                            facecolor='white',
                            dpi=300)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def niche_scale_scatter_plot_against_key_vars(self, the_panel):
        key_vars = ['Imputedprice', 'offersIAPTrue', 'containsAdsTrue', 'paidTrue']
        selected_vars = [i + '_' + the_panel for i in key_vars]
        df2 = self.cdf.copy(deep=True)
        # ------------ fill in self.niche_kv_dfs -----------------------------------------------------
        self.niche_kv_dfs = {}
        for key, sub_sample_dummies in self.ssnames.items():
            self.niche_kv_dfs[key] = {}
            if key == 'full':
                svars = copy.deepcopy(selected_vars)
                svars.extend(['full_full_kmeans_labels'])
                df = df2.loc[:, svars]
                self.niche_kv_dfs['full']['full'] = df
            elif key == 'developer':
                svars = copy.deepcopy(selected_vars)
                svars.extend(['top_digital_firms',
                              'developer_top_kmeans_labels',
                              'developer_non-top_kmeans_labels'])
                self.niche_kv_dfs['developer']['top'] = df2.loc[df2['top_digital_firms']==1, svars]
                self.niche_kv_dfs['developer']['non-top'] = df2.loc[
                    df2['top_digital_firms'] == 0, svars]
            else:
                for ss_dummy in sub_sample_dummies:
                    svars = copy.deepcopy(selected_vars)
                    svars.extend([ss_dummy, key + '_' + ss_dummy + '_kmeans_labels'])
                    self.niche_kv_dfs[key][ss_dummy] = df2.loc[df2[ss_dummy]==1, svars]
        # ------------ create niche indicator according to group size (prepare to graph)  -----------
        for name1, content1 in self.niche_kv_dfs.items():
            for name2, df in content1.items():
                df2 = self._create_index_indicator_based_on_group_size(name1=name1, name2=name2, df=df)
                # the reason there are nan in niche indicators is because in 202104 the niche labels are generated with different subsamples
                # because the cutoff points are adjusted in 202105. Thus some previous memebrs were not in this group, thus they were
                # not accounted for when generating text labels within each sub samples.
                print(name1, name2, ' BEFORE dropping nan in niche indicators : ', df.shape)
                df2.dropna(subset=['niche_indicators'], inplace=True)
                print(name1, name2, ' AFTER dropping nan in niche indicators : ', df.shape)
                fig = self._scatter_graph_niche_indicator_against_a_key_var(name1=name1,
                                                                            name2=name2,
                                                                            df=df2,
                                                                            key_vars=selected_vars,
                                                                            the_panel=the_panel)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def _create_index_indicator_based_on_group_size(self, name1, name2, df):
        """
        :param df: the df stored in self.niche_kv_dfs, after running self.niche_scale_scatter_plot_against_key_vars
        :return:
        """
        df2 = df.copy(deep=True)
        df3 = df2.groupby([name1 + '_' + name2 + '_kmeans_labels']).size().sort_values(ascending=False).to_frame().reset_index()
        # niche_indicator: 0 the group with most members (most broad), 1 decreasing members ...
        df3['niche_indicators'] = np.arange(df3.shape[0])
        niche_label_and_indicators = list(zip(df3[name1 + '_' + name2 + '_kmeans_labels'], df3['niche_indicators']))
        df2['niche_indicators'] = None
        for i in niche_label_and_indicators:
            df2.at[df2[name1 + '_' + name2 + '_kmeans_labels'] == i[0], 'niche_indicators'] = i[1]
        return df2

    def _scatter_graph_niche_indicator_against_a_key_var(self, name1, name2, df, key_vars, the_panel):
        """
        :param df: the df output of self._create_index_indicator_based_on_group_size
        :param key_vars: a list of key variables
        :return:
        """
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(11, 8.5))
        fig.tight_layout(pad=3)
        for i in range(len(key_vars)):
            if 'price' in key_vars[i]:
                pass
                ax.flat[i] = df.plot.scatter(x='niche_indicators',
                                        y=key_vars[i], ax=ax.flat[i])
                ax.flat[i].set_ylabel('Price')
                np1 = df['niche_indicators'].to_numpy()
            else:
                df2 = df.copy(deep=True)
                df3 = df2[['niche_indicators', key_vars[i]]]
                df3['Apps'] = 1 # count column
                df4 = df3.groupby(['niche_indicators', key_vars[i]]).sum().unstack()
                df4 = df4.fillna(0) # beacuse nan means actually there is no member in the group
                ax.flat[i] = df4.plot(kind='bar', y='Apps', stacked=True, ax=ax.flat[i])
                ax.flat[i].set_ylabel('Apps')
                np1 = df4.index.to_numpy()
            if name1 == 'full':
                ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 50.0).astype(int))
            elif name1 == 'developer':
                if name2 == 'top':
                    ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 10.0).astype(int))
                else:
                    ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 25.0).astype(int))
            elif name1 == 'minInstalls':
                if name2 == 'ImputedminInstalls_tier1':
                    ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 50.0).astype(int))
                elif name2 == 'ImputedminInstalls_tier2':
                    ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 50.0).astype(int))
                else:
                    ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 10.0).astype(int))
            else:
                ax.flat[i].set_xticks(np.arange(min(np1), max(np1) + 1, 5.0).astype(int))
            ax.flat[i].set_xticklabels(ax.flat[i].get_xticks(), rotation=0)
            ax.flat[i].set_xlabel('Niche Scale (0 is the most broad type)')
        # ------------ set title --------------------------------------
        if name1 != 'genreId':
            subsample_name = name1 + ' ' + name2
            title = self.initial_panel + ' Dataset -- Panel ' + the_panel + ' ' \
                     + reg_preparation_essay_2_3.graph_subsample_title_dict[subsample_name] \
                     + '\nPricing Variables Against Niche Scale'
        else:
            title = self.initial_panel + ' Dataset -- Panel ' + the_panel + ' ' \
                     + name1 + ' ' + name2 \
                     + '\nPricing Variables Against Niche Scale'
            title = title.replace("genreId", "Category")
            title = title.replace("_", " ")
            title = title.lower()
        title = title.title()
        fig.suptitle(title, fontsize=14)
        plt.subplots_adjust(top=0.9)
        filename = self.initial_panel + '_' + the_panel + '_' + name1 + '_' + name2 + '_niche_scale_scatter.png'
        fig.savefig(reg_preparation_essay_2_3.descriptive_stats_graphs / 'niche_scale_scatter' / filename,
                    facecolor='white',
                    dpi=300)
        return fig

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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def create_NicheScaleDummies_for_full_sample(self, n):
        """
        I am going to skip this step for essay 2 and essay 3 because in essay 1 this does not seem to produce meaningful results.
        """
        df2 = self.cdf.copy(deep=True)
        df3 = df2.groupby(['full_full_kmeans_labels']).size().sort_values(ascending=False)
        x = round(len(df3) / n)
        frames = [df3.iloc[j * x:(j + 1) * x].copy() for j in range(n - 1)]
        last_df = df3.iloc[(n - 1) * x: len(df3)]
        frames.extend([last_df])
        labels = [list(dff.index.values) for dff in frames]
        for z in range(n):
            self.cdf['full_full_NicheScaleDummy_' + str(z)] = self.cdf['full_full_kmeans_labels'].apply(
                lambda x: 1 if x in labels[z] else 0)
        nichescales = []
        for i in self.cdf.columns:
            if 'full_full_NicheScaleDummy_' in i:
                nichescales.append(i)
        print('FINISHED creating niche scale dummies for full sample: ')
        print(nichescales)
        return reg_preparation_essay_2_3(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
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
                                   niche_keyvar_dfs=self.niche_kv_dfs,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   text_label_count_df=self.tlc_df,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

