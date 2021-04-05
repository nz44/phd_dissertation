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

# *********************************************************************************************
# *********************************************************************************************
# ******************* COMBINE DATAFRAMES ******************************************************
# *********************************************************************************************
# *********************************************************************************************

class combine_dataframes():
    combined_df_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/combined_with_labels')

    def __init__(self,
                 initial_panel,
                 all_panels,
                 consec_panels,
                 appid_imputed_and_deleted_missing_df,
                 dev_index_gecoded_df,
                 predicted_labels,
                 dataframe_with_labels=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
        self.dfa = appid_imputed_and_deleted_missing_df
        self.dfd = dev_index_gecoded_df
        self.pl = predicted_labels
        self.dfl = dataframe_with_labels

    def combine_subsamples_with_predicted_labels(self):
        """
        run tnis after self.prepare_text_col and self.impute_text_cols and self.find_appids_to_remove_before_imputing
        """
        self.dfa['genreIdGame'] = self.dfa['genreId_' + self.all_panels[-1]].apply(lambda x: 1 if 'GAME' in x else 0)
        dfl = self.dfa.copy(deep=True)
        dfl = dfl.join(self.pl['full'], how='inner')
        dfl = dfl.join(self.pl['game'], how='left')
        dfl = dfl.join(self.pl['nongame'], how='left')
        # ------------------------------------------------------------------------------
        print(self.initial_panel, ' check number of non-missing rows in each label columns: ')
        print('full dataframe dimension : ', dfl.shape)
        dff = dfl[dfl['full_kmeans_labels'].notnull()]
        print('full sample non-missing rows : ', dff.shape)
        dfg = dfl[dfl['genreIdGame']==1]
        print('game dataframe dimension : ', dfg.shape)
        dfg = dfl[dfl['game_kmeans_labels'].notnull()]
        print('game subsample non-missing rows : ', dfg.shape)
        dfn = dfl[dfl['genreIdGame'] == 0]
        print('nongame dataframe dimension : ', dfn.shape)
        dfn = dfl[dfl['nongame_kmeans_labels'].notnull()]
        print('nongame subsample non-missing rows : ', dfn.shape)
        self.dfl = dfl
        # ------------------------------------------------------------------------------
        f_name = self.initial_panel + '_dataframe_with_labels.pickle'
        q = combine_dataframes.combined_df_path / f_name
        pickle.dump(self.dfl, open(q, 'wb'))
        return combine_dataframes(
                    initial_panel=self.initial_panel,
                    all_panels=self.all_panels,
                    consec_panels=self.consec_panels,
                    appid_imputed_and_deleted_missing_df=self.dfa,
                    dev_index_gecoded_df=self.dfd,
                    predicted_labels=self.pl,
                    dataframe_with_labels=self.dfl)

# *********************************************************************************************
# *********************************************************************************************
# ******************* REGRESSION CLASS ********************************************************
# *********************************************************************************************
# *********************************************************************************************

class regression_analysis():
    """by default, regression analysis will either use cross sectional data or panel data with CONSECUTIVE panels,
    this is because we can calculate t-1 easily."""
    reg_table_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/reg_results_tables')
    descriptive_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/tables')
    descriptive_stats_graphs = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/graphs')
    var_latex_map = {
               'const': 'Constant',
               'score': 'Rating',
               'DeMeanedscore': 'Demeaned Rating',
               'reviews': 'Reviews',
               'ZSCOREreviews': 'Z Score Reviews',
               'DeMeanedZSCOREreviews': 'Demeaned Z Score Reviews',
               'minInstallsTop': '\makecell[l]{High Level \\\ Minimum Installs}',
               'DeMeanedminInstallsTop': '\makecell[l]{Demeaned High Level \\\ Minimum Installs}',
               'minInstallsMiddle': '\makecell[l]{Medium Level \\\ Minimum Installs}',
               'DeMeanedminInstallsMiddle': '\makecell[l]{Demeaned Medium Level \\\ Minimum Installs}',
               'minInstallsBottom': '\makecell[l]{Low Level \\\ Minimum Installs}',
               'DeMeanedminInstallsBottom': '\makecell[l]{Demeaned Low Level \\\ Minimum Installs}',
               'niche_app': 'Niche',
               'genreIdGame': 'Hedonic',
               'contentRatingAdult': 'Age Restrictive',
               'DaysSinceReleased': 'Released',
               'paidTrue': 'Paid',
               'offersIAPTrue': 'Offers IAP',
               'containsAdsTrue': 'Contains ads',
               'price': 'Price',
               'F stat': 'F statistic',
               'P-value': 'P Value',
               'rsquared': 'R Squared',
               'nobs': '\makecell[l]{number of \\\ observations}',
               '_cov_type': 'Covariance Type'}

    var_definition = {
            'Rating_{i,t}': '\makecell[l]{Weighted average (from 1 to 5) of cumulative consumer ratings of app $i$}',
            'Demeaned Rating_{i,t}': '\makecell[l]{Time demean $Rating_{i,t}$ by subtracting \\\ the mean of 7 consecutive monthly periods}',
            'Reviews_{i,t}': '\makecell[l]{Number of cumulative consumer reviews \\\ for the app $i$ between its release and period $t$}',
            'Z Score Reviews_{i,t}': 'Normalize number of reviews for App $i$ in period $t$ using Z-Score',
            'Demeaned Z Score Reviews_{i,t}': '\makecell[l]{Time demeaned z-score reviews \\\ by subtracting the mean from 7 consecutive periods}',
            '\makecell[l]{High Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the minimum cumulative installs of the app $i$ in \\\ period $t$ is above 10,000,000, otherwise 0.}',
            '\makecell[l]{Demeaned High Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Time demean High Level Minimum $Installs_{i,t}$ \\\ by subtracting the mean from 7 consecutive periods}',
            '\makecell[l]{Medium Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the minimum cumulative installs of the app $i$ \\\ in period $t$ is between 10,000 and 10,000,000, otherwise 0.}',
            '\makecell[l]{Demeaned Medium Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Time demean Medium Level Minimum $Installs_{i,t}$ \\\ by subtracting the mean from 7 consecutive periods}',
            '\makecell[l]{Low Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the minimum cumulative installs of the app $i$ \\\ in period $t$ is below 10,000, otherwise 0.}',
            '\makecell[l]{Demeaned Low Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Time demean Low Level Minimum $Installs_{i,t}$ \\\ by subtracting the mean from 7 consecutive periods}',
            'Niche_{i}': '\makecell[l]{Time invariant dummy variable which \\\ equals to 1 if App $i$ is niche, otherwise 0}',
            'Hedonic_{i}': '\makecell[l]{Time invariant dummy variable which \\\ equals to 1 if App $i$ is in the category GAME, otherwise 0}',
            'Age Restrictive_{i}': '\makecell[l]{Time invariant dummy variable which \\\ equals to 1 if App $i$ contains mature (17+) \\\ or adult (18+) content, otherwise 0}',
            'Released_{i}': '\makecell[l]{The number of days \\\ since App $i$ was released}',
            'Paid_{i,t}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the App $i$ is paid in period $t$}',
            'Offers IAP_{i,t}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the App $i$ offers within app purchases (IAP)}',
            'Contains ads_{i,t}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the App $i$ contains advertisement in period $t$}',
            'Price_{i,t}': 'Price of App $i$ in period $t$'}

    descriptive_stats_table_row_order = {
        'niche_app': 0,
        'price': 1,
        'paidTrue': 2,
        'offersIAPTrue': 3,
        'containsAdsTrue': 4,
        'genreIdGame': 5,
        'contentRatingAdult': 6,
        'DaysSinceReleased': 7,
        'minInstallsTop': 8,
        'DeMeanedminInstallsTop': 9,
        'minInstallsMiddle': 10,
        'DeMeanedminInstallsMiddle': 11,
        'minInstallsBottom': 12,
        'DeMeanedminInstallsBottom': 13,
        'score': 14,
        'DeMeanedscore': 15,
        'reviews': 16,
        'ZSCOREreviews': 17,
        'DeMeanedZSCOREreviews': 18,
    }

    descriptive_stats_table_column_map = {
        'mean': 'Mean',
        'median': 'Median',
        'std': '\makecell[l]{Standard \\\ Deviation}',
        'min': 'Min',
        'max': 'Max',
        'count': '\makecell[l]{Total \\\ Observations}',
        '0_Count': '\makecell[l]{False \\\ Observations}',
        '1_Count': '\makecell[l]{True \\\ Observations}',
    }

    def __init__(self,
                 df,
                 initial_panel,
                 consec_panels,
                 subsample_names,
                 text_label_count_df=None,
                 panel_long_df=None,
                 panel_long_game_subsamples=None,
                 individual_dummies_df=None,
                 descriptive_stats_tables=None,
                 several_reg_results_pandas=None):
        self.df = df # df is the output of combine_imputed_deleted_missing_with_text_labels
        self.initial_panel = initial_panel
        self.consec_panels = consec_panels
        self.ssnames = subsample_names
        self.tlc_df = text_label_count_df
        self.panel_long_df = panel_long_df
        self.panel_long_game_subsamples = panel_long_game_subsamples
        self.i_dummies_df = individual_dummies_df
        self.descriptive_stats_tables = descriptive_stats_tables
        self.several_reg_results = several_reg_results_pandas

# *********************************************************************************************
# ******************* BASIC FUNCTIONS *********************************************************
# *********************************************************************************************

    def print_col_names(self, the_panel):
        vars_in_a_panel = []
        for i in self.df.columns:
            if the_panel in i:
                print(i)
                vars_in_a_panel.append(i)
        return vars_in_a_panel

    def print_unique_value_of_var_panel(self, single_var, the_panel=None):
        if the_panel is not None:
            col_name = single_var + '_' + the_panel
            unique_l = self.df[col_name].unique()
            print(col_name, 'contains', len(unique_l), 'unique values')
            print(unique_l)
            return unique_l
        else:
            col_name = [single_var+'_'+i for i in self.consec_panels]
            d = dict.fromkeys(col_name)
            for j in d.keys():
                unique_l = self.df[j].unique()
                print(j, 'contains', len(unique_l), 'unique values')
                print(unique_l)
                d[j]=unique_l
                print()
            return d

    def select_partial_vars(self, text):
        l1 = []
        for i in self.df.columns:
            if text in i:
                contains_digit = any(map(str.isdigit, i))
                if contains_digit is True: # seperate the panel information from the variable sub-string
                    x = i.split("_", -1)
                    x = x[:-1]
                    y = '_'.join(x)
                    l1.append(y)
                else: # the string column name does not contain panel informaiton (time-invariant variables)
                    l1.append(i)
        return list(set(l1))

    def select_vars(self, the_panel=None, **kwargs):
        if 'var_list' in kwargs.keys():
            variables = kwargs['var_list']
        elif 'partial_var' in kwargs.keys():
            variables = self.select_partial_vars(text=kwargs['partial_var'])
        elif 'single_var' in kwargs.keys():
            variables = [kwargs['single_var']]
        if the_panel is None:
            selected_cols = []
            for p in self.consec_panels:
                cols = [item + '_' + p for item in variables]
                selected_cols.extend(cols)
        else:
            selected_cols = [item + '_' + the_panel for item in variables]
        new_df = self.df.copy(deep=True)
        selected_df = new_df[selected_cols]
        return selected_df

    def select_panel_vars(self, time_invariant_vars, time_variant_vars, dep_vars, demaned_time_variant_vars=None):
        vars = ['full_kmeans_labels',
                'game_kmeans_labels',
                'nongame_kmeans_labels']
        vars.extend(time_invariant_vars)
        vars_2 = []
        for i in time_variant_vars:
            vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        if demaned_time_variant_vars is not None:
            for i in demaned_time_variant_vars:
                vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        for i in dep_vars:
            vars_2.extend([i + '_' + panel for panel in self.consec_panels])
        vars.extend(vars_2)
        new_df = self.df.copy(deep=True)
        new_df = new_df[vars]
        return new_df

    def peek_at_missing(self, **kwargs):
        df1 = self.select_vars(**kwargs)
        null_data = df1[df1.isnull().any(axis=1)]
        return null_data

    def replace_literal_true(self, cat_var): # after checking unique value of cat_var, some cat_var has 'True' instead of True
        cols = [cat_var+'_'+i for i in self.consec_panels]
        for j in cols:
            self.df.loc[self.df[j] == 'True', j] = True
        return self.df

    def convert_df_from_wide_to_long(self, time_variant_vars, time_invariant_vars, dep_vars, demaned_time_variant_vars=None):
        new_df = self.select_panel_vars(time_invariant_vars, time_variant_vars, dep_vars, demaned_time_variant_vars)
        new_df = new_df.reset_index()
        stub_names = copy.deepcopy(time_variant_vars)
        stub_names.extend(dep_vars)
        if demaned_time_variant_vars is not None:
            stub_names.extend(demaned_time_variant_vars)
        new_df = pd.wide_to_long(new_df, stubnames=stub_names, i="index", j="panel", sep='_') # here you can add developer for multiindex output
        new_df = new_df.sort_index()
        self.panel_long_df = new_df
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def combine_individual_dummies_to_long_panel(self):
        """
        run after self.create_individual_app_dummies()
        """
        dfl = self.panel_long_df.copy(deep=True)
        dfr = self.i_dummies_df.copy(deep=True)
        df = dfl.join(dfr, how='left')
        self.panel_long_df = df
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def create_long_df_game_subsamples(self):
        """
        run this at the very last step (after creating all the dummies,
        because I do not want to revise all other functions with this subsamples)
        even after converted from wide to long
        """
        df2 = self.panel_long_df.copy(deep=True)
        df_game = df2.loc[df2['genreIdGame'] == 1]
        df_nongame = df2.loc[df2['genreIdGame'] == 0]
        game_subsamples = {'game': df_game,
                           'none_game': df_nongame}
        self.panel_long_game_subsamples = game_subsamples
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

# *********************************************************************************************
# ******************* Descriptive Statistics **************************************************
# *********************************************************************************************
    def correlation_matrix(self, dep_vars, time_variant_vars, time_invariant_vars, the_panel):
        """
        This is for the purpose of checking multicolinearity between independent variables
        """
        vars = []
        vars.extend([i + '_' + the_panel for i in time_variant_vars])
        vars.extend(time_invariant_vars)
        vars.extend([i + '_' + the_panel for i in dep_vars])
        new_df = self.df.copy(deep=True)
        hdf = new_df[vars]
        df_corr = hdf.corr()
        return df_corr

    def text_cluster_group_count(self):
        df2 = self.df.copy(deep=True)
        label_cols = [i + '_kmeans_labels' for i in self.ssnames]
        df2 = df2[label_cols]
        count_series = []
        for i in label_cols:
            c = df2.groupby([i]).size().sort_values(ascending=False)
            count_series.append(c)
        df3 = pd.concat(count_series, axis=1)
        df3.columns = self.ssnames
        # # Only show the top 60 groups
        df4 = df3.copy(deep=True)
        df4 = df4.iloc[:30,]
        df4.columns = ['\makecell[l]{all apps}',
                       '\makecell[l]{hedonic apps}',
                       '\makecell[l]{none \\\ hedonic apps}']
        filename = self.initial_panel + '_text_cluster_label_group.tex'
        df4.to_latex(buf=regression_analysis.descriptive_stats_tables / filename,
                       multirow=True,
                       multicolumn=True,
                       caption=('Number of Apps In Top 30 Text Cluster Groups for Three Subsamples'),
                       position='h!',
                       label='table:3',
                       escape=False)
        self.tlc_df = df3
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def text_cluster_bar_chart(self, top_n):
        df2 = self.tlc_df.copy(deep=True)
        for i in self.ssnames:
            df3 = df2[[i]].sort_values(i, ascending=False)
            df3.reset_index(inplace=True)
            df3.columns = [i+' text clusters', 'number of apps']
            # -------------- plot ----------------------------------------------------------------
            fig, ax = plt.subplots()
            # color the top_n bars
            color = ['red'] * top_n
            rest = len(df3.index) - top_n
            color.extend(['blue'] * rest)
            ax = df3.plot.bar(x=i+' text clusters',
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
            df4 = df3.iloc[:top_n, ]
            for index, row in df4.iterrows():
                value = row['number of apps']
                ax.annotate(value,
                            (index, value),
                            xytext=(0, 0.1), # 2 points to the right and 15 points to the top of the point I annotate
                            textcoords='offset points')
            ax.set_xlabel("350 Text Clusters")
            ax.set_ylabel("Number of Apps")
            ax.set_title(self.initial_panel + ' ' + i + ' Text Cluster Bar Graph')
            filename = self.initial_panel + '_' + i + '_text_cluster_bar.png'
            fig.savefig(regression_analysis.descriptive_stats_graphs / filename,
                        facecolor='white',
                        dpi=300)
        return df2

    def bar_chart_a_dummy_against_dummy_or_cat(self, df, dummy1, dummy2):
        fig, ax = plt.subplots()
        df2 = df.groupby([dummy1, dummy2]).size()
        ax = df2.unstack().plot.bar(stacked=True, ax=ax)
        total_1 = 0
        total_2 = 0
        for p in ax.patches:
            if p.xy[0] == -0.25:
                total_1 += p.get_height()
            elif p.xy[0] == 0.75:
                total_2 += p.get_height()
        for p in ax.patches:
            if p.xy[0] == -0.25:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total_1)
            elif p.xy[0] == 0.75:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total_2)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height() / 2
            ax.annotate(percentage, (x, y))
        if dummy1 == 'nicheDummy':
            ax.set_title(self.initial_panel + ' Dataset' + ' Niche Dummy against ' + dummy2)
        filename = self.initial_panel + '_' + dummy1 + '_' + dummy2 + '.png'
        fig.savefig(regression_analysis.descriptive_stats_graphs / filename,
                    facecolor='white',
                    dpi=300)
        return ax

    def kde_plot_by_dummy(self, df, dummy1, continuous1):
        fig, ax = plt.subplots()
        ax = sns.kdeplot(data=df, x=continuous1, hue=dummy1,
                         fill=True, common_norm=False,
                         # palette="crest", remove palette because the color contrast is too low
                         alpha=.4, linewidth=0, ax=ax)
        if dummy1 == 'nicheDummy':
            ax.set_title(self.initial_panel + ' Dataset' + ' Niche Dummy against ' + continuous1)
        filename = self.initial_panel + '_' + dummy1 + '_' + continuous1 + '.png'
        fig.savefig(regression_analysis.descriptive_stats_graphs / filename,
                    facecolor='white',
                    dpi=300)
        return ax

    def ONEDummy_relationship_to_keyvars(self, ONEDummy, the_panel, n_niche_scale_dummies):
        """
        make sure you run self.create_nicheDummy and create_n_nichedummies before running this
        The ONEDummy is usually time-invariant, for example, nicheDummy or genreIdGame
        """
        # ----------------- select relationship with key variables -----------------------------
        df2 = self.df.copy(deep=True)
        key_vars = ['score',
                    'ratings',
                    'reviews',
                    'minInstalls',
                    'minInstallsTop',
                    'minInstallsMiddle',
                    'minInstallsBottom',
                    'CategoricalminInstalls',
                    'price',
                    'paidTrue',
                    'containsAdsTrue',
                    'offersIAPTrue']
        kvars = [i + '_' + the_panel for i in key_vars]
        time_invariants_vars = [
                     'combined_panels_kmeans_labels',
                     'combined_panels_kmeans_labels_count',
                     'genreIdGame',
                     'nicheDummy',
                     'contentRatingAdult',
                     'DaysSinceReleased']
        kvars.extend(time_invariants_vars)
        nicheScaleDummies = ['nicheScaleDummy' + str(i) for i in range(n_niche_scale_dummies)]
        kvars.extend(nicheScaleDummies)
        df4 = df2[kvars]
        # ---------------------------------------------------------------------------------------
        compare_against1 = ['nicheDummy', 'genreIdGame', 'contentRatingAdult',
                           'paidTrue_'+the_panel, 'offersIAPTrue_'+the_panel, 'containsAdsTrue_'+the_panel,
                           'CategoricalminInstalls_'+the_panel]
        compare_against1.remove(ONEDummy)
        for i in compare_against1:
            print(i)
            ax = self.bar_chart_a_dummy_against_dummy_or_cat(df4, ONEDummy, i)
        compare_against2 = ['score_'+the_panel, 'ratings_'+the_panel, 'reviews_'+the_panel]
        for i in compare_against2:
            ax = self.kde_plot_by_dummy(df4, ONEDummy, i)
        return ax

    def key_var_definition(self):
        df = pd.Series(regression_analysis.var_definition).to_frame().reset_index()
        df.columns = ['Variable', 'Definition']
        df.set_index('Variable', inplace=True)
        # -------------- convert to latex --------------------------------------------------
        filename = self.initial_panel + '_variable_definition.tex'
        df2 = df.to_latex(buf=regression_analysis.descriptive_stats_tables / filename,
                           multirow=True,
                           multicolumn=True,
                           caption=('Descriptive Statistics of Key Variables'),
                           position='h!',
                           label='table:1',
                           na_rep='',
                           escape=False)
        return df

    def add_sum_row(self, df):
        sum_row = df.sum(axis=0)
        sum_row = sum_row.to_frame().T
        sum_row.index = ['sum']
        df = pd.concat([df, sum_row], join="inner")
        return df

    def add_sum_col(self, df):
        sum_row = df.sum(axis=1)
        df['sum'] = sum_row
        return df

    def descriptive_stats_for_single_panel(self,
                                           continuous_vars,
                                           dummy_vars,
                                           cat_vars,
                                           time_invar_dum,
                                           time_invar_con,
                                           the_panel,
                                           add_sum_row_col=True):
        """
        This is must be run after self.create_new_dummies_from_cat_var to get updated self.df
        """
        # ----- Select Vars --------------------------------------------------------------
        con_vars = [i + '_' + the_panel for i in continuous_vars]
        con_vars.extend(time_invar_con)
        dum_vars = [i + '_' + the_panel for i in dummy_vars]
        dum_vars.extend(time_invar_dum)
        cat_vars = [i + '_' + the_panel for i in cat_vars]
        con_and_dum_vars = copy.deepcopy(con_vars)
        con_and_dum_vars.extend(dum_vars)
        # ----- Select DFs ---------------------------------------------------------------
        new_df = self.df.copy(deep=True)
        con_vars_df = new_df[con_vars]
        dum_vars_df = new_df[dum_vars]
        cat_vars_df = new_df[cat_vars]
        con_and_dum_df = new_df[con_and_dum_vars]
        # ----- Continuous Variables Summary Stats ---------------------------------------
        con_vars_sum_stats = con_vars_df.agg(['mean', 'std', 'min', 'median', 'max', 'count'], axis=0)
        # ----- Dummy Variables Count ----------------------------------------------------
        dum_stats_dfs = []
        for i in dum_vars:
            dum_vars_df['Count'+i] = 0
            df = dum_vars_df[[i, 'Count'+i]].groupby(i).count()
            dum_stats_dfs.append(df)
        dum_vars_sum_stats = functools.reduce(lambda a, b: a.join(b, how='inner'), dum_stats_dfs)
        if add_sum_row_col is True:
            dum_vars_sum_stats = self.add_sum_row(dum_vars_sum_stats)
        # ----- Continuous and Dummy Variables Together ----------------------------------
        con_and_dum_vars_stats = con_and_dum_df.agg(['mean', 'std', 'min', 'median', 'max', 'count'], axis=0)
        con_and_dum_vars_stats = con_and_dum_vars_stats.T
        con_and_dum_vars_stats['count'] = con_and_dum_vars_stats['count'].astype(int)
        dum_stats_dfs = []
        for i in dum_vars:
            dum_vars_df['Count_' + i] = 0
            df = dum_vars_df[[i, 'Count_' + i]].groupby(i).count()
            dum_stats_dfs.append(df)
        dum_vars_sum_stats = functools.reduce(lambda a, b: a.join(b, how='inner'), dum_stats_dfs)
        for i in dum_vars_sum_stats.columns:
            dum_vars_sum_stats.rename(columns={i: i.lstrip('Count').lstrip('_')}, inplace=True)
        for i in dum_vars_sum_stats.index:
            dum_vars_sum_stats.rename(index={i: str(i) + '_Count'}, inplace=True)
        dum_vars_sum_stats = dum_vars_sum_stats.T
        cd_sum_stats = con_and_dum_vars_stats.join(dum_vars_sum_stats, how='left')
        # ---- Categorical Variables Count -----------------------------------------------
        cat_stats_dict = dict.fromkeys(cat_vars)
        for i in cat_vars:
            cat_vars_df['Count'+i] = 0
            df = cat_vars_df[[i, 'Count'+i]].groupby(i).count()
            if 'minInstalls' in i:
                df.sort_index(inplace=True)
            else:
                df.sort_values(by='Count'+i, ascending=False, inplace=True)
            if add_sum_row_col is True:
                df = self.add_sum_row(df)
            cat_stats_dict[i] = df
        # ----- Dummy by Dummy and Dummy by Category --------------------------------------
        dummy_cat_dfs = []
        for i in dum_vars:
            sub_dummy_cat_dfs = []
            for j in cat_vars:
                df = new_df[[i, j]]
                df2 = pd.crosstab(df[i], df[j])
                df2.columns = [str(c) + '_' + the_panel for c in df2.columns]
                sub_dummy_cat_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), sub_dummy_cat_dfs)
            df3.index = [i + '_' + str(j) for j in df3.index]
            dummy_cat_dfs.append(df3)
        dummy_cat_cocat_df = functools.reduce(lambda a, b: pd.concat([a,b], join='inner'), dummy_cat_dfs)
        if add_sum_row_col is True:
            dummy_cat_cocat_df = self.add_sum_col(dummy_cat_cocat_df)
        # ----- Categorical Var by Categorical Var ----------------------------------------
        i, j = cat_vars[0], cat_vars[1]
        df = new_df[[i, j]]
        cc_df = pd.crosstab(df[i], df[j])
        cc_df.columns = [str(c) + '_' + the_panel for c in cc_df.columns]
        cc_df.index = [str(c) + '_' + the_panel for c in cc_df.index]
        if add_sum_row_col is True:
            cc_df = self.add_sum_row(cc_df)
            cc_df = self.add_sum_col(cc_df)
        # ----- Continuous Variables by Dummy ---------------------------------------------
        continuous_by_dummies = dict.fromkeys(con_vars)
        for i in con_vars:
            sub_groupby_dfs = []
            for j in dum_vars:
                df = new_df[[i, j]]
                agg_func_math = {i:['mean', 'std', 'min', 'median', 'max', 'count']}
                df2 = df.groupby([j]).agg(agg_func_math, axis=0)
                df2.columns = ['mean', 'std', 'min', 'median', 'max', 'count']
                sub_groupby_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: pd.concat([a, b], join='inner'), sub_groupby_dfs)
            df3.index = [j + '_' + str(z) for z in df3.index]
            continuous_by_dummies[i] = df3
        # ----- Continuous Variables by Categorical ---------------------------------------
        groupby_cat_dfs = dict.fromkeys(cat_vars)
        for i in cat_vars:
            sub_groupby_dfs = []
            for j in con_vars:
                df = new_df[[i, j]]
                agg_func_math = {j:['mean', 'std', 'min', 'median', 'max', 'count']}
                df2 = df.groupby([i]).agg(agg_func_math, axis=0)
                sub_groupby_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), sub_groupby_dfs)
            df3.index = [str(c) + '_' + the_panel for c in df3.index]
            groupby_cat_dfs[i] = df3
        # ----- Update Instance Attributes ------------------------------------------------
        self.descriptive_stats_tables = {'continuous_vars_stats': con_vars_sum_stats,
                                         'dummy_vars_stats': dum_vars_sum_stats,
                                         'continuous_and_dummy_vars_stats': cd_sum_stats,
                                         'categorical_vars_count': cat_stats_dict,
                                         'crosstab_dummy_categorical_vars': dummy_cat_cocat_df,
                                         'crosstab_two_categorical_vars': cc_df,
                                         'continuous_vars_by_dummies': continuous_by_dummies,
                                         'continuous_vars_by_categorical': groupby_cat_dfs}
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def customize_and_output_descriptive_stats_pandas_to_latex(self, the_panel):
        """
        :param df_dict: self.descriptive_stats_tables, the output of self.descriptive_stats_for_single_panel()
        since 'continuous_and_dummy_vars_stats' already included all the variables of interest to show their summary stats
        so I will not select more varibales.
        :return:
        """
        df2 = self.descriptive_stats_tables['continuous_and_dummy_vars_stats'].copy(deep=True)
        # -------------- round -------------------------------------------------------------
        for i in df2.columns:
            if i not in ['1_Count', '0_Count', 'count']:
                df2[i] = df2[i].astype(float).round(decimals=2)
        # for i in df2.columns:
        #     if i in ['DaysSinceReleased', 'reviews_'+the_panel]:
        #         df2[i] = df2[i].apply(lambda x: int(x) if not math.isnan(x) else x)
        # df2 = df2.T
        # -------------- adjust the order of rows and columns to display --------------------
        def set_row_order(x, the_panel):
            if the_panel in x:
                x = x.rstrip(the_panel).rstrip('_')
            for k in regression_analysis.descriptive_stats_table_row_order.keys():
                if k == x:
                    return regression_analysis.descriptive_stats_table_row_order[k]
        df2 = df2.reset_index()
        df2.rename(columns={'index': 'Variable'}, inplace=True)
        df2['row_order'] = df2['Variable'].apply(lambda x: set_row_order(x, the_panel))
        df2.sort_values(by='row_order', inplace=True)
        df2.set_index('Variable', inplace=True)
        df2.drop(['row_order'], axis=1, inplace=True)
        df2 = df2[['mean', 'std', 'min', 'median', 'max', '1_Count', '0_Count', 'count']]
        # -------------- change row and column names ---------------------------------------
        for i in df2.columns:
            for j in regression_analysis.descriptive_stats_table_column_map.keys():
                if j == i:
                    df2.rename(columns={i: regression_analysis.descriptive_stats_table_column_map[j]}, inplace=True)
        def set_row_names(x, the_panel):
            if the_panel in x:
                x = x.rstrip(the_panel).rstrip('_')
            for j in regression_analysis.var_latex_map.keys():
                if j == x:
                    return regression_analysis.var_latex_map[j]
        df2 = df2.reset_index()
        df2['Variable'] = df2['Variable'].apply(lambda x: set_row_names(x, the_panel))
        df2 = df2.set_index('Variable')
        # -------------- convert to latex --------------------------------------------------
        filename = self.initial_panel + '_descriptive_stats_for_' + the_panel + '.tex'
        df3 = df2.to_latex(buf=regression_analysis.descriptive_stats_tables / filename,
                           multirow=True,
                           multicolumn=True,
                           caption=('Descriptive Statistics of Key Variables'),
                           position='h!',
                           label='table:2',
                           na_rep='',
                           escape=False)
        return df2

# *********************************************************************************************
# ****************************** Regression ***************************************************
# *********************************************************************************************
    """
    # http://www.data-analysis-in-python.org/t_statsmodels.html
    # https://towardsdatascience.com/a-guide-to-panel-data-regression-theoretics-and-implementation-with-python-4c84c5055cf8
    # https://bashtage.github.io/linearmodels/doc/panel/models.html
    """
    def regression(self, dep_var, time_variant_vars, time_invariant_vars,
                   cross_section, reg_type, the_panel=None):
        """
        run convert_df_from_wide_to_long first and get self.panel_long_df updated, then run this method
        https://bashtage.github.io/linearmodels/doc/panel/models.html
        I have observed that using demeaned time variant independent variables in POOLED_OLS generats the same
        coefficient estimates (for time variant) as the ones from FE model with un-demeaned data.
        The dependent variable is better to interpret as staying un-demeaned.
        The coefficient of time invariant variables are then interpreted as when all other time variant variables are
        set to their group specific means.
        The weird thing, why does niche_app still get appear in FE model？
        """
        if cross_section is True:
            x_vars = [i + '_' + the_panel for i in time_variant_vars]
            x_vars.extend(time_invariant_vars)
            new_df = self.df.copy(deep=True)
            independents_df = new_df[x_vars]
            X = sm.add_constant(independents_df)
            dep_var = dep_var + '_' + the_panel
            y = new_df[[dep_var]]
            if reg_type == 'OLS':
                model = sm.OLS(y, X)
                results = model.fit()
                results = results.get_robustcov_results()
            elif reg_type == 'logit':
                model = sm.Logit(y, X)
                results = model.fit()
            elif reg_type == 'probit':
                model = sm.Probit(y, X)
                results = model.fit()
        else: # panel regression models
            x_vars = []
            x_vars.extend(time_variant_vars)
            x_vars.extend(time_invariant_vars)
            if reg_type == 'PPOLED_OLS_with_individual_dummies':
                x_vars.extend(self.i_dummies_df.columns)
            new_df = self.panel_long_df.copy(deep=True)
            independents_df = new_df[x_vars]
            X = sm.add_constant(independents_df)
            y = new_df[[dep_var]]
            # https://bashtage.github.io/linearmodels/panel/panel/linearmodels.panel.model.PanelOLS.html
            if reg_type == 'POOLED_OLS':
                model = PooledOLS(y, X)
                results = model.fit(cov_type='clustered', cluster_entity=True)
            elif reg_type == 'PPOLED_OLS_with_individual_dummies':
                model = PooledOLS(y, X)
                results = model.fit(cov_type='clustered', cluster_entity=True)
            elif reg_type == 'FE': # niche_type will be abosrbed in this model because it is time-invariant
                model = PanelOLS(y, X,
                                 drop_absorbed=True,
                                 entity_effects=True,
                                 time_effects=True)
                results = model.fit(cov_type = 'clustered', cluster_entity = True)
            elif reg_type == 'RE':
                model = RandomEffects(y, X,)
                results = model.fit(cov_type = 'clustered', cluster_entity = True)
        return results

    def several_regressions(self, dep_vars, time_variant_vars, time_invariant_vars, cross_section, reg_types, the_panel=None):
        results_dict = dict.fromkeys(reg_types, dict.fromkeys(dep_vars))
        for reg_type in reg_types:
            for dep_var in dep_vars:
                result = self.regression(
                    dep_var,
                    time_variant_vars,
                    time_invariant_vars,
                    cross_section,
                    reg_type,
                    the_panel)
                results_dict[reg_type][dep_var] = result
        return results_dict

    def compare_several_panel_reg_results(self, results_dict, dep_vars):
        """
        https://bashtage.github.io/linearmodels/panel/examples/examples.html
        results_dict is the output of several_regressions
        """
        panels_results = dict.fromkeys(dep_vars, {})
        panels_compare = dict.fromkeys(dep_vars)
        for reg_type, results in results_dict.items():
            if reg_type in ['FE', 'RE', 'POOLED_OLS', 'PPOLED_OLS_with_individual_dummies']:
                for dep_var in results.keys():
                    panels_results[dep_var][reg_type] = results[dep_var]
                    compared_results = compare(panels_results[dep_var])
                    print(compared_results)
                    panels_compare[dep_var] = compared_results
        return panels_compare

    def compile_several_reg_results_into_pandas(self, panels_compare):
        """
        panel_compare is the output of self.compare_several_panel_reg_results(self, results_dict)
        """
        def change_index_name(df, end_fix):
            dfn = df.copy(deep=True)
            for i in dfn.index:
                dfn.rename(index={i: i+'_'+end_fix}, inplace=True)
            return dfn
        df5 = []
        for dep_var, compare in panels_compare.items():
            df_params = change_index_name(compare.params, end_fix='coef')
            df_tstats = change_index_name(compare.tstats, end_fix='tstats')
            df_pvalues = change_index_name(compare.pvalues, end_fix='pvalues')
            df1 = [df_params, df_tstats, df_pvalues]
            df2 = functools.reduce(lambda a, b: pd.concat([a, b], axis=0), df1)
            df3 = [df2,
                   compare.f_statistic.T,
                   compare.rsquared.to_frame().T,
                   compare.nobs.to_frame().T,
                   compare.cov_estimator.to_frame().T]
            df4 = functools.reduce(lambda a, b: pd.concat([a, b], axis=0), df3)
            for i in df4.columns:
                df4.rename(columns={i: dep_var+'_'+i}, inplace=True)
            df5.append(df4)
        df6 = functools.reduce(lambda a, b: a.join(b, how='inner'), df5)
        # round the entire dataframe
        df7 = df6.T
        for i in df7.columns:
            if i not in ['_cov_type', 'nobs']:
                df7[i] = df7[i].astype(float).round(decimals=3)
            elif i == 'nobs':
                df7[i] = df7[i].astype(int)
        df8 = df7.T
        return df8

    def customize_reg_results_pandas_before_output_latex(self,
                                             df,
                                             selective_dep_vars,
                                             selective_reg_types,
                                             p_value_as_asterisk=False):
        """
        :param df6: the output of self.compile_several_reg_results_into_pandas(panels_compare)
        :return: df8: the customized version of pandas that will ultimately transformed into latex
        """
        cols_to_keep = [i + '_' + j for j in selective_reg_types for i in selective_dep_vars]
        df7 = df.copy(deep=True)
        df8 = df7[cols_to_keep]
        if p_value_as_asterisk is True:
            df9 = df8.T
        for j in df9.columns:
            if '_pvalues' in j:
                df9[j + '<0.01'] = df9[j].apply(lambda x: '***' if x < 0.01 else 'not sig at 1%')
                df9[j + '<0.05'] = df9[j].apply(lambda x: '**' if x < 0.05 else 'not sig at 5%')
                df9[j + '<0.1'] = df9[j].apply(lambda x: '*' if x < 0.1 else 'not sig at 10%')
        ind_vars_sig_at_1_percent = []
        ind_vars_sig_at_5_percent = []
        ind_vars_sig_at_10_percent = []
        for j in df9.columns:
            if '***' in df9[j].values:
                ind_var = j.rstrip('pvalues<0.01').rstrip('_')
                ind_vars_sig_at_1_percent.append(ind_var)
            elif '**' in df9[j].values:
                ind_var = j.rstrip('pvalues<0.05').rstrip('_')
                ind_vars_sig_at_5_percent.append(ind_var)
            elif '*' in df9[j].values:
                ind_var = j.rstrip('pvalues<0.1').rstrip('_')
                ind_vars_sig_at_10_percent.append(ind_var)
        for i in ind_vars_sig_at_1_percent:
            ind_vars_sig_at_5_percent.remove(i)
            ind_vars_sig_at_10_percent.remove(i)
            df9[i + '_coef'] = df9.apply(
                        lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.01'] if row[i + '_pvalues<0.01'] != 'not sig at 1%' else str(row[i + '_coef']),
                         axis=1)
        for i in ind_vars_sig_at_5_percent:
            ind_vars_sig_at_10_percent.remove(i)
            df9[i + '_coef'] = df9.apply(
                        lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.05'] if row[i + '_pvalues<0.05'] != 'not sig at 5%' else str(row[i + '_coef']),
                        axis=1)
        for i in ind_vars_sig_at_10_percent:
            df9[i + '_coef'] = df9.apply(
                        lambda row: str(row[i + '_coef']) + row[i + '_pvalues<0.1'] if row[i + '_pvalues<0.1'] != 'not sig at 10%' else str(row[i + '_coef']),
                        axis=1)
        cols_to_keep = [i for i in df9.columns if '_coef' in i]
        cols_to_keep.extend(['F stat', 'P-value', 'rsquared', 'nobs', '_cov_type'])
        df10 = df9[cols_to_keep]
        df10 = df10.T
        return df10

    def output_reg_results_pandas_to_latex(self, df, the_reg_type):
        """
        :param df: the output of self.customize_pandas_before_output_latex()
        :return: df10:
        """
        df2 = df.copy(deep=True)
        # ------------ rename columns ---------------------------------------------
        for i in df2.columns:
            z = i.rstrip(the_reg_type).rstrip('_')
            for j in regression_analysis.var_latex_map.keys():
                if j == z:
                    df2.rename(columns={i: regression_analysis.var_latex_map[j]}, inplace=True)
        # ------------ prepare column for multiindex rows --------------------------
        df2 = df2.reset_index()
        def set_row_level_0(x):
            if x in ['niche_app_coef']:
                return '\makecell[l]{Niche \\\ Indicators}'
            elif x in ['const_coef', 'genreIdGame_coef', 'contentRatingAdult_coef', 'DaysSinceReleased_coef']:
                return '\makecell[l]{Time-invariant \\\ Variables}'
            elif x in ['DeMeanedscore_coef', 'DeMeanedZSCOREreviews_coef', 'DeMeanedminInstallsTop_coef', 'DeMeanedminInstallsMiddle_coef']:
                return '\makecell[l]{Time-variant \\\ Variables}'
            else:
                return '\makecell[l]{Regression \\\ Statistics}'
        df2['Variable Groups'] = df2['index'].apply(lambda x: set_row_level_0(x))
        # ------------ rename rows -------------------------------------------------
        def set_row_level_1(x):
            if '_coef' in x:
                x = x.rstrip('coef').rstrip('_')
            for i in regression_analysis.var_latex_map.keys():
                if i == x:
                    return regression_analysis.var_latex_map[i]
        df2['Independent Variables'] = df2['index'].apply(lambda x: set_row_level_1(x))
        # manually set the order of rows you want to present in final latex table
        def set_row_order_level_0(x):
            if x == '\makecell[l]{Niche \\\ Indicators}':
                return 0
            elif x == '\makecell[l]{Time-invariant \\\ Variables}':
                return 1
            elif x == '\makecell[l]{Time-variant \\\ Variables}':
                return 2
            elif x == '\makecell[l]{Regression \\\ Statistics}':
                return 3
        df2['row_order_level_0'] = df2['Variable Groups'].apply(lambda x: set_row_order_level_0(x))
        df2.sort_values(by='row_order_level_0', inplace=True)
        df2.set_index(['Variable Groups', 'Independent Variables'], inplace=True)
        df2.drop(['index', 'row_order_level_0'], axis=1, inplace=True)
        df2.columns = pd.MultiIndex.from_product([['pooled OLS with demeaned time-variant variables'],
                                                  df2.columns.tolist()])
        # -------------------- save and output to latex -------------------------------
        filename = self.initial_panel + '_POOLED_OLS_demeaned.tex'
        df3 = df2.to_latex(buf=regression_analysis.reg_table_path / filename,
                           multirow=True,
                           multicolumn=True,
                           caption=('Regression Results'),
                           position='h!',
                           label='table:2',
                           escape=False)
        return df2

# *********************************************************************************************
# *********** create categorical variables and time-invariant variables ***********************
# *********************************************************************************************

    def cat_var_count(self, cat_var, the_panel=None):
        if the_panel is not None:
            col_name = cat_var + '_' + the_panel
            rd = self.df.groupby(col_name)['count_'+the_panel].count()
            if cat_var == 'minInstalls': # minInstalls should not be sorted by the number of apps in each group, rather by index
                rd = rd.sort_index(ascending=False)
            else:
                rd = rd.sort_values(ascending=False)
            print(rd)
            return rd
        else:
            col_name = [cat_var+'_'+i for i in self.consec_panels]
            df_list = []
            for j in range(len(col_name)):
                rd = self.df.groupby(col_name[j])['count_'+self.consec_panels[j]].count()
                if cat_var == 'minInstalls':
                    rd = rd.sort_index(ascending=False)
                else:
                    rd = rd.sort_values(ascending=False)
                rd = rd.to_frame()
                df_list.append(rd)
            dfn = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
            print(dfn)
            return dfn

    def find_time_variant_rows(self, cat_var):
        df2 = self.select_vars(single_var=cat_var)
        df_time_variant = []
        for index, row in df2.iterrows():
            row_time_variant = []
            for j in range(len(df2.columns) - 1):
                if row[df2.columns[j]] == row[df2.columns[j + 1]]:
                    row_time_variant.append(False)
                else:
                    row_time_variant.append(True)
            if any(row_time_variant) is True:
                df_time_variant.append(True)
            else:
                df_time_variant.append(False)
        time_variant_df = df2[df_time_variant]
        time_variant_appids = time_variant_df.index.tolist()
        return time_variant_df, time_variant_appids

    def change_time_variant_to_invariant(self, cat_var):
        """
        use the last panel as the standard value for time-invariant variables
        """
        time_variant_df, time_variant_appids = self.find_time_variant_rows(cat_var=cat_var)
        col_names = [cat_var + '_' + i for i in self.consec_panels]
        for i in time_variant_appids:
            for j in col_names:
                self.df.at[i, j] = self.df.at[i, cat_var+'_' + self.consec_panels[-1]] # this one intends to change class attributes
        return self.df

    def create_new_dummies_from_cat_var(self, cat_var, time_invariant=False):
        if time_invariant is True:
            self.df = self.change_time_variant_to_invariant(cat_var)
        else:
            pass
        if cat_var == 'contentRating':
            df1 = self.select_vars(single_var=cat_var)
            if time_invariant is True:
                df1['contentRatingAdult'] = df1['contentRating_' + self.consec_panels[-1]].apply(
                    lambda x: 0 if 'Everyone' in x else 1)
            else:
                for i in self.consec_panels:
                    df1['contentRatingAdult_'+i] = df1['contentRating_'+i].apply(lambda x: 0 if 'Everyone' in x else 1)
            dcols = ['contentRating_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        elif cat_var == 'minInstalls':
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1['minInstallsTop_'+i] = df1['minInstalls_'+i].apply(lambda x: 1 if x >= 1.000000e+07 else 0)
                df1['minInstallsMiddle_' + i] = df1['minInstalls_' + i].apply(lambda x: 1 if x < 1.000000e+07 and x >= 1.000000e+04 else 0)
                df1['minInstallsBottom_' + i] = df1['minInstalls_' + i].apply(lambda x: 1 if x < 1.000000e+04 else 0)
            dcols = ['minInstalls_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        elif cat_var == 'free':
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1['paidTrue_' + i] = df1['free_' + i].apply(lambda x: 1 if x is False else 0)
            dcols = ['free_' + i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        else:
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1[cat_var + 'True_' + i] = df1[cat_var + '_' + i].apply(lambda x: 1 if x is True else 0)
            dcols = [cat_var + '_' + i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        return regression_analysis(df=self.df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels)

    def create_categorical_from_exhaustive_dummies(self, me_dummys):
        me_dummys_p = [i + '_' + p for p in self.consec_panels for i in me_dummys]
        df2 = self.df.copy(deep=True)
        df3 = df2[me_dummys_p]
        if me_dummys == ['minInstallsTop', 'minInstallsMiddle', 'minInstallsBottom']:
            for p in self.consec_panels:
                df3.loc[df3['minInstallsTop'+'_'+p] == 1, 'CategoricalminInstalls'+'_'+p] = 'Top'
                df3.loc[df3['minInstallsMiddle'+'_'+p] == 1, 'CategoricalminInstalls'+'_'+p] = 'Middle'
                df3.loc[df3['minInstallsBottom'+'_'+p] == 1, 'CategoricalminInstalls'+'_'+p] = 'Bottom'
        df3.drop(me_dummys_p, axis=1, inplace=True)
        self.df = self.df.join(df3, how='inner')
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def create_NicheDummy(self, top_n):
        """
        make sure to run this after self.text_cluster_group_count()
        """
        df2 = self.tlc_df.copy(deep=True)
        broad_labels_dict = dict.fromkeys(self.ssnames)
        for i in self.ssnames:
            df3 = df2[[i]].sort_values(i, ascending=False)
            top = df3.iloc[:top_n, ]
            broad_labels_dict[i] = list(top.index.values)
        for i in self.ssnames:
            self.df[i+'NicheDummy'] = self.df[i+'_kmeans_labels'].apply(lambda x: 0 if x in broad_labels_dict[i] else 1)
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def create_n_NicheDummies(self, n):
        """
        note that here 1 does not necessarily means being niche, and 0 means being broad.
        In the create nicheDummy, 1 indeed means being niche and 0 means being broad because I defined the top 3 largest labels as broad.
        Here, 1 only means the app belongs to the top nth largest group of labels. So nicheScaleDummy0 == 1 could mean a broad app, while
        nicheScaleDummy19 == 1 definitely means being a niche app.
        """
        df2 = self.tlc_df.copy(deep=True)
        for i in self.ssnames:
            df3 = df2[[i]].sort_values(i, ascending=False)
            x = round(len(df3) / n)
            frames = [df3.iloc[j * x:(j + 1) * x].copy() for j in range(n - 1)]
            last_df = df3.iloc[(n - 1) * x : len(df3)]
            frames.extend([last_df])
            labels = [list(dff.index.values) for dff in frames]
            for z in range(n):
                self.df[i + 'NicheScaleDummy' + str(z)] = self.df[i+'_kmeans_labels'].apply(
                    lambda x: 1 if x in labels[z] else 0)
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def create_individual_app_dummies(self):
        df = self.df.copy(deep=True)
        df.reset_index(inplace=True)
        df['appId'] = df['index']
        df.set_index('index', inplace=True)
        dummies = df[['appId']]
        dummies = pd.get_dummies(dummies, columns=['appId'], drop_first=True)
        self.i_dummies_df = dummies
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def create_demean_time_variant_vars(self, time_variant_vars):
        """
        Because individual dummies regression takes too much time, I decide use this for FE, so that I could also include time invariant variables.
        """
        df = self.df.copy(deep=True)
        dfs = []
        for i in time_variant_vars:
            ts_i = [i + '_' + p for p in self.consec_panels]
            sub_df = df[ts_i]
            sub_df['PanelMean'+i] = sub_df.mean(axis=1)
            for p in self.consec_panels:
                sub_df['DeMeaned'+i+'_'+p] = sub_df[i+'_'+p] - sub_df['PanelMean'+i]
            ts_idm = ['DeMeaned' + i + '_' + p for p in self.consec_panels]
            dfs.append(sub_df[ts_idm])
        df_new = functools.reduce(lambda a, b: a.join(b, how='inner'), dfs)
        self.df = self.df.join(df_new, how='inner')
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def count_number_of_days_since(self, var):
        """
        :param var: time invariant independent variables, could either be released or updated
        :return: a new variable which is the number of days between today() and the datetime
        """
        df2 = self.select_vars(single_var=var)
        if var == 'released':
            df2 = df2[df2.columns[0]] # since released is time-invariant variable, so just pick a single column
            df2 = df2.rename('released').to_frame()
            df2['DaysSinceReleased'] = pd.Timestamp.now().normalize() - df2['released']
            df2['DaysSinceReleased'] = df2['DaysSinceReleased'].apply(lambda x: int(x.days))
        self.df = self.df.join(df2, how='inner')
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def standardize_continuous_vars(self, con_var, method):
        vars = [con_var + '_' + i for i in self.consec_panels]
        new_df = self.df.copy(deep=True)
        df2 = new_df[vars]
        print('before standardization:')
        for i in df2.columns:
            print(i)
            print(df2[i].describe())
            print()
        if method == 'zscore':
            scaler = preprocessing.StandardScaler()
            df3 = scaler.fit_transform(df2)
            df3 = pd.DataFrame(df3)
            scaled_col_names = ['ZSCORE' + i for i in vars]
            df3.columns = scaled_col_names
            df3.index = df2.index.tolist()
        print('after standardization:')
        for i in df3.columns:
            print(i)
            print(df3[i].describe())
            print()
        self.df = self.df.join(df3, how='inner')
        return regression_analysis(df=self.df,
                                   initial_panel=self.initial_panel,
                                   consec_panels=self.consec_panels,
                                   subsample_names=self.ssnames,
                                   text_label_count_df=self.tlc_df,
                                   panel_long_df=self.panel_long_df,
                                   panel_long_game_subsamples=self.panel_long_game_subsamples,
                                   individual_dummies_df=self.i_dummies_df)

    def check_whether_cat_dummies_are_mutually_exclusive(self, the_panel, **kwargs):
        df1 = self.select_vars(the_panel=the_panel, **kwargs)
        df1['sum_dummies'] = df1.sum(axis=1)
        l = df1.sum_dummies.unique()
        print('The unique value of all dummy columns summation should only be 1 if those dummies are mutually exclusive')
        print(l)
        if len(l) > 1:
            df2 = df1.loc[df1['sum_dummies']!=1]
            return df2
        else:
            return df1




