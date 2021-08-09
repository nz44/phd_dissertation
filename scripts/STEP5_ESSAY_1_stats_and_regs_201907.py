import pandas as pd
from pathlib import Path
import pickle
import copy
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from sklearn import preprocessing
import statsmodels.api as sm
from linearmodels import PooledOLS
from linearmodels import RandomEffects
from datetime import datetime
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")


class essay_1_stats_and_regs_201907:
    """2021 July 18
    This is the new version written based on the STEP10_ESSAY_2_3_Long_Table_Prep.py
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

    panel_essay_1_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_1_panels___')
    main_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_1___')
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
    all_y_reg_vars = ['LogImputedprice',
                      'LogImputedminInstalls',
                      'offersIAPTrue',
                      'containsAdsTrue']
    graph_dep_vars_descriptive_stats_names = {
        'LogImputedminInstalls': 'Log Minimum Installs',
        'LogImputedprice': 'Log Price',
        'both_IAP_and_ADS': 'Percentage of Apps that Offers IAP and Contains Ads'
    }
    dep_vars_reg_table_names = {
        'LogImputedprice': 'Log Price',
        'LogImputedminInstalls': 'Log Minimum Installs',
        'containsAdsTrue': 'Contains Ads',
        'offersIAPTrue': 'Offers IAP'
    }
    combo_name12_reg_table_names = {'full_full': 'Full',
                                    'minInstalls_Tier1' : 'Minimum Installs \nTier 1',
                                    'minInstalls_Tier2': 'Minimum Installs \nTier 2',
                                    'minInstalls_Tier3': 'Minimum Installs \nTier 3',
                                    'categories_category_GAME': 'Gaming \nApps',
                                    'categories_category_SOCIAL': 'Social \nApps',
                                    'categories_category_LIFESTYLE': 'Lifestyle \nApps',
                                    'categories_category_MEDICAL': 'Medical \nApps',
                                    'categories_category_BUSINESS': 'Business \nApps',
                                    'genreId_ART_AND_DESIGN': 'Art And Design',
                                    'genreId_COMICS': 'Comics',
                                    'genreId_PERSONALIZATION': 'Personalization',
                                    'genreId_PHOTOGRAPHY': 'Photography',
                                    'genreId_AUTO_AND_VEHICLES': 'Auto And Vehicles',
                                    'genreId_GAME_ROLE_PLAYING': 'Game Role Playing',
                                    'genreId_GAME_ACTION': 'Game Action',
                                    'genreId_GAME_RACING': 'Game Racing',
                                    'genreId_TRAVEL_AND_LOCAL': 'Travel And Local',
                                    'genreId_GAME_ADVENTURE': 'Game Adventure',
                                    'genreId_SOCIAL': 'Social',
                                    'genreId_GAME_SIMULATION': 'Game Simulation',
                                    'genreId_LIFESTYLE': 'Lifestyle',
                                    'genreId_EDUCATION': 'Education',
                                    'genreId_BEAUTY': 'Beauty',
                                    'genreId_GAME_CASUAL': 'Game Casual',
                                    'genreId_BOOKS_AND_REFERENCE': 'Books And Reference',
                                    'genreId_BUSINESS': 'Business',
                                    'genreId_FINANCE': 'Finance',
                                    'genreId_GAME_STRATEGY': 'Game Strategy',
                                    'genreId_SPORTS': 'Sports',
                                    'genreId_COMMUNICATION': 'Communication',
                                    'genreId_DATING': 'Dating',
                                    'genreId_ENTERTAINMENT': 'Entertainment',
                                    'genreId_GAME_BOARD': 'Game Board',
                                    'genreId_EVENTS': 'Events',
                                    'genreId_SHOPPING': 'Shopping',
                                    'genreId_FOOD_AND_DRINK': 'Food And Drink',
                                    'genreId_HEALTH_AND_FITNESS': 'Health And Fitness',
                                    'genreId_HOUSE_AND_HOME': 'House And Home',
                                    'genreId_TOOLS': 'Tools',
                                    'genreId_LIBRARIES_AND_DEMO': 'Libraries And Demo',
                                    'genreId_MAPS_AND_NAVIGATION': 'Maps And Navigation',
                                    'genreId_MEDICAL': 'Medical',
                                    'genreId_MUSIC_AND_AUDIO': 'Music And Audio',
                                    'genreId_NEWS_AND_MAGAZINES': 'News And Magazines',
                                    'genreId_PARENTING': 'Parenting',
                                    'genreId_GAME_PUZZLE': 'Game Puzzle',
                                    'genreId_VIDEO_PLAYERS': 'Video Players',
                                    'genreId_PRODUCTIVITY': 'Productivity',
                                    'genreId_WEATHER': 'Weather',
                                    'genreId_GAME_ARCADE': 'Game Arcade',
                                    'genreId_GAME_CASINO': 'Game Casino',
                                    'genreId_GAME_CARD': 'Game Card',
                                    'genreId_GAME_EDUCATIONAL': 'Game Educational',
                                    'genreId_GAME_MUSIC': 'Game Music',
                                    'genreId_GAME_SPORTS': 'Game Sports',
                                    'genreId_GAME_TRIVIA': 'Game Trivia',
                                    'genreId_GAME_WORD': 'Game Word',
                                    'starDeveloper_top_digital_firms': 'Apps from \nTop Tier \nFirms',
                                    'starDeveloper_non-top_digital_firms': 'Apps from \nNon-top Tier \nFirms'}
    @property
    def ssnames(self):
        d = self._open_predicted_labels_dict()
        res = dict.fromkeys(d.keys())
        for name1, content1 in d.items():
            res[name1] = list(content1.keys())
        return res

    @property
    def graph_combo_ssnames(self):
        res = dict.fromkeys(['combo1', 'combo2', 'combo3', 'combo4'])
        combo_contains = {'combo1': ['full', 'minInstalls'],
                          'combo2': ['full', 'starDeveloper'],
                          'combo3': ['full', 'categories'],
                          'combo4': ['genreId']}
        for combo_name, content1 in combo_contains.items():
            l = []
            for name1 in content1:
                for name2 in self.ssnames[name1]:
                    l.append(name1 + '_' + name2)
            res[combo_name] = l
        return res

    @classmethod
    def _open_imputed_deleted_divided_df(cls):
        f_name = cls.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = cls.panel_essay_1_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    @classmethod
    def _open_predicted_labels_dict(cls):
        f_name = cls.initial_panel + '_predicted_labels_dict.pickle'
        q = cls.panel_essay_1_path / 'predicted_text_labels' / f_name
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

    @classmethod
    def _open_app_level_text_cluster_stats(cls):
        filename = cls.initial_panel + '_dict_app_level_text_cluster_stats.pickle'
        q = cls.panel_essay_1_path / 'app_level_text_cluster_stats' / filename
        with open(q, 'rb') as f:
            d = pickle.load(f)
        return d

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
                title = cls.initial_panel + ' ' + cls.graph_subsample_title_dict[name1+'_'+name2] + ' \n' + graph_title
            else:
                title = cls.initial_panel + ' ' + graph_title
            title = title.title()
            fig.suptitle(title, fontsize='medium')
        # ------------ save ------------------------------------------------------------------------------
        filename = cls.initial_panel + '_' + name1 + '_' + name2 + '_' + file_keywords + '.png'
        fig.savefig(cls.des_stats_graphs_essay_1 / relevant_folder_name / filename,
                    facecolor='white',
                    dpi=300)
    # --------------------------------------------------------------------------------------------------------
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
        q = self.panel_essay_1_path / 'cross_section_dfs' / filename
        with open(q, 'rb') as f:
            self.cdf = pickle.load(f)
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def save_cross_section_reg_df(self):
        filename = self.initial_panel + '_cross_section_df.pickle'
        q = essay_1_stats_and_regs_201907.panel_essay_1_path / 'cross_section_dfs' / filename
        pickle.dump(self.cdf, open(q, 'wb'))
        print(self.initial_panel, ' SAVED CROSS SECTION DFS. ')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

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

    def _cross_section_reg_get_xy_var_list(self, name1, name2, y_var, the_panel):
        """
        :param y_var: 'LogImputedprice','LogImputedminInstalls','offersIAPTrue','containsAdsTrue'
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
        """
        # check the correlation among variables
        dfcorr = df.corr(method='pearson').round(2)
        print('The correlation table of the cross section regression dataframe is:')
        print(dfcorr)
        print()
        all_vars = df.columns.values.tolist()
        # y_var is a string without panel substring
        for i in all_vars:
            if y_var in i:
                all_vars.remove(i)
        independents_df = df[all_vars]
        X = sm.add_constant(independents_df)
        y = df[[y_var + '_' + the_panel]]
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
                        reg_results[i][name1][name2] = self._cross_section_regression(
                                              y_var=y_var,
                                              df=df,
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
                    reg_results[name1][name2] = self._panel_reg_pooled_ols(
                                                                  y_var=y_var,
                                                                  df=ldf)
        else:
            reg_results = {}
        self._extract_and_save_panel_results(result=reg_results, y_var=y_var)
        return reg_results

    def reg_for_all_subsamples_for_all_y_vars(self, reg_type):
        res = dict.fromkeys(self.all_y_reg_vars)
        for y in self.all_y_reg_vars:
            res[y] = self._reg_for_all_subsamples_for_single_y_var(reg_type=reg_type, y_var=y)
        self.reg_results = res
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _extract_and_save_panel_results(self, result, y_var):
        for name1, content1 in self.ssnames.items():
            for name2 in content1:
                # ---------- get the coefficients ----------------------
                x = result[name1][name2].params
                x = x.to_frame()
                y = x.loc[[
                    'const',
                     name1 + '_' + name2 + '_NicheDummy',
                    'PostDummy',
                     name1 + '_' + name2 + '_PostXNicheDummy']]
                # ---------- get the pvalues ---------------------------
                z1 = result[name1][name2].pvalues
                z1 = z1.to_frame()
                z2 = z1.loc[[
                    'const',
                     name1 + '_' + name2 + '_NicheDummy',
                    'PostDummy',
                     name1 + '_' + name2 + '_PostXNicheDummy']]
                y2 = y.join(z2, how='inner')
                y2 = y2.round(3)
                filename = y_var + '_' + name1 + '_' + name2 + '_POOLED_OLS.csv'
                y2.to_csv(self.main_path / 'reg_results_tables' / filename)
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

    def _put_panel_results_into_pandas_for_single_y_var(self, y_var):
        """
        :param result: is the output of self._reg_for_all_subsamples(
            reg_type='panel_pooled_ols',
            y_var=any one of ['LogImputedprice', 'LogImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue'])
            the documentation of the PanelResult class (which result is)
        :return:
        """
        # ============= 1. extract results info and put them into dicts ==================
        params_pvalues_dict = dict.fromkeys(self.ssnames.keys())
        for name1, content1 in self.ssnames.items():
            params_pvalues_dict[name1] = dict.fromkeys(content1)
            for name2 in content1:
                # ---------- get the coefficients ----------------------
                x = self.reg_results[y_var][name1][name2].params
                x = x.to_frame()
                y = x.loc[[
                    'const',
                     name1 + '_' + name2 + '_NicheDummy',
                    'PostDummy',
                     name1 + '_' + name2 + '_PostXNicheDummy']]
                # ---------- get the pvalues ---------------------------
                z1 = self.reg_results[y_var][name1][name2].pvalues
                z1 = z1.to_frame()
                z2 = z1.loc[[
                    'const',
                     name1 + '_' + name2 + '_NicheDummy',
                    'PostDummy',
                     name1 + '_' + name2 + '_PostXNicheDummy']]
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
                y2['Samples'] = y2['Samples'].apply(lambda x: self.combo_name12_reg_table_names[x] if x in self.combo_name12_reg_table_names.keys() else 'None')
                y2.rename(columns={'index': 'Independent Vars',
                                   'parameter': self.dep_vars_reg_table_names[y_var]},
                          inplace=True)
                params_pvalues_dict[name1][name2] = y2
        # ========= concatenate dataframes into a single dataframe for each combo ==========
        params_pvalues_combo_dict = self._rearrange_combo_df_dict(d = params_pvalues_dict)
        res = dict.fromkeys(params_pvalues_combo_dict.keys())
        for combo, content1 in params_pvalues_combo_dict.items():
            df_list = []
            for name12, df in content1.items():
                df_list.append(df)
            adf = functools.reduce(lambda a, b: a.append(b), df_list)
            res[combo] = adf
        return res

    def put_panel_results_into_pandas_for_all_y_var(self):
        res1 = dict.fromkeys(self.all_y_reg_vars)
        for y in self.all_y_reg_vars:
            res1[y] = self._put_panel_results_into_pandas_for_single_y_var(y_var=y)
        res2 = dict.fromkeys(['combo1', 'combo2', 'combo3', 'combo4'])
        for combo in res2.keys():
            df_list = []
            for y in self.all_y_reg_vars:
                df_list.append(res1[y][combo])
            adf = functools.reduce(lambda a, b: a.merge(b, how='inner',
                                                        on=['Samples', 'Independent Vars']),
                                   df_list)
            print(adf)
            filename = combo + '_POOLED_OLS_REG_RESULTS.csv'
            adf.to_csv(self.main_path / 'reg_tables_ready_for_latex' / filename)
            res2[combo] = adf
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def select_cols_and_rows_result_PostXNicheDummy(self, result_type, table_type):
        """
        table_type == table_1:
        include full, minInstalls and developer sub-samples, pooled OLS results with
        post * niche dummy (PostXNicheDummy) for now.
        table_type == table_2:
        include genreId sub-samples, pooled OLS results with
        post * niche dummy (PostXNicheDummy) for now.
        table_type == table_3:
        include full sample, pooled OLS results with
        post * niche scale dummies 1-19 (PostXNicheDummy) for now.
        """
        if result_type == 'panel':
            q = regression.panel_path / 'reg_results' / 'panel' / 'NicheDummy_combined.csv'
        else:
            q = regression.panel_path / 'reg_results' / 'cross_section' / 'NicheDummy_combined.csv'
        df = pd.read_csv(q)
        df.set_index('index', inplace=True)
        # -------------------------------------------------------------------------------
        if result_type == 'panel':
            selected_col_names = []
            substrings = ['POOLED_OLS', 'POOLED_OLS_pvalue']
            for i in df.columns:
                if any([substring in i for substring in substrings]):
                    selected_col_names.append(i)
        else:
            selected_col_names = df.columns.tolist()
        # -------------------------------------------------------------------------------
        selected_row_indices = []
        if table_type == 'table_1':
            if result_type == 'panel':
                contain = ['PostX', 'NicheDummy', 'PostDummy', 'Constant']
            else:
                contain = ['NicheDummy', 'Constant']
            not_contain = ['genreId', 'NicheScaleDummy', 'NicheScaleDummies']
            for i in df.index.values:
                if any([substring in i for substring in contain]) and not any([substring in i for substring in not_contain]):
                    selected_row_indices.append(i)
        elif table_type == 'table_2':
            if result_type == 'panel':
                substrings = ['PostX', 'NicheDummy', 'PostDummy', 'Constant']
            else:
                substrings = ['NicheDummy', 'Constant']
            for i in df.index.values:
                if any([substring in i for substring in substrings]) and 'genreId' in i:
                    selected_row_indices.append(i)
        else:
            if result_type == 'panel':
                substrings = ['PostXfull_full_NicheScaleDummy',
                              'full_full_NicheScaleDummy',
                              'NicheScaleDummies_PostDummy',
                              'NicheScaleDummies_Constant']
            else:
                substrings = ['NicheScaleDummy',
                              'NicheScaleDummies_Constant']
            for i in df.index.values:
                if any([substring in i for substring in substrings]):
                    selected_row_indices.append(i)
        # -------------------------------------------------------------------------------
        df2 = df.loc[selected_row_indices, selected_col_names]
        return df2

    def add_pvalue_asterisk_to_results(self, df):
        """
        This should be run after self.select_cols_and_rows_result_PostXNicheDummy(self, result_type, table_type).
        df is the output of self.select_cols_and_rows_result_PostXNicheDummy
        """
        # convert p_value columns to asterisks
        df2 = df.copy(deep=True)
        pvaluecols = [i for i in df2.columns if 'pvalue' in i]
        def f(x):
            if 0.05 < float(x) <= 0.1:
                return '*'
            elif 0.01 < float(x) <= 0.05:
                return '**'
            elif float(x) <= 0.01:
                return '***'
            else:
                return ''
        for i in pvaluecols:
            df2[i] = df2[i].apply(f)
        # concat pvalue cols to coefficient cols
        for i in df2.columns:
            if i not in pvaluecols:
                df2[i] = df2[i].astype(str) + df2[i+'_pvalue']
        df2.drop(pvaluecols, axis=1, inplace=True)
        return df2

    def set_row_and_column_groups(self, df, result_type, table_type):
        """
        The input df is the output of self.add_pvalue_asterisk_to_results(df)
        """
        df2 = df.copy(deep=True)
        # group columns ---------------------------------------------------------
        for i in df2.columns:
            new_i = i.replace('_POOLED_OLS', '')
            df2.rename(columns={i: new_i}, inplace=True)
        df2.rename(columns={'offersIAPTrue': 'OffersIAP',
                            'containsAdsTrue': 'ContainsAds',
                            'paidTrue': 'Paid',
                            'Imputedprice': 'Price'},
                   inplace=True)
        df2.columns = pd.MultiIndex.from_product([['Dependant Variables'],
                                                  df2.columns.tolist()])
        df2 = df2.reset_index()
        def reformat_index(x):
            x = x.replace('_', ' ')
            x = x.lower()
            return x
        df2['index'] = df2['index'].apply(reformat_index)
        df2.rename(columns={'index': 'Independent Vars'}, inplace=True)
        df2['Samples'] = None
        def replace_all(text, dic):
            for i, j in dic.items():
                text = text.replace(i, j)
            return text
        def set_sample(x):
            if 'full' in x:
                x = 'full'
            elif 'genreid' in x:
                rep = {' nichedummy': '',
                       'postxgenreid ': '',
                        'genreid ': '',
                       ' postdummy': '',
                       ' constant': ''}
                x = replace_all(x, rep)
            elif 'mininstalls imputedmininstalls ' in x:
                rep = {' nichedummy': '',
                       'postxmininstalls imputedmininstalls ': '',
                        'mininstalls imputedmininstalls ': '',
                       ' postdummy': '',
                       ' constant': ''}
                x = replace_all(x, rep)
            elif 'developer' in x:
                rep = {' nichedummy': '',
                       'postxdeveloper ': '',
                        'developer ': '',
                       ' postdummy': '',
                       ' constant': ''}
                x = replace_all(x, rep)
            else:
                x = x
            return x
        df2['Samples'] = df2['Independent Vars'].apply(set_sample)
        df2['Samples'] = df2['Samples'].apply(lambda x: x.capitalize())
        # ----------------------------------------------------------------------------------
        def set_row_indep_var_panel(x):
            if 'constant' in x:
                return 'Constant'
            elif 'postdummy' in x:
                return 'Post'
            elif 'nichedummy' in x and 'postx' not in x:
                return 'Niche'
            elif 'nichedummy' in x and 'postx' in x:
                return 'PostNiche'
            elif 'postx' in x and 'nichescaledummy' in x:
                rep = {'postxfull full ': 'Post',
                       'nichescaledummy': 'NicheScale'}
                x = replace_all(x, rep)
                return x
            elif 'full full nichescaledummy' in x:
                rep = {'full full nichescaledummy': 'NicheScale'}
                x = replace_all(x, rep)
                return x
            else:
                return x
        def set_row_indep_var_ols(x):
            if 'constant' in x:
                return 'Constant'
            elif 'nichedummy' in x:
                return 'Niche'
            elif 'nichescaledummy' in x:
                return 'NicheScale'
            else:
                return x
        # ----------------------------------------------------------------------------------
        if result_type == 'panel':
            df2['Independent Vars'] = df2['Independent Vars'].apply(set_row_indep_var_panel)
        else:
            df2['Independent Vars'] = df2['Independent Vars'].apply(set_row_indep_var_ols)
        # sort independent vars in the order Post Niche and Postniche
        df2['ind_var_order'] = None
        df2.at[df2['Independent Vars'] == 'Constant', 'ind_var_order'] = 0
        df2.at[df2['Independent Vars'] == 'Niche', 'ind_var_order'] = 1
        df2.at[df2['Independent Vars'] == 'Post', 'ind_var_order'] = 2
        df2.at[df2['Independent Vars'] == 'PostNiche', 'ind_var_order'] = 3
        for i in range(1, 20):
            df2.at[df2['Independent Vars'] == 'NicheScale ' + str(i), 'ind_var_order'] = i+2
            df2.at[df2['Independent Vars'] == 'PostNicheScale ' + str(i), 'ind_var_order'] = i+2
        df2 = df2.groupby(['Samples'], sort=False) \
            .apply(lambda x: x.sort_values(['ind_var_order'], ascending=True)) \
            .reset_index(drop=True).drop(['ind_var_order'], axis=1)
        if table_type == 'table_3':
            df2.drop(['Samples'], axis=1, inplace=True)
            df2.set_index('Independent Vars', inplace=True)
        else:
            df2.set_index(['Samples', 'Independent Vars'], inplace=True)
        return df2

    def convert_to_latex(self, df, result_type, table_type):
        """
        df is the output of self.set_row_and_column_groups(self, df, result_type)
        """
        f_name = result_type + '_' + table_type + '_NicheDummy_Combined.tex'
        df.to_latex(buf=regression.reg_table_path / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        return df

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
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def check_text_label_contents(self):
        df2 = self.cdf.copy(deep=True)
        d = essay_1_stats_and_regs_201907._open_predicted_labels_dict()
        for name1, content in d.items():
            for name2, text_label_col in content.items():
                label_col_name = name1 + '_' + name2 + '_kmeans_labels'
                unique_labels = df2[label_col_name].unique()
                for label_num in unique_labels:
                    df3 = df2.loc[df2[label_col_name]==label_num, [self.tcn + 'ModeClean']]
                    if len(df3.index) >= 10:
                        df3 = df3.sample(n=10)
                    f_name = self.initial_panel + '_' + name1 + '_' + name2 + '_' + 'TL_' + str(label_num) + '_' + self.tcn + '_sample.csv'
                    q = essay_1_stats_and_regs_201907.panel_essay_1_path / 'check_predicted_label_text_cols' / f_name
                    df3.to_csv(q)
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

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
                                         bins=essay_1_stats_and_regs_201907.text_cluster_size_bins,
                                         include_lowest=True,
                                         labels=essay_1_stats_and_regs_201907.text_cluster_size_labels)
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
                                                graph_title='Histogram of Clusters In Each Cluster Size Bin',
                                                relevant_folder_name='numClusters_per_cluster_size_bin')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def _numApps_per_cluster_size_bin(self):
        d1 = self._numApps_per_cluster()
        d3 = essay_1_stats_and_regs_201907._open_predicted_labels_dict()
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
                                   bins=essay_1_stats_and_regs_201907.text_cluster_size_bins,
                                   include_lowest=True,
                                   labels=essay_1_stats_and_regs_201907.text_cluster_size_labels)
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
                                                graph_title='Histogram of Apps Count In Each Cluster Size Bin',
                                                relevant_folder_name='numApps_per_cluster_size_bin')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def text_cluster_stats_at_app_level(self):
        d1 = essay_1_stats_and_regs_201907._open_predicted_labels_dict()
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
                                   bins=essay_1_stats_and_regs_201907.text_cluster_size_bins,
                                   include_lowest=True,
                                   labels=essay_1_stats_and_regs_201907.text_cluster_size_labels)
                # create a new column indicating number of cluster for each cluster size bin
                df[numClusters_in_cluster_size_bin] = df[cluster_size_bin].apply(
                    lambda x: d3[name1][name2].loc[x].squeeze())
                # create a new column indicating grouped sum of numApps_in_cluster for each cluster_size
                df[numApps_in_cluster_size_bin] = df[cluster_size_bin].apply(
                    lambda x: d4[name1][name2].loc[x].squeeze())
                res[name1][name2] = df
        filename = self.initial_panel + '_dict_app_level_text_cluster_stats.pickle'
        q = essay_1_stats_and_regs_201907.panel_essay_1_path / 'app_level_text_cluster_stats' / filename
        pickle.dump(res, open(q, 'wb'))
        return essay_1_stats_and_regs_201907(
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
        fig, ax = plt.subplots(figsize=self.combo_barh_figsize[combo])
        fig.subplots_adjust(left=0.2)
        # -------------------------------------------------------------------------
        res = self._groupby_subsample_dfs_by_nichedummy()
        df = self._combine_name2s_into_single_df(name12_list=self.graph_combo_ssnames[combo],
                                                 d=res)
        df.plot.barh(stacked=True,
                     color={"Broad Apps": "orangered",
                            "Niche Apps": "lightsalmon"},
                     ax=ax)
        ax.set_ylabel('Samples')
        ax.set_yticklabels(ax.get_yticklabels(),
                           fontsize=self.combo_barh_yticklabel_fontsize[combo])
        ax.set_xlabel('Apps Count')
        ax.xaxis.grid()
        ax.spines['right'].set_visible(False)
        ax.spines['top'].set_visible(False)
        graph_title = self.initial_panel + ' ' + self.combo_graph_titles[combo] + \
                      '\n Apps Count by Niche and Broad Types'
        ax.set_title(graph_title)
        ax.legend()
        # ------------------ save file -----------------------------------------------------------------
        self._set_title_and_save_graphs(fig=fig,
                                        file_keywords=self.combo_graph_titles[combo].lower().replace(' ', '_'),
                                        relevant_folder_name='nichedummy_count_by_subgroup')
        return essay_1_stats_and_regs_201907(
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
        key_vars = ['LogImputedprice',
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
                        stubnames=['LogImputedprice', 'LogImputedminInstalls',
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
        res = dict.fromkeys(self.graph_combo_ssnames.keys())
        for combo, name1_2 in self.graph_combo_ssnames.items():
            res[combo] = dict.fromkeys(name1_2)
        for combo in res.keys():
            for name1, content1 in self.ssnames.items():
                for name2 in content1:
                    the_name = name1 + '_' + name2
                    if the_name in res[combo].keys():
                        res[combo][the_name] = d[name1][name2]
        return res

    def graph_descriptive_stats_pricing_vars(self, combo, the_panel):
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
        key_vars = ['LogImputedprice', 'LogImputedminInstalls', 'both_IAP_and_ADS']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo]['figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = list(res12[combo].keys())
            for j in range(len(name1_2_l)):
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                if key_vars[i] in ['LogImputedprice', 'LogImputedminInstalls']:
                    sns.violinplot(
                        x= name1_2_l[j] + '_cluster_size_bin',
                        y= key_vars[i] + "_" + the_panel,
                        data=res12[combo][name1_2_l[j]],
                        color=".8",
                        inner=None,  # because you are overlaying stripplot
                        cut=0,
                        ax=ax.flat[j])
                    # overlay swamp plot with violin plot
                    sns.stripplot(
                        x= name1_2_l[j] + '_cluster_size_bin',
                        y= key_vars[i] + "_" + the_panel,
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
                graph_title = essay_1_stats_and_regs_201907.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_ylim(bottom=0)
                ax.flat[j].set_xlabel('Text Cluster Sizes Bins')
                y_label_dict = {'LogImputedminInstalls': 'Log Minimum Installs',
                                'LogImputedprice': 'Log Price',
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
                    fig.legend(handles=[top_bar, middle_bar, bottom_bar],
                               labels=['Total (100%)', 'Contains Ads (%)', 'Offers IAP (%)'],
                               loc='upper right',
                               ncol=1,  frameon=False)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            file_keywords=key_vars[i] + '_' + combo + '__' + the_panel,
                                            graph_title=self.multi_graph_combo_suptitle[combo] + \
                                                        ' Cross Section Descriptive Statistics of \n' + \
                                                        self.graph_dep_vars_descriptive_stats_names[key_vars[i]] + the_panel,
                                            relevant_folder_name='pricing_vars_stats')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_group_mean_subsamples_parallel_trends(self, combo):
        """
        https://github.com/mwaskom/seaborn/blob/master/seaborn/relational.py
        seaborn line plot create a band for measuring central tendancy (95% CI),
        I am curious to know how they calculated the band.
        """
        res12, res34 = self._prepare_pricing_vars_for_graph_group_by_var(
            group_by_var='NicheDummy')
        res12 = self._rearrange_combo_df_dict(d=res12)
        res34 = self._rearrange_combo_df_dict(d=res34)
        key_vars = ['LogImputedminInstalls', 'LogImputedprice', 'TRUE%_offersIAPTrue', 'TRUE%_containsAdsTrue']
        # --------------------------------------- graph -------------------------------------------------
        for i in range(len(key_vars)):
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo][
                                       'figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = list(res12[combo].keys())
            for j in range(len(name1_2_l)):
                nichedummy = name1_2_l[j] + "_NicheDummy"
                sns.set(style="whitegrid")
                sns.despine(right=True, top=True)
                hue_order = [1, 0]
                if key_vars[i] in ['LogImputedminInstalls', 'LogImputedprice']:
                    sns.lineplot(
                        data=res12[combo][name1_2_l[j]],
                        x="panel",
                        y= key_vars[i],
                        hue=nichedummy,
                        hue_order=hue_order,
                        markers=True,
                        style=nichedummy,
                        dashes=False,
                        ax = ax.flat[j])
                    ylimits = {'LogImputedminInstalls':{'bottom':0, 'top':25},
                               'LogImputedprice':{'bottom':0, 'top':2}}
                    ax.flat[j].set_ylim(bottom=ylimits[key_vars[i]]['bottom'],
                                        top=ylimits[key_vars[i]]['top'])
                else:
                    sns.lineplot(
                        data=res34[combo][name1_2_l[j]],
                        x="panel",
                        y= key_vars[i],
                        hue=nichedummy,
                        hue_order=hue_order,
                        markers=True,
                        style=nichedummy,
                        dashes=False,
                        ax = ax.flat[j])
                    ax.flat[j].set_ylim(bottom=0, top=100)
                graph_title = essay_1_stats_and_regs_201907.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].axvline(x='2020-03', linewidth=2, color='red')
                ax.flat[j].set_xlabel("Time")
                y_label_dict = {'LogImputedminInstalls': 'Log Minimum Installs',
                                'LogImputedprice': 'Log Price',
                                'TRUE%_offersIAPTrue': 'Percentage of Apps Offers IAP',
                                'TRUE%_containsAdsTrue': 'Percentage of Apps Contains Ads'}
                ax.flat[j].set_ylabel(y_label_dict[key_vars[i]])
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                ax.flat[j].legend().set_visible(False)
            fig.legend(labels=['Niche App : Yes', 'Niche App : No'],
                       loc='lower right', ncol=2)
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            file_keywords=key_vars[i] + '_' + combo + '_group_mean_parallel_trends',
                                            graph_title=essay_1_stats_and_regs_201907.multi_graph_combo_suptitle[combo] + \
                                                        " Group Mean Parallel Trends",
                                            relevant_folder_name='parallel_trend_group_mean')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def graph_beta_parallel_trends(self, combo, reg_type, alpha):
        """
        :return: six graphs per page (each graph is 1 sub-sample), 1 page has 1 dep var, hues are leaders and non-leaders
        """
        res = self._create_cross_section_reg_results_df_for_parallel_trend_beta_graph(alpha)
        for dep_var in ['LogImputedprice', 'LogImputedminInstalls', 'offersIAPTrue', 'containsAdsTrue']:
            fig, ax = plt.subplots(nrows=self.multi_graph_combo_fig_subplot_layout[combo]['nrows'],
                                   ncols=self.multi_graph_combo_fig_subplot_layout[combo]['ncols'],
                                   figsize=self.multi_graph_combo_fig_subplot_layout[combo]['figsize'],
                                   sharey='row',
                                   sharex='col')
            fig.subplots_adjust(bottom=0.2)
            name1_2_l = self.graph_combo_ssnames[combo]
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
                graph_title = self.graph_subsample_title_dict[name1_2_l[j]]
                ax.flat[j].set_title(graph_title)
                ax.flat[j].set_xlabel("Time")
                ax.flat[j].set_ylabel('Niche Dummy Coefficient')
                ax.flat[j].grid(True)
                ax.flat[j].xaxis.set_visible(True)
                for tick in ax.flat[j].get_xticklabels():
                    tick.set_rotation(45)
                handles, labels = ax.flat[j].get_legend_handles_labels()
                fig.legend(handles, labels, loc='upper right', ncol=2)
            dep_var_dict = {'LogImputedminInstalls': 'Log Minimum Installs',
                            'LogImputedprice': 'Log Price',
                            'offersIAPTrue': 'Percentage of Apps Offers IAP',
                            'containsAdsTrue': 'Percentage of Apps Contains Ads'}
            # ------------ set title and save ---------------------------------------------
            self._set_title_and_save_graphs(fig=fig,
                                            file_keywords=dep_var + '_' + combo + '_beta_nichedummy_parallel_trends',
                                            graph_title=essay_1_stats_and_regs_201907.multi_graph_combo_suptitle[combo] + \
                                                  ' ' + dep_var_dict[dep_var] + \
                                                  " \nRegress on Niche Dummy Coefficient Parallel Trends",
                                            relevant_folder_name='parallel_trend_nichedummy')
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def cat_var_count(self, cat_var, the_panel=None):
        if the_panel is not None:
            col_name = cat_var + '_' + the_panel
            rd = self.cdf.groupby(col_name)['count_' + the_panel].count()
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def log_transform_pricing_vars(self):
        df1 = self._select_vars(df=self.cdf, time_variant_vars_list=['Imputedprice', 'ImputedminInstalls'])
        for i in self.all_panels:
            df1['LogImputedprice_' + i] = np.log(df1['Imputedprice_' + i] + 1)
            df1['LogImputedminInstalls_' + i] = np.log(df1['ImputedminInstalls_' + i] + 1)
        dcols = ['Imputedprice_' + i for i in self.all_panels] + ['ImputedminInstalls_' + i for i in self.all_panels]
        df1.drop(dcols, axis=1, inplace=True)
        self.cdf = self.cdf.join(df1, how='inner')
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
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
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    # I will first standardize variables and then demean them
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
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def create_var_definition_latex(self):
        df = pd.DataFrame()
        # the variable name here are the actual one shown in the paper, not the ones embedded in the code
        # MAPPING -------------------------------------------------
        var_definitions = {
                    'containsAdsTrue': '\makecell[l]{Dummy variable equals 1 if an app \\\ contains advertisement, 0 otherwise.}',
                    'offersIAPTrue': '\makecell[l]{Dummy variable equals 1 if an app \\\ offers in-app-purchase, 0 otherwise.}',
                    'paidTrue': '\makecell[l]{Dummy variable equals 1 if an app charges a positive \\\ price upfront, 0 otherwise.}',
                    'Imputedprice': '\makecell[l]{App price in USD.}',
                    'PostDummy': """ \makecell[l]{Dummy variable equals 1 if the observation \\\ 
                                    is after Mar 2020 (inclusive), and 0 if the \\\ 
                                    observation is before Mar 2020} """,
                    'PostNiche': """ \makecell[l]{Interaction variable between Post and Niche,\\\  
                                    which equals 1 if and only if the observation \\\ 
                                    equals to 1 for both Post and Niche.} """,
                    'PostNicheScale': """ \makecell[l]{Interaction variable between Post and NicheScale dummies,\\\  
                                    which equals 1 if and only if the observation \\\ 
                                    equals to 1 for both Post and NicheScale.} """,
                    'Imputedscore': '\makecell[l]{Average score, between 1 and 5, of an app in a period. }',
                    'DeMeanedImputedscore': """ \makecell[l]{Demeaning the score across all panels \\\ 
                                            within each app.} """,
                    'minInstallsTop': """ \makecell[l]{Dummy variable equals 1 if the app \\\ 
                                            has minimum installs above 10,000,000 \\\ 
                                            (inclusive) in a panel, 0 otherwise.} """,
                    'DeMeanedminInstallsTop': '\makecell[l]{Demeaning Tier1 across all panels \\\ within each app.}',
                    'minInstallsMiddle': """ \makecell[l]{Dummy variable equals 1 if the app \\\ 
                                            has minimum installs below 10,000,000 and \\\ 
                                            above 100,000 (inclusive), 0 otherwise.} """,
                    'DeMeanedminInstallsMiddle': '\makecell[l]{Demeaning Tier2 across all panels \\\ within each app.}',
                    'Imputedreviews': '\makecell[l]{Number of reviews of an app in a panel.}',
                    'ZScoreDeMeanedImputedreviews': """ \makecell[l]{Demeaning Z-Score standardized \\\ 
                                                        number of reviews across all panels within each app.} """,
                    'contentRatingAdult': '\makecell[l]{Dummy variable equals 1 if the app has \\\ adult content, 0 otherwise.}',
                    'size': '\makecell[l]{Size (MB)}',
                    'DaysSinceReleased': '\makecell[l]{Number of days since the app was \\\ launched on Google Play Store.}',
                    'top_digital_firms': '\makecell[l]{Dummy variable equals 1 if the app is \\\ owned by a Top Digital Firm.}',
                    'NicheDummy': """ \makecell[l]{Dummy variable equals 1 if an app \\\ 
                                        is a niche-type app, 0 if an app is broad-type app.}""",
                    'NicheScaleDummies': """ \makecell[l]{20 dummy variables representing the degree of niche property of an app. \\\ 
                                            NicheScale_{0} equals to 1 represents that the app is the most broad type, \\\ 
                                            while NicheScale_{19} equals to 1 represents that the app is the most niche type.} """}

        time_variancy = {
            'Time Variant': ['containsAdsTrue',
                                     'offersIAPTrue',
                                     'paidTrue',
                                     'Imputedprice',
                                     'PostDummy',
                                     'PostNiche',
                                     'PostNicheScale',
                                     'Imputedscore',
                                     'DeMeanedImputedscore',
                                     'minInstallsTop',
                                     'DeMeanedminInstallsTop',
                                     'minInstallsMiddle',
                                     'DeMeanedminInstallsMiddle',
                                     'Imputedreviews',
                                     'ZScoreDeMeanedImputedreviews'],
            'Time Invariant': ['contentRatingAdult',
                                       'size',
                                       'DaysSinceReleased',
                                       'top_digital_firms',
                                       'NicheDummy',
                                       'NicheScaleDummies']}

        df['reference'] = ['containsAdsTrue',
                                            'offersIAPTrue',
                                            'paidTrue',
                                            'Imputedprice',
                                            'PostDummy',
                                            'PostNiche',
                                            'PostNicheScale',
                                            'Imputedscore',
                                            'DeMeanedImputedscore',
                                            'minInstallsTop',
                                            'DeMeanedminInstallsTop',
                                            'minInstallsMiddle',
                                            'DeMeanedminInstallsMiddle',
                                            'Imputedreviews',
                                            'ZScoreDeMeanedImputedreviews',
                                            'contentRatingAdult',
                                            'size',
                                            'DaysSinceReleased',
                                            'top_digital_firms',
                                            'NicheDummy',
                                            'NicheScaleDummies']

        df['Variables'] = [regression.var_names[i] for i in df['reference']]

        df['Definitions'] = None
        for var, definition in var_definitions.items():
            df.at[df['reference'] == var, 'Definitions'] = definition

        df['Type'] = None
        for type, var_list in time_variancy.items():
            for j in var_list:
                for i in df['reference']:
                    if i == j:
                        df.at[df['reference'] == i, 'Type'] = type

        df.drop('reference', axis=1, inplace=True)
        df.set_index(['Type', 'Variables'], inplace=True)

        f_name = 'variable_definition.tex'
        df.to_latex(buf=regression.descriptive_stats_tables / f_name,
                    multirow=True,
                    multicolumn=True,
                    longtable=True,
                    position='h!',
                    escape=False)
        return df

    def single_panel_descriptive_stats(self, var_list, the_panel, type):
        """
        run self.slice_single_panel_descriptive_stats before running this,
        DO NOT run self.select_x_y_for_subsample() and self.slice_single_panel() before this, because
        regression need different dataframes from descriptive stats.
        """
        for name1, content1 in self.single_panel_df.items():
            for name2, df in content1.items():
                df2 = df.copy(deep=True)
                if type == 'dummy':
                    df_list = []
                    for i in var_list:
                    # some dummy var has been deleted because of they are perfectly correlated with other dummy var
                        if i in df2.columns:
                            if df2[i].nunique() == 2:
                                df3 = df2[i].value_counts()
                                df3 = df3.rename(i).to_frame().T.rename(columns={0: 'False', 1: 'True'})
                            else: # sometimes the entire column is 1 or 0
                                if df2[i].unique() == 1:
                                    df3 = df2[i].value_counts()
                                    df3 = df3.rename(i).to_frame().T.rename(columns={1: 'True'})
                                    df3['False'] = 0
                                else:
                                    df3 = df2[i].value_counts()
                                    df3 = df3.rename(i).to_frame().T.rename(columns={0: 'False'})
                                    df3['True'] = 0
                            df3['count'] = df3['False'] + df3['True']
                            df_list.append(df3)
                    f_name = self.initial_panel + '_PANEL_' + the_panel + '_' + name1 + '_' + name2  + '_Dummy_Stats.csv'
                    q = regression.descriptive_stats_tables / 'dummy_stats' / f_name
                elif type == 'continuous':
                    df_list = []
                    for i in var_list:
                        if i in df2.columns:
                            df3 = df2[i].describe(include=[np.number]).round(3)
                            df3 = df3.rename(i).to_frame().T
                            df_list.append(df3)
                    f_name = self.initial_panel + '_PANEL_' + the_panel + '_' + name1 + '_' + name2  + '_Continuous_Stats.csv'
                    q = regression.descriptive_stats_tables / 'continuous_stats' / f_name
                else: # for DeMeaned Variables Only (They are continuous)
                    df_list = []
                    for i in var_list:
                        if i in df2.columns:
                            df3 = df2[i].describe(include=[np.number]).round(3)
                            df3 = df3.rename(i).to_frame().T
                            df_list.append(df3)
                    f_name = self.initial_panel + '_PANEL_' + the_panel + '_' + name1 + '_' + name2 + '_DeMeaned_Stats.csv'
                    q = regression.descriptive_stats_tables / 'demeaned_stats' / f_name
                stats_df = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
                stats_df.to_csv(q)
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)

    def export_descriptive_stats_to_latex(self, the_panel):
        # =============== DUMMY TABLE ==============================================================
        df_list = []
        for name, mapped_name in sample_name.items():
            f_name = self.initial_panel + '_PANEL_' + the_panel + name + '_Dummy_Stats.csv'
            q = regression.descriptive_stats_tables / 'dummy_stats' / f_name
            df = pd.read_csv(q)
            df['Sample'] = mapped_name
            df_list.append(df)
        df2 = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
        df2['Variables'] = [regression.var_names[i] for i in df2.iloc[:,0]]
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2 = df2[['Sample', 'Variables', 'True', 'False', 'count']]
        df2.set_index(['Sample', 'Variables'], inplace=True)
        f_name = self.initial_panel + '_PANEL_' + the_panel + '_dummy_vars_descriptive_stats.tex'
        df2.to_latex(buf=regression.descriptive_stats_tables / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        # =============== CONTINUOUS TABLE =========================================================
        df_list = []
        for name, mapped_name in sample_name.items():
            f_name = self.initial_panel + '_PANEL_' + the_panel + name + '_Continuous_Stats.csv'
            q = regression.descriptive_stats_tables / 'continuous_stats' / f_name
            df = pd.read_csv(q)
            df['Sample'] = mapped_name
            df_list.append(df)
        df2 = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
        df2['Variables'] = [regression.var_names[i] for i in df2.iloc[:,0]]
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2.rename(columns={'50%': 'median'}, inplace=True)
        df2 = df2[['Sample', 'Variables', 'mean', 'std', 'min', 'median', 'max', 'count']]
        df2.set_index(['Sample', 'Variables'], inplace=True)
        # --- convert data type and round ----
        df2['count'] = df2['count'].astype(int)
        for i in ['mean', 'std', 'min', 'median', 'max']:
            df2[i] = df2[i].apply(lambda x: round(x, 2))
        f_name = self.initial_panel + '_PANEL_' + the_panel + '_continuous_vars_descriptive_stats.tex'
        df2.to_latex(buf=regression.descriptive_stats_tables / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        # =============== DEMEANED TABLE =========================================================
        df_list = []
        for name, mapped_name in sample_name.items():
            f_name = self.initial_panel + '_PANEL_' + the_panel + name + '_DeMeaned_Stats.csv'
            q = regression.descriptive_stats_tables / 'demeaned_stats' / f_name
            df = pd.read_csv(q)
            df['Sample'] = mapped_name
            df_list.append(df)
        df2 = functools.reduce(lambda a, b: pd.concat([a, b]), df_list)
        df2['Variables'] = [regression.var_names[i] for i in df2.iloc[:,0]]
        df2.drop(df2.columns[0], axis=1, inplace=True)
        df2.rename(columns={'50%': 'median'}, inplace=True)
        df2 = df2[['Sample', 'Variables', 'mean', 'std', 'min', 'median', 'max', 'count']]
        df2.set_index(['Sample', 'Variables'], inplace=True)
        # --- convert data type and round ----
        df2['count'] = df2['count'].astype(int)
        for i in ['mean', 'std', 'min', 'median', 'max']:
            df2[i] = df2[i].apply(lambda x: round(x, 2))
        f_name = self.initial_panel + '_PANEL_' + the_panel + '_demeaned_vars_descriptive_stats.tex'
        df2.to_latex(buf=regression.descriptive_stats_tables / f_name,
                   multirow=True,
                   multicolumn=True,
                   longtable=True,
                   position='h!',
                   escape=False)
        return essay_1_stats_and_regs_201907(
                                   tcn=self.tcn,
                                   combined_df=self.cdf,
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   broadDummy_labels=self.broadDummy_labels,
                                   reg_results=self.reg_results)


