import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
import numpy as np
from datetime import datetime
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")

class combine_dataframes():

    def __init__(self,
                 initial_panel,
                 all_panels,
                 consec_panels,
                 appid_imputed_and_deleted_missing_df,
                 dev_index_gecoded_df,
                 appid_text_cluster_labeled_df):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
        self.dfa = appid_imputed_and_deleted_missing_df
        self.dfd = dev_index_gecoded_df
        self.dfl = appid_text_cluster_labeled_df

    def combine_imputed_deleted_missing_with_text_labels(self):
        inter_df = self.dfa.join(self.dfl, how='inner')
        inter_df.index.names = ['appid']
        inter_df.reset_index(inplace=True)
        dfd2 = self.dfd.reset_index()
        dfd2 = dfd2[['developer', 'location', 'longitude', 'latitude']]
        inter_df.rename(columns={'developer_'+self.initial_panel: 'developer'}, inplace=True)
        result_df = inter_df.merge(dfd2, on='developer', how='left', validate='m:1')
        return result_df

    def convert_to_dev_multiindex(self):
        result_df = self.combine_imputed_deleted_missing_with_text_labels()
        num_apps_df = result_df.groupby('developer')['appid'].nunique().rename('num_apps_owned').to_frame()
        result_df = result_df.merge(num_apps_df, on='developer', how='inner')
        result_df.set_index(['developer', 'appid'], inplace=True)
        return result_df

    def eyeball_labels_by_group(self, label):
        # get a count df
        result_df = self.convert_to_dev_multiindex()
        gdf = result_df.groupby(by=['predicted_labels']).count()
        gdf = gdf.iloc[:, 0].to_frame()
        gdf.index.names = ['predicted_labels']
        gdf.rename(columns={gdf.columns[0]: 'count'}, inplace=True)
        gdf.sort_values(by='count', ascending=False, inplace=True)
        # display on a specific label
        sdf = result_df.loc[result_df['predicted_labels'] == label,
                            ['summary_202102', 'description_202102', 'predicted_labels']]
        return gdf, sdf

#########################################################################################
#######   REGRESSION   ##################################################################
#########################################################################################

class regression_analysis():
    """by default, regression analysis will either use cross sectional data or panel data with CONSECUTIVE panels,
    this is because we can calculate t-1 easily."""

    def __init__(self,
                 df,
                 initial_panel,
                 consec_panels,
                 dep_var='minInstalls',
                 h1_ind_vars=None,
                 h2_ind_vars=None,
                 h3_ind_vars=None):
        self.df = df # df is the output of convert_to_dev_multiiindex
        self.initial_panel = initial_panel
        self.consec_panels = consec_panels
        self.dep_var = dep_var
        self.h1_ivars = h1_ind_vars
        self.h2_ivars = h2_ind_vars
        self.h3_ivars = h3_ind_vars

    def print_col_names(self, the_panel):
        for i in self.df.columns:
            if the_panel in i:
                print(i)

    def select_vars_contain_text(self, text):
        l1 = []
        for i in self.df.columns:
            if text in i:
                x = i.split("_", -1)
                x = x[:-1]
                y = '_'.join(x)
                l1.append(y)
        return list(set(l1))

    def select_vars(self, the_panel=None, **kwargs):
        if 'var_list' in kwargs.keys():
            variables = kwargs['var_list']
        elif 'var_text' in kwargs.keys():
            variables = self.select_vars_contain_text(text=kwargs['var_text'])
        if the_panel is None:
            selected_cols = []
            for p in self.consec_panels:
                cols = [item + '_' + p for item in variables]
                selected_cols.extend(cols)
        else:
            selected_cols = [item + '_' + the_panel for item in variables]
        selected_df = self.df[selected_cols]
        return selected_df

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

    def groupby_count_cat_dummies(self, dummies_list, the_panel):
        df1 = self.select_vars(var_list=dummies_list, the_panel=the_panel)
        return df1

    # ------------------------------------------------------------------------------------------
    def convert_cat_to_dummies(self):
        everyone_pattern = re.compile(r'(Everyone+)|(10\++)')
        teen_pattern = re.compile(r'(Teen+)|(Mature+)')
        adult_pattern = re.compile(r'(Adults+)|(18\++)')
        def combine_contentRating_into_3_groups(x):
            if x is not None:
                if re.search(everyone_pattern, x):
                    return 'Everyone'
                elif re.search(teen_pattern, x):
                    return 'Teen'
                elif re.search(adult_pattern, x):
                    return 'Adult'
            else:
                return x
        game_pattern = re.compile(r'(GAME+)|(VIDEO_PLAYERS+)')
        productivity_pattern = re.compile(
            r'^(?<!GAME_)(TOOLS+)|(EDUCATION+)|(MEDICAL+)|(LIBRARIES+)|(PARENTING+)|(AUTO_AND_VEHICLES+)|(WEATHER+)|(FINANCE+)|(TRAVEL+)|(BUSINESS+)')
        social_entertainment_pattern = re.compile(
            r'^(?<!GAME_)(ENTERTAINMENT+)|(LIFESTYLE+)|(EVENTS+)|(COMICS+)|(DATING+)|(ART_AND_DESIGN+)|(BEAUTY+)|(SOCIAL+)|(MUSIC_AND_AUDIO+)|(SHOPPING+)|(MAPS_AND_NAVIGATION+)|(PERSONALIZATION+)')
        def combine_genreId_into_3_groups(x):
            if x is not None:
                if re.search(game_pattern, x):
                    return 'Game'
                elif re.search(productivity_pattern, x):
                    return 'Productivity'
                elif re.search(social_entertainment_pattern, x):
                    return 'Entertainment'
            else:
                return x
        df['contentRating'] = df['contentRating'].apply(unlist_a_col_containing_list_of_strings)
        df['contentRating'] = df['contentRating'].apply(combine_contentRating_into_3_groups)
        df['genreId'] = df['genreId'].apply(combine_genreId_into_3_groups)

    # ------------------------------------------------------------------------------------------
    def unlist_a_col_containing_list_of_strings(x):
        if x is not None and re.search(r'\[\]+', x):
            s = eval(x)
            if isinstance(s, list):
                s2 = ', '.join(str(ele) for ele in s)
                return s2
        else:
            return x

    def select_vars_for_reg_df(self, cross_section=True):
        pass




