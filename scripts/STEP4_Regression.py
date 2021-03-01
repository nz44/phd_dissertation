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
        inter_df.rename(columns={'developer_'+self.initial_panel: 'developer'}, inplace=True) # here I did not delete apps that changed developer over time
        result_df = inter_df.merge(dfd2, on='developer', how='left', validate='m:1')
        result_df.set_index('appid', inplace=True)
        new_count_cols = ['count_'+i for i in self.consec_panels]
        for i in new_count_cols: # for the purpose of creating dataframes in the groupby count
            result_df[i] = 0
        return result_df

    def convert_to_dev_multiindex(self):
        result_df = self.combine_imputed_deleted_missing_with_text_labels()
        result_df.reset_index(inplace=True)
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

    def print_unique_value_of_var_panel(self, the_var, the_panel=None):
        if the_panel is not None:
            col_name = the_var + '_' + the_panel
            unique_l = self.df[col_name].unique()
            print(col_name, 'contains', len(unique_l), 'unique values')
            print(unique_l)
            return unique_l
        else:
            col_name = [the_var+'_'+i for i in self.consec_panels]
            d = dict.fromkeys(col_name)
            for j in d.keys():
                unique_l = self.df[j].unique()
                print(j, 'contains', len(unique_l), 'unique values')
                print(unique_l)
                d[j]=unique_l
                print()
            return d

    def cat_var_count(self, cat_var, the_panel=None):
        if the_panel is not None:
            col_name = cat_var + '_' + the_panel
            rd = self.df.groupby(col_name)['count_'+the_panel].count()
            rd = rd.sort_values(ascending=False)
            print(rd)
            return rd
        else:
            col_name = [cat_var+'_'+i for i in self.consec_panels]
            df_list = []
            for j in range(len(col_name)):
                rd = self.df.groupby(col_name[j])['count_'+self.consec_panels[j]].count()
                rd = rd.sort_values(ascending=False)
                rd = rd.to_frame()
                df_list.append(rd)
            dfn = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
            print(dfn)
            return dfn

    def df_that_changed_cat_var_over_time(self, cat_var):
        df2 = self.select_vars(single_var=cat_var)
        df_time_variant = []
        for index, row in df2.iterrows():
            row_time_variant = []
            for j in range(len(df2.columns) - 1):
                if row[df2.columns[j]] == row[df2.columns[j + 1]]:
                    row_time_variant.append(False)
                else:
                    row_time_variant.append(True)
            if all(row_time_variant) is True:
                df_time_variant.append(True)
            else:
                df_time_variant.append(False)
        time_variant_df = df2[df_time_variant]
        time_variant_appids = time_variant_df.index.tolist()
        return time_variant_df, time_variant_appids

    def change_rows_with_time_variant_cat_var_to_last_panel(self, cat_var):
        time_variant_df, time_variant_appids = self.df_that_changed_cat_var_over_time(cat_var=cat_var)
        col_names = [cat_var + '_' + i for i in self.consec_panels]
        for i in time_variant_appids:
            for j in col_names:
                self.df.at[i, j] = self.df.at[i, cat_var+'_'+self.consec_panels[-1]] # this one intends to change class attributes
        return self.df

    def create_new_dummies_from_cat_var(self, cat_var, time_invariant=False):
        if time_invariant is True:
            self.df = self.change_rows_with_time_variant_cat_var_to_last_panel(cat_var)
        else:
            pass
        if cat_var == 'genreId':
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1['genreId_game_'+i] = df1['genreId_'+i].apply(lambda x: 1 if 'GAME' in x else 0)
            dcols = ['genreId_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        elif cat_var == 'contentRating':
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1['contentRating_everyone_'+i] = df1['contentRating_'+i].apply(lambda x: 1 if 'Everyone' in x else 0)
            dcols = ['contentRating_'+ i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        else:
            df1 = self.select_vars(single_var=cat_var)
            for i in self.consec_panels:
                df1[cat_var + '_true_' + i] = df1[cat_var + '_' + i].apply(lambda x: 1 if x is True else 0)
            dcols = [cat_var + '_' + i for i in self.consec_panels]
            df1.drop(dcols, axis=1, inplace=True)
            self.df = self.df.join(df1, how='inner')
        return regression_analysis(df=self.df, initial_panel=self.initial_panel, consec_panels=self.consec_panels)

    def select_partial_vars(self, text):
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




