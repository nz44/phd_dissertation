import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
import numpy as np
from datetime import datetime
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

class regression_analysis(combine_dataframes):

    def __init__(self, **kwargs):
        super().__init__(**kwargs)



