from pathlib import Path
import pandas as pd
import warnings
warnings.simplefilter('ignore')
import pickle

# ********************************************************************************************************

class Mac_Paths:
    input_path = Path('/Users/zhunaixin/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____')
    output_path = Path('/Users/zhunaixin/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/___cleaned_datasets___')
    table_output = Path('/Users/zhunaixin/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/tables')
    graph_output = Path('/Users/zhunaixin/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs')
    nlp_output = Path('/Users/zhunaixin/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp')

class Win_Paths:
    input_path = Path(r'C:\Users\naixi\OneDrive\_____GWU_ECON_PHD_____\___Dissertation___\____WEB_SCRAPER____')
    output_path = Path(r'C:\Users\naixi\OneDrive\_____GWU_ECON_PHD_____\___Dissertation___\____WEB_SCRAPER____\___cleaned_datasets___')
    table_output = Path(r'C:\Users\naixi\OneDrive\__CODING__\PycharmProjects\GOOGLE_PLAY\tables')
    graph_output = Path(r'C:\Users\naixi\OneDrive\__CODING__\PycharmProjects\GOOGLE_PLAY\graphs')
    nlp_output = Path(r'C:\Users\naixi\OneDrive\__CODING__\PycharmProjects\GOOGLE_PLAY\nlp')

class Linux_Paths:
    input_path = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____')
    output_path = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/___cleaned_datasets___')
    table_output = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/tables')
    graph_output = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs')
    nlp_output = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp')

# ********************************************************************************************************
# ********************************************************************************************************
class open_files(Linux_Paths):
    def __init__(self, initial_panel=None, cluster_type=None, current_panel=None, all_panels=None):
        self.initial_panel = initial_panel
        self.cluster_type = cluster_type
        self.current_panel = current_panel
        self.all_panels = all_panels

    def open_panel_df(self, type):
        f_name = self.initial_panel + '_' + type + '.pkl'
        q = super().input_path / '__PANELS__' / f_name
        df = pd.read_pickle(q)
        return df

    def open_token_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        f_name = 'description_converted_to_spacy_tokens.pkl'
        q = super().input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df

    def open_topic_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        f_name = 'description_tokens_converted_to_topics.pkl'
        q = super().input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df

    def open_cluster_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        if self.cluster_type == 'k-means':
            f_name = 'df_with_k_means_labels.pkl'
        elif self.cluster_type == 'fuzzy-c-means':
            f_name = 'df_with_fuzzy_c_means_labels.pkl'
        q = super().input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df

    def open_df_to_id_for_scraping(self):
        filename = "ALL_APP_DETAILS_" + self.initial_panel + '.pickle'
        q = super().input_path / "NEW_ALGORITHM_MONTHLY_SCRAPE" / self.initial_panel / filename
        df = pd.read_pickle(q)
        return df

    def open_initial_app_dict(self): # just open the initially scraped panel
        filename = 'ALL_APP_DETAILS_' + self.initial_panel + '.pickle'
        q = super().input_path / "NEW_ALGORITHM_MONTHLY_SCRAPE" / self.initial_panel / filename
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    def open_initial_panel_with_its_tracking_panels(self): # for opening the initial month and the following panels tracking the initial month
        # here the tracking panels are actually all panels including the initial panel
        dfs = dict.fromkeys(self.all_panels)
        for i in range(len(self.all_panels)):
            if i == 0: # open the initial panel
                filename = 'ALL_APP_DETAILS_' + self.initial_panel + '.pickle'
                q = super().input_path / "NEW_ALGORITHM_MONTHLY_SCRAPE" / self.initial_panel / filename
                with open(q, 'rb') as f:
                    dfs[self.initial_panel] = pickle.load(f)
            else:
                filename = 'TRACKING_' + self.initial_panel + '.pickle'
                q = super().input_path / "TRACKING_THE_SAME_ID_MONTHLY_SCRAPE" / self.all_panels[i] / filename
                with open(q, 'rb') as f:
                    dfs[self.all_panels[i]] = pickle.load(f)
        return dfs



# ********************************************************************************************************
# ********************************************************************************************************
class save_files(Linux_Paths):

    def __init__(self, initial_panel=None, current_panel=None,
                 app_details_dict=None, df=None, fig=None):
        self.initial_panel = initial_panel
        self.current_panel = current_panel
        self.app_details_dict = app_details_dict
        self.df = df
        self.fig = fig

    def save_scraped_app_details(self):
        filename = 'TRACKING_' + self.initial_panel + '.pickle'
        q = super().input_path / "TRACKING_THE_SAME_ID_MONTHLY_SCRAPE" / self.current_panel / filename
        pickle.dump(self.app_details_dict, open(q, 'wb'))

    def save_panel_df(self, type):
        filename = self.initial_panel + '_' + type + '.pkl'
        q = super().input_path / '__PANELS__' / filename
        pickle.dump(self.df, open(q, 'wb'))
        # self.merged_df.to_pickle(q) # cannot use this

    def save_fig(self, fig_name, sub_folder):
        q = super().graph_output / sub_folder / fig_name
        self.fig.savefig(fname=q, facecolor=self.fig.get_facecolor())

