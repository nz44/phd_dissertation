from pathlib import Path
import pandas as pd
import warnings
warnings.simplefilter('ignore')

# ********************************************************************************************************
# ********************************************************************************************************
class Paths:
    def __init__(self, initial_panel):
        self.initial_panel = initial_panel
# ********************************************************************************************************

class Mac_Paths(Paths):
    def __init__(self, initial_panel):
        super().__init__(initial_panel)
        self.input_path = Path('/Users/zhunaixin/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____')
        self.output_path = Path('/Users/zhunaixin/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/___cleaned_datasets___')
        self.table_output = Path('/Users/zhunaixin/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/tables')
        self.graph_output = Path('/Users/zhunaixin/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs')
        self.nlp_output = Path('/Users/zhunaixin/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp')

class Win_Paths(Paths):
    def __init__(self, initial_panel):
        super().__init__(initial_panel)
        self.input_path = Path(r'C:\Users\naixi\OneDrive\_____GWU_ECON_PHD_____\___Dissertation___\____WEB_SCRAPER____')
        self.output_path = Path(r'C:\Users\naixi\OneDrive\_____GWU_ECON_PHD_____\___Dissertation___\____WEB_SCRAPER____\___cleaned_datasets___')
        self.table_output = Path(r'C:\Users\naixi\OneDrive\__CODING__\PycharmProjects\GOOGLE_PLAY\tables')
        self.graph_output = Path(r'C:\Users\naixi\OneDrive\__CODING__\PycharmProjects\GOOGLE_PLAY\graphs')
        self.nlp_output = Path(r'C:\Users\naixi\OneDrive\__CODING__\PycharmProjects\GOOGLE_PLAY\nlp')

class Linux_Paths(Paths):
    def __init__(self, initial_panel):
        super().__init__(initial_panel)
        self.input_path = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____')
        self.output_path = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/___cleaned_datasets___')
        self.table_output = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/tables')
        self.graph_output = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp')
        self.nlp_output = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs')

# ********************************************************************************************************
# ********************************************************************************************************
class open_or_save_files(Linux_Paths):
    def __init__(self, initial_panel, cluster_type):
        self.cluster_type = cluster_type
        super().__init__(initial_panel)

    def open_merged_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        f_name = self.initial_panel + '_MERGED.pickle'
        q = self.input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df

    def open_token_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        f_name = 'description_converted_to_spacy_tokens.pkl'
        q = self.input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df

    def open_topic_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        f_name = 'description_tokens_converted_to_topics.pkl'
        q = self.input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df

    def open_cluster_df(self):
        folder_name = self.initial_panel + '_PANEL_DF'
        if self.cluster_type == 'k-means':
            f_name = 'df_with_k_means_labels.pkl'
        elif self.cluster_type == 'fuzzy-c-means':
            f_name = 'df_with_fuzzy_c_means_labels.pkl'
        q = self.input_path / '__PANELS__' / folder_name / f_name
        df = pd.read_pickle(q)
        return df