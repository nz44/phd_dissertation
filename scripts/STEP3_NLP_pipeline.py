import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pickle
import re
from tqdm import tqdm
tqdm.pandas()
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
pd.options.display.max_columns = None
pd.options.display.max_colwidth = None
import nltk
from sklearn.decomposition import NMF, LatentDirichletAllocation, TruncatedSVD
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.manifold import TSNE
import chart_studio.plotly as py
import plotly.graph_objects as go
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.datasets import fetch_20newsgroups
from sklearn.pipeline import Pipeline
from sklearn import metrics
vectorizer = TfidfVectorizer()
from sklearn import decomposition
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from scipy.sparse import random as sparse_random
from collections import Counter
import random
import skfuzzy as fuzz
from fcmeans import FCM
import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
import functools

class nlp_pipeline():
    """
    the input dataframe is by default appid index
    df is the MERGED dataframe
    df_labels is the output of self.kmeans_cluster and saved with name 'predicted_kmeans_labels
    03/08/21
        After eyeballing self.eyeball_app_changed_niche_label(), I see that they do not ever change over time, the description.
        So I decide from now on to combine all panel's description together.

    03/09/21
        I found that those appids are with exactly the same text information in each panel (after deleting None and impute missing).
        So I conclude that the apps that changed from niche to broad type is completely due to the algorithm only.
        It is pointless to conduct panel specific text label prediction.
        From this particular method, I prove that apps' text labels are time INVARIANT.
        Now I should find a suitable panel regression model that incorporate time-invariant independent variables.

    04/02/21
        1.  I have deleted combined_panel, because I have checked that descriptions in every panel is the same,
            and it is time-invariant, so niche label will be generated by combining all panels.
        2.  I will delete consecutive_panel = True, and uses all panels' text columns in generating niche label,
            this is because I will be using difference-in-difference for pre and after covid, therefore I need to use all columns.
            Even if you are not using diff-in-diff, since niche label is time-invariant, so using all panels will not be any different
            from using only consecutive panels.
        3.  I have deleted functions that assign niche dummy or niche scale dummy, and descriptive stats functions
            relates to niche labels because they are moved to regression_analysis class
        4.  I will divide full sample into game and nongame, and do the NLP algorithm within each subsample.
            Note that game and nongame are time-invariant.
            Accordingly, I will delete the function that generate genreIdGame in regression_analysis class, so
            this dummy is only generated once in the entire code.
    """
    nlp_graph_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp/graphs')
    stoplist = nltk.corpus.stopwords.words('english')
    label_df_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__')
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    def __init__(self,
                 tcn,
                 initial_panel,
                 all_panels,
                 df=None,
                 game_subsamples=None,
                 input_text_cols=None,
                 tf_idf_matrices=None,
                 svd_matrices=None,
                 output_labels=None):
        self.tcn = tcn
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.game_subsamples = game_subsamples
        self.input_text_cols = input_text_cols
        self.tf_idf_matrices = tf_idf_matrices
        self.svd_matrices = svd_matrices
        self.output_labels = output_labels

    def open_imputed_missing_df(self):
        """
        The reason to use imputed missing dataframe instead of imputed and deleted missing is that everytime you delete different number of rows
        (depends on the definition of missig), but not every time you need to re-run text clustering (because it is time-consuming),
        so I will just use the FULL converted data and you could merged this predicted labels to imputed and deleted missing in combine_dataframes class.
        """
        f_name = self.initial_panel + '_imputed_missing.pickle'
        q = nlp_pipeline.label_df_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    ################## deleting, impute and format text columns for the FULL SAMPLE ####################################
    def find_appids_to_remove_before_imputing(self):
        """
        deleting appids that have all missing values in text col for ALL panels
        """
        cols = [self.tcn + '_' + item for item in self.all_panels]
        text_df = self.df[cols]
        null_data = text_df[text_df.isnull().any(axis=1)]
        null_data_t = null_data.T
        appids_to_remove = []
        for appid in null_data_t.columns:
            if null_data_t[appid].isnull().all():
                appids_to_remove.append(appid)
        print('before removing rows with all none in text cols, we have', len(self.df.index))
        self.df = self.df.drop(appids_to_remove, axis=0)
        print('after removing rows with all none in text cols, we have', len(self.df.index))
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def impute_text_cols(self):
        """
        impute the missing panels using its non-missing panels
        """
        cols = [self.tcn + '_' + item for item in self.all_panels]
        for j in range(len(cols)):
            self.df[cols[j]] = self.df[cols[j]].fillna('')
            if j == 0:
                self.df['all_panel_'+self.tcn] = self.df[cols[j]]
            else:
                self.df['all_panel_' + self.tcn] = self.df['all_panel_'+self.tcn] + self.df[cols[j]]
        all_text_col = self.df['all_panel_' + self.tcn]
        null_data = all_text_col[all_text_col.isnull()]
        if len(null_data) == 0:
            print('successfully imputed missing text columns for TEXT COLUMN (combined from all panels) of the dataset', self.initial_panel)
            print('assuming all text columns are time-invariant')
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def remove_stopwords(self, text):
        text = text.lower()
        tokens_without_sw = [word for word in text.split() if word not in nlp_pipeline.stoplist]
        filtered_sentence = (" ").join(tokens_without_sw)
        return filtered_sentence

    def prepare_text_col(self):  # use take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
        """
        # _________________ process text __________________________________________________
        # Adding ^ in []  excludes any character in
        # the set. Here, [^ab5] it matches characters that are
        # not a, b, or 5.
        """
        self.df['clean_all_panel_' + self.tcn] = self.df['all_panel_' + self.tcn]
        self.df['clean_all_panel_' + self.tcn] = self.df['clean_all_panel_' + self.tcn].apply(
            lambda x: re.sub(r'[^\w\s]', '', x)).apply(
            lambda x: re.sub(r'[0-9]', '', x)).apply(
            lambda x: self.remove_stopwords(x))
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    ################# Divide processed df with processed and clean text column into subsamples ################
    def divide_into_subsamples(self):
        """
        run tnis after self.prepare_text_col and self.impute_text_cols and self.find_appids_to_remove_before_imputing
        """
        self.df['genreIdGame'] = self.df['ImputedgenreId_' + self.all_panels[-1]].apply(lambda x: 1 if 'GAME' in x else 0)
        df2 = self.df.copy(deep=True)
        df_game = df2.loc[df2['genreIdGame'] == 1]
        df_nongame = df2.loc[df2['genreIdGame'] == 0]
        self.game_subsamples = {'game': df_game,
                                'nongame': df_nongame}
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def generate_save_input_text_col(self):
        """
        Purpose of creating this cell is to avoid creating run_subsample switch in each function below.
        Everytime you run the full sample NLP, you run the sub samples NLP simultaneously, it would take longer, but anyways.
        """
        full_sample = self.df['clean_all_panel_' + self.tcn].copy(deep=True)
        game_subsample = self.game_subsamples['game']['clean_all_panel_' + self.tcn].copy(deep=True)
        nongame_subsample = self.game_subsamples['nongame']['clean_all_panel_' + self.tcn].copy(deep=True)
        self.input_text_cols = {'full': full_sample,
                                'game': game_subsample,
                                'nongame': nongame_subsample}
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def tf_idf_transformation(self):
        pipe = Pipeline(steps=[('tfidf',
                                TfidfVectorizer(
                                    stop_words='english',
                                    strip_accents='unicode',
                                    max_features=2000))])
        matrix_df_dict = {}
        for sample, input_text_col in self.input_text_cols.items():
            matrix = pipe.fit_transform(input_text_col)
            print(type(matrix))
            # transform the matrix to matrix dataframe
            matrix_df = pd.DataFrame(matrix.toarray(),
                                     columns=pipe['tfidf'].get_feature_names())
            matrix_df['app_ids'] = input_text_col.index.tolist()
            matrix_df.set_index('app_ids', inplace=True)
            matrix_df_dict[sample] = matrix_df
        self.tf_idf_matrices = matrix_df_dict
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def find_optimal_svd_component_plot(self):
        # https://medium.com/swlh/truncated-singular-value-decomposition-svd-using-amazon-food-reviews-891d97af5d8d
        n_comp = [4, 10, 15, 20, 50, 100, 150, 200, 500, 700, 800, 900, 1000, 1100, 1200, 1300,
                  1400, 1500, 1600, 1700, 1800, 1900]  # list containing different values of components
        for sample, matrix in self.tf_idf_matrices.items():
            explained = []
            for x in tqdm(n_comp):
                svd = TruncatedSVD(n_components=x)
                svd.fit(matrix)
                explained.append(svd.explained_variance_ratio_.sum())
                print("Number of components = %r and explained variance = %r" % (x, svd.explained_variance_ratio_.sum()))
            fig, ax = plt.subplots()
            ax.plot(n_comp, explained)
            ax.grid()
            plt.xlabel('Number of components')
            plt.ylabel("Explained Variance")
            plt.title(sample + " Plot of Number of components v/s explained variance")
            filename = self.initial_panel + '_' + sample + '_optimal_svd_graph.png'
            fig.savefig(nlp_pipeline.nlp_graph_path / 'optimal_svd_comp' / filename, facecolor='white', dpi=300)
            plt.show()
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def truncate_svd(self, n_comp, random_state):
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)
        matrix_df_dict = {}
        for sample, matrix in self.tf_idf_matrices.items():
            matrix_transformed = svd.fit_transform(matrix)
            print(sample)
            print(matrix_transformed.shape)
            matrix_transformed_df = pd.DataFrame(matrix_transformed)  # do not need to assign column names because those do not correspond to each topic words (they have been transformed)
            matrix_transformed_df['app_ids'] = matrix.index.tolist()
            matrix_transformed_df.set_index('app_ids', inplace=True)
            matrix_df_dict[sample] = matrix_transformed_df
        self.svd_matrices = matrix_df_dict
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def find_optimal_cluster_plot(self):
        """
        https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        """
        n_cluster_list = [25, 50, 75, 100, 125, 150, 175, 200, 225, 250, 275, 300, 325, 350, 375, 400]
        for sample, matrix in self.svd_matrices.items():
            Sum_of_squared_distances = []
            for k in n_cluster_list:
                km = KMeans(n_clusters=k)
                km = km.fit(matrix)
                Sum_of_squared_distances.append(km.inertia_)
            fig, ax = plt.subplots()
            ax.plot(n_cluster_list, Sum_of_squared_distances, 'bx-')
            ax.grid()
            plt.xlabel('k')
            plt.ylabel('Sum_of_squared_distances')
            plt.title(sample + ' Elbow Method For Optimal k')
            filename = self.initial_panel + '_' + sample + '_optimal_cluster_of_svd_matrix.png'
            fig.savefig(nlp_pipeline.nlp_graph_path / 'optimal_clusters' / filename, facecolor='white', dpi=300)
            plt.show()
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def kmeans_cluster(self,
                       n_clusters_dict,
                       init,
                       random_state):
        labels_dict = {}
        for sample, matrix in self.svd_matrices.items():
            kmeans = KMeans(n_clusters=n_clusters_dict[sample], init=init, random_state=random_state)
            y_kmeans = kmeans.fit_predict(matrix)  # put matrix_transformed_df here would generate same result as put matrix_transformed
            matrix[sample + '_kmeans_labels'] = y_kmeans
            label_single = matrix[[sample + '_kmeans_labels']]
            labels_dict[sample]=label_single
        self.output_labels = labels_dict
        # --------------------------- save -------------------------------------------------
        # for this one, you do not need to run text cluster label every month when you scraped new data, because they would more or less stay the same
        filename = self.initial_panel + '_predicted_labels_dict.pickle'
        q = nlp_pipeline.label_df_path / 'predicted_text_labels' / filename
        pickle.dump(self.output_labels, open(q, 'wb'))
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)
