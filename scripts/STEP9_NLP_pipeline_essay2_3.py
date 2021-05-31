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
from sklearn.metrics import silhouette_score
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

class nlp_pipeline_essay_2_3():
    """
    2021-05-29
    Generating the niche scale within the leader / non-leader and 5 categories within each of them.
    """
    nlp_graph_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2___/nlp/graphs')
    nlp_graph_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_3___/nlp/graphs')
    panel_essay_2_3_common_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_3_common_panels___')
    panel_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_panels___')
    panel_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_3_panels___')
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    # after examining the optimal svd cluster graphs, I write dict here as class attributes for each dataset
    # ------------------------------------------------------------------------
    optimal_svd_components_201907 = {
        'Leaders': {'full': 860,
                'category_GAME': 630,
                'category_BUSINESS': 210,
                'category_SOCIAL': 200,
                'category_LIFESTYLE': 300,
                'category_MEDICAL': 65},
        'Non-leaders': {'full': 1150,
                    'category_GAME': 950,
                    'category_BUSINESS': 750,
                    'category_SOCIAL': 700,
                    'category_LIFESTYLE': 850,
                    'category_MEDICAL': 350}}

    # after examining the optimal km cluster graphs (both elbow and sihouette graphs),
    # I write dict here as class attributes for each dataset
    # ------------------------------------------------------------------------
    optimal_km_clusters_201907 = {
        'Leaders': {'full': 130,
                    'category_GAME': 50,
                    'category_BUSINESS': 47,
                    'category_SOCIAL': 40,
                    'category_LIFESTYLE': 75,
                    'category_MEDICAL': 8},
        'Non-leaders': {'full': 300,
                        'category_GAME': 50,
                        'category_BUSINESS': 100,
                        'category_SOCIAL': 100,
                        'category_LIFESTYLE': 130,
                        'category_MEDICAL': 85}}

    def __init__(self,
                 tcn,
                 initial_panel,
                 all_panels,
                 df=None,
                 sub_sample_text_cols=None,
                 tf_idf_matrices=None,
                 svd_matrices=None,
                 output_labels=None):
        self.tcn = tcn
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.ss_text_cols = sub_sample_text_cols
        self.tf_idf_matrices = tf_idf_matrices
        self.svd_matrices = svd_matrices
        self.output_labels = output_labels

    def open_divided_df(self):
        f_name = self.initial_panel + '_imputed_deleted_subsamples.pickle'
        q = nlp_pipeline_essay_2_3.panel_essay_2_3_common_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def generate_save_input_text_col(self):
        """
        leaders -- five categories
        non-leaders -- five categories
        """
        # --------------- compile text cols ----------------------------------------------------------
        self.ss_text_cols = {'Leaders': {'full': None,
                                         'category_GAME': None,
                                         'category_BUSINESS': None,
                                         'category_SOCIAL': None,
                                         'category_LIFESTYLE': None,
                                         'category_MEDICAL': None},
                             'Non-leaders': {'full': None,
                                             'category_GAME': None,
                                             'category_BUSINESS': None,
                                             'category_SOCIAL': None,
                                             'category_LIFESTYLE': None,
                                             'category_MEDICAL': None}}
        # --------------- full ------------------------------------------------------
        for subsample1, content1 in self.ss_text_cols.items():
            for subsample2, content2 in content1.items():
                if subsample2 == 'full':
                    self.ss_text_cols[subsample1][subsample2] = self.df.loc[
                        self.df[subsample1] == 1,
                        self.tcn + 'ModeClean'].copy(deep=True)
                else:
                    self.ss_text_cols[subsample1][subsample2] = self.df.loc[
                        (self.df[subsample1] == 1) & (self.df[subsample2] == 1),
                        self.tcn + 'ModeClean'].copy(deep=True)
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def tf_idf_transformation(self):
        pipe = Pipeline(steps=[('tfidf',
                                TfidfVectorizer(
                                    stop_words='english',
                                    strip_accents='unicode',
                                    max_features=1500))])
        matrix_df_dict = dict.fromkeys(self.ss_text_cols.keys())
        for sample, content in matrix_df_dict.items():
            matrix_df_dict[sample] = dict.fromkeys(self.ss_text_cols[sample].keys())
        for sample, content in self.ss_text_cols.items():
            for ss_name, col in content.items():
                print('TF-IDF TRANSFORMATION')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                matrix = pipe.fit_transform(col)
                matrix_df = pd.DataFrame(matrix.toarray(),
                                         columns=pipe['tfidf'].get_feature_names())
                matrix_df['app_ids'] = col.index.tolist()
                matrix_df.set_index('app_ids', inplace=True)
                matrix_df_dict[sample][ss_name] = matrix_df
                print(matrix_df.shape)
        self.tf_idf_matrices = matrix_df_dict
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def find_optimal_svd_component_plot(self):
        """
        https://medium.com/swlh/truncated-singular-value-decomposition-svd-using-amazon-food-reviews-891d97af5d8d
        https://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html
        """
        for sample, content in self.tf_idf_matrices.items():
            for ss_name, matrix in content.items():
                print('FIND OPTIMAL SVD COMPONENTS')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                n_comp = np.round(np.linspace(0, matrix.shape[1]-1, 20))
                n_comp = n_comp.astype(int)
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
                plt.title(self.initial_panel + ' ' + sample + ' ' + ss_name + " Plot of Number of components v/s explained variance")
                filename = self.initial_panel + '_' + sample + '_' + ss_name + '_optimal_svd_graph.png'
                if sample == 'Leaders':
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_3_path / 'optimal_svd_comp' / filename,
                                facecolor='white', dpi=300)
                else:
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_2_path / 'optimal_svd_comp' / filename,
                                facecolor='white', dpi=300)
                plt.show()
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def truncate_svd(self, random_state):
        matrix_df_dict = dict.fromkeys(self.ss_text_cols.keys())
        for sample, content in matrix_df_dict.items():
            matrix_df_dict[sample] = dict.fromkeys(self.ss_text_cols[sample].keys())
        for sample, content in self.tf_idf_matrices.items():
            for ss_name, matrix in content.items():
                print('TRUNCATE SVD')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                svd = TruncatedSVD(n_components=nlp_pipeline_essay_2_3.optimal_svd_components_201907[sample][ss_name],
                                   random_state=random_state)
                matrix_transformed = svd.fit_transform(matrix)
                print(matrix_transformed.shape)
                matrix_transformed_df = pd.DataFrame(matrix_transformed)  # do not need to assign column names because those do not correspond to each topic words (they have been transformed)
                matrix_transformed_df['app_ids'] = matrix.index.tolist()
                matrix_transformed_df.set_index('app_ids', inplace=True)
                matrix_df_dict[sample][ss_name] = matrix_transformed_df
        self.svd_matrices = matrix_df_dict
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def find_optimal_cluster_elbow(self):
        """
        https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        https://scikit-learn.org/stable/modules/clustering.html
        """
        for sample, content in self.svd_matrices.items():
            for ss_name, matrix in content.items():
                print('ELBOW METHOD TO FIND OPTIMAL KM CLUSTERS')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                # note here, for svd comps, the maximum components are the total number of features (columns)
                # here, the total number of clusters are the total number of apps (at extreme, 1 app per cluster)
                # starts from 1 because this is only within cluster sum of squared distances
                # the maximum number of clusters is controlled below 1/5 of total number of points because first it would
                # take extremely long time to compute without controlling for maximum number of clusters, second,
                # it is reasonable to assume that one would not want only 4 points in a cluster.
                n_cluster_list = np.round(np.linspace(1, matrix.shape[0] - 0.8 * matrix.shape[0], 10))
                n_cluster_list = n_cluster_list.astype(int)
                sum_of_squared_distances = []
                for k in tqdm(n_cluster_list):
                    km = KMeans(n_clusters=k)
                    km = km.fit(matrix)
                    sum_of_squared_distances.append(km.inertia_)
                fig, ax = plt.subplots()
                ax.plot(n_cluster_list, sum_of_squared_distances, 'bx-')
                ax.grid()
                plt.xlabel('k')
                plt.ylabel('Sum_of_squared_distances')
                plt.title(self.initial_panel + ' ' + sample + ' ' + ss_name + ' Elbow Method For Optimal k')
                filename = self.initial_panel + '_' + sample + '_' + ss_name + '_elbow_optimal_cluster.png'
                if sample == 'Leaders':
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_3_path / 'optimal_clusters' / filename,
                                facecolor='white', dpi=300)
                else:
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_2_path / 'optimal_clusters' / filename,
                                facecolor='white', dpi=300)
                plt.show()
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def find_optimal_cluster_silhouette(self):
        """
        https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
        https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
        """
        for sample, content in self.svd_matrices.items():
            for ss_name, matrix in content.items():
                print('SILHOUETTE SCORE TO FIND OPTIMAL KM CLUSTERS')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                # starting from 2 because this score need to calculate between cluster estimators
                n_cluster_list = np.round(np.linspace(2, matrix.shape[0] - 0.8 * matrix.shape[0], 10))
                n_cluster_list = n_cluster_list.astype(int)
                silhouette_scores = []
                for k in tqdm(n_cluster_list):
                    km = KMeans(n_clusters=k)
                    km = km.fit(matrix)
                    labels = km.labels_
                    silhouette_scores.append(silhouette_score(matrix, labels, metric='cosine'))
                fig, ax = plt.subplots()
                ax.plot(n_cluster_list, silhouette_scores, 'bx-')
                ax.grid()
                plt.xlabel('k')
                plt.ylabel('silhouette_scores (cosine distance)')
                plt.title(self.initial_panel + ' ' + sample + ' ' + ss_name + ' Silhouette Scores For Optimal k')
                filename = self.initial_panel + '_' + sample + '_' + ss_name + '_silhouette_optimal_cluster.png'
                if sample == 'Leaders':
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_3_path / 'optimal_clusters' / filename,
                                facecolor='white', dpi=300)
                else:
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_2_path / 'optimal_clusters' / filename,
                                facecolor='white', dpi=300)
                plt.show()
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)

    def kmeans_cluster(self,
                       init,
                       random_state):
        label_dict = dict.fromkeys(self.ss_text_cols.keys())
        for sample, content in label_dict.items():
            label_dict[sample] = dict.fromkeys(self.ss_text_cols[sample].keys())
        for sample, content in self.svd_matrices.items():
            for ss_name, matrix in content.items():
                print('KMEANS CLUSTER')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                kmeans = KMeans(n_clusters=nlp_pipeline_essay_2_3.optimal_km_clusters_201907[sample][ss_name],
                                init=init,
                                random_state=random_state)
                y_kmeans = kmeans.fit_predict(matrix)  # put matrix_transformed_df here would generate same result as put matrix_transformed
                matrix[sample + '_' + ss_name + '_kmeans_labels'] = y_kmeans
                label_single = matrix[[sample + '_' + ss_name + '_kmeans_labels']]
                label_dict[sample][ss_name] = label_single
        self.output_labels = label_dict
        # --------------------------- save -------------------------------------------------
        # for this one, you do not need to run text cluster label every month when you scraped new data, because they would more or less stay the same
        filename = self.initial_panel + '_predicted_labels_dict.pickle'
        q = nlp_pipeline_essay_2_3.panel_essay_2_3_common_path / 'predicted_text_labels' / filename
        pickle.dump(self.output_labels, open(q, 'wb'))
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 tf_idf_matrices=self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)
