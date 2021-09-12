import warnings
warnings.filterwarnings('ignore')
from pathlib import Path
import pickle
import copy
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
    nlp_stats_essay_2_3_common_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/___essay_2_3_common___/nlp/stats')
    panel_essay_2_3_common_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_3_common_panels___')
    panel_essay_2_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_2_panels___')
    panel_essay_3_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__/___essay_3_panels___')
    tokenizer = nlp.Defaults.create_tokenizer(nlp)

    def __init__(self,
                 tcn,
                 initial_panel,
                 all_panels,
                 ssvars=None,
                 df=None,
                 sub_sample_text_cols=None,
                 tf_idf_matrices=None,
                 optimal_svd_dict=None,
                 svd_matrices=None,
                 optimal_k_cluster_dict=None,
                 output_labels=None):
        self.tcn = tcn
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.ssvars = ssvars
        self.df = df
        self.ss_text_cols = sub_sample_text_cols
        self.tf_idf_matrices = tf_idf_matrices
        self.optimal_svd_dict = optimal_svd_dict
        self.svd_matrices = svd_matrices
        self.optimal_k_cluster_dict = optimal_k_cluster_dict
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
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def create_subsample_slice_vars(self):
        sub_categories = ['full', 'category_GAME', 'category_BUSINESS', 'category_SOCIAL',
                          'category_LIFESTYLE', 'category_MEDICAL']
        self.ssvars = {'Leaders': sub_categories,
                       'Non-leaders': sub_categories}
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def slice_text_cols_for_sub_samples(self):
        """
        leaders -- five categories
        non-leaders -- five categories
        """
        d = dict.fromkeys(self.ssvars.keys())
        for subsample1, content1 in self.ssvars.items():
            d[subsample1] = dict.fromkeys(content1)
            for subsample2 in content1:
                if subsample2 == 'full':
                    d[subsample1][subsample2] = self.df.loc[
                        self.df[subsample1] == 1,
                        self.tcn + 'ModeClean'].copy(deep=True)
                else:
                    d[subsample1][subsample2] = self.df.loc[
                        (self.df[subsample1] == 1) & (self.df[subsample2] == 1),
                        self.tcn + 'ModeClean'].copy(deep=True)
        self.ss_text_cols = d
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
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
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
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
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def find_optimal_svd_component_dict(self, cutoff_percent_explained):
        d = dict.fromkeys(self.ssvars)
        for k, ss in self.ssvars.items():
            d[k] = dict.fromkeys(ss)
            for i in ss:
                print('FIND OPTIMAL SVD COMPONENTS')
                matrix = self.tf_idf_matrices[k][i]
                n_comp = np.round(np.linspace(0, matrix.shape[1] - 1, 40))
                n_comp = n_comp.astype(int)
                x = 0
                while x <= len(n_comp)-1:
                    svd = TruncatedSVD(n_components=n_comp[x])
                    svd.fit(matrix)
                    print(self.initial_panel, ' -- ', k, ' -- ', i)
                    print('Number of Components: ', n_comp[x])
                    print('Explained Variance Ratio: ', svd.explained_variance_ratio_.sum())
                    if svd.explained_variance_ratio_.sum() < cutoff_percent_explained:
                        x += 1 # continue the while loop to test next ncomp
                    else:
                        d[k][i] = n_comp[x]
                        print("The Optimal SVD Component is = %r and the explained variance = %r" % (
                        n_comp[x], svd.explained_variance_ratio_.sum()))
                        x = len(n_comp) # set the x value so to break the while loop
        self.optimal_svd_dict = d
        # ----------------- save -----------------------------------------------
        filename = self.initial_panel + '_optimal_svd_dict.pickle'
        q = nlp_pipeline_essay_2_3.nlp_stats_essay_2_3_common_path / filename
        pickle.dump(d, open(q, 'wb'))
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def truncate_svd(self, random_state):
        f_name = self.initial_panel + '_optimal_svd_dict.pickle'
        q = nlp_pipeline_essay_2_3.nlp_stats_essay_2_3_common_path / f_name
        with open(q, 'rb') as f:
            self.optimal_svd_dict = pickle.load(f)
        # -------------------------------------------------------------------------
        d = dict.fromkeys(self.tf_idf_matrices.keys())
        for sample, content in self.tf_idf_matrices.items():
            d[sample] = dict.fromkeys(content.keys())
            for ss_name, matrix in content.items():
                print('TRUNCATE SVD')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                svd = TruncatedSVD(n_components=self.optimal_svd_dict[sample][ss_name],
                                   random_state=random_state)
                matrix_transformed = svd.fit_transform(matrix)
                print(matrix_transformed.shape)
                matrix_transformed_df = pd.DataFrame(matrix_transformed)  # do not need to assign column names because those do not correspond to each topic words (they have been transformed)
                matrix_transformed_df['app_ids'] = matrix.index.tolist()
                matrix_transformed_df.set_index('app_ids', inplace=True)
                d[sample][ss_name] = matrix_transformed_df
        self.svd_matrices = d
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def optimal_k_silhouette(self):
        """
        https://medium.com/analytics-vidhya/how-to-determine-the-optimal-k-for-k-means-708505d204eb
        https://medium.com/@kunal_gohrani/different-types-of-distance-metrics-used-in-machine-learning-e9928c5e26c7
        I think it is just better off using sihouette score because global maximum is better to find out than the elbow point.
        The Silhouette Coefficient is calculated using the mean intra-cluster distance (a)
        and the mean nearest-cluster distance (b) for each sample. The Silhouette Coefficient for a sample is (b - a) / max(a, b).
        To clarify, b is the distance between a sample and the nearest cluster that the sample is not a part of.
        Note that Silhouette Coefficient is only defined if number of labels is 2 <= n_labels <= n_samples - 1.
        """
        d = dict.fromkeys(self.ssvars.keys())
        for sample, content in self.svd_matrices.items():
            d[sample] = dict.fromkeys(content.keys())
            for ss_name, matrix in content.items():
                # starting from 2 because this score need to calculate between cluster estimators
                n_cluster_list = np.round(np.linspace(2, matrix.shape[0] - 0.9 * matrix.shape[0], 10))
                n_cluster_list = n_cluster_list.astype(int)
                silhouette_scores = []
                silhouette_scores_dict = {}
                print('SILHOUETTE SCORE -- ', self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                for k in tqdm(n_cluster_list):
                    km = KMeans(n_clusters=k)
                    km = km.fit(matrix)
                    labels = km.labels_
                    s_score = silhouette_score(matrix, labels, metric='cosine')
                    silhouette_scores.append(s_score)
                    silhouette_scores_dict[k] = s_score
                d[sample][ss_name] = silhouette_scores_dict
                fig, ax = plt.subplots()
                ax.plot(n_cluster_list, silhouette_scores, 'bx-')
                ax.grid()
                plt.xlabel('k')
                plt.ylabel('silhouette_scores (cosine distance)')
                plt.title(self.initial_panel + sample + ss_name + ' Silhouette Scores For Optimal k')
                filename = self.initial_panel + '_' + sample + '_' + ss_name + '_silhouette_optimal_cluster.png'
                if sample == 'Leaders':
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_3_path / 'optimal_clusters' / filename,
                                facecolor='white', dpi=300)
                else:
                    fig.savefig(nlp_pipeline_essay_2_3.nlp_graph_essay_2_path / 'optimal_clusters' / filename,
                                facecolor='white', dpi=300)
                plt.show()
        # ----------------- save -----------------------------------------------
        dict_f_name = self.initial_panel + '_silhouette_score_dict.pickle'
        q = nlp_pipeline_essay_2_3.nlp_stats_essay_2_3_common_path / dict_f_name
        pickle.dump(d, open(q, 'wb'))
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def determine_optimal_k_from_silhouette(self):
        dict_f_name = self.initial_panel + '_silhouette_score_dict.pickle'
        q = nlp_pipeline_essay_2_3.nlp_stats_essay_2_3_common_path / dict_f_name
        with open(q, 'rb') as f:
            res = pickle.load(f)
        d = dict.fromkeys(self.ssvars.keys())
        for sample1, content in self.ssvars.items():
            d[sample1] = dict.fromkeys(content)
            for sample2 in content:
                df = copy.deepcopy(res[sample1][sample2])
                df2 = pd.DataFrame(df, index=[0])
                df3 = df2.T
                optimal_k = df3.idxmax(axis=0)
                print(self.initial_panel, ' -- ', sample1, ' -- ', sample2, ' -- ',
                      ' Optimal K From Global Max of Silhouette Score')
                print(optimal_k)
                print()
                d[sample1][sample2] = optimal_k
        self.optimal_k_cluster_dict = d
        # ----------------- save -----------------------------------------------
        dict_f_name = self.initial_panel + '_optimal_k_from_global_max_of_silhouette_score.pickle'
        q = nlp_pipeline_essay_2_3.nlp_stats_essay_2_3_common_path / dict_f_name
        pickle.dump(d, open(q, 'wb'))
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def open_optimal_k(self):
        dict_f_name = self.initial_panel + '_optimal_k_from_global_max_of_silhouette_score.pickle'
        q = nlp_pipeline_essay_2_3.nlp_stats_essay_2_3_common_path / dict_f_name
        with open(q, 'rb') as f:
            self.optimal_k_cluster_dict = pickle.load(f)
        return nlp_pipeline_essay_2_3(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 sub_sample_text_cols=self.ss_text_cols,
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)

    def kmeans_cluster(self,
                       random_state):
        label_dict = dict.fromkeys(self.svd_matrices.keys())
        for sample, content in self.svd_matrices.items():
            label_dict[sample] = dict.fromkeys(self.svd_matrices[sample].keys())
            for ss_name, matrix in content.items():
                print('KMEANS CLUSTER')
                print(self.initial_panel, ' -- ', sample, ' -- ', ss_name)
                print('input matrix shape')
                print(matrix.shape)
                print('optimal k clusters')
                k = self.optimal_k_cluster_dict[sample][ss_name]
                print(k)
                y_kmeans = KMeans(
                    n_clusters=int(k),
                    random_state=random_state
                ).fit_predict(
                    matrix
                )  # it is equivalent as using fit then .label_.
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
                 ssvars=self.ssvars,
                 tf_idf_matrices=self.tf_idf_matrices,
                 optimal_svd_dict=self.optimal_svd_dict,
                 svd_matrices=self.svd_matrices,
                 optimal_k_cluster_dict=self.optimal_k_cluster_dict,
                 output_labels=self.output_labels)
