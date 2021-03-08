import warnings
warnings.filterwarnings('ignore')
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
from sklearn.cluster import KMeans
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
    # the input dataframe is by default appid index
    def __init__(self,
                 df,
                 text_col_name,
                 tokenizer=nlp.Defaults.create_tokenizer(nlp),
                 initial_panel=None,
                 all_panels=None,
                 consec_panels=None,
                 stoplist = nltk.corpus.stopwords.words('english')):
        self.df = df
        self.tcn = text_col_name
        self.tokenizer = tokenizer
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
        self.stoplist = stoplist

    def select_text_cols(self, consecutive):
        if consecutive is True:
            cols = [self.tcn + '_' + item for item in self.consec_panels]
        else:
            cols = [self.tcn + '_' + item for item in self.all_panels]
        new_df = self.df[cols]
        return new_df

    def select_text_cols_into_list(self, consecutive): # functional tool operation, mostly for combined panel process
        if consecutive is True:
            cols = [self.tcn + '_' + item for item in self.consec_panels]
        else:
            cols = [self.tcn + '_' + item for item in self.all_panels]
        new_df_list = []
        for j in cols:
            new_df = self.df[j].to_frame()
            new_df_list.append(new_df)
        return new_df_list

    def select_text_cols_into_dict(self, consecutive): # for single panel processes
        if consecutive is True:
            new_df_dict = dict.fromkeys(self.consec_panels)
        else:
            new_df_dict = dict.fromkeys(self.all_panels)
        for panel_name in new_df_dict.keys():
            new_df_dict[panel_name] = self.df[self.tcn + '_' + panel_name].to_frame()
        return new_df_dict

    def remove_stopwords(self, text):
        text = text.lower()
        tokens_without_sw = [word for word in text.split() if word not in self.stoplist]
        filtered_sentence = (" ").join(tokens_without_sw)
        return filtered_sentence

    def prepare_text_col(self, consecutive=True, combine_panels=True):  # use take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
        """
        # _________________ process text __________________________________________________
        # Adding ^ in []  excludes any character in
        # the set. Here, [^ab5] it matches characters that are
        # not a, b, or 5.
        """
        if combine_panels is True: # here combine all panels' text together to create time invariant text cluster labels
            new_df_list = self.select_text_cols_into_list(consecutive=consecutive)
            single_text_col = functools.reduce(lambda a, b: a.squeeze().fillna('') + b.squeeze().fillna(''), new_df_list)
            single_text_col = single_text_col.apply(
                lambda x: re.sub(r'[^\w\s]', '', x)).apply(
                lambda x: re.sub(r'[0-9]', '', x)).apply(
                lambda x: self.remove_stopwords(x))
            return single_text_col
        else: # here do NOT combine, create a text cluster label for each panel; it implies that text cluster label is time Variant, this is important for FE regression
            new_df_dict = self.select_text_cols_into_dict(consecutive=consecutive)
            if consecutive is True:
                text_col_dict = dict.fromkeys(self.consec_panels)
            else:
                text_col_dict = dict.fromkeys(self.all_panels)
            for panel_name, single_text_col in new_df_dict.items():
                single_text_col = single_text_col.fillna('')
                single_text_col[single_text_col.columns[0]] = single_text_col[single_text_col.columns[0]].apply(
                    lambda x: re.sub(r'None', '', x)).apply(
                    lambda x: re.sub(r'[^\w\s]', '', x)).apply(
                    lambda x: re.sub(r'[0-9]', '', x)).apply(
                    lambda x: self.remove_stopwords(x))
                print(panel_name)
                text_col_dict[panel_name] = single_text_col
            return text_col_dict

    def tf_idf_transformation_single(self, input_pd_series):
        pipe = Pipeline(steps=[('tfidf',
                                TfidfVectorizer(
                                    stop_words='english',
                                    strip_accents='unicode',
                                    max_features=2000))])
        matrix = pipe.fit_transform(input_pd_series)
        print(type(matrix))
        # transform the matrix to matrix dataframe
        matrix_df = pd.DataFrame(matrix.toarray(),
                                 columns=pipe['tfidf'].get_feature_names())
        matrix_df['app_ids'] = input_pd_series.index.tolist()
        matrix_df.set_index('app_ids', inplace=True)
        return matrix_df

    def tf_idf_transformation(self,
                              consecutive=True,
                              combine_panels=True):
        if combine_panels is True:
            single_text_col = self.prepare_text_col(consecutive=consecutive,
                                                    combine_panels=combine_panels)
            return self.tf_idf_transformation_single(single_text_col)
        else:
            text_col_dict = self.prepare_text_col(consecutive=consecutive,
                                                  combine_panels=combine_panels)
            if consecutive is True:
                matrix_dict = dict.fromkeys(self.consec_panels)
            else:
                matrix_dict = dict.fromkeys(self.all_panels)
            for panel_name, col in text_col_dict.items():
                print(panel_name)
                matrix_dict[panel_name] = self.tf_idf_transformation_single(col.squeeze())
            return matrix_dict

    def find_optimal_svd_component_plot_single(self, matrix):
        # https://medium.com/swlh/truncated-singular-value-decomposition-svd-using-amazon-food-reviews-891d97af5d8d
        n_comp = [4, 10, 15, 20, 50, 100, 150, 200, 500, 700, 800, 900, 1000, 1100, 1200, 1300,
                  1400, 1500, 1600, 1700, 1800, 1900]  # list containing different values of components
        explained = []  # explained variance ratio for each component of Truncated SVD
        for x in tqdm(n_comp):
            svd = TruncatedSVD(n_components=x)
            svd.fit(matrix)
            explained.append(svd.explained_variance_ratio_.sum())
            print("Number of components = %r and explained variance = %r" % (x, svd.explained_variance_ratio_.sum()))
        plt.plot(n_comp, explained)
        plt.xlabel('Number of components')
        plt.ylabel("Explained Variance")
        plt.title("Plot of Number of components v/s explained variance")
        plt.show()

    def find_optimal_svd_component_plot(self,
                                        consecutive=True,
                                        combine_panels=True):
        if combine_panels is True:
            matrix = self.tf_idf_transformation(
                consecutive=consecutive,
                combine_panels=combine_panels)
            self.find_optimal_svd_component_plot_single(matrix=matrix)
        else:
            matrix_dict = self.tf_idf_transformation(
                consecutive=consecutive,
                combine_panels=combine_panels
            )
            for panel_name, matrix in matrix_dict.items():
                print(panel_name)
                self.find_optimal_svd_component_plot_single(matrix=matrix)

    def truncate_svd_single(self, matrix, n_comp, random_state):
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)  # specify random state for the results to be replicable
        matrix_transformed = svd.fit_transform(
            matrix)  # matrix is the output of self.tf_idf_transformation
        print(matrix_transformed.shape)
        matrix_transformed_df = pd.DataFrame(matrix_transformed)  # do not need to assign column names because those do not correspond to each topic words (they have been transformed)
        matrix_transformed_df['app_ids'] = matrix.index.tolist()
        matrix_transformed_df.set_index('app_ids', inplace=True)
        return matrix_transformed_df

    def truncate_svd(self,
                     n_comp,
                     random_state,
                     consecutive=True,
                     combine_panels=True):
        if combine_panels is True:
            matrix = self.tf_idf_transformation(
                consecutive=consecutive,
                combine_panels=combine_panels)
            return self.truncate_svd_single(matrix, n_comp, random_state)
        else:
            if consecutive is True:
                svd_matrix_dict = dict.fromkeys(self.consec_panels)
            else:
                svd_matrix_dict = dict.fromkeys(self.all_panels)
            matrix_dict = self.tf_idf_transformation(
                consecutive=consecutive,
                combine_panels=combine_panels
            )
            for panel_name, matrix in matrix_dict.items():
                print(panel_name)
                svd_matrix_dict[panel_name] = self.truncate_svd_single(matrix, n_comp, random_state)
            return svd_matrix_dict

    def find_optimal_cluster_plot_single(self, n_cluster_list, matrix):
        """
        https://blog.cambridgespark.com/how-to-determine-the-optimal-number-of-clusters-for-k-means-clustering-14f27070048f
        """
        Sum_of_squared_distances = []
        for k in n_cluster_list:
            km = KMeans(n_clusters=k)
            km = km.fit(matrix)
            Sum_of_squared_distances.append(km.inertia_)
        plt.plot(n_cluster_list, Sum_of_squared_distances, 'bx-')
        plt.xlabel('k')
        plt.ylabel('Sum_of_squared_distances')
        plt.title('Elbow Method For Optimal k')
        plt.show()

    def find_optimal_cluster_plot(self, n_cluster_list, n_comp, random_state, consecutive=True, combine_panels=True):
        if combine_panels is True:
            svd_matrix = self.truncate_svd(
                  n_comp=n_comp,
                  random_state=random_state,
                  consecutive=consecutive,
                  combine_panels=combine_panels)
            self.find_optimal_cluster_plot_single(n_cluster_list, svd_matrix)
        else:
            svd_matrix_dict = self.truncate_svd(
                n_comp=n_comp,
                random_state=random_state,
                consecutive=consecutive,
                combine_panels=combine_panels)
            for panel_name, svd_matrix in svd_matrix_dict.items():
                print(panel_name)
                self.find_optimal_cluster_plot_single(n_cluster_list, svd_matrix)

    def kmeans_cluster(self,
                       n_clusters,
                       init,
                       random_state,
                       n_comp,
                       consecutive=True,
                       combine_panels=True):
        if combine_panels is True:
            svd_matrix = self.truncate_svd(
                  n_comp=n_comp,
                  random_state=random_state,
                  consecutive=consecutive,
                  combine_panels=combine_panels)
            kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=random_state)
            y_kmeans = kmeans.fit_predict(svd_matrix)  # put matrix_transformed_df here would generate same result as put matrix_transformed
            svd_matrix['combined_panels_kmeans_labels'] = y_kmeans
            label_col = svd_matrix[['combined_panels_kmeans_labels']]
            return label_col
        else:
            svd_matrix_dict = self.truncate_svd(
                n_comp=n_comp,
                random_state=random_state,
                consecutive=consecutive,
                combine_panels=combine_panels)
            pure_label_list = []
            for panel_name, svd_matrix in svd_matrix_dict.items():
                print(panel_name)
                kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=random_state)
                y_means_for_this_matrix = kmeans.fit_predict(svd_matrix) # put matrix_transformed_df here would generate same result as put matrix_transformed
                svd_matrix[panel_name+'_kmeans_labels'] = y_means_for_this_matrix
                pure_label_list.append(svd_matrix[[panel_name+'_kmeans_labels']])
            label_df = functools.reduce(lambda a, b: a.join(b, how='inner'), pure_label_list)
            return label_df
