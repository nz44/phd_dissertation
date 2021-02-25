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
from sklearn.random_projection import sparse_random_matrix
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
                 consec_panels=None):
        self.df = df
        self.tcn = text_col_name
        self.tokenizer = tokenizer
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels

    def select_text_cols(self, consecutive):
        if consecutive is True:
            cols = [self.tcn + '_' + item for item in self.consec_panels]
        else:
            cols = [self.tcn + '_' + item for item in self.all_panels]
        new_df = self.df[cols]
        return new_df

    def select_text_cols_into_list(self, consecutive):
        if consecutive is True:
            cols = [self.tcn + '_' + item for item in self.consec_panels]
        else:
            cols = [self.tcn + '_' + item for item in self.all_panels]
        new_df_list = []
        for j in cols:
            new_df = self.df[j] # a list of pandas series
            new_df_list.append(new_df)
        return new_df_list

    def combine_text_cols_into_single_col(self, consecutive=False):  # use take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
        new_df_list = self.select_text_cols_into_list(consecutive=consecutive)
        single_text_col = functools.reduce(lambda a, b: a.fillna('') + b.fillna(''), new_df_list)
        single_text_col = single_text_col.apply(lambda x: re.sub(r'[0-9]', '', x))
        return single_text_col

    def tf_idf_transformation(self, consecutive=False):
        single_text_col = self.combine_text_cols_into_single_col(consecutive=consecutive)
        pipe = Pipeline(steps=[('tfidf',
                                TfidfVectorizer(
                                    stop_words='english',
                                    strip_accents='unicode',
                                    max_features=2000))])
        matrix = pipe.fit_transform(single_text_col)
        print(type(matrix))
        # transform the matrix to matrix dataframe
        matrix_df = pd.DataFrame(matrix.toarray(),
                                 columns=pipe['tfidf'].get_feature_names())
        matrix_df['app_ids'] = single_text_col.index.tolist()
        matrix_df.set_index('app_ids', inplace=True)
        return matrix, matrix_df

    def truncate_svd_threshhold_plot(self, consecutive=False):
        matrix, matrix_df = self.tf_idf_transformation(consecutive=consecutive)
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

    def truncate_svd(self, n_comp, random_state, consecutive=False):
        matrix, matrix_df = self.tf_idf_transformation(consecutive=consecutive)
        svd = TruncatedSVD(n_components=n_comp, random_state=random_state)  # specify random state for the results to be replicable
        matrix_transformed = svd.fit_transform(
            matrix)  # I tried to substitute matrix_df here, and got exactly same output np array as using matrix
        print(matrix_transformed.shape)
        matrix_transformed_df = pd.DataFrame(matrix_transformed)  # do not need to assign column names because those do not correspond to each topic words (they have been transformed)
        matrix_transformed_df['app_ids'] = matrix_df.index.tolist()
        matrix_transformed_df.set_index('app_ids', inplace=True)
        return matrix_transformed, matrix_transformed_df

    def kmeans_cluster(self, n_clusters, init, random_state, n_comp, consecutive=False):
        matrix_transformed, matrix_transformed_df = self.truncate_svd(n_comp=n_comp,
                                                                      random_state=random_state,
                                                                      consecutive=consecutive)
        kmeans = KMeans(n_clusters=n_clusters, init=init, random_state=random_state)
        y_kmeans = kmeans.fit_predict(
            matrix_transformed)  # put matrix_transformed_df here would generate same result as put matrix_transformed
        return y_kmeans, matrix_transformed_df

    def add_predicted_cluster_labels_to_df(self, n_clusters, init, random_state, n_comp, consecutive=False):
        y_kmeans, matrix_transformed_df = self.kmeans_cluster(
            n_clusters=n_clusters,
            init=init,
            random_state=random_state,
            n_comp=n_comp,
            consecutive=consecutive)
        matrix_transformed_df['predicted_labels'] = y_kmeans
        new_df = matrix_transformed_df[['predicted_labels']]
        return new_df