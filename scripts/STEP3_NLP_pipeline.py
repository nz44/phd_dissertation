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
    """
    the input dataframe is by default appid index
    df is the MERGED dataframe
    df_labels is the output of self.kmeans_cluster and saved with name 'predicted_kmeans_labels'
    """
    def __init__(self,
                 df,
                 text_col_name,
                 tokenizer=nlp.Defaults.create_tokenizer(nlp),
                 df_labels=None,
                 df_niche_labels=None,
                 initial_panel=None,
                 all_panels=None,
                 consec_panels=None,
                 stoplist = nltk.corpus.stopwords.words('english'),
                 nlp_input_path = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp')):
        self.df = df
        self.dfl = df_labels
        self.dfn = df_niche_labels
        self.tcn = text_col_name
        self.tokenizer = tokenizer
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.consec_panels = consec_panels
        self.stoplist = stoplist
        self.nlp_path = nlp_input_path

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

    def find_appids_to_remove_before_imputing(self, consecutive=True):
        if consecutive is True:
            cols = [self.tcn + '_' + item for item in self.consec_panels]
        else:
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
                 df=self.df,
                 text_col_name=self.tcn,
                 tokenizer=nlp.Defaults.create_tokenizer(nlp),
                 df_labels=self.dfl,
                 df_niche_labels=self.dfn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 consec_panels=self.consec_panels,
                 stoplist = nltk.corpus.stopwords.words('english'),
                 nlp_input_path = Path('/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp'))

    def impute_text_cols(self, consecutive=True, combine_panels=True):
        """
        run this method on the new instance object returned from self.find_appids_to_remove_before_imputing
        By observation, I assume description is time-invariant, so one would combine_panels by default, however,
        I will also impute the missing panel with text from another other column before or after this missing panel.
        Use combine_panel = False if you do not want to run text cluster for so long time.
        """
        if consecutive is True:
            cols = [self.tcn + '_' + item for item in self.consec_panels]
        else:
            cols = [self.tcn + '_' + item for item in self.all_panels]
        if combine_panels is True:
            for j in range(len(cols)):
                self.df[cols[j]] = self.df[cols[j]].fillna('')
                if j == 0:
                    self.df['combined_panel_'+self.tcn] = self.df[cols[j]]
                else:
                    self.df['combined_panel_' + self.tcn] = self.df['combined_panel_'+self.tcn] + self.df[cols[j]]
            combined_text_col = self.df['combined_panel_' + self.tcn]
            null_data = combined_text_col[combined_text_col.isnull()]
            if len(null_data) == 0:
                print('successfully imputed missing text columns for COMBINED TEXT COLUMN of the dataset', self.initial_panel)
                print('assuming all text columns are time-invariant')
        else:
            text_df = self.df[cols]
            null_data = text_df[text_df.isnull().any(axis=1)]
            null_data_t = null_data.T
            appids_and_none_missing_text = dict.fromkeys(null_data_t.columns)
            for appid in null_data_t.columns:
                not_missing = null_data_t[appid].dropna()
                not_missing_text = not_missing.iloc[0]
                appids_and_none_missing_text[appid] = not_missing_text
            for appid, text in appids_and_none_missing_text.items():
                for j in cols: # since assuming time-invariant, all panels could be assigned to the same text
                    self.df.at[appid, j] = text
            text_df = self.df[cols] # alternation on the original file does not reflect on the shallow copy
            null_data = text_df[text_df.isnull().any(axis=1)]
            if len(null_data.index) == 0:
                print('successfully imputed missing text columns for EACH PANEL of the dataset', self.initial_panel)
                print('assuming all text columns are time-invariant')
        return nlp_pipeline(
            df=self.df,
            text_col_name=self.tcn,
            tokenizer=nlp.Defaults.create_tokenizer(nlp),
            df_labels=self.dfl,
            df_niche_labels=self.dfn,
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            consec_panels=self.consec_panels,
            stoplist=nltk.corpus.stopwords.words('english'),
            nlp_input_path=Path(
                '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/nlp'))

    def prepare_text_col(self, consecutive=True, combine_panels=True):  # use take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
        """
        # _________________ process text __________________________________________________
        # Adding ^ in []  excludes any character in
        # the set. Here, [^ab5] it matches characters that are
        # not a, b, or 5.
        """
        if combine_panels is True: # here combine all panels' text together to create time invariant text cluster labels
            # new_df_list = self.select_text_cols_into_list(consecutive=consecutive)
            # single_text_col = functools.reduce(lambda a, b: a.squeeze().fillna('') + b.squeeze().fillna(''), new_df_list)
            single_text_col = self.df['combined_panel_'+self.tcn]
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
            label_single = svd_matrix[['combined_panels_kmeans_labels']]
            return label_single
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

    # --------------------------------------------------------------------------------------------------
    def find_top_count_labels(self, label_name, top_n, combine_panels=True):
        if combine_panels is True:
            new_df = self.dfl.copy(deep=True)
            new_df[label_name + '_count'] = 0
            gdf = new_df[[label_name, label_name + '_count']].groupby(by=[label_name]).count()
            gdf.sort_values(by=label_name + '_count', ascending=False, inplace=True)
            gdf.reset_index(inplace=True)
            top_labels = gdf.iloc[:top_n, 0].tolist()
            return gdf, top_labels
        else:
            df_list = []
            top_labels_dict = {} # the top 3 labels with most apps
            for i in self.dfl.columns:
                x = i.split("_", -1)
                panel = x[0]
                new_df = self.dfl.copy(deep=True)
                new_df[panel + '_' + label_name + '_count'] = 0
                gdf = new_df[[i, panel + '_' + label_name + '_count']].groupby(by=[i]).count()
                gdf.sort_values(by=panel + '_' + label_name + '_count', ascending=False, inplace=True)
                gdf.reset_index(inplace=True)
                df_list.append(gdf)
                top_labels = gdf.iloc[:top_n, 0].tolist()
                top_labels_dict[panel] = top_labels
            rdf = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
            return rdf, top_labels_dict

    def label_broad_niche_apps(self, label_name, top_n, combine_panels=True):
        rdf, top_labels_dict = self.find_top_count_labels(label_name=label_name, top_n=top_n, combine_panels=combine_panels)
        new_df = self.dfl.copy(deep=True)
        if combine_panels is True:
            pdf = new_df[[label_name]]
            pdf['niche_app'] = pdf[label_name].apply(lambda x: 0 if x in top_labels_dict else 1)
            # add rdf count per label in each panel to this new dataframe
            df_l_c = rdf[[label_name, label_name + '_count']]
            ndf = pdf.reset_index().merge(df_l_c, on=label_name, how='left').set_index('app_ids')
            count_df = ndf.groupby(by='niche_app').count()
            count_df = count_df.iloc[:, 0]
            count_df.rename('count', inplace=True)
            print(count_df)
            return ndf
        else:
            niche_app_dfs = []
            for panel, top_labels in top_labels_dict.items():
                pdf = new_df[[panel+'_'+label_name]]
                pdf[panel+'_niche_app'] = pdf[panel+'_'+label_name].apply(lambda x: 0 if x in top_labels else 1)
                # add rdf count per label in each panel to this new dataframe
                df_l_c = rdf[[panel+'_'+label_name, panel+'_'+label_name+'_count']]
                ndf = pdf.reset_index().merge(df_l_c, on=panel+'_'+label_name, how='left').set_index('app_ids')
                niche_app_dfs.append(ndf)
            result_df = functools.reduce(lambda a, b: a.join(b, how='inner'), niche_app_dfs)
            return result_df

    def check_niche_label_is_time_variant(self, label_name, top_n, combine_panels=False, consecutive=True):
        """
        This method is only for if you have previously created text cluster labels seperately by panels.
        03/09/21
        I have discovered that appids that changed from niche to broad or vice versa actually have the exactly same descriptions.
        So from now on I will just predict text cluster labels with combined text columns.
        """
        result_df = self.label_broad_niche_apps(label_name=label_name, top_n=top_n, combine_panels=combine_panels)
        if consecutive is True:
            cols = [i + '_niche_app' for i in self.consec_panels]
        else:
            cols = [i + '_niche_app' for i in self.all_panels]
        df2 = result_df[cols]
        time_invariant_niche = len(df2.columns)
        time_invariant_broad = 0
        df2['sum_niche_labels'] = df2[cols].sum(axis=1)
        df2['time_invariant_broad_apps'] = df2['sum_niche_labels'].apply(lambda x: 1 if x == time_invariant_broad else 0)
        df2['time_invariant_niche_apps'] = df2['sum_niche_labels'].apply(
            lambda x: 1 if x == time_invariant_niche else 0)
        df2['app_changed_niche_label'] = df2['sum_niche_labels'].apply(
            lambda x: 1 if x != time_invariant_niche and x != time_invariant_broad else 0)
        count_df = df2[['time_invariant_broad_apps', 'time_invariant_niche_apps', 'app_changed_niche_label']].sum(axis=0)
        print(count_df)
        df3 = df2.loc[df2['app_changed_niche_label']==1]
        appids_changed_label = df3.index.tolist()
        label_count_df = result_df.loc[appids_changed_label]
        return df3, label_count_df

    def loop_through_cutoffs_for_niche_broad(self, label_name, top_n_list, combine_panels=False, consecutive=True):
        """
        The only reason to loop through different niche/broad threshhold is to check whether the number of apps belonging to
        niche group or broad group varies across panels.
        I thought I could find the optimal threshhold (n_top) that would lead to the least number of apps that change niche/broad label
        over time. The results showed that as n_top increases, the number of apps changed niche/broad labels monotonically increases as well.
        So this method is only used if you had previous predicted panel specific text labels.
        """
        changed_appids_dfs = []
        changed_appids_count_dfs = []
        for top_n in top_n_list:
            print('top', top_n, 'labels defined as broad apps')
            df3, label_count_df = self.check_niche_label_is_time_variant(label_name=label_name, top_n=top_n, combine_panels=combine_panels, consecutive=consecutive)
            print()
            changed_appids_dfs.append(df3)
            changed_appids_count_dfs.append(label_count_df)
        return changed_appids_dfs, changed_appids_count_dfs

    def eyeball_app_with_label(self, label_name, top_n, niche=True, combine_panels=True, the_panel=None):
        df2 = self.label_broad_niche_apps(label_name=label_name, top_n=top_n, combine_panels=combine_panels)
        if combine_panels is True:
            if niche is True:
                df3 = df2.loc[df2['niche_app'] == 1]
            else:
                df3 = df2.loc[df2['niche_app'] == 0]
            df3 = df3.sort_values(by=label_name + '_count', ascending=False)
            labels = df3[label_name].unique()
            labels_and_corresponding_appids = dict.fromkeys(labels)
            for label in labels_and_corresponding_appids.keys():
                df4 = df3.loc[df3[label_name] == label]
                labels_and_corresponding_appids[label] = df4.index.tolist()
        else:
            if niche is True:
                df3 = df2.loc[df2[the_panel+'_niche_app'] == 1]
            else:
                df3 = df2.loc[df2[the_panel+'_niche_app'] == 0]
            df3 = df3.sort_values(by=the_panel+'_'+label_name+'_count', ascending=False)
            labels = df3[the_panel+'_'+label_name].unique()
            labels_and_corresponding_appids = dict.fromkeys(labels)
            for label in labels_and_corresponding_appids.keys():
                df4 = df3.loc[df3[the_panel+'_'+label_name] == label]
                labels_and_corresponding_appids[label] = df4.index.tolist()
        labels_and_corresponding_text = dict.fromkeys(labels)
        # --------------------- save eyeball results to text file ---------------------------
        if niche is True:
            filename = self.initial_panel + '_eyeball_app_with_niche_label.txt'
        else:
            filename = self.initial_panel + '_eyeball_app_with_broad_label.txt'
        my_file = open(self.nlp_path / filename, "w")
        for label, appids in labels_and_corresponding_appids.items():
            df5 = self.df.loc[appids]
            if combine_panels is True:
                df6 = df5[['combined_panel_' + self.tcn]]
            else:
                df6 = df5[[self.tcn+'_' + the_panel]]
            labels_and_corresponding_text[label] = df6
            if niche is True:
                my_file.write('niche label : ' + str(label) + '\n')
            else:
                my_file.write('broad label : ' + str(label) + '\n')
            my_file.write('count in the label :' + str(len(df6.index)) + '\n')
            if len(df6.index) >= 3:
                df6 = df6.sample(3)
            for index, row in df6.iterrows():
                my_file.write(index + '\n')
                my_file.write('='*150+'\n')
                if combine_panels is True:
                    my_file.write(row['combined_panel_' + self.tcn] + '\n\n\n')
                else:
                    my_file.write(row[self.tcn + '_' + the_panel] + '\n\n\n')
            my_file.write('\n\n\n\n\n')
        my_file.close()
        return labels_and_corresponding_text

    def eyeball_app_changed_niche_label(self, label_name, top_n, combine_panels=False, consecutive=True):
        """
        The method is only used if you have previous calculated panel specific text labels and would like to look closer
        at those appid text columns because their niche/broad label changes over time.
        03/09/21
        I found that those appids are with exactly the same text information in each panel (after deleting None and impute missing).
        So I conclude that the apps that changed from niche to broad type is completely due to the algorithm only.
        It is pointless to conduct panel specific text label prediction.
        From this particular method, I prove that apps' text labels are time INVARIANT.
        Now I should find a suitable panel regression model that incorporate time-invariant independent variables.
        """
        df1, df2 = self.check_niche_label_is_time_variant(
            label_name=label_name,
            top_n=top_n,
            combine_panels=combine_panels,
            consecutive=consecutive)
        df3 = self.select_text_cols(consecutive=consecutive)
        df4 = df3.loc[df1.index.tolist()]
        # --------------------- save eyeball results to text file ---------------------------
        filename = self.initial_panel + '_eyeball_app_changed_niche_label.txt'
        my_file = open(self.nlp_path / filename, "w")
        my_file.write('number of apps changed niche label over time : ' + str(len(df4.index)) + '\n')
        df4 = df4.sample(5)
        for index, row in df4.iterrows():
            my_file.write(index+'\n')
            my_file.write('=' * 150 + '\n')
            for i in df4.columns:
                my_file.write(i+ '\n')
                my_file.write('-'*100+'\n')
                my_file.write(row[i])
                my_file.write('\n\n')
            my_file.write('\n\n\n\n\n')
        my_file.close()
        return df2

# ====================================================================================================================
"""
RESEARCH NOTES:


2021/03/08
After eyeballing self.eyeball_app_changed_niche_label(), I see that they do not ever change over time, the description. 
So I decide from now on to combine all panel's description together. 


03/09/21
I found that those appids are with exactly the same text information in each panel (after deleting None and impute missing).
So I conclude that the apps that changed from niche to broad type is completely due to the algorithm only.
It is pointless to conduct panel specific text label prediction.
From this particular method, I prove that apps' text labels are time INVARIANT.
Now I should find a suitable panel regression model that incorporate time-invariant independent variables. 
"""
# =====================================================================================================================