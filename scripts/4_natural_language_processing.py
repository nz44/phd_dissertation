# Oct 20, 2020
# find the products in a niche market (apps belonging to a very narrowly defined category, which is figured out by
# analyzing descriptions and names and figure out if they serve the same function or purpose)
import warnings
warnings.filterwarnings('ignore')
import pickle
import re
from tqdm import tqdm, tqdm_notebook
tqdm_notebook().pandas()
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

import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_sm')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
# print(stopwords)
#spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS

# create tokenizer
tokenizer = nlp.Defaults.create_tokenizer(nlp)

# ********************************************************************************************************
### open_file_func functional parameter
def open_merged_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = initial_panel + '_MERGED.pickle'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df

def open_token_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_converted_to_spacy_tokens.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df

def open_topic_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_tokens_converted_to_topics.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df

def open_cluster_df(initial_panel, cluster_type):
    folder_name = initial_panel + '_PANEL_DF'
    if cluster_type == 'k-means':
        f_name = 'df_with_k_means_labels.pkl'
    elif cluster_type == 'fuzzy-c-means':
        f_name = 'df_with_fuzzy_c_means_labels.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df
# ********************************************************************************************************


# ********************************************************************************************************
### Preprocessing
# ********************************************************************************************************

def take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name): # open_merged_df
    df = open_file_func(initial_panel)
    text_cols = df.filter(regex=text_column_name).columns
    f =df[text_cols]
    f.fillna('not available', inplace=True)
    return f

#### 1. Tokenization: break text into sentences and then into words, remove punctuations and stopword
#### 2. Lemmatization: for example, change third person verb to first person verb, change past tense to present tense
#### 3. Stem: words are reduced to their root form
#### 4. remove hyperlinks
# https://stackabuse.com/using-regex-for-text-manipulation-in-python/
# https://docs.python.org/3/howto/regex.html

# X input is a pure text string (unicode string is alright because python 3 does not make a distinction)
def break_text_string_into_pure_tokens(X):
    X = X.replace('b\'', '')
    X = X.replace('b\"', '')
    X = X.lower()
    p = re.compile(r"\\r\\n")
    X = p.sub(' ', X)
    p = re.compile(r'\<b\>')
    X = p.sub('', X)
    p = re.compile(r'\<\/b\>')
    X = p.sub('', X)
    p = re.compile(r'[$|@|&|-|+|!|=|*|(|)|#|>]')
    X = p.sub(' ', X)

    text = []
    hyperlinks = ['http', '.com']
    Y = nlp(X)
    for w in Y:
        if not w.is_stop and not w.is_punct and not w.is_space:
            if not any(link_word in w.text for link_word in hyperlinks):
                text.append(w.lemma_)
    return text


# convert a dataframe to nlp tokens
# XD is a dataframe output of take_out_the_text_colume_from_merged_df
def convert_df_to_nlp_tokens(initial_panel, text_column_name, **kwargs):
    xd = take_out_the_text_colume_from_merged_df(initial_panel, text_column_name)
    if 'sample' in kwargs.keys():
        xd = xd.sample(n=kwargs['sample'])
    for i in xd.columns:
        xd[i] = xd[i].progress_apply(lambda x: break_text_string_into_pure_tokens(x))
        # below is a way to include tqmd
        # a = np.arange(0, XD.shape[0], 50).tolist()
        # a.insert(len(a), XD.shape[0])
        # list_of_dfs = []
        # for n in range(len(a)-1):
        #     list_of_dfs.append(XD.iloc[a[n]:a[n+1]])
        # for i in tqdm(range(len(list_of_dfs)), desc='converting the next dataframe'):
        #     for j in list_of_dfs[i].columns:
        #         list_of_dfs[i][j] = list_of_dfs[i][j].apply(lambda x: break_text_string_into_pure_tokens(x))
        # XE = pd.concat(list_of_dfs)
    # save
    folder_name = initial_panel + '_PANEL_DF'
    f_name = text_column_name + '_converted_to_spacy_tokens.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    xd.to_pickle(q)
    return xd


# ********************************************************************************************************
# Topic Modeling
# SKlean negative matrix model
# please refer to topic_modeling_playground.ipynb
# ********************************************************************************************************

def combine_tokenized_cols_into_single_col(open_file_func, initial_panel, panels_have_text): # open_token_df
    df = open_file_func(initial_panel)
    tokenized_cols = ['description_' + item for item in panels_have_text]
    for i in range(len(tokenized_cols)):
        if i == 0:
            df['combined_panels_description'] = df[tokenized_cols[i]]
        else:
            df['combined_panels_description'] = df['combined_panels_description'] + df[tokenized_cols[i]]
    return df

# https://medium.com/ml2vec/topic-modeling-is-an-unsupervised-learning-approach-to-clustering-documents-to-discover-topics-fdfbf30e27df
def get_nmf_topics(pipe, doc, num_topics, num_top_words, format):
    try:
        pipe.fit_transform(doc)
        feat_names = pipe['tfidf'].get_feature_names()
        if format == 'dataframe':
            word_dict = {}
            for i in range(num_topics):
                words_ids = pipe['nmf'].components_[i].argsort()[:-num_top_words - 1:-1]
                words = [feat_names[key] for key in words_ids]
                word_dict['Topic # ' + '{:02d}'.format(i+1)] = words
            return pd.DataFrame(word_dict)
        if format == 'list':
            topic_words = []
            for i in range(num_topics):
                words_ids = pipe['nmf'].components_[i].argsort()[:-num_top_words - 1:-1]
                words = [feat_names[key] for key in words_ids]
                topic_words.extend(words)
            return topic_words
    except:
        print(doc)
        return 'none'


def get_nmf_topics_for_each_row( initial_panel, panels_have_text, pipe, num_topics, num_top_words, format, **kwargs ):
    df = combine_tokenized_cols_into_single_col(initial_panel, panels_have_text)
    if 'sample' in kwargs.keys():
        df = df.sample(n=kwargs['sample'])
    # the columns to store the ranked topics
    df['topic_words'] = np.nan
    df['topic_words'] = df['combined_panels_description'].progress_apply(lambda x: get_nmf_topics(pipe, x, num_topics, num_top_words, format))
    # save
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_tokens_converted_to_topics.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df.to_pickle(q)
    return df


# ********************************************************************************************************
# Broad vs. Niche from Analyzing Topic Models (text k-means clustering)
# https://medium.com/@lucasdesa/text-clustering-with-k-means-a039d84a941b
# from above article, I got inspired by 'hierarchical clustering' mentioned at the end.
# https://scikit-learn.org/stable/modules/clustering.html
# https://medium.com/@sametgirgin/hierarchical-clustering-model-in-5-steps-with-python-6c45087d4318
# ********************************************************************************************************

################ get the documents (single string) for each app for tf-idf and clustering ################
# ********************************************************************************************************
def combine_descriptions_into_str(open_file_func, initial_panel, text_column_name, panels_with_text): # use take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
    df = take_out_the_text_colume_from_merged_df(open_file_func, initial_panel, text_column_name)
    description_cols = ['description_' + item for item in panels_with_text]
    for i in range(len(description_cols)):
        if i == 0:
            df['description_all_panels'] = df[description_cols[i]]
        else:
            df['description_all_panels'] = df['description_all_panels'] + df[description_cols[i]]
    df = df[['description_all_panels']]
    return df

def combine_topics_into_a_dict(open_file_func, initial_panel):
    df = open_file_func(initial_panel)
    topic_dict = dict.fromkeys(df.index)
    for index, row in df.iterrows():
        topic_dict[index] = row['topic_words']
    return topic_dict

# in order to apply td-idf transformation, you need each document to be a string, containing all the topic words
# rather than a list of topic words
def combine_topics_into_a_list(open_file_func, initial_panel):
    df = open_file_func(initial_panel)
    topic_list = []
    for index, row in df.iterrows():
        listToStr = ' '.join([str(elem) for elem in row['topic_words']])
        topic_list.append(listToStr)
    return topic_list

def combine_topics_into_pandas(open_file_func, initial_panel):
    df = open_file_func(initial_panel)
    for index, row in df.iterrows():
        listToStr = ' '.join([str(elem) for elem in row['topic_words']])
        df.at[index, 'topic_string'] = listToStr
    df_out = df[['topic_string']] # single brackets will only return you a pandas series instead of pandas dataframe!
    return df_out
# ********************************************************************************************************


# ********************************************************************************************************
# one of the list is the index of orginal dataframe (storing app ids), the other list is the cluster labels, they are of same length
def merge_lists_to_list_of_tuples(list1, list2):
    merged_list = [(list1[i], list2[i]) for i in range(0, len(list1))]
    return merged_list

def merge_lists_to_dataframe(appid, cluster_label, type_of_cluster):
    df = pd.DataFrame({'app_id': appid, type_of_cluster: cluster_label})
    df.set_index('app_id', inplace=True)
    return df

# after obtainning each app id's cluster lable, create a new column to the original pandas dataframe and save
# merged_df is obtained through merge_lists_to_dataframe
def add_cluster_label_to_df(open_file_func, initial_panel, merged_df, cluster_type):
    df = open_file_func(initial_panel)
    df2 = df.merge(merged_df, left_index=True, right_index=True)
    # save
    folder_name = initial_panel + '_PANEL_DF'
    if cluster_type == 'k-means':
        f_name = 'df_with_k_means_labels.pkl'
    elif cluster_type == 'fuzzy-c-means':
        f_name = 'df_with_fuzzy_c_means_labels.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df2.to_pickle(q)
    return df2

def select_cluster_labels(open_file_func, initial_panel, cluster_type, cluster_label_value):
    df = open_file_func(initial_panel, cluster_type)
    label = df[cluster_type] == cluster_label_value
    df2 = df[label][['combined_panels_description', cluster_type]]
    return df2

# given an app's id, see which other apps are in the same cluster (k-means)
def see_apps_from_the_same_cluster_of_a_given_app(open_file_func, initial_panel, cluster_type, given_app_id): # use open_topic_and_cluster_df
    df = open_file_func(initial_panel, cluster_type)
    app_cluster_number = df.loc[given_app_id, cluster_type]
    print(given_app_id, 'has cluster label', str(app_cluster_number))
    label = df[cluster_type] == app_cluster_number
    df2 = df[label][['combined_panels_description', cluster_type]]
    return df2

# look at how many apps are allocated inside each cluster (k-means)
# def how_many_apps_are_inside_each_cluster(open_file_func, initial_panel): # use open_topic_and_cluster_df
