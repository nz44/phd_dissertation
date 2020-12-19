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
### Preprocessing
# ********************************************************************************************************

def take_out_the_text_colume_from_merged_df(initial_panel, text_column_name):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = initial_panel + '_MERGED.pickle'
    q = input_path / '__PANELS__' / folder_name / f_name
    with open(q, 'rb') as f:
        D = pickle.load(f)
    text_cols = D.filter(regex=text_column_name).columns
    F =D[text_cols]
    F.fillna('not available', inplace=True)
    return F

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
def open_token_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_converted_to_spacy_tokens.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    DF = pd.read_pickle(q)
    return DF

def combine_tokenized_cols_into_single_col(initial_panel, panels_have_text):
    df = open_token_df(initial_panel)
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
def open_topic_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_tokens_converted_to_topics.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df

# 2020 Dec 12: I decide not to combine all topics words of each app into a single document.
# Instead, I will keep the n topic words for one app as a unity, and use hierarchical clustering to
# group those small unities, so you will not need to set arbitrary criteria for comparing the topic words within one app
# with the top topic words generated from a single document.

def combine_topics_into_a_dict(initial_panel):
    df = open_topic_df(initial_panel)
    topic_dict = dict.fromkeys(df.index)
    for index, row in df.iterrows():
        topic_dict[index] = row['topic_words']
    return topic_dict


# in order to apply td-idf transformation, you need each document to be a string, containing all the topic words
# rather than a list of topic words
def combine_topics_into_a_list(initial_panel):
    df = open_topic_df(initial_panel)
    topic_list = []
    for index, row in df.iterrows():
        listToStr = ' '.join([str(elem) for elem in row['topic_words']])
        topic_list.append(listToStr)
    return topic_list

def combine_topics_into_pandas(initial_panel):
    df = open_topic_df(initial_panel)
    for index, row in df.iterrows():
        listToStr = ' '.join([str(elem) for elem in row['topic_words']])
        df.at[index, 'topic_string'] = listToStr
    df_out = df[['topic_string']] # single brackets will only return you a pandas series instead of pandas dataframe!
    return df_out