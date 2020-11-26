# Oct 20, 2020
# find the products in a niche market (apps belonging to a very narrowly defined category, which is figured out by
# analyzing descriptions and names and figure out if they serve the same function or purpose)
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
from sklearn.feature_extraction.text import TfidfVectorizer
vectorizer = TfidfVectorizer()
from sklearn import decomposition


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

def tfi_vectorizer_for_an_app_description(tokenized_app_description):
    c = vectorizer.fit_transform(tokenized_app_description)
    terms = vectorizer.get_feature_names()
    return c, terms

def create_model(init, k, tokenized_app_description):
    model = decomposition.NMF(init=init, n_components=k)
    c, terms = tfi_vectorizer_for_an_app_description(tokenized_app_description)
    # apply the model and extract the two factor matrices
    W = model.fit_transform(c)
    H = model.components_
    return W, H

# below function is directly copied from
# https://github.com/derekgreene/topic-model-tutorial/blob/master/2%20-%20NMF%20Topic%20Models.ipynb
# The top ranked terms from the H factor for each topic can give us an insight into the content of that topic.
# This is often called the topic descriptor. Let's define a function that extracts the descriptor for a specified topic:
def get_descriptor( init, k, tokenized_app_description, topic_index, top ):
    C, terms = tfi_vectorizer_for_an_app_description(tokenized_app_description)
    W, H = create_model(init, k, tokenized_app_description)
    # reverse sort the values to sort the indices
    top_indices = np.argsort( H[topic_index,:] )[::-1]
    # now get the terms corresponding to the top-ranked indices
    top_terms = []
    for term_index in top_indices[0:top]:
        top_terms.append( terms[term_index] )
    return top_terms


def get_descriptor_in_tuple( init, k, tokenized_app_description, top ):
    descriptors = []
    for topic_index in range(k):
        d = get_descriptor(init, k, tokenized_app_description, topic_index, top)
        str_descriptor = ", ".join(d)
        topic_tuple = (topic_index+1, str_descriptor)
        descriptors.append(topic_tuple)
    return descriptors


def get_descriptor_for_each_row( initial_panel, panels_have_text, init, k, top, **kwargs ):
    df = open_token_df(initial_panel)
    # the columns to store the ranked topics
    topic_cols = ['description_topics_' + item for item in panels_have_text]
    for i in topic_cols:
        df[i] = np.nan
    if 'sample' in kwargs.keys():
        df = df.sample(n=kwargs['sample'])
    for i in panels_have_text:
        df['description_topics_'+i] = df['description_'+i].progress_apply(lambda x: get_descriptor_in_tuple( init, k, x, top ))
    # save
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_tokens_converted_to_topics.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df.to_pickle(q)
    return df