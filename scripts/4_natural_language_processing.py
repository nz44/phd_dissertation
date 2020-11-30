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
from collections import Counter


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


# embed try except inside the function you are going to apply to pandas
# every time you run get_descriptor function, it will start from a new random initial position, which will
# result in slightly different topic words.
# so in order to keep consistency, you cannot run get_descriptor_in_tuple_or_list twice, once get tuple and once
# get list, because this way you are actually running get_descriptor twice and get different topic words in tuple and list.
def get_descriptor_in_tuple( init, k, tokenized_app_description, top):
    try:
        descriptors = []
        for topic_index in range(k):
            d = get_descriptor(init, k, tokenized_app_description, topic_index, top)
            str_descriptor = ", ".join(d)
            topic_tuple = (topic_index+1, str_descriptor)
            descriptors.append(topic_tuple)
        return descriptors
    except:
        str_tokenized = ', '.join(tokenized_app_description)
        print('error : ' + str_tokenized)


def take_words_from_tuple(list_of_tuples):
    a = []
    if list_of_tuples is not None:
        for tup in list_of_tuples:
            # corresponding to [(1, 'marker, photo'), (2, 'design, instagram')]
            if isinstance(tup[1], str):
                x = tup[1].split(', ')
                a.extend(x) # use extend bc x is a list of strings
            # corresponding to [('game', 14420), ('app', 13258)]
            elif isinstance(tup[0], str):
                a.append(tup[0]) # use append bc tup[0] is just a string
        # need to filter out duplicated topic words within each app's description
        # this is good for later comparing them with the most frequently occurring topic words across all apps.
        return list(set(a))
    else:
        return 'none' # to avoid 'float' type is not iterable in below row[i]


def get_descriptor_for_each_row( initial_panel, panels_have_text, init, k, top, **kwargs ):
    df = open_token_df(initial_panel)
    # the columns to store the ranked topics
    topic_tuple_cols = ['topic_tuple_' + item for item in panels_have_text]
    topic_list_cols = ['topic_list_' + item for item in panels_have_text]
    for i in topic_tuple_cols:
        df[i] = np.nan
    for i in topic_list_cols:
        df[i] = np.nan
    if 'sample' in kwargs.keys():
        df = df.sample(n=kwargs['sample'])
    for i in panels_have_text:
        df['topic_tuple_'+i] = df['description_'+i].progress_apply(lambda x: get_descriptor_in_tuple( init, k, x, top ))
        df['topic_list_' + i] = df['topic_tuple_' + i].progress_apply(lambda x: take_words_from_tuple(x))

    # save
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_tokens_converted_to_topics.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df.to_pickle(q)
    return df


# ********************************************************************************************************
# Broad vs. Niche from Analyzing Topic Models
# https://towardsdatascience.com/lovecraft-with-natural-language-processing-part-2-tokenisation-and-word-counts-f970f6ff5690
# file:///home/naixin/Downloads/4693-Article%20Text-7732-1-10-20190707.pdf
# ********************************************************************************************************
def open_topic_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'description_tokens_converted_to_topics.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df

# https://towardsdatascience.com/lovecraft-with-natural-language-processing-part-2-tokenisation-and-word-counts-f970f6ff5690
# bag-of-words: an unordered aggregated representation of a larger volume of text, which can be a document, chapter, paragraph, sentence, etc….
# The grammar, punctuation, and word order from the original text are ignored, the only thing that’s kept is the
# unique words and a number attached to them. That number can be the frequency with which the word occurs in the text, or it can be a binary 0 or 1,
# simply measuring whether the word is in the text or not.

## put all the topic words from relevent columns into a list (dict value) corresponding to panel names (dict key).
def bag_of_words(initial_panel, panels_have_text):
    df = open_topic_df(initial_panel)
    topic_list_cols = ['topic_list_' + item for item in panels_have_text]
    bow = dict.fromkeys(topic_list_cols, [])
    for i in bow.keys():
        for index, row in df.iterrows():
            bow[i].extend(row[i])
    return bow

## count the frequency of a word appearing in a list (dict value) corresponding to panel names (dict key)
def frequencies_bag_of_words(initial_panel, panels_have_text):
    bow = bag_of_words(initial_panel, panels_have_text)
    fre_bow = dict.fromkeys(bow.keys())
    for k, v in bow.items():
        fre_bow[k] = Counter(v)
    return fre_bow

# create a function that arbitrarily define a list of broad topic words for each panel
def broad_or_niche_topic_words(initial_panel, panels_have_text, type, tup, threshold):
    fre_bow = frequencies_bag_of_words(initial_panel, panels_have_text)
    if type == 'broad':
        if tup is True:
            l = dict.fromkeys(fre_bow.keys())
            for k, v in fre_bow.items():
                l[k] = []
                for word, frequency in v.items():
                    if frequency >= threshold:
                        l[k].append(tuple((word, frequency)))
                l[k] = sorted(l[k], key=lambda x: x[1], reverse=True)
            return l
        else:
            l = dict.fromkeys(fre_bow.keys())
            for k, v in fre_bow.items():
                l[k] = []
                for word, frequency in v.items():
                    if frequency >= threshold:
                        l[k].append(word)
            return l
    if type == 'niche':
        if tup is True:
            l = dict.fromkeys(fre_bow.keys())
            for k, v in fre_bow.items():
                l[k] = []
                for word, frequency in v.items():
                    if frequency < threshold:
                        l[k].append(tuple((word, frequency)))
                l[k] = sorted(l[k], key=lambda x: x[1], reverse=True)
            return l
        else:
            l = dict.fromkeys(fre_bow.keys())
            for k, v in fre_bow.items():
                l[k] = []
                for word, frequency in v.items():
                    if frequency < threshold:
                        l[k].append(word)
            return l


## compare each app's topic words with a list of top topic words (above an arbitrary threshold, for example, >= 1000)
## each app has 6 topic words for now (they are not unique), first turn them
## classifying niche and broad, you do not need to do it for every year because we assume this is time-invariant feature
def df_classify_app_into_broad_or_niche(initial_panel, the_panel, threshold_broad, threshold_niche, **kwargs):
    broad_words = broad_or_niche_topic_words(initial_panel, the_panel, type = 'broad', tup = False, threshold = threshold_broad)
    niche_words = broad_or_niche_topic_words(initial_panel, the_panel, type = 'niche', tup = False, threshold = threshold_niche)
    df = open_topic_df(initial_panel)
    if 'sample' in kwargs.keys():
        df = df.sample(n=kwargs['sample'])
    broad_cols = ['broad_app_' + item for item in the_panel]
    for i in broad_cols:
        df[i] = np.nan
    niche_cols = ['niche_app_' + item for item in the_panel]
    for i in niche_cols:
        df[i] = np.nan

    for i in the_panel:
        with tqdm(total=df.shape[0]) as pbar:
            for index, row in df.iterrows():
                pbar.update(1)
                if any(item in row['topic_list_'+i] for item in broad_words['topic_list_'+i]):
                    df.at[index, 'broad_app_'+i] = 1
                if any(item in row['topic_list_'+i] for item in niche_words['topic_list_'+i]):
                    df.at[index, 'niche_app_'+i] = 1
    for i in the_panel:
        df.fillna({'broad_app_'+i: 0, 'niche_app_'+i: 0}, inplace=True)
    # save
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'niche_broad_classified.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df.to_pickle(q)
    return df

# ************************************************************************************************************
def open_niche_classified_df(initial_panel):
    folder_name = initial_panel + '_PANEL_DF'
    f_name = 'niche_broad_classified.pkl'
    q = input_path / '__PANELS__' / folder_name / f_name
    df = pd.read_pickle(q)
    return df