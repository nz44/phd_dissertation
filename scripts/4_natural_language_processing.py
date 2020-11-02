# Oct 20, 2020
# find the products in a niche market (apps belonging to a very narrowly defined category, which is figured out by
# analyzing descriptions and names and figure out if they serve the same function or purpose)
import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
#import gensim
#import gensim.corpora as corpora


import spacy
from spacy import displacy
nlp = spacy.load('en_core_web_md')
from spacy.lang.en.stop_words import STOP_WORDS
stopwords = list(STOP_WORDS)
print(stopwords)
#spacy_stopwords = spacy.lang.en.stop_words.STOP_WORDS


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

    # below takes too long
    for i in F.columns:
        new_i = 'nlp' + i
        F[new_i] = F[i].apply(lambda x: nlp(x))



#### 1. Tokenization: break text into sentences and then into words, remove punctuations and stopwords



#### 2. Lemmatization: for example, change third person verb to first person verb, change past tense to present tense


#### 3. Stem: words are reduced to their root form




# ********************************************************************************************************
### Topic Modeling
# ********************************************************************************************************