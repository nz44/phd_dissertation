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