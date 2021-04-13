from pathlib import Path
import pickle
import pandas as pd
pd.set_option('display.max_colwidth', -1)
pd.options.display.max_rows = 999
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import math
import random
from datetime import datetime
from datetime import date
import collections
import operator
import functools
import itertools
import re
import os
import geopy
from geopy.geocoders import AzureMaps
from tqdm import tqdm
tqdm.pandas()
import matplotlib
import seaborn
import copy

"""
In this class, i will create time-invariant dummies indicating sub-smples, in the regression_analysis class, 
I will run both regression for sub samples and regression for the full sample with sub-sample dummies (and interactions).
The niche text label will be generated within each subsample. 
"""
class divide():
    panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__')

    def __init__(self,
                 initial_panel,
                 all_panels,
                 df=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df

    def open_imputed_and_deleted_missing_df(self):
        f_name = self.initial_panel + '_imputed_and_deleted_missing.pickle'
        q = divide.panel_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)
        return divide(initial_panel=self.initial_panel,
                      all_panels=self.all_panels,
                      df=self.df)

    def count_apps_in_each_intended_subsample(self, cat_var):
        """
        The aim of this function is to see which group has too few apps, and that may not be a suitable choice of a sub-sample
        (degree of freedom issue when running regression)
        """
        df2 = self.df.copy(deep=True)
        c = df2.groupby([cat_var]).size().sort_values(ascending=False)
        return c

    def divide_into_categorical_subsamples(self):
        pass

    def divide_into_mininstalls_subsamples(self):
        pass

    def divide_into_developers_subsamples(self):
        pass