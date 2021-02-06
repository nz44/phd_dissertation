import pandas as pd
from tqdm import tqdm
import numpy as np
import random
from datetime import datetime as dt
from datetime import date
import functools
import collections
import operator
import functools
import re
import matplotlib
import seaborn

# the input dataframe are the output of merge_panels_into_single_df() method of app_detail_dicts class
class merged_panel_df():

    def __init__(self, merged_df):
        self.merged_df = merged_df

    def print_col_names(self):
        for col in self.merged_df.columns:
            print(col)

    def print_num_rows(self):
        print(len(self.merged_df.index))
