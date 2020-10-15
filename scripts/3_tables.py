import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
pd.set_option('display.max_rows', 500)
pd.set_option('display.max_columns', 500)
import datetime
import os.path
import glob
import ast
import pickle
from pathlib import Path
import tabulate
from functools import reduce

graph_output = '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs'
table_output = '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/tables'
input_path = Path("/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____")
################################################################################################
# Oct 14 2020
# the motivation for visualization data in tables is because I see most published papers in economic feild,
# descriptive stats are organized into latex tables
################################################################################################
def export_and_save_qauntile_tables(initial_dates):

    # open dataframe
    for i in initial_dates:
        folder_name = i + '_PANEL_DF'
        f_name = i + '_MERGED.pickle'
        q = input_path / '__PANELS__' / folder_name / f_name
        with open(q, 'rb') as f:
            DF = pickle.load(f)
        E = DF.quantile([.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
        f_name = i + '_quantile_table.html'
        E.to_html(os.path.join(table_output, f_name))



##################################################################################################
## BIG THING (DEVIDE APPS ACCORDING TO MIN-INSTALLS)
# I suspect different groups of apps may exhibit entirely different characteristics (such as the distribution of score, reviews, text sentiment and so on)
# the most obvious way of grouping is by the range of minInstalls, apps with super large number of installs may be quite different from niche products which only have a few installs
# I think this is more important than looking at categories, the thing with categories is that some of them have blurry lines (social, media, leisure shopping all combined together in one app)
# the other thing is the category is only important when one analyze gaming and non-gaming apps.
# The function below will transform data into different minimum install groups, and then feed the transformed data back to the graphing functions above, to see violin plots for different install groups


# the function below split the dataframe according to quantile cutoff points in any particular panel
def divide_dataframe_by_variable_level(initial_date, the_panel, variable, in_place):
    # open original dataframe
    folder_name = initial_date + '_PANEL_DF'
    f_name = initial_date + '_MERGED.pickle'
    q = input_path / '__PANELS__' / folder_name / f_name
    with open(q, 'rb') as f:
        DF = pickle.load(f)

    # set break points
    if variable == 'minInstalls':
        # from export_and_save_quantile_tables, it looks like the most meaningful cutoff point for analyzing apps are
        # below 500000.0 (0.3 quantile), 5000000.0 - 5000000.0 (0.3 - 0.7 quantile) and above 5000000.0 (0.7 - 1.0 quantile)
        break_points = [500000, 5000000]

    # create a new dictionary containing all the subsetted dataframes according to break points
    if in_place is False:
        DF_dict = {}
        col_name = variable + '_' + the_panel
        for i in range(len(break_points) + 1):
            if i == 0:
                name = 'below ' + str(break_points[i])
                E = DF[DF[col_name] <= break_points[i]]
                DF_dict[name] = E
            if 0 < i < len(break_points):
                name = 'between ' + str(break_points[i - 1]) + ' and ' + str(break_points[i])
                E = DF[(DF[col_name] > break_points[i - 1]) & (DF[col_name] <= break_points[i])]
                DF_dict[name] = E
            if i == len(break_points):
                name = 'above ' + str(break_points[i - 1])
                E = DF[DF[col_name] > break_points[i - 1]]
                DF_dict[name] = E
        # check
        print(initial_date + ' panel divided according to static ' + variable + ' in ' + the_panel)
        for k, v in DF_dict.items():
            print(k, v.shape)
            print()
        # save
        f_name = 'section_df_static_' + variable + '_' + the_panel + '.pickle'
        q = input_path / '__PANELS__' / folder_name / f_name
        pickle.dump(DF_dict, open(q, 'wb'))
        return DF_dict

    elif in_place is True:
        col_name = variable + '_' + the_panel
        conditions = []
        choices = []
        for i in range(len(break_points) + 1):
            if i == 0:
                name = 'below ' + str(break_points[i])
                conditions.insert(i, (DF[col_name] <= break_points[i]))
                choices.insert(i, name)
            if 0 < i < len(break_points):
                name = 'between ' + str(break_points[i - 1]) + ' and ' + str(break_points[i])
                conditions.insert(i, (DF[col_name] > break_points[i-1]) & (DF[col_name] <= break_points[i]))
                choices.insert(i, name)
            if i == len(break_points):
                name = 'above ' + str(break_points[i - 1])
                conditions.insert(i, (DF[col_name] > break_points[i-1]))
                choices.insert(i, name)
        # map the conditions with choices in the new column
        new_col_name = 'group_static_' + variable
        DF[new_col_name] = np.select(conditions, choices, default=None)
        print(initial_date)
        print(DF[new_col_name].value_counts())
        # save in place
        f_name = initial_date + '_MERGED.pickle'
        q = input_path / '__PANELS__' / folder_name / f_name
        pickle.dump(DF, open(q, 'wb'))
        return DF

# the function below split the dataframe according to whether the variable has changed during the time period
# because I believe the apps that increased in the variable may exhibit different characteristics than apps that do not have increase in the variable
def divide_dataframe_by_variable_change(initial_date, end_date, variable, in_place):
    # open original dataframe
    folder_name = initial_date + '_PANEL_DF'
    f_name = initial_date + '_MERGED.pickle'
    q = input_path / '__PANELS__' / folder_name / f_name
    with open(q, 'rb') as f:
        DF = pickle.load(f)

    # set break points
    if variable == 'minInstalls':
        # it looks like majority of apps have no change in minInstalls between initial date and end date
        # so the analysis will simply break dataframe into two parts, apps with positive and zero increase in minInstalls.
        change_points = [0]

    # create the column containing change in the variable and take a look at the quantile distribuiton of the new column
    new_col_name = 'change_in_' + variable
    col_name_1 = variable + '_' + initial_date
    col_name_2 = variable + '_' + end_date
    DF[new_col_name] = DF[col_name_2] - DF[col_name_1]
    E = DF.quantile([.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9])
    print(E[new_col_name])

    # create a new dictionary containing all the subsetted dataframes according to break points
    if in_place is False:
        DF_dict = {}
        for i in range(len(change_points) + 1):
            if i == 0:
                name = 'no change'
                F = DF[DF[new_col_name] == 0]
                DF_dict[name] = F
            if 0 < i < len(change_points):
                name = 'increased between ' + str(change_points[i - 1]) + ' and ' + str(change_points[i])
                F = DF[(DF[new_col_name] > change_points[i - 1]) & (DF[new_col_name] <= change_points[i])]
                DF_dict[name] = F
            if i == len(change_points):
                name = 'increase more than ' + str(change_points[i - 1])
                F = DF[DF[new_col_name] > change_points[i - 1]]
                DF_dict[name] = F
        # check
        print(initial_date + ' panel divided according to change in ' + variable + ' between the initial date and ' + end_date)
        for k, v in DF_dict.items():
            print(k, v.shape)
            print()

        # save
        f_name = 'section_df_change_in_' + variable + '_from_' + initial_date + '_to_' + end_date + '.pickle'
        q = input_path / '__PANELS__' / folder_name / f_name
        pickle.dump(DF_dict, open(q, 'wb'))
        return DF_dict

    elif in_place is True:
        conditions = []
        choices = []
        for i in range(len(change_points) + 1):
            if i == 0:
                name = 'no change'
                conditions.insert(i, (DF[new_col_name] == 0))
                choices.insert(i, name)
            if 0 < i < len(change_points):
                name = 'increased between ' + str(change_points[i - 1]) + ' and ' + str(change_points[i])
                conditions.insert(i, (DF[new_col_name] > change_points[i-1]) & (DF[new_col_name] <= change_points[i]))
                choices.insert(i, name)
            if i == len(change_points):
                name = 'increase more than ' + str(change_points[i - 1])
                conditions.insert(i, (DF[new_col_name] > change_points[i-1]))
                choices.insert(i, name)
        # map the conditions with choices in the new column
        col2_name = 'group_change_' + variable
        DF[col2_name] = np.select(conditions, choices, default=None)
        print(initial_date)
        print(DF[col2_name].value_counts())
        # save in place
        f_name = initial_date + '_MERGED.pickle'
        q = input_path / '__PANELS__' / folder_name / f_name
        pickle.dump(DF, open(q, 'wb'))
        return DF


################################################################################################
# make descriptive stats table according to different minInstalls (level and change)
################################################################################################
def descriptive_stats_merged_df(level_1_var, initial_date, the_panel, level_2_vars, level_3_vars, **kwargs):
    # open file
    folder_name = initial_date + '_PANEL_DF'
    f_name = initial_date + '_MERGED.pickle'
    q = input_path / '__PANELS__' / folder_name / f_name
    with open(q, 'rb') as f:
        DF = pickle.load(f)
    #print(DF.head())

    # create the descriptive stats dataframes
    ## the groups -- and subgroups I want the descriptive stats for
    ## the numerical variables that I want mean, media, quartile for within each sub groups
    ### level 1 group var: minInstalls groups
    level_2_col = level_2_vars + '_' + the_panel
    level_3_col = level_3_vars + '_' + the_panel
    DF_LIST = []
    for var_name in kwargs.values():
        var_col = var_name + '_' + the_panel
        stats = ['count', 'min', 'mean', 'median', 'max']
        grouped_multiple = DF.groupby([level_1_var, level_2_col, level_3_col]).agg({var_col: stats})
        grouped_multiple.columns = list(map(lambda x: var_name + '_' + str(x), stats))
        grouped_multiple = grouped_multiple.reset_index()
        count_var = var_name + '_count'
        grouped_multiple.rename(columns = {count_var: 'count'}, inplace = True)
        DF_LIST.append(grouped_multiple)
    # merge stats of each variable into a single dataframe
    result_df = reduce(lambda x, y: pd.merge(x, y, on=[level_1_var, level_2_col, level_3_col, 'count']), DF_LIST)
    # save
    f_name = initial_date + '_summary_stats_by_group.html'
    result_df.to_html(os.path.join(table_output, f_name))
    return(result_df)
