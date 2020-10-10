# The file include functions that transform raw scraped app data into
# 1. categorical level and developer level
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
import pandas as pd
import datetime
import os.path
import ast
import pickle

graph_output = '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs'
input_path = Path("/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____")

#*****************************************************************************************************
#*** INPUT DATA (scraped before 202009) is transformed to dataframe
#*****************************************************************************************************
# change the key names to the same one scraped using google_play_scraper
# new data is the output of convert_data_before_202009_to_dict_with_appid_keys,
# which is scraper using the older package play_scraper, thus they have different key attribute
# so I have to standardize that.
######################### old attributes --> new attributes ############################
# title --> # title
# icon --> # icon
# screenshots --> # screenshots
# video --> # video
# category --> # genreId
# score --> # score
# histogram --> # histogram
# reviews --> # reviews
# description --> # description
# description_html --> # descriptionHTML
# recent_changes --> # recentChanges
# editors_choice --> do not exist in new scraper, suggest to delete it
# price --> # price
# free --> # free
# iap --> # offersIAP
# developer_id --> # developerId
# updated --> # updated
# size --> # size
# installs --> # installs and create minInstall variable
# current_version --> # version
# required_android_version --> # androidVersion
# content_rating --> # ratings
# iap_range --> # inAppProductPrice
# interactive_elements --> do not exist in new scraper, suggest to delete it
# developer --> # developer
# developer_email --> # developerEmail
# developer_url --> # developerWebsite
# developer_address --> # developerAddress
# app_id --> # appId
# url --> # url

def transform_old_scraper_dict_dataframe(raw_data, date_panel_scraped):
    # C is the output of convert_data_before_202009_to_dict_with_appid_keys
    D = pd.DataFrame.from_dict(raw_data)
    E = D.T
    E.rename(columns={"category": "genreId", "description_html": "descriptionHTML",
                      "recent_changes": "recentChanges", "iap": "offersIAP",
                      "developer_id": "developerId", "current_version": "version",
                      "required_android_version": "androidVersion",
                      "content_rating": "contentRating", "iap_range": "inAppProductPrice",
                      "developer_email": "developerEmail", "developer_url": "developerWebsite",
                      "developer_address": "developerAddress", "app_id": "appId"}, inplace=True)

    E.drop(['editors_choice', 'interactive_elements',
            'descriptionHTML', 'recentChanges'], axis=1, inplace=True)

    E = pd.get_dummies(E, columns=['free', 'offersIAP'], dtype=int)

    # create new minInstalls column
    E["minInstalls"] = E['installs'].str.replace('[^\w\s]', '')
    E['minInstalls'] = E['minInstalls'].astype(float)

    # change data type in columns, the old scraper did not include summary and comments columns
    E['title'] = E['title'].str.encode("ascii", "ignore").astype("string")
    E['description'] = E['description'].str.encode("ascii", "ignore").astype("string")
    #E['summary'] = E['summary'].str.encode("ascii", "ignore").astype("string")
    E['developer'] = E['developer'].str.encode("ascii", "ignore").astype("string")
    E['developerId'] = E['developerId'].str.encode("ascii", "ignore").astype("string")
    E['developerEmail'] = E['developerEmail'].str.encode("ascii", "ignore").astype("string")
    E['developerWebsite'] = E['developerWebsite'].str.encode("ascii", "ignore").astype("string")
    E['developerAddress'] = E['developerAddress'].str.encode("ascii", "ignore").astype("string")
    #E['comments'] = E['comments'].str.encode("ascii", "ignore").astype("string")

    # remove the list brackets in certain columns
    #E['genreId'] = E['genreId'].apply(pd.Series)

    E['GAME'] = np.where(E['genreId'].str.contains("GAME"), 1, 0)
    utils = ['MEDICAL', 'EDUCATION', 'HEALTH_AND_FITNESS',
             'PARENTING', 'LIBRARIES_AND_DEMO',
             'HOUSE_AND_HOME', 'FINANCE', 'WEATHER', 'BOOKS_AND_REFERENCE',
             'TOOLS', 'BUSINESS', 'PRODUCTIVITY', 'TRAVEL_AND_LOCAL',
             'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES']
    E['UTILITIES'] = np.where(E['genreId'].str.contains('|'.join(utils)), 1, 0)

    social_media_l = ['COMICS', 'BEAUTY', 'ART_AND_DESIGN', 'DATING',
                    'VIDEO_PLAYERS', 'SPORTS', 'LIFESTYLE',
                    'PERSONALIZATION', 'SHOPPING', 'FOOD_AND_DRINK',
                    'MUSIC_AND_AUDIO',
                    'NEWS_AND_MAGAZINES', 'PHOTOGRAPHY', 'COMMUNICATION',
                    'ENTERTAINMENT', 'SOCIAL', 'EVENTS']
    E['SOCIAL_MEDIA_LEISURE'] = np.where(E['genreId'].str.contains('|'.join(social_media_l)), 1, 0)

    # remove duplicate groups categories
    #E.loc[(E.UTILITIES == 1 & ), 'Event'] = 'Hip-Hop'
    E['UTILITIES'] = np.where(((E.GAME == 1) & (E.UTILITIES == 1)), 0, E.UTILITIES)
    E['SOCIAL_MEDIA_LEISURE'] = np.where(((E.GAME == 1) & (E.SOCIAL_MEDIA_LEISURE == 1)), 0, E.SOCIAL_MEDIA_LEISURE)
    # because the conflicts mostly occur where family education is in the same category as entertainment or art and design
    E['SOCIAL_MEDIA_LEISURE'] = np.where(((E.UTILITIES == 1) & (E.SOCIAL_MEDIA_LEISURE == 1)), 0, E.SOCIAL_MEDIA_LEISURE)

    E['check_all_genre_covered'] = E['GAME'] + E['UTILITIES'] + E['SOCIAL_MEDIA_LEISURE']

    # categorical dummies for content rating
    E['contentRating_Adults only 18+'] = np.where(E['contentRating'].str.contains("18+"), 1, 0)
    E['contentRating_Mature 17+'] = np.where(E['contentRating'].str.contains("17+"), 1, 0)
    E['contentRating_Everyone 10+'] = np.where(E['contentRating'].str.contains("10+"), 1, 0)
    E['contentRating_Everyone'] = np.where(E['contentRating'].str.contains("Everyone"), 1, 0)
    E['contentRating_Teen'] = np.where(E['contentRating'].str.contains("Teen"), 1, 0)

    # a column is a pandas series, so this problem is creating pandas series from dictionary
    # Since it is in string format, so first turn string into dictionary, then extract the dictioanry values and turn them into a list
    # now I can use pd.Series to break this list into several columns
    # note that unlike data scraped using the new scraper, they put the 5-star number of ratings in the first position
    # initial data or trakcing data scraped in 201907, 201908, 201912, 202001, 202002 do not have histogram, using
    # ast.literal_eval will raise value error, note str.contains is only a pandas series method
    date_without_score_histogram = ['201907', '201908', '201909', '201912', '202001', '202002']
    if any(x in date_panel_scraped for x in date_without_score_histogram):
        E[['score_5', 'score_4', 'score_3', 'score_2', 'score_1']] = np.nan
        E['ratings'] = np.nan

    else:
        X = E['histogram'].apply(lambda x: ast.literal_eval(x))
        Y = X.apply(lambda x: list(x.values()))
        E[['score_5', 'score_4', 'score_3', 'score_2', 'score_1']] = Y.apply(pd.Series)
        E['ratings'] = E['score_5'] + E['score_4'] + E['score_3'] + E['score_2'] + E['score_1']

    # convert IAP price range into IAP price low or IAP price high -- skip the null
    # for 201812,it seems that if you use list(). the list would break every digit into an element, so I will use str.split into two columns here
    # however, for 2019 and onwards, the inAppProductPrice is already in list, so apply pd.Series worked
    # in Both cases, they have some weird contents, mostly from content rating, in IAPprice column, I do not know why. I just removed all letters [a-z]

    if '201812' in date_panel_scraped:
        E['inAppProductPrice'] = np.where(E['inAppProductPrice'].str.contains('[a-z]', regex = True), np.nan,
                                          E.inAppProductPrice)
        E['inAppProductPrice'] = E['inAppProductPrice'].str.replace('$', '')
        E['inAppProductPrice'] = E['inAppProductPrice'].str.replace('(', '')
        E['inAppProductPrice'] = E['inAppProductPrice'].str.replace(')', '')
        E['inAppProductPrice'] = E['inAppProductPrice'].str.replace("'", '')
        E[['IAP_low', 'IAP_high']] = E['inAppProductPrice'].str.split(",", n=2, expand=True)
        E[['IAP_low', 'IAP_high']] = E[['IAP_low', 'IAP_high']].astype(float)

    else:
        E[['IAP_low', 'IAP_high']] = E['inAppProductPrice'].apply(lambda x: pd.Series(x))
        E['IAP_low'] = E['IAP_low'].str.replace('$', '')
        E['IAP_high'] = E['IAP_high'].str.replace('$', '')
        E['IAP_low'] = np.where(E['IAP_low'].str.contains('[a-z]', regex = True), np.nan,
                                          E.IAP_low)
        E['IAP_high'] = np.where(E['IAP_high'].str.contains('[a-z]', regex = True), np.nan,
                                          E.IAP_high)
        E[['IAP_low', 'IAP_high']] = E[['IAP_low', 'IAP_high']].astype(float)

    E['price'] = E['price'].str.replace('$', '')
    E['price'] = E['price'].astype(float)

    #change column names with date suffix so that
    E = E.add_suffix('_' + date_panel_scraped)
    #fill none with nan, and remove rows that contain nan in ALL columns
    E = E.fillna(value=np.nan)
    E = E.dropna(axis=0, how="all")

    return(E)

#*****************************************************************************************************
#*** INPUT DATA (scraped in and after 202009) is transformed to dataframe
#*****************************************************************************************************

def transform_dict_dataframe(raw_data, date_panel_scraped):
    # the raw data is scraped by 1_functions_scraping_data.py function scraping_apps_according_to_id
    D = pd.DataFrame.from_dict(raw_data)
    E = D.T
    #E['updated_datetime'] = pd.to_datetime(E['updated'], origin='unix')
    E['released_datetime'] = pd.to_datetime(E['released']).dt.date
    E['today_datetime'] = datetime.date.today()
    E['days_since_released'] = (E['today_datetime'] - E['released_datetime']).dt.days
    E = pd.get_dummies(E, columns=['free', 'offersIAP',
                                   'adSupported', 'containsAds', 'contentRating'], dtype=int)
    E['GAME'] = np.where(E['genreId'].str.contains("GAME"), 1, 0)
    E['UTILITIES'] = np.where(np.isin(E['genreId'], ['MEDICAL', 'EDUCATION', 'HEALTH_AND_FITNESS',
                                                     'PARENTING', 'LIBRARIES_AND_DEMO', 'BOOKS_AND_REFERENCE',
                                                     'HOUSE_AND_HOME', 'FINANCE', 'WEATHER',
                                                     'TOOLS', 'BUSINESS', 'PRODUCTIVITY','TRAVEL_AND_LOCAL',
                                                     'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES']), 1, 0)
    E['SOCIAL_MEDIA_LEISURE'] = np.where(np.isin(E['genreId'], ['COMICS', 'BEAUTY','ART_AND_DESIGN','DATING',
                                                                'VIDEO_PLAYERS','SPORTS', 'LIFESTYLE',
                                                                'PERSONALIZATION', 'SHOPPING','FOOD_AND_DRINK',
                                                                'MUSIC_AND_AUDIO',
                                                                'NEWS_AND_MAGAZINES','PHOTOGRAPHY','COMMUNICATION',
                                                                'ENTERTAINMENT', 'SOCIAL','EVENTS']), 1, 0)
    E['check_all_genre_covered'] = E['GAME'] + E['UTILITIES'] + E['SOCIAL_MEDIA_LEISURE']
    #print("Non missing values in each column ", E.count())

    #drop columns that will not be needed
    E.drop(['descriptionHTML', 'summaryHTML', 'contentRatingDescription',
            'recentChanges'], axis=1, inplace=True)

    #first remove none ascii characters from the string then convert object to string type
    E['title'] = E['title'].str.encode("ascii", "ignore").astype("string")
    E['description'] = E['description'].str.encode("ascii", "ignore").astype("string")
    E['summary'] = E['summary'].str.encode("ascii", "ignore").astype("string")
    E['developer'] = E['developer'].str.encode("ascii", "ignore").astype("string")
    E['developerId'] = E['developerId'].str.encode("ascii", "ignore").astype("string")
    E['developerEmail'] = E['developerEmail'].str.encode("ascii", "ignore").astype("string")
    E['developerWebsite'] = E['developerWebsite'].str.encode("ascii", "ignore").astype("string")
    E['developerAddress'] = E['developerAddress'].str.encode("ascii", "ignore").astype("string")
    E['comments'] = E['comments'].str.encode("ascii", "ignore").astype("string")

    # a column is a pandas series, so this problem is creating pandas series from list
    E[['score_1', 'score_2', 'score_3', 'score_4', 'score_5']] = E['histogram'].apply(pd.Series)
    #change column names with date suffix so that
    E = E.add_suffix('_' + date_panel_scraped)
    #fill none with nan, and remove rows that contain nan in ALL columns
    E = E.fillna(value=np.nan)
    E = E.dropna(axis=0, how="all")
    #check
    print("after dropping rows all NA in all columns, the non-missing values in each column ", E.count())
    print(E.dtypes)
    return(E)


#*****************************************************************************************************
#*** MERGE
#*****************************************************************************************************

def merge_dataframes_panels(initial_date, subsequent_date_list):

    folder = initial_date + '_PANEL_DF'
    f_name = 'INITIAL_' + initial_date + '.pickle'
    q = input_path / "__PANELS__" / folder / f_name
    with open(q, 'rb') as filename:
        DF1 = pickle.load(filename)

    panel_dfs = []
    for i in subsequent_date_list:
        f_name = i + '_PANEL.pickle'
        q = input_path / "__PANELS__" / folder / f_name
        with open(q, 'rb') as filename:
            DF2 = pickle.load(filename)
            panel_dfs.append(DF2)

    new = DF1.join(panel_dfs[0], how = 'inner')
    for j in range(1, len(panel_dfs), 1):
        new = new.join(panel_dfs[j], how = 'inner')

    return(new)

######################################################################################
## plot single dataframe
######################################################################################
def plot_dataframe(E, date_track_origin_date):
    # the input is the output of transform_dict_dataframe
    fig = plt.figure(figsize=(8.27, 11.69))
    spec = fig.add_gridspec(ncols=2, nrows=3)

    ax1 = fig.add_subplot(spec[0, 0])
    ax1 = E['minInstalls'].dropna().plot.hist(bins=3)
    ax1.set_ylabel('frequency', fontsize=6)
    ax1.set_yticklabels(ax1.get_xticks(), fontsize=5)
    ax1.set_xlabel('minInstalls', fontsize=6)
    ax1.set_xticklabels(ax1.get_xticks(), fontsize=5)
    ax1.set_title('The Histogram of \n Accumulative Minimum Installs', fontsize=8)

    ax2 = fig.add_subplot(spec[1, 0])
    ax2 = E['score'].dropna().plot.hist()
    ax2.set_ylabel('frequency', fontsize=6)
    ax2.set_xlabel('score', fontsize=6)
    ax2.set_title('The Histogram of \n Weighted Average Score', fontsize=8)

    ax3 = fig.add_subplot(spec[2, 0])
    ax3 = E['score'].plot.hist()
    ax3.set_ylabel('frequency', fontsize=6)
    ax3.set_xlabel('score', fontsize=6)
    ax3.set_title('The Histogram of \n Weighted Average Score', fontsize=8)

    ax4 = fig.add_subplot(spec[0, 1])
    ax4 = E['score'].plot.hist()
    ax4.set_ylabel('frequency', fontsize=6)
    ax4.set_xlabel('score', fontsize=6)
    ax4.set_title('The Histogram of \n Weighted Average Score', fontsize=8)

    ax5 = fig.add_subplot(spec[1, 1])
    ax5 = E['score'].plot.hist()
    ax5.set_ylabel('frequency', fontsize=6)
    ax5.set_xlabel('score', fontsize=6)
    ax5.set_title('The Histogram of \n Weighted Average Score', fontsize=8)

    ax6 = fig.add_subplot(spec[2, 1])
    ax6 = E['score'].plot.hist()
    ax6.set_ylabel('frequency', fontsize=6)
    ax6.set_xlabel('score', fontsize=6)
    ax6.set_title('The Histogram of \n Weighted Average Score', fontsize=8)

    spec.tight_layout(fig)
    f_name = date_track_origin_date + '_visualization.png'
    fig.savefig(os.path.join(graph_output, f_name),
                facecolor='white', edgecolor='none', dpi=800)

######################################################################################
## plot panel trends
# https://python-graph-gallery.com/124-spaghetti-plot/
# https://matplotlib.org/3.1.0/gallery/subplots_axes_and_figures/broken_axis.html
# https://jakevdp.github.io/PythonDataScienceHandbook/04.09-text-and-annotation.html
# here, we are going to use nested functions
######################################################################################

# FUNCTION_1: Create a dataframe just for graphing
# each column represent an app, and each row represent a time point (a panel)
# the entire dataframe is about one variable: minInstalls

def dataframe_for_line_plot(initial_date, panels, variable, **kwargs):

    # open the merged panel dataframe
    folder = initial_date + '_PANEL_DF'
    df_name = initial_date + '_MERGED.pickle'
    q = input_path / "__PANELS__" / folder / df_name
    with open(q, 'rb') as filename:
        B = pickle.load(filename)

    # either randomly choose a subset or take the whole sample
    if 'sample' in kwargs:
        C = B.sample(kwargs['sample'])
    else:
        C = B

    # select only the columns related to the variable
    panels.insert(0, initial_date)
    var_names = variable + '_'
    col_names = list(map(lambda x: var_names + str(x), panels))
    C = C[col_names]
    for i in col_names:
        C[i] = C[i].astype(float)

    # figure out which apps have increase in the variable over the entire period
    end_date = panels[-1]
    end_date_var = variable + '_' + end_date
    initial_date_var = variable + '_' + initial_date
    C.loc[:, 'increase_over_interval'] = C.loc[:, end_date_var] - C.loc[:, initial_date_var]
    C = C.sort_values(by='increase_over_interval', ascending=False)
    n_rows = int(len(C.axes[0]))
    if 'break_rule' in kwargs:
        top_performers = C.loc[C['increase_over_interval'] > kwargs['break_rule']]
        top_performers = top_performers.index.tolist()
    elif n_rows >= 10:
        top_performers = C.head(10)
        top_performers = top_performers.index.tolist()
    else:
        top_performers = C
        top_performers = top_performers.index.tolist()
    C.drop('increase_over_interval', inplace=True, axis=1)
    C = C.T

    # create a column for graphing xticks (since you already modified panels, so you can skip adding the initial date)
    time_axis = []
    for j in panels:
        time_axis.append(datetime.datetime.strptime(j, '%Y%m').strftime('%Y %b'))
    C['x'] = time_axis

    # take the title of app as the value of a dictionary for annotation in plotting
    top_performers_dict = dict.fromkeys(top_performers)
    title_column = 'title_' + initial_date
    for app_id in top_performers_dict.keys():
        top_performers_dict[app_id] = B.at[app_id, title_column]

    return(C, top_performers_dict)

###############################################################################################
## LINE PLOTS
# create new dataframe which each column holds the series of one app's data across panels
###############################################################################################
def graph_line_plots(C, initial_date, panels, variable, **kwargs):
    fig = plt.figure(figsize=(11.69, 8.27))
    plt.style.use('seaborn-paper')
    spec = fig.add_gridspec(ncols=1, nrows=1)

    # first subplot is the overall picture
    ax1 = fig.add_subplot(spec[0, 0])
    for column in C.drop('x', axis=1):
        ax1.plot(C['x'], C[column], marker='', color='powderblue', linewidth=1, alpha=0.9, label=column)

    # if top performers are specified, highlight those lines
    last_panel = panels[-1]
    annotate_pos = variable + '_' + last_panel
    string_initial_date = datetime.datetime.strptime(initial_date, '%Y%m').strftime('%Y %b')
    string_end_date = datetime.datetime.strptime(panels[-1], '%Y%m').strftime('%Y %b')
    if 'top_performers' in kwargs:
        x_coord_frac = 0.95
        y_coord_frac = 0.95
        for app_id, app_title in kwargs['top_performers'].items():
            ax1.plot(C['x'], C[app_id], marker='', color='deepskyblue', linewidth=2, alpha=0.7)
            ax1.annotate(app_title,
                        xy=(string_end_date, C.at[annotate_pos, app_id]),
                        xycoords='data',
                        xytext=(x_coord_frac, y_coord_frac),
                        textcoords='axes fraction',
                        arrowprops={'arrowstyle': '->'},
                        horizontalalignment='right',
                        verticalalignment='bottom')
            x_coord_frac += -0.08
            y_coord_frac += -0.08

    #ax1.grid(True)
    ax1.set_xlabel('time')
    if variable == 'minInstalls':
        ax1.set_ylabel('cumulative minimum installs')
        title = 'The Number of Minimum Installs of Apps from ' + string_initial_date + ' to ' + string_end_date
        ax1.set_title(title)

    # highlight the line that has the highest increase in min installs, and label them out in the legend
    spec.tight_layout(fig)
    f_name = initial_date + '_panels_' + variable + '_multiple_lines.png'
    fig.savefig(os.path.join(graph_output, variable, f_name),
                facecolor='white', edgecolor='none')

    # second subplot zoom into the portion where the installs are lower and cramped together


###############################################################################################
## SCATTER PLOTS
# (different color represents different category, and x-axis is the time)
###############################################################################################
def dataframe_for_scatter_plot_by_group(initial_date, panels, variable, group, **kwargs):
    # open the merged panel dataframe
    folder = initial_date + '_PANEL_DF'
    df_name = initial_date + '_MERGED.pickle'
    q = input_path / "__PANELS__" / folder / df_name
    with open(q, 'rb') as filename:
        B = pickle.load(filename)

    # either randomly choose a subset or take the whole sample
    if 'sample' in kwargs:
        C = B.sample(kwargs['sample'])
    else:
        C = B

    # select only the columns related to the variable, the panels and the group
    panels.insert(0, initial_date)
    var_names = variable + '_'
    var_cols = list(map(lambda x: var_names + str(x), panels))
    for i in var_cols:
        C[i] = C[i].astype(float)
    last_panel = panels[-1] # use the category specified in the most recent panel
    group_cols = list(map(lambda x: str(x) + '_' + last_panel, group))
    selected_cols = var_cols + group_cols
    C = C[selected_cols]

    # reshape the data with y-axis value in one column, x-axis value in another column (all panels stacked into one column)
    # and categories in the third column
    C['category'] = np.nan
    for i in group:
        col_name = str(i) + '_' + last_panel
        C.loc[C[col_name] == 1, 'category'] = i

    C['app_id'] = C.index
    C.reset_index(drop=True, inplace=True)
    D = pd.wide_to_long(C, stubnames=variable, i='app_id', j='panel', sep='_')
    D.reset_index(inplace=True)

    # convert to datetime format for panel column
    D['panel'] = pd.to_datetime(D['panel'], format='%Y%m') # when you convert object to datetime, the format has to match the original format
    D['panel'] = D['panel'].dt.strftime('%Y %b') # then you can change the format to the desired datetime format

    return(D)



###############################################################################################
## VIOLIN PLOTS
# create new dataframe which each column holds the series of one app's data across panels
###############################################################################################
def dataframe_for_violin_plots(initial_date, panels, variable, group, binary, panel_for_group_and_binary):
    folder_1 = initial_date + '_PANEL_DF'
    merged_df = initial_date + '_MERGED.pickle'
    q = input_path / '__PANELS__' / folder_1 / merged_df
    with open(q, 'rb') as f:
        D = pickle.load(f)

    # subset dataframe with the same variable in all the panels
    panels.insert(0, initial_date)
    var_cols = list(map(lambda x: variable + '_' + str(x), panels))
    for i in var_cols:
        D[i] = D[i].astype(float)

    if group == 'category':
        group_names = ['GAME', 'UTILITIES', 'SOCIAL_MEDIA_LEISURE']
        group_cols = []
        for i in group_names:
            group_cols.extend(list(map(lambda x: i + '_' + str(x), panels)))

        # since group definition for the same app may change over time, for the purpose of graph, we only select one panel of group definition
        group_panel_col = list(map(lambda x: str(x) + '_' + panel_for_group_and_binary, group_names))

    if binary in ['free_True', 'offersIAP_True', 'containsAds_True']:
        binary_cols = list(map(lambda x: binary + '_' + str(x), panels))
        binary_panel_col = [binary + '_' + panel_for_group_and_binary]

    selected_cols = var_cols + group_panel_col + binary_panel_col
    D = D[selected_cols]

    # reshape the dataframe from wide to long (stack all panels into a single column)
    D['app_id'] = D.index
    D.reset_index(drop=True, inplace=True)
    E = pd.wide_to_long(D, stubnames=variable, i='app_id', j='panels', sep='_')
    ## drop the last panel from group and binary column names
    A = group_panel_col + binary_panel_col
    for i in A:
        need_to_be_removed = "_" + panel_for_group_and_binary
        new_name = i.replace(need_to_be_removed, "")
        E.rename(columns={i: new_name}, inplace=True)

    # rename binary variable
    new_name = binary.replace('_True', "")
    E.rename(columns={binary: new_name}, inplace=True)

    # convert to datetime format for panel column
    E.reset_index(inplace=True) # because app_id and panel are automatically turned into indices after wide_to_long, this step turns them back into columns
    E['panels'] = pd.to_datetime(E['panels'], format='%Y%m') # when you convert object to datetime, the format has to match the original format
    E['panels'] = E['panels'].dt.strftime('%Y %b') # then you can change the format to the desired datetime format

    # replace the 1 0 with appropriate content for showing in the legend of graphs
    if group == 'category':
        E['SOCIAL_MEDIA_LEISURE'].replace(1, 'social_media_leisure', inplace=True)
        E['SOCIAL_MEDIA_LEISURE'].replace(0, '', inplace=True)
        E['UTILITIES'].replace(1, 'utilities', inplace=True)
        E['UTILITIES'].replace(0, '', inplace=True)
        E['GAME'].replace(1, 'game', inplace=True)
        E['GAME'].replace(0, '', inplace=True)
        E[group] = E['SOCIAL_MEDIA_LEISURE'] + E['UTILITIES'] + E['GAME']
        # delete the rows with '' in category, I have checked the original data, the reason being in some panels genreId is nan
        E = E[E[group]!='']
    if binary == 'free_True':
        E['free'].replace(1, 'free', inplace=True)
        E['free'].replace(0, 'paid', inplace=True)

    return(E)

##################################################################################################
def graph_violin_plots(E, x, y, hue, col):
    # set background theme
    sns.set_theme(style="whitegrid")
    # plot
    plot = sns.catplot(x=x, y=y,
                    hue=hue, col=col,
                    data=E, kind="violin",
                    scale="count",
                    scale_hue = True,
                    inner="quartile",
                    split=True,
                    bw=.1,
                    height=4, aspect=.7)

    initial_date = E['panels'].unique()[0]
    end_date = E['panels'].unique()[-1]
    # add plot title
    plt.subplots_adjust(top=0.8)
    title_text = 'Violin Plots of ' + y + ' by ' + hue + ' and ' + col + ' from ' + initial_date + ' to ' + end_date
    plot.fig.suptitle(title_text)
    # rotate plot x-axis so that the texts do not overlap
    plot.set_xticklabels(rotation=60)
    # save plot
    f_name = initial_date + '_' + y + '_by_' + hue + '_and_' + col + '_panel_violin.png'
    plot.savefig(os.path.join(graph_output, y, f_name),
                facecolor='white', edgecolor='none', dpi=500)

##################################################################################################
##################################################################################################
## BIG THING (DEVIDE APPS ACCORDING TO MIN-INSTALLS)
# I suspect different groups of apps may exhibit entirely different characteristics (such as the distribution of score, reviews, text sentiment and so on)
# the most obvious way of grouping is by the range of minInstalls, apps with super large number of installs may be quite different from niche products which only have a few installs
# I think this is more important than looking at categories, the thing with categories is that some of them have blurry lines (social, media, leisure shopping all combined together in one app)
# the other thing is the category is only important when one analyze gaming and non-gaming apps.
# The function below will transform data into different minimum install groups, and then feed the transformed data back to the graphing functions above, to see violin plots for different install groups
##################################################################################################
##################################################################################################
