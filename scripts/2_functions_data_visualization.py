# The file include functions that transform raw scraped app data into
# 1. categorical level and developer level
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import datetime
import os.path
import ast

graph_output = '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs'

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
    E['contentRating_18+'] = np.where(E['contentRating'].str.contains("18+"), 1, 0)
    E['contentRating_17+'] = np.where(E['contentRating'].str.contains("17+"), 1, 0)
    E['contentRating_10+'] = np.where(E['contentRating'].str.contains("10+"), 1, 0)
    E['contentRating_everyone'] = np.where(E['contentRating'].str.contains("Everyone"), 1, 0)
    E['contentRating_teen'] = np.where(E['contentRating'].str.contains("Teen"), 1, 0)

    # a column is a pandas series, so this problem is creating pandas series from dictionary
    # Since it is in string format, so first turn string into dictionary, then extract the dictioanry values and turn them into a list
    # now I can use pd.Series to break this list into several columns
    # note that unlike data scraped using the new scraper, they put the 5-star number of ratings in the first position
    # initial data or trakcing data scraped in 201907, 201908, 201912, 202001, 202002 do not have histogram, using
    # ast.literal_eval will raise value error, note str.contains is only a pandas series method
    date_without_score_histogram = ['201907', '201908', '201912', '202001', '202002']
    if any(x in date_panel_scraped for x in date_without_score_histogram):
        E[['score_5', 'score_4', 'score_3', 'score_2', 'score_1']] = np.nan

    else:
        X = E['histogram'].apply(lambda x: ast.literal_eval(x))
        Y = X.apply(lambda x: list(x.values()))
        E[['score_5', 'score_4', 'score_3', 'score_2', 'score_1']] = Y.apply(pd.Series)

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

    E['price'] = E['price'].str.replace('$', '')
    E['price'] = E['price'].astype(float)

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

######################################################################################
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