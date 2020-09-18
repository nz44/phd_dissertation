# The file include functions that transform raw scraped app data into
# 1. categorical level and developer level
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
import datetime
import os.path

graph_output = '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/graphs'

#*****************************************************************************************************
#*** INPUT DATA is transformed to dataframe
#*****************************************************************************************************

def transform_dict_dataframe(raw_data):
    # the raw data is scraped by 1_functions_scraping_data.py function scraping_apps_according_to_id
    D = pd.DataFrame.from_dict(raw_data)
    E = D.T
    E = E.fillna(value=np.nan)
    E = E.dropna(axis=0, how="all")
    print("after dropping rows all NA in all columns, the non-missing values in each column ", E.count())
    #E['updated_datetime'] = pd.to_datetime(E['updated'], origin='unix')
    E['released_datetime'] = pd.to_datetime(E['released']).dt.date
    E['today_datetime'] = datetime.date.today()
    E['days_since_released'] = (E['today_datetime'] - E['released_datetime']).dt.days
    E = pd.get_dummies(E, columns=['free', 'sale', 'adSupported', 'containsAds'], dtype=float)
    E['GAME'] = np.where(E['genreId'].str.contains("GAME"), 1, 0)
    E['UTILITIES'] = np.where(np.isin(E['genreId'], ['MEDICAL', 'EDUCATION', 'HEALTH_AND_FITNESS',
                                                     'PARENTING', 'LIBRARIES_AND_DEMO',
                                                     'HOUSE_AND_HOME', 'FINANCE', 'WEATHER',
                                                     'TOOLS', 'BUSINESS', 'PRODUCTIVITY','TRAVEL_AND_LOCAL',
                                                     'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES']), 1, 0)
    E['SOCIAL_MEDIA_LEISURE'] = np.where(np.isin(E['genreId'], ['COMICS', 'BEAUTY','ART_AND_DESIGN','DATING',
                                                                'VIDEO_PLAYERS','SPORTS', 'LIFESTYLE',
                                                                'PERSONALIZATION', 'SHOPPING','FOOD_AND_DRINK',
                                                                'MUSIC_AND_AUDIO', 'BOOKS_AND_REFERENCE',
                                                                'NEWS_AND_MAGAZINES','PHOTOGRAPHY','COMMUNICATION',
                                                                'ENTERTAINMENT', 'SOCIAL','EVENTS']), 1, 0)
    E['check_all_genre_covered'] = E['GAME'] + E['UTILITIES'] + E['SOCIAL_MEDIA_LEISURE']
    #print("Non missing values in each column ", E.count())
    print(E.dtypes)
    return(E)

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