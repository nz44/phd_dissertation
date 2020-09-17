# The file include functions that transform raw scraped app data into
# 1. categorical level and developer level
import numpy as np
import matplotlib.pyplot as plt

def transform_raw_to_category_data(raw_data):
    # get a list of unique categories in the raw_data
    categories = []
    for appid, appdetails in raw_data.items():
        if appdetails is not None:
            categories.append(appdetails['genreId'])
    categories = list(set(categories))
    appdetails_categories = dict.fromkeys(categories)

    for category in appdetails_categories.keys():
        content = {}
        for appid, appdetails in raw_data.items():
            if appdetails is not None:
                if appdetails['genreId'] == category:
                    content[appdetails['appId']] = appdetails
        appdetails_categories[category] = content

    num_in_cat = dict.fromkeys(categories)
    for k, v in appdetails_categories.items():
        num_in_cat[k] = len(v)
    sort_num_in_cat = sorted(num_in_cat.items(), key=lambda x: x[1], reverse=True)
    print(sort_num_in_cat)

    return(appdetails_categories)


# explore the summary stats within each category
# appdetails is any data file that is organized in dictionaries with appid as dictionary key
# attribute is the variable you want to graph against
# variables can be histogramed (below are the key names in appdetails)
# -- minInstalls
# -- score (weighted average score)
# -- ratings (cumulative number of ratings)
# -- reviews (cumulative number of reviews)
# -- free (T/F)
# -- price
# -- offersIAP
# -- inAppProductPrice
# -- released (date, could graph in number of apps released in a date range)
# -- updated (unix date format)

# https://www.statista.com/statistics/269884/android-app-downloads/
def histogram(appdetails, attribute):
    # put data of this attribute of all apps into a numpy array
    X = np.array([])
    for appid, details in appdetails.items():
        if details is not None:
            if details[attribute] is not None:
                X = np.append(X, details[attribute])

    if attribute == "minInstalls":
        bin_list = [0, 500, 1000, 5000, 10000, 50000, 100000, 500000, 1000000, 5000000, 10000000, 50000000, 100000000, max(X)]
        a = np.histogram(X, bins = bin_list)
        print(attribute, a)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(X, bin_list, color='#0504aa', alpha=0.7)

    else:
        a = np.histogram(X, bins = 'auto')
        print(attribute, a)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.hist(X, 'auto', color='#0504aa', alpha=0.7)

    ax.grid(True)
    ax.set_ylabel('frequency')
    ax.set_xlabel(attribute)
    fig.savefig(attribute + '_' + 'histogram.png', facecolor='white', edgecolor='none', dpi = 300)


def boxplot(appdetails, attribute):
    # put data of this attribute of all apps into a numpy array
    X = np.array([])
    for appid, details in appdetails.items():
        if details is not None:
            X = np.append(X, details[attribute])
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.boxplot(X)
    ax.grid(True)
    ax.set_ylabel('frequency')
    ax.set_xlabel(attribute)
    fig.savefig(attribute + '_' + 'boxplot.png', facecolor='white', edgecolor='none', dpi = 300)