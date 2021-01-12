from google_play_scraper import app
import pandas as pd
from tqdm import tqdm
import numpy as np
import random
import datetime
import functools
import collections
import operator


# get ID list
# data scraped before 2020 Sep are organized in a list, each dictionary inside the list contains attributes and their values
# for the data scraped in 2020 Sep and onwards, they are organized in dictionary with key as appid, and then their appdetails,
# so id_list should just be C.keys()
def get_id_from_data_before_202009(C):
    id_list = []
    for i in C:
        if 'appId' in i.keys():
            id_list.append(i['appId'])
        else:
            id_list.append(i['app_id'])
    return(id_list)


# scraping using google_play_scraper app function
# the input is from get_id_from_old_data
def scraping_apps_according_to_id(id_list):
    app_details = dict.fromkeys(id_list)
    # print(app_details)
    for j in tqdm(range(len(id_list)), desc="scraping..."):
        try:
            app_details[id_list[j]] = app(id_list[j])
        except:
            pass
    return(app_details)


class app_detail_dicts():
    # this self is opened from open_files class open_app_details_dict method or open_initial_panel_with_its_tracking_panels method
    def __init__(self, d):
        self.d = d
        # for some dataframe, there are two very similar cols and the contents could be combine
        # first combine, and then rename
        self.df1_to_combine = ['developerId', 'developerWebsite', 'developerEmail', 'developerAddress', 'offersIAP',
                          'appId', 'contentRating']
        self.df2_to_combine = ['developer_id', 'developer_url', 'developer_email', 'developer_address', 'iap',
                          'app_id', 'content_rating']
        self.all_cols_to_combine = self.df1_to_combine + self.df2_to_combine

        self.cols_to_rename = {"category": "genreId", "iap": "offersIAP", "developer_id": "developerId",
                          "content_rating": "contentRating", "iap_range": "inAppProductPrice",
                          "developer_email": "developerEmail", "developer_url": "developerWebsite",
                          "developer_address": "developerAddress", "app_id": "appId"}

        self.cols_to_drop = ['editors_choice', 'interactive_elements', 'description_html', 'descriptionHTML', 'privacyPolicy',
                        'recentChanges', 'recent_changes', 'summaryHTML', 'androidVersionText', 'icon', 'genre',
                        'headerImage', 'screenshots', 'video', 'videoImage', 'developerInternalID', 'content_rating',
                        'contentRatingDescription', 'installs', 'url', 'free_False', 'offersIAP_False',
                        'recentChangesHTML', 'adSupported_False', 'containsAds_False', 'genreId', 'recentChanges',
                        'category', 'iap_range', 'app_id', 'androidVersionText', 'androidVersion',
                        'current_version', 'version', 'required_android_version', 'sale', 'saleTime', 'originalPrice',
                        'saleText']

        self.cols_to_dummy = ['free', 'offersIAP', 'adSupported', 'containsAds', 'genreId', 'contentRating']

        self.cols_to_string = ['title', 'description', 'summary', 'developer', 'developerId', 'developerEmail',
                          'developerWebsite', 'developerAddress']

        self.cols_to_float = ['minInstalls', 'score', 'ratings', 'reviews', 'size', 'androidVersion']

        self.cols_in_list_or_dict_need_to_expand = ['histogram', 'price', 'comments']

        self.cols_to_datetime = ['released', 'updated']


    def convert_keys_to_datetime(self):
        datetime_list = []
        for i in self.d.keys():
            date_time_obj = datetime.datetime.strptime(i, '%Y%m')
            datetime_list.append(date_time_obj.date())
        return datetime_list

    def get_a_glimpse(self):
        dict_data = self.convert_list_data_to_dict_with_appid_keys()
        for k, v in dict_data.items():
            print('PANEL', k)
            random_app_id = random.choice(list(v))
            print('a random app id is:', random_app_id)
            the_app_detail = v[random_app_id]
            if the_app_detail is not None:
                for feature, content in the_app_detail.items():
                    print(feature, " : ", content)
                print()
            else:
                print('the app detail is None')
                print()

    def check_for_duplicate_cols(self):
        df_dict = self.combine_similar_and_drop_extra_cols()
        for i, df in df_dict.items():
            # check for duplicate columns
            unique_cols = set(df.columns)
            print(i)
            if len(unique_cols) == len(df.columns):
                print('There are no duplicate columns')
                print(df.columns)
            else:
                print('The following columns are duplicates:')
                print([item for item, count in collections.Counter(df.columns).items() if
                       count > 1])

    def combine_similar_and_drop_extra_cols(self):
        new_data = dict.fromkeys(self.d.keys())
        old_data = self.convert_from_dict_to_df()
        for i, df in old_data.items():
            ## __________________________________________ Combine Similar Cols ___________________________________________________
            # first check whether the dataframe contains BOTH old and new cols, yes then combine, no keep the same
            if all(x in df.columns for x in self.all_cols_to_combine):
                # combine duplicate columns, first take them out as separate dataframes
                combine_df1 = df[self.df1_to_combine].fillna('').astype(str)
                combine_df2 = df[self.df2_to_combine].fillna('').astype(str)
                df.drop(self.all_cols_to_combine, axis=1, inplace=True)

                # second, rename the columns in the second dataframe to the same as the first dataframe
                for old_col_name, new_col_name in self.cols_to_rename.items():
                    if old_col_name in combine_df2.columns:
                        combine_df2.rename(columns={old_col_name: new_col_name}, inplace=True)

                # third, combine the dataframe with overwrite == False so that if one col exist in one but not another, they get kept the same
                df3 = combine_df1.combine(combine_df2, operator.add, overwrite=False)
                df = df.join(df3, how='inner')

            ## __________________________________________ Drop Extra Cols and Add Panel Suffix ____________________________________
            # drop columns that will not be needed
            # after dropping columns, the only useful col that exist in and after 202009 is inAppProductPrice, which does not existed before
            df.drop([x for x in df.columns if x in self.cols_to_drop], axis=1, inplace=True)
            # # change column names with date suffix so that
            df = df.add_suffix('_' + i)
            new_data[i] = df
        return new_data


    def format_dfs(self):
        x = datetime.datetime(2020, 9, 1)
        i = '201812'
        date_time_obj = datetime.datetime.strptime(i, '%Y%m')
        return i

    def convert_from_dict_to_df(self):
        old_data = self.convert_list_data_to_dict_with_appid_keys()
        new_data = dict.fromkeys(self.d.keys())
        for i in old_data.keys():
            df = pd.DataFrame.from_dict(old_data[i])
            df2 = df.T
            new_data[i] = df2
        return new_data


    # convert the data scraped before 202009 from list of dictionaries to a dictionary with app ids as keys
    # some of them uses app_id as key and some of them uses appId as key
    # from 202009 onwards, the key conversion happened in scraping stage
    def convert_list_data_to_dict_with_appid_keys(self):
        new_data = dict.fromkeys(self.d.keys(), {})
        x = datetime.datetime(2020, 9, 1)
        for i in self.d.keys():
            date_time_obj = datetime.datetime.strptime(i, '%Y%m')
            if date_time_obj < x: # check the date is before 2020 September
                for j in self.d[i]: # self[i] is a list here
                    if j is not None:
                        if 'app_id' in j.keys():
                            new_data[i][j['app_id']] = j
                        elif 'appId' in j.keys():
                            new_data[i][j['appId']] = j
            else: # for data scraped in 202009 and onwards, leave them as they are
                new_data[i] = self.d[i]
        return new_data


# ================================================================
# extra code
##################################################################


                ## __________________________________________ Combine Similar Cols ___________________________________________________
                # df_dummies = df2[[x for x in df2.columns if x in cols_to_dummy]]
                # df_cat = pd.get_dummies(df_dummies.categories.apply(pd.Series).stack(),
                #                     dummy_na=True, dtype=int).sum(level=0)
                # df2 = df2.join(df_cat, how='inner')
                #
                # # create new minInstalls column
                # df2["minInstalls"] = df2['installs'].str.replace('[^\w\s]', '')
                # df2['minInstalls'] = df2['minInstalls'].astype(float)
                #
                # df2[[x for x in df2.columns if x in cols_to_string]].astype("string")
        #
        #         # remove the list brackets in certain columns
        #         df2['genreId'] = df2['genreId'].apply(pd.Series)
        #
        #         df2['GAME'] = np.where(df2['genreId'].str.contains("GAME"), 1, 0)
        #         utils = ['MEDICAL', 'EDUCATION', 'HEALTH_AND_FITNESS',
        #                  'PARENTING', 'LIBRARIES_AND_DEMO',
        #                  'HOUSE_AND_HOME', 'FINANCE', 'WEATHER', 'BOOKS_AND_REFERENCE',
        #                  'TOOLS', 'BUSINESS', 'PRODUCTIVITY', 'TRAVEL_AND_LOCAL',
        #                  'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES']
        #         df2['UTILITIES'] = np.where(df2['genreId'].str.contains('|'.join(utils)), 1, 0)
        #
        #         social_media_l = ['COMICS', 'BEAUTY', 'ART_AND_DESIGN', 'DATING',
        #                           'VIDEO_PLAYERS', 'SPORTS', 'LIFESTYLE',
        #                           'PERSONALIZATION', 'SHOPPING', 'FOOD_AND_DRINK',
        #                           'MUSIC_AND_AUDIO',
        #                           'NEWS_AND_MAGAZINES', 'PHOTOGRAPHY', 'COMMUNICATION',
        #                           'ENTERTAINMENT', 'SOCIAL', 'EVENTS']
        #         df2['SOCIAL_MEDIA_LEISURE'] = np.where(df2['genreId'].str.contains('|'.join(social_media_l)), 1, 0)
        #
        #         # remove duplicate groups categories
        #         # df2.loc[(E.UTILITIES == 1 & ), 'Event'] = 'Hip-Hop'
        #         df2['UTILITIES'] = np.where(((df2.GAME == 1) & (df2.UTILITIES == 1)), 0, df2.UTILITIES)
        #         df2['SOCIAL_MEDIA_LEISURE'] = np.where(((df2.GAME == 1) & (df2.SOCIAL_MEDIA_LEISURE == 1)), 0, df2.SOCIAL_MEDIA_LEISURE)
        #         # because the conflicts mostly occur where family education is in the same category as entertainment or art and design
        #         df2['SOCIAL_MEDIA_LEISURE'] = np.where(((df2.UTILITIES == 1) & (df2.SOCIAL_MEDIA_LEISURE == 1)), 0,
        #                                              df2.SOCIAL_MEDIA_LEISURE)
        #
        #         df2['check_all_genre_covered'] = df2['GAME'] + df2['UTILITIES'] + df2['SOCIAL_MEDIA_LEISURE']
        #
        #         # a column is a pandas series, so this problem is creating pandas series from dictionary
        #         # Since it is in string format, so first turn string into dictionary, then extract the dictioanry values and turn them into a list
        #         # now I can use pd.Series to break this list into several columns
        #         # note that unlike data scraped using the new scraper, they put the 5-star number of ratings in the first position
        #         # initial data or trakcing data scraped in 201907, 201908, 201912, 202001, 202002 do not have histogram, using
        #         # ast.literal_eval will raise value error, note str.contains is only a pandas series method
        #         y = datetime.datetime(2020, 3, 1)
        #         if date_time_obj < y:  # before 202003, you do not have score histograms
        #             df2[['score_5', 'score_4', 'score_3', 'score_2', 'score_1']] = np.nan
        #             df2['ratings'] = np.nan
        #
        #         else: # there are actually those dataframe scraped between 202003 and 202009
        #             temp1 = df2['histogram'].apply(lambda x: ast.literal_eval(x))
        #             temp2 = temp1.apply(lambda x: list(x.values()))
        #             df2[['score_5', 'score_4', 'score_3', 'score_2', 'score_1']] = temp2.apply(pd.Series)
        #             df2['ratings'] = df2['score_5'] + df2['score_4'] + df2['score_3'] + df2['score_2'] + df2['score_1']
        #
        #         # convert IAP price range into IAP price low or IAP price high -- skip the null
        #         # for 201812,it seems that if you use list(). the list would break every digit into an element, so I will use str.split into two columns here
        #         # however, for 2019 and onwards, the inAppProductPrice is already in list, so apply pd.Series worked
        #         # in Both cases, they have some weird contents, mostly from content rating, in IAPprice column, I do not know why. I just removed all letters [a-z]
        #         z = datetime.datetime(2018, 12, 1)
        #         if date_time_obj == z:
        #             df2['inAppProductPrice'] = np.where(df2['inAppProductPrice'].str.contains('[a-z]', regex=True), np.nan,
        #                                               df2.inAppProductPrice)
        #             df2['inAppProductPrice'] = df2['inAppProductPrice'].str.replace('$', '')
        #             df2['inAppProductPrice'] = df2['inAppProductPrice'].str.replace('(', '')
        #             df2['inAppProductPrice'] = df2['inAppProductPrice'].str.replace(')', '')
        #             df2['inAppProductPrice'] = df2['inAppProductPrice'].str.replace("'", '')
        #             df2[['IAP_low', 'IAP_high']] = df2['inAppProductPrice'].str.split(",", n=2, expand=True)
        #             df2[['IAP_low', 'IAP_high']] = df2[['IAP_low', 'IAP_high']].astype(float)
        #
        #         else:
        #             df2[['IAP_low', 'IAP_high']] = df2['inAppProductPrice'].apply(lambda x: pd.Series(x))
        #             df2['IAP_low'] = df2['IAP_low'].str.replace('$', '')
        #             df2['IAP_high'] = df2['IAP_high'].str.replace('$', '')
        #             df2['IAP_low'] = np.where(df2['IAP_low'].str.contains('[a-z]', regex=True), np.nan,
        #                                     df2.IAP_low)
        #             df2['IAP_high'] = np.where(df2['IAP_high'].str.contains('[a-z]', regex=True), np.nan,
        #                                      df2.IAP_high)
        #             df2[['IAP_low', 'IAP_high']] = df2[['IAP_low', 'IAP_high']].astype(float)
        #
        #         df2['price'] = df2['price'].str.replace('$', '').astype(float)
        #         # E['price'] = E['price'].astype(float)
        #
                # fill none with nan, and remove rows that contain nan in ALL columns
                # df2 = df2.fillna(value=np.nan)
                # df2 = df2.dropna(axis=0, how="all")
                # df2.drop([x for x in df2.columns if x in cols_to_drop], axis=1, inplace=True)
                # # change column names with date suffix so that
                # df2 = df2.add_suffix('_' + i)
                # new_data.append(df2)
        #
        #     else: # panels scraped in and after 202009
        #         df = pd.DataFrame.from_dict(dict_data[i])
        #         df2 = df.T
        #         # df2['updated_datetime'] = pd.to_datetime(df2['updated'], origin='unix')
        #         df2['released_datetime'] = pd.to_datetime(df2['released']).dt.date
        #         df2['today_datetime'] = datetime.date.today()
        #         df2['days_since_released'] = (df2['today_datetime'] - df2['released_datetime']).dt.days
        #         df2 = pd.get_dummies(df2, columns=['free', 'offersIAP',
        #                                        'adSupported', 'containsAds', 'contentRating'], dtype=int)
        #         df2['GAME'] = np.where(df2['genreId'].str.contains("GAME"), 1, 0)
        #         df2['UTILITIES'] = np.where(np.isin(df2['genreId'], ['MEDICAL', 'EDUCATION', 'HEALTH_AND_FITNESS',
        #                                                          'PARENTING', 'LIBRARIES_AND_DEMO',
        #                                                          'BOOKS_AND_REFERENCE',
        #                                                          'HOUSE_AND_HOME', 'FINANCE', 'WEATHER',
        #                                                          'TOOLS', 'BUSINESS', 'PRODUCTIVITY',
        #                                                          'TRAVEL_AND_LOCAL',
        #                                                          'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES']), 1, 0)
        #         df2['SOCIAL_MEDIA_LEISURE'] = np.where(
        #             np.isin(df2['genreId'], ['COMICS', 'BEAUTY', 'ART_AND_DESIGN', 'DATING',
        #                                    'VIDEO_PLAYERS', 'SPORTS', 'LIFESTYLE',
        #                                    'PERSONALIZATION', 'SHOPPING', 'FOOD_AND_DRINK',
        #                                    'MUSIC_AND_AUDIO',
        #                                    'NEWS_AND_MAGAZINES', 'PHOTOGRAPHY',
        #                                    'COMMUNICATION',
        #                                    'ENTERTAINMENT', 'SOCIAL', 'EVENTS']), 1, 0)
        #         df2['check_all_genre_covered'] = df2['GAME'] + df2['UTILITIES'] + df2['SOCIAL_MEDIA_LEISURE']
        #         # print("Non missing values in each column ", E.count())
        #
        #         # first remove none ascii characters from the string then convert object to string type
        #         df2['title'] = df2['title'].str.encode("ascii", "ignore").astype("string")
        #         df2['description'] = df2['description'].str.encode("ascii", "ignore").astype("string")
        #         df2['summary'] = df2['summary'].str.encode("ascii", "ignore").astype("string")
        #         df2['developer'] = df2['developer'].str.encode("ascii", "ignore").astype("string")
        #         df2['developerId'] = df2['developerId'].str.encode("ascii", "ignore").astype("string")
        #         df2['developerdf2mail'] = df2['developerdf2mail'].str.encode("ascii", "ignore").astype("string")
        #         df2['developerWebsite'] = df2['developerWebsite'].str.encode("ascii", "ignore").astype("string")
        #         df2['developerAddress'] = df2['developerAddress'].str.encode("ascii", "ignore").astype("string")
        #         df2['comments'] = df2['comments'].str.encode("ascii", "ignore").astype("string")
        #
        #         # a column is a pandas series, so this problem is creating pandas series from list
        #         df2[['score_1', 'score_2', 'score_3', 'score_4', 'score_5']] = df2['histogram'].apply(pd.Series)
                # fill none with nan, and remove rows that contain nan in ALL columns
                # df2 = df2.fillna(value=np.nan)
                # df2 = df2.dropna(axis=0, how="all")

        #
        #         new_data.append(df2)
        # # merge all the dataframes in the list
        # new_data_df = functools.reduce(lambda x1, x2: x1.join(x2, how='inner'), new_data)
