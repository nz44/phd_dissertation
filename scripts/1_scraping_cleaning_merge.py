from google_play_scraper import app
from tqdm import tqdm
import numpy as np
import random
import datetime


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

    def convert_keys_to_datetime(self):
        datetime_list = []
        for i in self.d.keys():
            date_time_obj = datetime.datetime.strptime(i, '%Y%m')
            datetime_list.append(date_time_obj.date())
        return datetime_list

    def get_a_glimpse(self):
        for k, v in self.d.items():
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


    def transform_old_scraper_dict_dataframe(self):
        # c are panels scraped before 202009
        # that means in the folder TRACKING_THE_SAME_ID_MONTHLY_SCRAPE 202009 and later does not need to use this function
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
        # E['summary'] = E['summary'].str.encode("ascii", "ignore").astype("string")
        E['developer'] = E['developer'].str.encode("ascii", "ignore").astype("string")
        E['developerId'] = E['developerId'].str.encode("ascii", "ignore").astype("string")
        E['developerEmail'] = E['developerEmail'].str.encode("ascii", "ignore").astype("string")
        E['developerWebsite'] = E['developerWebsite'].str.encode("ascii", "ignore").astype("string")
        E['developerAddress'] = E['developerAddress'].str.encode("ascii", "ignore").astype("string")
        # E['comments'] = E['comments'].str.encode("ascii", "ignore").astype("string")

        # remove the list brackets in certain columns
        # E['genreId'] = E['genreId'].apply(pd.Series)

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
        # E.loc[(E.UTILITIES == 1 & ), 'Event'] = 'Hip-Hop'
        E['UTILITIES'] = np.where(((E.GAME == 1) & (E.UTILITIES == 1)), 0, E.UTILITIES)
        E['SOCIAL_MEDIA_LEISURE'] = np.where(((E.GAME == 1) & (E.SOCIAL_MEDIA_LEISURE == 1)), 0, E.SOCIAL_MEDIA_LEISURE)
        # because the conflicts mostly occur where family education is in the same category as entertainment or art and design
        E['SOCIAL_MEDIA_LEISURE'] = np.where(((E.UTILITIES == 1) & (E.SOCIAL_MEDIA_LEISURE == 1)), 0,
                                             E.SOCIAL_MEDIA_LEISURE)

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
            E['inAppProductPrice'] = np.where(E['inAppProductPrice'].str.contains('[a-z]', regex=True), np.nan,
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
            E['IAP_low'] = np.where(E['IAP_low'].str.contains('[a-z]', regex=True), np.nan,
                                    E.IAP_low)
            E['IAP_high'] = np.where(E['IAP_high'].str.contains('[a-z]', regex=True), np.nan,
                                     E.IAP_high)
            E[['IAP_low', 'IAP_high']] = E[['IAP_low', 'IAP_high']].astype(float)

        E['price'] = E['price'].str.replace('$', '')
        E['price'] = E['price'].astype(float)

        # change column names with date suffix so that
        E = E.add_suffix('_' + date_panel_scraped)
        # fill none with nan, and remove rows that contain nan in ALL columns
        E = E.fillna(value=np.nan)
        E = E.dropna(axis=0, how="all")

        return (E)

    def transform_dict_dataframe(raw_data, date_panel_scraped):
        # the raw data is scraped by 1_functions_scraping_data.py function scraping_apps_according_to_id
        D = pd.DataFrame.from_dict(raw_data)
        E = D.T
        # E['updated_datetime'] = pd.to_datetime(E['updated'], origin='unix')
        E['released_datetime'] = pd.to_datetime(E['released']).dt.date
        E['today_datetime'] = datetime.date.today()
        E['days_since_released'] = (E['today_datetime'] - E['released_datetime']).dt.days
        E = pd.get_dummies(E, columns=['free', 'offersIAP',
                                       'adSupported', 'containsAds', 'contentRating'], dtype=int)
        E['GAME'] = np.where(E['genreId'].str.contains("GAME"), 1, 0)
        E['UTILITIES'] = np.where(np.isin(E['genreId'], ['MEDICAL', 'EDUCATION', 'HEALTH_AND_FITNESS',
                                                         'PARENTING', 'LIBRARIES_AND_DEMO', 'BOOKS_AND_REFERENCE',
                                                         'HOUSE_AND_HOME', 'FINANCE', 'WEATHER',
                                                         'TOOLS', 'BUSINESS', 'PRODUCTIVITY', 'TRAVEL_AND_LOCAL',
                                                         'MAPS_AND_NAVIGATION', 'AUTO_AND_VEHICLES']), 1, 0)
        E['SOCIAL_MEDIA_LEISURE'] = np.where(np.isin(E['genreId'], ['COMICS', 'BEAUTY', 'ART_AND_DESIGN', 'DATING',
                                                                    'VIDEO_PLAYERS', 'SPORTS', 'LIFESTYLE',
                                                                    'PERSONALIZATION', 'SHOPPING', 'FOOD_AND_DRINK',
                                                                    'MUSIC_AND_AUDIO',
                                                                    'NEWS_AND_MAGAZINES', 'PHOTOGRAPHY',
                                                                    'COMMUNICATION',
                                                                    'ENTERTAINMENT', 'SOCIAL', 'EVENTS']), 1, 0)
        E['check_all_genre_covered'] = E['GAME'] + E['UTILITIES'] + E['SOCIAL_MEDIA_LEISURE']
        # print("Non missing values in each column ", E.count())

        # drop columns that will not be needed
        E.drop(['descriptionHTML', 'summaryHTML', 'contentRatingDescription',
                'recentChanges'], axis=1, inplace=True)

        # first remove none ascii characters from the string then convert object to string type
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
        # change column names with date suffix so that
        E = E.add_suffix('_' + date_panel_scraped)
        # fill none with nan, and remove rows that contain nan in ALL columns
        E = E.fillna(value=np.nan)
        E = E.dropna(axis=0, how="all")
        # check
        print("after dropping rows all NA in all columns, the non-missing values in each column ", E.count())
        print(E.dtypes)
        return (E)

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

        new = DF1.join(panel_dfs[0], how='inner')
        for j in range(1, len(panel_dfs), 1):
            new = new.join(panel_dfs[j], how='inner')

        return (new)




