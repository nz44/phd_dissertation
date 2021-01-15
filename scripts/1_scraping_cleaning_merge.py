from google_play_scraper import app
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
import math
from collections.abc import Iterable


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
                        'recentChangesHTML', 'adSupported_False', 'containsAds_False', 'recentChanges',
                        'category', 'iap_range', 'app_id', 'androidVersionText', 'androidVersion',
                        'current_version', 'version', 'required_android_version', 'sale', 'saleTime', 'originalPrice',
                        'saleText']


    def convert_keys_to_datetime(self):
        datetime_list = []
        for i in self.d.keys():
            date_time_obj = dt.strptime(i, '%Y%m')
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


    def merge_panels_into_single_df(self):
        old_data = self.format_cols() # this is a dictionary with panel as keys
        df_list = []
        for panel, df in old_data.items():
            df.drop([x for x in df.columns if x in self.cols_to_drop], axis=1, inplace=True)
            df = df.add_suffix('_' + panel)
            df_list.append(df)
        merged_df = functools.reduce(lambda x, y: x.join(y, how='inner'), df_list)
        return merged_df


    def check_data_type(self):
        data = self.combine_similar_and_drop_extra_cols()
        for panel, df in data.items():
            print(panel)
            print(df.dtypes)
            print()


    def check_unique_value_in_cols(self, col_name):
        data = self.format_cols()
        for panel, df in data.items():
            print(panel, col_name)
            print('Unique Values:')
            print(df[col_name].unique())
            print()
            print('Frequency Table:')
            print(df[col_name].value_counts(dropna=False))
            print()
            print()


    def format_cols(self):
        old_data = self.format_missing_values()
        new_data = dict.fromkeys(self.d.keys())
        numeric_cols = ['minInstalls', 'score', 'ratings', 'reviews', 'size', 'price']
        datetime_cols = ['released', 'updated']
        dummy_cols = ['adSupported', 'containsAds', 'free', 'offersIAP', 'contentRating', 'genreId']
        # ---------------------------------------------------------------------------------------------
        def convert_string_to_datetime(x):
            for fmt in ('%b %d, %Y', '%d-%b-%y', '%B %d, %Y'):
                try:
                    return dt.strptime(x, fmt).date()
                except:
                    pass

        def convert_unix_to_datetime(x):
            if isinstance(x, dt) or isinstance(x, date):
                pass
                return x
            elif x is None:
                pass
                return np.datetime64("NaT")
            else:
                date_time_obj = dt.fromtimestamp(x)
                return date_time_obj.date()

        def remove_characters_from_numeric_cols(x):
            num = re.sub(r'\D+', "", x)
            return num

        def unlist_a_col_containing_list_of_strings(x):
            if x is not None and re.search(r'\[\]+', x):
                s = eval(x)
                if isinstance(s, list):
                    s2 = ', '.join(str(ele) for ele in s)
                    return s2
            else:
                return x

        everyone_pattern = re.compile(r'(Everyone+)|(10\++)')
        teen_pattern = re.compile(r'(Teen+)|(Mature+)')
        adult_pattern = re.compile(r'(Adults+)|(18\++)')
        def combine_contentRating_into_3_groups(x):
            if x is not None:
                if re.search(everyone_pattern, x):
                    return 'Everyone'
                elif re.search(teen_pattern, x):
                    return 'Teen'
                elif re.search(adult_pattern, x):
                    return 'Adult'
            else:
                return x
        # ---------------------------------------------------------------------------------------------
        for panel, df in old_data.items():
            df['size'] = df['size'].apply(lambda x: remove_characters_from_numeric_cols(x) if isinstance(x, str) else x)
            df['price'] = df['price'].apply(lambda x: remove_characters_from_numeric_cols(x) if isinstance(x, str) else x)
            df['released'] = df['released'].apply(lambda x: convert_string_to_datetime(x) if isinstance(x, str) else convert_unix_to_datetime(x))
            df['updated'] = df['updated'].apply(lambda x: convert_string_to_datetime(x) if isinstance(x, str) else convert_unix_to_datetime(x))
            df['contentRating'] = df['contentRating'].apply(unlist_a_col_containing_list_of_strings)
            df['contentRating'] = df['contentRating'].apply(combine_contentRating_into_3_groups)
            # since AdSupported only contains True and None, so fill None with False
            df['adSupported'].fillna(False, inplace=True)
            # df_temp = pd.get_dummies(df[dummy_cols], dummy_na=True)
            # df = df.join(df_temp, how='inner')
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric)
            new_data[panel] = df
        return new_data


    def format_missing_values(self):
        old_data = self.combine_similar_and_drop_extra_cols()
        new_data = dict.fromkeys(self.d.keys())
        numeric_cols = ['minInstalls', 'score', 'ratings', 'reviews', 'size', 'price', 'inAppProductPrice'] # use np.nan
        none_numeric_cols = ['adSupported', 'containsAds', 'free', 'offersIAP', 'contentRating', 'genreId',
                             'released', 'updated', 'summary', 'description', 'title', 'comments',
                             'developerId', 'developerEmail', 'developerWebsite', 'developerAddress'] # use None
        for panel, df in old_data.items():
            # where(condition, df2) if True, leave as they are (df1), if False, use df2
            # use list comprehension because dataframe before 202009 does not have inAppProductPrice column
            df[[x for x in df.columns if x in numeric_cols]] = df[[x for x in df.columns if x in numeric_cols]].where(df.notnull(), np.nan) # replace None with nan
            df[none_numeric_cols] = df[none_numeric_cols].replace(r'^\s*$', np.nan, regex=True) # fill whitespaces with np.nan
            df[none_numeric_cols] = df[none_numeric_cols].where(df.notnull(), None) # replace nan with None
            new_data[panel] = df
        return new_data


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
        old_data = self.convert_from_dict_to_df()
        new_data = dict.fromkeys(self.d.keys())
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
            new_data[i] = df
        return new_data


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
        x = dt(2020, 9, 1)
        for i in self.d.keys():
            date_time_obj = dt.strptime(i, '%Y%m')
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

