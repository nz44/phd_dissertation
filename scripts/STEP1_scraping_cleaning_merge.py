from google_play_scraper import app
import pandas as pd
from pathlib import Path
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
import os
import math
import pickle
from collections.abc import Iterable

#############################################################################################################################
# get ID list
# data scraped before 2020 Sep are organized in a list, each dictionary inside the list contains attributes and their values
# for the data scraped in 2020 Sep and onwards, they are organized in dictionary with key as appid, and then their appdetails,
# so id_list should just be C.keys()
class scrape():
    initial_dict_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/NEW_ALGORITHM_MONTHLY_SCRAPE')
    tracking_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/TRACKING_THE_SAME_ID_MONTHLY_SCRAPE')

    def __init__(self,
                 initial_panel,
                 current_panel,
                 initial_panel_data=None,
                 id_list=None,
                 scraped_dict=None):
        self.initial_panel = initial_panel
        self.current_panel = current_panel
        self.initial_panel_data = initial_panel_data
        self.id_list = id_list,
        self.scraped_dict = scraped_dict

    def create_dir(self):
        ispath = os.path.join(scrape.tracking_path, self.current_panel)
        os.makedirs(ispath, exist_ok=True)
        return scrape(initial_panel=self.initial_panel,
                      current_panel=self.current_panel,
                      initial_panel_data=self.initial_panel_data,
                      id_list=self.id_list,
                      scraped_dict=self.scraped_dict)

    def open_initial_panel_data(self):
        filename = "ALL_APP_DETAILS_" + self.initial_panel + '.pickle'
        q = scrape.initial_dict_path / self.initial_panel / filename
        self.initial_panel_data = pd.read_pickle(q)
        return scrape(initial_panel=self.initial_panel,
                      current_panel=self.current_panel,
                      initial_panel_data=self.initial_panel_data,
                      id_list=self.id_list,
                      scraped_dict=self.scraped_dict)

    def get_appids_from_initial_panel_data(self):
        self.id_list = []
        for i in self.initial_panel_data:
            if 'appId' in i.keys():
                self.id_list.append(i['appId'])
            else:
                self.id_list.append(i['app_id'])
        print(self.initial_panel, ' first panel contains ', len(self.id_list), ' IDs.')
        return scrape(initial_panel=self.initial_panel,
                      current_panel=self.current_panel,
                      initial_panel_data=self.initial_panel_data,
                      id_list=self.id_list,
                      scraped_dict=self.scraped_dict)

    def scraping_apps_according_to_id(self):
        """
        scraping using google_play_scraper app function
        for unknown reason,, self.id_list in weird tuple, so first turn it into list and flatten it
        """
        filename = 'TRACKING_' + self.initial_panel + '.pickle'
        q = scrape.tracking_path / self.current_panel / filename
        isfile = os.path.isfile(q)
        if isfile is True:
            print(self.current_panel, ' tracking ', self.initial_panel, ' has already been scraped.')
        else:
            d2list = list(self.id_list)
            appids = [i for sublist in d2list for i in sublist]
            app_details = dict.fromkeys(appids)
            print('start scraping apps with initial panel', self.initial_panel)
            for j in tqdm(range(len(appids)), desc="scraping..."):
                try:
                    app_details[appids[j]] = app(appids[j])
                except:
                    pass
            # ---------------------- save --------------------------------------
            self.scraped_dict = app_details
            pickle.dump(self.scraped_dict, open(q, 'wb'))
            print('Saved scarped app details with initial panel', self.initial_panel)
        return scrape(initial_panel=self.initial_panel,
                      current_panel=self.current_panel,
                      initial_panel_data=self.initial_panel_data,
                      id_list=self.id_list,
                      scraped_dict=self.scraped_dict)

#############################################################################################################################
#############################################################################################################################

class convert():

    initial_dict_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/NEW_ALGORITHM_MONTHLY_SCRAPE')
    tracking_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/TRACKING_THE_SAME_ID_MONTHLY_SCRAPE')
    imputed_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/___essay_1___')
    df1_to_combine = ['developerId',
                      'developerWebsite',
                      'developerEmail',
                      'developerAddress',
                      'offersIAP',
                      'appId',
                      'contentRating']
    df2_to_combine = ['developer_id',
                      'developer_url',
                      'developer_email',
                      'developer_address',
                      'iap',
                      'app_id',
                      'content_rating']
    all_cols_to_combine = df1_to_combine + df2_to_combine
    cols_to_rename = {'category': 'genreId',
                      'iap': 'offersIAP',
                      'developer_id': 'developerId',
                      'content_rating': 'contentRating',
                      'iap_range': 'inAppProductPrice',
                      'developer_email': 'developerEmail',
                      'developer_url': 'developerWebsite',
                      'developer_address': 'developerAddress',
                      'app_id': 'appId'}
    cols_to_drop = ['editors_choice',
                    'interactive_elements',
                    'description_html',
                    'descriptionHTML',
                    'privacyPolicy',
                    'recentChanges',
                    'recent_changes',
                    'summaryHTML',
                    'androidVersionText',
                    'icon',
                    'genre',
                    'headerImage',
                    'screenshots',
                    'video',
                    'videoImage',
                    'developerInternalID',
                    'content_rating',
                    'contentRatingDescription',
                    'installs',
                    'url',
                    'recentChangesHTML',
                    'recentChanges',
                    'category',
                    'iap_range',
                    'app_id',
                    'androidVersionText',
                    'androidVersion',
                    'current_version',
                    'version',
                    'required_android_version',
                    'sale',
                    'saleTime',
                    'originalPrice',
                    'saleText']

    def __init__(self,
                 initial_panel,
                 all_panels,
                 d=None,
                 merged_df=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.d = d
        self.merged_df = merged_df

    def open_app_detail_dict(self):
        dfs = dict.fromkeys(self.all_panels)
        for i in range(len(self.all_panels)):
            if i == 0:  # open the initial panel
                filename = 'ALL_APP_DETAILS_' + self.initial_panel + '.pickle'
                q = convert.initial_dict_path / self.initial_panel / filename
                with open(q, 'rb') as f:
                    dfs[self.initial_panel] = pickle.load(f)
            else:
                filename = 'TRACKING_' + self.initial_panel + '.pickle'
                q = convert.tracking_path / self.all_panels[i] / filename
                with open(q, 'rb') as f:
                    dfs[self.all_panels[i]] = pickle.load(f)
        self.d = dfs
        return convert(initial_panel=self.initial_panel,
                       all_panels=self.all_panels,
                       d=self.d,
                       merged_df=self.merged_df)

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

    def merge_panels_into_single_df(self, balanced_panel):
        old_data = self.format_cols() # this is a dictionary with panel as keys
        # ------------ singling out the initial month -------------------------
        all_months = []
        for panel, df in old_data.items():
            panel_dt = dt.strptime(panel, "%Y%m")
            all_months.append(panel_dt)
        init_month = min(all_months)
        init_month_str = dt.strftime(init_month, "%Y%m")
        # ---------------------------------------------------------------------
        print('start merging the dataframe for panel starting ', init_month_str)
        for panel, df in old_data.items():
            df = df.add_suffix('_' + panel)
            if panel == init_month_str:
                merged_df = df
            else:
                indicator_col = 'merge_' + panel
                if balanced_panel is True:
                    # inner join will delete all the apps that disappear in the subsequent months
                    merged_df = merged_df.merge(df, how='inner', left_index=True, right_index=True, indicator=indicator_col)
                    filename = self.initial_panel + '_balanced_MERGED.pickle'
                else:
                    # in essay three we will use a noisy death measure (apps that disappear over time)
                    merged_df = merged_df.merge(df, how='left', left_index=True, right_index=True, indicator=indicator_col)
                    filename = self.initial_panel + '_unbalanced_MERGED.pickle'
        print('merged dataframe for all the panels above')
        self.merged_df = merged_df
        # --------------------------- save --------------------------------------
        q = convert.imputed_path / filename
        pickle.dump(self.merged_df, open(q, 'wb'))
        print('panel data ', self.initial_panel, ' has shape : ', self.merged_df.shape)
        print(self.merged_df.columns)
        return convert(initial_panel=self.initial_panel,
                       all_panels=self.all_panels,
                       d=self.d,
                       merged_df=self.merged_df)

    def check_for_duplicate_cols(self):
        df_dict = self.format_cols()
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

    def format_cols(self):
        old_data = self.format_missing_values()
        new_data = dict.fromkeys(self.d.keys())
        numeric_cols = ['minInstalls', 'score', 'ratings', 'reviews', 'size', 'price']
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

        # ---------------------------------------------------------------------------------------------
        for panel, df in old_data.items():
            df['size'] = df['size'].apply(lambda x: remove_characters_from_numeric_cols(x) if isinstance(x, str) else x)
            df['price'] = df['price'].apply(lambda x: remove_characters_from_numeric_cols(x) if isinstance(x, str) else x)
            df['released'] = df['released'].apply(lambda x: convert_string_to_datetime(x) if isinstance(x, str) else convert_unix_to_datetime(x))
            df['updated'] = df['updated'].apply(lambda x: convert_string_to_datetime(x) if isinstance(x, str) else convert_unix_to_datetime(x))
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


    def combine_similar_and_drop_extra_cols(self):
        old_data = self.convert_from_dict_to_df()
        new_data = dict.fromkeys(self.d.keys())
        for i, df in old_data.items():
            ## __________________________________________ Combine Similar Cols ___________________________________________________
            # first check whether the dataframe contains BOTH old and new cols, yes then combine, no keep the same
            if all(x in df.columns for x in convert.all_cols_to_combine):
                # combine duplicate columns, first take them out as separate dataframes
                combine_df1 = df[convert.df1_to_combine].fillna('').astype(str)
                combine_df2 = df[convert.df2_to_combine].fillna('').astype(str)
                df.drop(convert.all_cols_to_combine, axis=1, inplace=True)

                # second, rename the columns in the second dataframe to the same as the first dataframe
                for old_col_name, new_col_name in convert.cols_to_rename.items():
                    if old_col_name in combine_df2.columns:
                        combine_df2.rename(columns={old_col_name: new_col_name}, inplace=True)

                # third, combine the dataframe with overwrite == False so that if one col exist in one but not another, they get kept the same
                df3 = combine_df1.combine(combine_df2, operator.add, overwrite=False)
                df = df.join(df3, how='inner')

            ## __________________________________________ Drop Extra Cols and Add Panel Suffix ____________________________________
            # drop columns that will not be needed
            # after dropping columns, the only useful col that exist in and after 202009 is inAppProductPrice, which does not existed before
            df.drop([x for x in df.columns if x in convert.cols_to_drop], axis=1, inplace=True)
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

