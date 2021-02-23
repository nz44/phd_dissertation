# DUPLICATES THEME
###############################################################################################################################
def unique_duplicate_appids(self, input_df):
    dup_df_according_to_each_col = []
    for i in input_df.columns:
        dup_df = input_df[input_df.duplicated(subset=i)]
        dup_df_according_to_each_col.append(dup_df)
    mm = list(map(lambda x: x.index.tolist(), dup_df_according_to_each_col))
    flat_list = [item for sublist in mm for item in sublist]
    unique_dup_appids = list(set(flat_list))
    return unique_dup_appids

def var_with_multiple_duplicate_values(self, var, consecutive=False, select_one_panel=None):
    col_list, panel_list = self.select_the_var(var=var, consecutive=consecutive, select_one_panel=select_one_panel)
    df2 = self.keep_cols(list_of_col_names=col_list)
    unique_dup_appids = self.unique_duplicate_appids(input_df=df2)
    print('in total, there are', len(unique_dup_appids), 'appids have duplicate developer info')
    print('before removing, we have', df2.shape)
    the_unique_df = df2.drop(index=unique_dup_appids)
    print('after removing, we have', the_unique_df.shape)
    # self.df: A,A,A, false, true, true ---> dup_df_1 = A,A; false, true ---> dup_df_2 = A ---> len(third dup_df.index == 0)
    dup_df_1 = df2.loc[unique_dup_appids]
    duplicate_dfs = [dup_df_1]
    while len(dup_df_1.index) > 0:
        unique_dup_appids = self.unique_duplicate_appids(input_df=dup_df_1)
        dup_df_2 = dup_df_1.loc[unique_dup_appids]
        duplicate_dfs.append(dup_df_2)
        dup_df_1= dup_df_2
    return the_unique_df, duplicate_dfs

def check_duplicate_indices(self, df_type): # df_type could be appid, or it could be developer, or dev_multi
    if df_type == 'appid':
        dup_index_list = self.df.index[self.df.index.duplicated()].tolist()
    return dup_index_list

def appids_have_time_variant_var(self, var, consecutive=False, format_text=False):
    df_list, diff_dfs = self.check_whether_var_is_time_invariant(var=var, consecutive=consecutive, format_text=format_text)
    combined_appids = []
    for df in diff_dfs:
        if len(df.index) != 0:
            appid_list = df.index.tolist()
            combined_appids.extend(appid_list)
    unique_index = list(set(combined_appids))
    return diff_dfs, unique_index

def format_text_for_developer(self, text):
    if text is not None:
        result_text = ''.join(c.lower() for c in text if not c.isspace()) # remove spaces
        punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~+''' # remove functuations
        for ele in result_text:
            if ele in punc:
                result_text = result_text.replace(ele, "")
        extra1 = re.compile(r'(corporated$)|(corporation$)|(corp$)|(company$)|(limited$)|(games$)|(game$)|(studios$)|(studio$)|(mobile$)')
        extra2 = re.compile(r'(technologies$)|(technology$)|(tech$)|(solutions$)|(solution$)|(com$)|(llc$)|(inc$)|(ltd$)|(apps$)|(app$)|(org$)|(gmbh$)')
        res1 = re.sub(extra1, '', result_text)
        res2 = re.sub(extra2, '', res1)
    else:
        res2 = np.nan
    return res2

def check_whether_var_is_time_invariant(self, var, consecutive=False, format_text=False): # note the var cannot be appId
    the_unique_df, dup_dfs_list = self.var_with_multiple_duplicate_values(var=var, consecutive=consecutive)
    df_list = []
    for j in the_unique_df.columns:
        new_df = the_unique_df[[j]]
        new_df.rename(columns={j:var}, inplace=True)
        if format_text is True:
            new_df[j]=new_df[j].apply(lambda x: self.format_text_for_developer(x))
        df_list.append(new_df)
    diff_dfs = []
    for i in range(len(df_list)-1):
        diff_df = self.dataframe_difference(df_list[i], df_list[i+1], var=var)
        if len(diff_df.index) != 0:
            print(var, 'is NOT time invariant')
            diff_dfs.extend([diff_df])
        else:
            print(var, 'is time invariant')
    return df_list, diff_dfs

def dataframe_difference(self, df1, df2, var): # ALL YOU need is left only, because you are compare df1 to df2, df2 to df3...
    """Find rows which are different between two DataFrames.
    https://hackersandslackers.com/compare-rows-pandas-dataframes/"""
    comparison_df = df1.reset_index().merge(
        df2,
        on = var,
        indicator = True,
        how = 'left'
    ).set_index('index')
    diff_df = comparison_df[comparison_df['_merge'] == 'left_only']
    return diff_df



def impute_missing_using_adjacent_panel(self, var): # the self.df here should be the newly passed df that has deleted all rows and cols that will not be imputed
    col, panels = self.select_the_var(var=var)
    df2 = self.keep_cols(list_of_col_names=col)
    for i in range(len(df2.columns)):
        if i == 0: # the first panel is missing, impute with the next panel
            df2[df2.columns[i]] = df2.apply(
                lambda row: row[df2.columns[i+1]] if np.isnan(row[df2.columns[i]]) else row[df2.columns[i]],
                axis=1
            )
        else: # all other panels impute with previous panels
            df2[df2.columns[i]] = df2.apply(
                lambda row: row[df2.columns[i-1]] if np.isnan(row[df2.columns[i]]) else row[df2.columns[i]],
                axis=1
            )
    return df2


def appids_that_have_missing_in_any_panels(self, var, consecutive=False):
    col_list, panel_list = self.select_the_var(var=var, consecutive=consecutive)
    df2 = self.keep_cols(list_of_col_names=col_list)
    data = df2[df2.isnull().any(axis=1)]
    appids = data.index.tolist()
    return appids, data