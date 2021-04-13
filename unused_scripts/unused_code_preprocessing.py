    def peek_at_outliers(self, var, method, quantiles, q_inter, **kwargs):
        # method determines which var you are using histogram over, if none, use it over the var itself
        # if average, or standard deviation, first calculate the average and std of the var over all panels, then draw the histogram on that average or std
        if method == 'std':
            df_with_stats = self.standard_deviation_of_var_panels(var=var)
        elif method == 'mean':
            df_with_stats = self.mean_of_var_panels(var=var)
        # -------------------------------------------------------------
        s_q = df_with_stats[[var + '_stats']].quantile(q=quantiles, axis=0, interpolation=q_inter)
        if 'ind' in kwargs.keys():
            ax = df_with_stats[var + '_stats'].plot.kde(ind=kwargs['ind'])
        else:
            ax = df_with_stats[var+'_stats'].plot.kde()
        return s_q, ax

    def define_outlier_appids(self, var, method, cutoff_q, q_inter): # first peek_at_outliers, then decide at which quantile to truncate the data
        if method == 'std':
            df_with_stats = self.standard_deviation_of_var_panels(var=var)
            s_q = df_with_stats[[var + '_stats']].quantile(q=cutoff_q, axis=0, interpolation=q_inter)
        # -------------------------------------------------------------
        cutoff_value = s_q.iat[0] # this is pandas series
        print('The cutoff value for', var, 'at', cutoff_q, '_th quantile is', cutoff_value)
        df_outliers = df_with_stats.loc[(df_with_stats[var+'_stats'] >= cutoff_value)]
        print('number of outliers are', len(df_outliers.index), 'out of', len(df_with_stats.index), 'total apps.')
        outlier_appids = df_outliers.index.tolist()
        return df_outliers, outlier_appids


    def cols_missing_ratio(self):
        num_of_cols_above_missing_threshold = 0
        missing_cols_and_missing_ratios = []
        missing_cols = []
        for col in self.df.columns:
            null_data = self.df[[col]][self.df[col].isnull()]
            r = len(null_data.index) / len(self.df.index)
            if r >= self.missing_ratio:
                num_of_cols_above_missing_threshold += 1
                missing_cols_and_missing_ratios.append((col, r))
                missing_cols.append(col)
        print('total number of columns contain missing value above', self.missing_ratio, 'is', num_of_cols_above_missing_threshold)
        print('out of total number of columns', len(self.df.columns))
        print(missing_cols_and_missing_ratios)
        return missing_cols_and_missing_ratios, missing_cols

    def rows_missing_ratio(self):
        df_t = self.df.T
        num_of_cols_above_missing_threshold = 0
        missing_cols_and_missing_ratios = []
        missing_cols = []
        for col in df_t.columns:
            null_data = df_t[[col]][df_t[col].isnull()]
            r = len(null_data.index) / len(df_t.index)
            if r >= self.missing_ratio:
                num_of_cols_above_missing_threshold += 1
                missing_cols_and_missing_ratios.append((col, r))
                missing_cols.append(col)
        print('total number of apps contain missing attributes above', self.missing_ratio, 'is',
              num_of_cols_above_missing_threshold)
        print('out of total number of apps', len(df_t.columns))
        print(missing_cols_and_missing_ratios)
        return missing_cols_and_missing_ratios, missing_cols

    def check_apps_with_consecutive_missing_panels(self, var, number_consec_panels_missing):
        df2 = self.select_the_var(var=var)
        null_data = df2[df2.isnull().any(axis=1)]
        null_data_t = null_data.T
        appids_with_consec_missing_panels = []
        for appid in null_data_t.columns:
            app_panels = null_data_t[[appid]]
            # https://stackoverflow.com/questions/29007830/identifying-consecutive-nans-with-pandas
            # https://numpy.org/doc/stable/reference/generated/numpy.cumsum.html
            # nan row (panel) will be 0 and none-missing rows (panel) wil be 1
            # and each row is the sum of all the rows preceding it
            # so when two rows have the same value means 1 missing row occured. Three rows have the same value, meaning two consecutive missing rows occured.
            cumlative_none_missing_df = app_panels.notnull().astype(int).cumsum()
            # https://stackoverflow.com/questions/35584085/how-to-count-duplicate-rows-in-pandas-dataframe
            missing_count = cumlative_none_missing_df.groupby(cumlative_none_missing_df.columns.tolist(), as_index=False).size()
            consec_missing_count = missing_count['size'].max()
            threshold = number_consec_panels_missing+1
            if consec_missing_count >= threshold: # == 2 means only 1 panel is missing, == 3 means 2 consecutive panels are missing, note greater or equal than the threshold
                appids_with_consec_missing_panels.append(appid)
        appids_intend_to_drop = null_data_t[appids_with_consec_missing_panels]
        print('number of apps with at least', number_consec_panels_missing, 'consecutive missing panels for', var, 'are', len(appids_with_consec_missing_panels))
        print('out of', len(df2.index), 'apps.')
        return appids_intend_to_drop, appids_with_consec_missing_panels

    def check_if_col_has_identical_value_except_for_missing(self, var):
        df2 = self.select_the_var(var=var)
        null_data = df2[df2.isnull().any(axis=1)]
        null_data_t = null_data.T
        appids_have_same_value_except_missing = []
        for j in null_data_t.columns:
            l1 = null_data_t[j]
            l2 = l1.dropna()
            l3 = list(set(l2.tolist()))
            if len(l3) == 1:
                appids_have_same_value_except_missing.append(j)
        dfr = null_data.loc[appids_have_same_value_except_missing]
        return dfr, appids_have_same_value_except_missing


    def convert_appid_to_developer_index(self, multiindex):
        time_invariant_df, time_invariant_appids = self.check_whether_var_varies_across_panels(var='developer')
        df2 = self.df.loc[time_invariant_appids]
        if multiindex is True:
            df2 = df2.reset_index().set_index(['developer_'+self.initial_panel, 'index'])
            df2.index.rename(['developer', 'appId'], inplace=True)
            # remove developers b/c we have only kept time-invariant developer information
            for j in df2.columns:
                if 'developer_' in j:
                    df2.drop(j, axis=1, inplace=True)
            # add number of apps variable to each row
            df3 = df2.reset_index().groupby('developer')['appId'].nunique().rename('num_apps_owned').to_frame()

            df2 = df2.reset_index().merge(df3, on='developer', how='left')
            df2.set_index(['developer', 'appId'], inplace=True)
            self.df_mi = df2
            return df2
        elif multiindex is False:
            dev_level_vars = ['developer', 'developerId', 'developerEmail', 'developerWebsite', 'developerAddress']
            cols = []
            for v in dev_level_vars:
                new_df = self.select_the_var(var=v)
                cols.extend(new_df.columns.tolist())
            df2 = df2[cols]
            df2 = df2.reset_index(drop=True).set_index('developer_'+self.initial_panel)
            df2.index.rename('developer', inplace=True)
            # drop duplicate index rows
            index = df2.index
            is_duplicate = index.duplicated(keep="first")
            not_duplicate = ~is_duplicate
            df2 = df2[not_duplicate]
            # remove developers b/c we have only kept time-invariant developer information
            for j in df2.columns:
                if 'developer_' in j:
                    df2.drop(j, axis=1, inplace=True)
            self.df_di = df2
            return pre_processing(df=self.df,
                                  initial_panel=self.initial_panel,
                                  all_panels=self.all_panels,
                                  df_developer_index_geocoded=self.df_dig,
                                  df_developer_index=self.df_di,
                                  df_multiindex=self.df_mi,
                                  appids_to_remove=self.appids_to_remove)

    def format_text_for_developer(self, text):
        if text is not None:
            result_text = ''.join(c.lower() for c in text if not c.isspace())  # remove spaces
            punc = '''!()-[]{};:'"\, <>./?@#$%^&*_~+'''  # remove functuations
            for ele in result_text:
                if ele in punc:
                    result_text = result_text.replace(ele, "")
            extra1 = re.compile(
                r'(corporated$)|(corporation$)|(corp$)|(company$)|(limited$)|(games$)|(game$)|(studios$)|(studio$)|(mobile$)')
            extra2 = re.compile(
                r'(technologies$)|(technology$)|(tech$)|(solutions$)|(solution$)|(com$)|(llc$)|(inc$)|(ltd$)|(apps$)|(app$)|(org$)|(gmbh$)')
            res1 = re.sub(extra1, '', result_text)
            res2 = re.sub(extra2, '', res1)
        else:
            res2 = np.nan
        return res2