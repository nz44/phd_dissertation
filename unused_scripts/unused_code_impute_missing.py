    def impute_missing(self, var, method):
        df2 = self.select_the_var(var=var)
        df_list = []
        for j in range(len(df2.columns)):
            if j <= adj_panels // 2 or j in [0, 1]:
                df = df2.iloc[:, 0:adj_panels + 1]
            elif j >= len(df2.columns) - adj_panels // 2 - 1:
                df = df2.iloc[:, len(df2.columns) - adj_panels - 1:len(df2.columns)]
            else:
                if adj_panels == 1:
                    df = df2.iloc[:, j - 1:j + 1]
                else:
                    df = df2.iloc[:, j - adj_panels // 2:j + adj_panels // 2 + 1]
            if method == 'mean':
                df[method] = df.mean(axis=1, skipna=True)
            elif method in ['mode', 'mode if none-missing are all the same']:
                df['mode'] = df.mode(axis=1, numeric_only=False, dropna=True).iloc[:, 0]
            elif method == 'previous':
                df[method] = df.iloc[:, 0]
            elif method == 'previous_mean':
                
            else:
                df[method] = 0
            dfd = copy.deepcopy(df)
            for col in dfd.columns:
                if method == 'mode if none-missing are all the same':
                    dfr, appids_have_same_value_except_missing = self.check_if_col_has_identical_value_except_for_missing(
                        var=var)
                    dfd.loc[appids_have_same_value_except_missing, col] = dfd['mode']
                else:
                    dfd.loc[dfd[col].isnull(), col] = dfd[method]
            dfd = dfd[[df2.columns[j]]]
            df_list.append(dfd)
        imputed_df = functools.reduce(lambda a, b: a.join(b, how='inner'), df_list)
        col_names = imputed_df.columns.tolist()
        self.df.drop(columns=col_names, inplace=True)
        self.df = self.df.join(imputed_df, how='inner')
        return pre_processing(df=self.df,
                              initial_panel=self.initial_panel,
                              all_panels=self.all_panels,
                              df_developer_index_geocoded=self.df_dig,
                              df_developer_index=self.df_di,
                              df_multiindex=self.df_mi)