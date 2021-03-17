# extra code
else:
    con_vars = [i + '_' + p for p in self.consec_panels for i in continuous_vars]
    dum_vars = [i + '_' + p for p in self.consec_panels for i in dummy_vars]
    cat_vars_all = [i + '_' + p for p in self.consec_panels for i in cat_vars]


else:
    cat_stats_dfs = []
    for i in cat_vars:
        cat_dfs = []
        for j in self.consec_panels:
            cat_vars_df['Count'+i+'_'+j] = 0
            df = cat_vars_df[[i+'_'+j, 'Count'+i+'_'+j]].groupby(i+'_'+j).count()
            if 'minInstalls' in i:
                df.sort_index(inplace=True)
            else:
                df.sort_values(by='Count'+i+'_'+j, ascending=False, inplace=True)
            cat_dfs.append(df)
        cat_df_i = functools.reduce(lambda a, b: a.join(b, how='inner'), cat_dfs)
        sum_row = cat_df_i.sum(axis=0)
        sum_row = sum_row.to_frame().T
        sum_row.index = ['sum']
        cat_df_i = pd.concat([cat_df_i, sum_row], join="inner")
        cat_stats_dfs.append(cat_df_i)