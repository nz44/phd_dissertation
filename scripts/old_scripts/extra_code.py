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

def compare_several_panel_reg_results(self, results_dict, dep_vars):
    """
    https://bashtage.github.io/linearmodels/panel/examples/examples.html
    results_dict is the output of several_regressions
    """
    panels_results = dict.fromkeys(dep_vars, {})
    panels_compare = dict.fromkeys(dep_vars)
    for reg_type, results in results_dict.items():
        if reg_type in ['FE', 'RE', 'POOLED_OLS', 'PPOLED_OLS_with_individual_dummies']:
            for dep_var in results.keys():
                panels_results[dep_var][reg_type] = results[dep_var]
                compared_results = compare(panels_results[dep_var])
                print(compared_results)
                panels_compare[dep_var] = compared_results
    return panels_compare

def compile_several_reg_results_into_pandas(self, panels_compare):
    """
    panel_compare is the output of self.compare_several_panel_reg_results(self, results_dict)
    """
    def change_index_name(df, end_fix):
        dfn = df.copy(deep=True)
        for i in dfn.index:
            dfn.rename(index={i: i+'_'+end_fix}, inplace=True)
        return dfn
    df5 = []
    for dep_var, compare in panels_compare.items():
        df_params = change_index_name(compare.params, end_fix='coef')
        df_tstats = change_index_name(compare.tstats, end_fix='tstats')
        df_pvalues = change_index_name(compare.pvalues, end_fix='pvalues')
        df1 = [df_params, df_tstats, df_pvalues]
        df2 = functools.reduce(lambda a, b: pd.concat([a, b], axis=0), df1)
        df3 = [df2,
                compare.f_statistic.T,
                compare.rsquared.to_frame().T,
                compare.nobs.to_frame().T,
                compare.cov_estimator.to_frame().T]
        df4 = functools.reduce(lambda a, b: pd.concat([a, b], axis=0), df3)
        for i in df4.columns:
            df4.rename(columns={i: dep_var+'_'+i}, inplace=True)
        df5.append(df4)
    df6 = functools.reduce(lambda a, b: a.join(b, how='inner'), df5)
    # round the entire dataframe
    df7 = df6.T
    for i in df7.columns:
        if i not in ['_cov_type', 'nobs']:
            df7[i] = df7[i].astype(float).round(decimals=3)
        elif i == 'nobs':
            df7[i] = df7[i].astype(int)
    df8 = df7.T
    return df8
