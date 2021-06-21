###########################################################################################################
# Generate Descriptive Stats Latex Tables
###########################################################################################################
var_latex_map = {
    'const': 'Constant',
    'score': 'Rating',
    'DeMeanedscore': 'Demeaned Rating',
    'reviews': 'Reviews',
    'ZSCOREreviews': 'Z Score Reviews',
    'DeMeanedZSCOREreviews': 'Demeaned Z Score Reviews',
    'minInstallsTop': '\makecell[l]{High Level \\\ Minimum Installs}',
    'DeMeanedminInstallsTop': '\makecell[l]{Demeaned High Level \\\ Minimum Installs}',
    'minInstallsMiddle': '\makecell[l]{Medium Level \\\ Minimum Installs}',
    'DeMeanedminInstallsMiddle': '\makecell[l]{Demeaned Medium Level \\\ Minimum Installs}',
    'minInstallsBottom': '\makecell[l]{Low Level \\\ Minimum Installs}',
    'DeMeanedminInstallsBottom': '\makecell[l]{Demeaned Low Level \\\ Minimum Installs}',
    'niche_app': 'Niche',
    'genreIdGame': 'Hedonic',
    'contentRatingAdult': 'Age Restrictive',
    'DaysSinceReleased': 'Released',
    'paidTrue': 'Paid',
    'offersIAPTrue': 'Offers IAP',
    'containsAdsTrue': 'Contains ads',
    'price': 'Price',
    'F stat': 'F statistic',
    'P-value': 'P Value',
    'rsquared': 'R Squared',
    'nobs': '\makecell[l]{number of \\\ observations}',
    '_cov_type': 'Covariance Type'}

var_definition = {
    'Rating_{i,t}': '\makecell[l]{Weighted average (from 1 to 5) of cumulative consumer ratings of app $i$}',
    'Demeaned Rating_{i,t}': '\makecell[l]{Time demean $Rating_{i,t}$ by subtracting \\\ the mean of 7 consecutive monthly periods}',
    'Reviews_{i,t}': '\makecell[l]{Number of cumulative consumer reviews \\\ for the app $i$ between its release and period $t$}',
    'Z Score Reviews_{i,t}': 'Normalize number of reviews for App $i$ in period $t$ using Z-Score',
    'Demeaned Z Score Reviews_{i,t}': '\makecell[l]{Time demeaned z-score reviews \\\ by subtracting the mean from 7 consecutive periods}',
    '\makecell[l]{High Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the minimum cumulative installs of the app $i$ in \\\ period $t$ is above 10,000,000, otherwise 0.}',
    '\makecell[l]{Demeaned High Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Time demean High Level Minimum $Installs_{i,t}$ \\\ by subtracting the mean from 7 consecutive periods}',
    '\makecell[l]{Medium Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the minimum cumulative installs of the app $i$ \\\ in period $t$ is between 10,000 and 10,000,000, otherwise 0.}',
    '\makecell[l]{Demeaned Medium Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Time demean Medium Level Minimum $Installs_{i,t}$ \\\ by subtracting the mean from 7 consecutive periods}',
    '\makecell[l]{Low Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the minimum cumulative installs of the app $i$ \\\ in period $t$ is below 10,000, otherwise 0.}',
    '\makecell[l]{Demeaned Low Level \\\ Minimum Installs_{i,t}}': '\makecell[l]{Time demean Low Level Minimum $Installs_{i,t}$ \\\ by subtracting the mean from 7 consecutive periods}',
    'Niche_{i}': '\makecell[l]{Time invariant dummy variable which \\\ equals to 1 if App $i$ is niche, otherwise 0}',
    'Hedonic_{i}': '\makecell[l]{Time invariant dummy variable which \\\ equals to 1 if App $i$ is in the category GAME, otherwise 0}',
    'Age Restrictive_{i}': '\makecell[l]{Time invariant dummy variable which \\\ equals to 1 if App $i$ contains mature (17+) \\\ or adult (18+) content, otherwise 0}',
    'Released_{i}': '\makecell[l]{The number of days \\\ since App $i$ was released}',
    'Paid_{i,t}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the App $i$ is paid in period $t$}',
    'Offers IAP_{i,t}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the App $i$ offers within app purchases (IAP)}',
    'Contains ads_{i,t}': '\makecell[l]{Dummy variable, which equals to 1 if \\\ the App $i$ contains advertisement in period $t$}',
    'Price_{i,t}': 'Price of App $i$ in period $t$'}

descriptive_stats_table_row_order = {
    'niche_app': 0,
    'price': 1,
    'paidTrue': 2,
    'offersIAPTrue': 3,
    'containsAdsTrue': 4,
    'genreIdGame': 5,
    'contentRatingAdult': 6,
    'DaysSinceReleased': 7,
    'minInstallsTop': 8,
    'DeMeanedminInstallsTop': 9,
    'minInstallsMiddle': 10,
    'DeMeanedminInstallsMiddle': 11,
    'minInstallsBottom': 12,
    'DeMeanedminInstallsBottom': 13,
    'score': 14,
    'DeMeanedscore': 15,
    'reviews': 16,
    'ZSCOREreviews': 17,
    'DeMeanedZSCOREreviews': 18,
}

descriptive_stats_table_column_map = {
    'mean': 'Mean',
    'median': 'Median',
    'std': '\makecell[l]{Standard \\\ Deviation}',
    'min': 'Min',
    'max': 'Max',
    'count': '\makecell[l]{Total \\\ Observations}',
    '0_Count': '\makecell[l]{False \\\ Observations}',
    '1_Count': '\makecell[l]{True \\\ Observations}',
}

time_variant_vars = ['score', 'reviews', ]
    def correlation_matrix(self, time_variant_vars, time_invariant_vars, the_panel):
        """
        This is for the purpose of checking multicollinearity between independent variables
        """
        df = self.select_vars(time_variant_vars_list=time_variant_vars,
                              time_invariant_vars_list=time_invariant_vars,
                              the_panel=the_panel)
        df_corr = df.corr()
        print(df_corr)
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def bar_chart_a_dummy_against_dummy_or_cat(self, df, dummy1, dummy2):
        fig, ax = plt.subplots()
        df2 = df.groupby([dummy1, dummy2]).size()
        ax = df2.unstack().plot.bar(stacked=True, ax=ax)
        total_1 = 0
        total_2 = 0
        for p in ax.patches:
            if p.xy[0] == -0.25:
                total_1 += p.get_height()
            elif p.xy[0] == 0.75:
                total_2 += p.get_height()
        for p in ax.patches:
            if p.xy[0] == -0.25:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total_1)
            elif p.xy[0] == 0.75:
                percentage = '{:.1f}%'.format(100 * p.get_height() / total_2)
            x = p.get_x() + p.get_width() + 0.02
            y = p.get_y() + p.get_height() / 2
            ax.annotate(percentage, (x, y))
        filename = self.initial_panel + '_' + dummy1 + '_' + dummy2 + '.png'
        fig.savefig(reg_preparation.descriptive_stats_graphs / filename,
                    facecolor='white',
                    dpi=300)
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def kde_plot_by_dummy(self, df, dummy1, continuous1):
        fig, ax = plt.subplots()
        ax = sns.kdeplot(data=df, x=continuous1, hue=dummy1,
                         fill=True, common_norm=False,
                         # palette="crest", remove palette because the color contrast is too low
                         alpha=.4, linewidth=0, ax=ax)
        ax.set_title(self.initial_panel + ' Dataset' + ' ' + dummy1 + ' against ' + continuous1)
        filename = self.initial_panel + '_' + dummy1 + '_' + continuous1 + '.png'
        fig.savefig(reg_preparation.descriptive_stats_graphs / filename,
                    facecolor='white',
                    dpi=300)
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def ONEDummy_relationship_to_keyvars(self, NicheDummy, the_panel):
        """
        NicheDummy is one of the NicheDummies for different subsamples
        """
        # ----------------- select relationship with key variables -----------------------------
        key_vars = ['score',
                    'ratings',
                    'reviews',
                    'minInstalls',
                    'minInstallsTop',
                    'minInstallsMiddle',
                    'minInstallsBottom',
                    'CategoricalminInstalls',
                    'price',
                    'paidTrue',
                    'containsAdsTrue',
                    'offersIAPTrue']
        kvars = [i + '_' + the_panel for i in key_vars]
        time_invariants_vars = [
                     'genreIdGame',
                     'contentRatingAdult',
                     'DaysSinceReleased']
        kvars.extend(time_invariants_vars)
        nichedummies = [i + 'NicheDummy' for i in self.ssnames]
        kvars.extend(nichedummies)
        # we are comparing niche dummies (under different samples) against all other dummies
        compare_against1 = ['genreIdGame',
                            'contentRatingAdult',
                            'paidTrue_' + the_panel,
                            'offersIAPTrue_' + the_panel,
                            'containsAdsTrue_' + the_panel,
                            'CategoricalminInstalls_' + the_panel]
        compare_against2 = ['score_' + the_panel,
                            'ratings_' + the_panel,
                            'reviews_' + the_panel]
        # --------------- LOOPING THROUGH EACH SUBSAMPLE ---------------------------------
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def key_var_definition(self):
        df = pd.Series(reg_preparation.var_definition).to_frame().reset_index()
        df.columns = ['Variable', 'Definition']
        df.set_index('Variable', inplace=True)
        # -------------- convert to latex --------------------------------------------------
        filename = self.initial_panel + '_variable_definition.tex'
        df2 = df.to_latex(buf=reg_preparation.descriptive_stats_tables / filename,
                           multirow=True,
                           multicolumn=True,
                           caption=('Descriptive Statistics of Key Variables'),
                           position='h!',
                           label='table:1',
                           na_rep='',
                           escape=False)
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def add_sum_row(self, df):
        sum_row = df.sum(axis=0)
        sum_row = sum_row.to_frame().T
        sum_row.index = ['sum']
        df = pd.concat([df, sum_row], join="inner")
        return df

    def add_sum_col(self, df):
        sum_row = df.sum(axis=1)
        df['sum'] = sum_row
        return df

    def descriptive_stats_for_single_panel(self,
                                           continuous_vars,
                                           dummy_vars,
                                           cat_vars,
                                           time_invar_dum,
                                           time_invar_con,
                                           the_panel,
                                           add_sum_row_col=True):
        """
        This is must be run after self.create_new_dummies_from_cat_var to get updated self.df
        """
        # ----- Select Vars --------------------------------------------------------------
        con_vars = [i + '_' + the_panel for i in continuous_vars]
        con_vars.extend(time_invar_con)
        dum_vars = [i + '_' + the_panel for i in dummy_vars]
        dum_vars.extend(time_invar_dum)
        cat_vars = [i + '_' + the_panel for i in cat_vars]
        con_and_dum_vars = copy.deepcopy(con_vars)
        con_and_dum_vars.extend(dum_vars)
        # ----- Select DFs ---------------------------------------------------------------
        new_df = self.df.copy(deep=True)
        con_vars_df = new_df[con_vars]
        dum_vars_df = new_df[dum_vars]
        cat_vars_df = new_df[cat_vars]
        con_and_dum_df = new_df[con_and_dum_vars]
        # ----- Continuous Variables Summary Stats ---------------------------------------
        con_vars_sum_stats = con_vars_df.agg(['mean', 'std', 'min', 'median', 'max', 'count'], axis=0)
        # ----- Dummy Variables Count ----------------------------------------------------
        dum_stats_dfs = []
        for i in dum_vars:
            dum_vars_df['Count'+i] = 0
            df = dum_vars_df[[i, 'Count'+i]].groupby(i).count()
            dum_stats_dfs.append(df)
        dum_vars_sum_stats = functools.reduce(lambda a, b: a.join(b, how='inner'), dum_stats_dfs)
        if add_sum_row_col is True:
            dum_vars_sum_stats = self.add_sum_row(dum_vars_sum_stats)
        # ----- Continuous and Dummy Variables Together ----------------------------------
        con_and_dum_vars_stats = con_and_dum_df.agg(['mean', 'std', 'min', 'median', 'max', 'count'], axis=0)
        con_and_dum_vars_stats = con_and_dum_vars_stats.T
        con_and_dum_vars_stats['count'] = con_and_dum_vars_stats['count'].astype(int)
        dum_stats_dfs = []
        for i in dum_vars:
            dum_vars_df['Count_' + i] = 0
            df = dum_vars_df[[i, 'Count_' + i]].groupby(i).count()
            dum_stats_dfs.append(df)
        dum_vars_sum_stats = functools.reduce(lambda a, b: a.join(b, how='inner'), dum_stats_dfs)
        for i in dum_vars_sum_stats.columns:
            dum_vars_sum_stats.rename(columns={i: i.lstrip('Count').lstrip('_')}, inplace=True)
        for i in dum_vars_sum_stats.index:
            dum_vars_sum_stats.rename(index={i: str(i) + '_Count'}, inplace=True)
        dum_vars_sum_stats = dum_vars_sum_stats.T
        cd_sum_stats = con_and_dum_vars_stats.join(dum_vars_sum_stats, how='left')
        # ---- Categorical Variables Count -----------------------------------------------
        cat_stats_dict = dict.fromkeys(cat_vars)
        for i in cat_vars:
            cat_vars_df['Count'+i] = 0
            df = cat_vars_df[[i, 'Count'+i]].groupby(i).count()
            if 'minInstalls' in i:
                df.sort_index(inplace=True)
            else:
                df.sort_values(by='Count'+i, ascending=False, inplace=True)
            if add_sum_row_col is True:
                df = self.add_sum_row(df)
            cat_stats_dict[i] = df
        # ----- Dummy by Dummy and Dummy by Category --------------------------------------
        dummy_cat_dfs = []
        for i in dum_vars:
            sub_dummy_cat_dfs = []
            for j in cat_vars:
                df = new_df[[i, j]]
                df2 = pd.crosstab(df[i], df[j])
                df2.columns = [str(c) + '_' + the_panel for c in df2.columns]
                sub_dummy_cat_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), sub_dummy_cat_dfs)
            df3.index = [i + '_' + str(j) for j in df3.index]
            dummy_cat_dfs.append(df3)
        dummy_cat_cocat_df = functools.reduce(lambda a, b: pd.concat([a,b], join='inner'), dummy_cat_dfs)
        if add_sum_row_col is True:
            dummy_cat_cocat_df = self.add_sum_col(dummy_cat_cocat_df)
        # ----- Categorical Var by Categorical Var ----------------------------------------
        i, j = cat_vars[0], cat_vars[1]
        df = new_df[[i, j]]
        cc_df = pd.crosstab(df[i], df[j])
        cc_df.columns = [str(c) + '_' + the_panel for c in cc_df.columns]
        cc_df.index = [str(c) + '_' + the_panel for c in cc_df.index]
        if add_sum_row_col is True:
            cc_df = self.add_sum_row(cc_df)
            cc_df = self.add_sum_col(cc_df)
        # ----- Continuous Variables by Dummy ---------------------------------------------
        continuous_by_dummies = dict.fromkeys(con_vars)
        for i in con_vars:
            sub_groupby_dfs = []
            for j in dum_vars:
                df = new_df[[i, j]]
                agg_func_math = {i:['mean', 'std', 'min', 'median', 'max', 'count']}
                df2 = df.groupby([j]).agg(agg_func_math, axis=0)
                df2.columns = ['mean', 'std', 'min', 'median', 'max', 'count']
                sub_groupby_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: pd.concat([a, b], join='inner'), sub_groupby_dfs)
            df3.index = [j + '_' + str(z) for z in df3.index]
            continuous_by_dummies[i] = df3
        # ----- Continuous Variables by Categorical ---------------------------------------
        groupby_cat_dfs = dict.fromkeys(cat_vars)
        for i in cat_vars:
            sub_groupby_dfs = []
            for j in con_vars:
                df = new_df[[i, j]]
                agg_func_math = {j:['mean', 'std', 'min', 'median', 'max', 'count']}
                df2 = df.groupby([i]).agg(agg_func_math, axis=0)
                sub_groupby_dfs.append(df2)
            df3 = functools.reduce(lambda a, b: a.join(b, how='inner'), sub_groupby_dfs)
            df3.index = [str(c) + '_' + the_panel for c in df3.index]
            groupby_cat_dfs[i] = df3
        # ----- Update Instance Attributes ------------------------------------------------
        self.descriptive_stats_tables = {'continuous_vars_stats': con_vars_sum_stats,
                                         'dummy_vars_stats': dum_vars_sum_stats,
                                         'continuous_and_dummy_vars_stats': cd_sum_stats,
                                         'categorical_vars_count': cat_stats_dict,
                                         'crosstab_dummy_categorical_vars': dummy_cat_cocat_df,
                                         'crosstab_two_categorical_vars': cc_df,
                                         'continuous_vars_by_dummies': continuous_by_dummies,
                                         'continuous_vars_by_categorical': groupby_cat_dfs}
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

    def customize_and_output_descriptive_stats_pandas_to_latex(self, the_panel):
        """
        :param df_dict: self.descriptive_stats_tables, the output of self.descriptive_stats_for_single_panel()
        since 'continuous_and_dummy_vars_stats' already included all the variables of interest to show their summary stats
        so I will not select more varibales.
        :return:
        """
        df2 = self.descriptive_stats_tables['continuous_and_dummy_vars_stats'].copy(deep=True)
        # -------------- round -------------------------------------------------------------
        for i in df2.columns:
            if i not in ['1_Count', '0_Count', 'count']:
                df2[i] = df2[i].astype(float).round(decimals=2)
        # for i in df2.columns:
        #     if i in ['DaysSinceReleased', 'reviews_'+the_panel]:
        #         df2[i] = df2[i].apply(lambda x: int(x) if not math.isnan(x) else x)
        # df2 = df2.T
        # -------------- adjust the order of rows and columns to display --------------------
        def set_row_order(x, the_panel):
            if the_panel in x:
                x = x.rstrip(the_panel).rstrip('_')
            for k in reg_preparation.descriptive_stats_table_row_order.keys():
                if k == x:
                    return reg_preparation.descriptive_stats_table_row_order[k]
        df2 = df2.reset_index()
        df2.rename(columns={'index': 'Variable'}, inplace=True)
        df2['row_order'] = df2['Variable'].apply(lambda x: set_row_order(x, the_panel))
        df2.sort_values(by='row_order', inplace=True)
        df2.set_index('Variable', inplace=True)
        df2.drop(['row_order'], axis=1, inplace=True)
        df2 = df2[['mean', 'std', 'min', 'median', 'max', '1_Count', '0_Count', 'count']]
        # -------------- change row and column names ---------------------------------------
        for i in df2.columns:
            for j in reg_preparation.descriptive_stats_table_column_map.keys():
                if j == i:
                    df2.rename(columns={i: reg_preparation.descriptive_stats_table_column_map[j]}, inplace=True)
        def set_row_names(x, the_panel):
            if the_panel in x:
                x = x.rstrip(the_panel).rstrip('_')
            for j in reg_preparation.var_latex_map.keys():
                if j == x:
                    return reg_preparation.var_latex_map[j]
        df2 = df2.reset_index()
        df2['Variable'] = df2['Variable'].apply(lambda x: set_row_names(x, the_panel))
        df2 = df2.set_index('Variable')
        # -------------- convert to latex --------------------------------------------------
        filename = self.initial_panel + '_descriptive_stats_for_' + the_panel + '.tex'
        df3 = df2.to_latex(buf=reg_preparation.descriptive_stats_tables / filename,
                           multirow=True,
                           multicolumn=True,
                           caption=('Descriptive Statistics of Key Variables'),
                           position='h!',
                           label='table:2',
                           na_rep='',
                           escape=False)
        return reg_preparation(initial_panel=self.initial_panel,
                                   all_panels=self.all_panels,
                                   tcn=self.tcn,
                                   subsample_names=self.ssnames,
                                   df=self.df,
                                   text_label_df=self.text_label_df,
                                   combined_df=self.cdf,
                                   
                                   broad_niche_cutoff=self.broad_niche_cutoff,
                                   nicheDummy_labels=self.nicheDummy_labels,
                                   long_cdf=self.long_cdf,
                                   individual_dummies_df=self.i_dummies_df)

