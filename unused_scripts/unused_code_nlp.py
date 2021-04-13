    ################## deleting, impute and format text columns for the FULL SAMPLE ####################################
    def find_appids_to_remove_before_imputing(self):
        """
        deleting appids that have all missing values in text col for ALL panels
        """
        cols = [self.tcn + '_' + item for item in self.all_panels]
        text_df = self.df[cols]
        null_data = text_df[text_df.isnull().any(axis=1)]
        null_data_t = null_data.T
        appids_to_remove = []
        for appid in null_data_t.columns:
            if null_data_t[appid].isnull().all():
                appids_to_remove.append(appid)
        print('before removing rows with all none in text cols, we have', len(self.df.index))
        self.df = self.df.drop(appids_to_remove, axis=0)
        print('after removing rows with all none in text cols, we have', len(self.df.index))
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)


    def generate_save_input_text_col(self):
        """
        Purpose of creating this cell is to avoid creating run_subsample switch in each function below.
        Everytime you run the full sample NLP, you run the sub samples NLP simultaneously, it would take longer, but anyways.
        """
        full_sample = self.df['clean_all_panel_' + self.tcn].copy(deep=True)
        game_subsample = self.game_subsamples['game']['clean_all_panel_' + self.tcn].copy(deep=True)
        nongame_subsample = self.game_subsamples['nongame']['clean_all_panel_' + self.tcn].copy(deep=True)
        self.input_text_cols = {'full': full_sample,
                                'game': game_subsample,
                                'nongame': nongame_subsample}
        return nlp_pipeline(
                 tcn=self.tcn,
                 initial_panel=self.initial_panel,
                 all_panels=self.all_panels,
                 df=self.df,
                 game_subsamples=self.game_subsamples,
                 input_text_cols = self.input_text_cols,
                 tf_idf_matrices = self.tf_idf_matrices,
                 svd_matrices=self.svd_matrices,
                 output_labels=self.output_labels)