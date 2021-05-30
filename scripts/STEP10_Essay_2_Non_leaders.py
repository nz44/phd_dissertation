class regression_non_leaders():
    panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/____WEB_SCRAPER____/__PANELS__')
    reg_table_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/reg_results_tables')
    descriptive_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/tables')
    descriptive_stats_graphs = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/__CODING__/PycharmProjects/GOOGLE_PLAY/descriptive_stats/graphs')
    graph_ylabel_dict = {'containsAdsTrue': 'ContainsAds',
                         'offersIAPTrue': 'OffersIAP',
                         'paidTrue': 'Paid',
                         'Imputedprice': 'Price'}
    graph_subsample_title_dict = {'minInstalls ImputedminInstalls_tier1': 'Tier 1 (Minimum Installs)',
                                  'minInstalls ImputedminInstalls_tier2': 'Tier 2 (Minimum Installs)',
                                  'minInstalls ImputedminInstalls_tier3': 'Tier 3 (Minimum Installs)',
                                  'developer top': 'Top (Companies)',
                                  'developer non-top': 'Non-top (Companies)',
                                  'full full': 'Full Sample'}

    def __init__(self,
                 initial_panel,
                 all_panels,
                 dep_vars,
                 independent_vars,
                 subsample_names=None,
                 reg_dict=None,
                 reg_dict_xy=None,
                 single_panel_df=None,
                 subsample_op_results=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.dep_vars = dep_vars
        self.independent_vars = independent_vars
        self.ssnames = subsample_names
        self.reg_dict = reg_dict
        self.reg_dict_xy = reg_dict_xy
        self.single_panel_df = single_panel_df
        self.subsample_op_results = subsample_op_results

    def open_imputed_and_deleted_missing_df(self):
        f_name = self.initial_panel + '_imputed_and_deleted_missing.pickle'
        q = divide.panel_path / f_name
        with open(q, 'rb') as f:
            self.df = pickle.load(f)

    def add_subsample_names(self):
        self.ssnames = dict.fromkeys(self.reg_dict.keys())
        for key, content in self.reg_dict.items():
            self.ssnames[key] = list(content.keys())
        return regression(initial_panel=self.initial_panel,
                          all_panels=self.all_panels,
                          dep_vars=self.dep_vars,
                          independent_vars=self.independent_vars,
                          subsample_names=self.ssnames,
                          reg_dict=self.reg_dict,
                          reg_dict_xy=self.reg_dict_xy,
                          single_panel_df=self.single_panel_df,
                          subsample_op_results=self.subsample_op_results)