import pandas as pd
from pathlib import Path
import pickle
import copy
import math
import locale
from statsmodels.compat import lzip
pd.options.display.max_rows = 999
pd.options.mode.chained_assignment = None
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib import dates
import matplotlib.patches as mpatches
from sklearn import preprocessing
from sklearn import metrics
from scipy.stats import boxcox
import statsmodels.formula.api as smf
import statsmodels.stats.api as sms
from statsmodels.compat.python import lzip
import statsmodels.api as sm
from linearmodels.panel import PooledOLS
from linearmodels.panel import PanelOLS
from datetime import datetime
from itertools import combinations
import functools
today = datetime.today()
yearmonth = today.strftime("%Y%m")


class stats_and_regs:
    """2021 July 18
    This is the new version written based on the STEP10_ESSAY_2_3_Long_Table_Prep.py
    2022 Mar 26
    Combine market leaders and followers regression and statistics into the same class.
    Run robustness checks which include regressions with deleted missings (without imputing the missing) and with imputing the missing
    and they ways to validate the regression coefficients of the same variables (loosely speaking niche and post niche) are different
    for different sub-samples by pooled regression with sample dummies.
    """
# linux paths
    full_sample_panel_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/___full_sample___')
    nlp_stats_path = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/__PANELS__/nlp_stats')
    des_stats = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____')
    des_stats_tables = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/TABLES')
    des_stats_graphs = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____DESCRIPTIVE_STATS____/GRAPHS')
    ols_results = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____OLS_RESULTS____')
    panel_results = Path(
        '/home/naixin/Insync/naixin88@sina.cn/OneDrive/_____GWU_ECON_PHD_____/___Dissertation___/___WEB_SCRAPER___/____PANEL_RESULTS____')
# win paths
#     full_sample_panel_path = Path(
#         "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//__PANELS__//___full_sample___")
#     nlp_stats_path = Path(
#         "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//__PANELS__//nlp_stats")
#     des_stats_tables = Path(
#         "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____DESCRIPTIVE_STATS____//TABLES")
#     des_stats_graphs = Path(
#         "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____DESCRIPTIVE_STATS____//GRAPHS")
#     ols_results = Path(
#         "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____OLS_RESULTS____")
#     panel_results = Path(
#         "C://Users//naixi//OneDrive//_____GWU_ECON_PHD_____//___Dissertation___//___WEB_SCRAPER___//____PANEL_RESULTS____")

    # ----------------------------- slicing variables --------------------------------------------------------------------
    sub_sample_key_level1 = ['FULL', 'ML', 'MF']
    FULL_sample_key_level2 = ['FULL', 'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                              'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']
    ML_sample_key_level2 = ['ML', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL']
    MF_sample_key_level2 = ['MF', 'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL']
    # These sample dummies will be used in regression with FULL, MF and ML
    reg_sample_dummies = ['Tier1', 'Tier2', 'top_firm', 'ML',
                          'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_MEDICAL',
                          'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_MEDICAL',
                          'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_MEDICAL']
    sub_sample_categorical_vars = ['MF_CAT', 'ML_CAT', 'FULL', 'ML_MF', 'FULL_CAT', 'FULL_TIER', 'FULL_FIRM']
    sub_sample_graph_cat_vars_d = {'FULL': ['FULL_CAT', 'FULL_TIER', 'FULL_FIRM'],
                                   'ML':   ['ML_CAT'],
                                   'MF':   ['MF_CAT']}
    cat_hue_order_d = {'FULL_TIER': ['Tier1', 'Tier2', 'Tier3'],
                       'FULL_FIRM': ['top_firm', 'non_top_firm'],
                       'FULL_CAT': ['FULL_BUSINESS', 'FULL_GAME', 'FULL_LIFESTYLE', 'FULL_MEDICAL', 'FULL_SOCIAL'],
                       'ML_CAT': ['ML_BUSINESS', 'ML_GAME', 'ML_LIFESTYLE', 'ML_MEDICAL', 'ML_SOCIAL'],
                       'MF_CAT': ['MF_BUSINESS', 'MF_GAME', 'MF_LIFESTYLE', 'MF_MEDICAL', 'MF_SOCIAL']}
    cat_label_order_d = {'FULL_TIER': ['Tier 1', 'Tier 2', 'Tier 3'],
                         'FULL_FIRM': ['Apps from Top Firm', 'Apps from Non-top Firm'],
                         'FULL_CAT': ['Business', 'Game', 'Lifestyle', 'Medical', 'Social'],
                         'ML_CAT': ['Business', 'Game', 'Lifestyle', 'Medical', 'Social'],
                         'MF_CAT': ['Business', 'Game', 'Lifestyle', 'Medical', 'Social']}
    regplot_color_palette = {'FULL':{'FULL_CAT': sns.color_palette("hls", 5),
                                     'FULL_TIER': sns.color_palette("hls", 3),
                                     'FULL_FIRM': sns.color_palette("hls", 2)},
                             'ML':{'ML_CAT': sns.color_palette("hls", 5)},
                             'MF':{'MF_CAT': sns.color_palette("hls", 5)}}

    sub_sample_d =  dict.fromkeys(['FULL', 'MF', 'ML'])

    sub_sample_l = FULL_sample_key_level2 + ML_sample_key_level2 + MF_sample_key_level2

    graph_layout_categorical = plt.subplots(3, 2)
    graph_layout_full_firm = plt.subplots(2, 1)
    graph_layout_full_tiers = plt.subplots(3, 1)

    # 20221001：I know 'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA' are created based on imputed data
    # so they should be imputed variables. Add them in here just because I need descriptive statistics for those variables
    # in the original data sample
    core_dummy_y_vars_d = {'original': ['containsAdsdummy', 'offersIAPdummy'],
                           'imputed':  ['imputed_containsAdsdummy', 'imputed_offersIAPdummy',
                                        'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA']}
    core_scaled_continuous_y_vars_d = {'original': ['nlog_price', 'nlog_minInstalls'],
                                       'imputed':  ['nlog_imputed_price', 'nlog_imputed_minInstalls']}
    imputed_y_vars = {'FULL': ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'imputed_offersIAPdummy', 'imputed_containsAdsdummy',
'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA'],
                      # by definition, for market follower, change to tier 1 and change to top do not exist
                      'MF': ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'imputed_offersIAPdummy', 'imputed_containsAdsdummy',
'noisy_death', 'MA'],
                      'ML': ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'imputed_offersIAPdummy', 'imputed_containsAdsdummy',
'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA']}
    all_imputed_control_vars = ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent', 'imputed_daysreleased', 'imputed_size']
    step_models_imputed_control_vars = {'FULL': {
                                            'nlog_imputed_price': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'nlog_imputed_minInstalls': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'imputed_offersIAPdummy': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_adultcontent'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_adultcontent', 'imputed_score'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'imputed_containsAdsdummy': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'noisy_death': {
                                                0: [],
                                                1: ['imputed_daysreleased'],
                                                2: ['imputed_daysreleased', 'imputed_score'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score', 'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'T_TO_TIER1_minInstalls': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'T_TO_top_firm': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_score'],
                                                3: ['nlog_imputed_reviews', 'imputed_adultcontent', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_adultcontent', 'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'MA': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_adultcontent'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score', 'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            }
                                        },
                                        'MF': {
                                            'nlog_imputed_price': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'nlog_imputed_minInstalls': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score', 'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'imputed_offersIAPdummy': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_adultcontent'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_adultcontent', 'imputed_score'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'imputed_containsAdsdummy': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score', 'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'noisy_death': {
                                                0: [],
                                                1: ['imputed_daysreleased'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score',
                                                    'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'T_TO_top_firm': {
                                                0: [],
                                                1: ['imputed_daysreleased'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score',
                                                    'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'MA': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_adultcontent'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score',
                                                    'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            }
                                        },
                                        'ML': {
                                            'nlog_imputed_price': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score', 'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'nlog_imputed_minInstalls': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_size'],
                                                3: ['nlog_imputed_reviews', 'imputed_size', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'imputed_offersIAPdummy': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_adultcontent'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_adultcontent', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'imputed_containsAdsdummy': {
                                                0: [],
                                                1: ['nlog_imputed_reviews'],
                                                2: ['nlog_imputed_reviews', 'imputed_size'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_size'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased',
                                                    'imputed_score', 'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'noisy_death': {
                                                0: [],
                                                1: ['imputed_daysreleased'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_adultcontent'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_adultcontent',
                                                    'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'T_TO_TIER1_minInstalls': {
                                                0: [],
                                                1: ['imputed_daysreleased'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score',
                                                    'imputed_adultcontent'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'T_TO_top_firm': {
                                                0: [],
                                                1: ['imputed_size'],
                                                2: ['imputed_size', 'imputed_daysreleased'],
                                                3: ['imputed_size', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score',
                                                    'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                            'MA': {
                                                0: [],
                                                1: ['imputed_daysreleased'],
                                                2: ['nlog_imputed_reviews', 'imputed_daysreleased'],
                                                3: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score'],
                                                4: ['nlog_imputed_reviews', 'imputed_daysreleased', 'imputed_score',
                                                    'imputed_size'],
                                                5: ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent',
                                                    'imputed_daysreleased', 'imputed_size']
                                            },
                                        }
                                    }

    chosen_step_for_reg = {
        'FULL': 4,
        'MF': 3,
        'ML': 3
    }
    sub_sample_dummies = {'FULL': ['ML'],  # base group is MF
                          'ML': ['ML_GAME', 'ML_SOCIAL', 'ML_BUSINESS', 'ML_MEDICAL'],  # base group is lifestyle
                          'MF': ['MF_GAME', 'MF_SOCIAL', 'MF_BUSINESS', 'MF_MEDICAL']}
    unstacked_ols_categorical_control_vars = {'FULL': ['FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_MEDICAL'],
                                              'ML': ['ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_MEDICAL'],
                                              'MF': ['MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_MEDICAL']}
    # since you are including interaction of niche and ML in the full sample regression, you need to include ML indicator in the regression as well
    # since you are including interaction between niche and categories in ML and MF sub-sample regression, so the categories remain
    stacked_ols_categorical_control_vars = {'FULL': ['ML', 'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_MEDICAL'], # lifestyle is the baseline
                                              'ML': ['ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_MEDICAL'],
                                              'MF': ['MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_MEDICAL']}

    # niche variables are calculated from kmeans on self.tcn + 'Clean' (which is in turn based on 'imputed_'+self.tcn)
    # essentially all niche variables are imputed variables (based on imputed app descriptions)
    niche_vars = ['continuous_niche']
    # time dummies and interactions only exist in long form dataframe
    time_dummies = ['period_0', 'period_1', 'period_2', 'period_3']
    time_interactions = ['period_0_continuous_niche', 'period_1_continuous_niche',
                           'period_2_continuous_niche', 'period_3_continuous_niche']
    # For the purpose of descriptive statistics, all variables are scaled and WITHOUT adding whitenoise (so that dummy stays dummy)
    scaled_vars = {'original': ['nlog_price', 'nlog_minInstalls', 'nlog_reviews'],
                   'imputed': ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'nlog_imputed_reviews']}
    # no need to specify original and imputed in scale_var_dict because every var in here has imputed form
    scale_var_dict = {
        'nlog_plus_one': ['reviews', 'minInstalls', 'price']
    }
    # y variable labels that make the graph easier to read
    super_y_label = {'original': {'nlog_price': 'Natural Log of App Price ($)',
                                  'nlog_minInstalls': 'Natural Log of the Lower Bound Install Brackets',
                                  'containsAdsdummy': 'Dummy Variable Indicating Whether an App Contains Ad',
                                  'offersIAPdummy': 'Dummy Variable Indicating Whether an App Has In-app Purchases'},
                     'imputed': {'nlog_imputed_price': 'Natural Log of App Price ($) (Imputed)',
                                 'nlog_imputed_minInstalls': 'Natural Log of the Lower Bound Install Brackets (Imputed)',
                                 'imputed_containsAdsdummy': 'Dummy Variable Indicating Whether an App Contains Ad (Imputed)',
                                 'imputed_offersIAPdummy': 'Dummy Variable Indicating Whether an App Has In-app Purchases (Imputed)',
                                 'noisy_death': 'Proxy Dummy Variable Indicating Whether an App has Died',
                                 'T_TO_TIER1_minInstalls': 'Dummy Variable Indicating Whether the Lower Bound of Cumulative Installs Crossed the Tier1 Threshhold',
                                 'T_TO_top_firm': 'Dummy Variable Indicating Whether an App Changed Ownership From a Non-top Firm to a Top Firm',
                                 'MA': 'Dummy Variable Indicating Whether an App Underwent a Merger or Acquisition'}}

    short_x_label = {'original': {'score': 'Rating',
                                  'nlog_reviews': 'Log Number of Reviews',
                                  'adultcontent': 'Contain Adult Content',
                                  'daysreleased': 'Time Since Launch (Days)',
                                  'size': 'App Size (MB)',
                                  'continuous_niche': 'Niche'},
                     'imputed': {'imputed_score': 'Rating (Imputed)',
                                 'nlog_imputed_reviews': 'Log Number of Reviews (Imputed)',
                                 'imputed_adultcontent': 'Contain Adult Content (Imputed)',
                                 'imputed_daysreleased': 'Time Since Launch (Days) (Imputed)',
                                 'imputed_size': 'App Size (MB) (Imputed)',
                                 'continuous_niche': 'Niche'}}

    # -------------------- test names -----------------------------------------------------------
    # https://www.statsmodels.org/dev/examples/notebooks/generated/regression_diagnostics.html
    jb_test_names = ["Jarque-Bera", "Chi^2 two-tail prob.", "Skew", "Kurtosis"]
    bp_test_names = ["Lagrange multiplier statistic", "p-value", "f-value", "f p-value"]

    ##################################################################################################################
    def __init__(self,
                 initial_panel,
                 all_panels,
                 df=None,
                 ss_data_dict=None,
                 reg_results=None):
        self.initial_panel = initial_panel
        self.all_panels = all_panels
        self.df = df
        self.ss_data_dict = ss_data_dict
        self.reg_results = reg_results

    # ====================== The set of functions below are regularly used common functions in pre_processing class =============================
    def _open_df(self, balanced, keyword):
        """
        :param balanced:
        :param keyword: could be any of 'merged', 'imputed', 'nlp' or 'reg_ready'
        :return:
        """
        print('------------------------ open_df ' + keyword + ' ---------------------------')
        f_name = self.initial_panel + '_' + balanced + '_' + keyword + '.pickle'
        q = self.full_sample_panel_path / f_name
        with open(q, 'rb') as f:
            df = pickle.load(f)
        return df

    def _save_df(self, DF, balanced, keyword):
        """
        I will save the df at the very end, which include imputed variables, sub-sample dummies and NLP niche variables
        :param balanced: Whether to include appids that are missing in each subsequent month as compared to the original month
        :param keyword: could be any of 'merged', 'imputed', 'nlp' or 'reg_ready'
        :return:
        """
        print('------------------------ _save_df ' + keyword + ' ---------------------------')
        f_name = self.initial_panel + '_' + balanced + '_' + keyword + '.pickle'
        q = self.full_sample_panel_path / f_name
        pickle.dump(DF, open(q, 'wb'))
        return None

    def _scale_var_printout_descriptive_stats(self, df, imputed):
        """
        :param imputed should either be 'original' or 'imputed'
        :param df: A subsetted dataframe has no missing in the original variable columns
                   The existing columns in this dataframe should all be numeric
        :param scale_var_dict: key is the method and the value is a list of core variable names
        :return:
        """
        print('---- _scale_var_printout_descriptive_stats ------')
        for method, vars in self.scale_var_dict.items():
            for var in vars:
                if imputed == 'imputed':
                    vs = ['imputed_' + var + '_' + m for m in self.all_panels]
                else:
                    vs = [var + '_' + m for m in self.all_panels]
                for v in vs:
                    if method == 'nlog_plus_one':
                        # print(v + ' BEFORE nlog_plus_one')
                        # print(df[v].describe())
                        df['nlog_' + v] = df.apply(lambda row: np.log(row[v] + 1), axis=1)
                        # print(v + ' AFTER nlog_plus_one')
                        # print(df['nlog_' + v].describe())
        # --- convert everything to numeric before regression or graphing ---------------
        # num_cols = [x for x in list(df.columns) if x not in self.sub_sample_categorical_vars]
        # print('numeric columns are')
        # print(num_cols)
        # for i in num_cols:
        #     df[i] = pd.to_numeric(df[i])
        return df

    def _create_categorical_sub_sample_vars(self, df):
        """
        :param df: should be self._open_df(balanced=balanced, keyword='imputed')
        :return:
        """
        print('------------------------ _create_categorical_sub_sample_vars ----------------------')
        # print(list(df.columns))
        # --------- create categorical sub-sample slicing variables for future graphing --------
        for v in ['ML', 'MF', 'ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_LIFESTYLE', 'ML_MEDICAL',
                  'MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_LIFESTYLE', 'MF_MEDICAL',
                  'Tier1', 'Tier2', 'Tier3', 'top_firm', 'non_top_firm',
                  'FULL_GAME', 'FULL_BUSINESS', 'FULL_SOCIAL', 'FULL_LIFESTYLE', 'FULL_MEDICAL']:
            # print(df[v].value_counts(dropna=False))
            df[v + '_cat'] = df.apply(lambda row: v if row[v] == 1 else '', axis=1)
            # print(df[v + '_cat'].value_counts(dropna=False))
        df['ML_MF'] = df['ML_cat'] + df['MF_cat']
        df['ML_CAT'] = df['ML_GAME_cat'] + df['ML_BUSINESS_cat'] + df['ML_SOCIAL_cat'] + \
                       df['ML_LIFESTYLE_cat'] + df['ML_MEDICAL_cat']
        df['MF_CAT'] = df['MF_GAME_cat'] + df['MF_BUSINESS_cat'] + df['MF_SOCIAL_cat'] + \
                       df['MF_LIFESTYLE_cat'] + df['MF_MEDICAL_cat']
        df['FULL_TIER'] = df['Tier1_cat'] + df['Tier2_cat'] + df['Tier3_cat']
        df['FULL_FIRM'] = df['top_firm_cat'] + df['non_top_firm_cat']
        df['FULL_CAT'] = df['FULL_GAME_cat'] + df['FULL_BUSINESS_cat'] + df['FULL_SOCIAL_cat'] + \
                         df['FULL_LIFESTYLE_cat'] + df['FULL_MEDICAL_cat']
        # easier for the purpose of groupby describe
        df['FULL'] = 'FULL'
        for v in ['FULL', 'ML_MF', 'ML_CAT', 'MF_CAT', 'FULL_TIER', 'FULL_FIRM', 'FULL_CAT']:
            df[v] = df[v].astype("category")
        print(df[['FULL', 'ML_MF', 'ML_CAT', 'MF_CAT', 'FULL_TIER', 'FULL_FIRM', 'FULL_CAT']].dtypes)
        df.drop(columns=['ML_cat', 'MF_cat', 'ML_GAME_cat', 'ML_BUSINESS_cat', 'ML_SOCIAL_cat', 'ML_LIFESTYLE_cat', 'ML_MEDICAL_cat',
                         'MF_GAME_cat', 'MF_BUSINESS_cat', 'MF_SOCIAL_cat', 'MF_LIFESTYLE_cat', 'MF_MEDICAL_cat',
                         'Tier1_cat', 'Tier2_cat', 'Tier3_cat', 'top_firm_cat', 'non_top_firm_cat', 'FULL_GAME_cat',
                         'FULL_BUSINESS_cat', 'FULL_SOCIAL_cat', 'FULL_LIFESTYLE_cat', 'FULL_MEDICAL_cat'], inplace=True)
        return df

    def create_subsample_dict_and_merge_in_niche(self, picked_tfidf_param, picked_k, balanced):
        """
        20220911, by default, I will only use niche variables generated from full sample in all
                  the descriptive stats and regression analysis in my dissertation. That means the ML and MF sub-samples and categorical sub-samples
                  will also use niche generated using the full sample. Interpretation would be that niche product is relative to the whole sample, not the sub-sample.
                  This would be easier to compare the coefficients in different sub-samples by running a pooled regression.
        :return: The slicing dummies are based on imputed variables such as imputed_minInstalls, top firms (imputed_developer_ and developerClean_), and imputed_genreId
                please refer to STEP2_pre_processing create_sub_sample_dummies.
        """
        print('------------------------ create_subsample_dict_and_merge_in_niche_vars ---------------------------')
        # df2 is the imputed dataframe and categorical variables have been created
        # ----------- open nlp k means cluster labels ------------------------------------------
        f_name = self.initial_panel + '_' + picked_tfidf_param + '_cluster_num_' + str(picked_k) + '_kmeans_labels_with_appids.pickle'
        q = self.nlp_stats_path / f_name
        # 20221001: there is only one set of niche index that is generated from the full sample
        # in the future sub-sample regression I will only use this single niche index
        full_niche_df = pickle.load(open(q, 'rb'))
        # -------- create empty dictionary placeholder for dictionary of dataframes ------------
        res_original = copy.deepcopy(self.sub_sample_d)
        res_imputed = copy.deepcopy(self.sub_sample_d)
        ss_data_d = {'original': res_original, 'imputed': res_imputed}
        # ----------- open imputed dataframe ---------------------------------------------------
        self.df = self._open_df(balanced=balanced, keyword='imputed')
        df2 = self.df.copy()
        df2 = self._create_categorical_sub_sample_vars(df=df2)
        print('----------- open imputed dataframe ------------')
        print(df2.shape)
        # print(list(df2.columns))
        for k in self.sub_sample_d:
            print(k + '--- before merging in niche variables')
            # ----------- slicing into sub-samples and merge in the nlp labels -----------------
            if k == 'FULL':
                df3 = df2.copy()
            else:
                df3 = df2.loc[df2[k] == 1]
            print(df3.shape)
            df3 = df3.merge(full_niche_df, how='inner', left_index=True, right_index=True)
            df3 = df3.fillna(value=np.nan)
            # make the time-invariant niche variables a set of time-variant variables
            for m in self.all_panels:
                df3['continuous_niche_' + m] = df3.apply(lambda row: row['continuous_niche'], axis=1)
            df3.drop(['continuous_niche'], axis=1, inplace=True)
            print(k + '--- after merging in niche variables')
            print(df3.shape)
            for im in ['original', 'imputed']:
                # --------------------------------------------------------
                print(k + '---' + im + '--- delete missing in all the relevant variables -------------- ')
                cols = self.des_stats_all_vars[im]['dummy'] + self.des_stats_all_vars[im]['continuous']
                for i in self.vars_to_remove + self.scaled_vars[im]:
                    cols.remove(i)
                cols2 = [i + '_' + m for m in self.all_panels for i in cols]
                print('relevant columns selected are: ')
                print(cols2)
                df4 = df3.dropna(axis=0, how='any', subset=cols2)
                df4 = df4.loc[:, cols2 + self.sub_sample_categorical_vars + self.reg_sample_dummies]
                print(df4.shape)
                # --------------------------------------------------------
                df4 = self._scale_var_printout_descriptive_stats(df=df4, imputed=im)
                # --------------------------------------------------------
                print(k + '---' + im + '-------------- final columns before go into saving --------------')
                print(df4.shape)
                print(list(df4.columns))
                ss_data_d[im][k] = df4
        self.ss_data_dict = ss_data_d
        filename = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_merged_kmeans_data.pickle'
        q = self.des_stats / filename
        pickle.dump(self.ss_data_dict, open(q, 'wb'))
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict)

    def _convert_to_long(self, df, sample, chosen_step):
        # input should be df = self.ss_data_dict[im][k]
        print('--------------------------- _convert_to_long --------------------------')
        df.reset_index(inplace=True)
        df.rename(columns={'index': 'app_id'}, inplace=True)
        # print(list(df2.columns))
        core_vars = self.imputed_y_vars[sample] + ['continuous_niche'] + self.all_imputed_control_vars
        cols_m = [i + '_' + m for m in self.all_panels for i in core_vars]
        df2 = df.loc[:, cols_m + ['app_id'] + self.reg_sample_dummies]
        print('------------ ' + sample + '---STEP---' + str(chosen_step) + ' before converting to long --------------------')
        print(df2.shape)
        print(list(df2.columns))
        df2 = pd.wide_to_long(df2, core_vars, sep='_', i='app_id', j='month')
        # get month as a column for the purpose of creating time dummies
        df2.reset_index(inplace=True)
        # panel regression needs entity and time indices
        df2.set_index(['app_id', 'month'], drop=False, inplace=True)
        df2['month'] = df2['month'].astype(str)
        print(df2['month'].value_counts(dropna=False))
        # print(df2.head(1))
        df2 = self._create_time_dummies_and_interactions(df=df2)
        print('------------' + sample + '---STEP---' + str(chosen_step) + ' after converting to long --------------------')
        print(df2.shape)
        print(list(df2.columns))
        return df2

    def des_stats_as_in_dissertation_format(self, balanced, picked_tfidf_param, picked_k):
        """
        The tables generated by this function follows the format in overleaf data section
        https://www.overleaf.com/project/623deb3069c58d0c7d2ac114
        that compares both the imputed and the original variables FULL, MF and ML sub-samples for a particular month
        20220911: From today, we will only use the continuous niche variable
        :return:
        """
        print('------------------------ des_stats_as_in_dissertation_format ---------------------------')
        filename_long = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_merged_kmeans_data_LONG.pickle'
        q_long = self.des_stats / filename_long
        ldf = pickle.load(open(q_long, 'rb'))
        df_list_store = {'continuous':[], 'continuous_cat':[], 'dummy':[], 'dummy_cat':[]}
        cat_vars = {'FULL': ['ML', 'MF'],
                    'ML': ['ML_GAME', 'ML_BUSINESS', 'ML_SOCIAL', 'ML_MEDICAL', 'ML_LIFESTYLE'],
                    'MF': ['MF_GAME', 'MF_BUSINESS', 'MF_SOCIAL', 'MF_MEDICAL', 'MF_LIFESTYLE']}
        for k in self.sub_sample_d:
            for im in ['original', 'imputed']:
                vscon = self.shortlist_var_descriptive_stats[im]['continuous']
                vsdum = self.shortlist_var_descriptive_stats[im]['dummy']
                df_long = ldf[im][k].copy()
                df_long['MF'] = np.where(df_long['ML'] == 0, 1, 0)
                df_long['ML_LIFESTYLE'] = np.where((df_long['ML']==1) & (df_long['ML_GAME']==0) & (
                        df_long['ML_BUSINESS']==0) & (df_long['ML_SOCIAL']==0) & (df_long['ML_MEDICAL']==0), 1, 0)
                df_long['MF_LIFESTYLE'] = np.where(
                    (df_long['ML'] == 0) & (df_long['MF_GAME'] == 0) & (df_long['MF_BUSINESS'] == 0) & (
                                df_long['MF_SOCIAL'] == 0) & (df_long['MF_MEDICAL'] == 0), 1, 0)
            # ------------------- statistics ----------------------------------------------------
                print('-------------------', k, ' statistics --------------------')
                print('ALL COLUMN before any subsetting')
                print(list(df_long.columns))
                dfv = df_long[vscon]
                dfv.reset_index(inplace=True)
                dfv.drop(['app_id'], axis=1, inplace=True)
                tablec = dfv.groupby('month').describe().unstack(1).to_frame()
                tablec.reset_index(inplace=True)
                tablec.rename(columns={'index': 'variable'}, inplace=True)
                # by categories
                cate_continuous_dfs = []
                for CAT in cat_vars[k]:
                    dfv = df_long[vscon + [CAT]]
                    dfv2 = dfv.loc[dfv[CAT] == 1]
                    dfv2.drop([CAT], axis=1, inplace=True)
                    tablec_cat = dfv2.groupby(['month']).describe().unstack(1).reset_index()
                    tablec_cat.rename(columns={'index': 'variable'}, inplace=True)
                    tablec_cat['CAT'] = CAT
                    print(tablec_cat)
                    cate_continuous_dfs.append(tablec_cat)
                tablec_cat = pd.concat(cate_continuous_dfs, axis=0)
                # -----------------------------------------------------------------------------------
                dfv = df_long[vsdum]
                dfv.reset_index(inplace=True)
                dfv.drop(['app_id'], axis=1, inplace=True)
                print(dfv.columns)
                value_counts_dfs = []
                value_counts_dfs_cat = []
                for v in vsdum:
                    # for some weird reason pandas would not run this, so the equivalent would be
                    # tabled = dfv[['month', v]].groupby('month').value_counts(dropna=False).unstack(1)
                    tabled = dfv[['month', v]].groupby(['month', v]).size().unstack(1)
                    tabled.reset_index(inplace=True)
                    tabled['variable'] = v
                    tabled.insert(0, 'variable', tabled.pop('variable'))
                    value_counts_dfs.append(tabled)
                    # by categories
                    cate_dummy_dfs = []
                    for CAT in cat_vars[k]:
                        dfv_cat = df_long[vsdum + [CAT]]
                        dfv_cat.reset_index(inplace=True)
                        dfv_cat.drop(['app_id'], axis=1, inplace=True)
                        print(dfv_cat.columns)
                        dfv2 = dfv_cat.loc[dfv_cat[CAT] == 1]
                        dfv2.drop([CAT], axis=1, inplace=True)
                        tabled_cat = dfv2[['month', v]].groupby(['month', v]).size().unstack(1)
                        tabled_cat['variable'] = v
                        tabled_cat.insert(0, 'variable', tabled_cat.pop('variable'))
                        tabled_cat['CAT'] = CAT
                        cate_dummy_dfs.append(tabled_cat)
                    dum_v_cat_table = pd.concat(cate_dummy_dfs, axis=0)
                    print(dum_v_cat_table)
                    value_counts_dfs_cat.append(dum_v_cat_table)
                tabledum = pd.concat(value_counts_dfs, axis=0)
                tabledum_cat = pd.concat(value_counts_dfs_cat, axis=0)
                table_dict = {'continuous': tablec, 'continuous_cat': tablec_cat, 'dummy': tabledum, 'dummy_cat': tabledum_cat}
                for vtype, table in table_dict.items():
                    # table.reset_index(inplace=True)
                    table['sample'] = k
                    table['im'] = im
                    table.insert(0, 'im', table.pop('im'))
                    table.insert(1, 'sample', table.pop('sample'))
                    print(table.head())
                    df_list_store[vtype].append(table)
        # save pooled continuous and dummy variables table ---------------------------------------------
        f_name = picked_tfidf_param + '_' + str(picked_k) + '_BY_MONTH_VAR_STATS.xlsx'
        q = self.des_stats_tables / balanced / 'table_in_dissertation' / f_name
        with pd.ExcelWriter(q) as writer:
            for n in ['continuous', 'continuous_cat', 'dummy', 'dummy_cat']:
                final_table = pd.concat(df_list_store[n], axis=0)
                final_table = final_table.round(2)
                final_table.to_excel(writer, sheet_name=n)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict)



    # 20220911: delete the table_cat_y_variables_against_niche_dummy function because my dissertation does not use niche dummy any longer
    def _y_countplot_against_binned_continuous_niche(self, balanced, im, m, k, y_var, y_var_m, df, CAT, picked_tfidf_param, picked_k):
        fig, axs = plt.subplots(nrows=1, ncols=len(self.cat_hue_order_d[CAT]),
                                figsize=(len(self.cat_hue_order_d[CAT])*10, 10),
                                constrained_layout=True, sharex=True, sharey=True)
        sns.set(style="darkgrid")
        # --------- for dummy y variables --------------------------------------------------------------------
        for i in range(len(self.cat_hue_order_d[CAT])):
            df3 = df.loc[df[CAT] == self.cat_hue_order_d[CAT][i]]
            # print(df3.shape)
            # print(df3.head())
            sns.countplot(data=df3, x="continuous_niche_cut",
                          hue=y_var_m, ax=axs[i])
            axs[i].grid(True)
            axs[i].set_xlabel('Niche')
            axs[i].set_ylabel('')
            axs[i].set_title(self.cat_label_order_d[CAT][i])
            axs[i].set_xticklabels(axs[i].get_xticklabels(), rotation=90)
        fig.text(-0.005, 0.5, self.super_y_label[im][y_var], ha='center', va='center', rotation='vertical')
        if 'FULL' in CAT:
            fig.suptitle('All Apps')
        elif 'ML' in CAT:
            fig.suptitle('Market Leading Apps')
        else:
            fig.suptitle('Market Follower Apps')
        # ----------------- save ---------------------------------------
        f_name = k + '_' + picked_tfidf_param + '_' + str(picked_k) + '_' + CAT + '_' + y_var_m + '.png'
        q = self.des_stats_graphs / balanced / im / m / 'continuous_niche' / f_name
        fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return None

    def _y_boxplot_against_binned_continuous_niche(self, balanced, im, m, k, y_var, y_var_m, df, CAT, picked_tfidf_param, picked_k):
        # --------- for continuous y variables ---------------------------------------------------------------
        # hue order and label order The tiers and top and nontop firm the hue name and label name are the same
        # the df is generated by the line df2 = df[var_cols] in function graph_y_against_binned_continuous_niche
        fig, axs = plt.subplots(2, figsize=(15, 10), constrained_layout=True, sharex=True, sharey=True)
        # draw it without hue and with hue
        sns.set(style="darkgrid")
        sns.boxplot(data=df, x="continuous_niche_cut", y=y_var_m, color='skyblue', ax=axs[0])
        # I checked the hue_order does not work, but the hue follows alphabetically
        sns.boxplot(data=df, x="continuous_niche_cut", y=y_var_m, hue=CAT, ax=axs[1])
        if 'CAT' in CAT:
            h, l = axs[1].get_legend_handles_labels()
            plt.legend(title="App Categories", handles=h,
                       labels=self.cat_label_order_d[CAT],
                       bbox_to_anchor=(1.02, 0.3), loc='upper left', borderaxespad=0.1)
        else:
            plt.legend(bbox_to_anchor=(1.02, 0.3), loc='upper left', borderaxespad=0.1)
        plt.xticks(rotation=90)
        for i in range(len(axs)):
            axs[i].grid(True)
            axs[i].set_xlabel('Niche')
            axs[i].set_ylabel('')
        # plt.ylabel('Natural Log of the Lower Bound Install Brackets (Imputed)') does not work
        fig.text(-0.02, 0.5, self.super_y_label[im][y_var], ha='center', va='center', rotation='vertical')
        if 'FULL' in CAT:
            fig.suptitle('All Apps')
        elif 'ML' in CAT:
            fig.suptitle('Market Leading Apps')
        else:
            fig.suptitle('Market Follower Apps')
        # ----------------- save ---------------------------------------
        f_name = k + '_' + picked_tfidf_param + '_' + str(picked_k) + '_' + CAT + '_' + y_var_m + '.png'
        q = self.des_stats_graphs / balanced / im / m / 'continuous_niche' / f_name
        fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return None

    def graph_y_against_binned_continuous_niche(self, balanced, bin_range, m, picked_tfidf_param, picked_k):
        """
        20220913: I don't like the old graphs, because they are too complicated. I will just bin continuous niche
        on the x-axis, and graph histogram for both continuous and dummy ys
        """
        print('------------------------ graph_y_histogram_against_binned_continuous_niche ---------------------------')
        filename = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_merged_kmeans_data.pickle'
        q = self.des_stats / filename
        self.ss_data_dict = pickle.load(open(q, 'rb'))
        for im in ['original', 'imputed']:
            for k in self.sub_sample_d:
                df = self.ss_data_dict[im][k].copy()
                print(k)
                for y_var in self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]:
                    x_var_m = 'continuous_niche_' + m
                    y_var_m = y_var + '_' + m
                    for CAT in self.sub_sample_graph_cat_vars_d[k]:
                        print('*************************** start graphing niche cut ' + y_var_m + ' against ' + CAT + ' *****************************')
                        var_cols = [x_var_m, y_var_m] + [CAT]
                        df2 = df[var_cols]
                        # bin continuous niche
                        df2['continuous_niche_cut'] = pd.cut(df2[x_var_m], bins=bin_range)
                        print(df2.continuous_niche_cut.unique())
                        if y_var in self.core_dummy_y_vars_d[im]:
                            self._y_countplot_against_binned_continuous_niche(
                                balanced=balanced, im=im, m=m, k=k, y_var=y_var, y_var_m=y_var_m, df=df2, CAT=CAT,
                                picked_tfidf_param=picked_tfidf_param, picked_k=picked_k)
                        else:
                            self._y_boxplot_against_binned_continuous_niche(
                                balanced=balanced, im=im, m=m, k=k, y_var=y_var, y_var_m=y_var_m, df=df2, CAT=CAT,
                                picked_tfidf_param=picked_tfidf_param, picked_k=picked_k)
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            ss_data_dict=self.ss_data_dict,
            reg_results=self.reg_results)

    def heatmap_among_variables(self, balanced, picked_tfidf_param, picked_k, month):
        print('*************************** heatmap_among_variables *************************** ')
        filename = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_merged_kmeans_data.pickle'
        q = self.des_stats / filename
        self.ss_data_dict = pickle.load(open(q, 'rb'))
        for im in ['original', 'imputed']:
            for k, s in self.sub_sample_d.items():
                print(k)
                df = self.ss_data_dict[im][k].copy()
                all_ys = self.core_dummy_y_vars_d[im] + self.core_scaled_continuous_y_vars_d[im]
                # get short y labels
                all_ys_labels = [self.short_y_label[im][i] for i in all_ys]
                all_ys_m = [i + '_' + month for i in all_ys]
                all_xs = self.core_scaled_control_vars[im] + self.niche_vars
                # get short x labels
                all_xs_labels = [self.short_x_label[im][i] for i in all_xs]
                all_xs_m = [i + '_' + month for i in all_xs]
                var_d = {'y_vars': all_ys_m, 'x_vars': all_xs_m}
                var_labels_d = {'y_vars': all_ys_labels, 'x_vars': all_xs_labels}
                for name, var_ls in var_d.items():
                    df2 = df[var_ls]
                    df3 = df2.corr()
                    print(list(df2.columns))
                    print(df3)
                    mask = np.triu(np.ones_like(df3.corr(), dtype=np.bool))
                    print(mask)
                    fig, axes = plt.subplots(nrows=1,
                                             ncols=1,
                                             figsize=(10, 10))
                    sns.set_style("whitegrid")
                    sns.heatmap(data=df3, xticklabels=var_labels_d[name], yticklabels=var_labels_d[name],
                                vmin=-1, vmax=1, annot=True,
                                mask=mask,
                                cmap='BrBG',
                                ax=axes)
                    f_name = k + '_' + month + '_' + picked_tfidf_param + '_' + str(picked_k) + '_' + name + '_heatmap.png'
                    q = self.des_stats_graphs / balanced / im / f_name
                    fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return stats_and_regs(
            initial_panel=self.initial_panel,
            all_panels=self.all_panels,
            df=self.df,
            ss_data_dict=self.ss_data_dict,
            reg_results=self.reg_results)

    def _check_cross_sectional_ols_assumptions(self, balanced, im, niche_v, k, y, sms_results, writer, multicolinearity_df):
        """
        :param sms_results: statsmodel results object
        :return:
        """
        # normality of residual --------------------------------------------------------------
        test = sms.jarque_bera(sms_results.resid)
        test = lzip(self.jb_test_names, test) # this is a list of tuples
        test_df = pd.DataFrame(test, columns =['test_statistics', 'value'])
        test_df.to_excel(writer, sheet_name=im + '_' + k + '_' + y + '_normality_of_residual')
        # multi-collinearity -----------------------------------------------------------------
        test = np.linalg.cond(sms_results.model.exog)
        multicolinearity_df = multicolinearity_df.append({im + '_' + k: test}, ignore_index=True)
        # heteroskedasticity Breush-Pagan test -------------------------------------------------
        test = sms.het_breuschpagan(sms_results.resid, sms_results.model.exog)
        test = lzip(self.bp_test_names, test)
        test_df = pd.DataFrame(test, columns =['test_statistics', 'value'])
        test_df.to_excel(writer, sheet_name=im + '_' + k + '_' + y + '_hetero_bp_test')
        # linearity Harvey-Collier -------------------------------------------------------------
        # this test not seem to work with my dataset because it raises singular matrix error message
        # I guess for the dummy_niche regressor, the relationship is not a linear one
        # I will visually plot the y variables against x variables to check linearity
        return multicolinearity_df

    def _create_forward_step_variables(self):
        controls =  {'original': ['score', 'nlog_reviews', 'adultcontent', 'daysreleased', 'size'],
                     'imputed': ['imputed_score', 'nlog_imputed_reviews', 'imputed_adultcontent', 'imputed_daysreleased', 'imputed_size']}
        stepwise_controls = {'original': {}, 'imputed': {}}
        for name in ['original', 'imputed']:
            for n in [0, 1, 2, 3, 4, 5]:
                temp = list(combinations(controls[name], n))
                stepwise_controls[name][n] = [list(ele) for ele in temp]
        print('---------------------stepwise controls-------------------')
        print(stepwise_controls)
        return stepwise_controls

    def select_model_according_to_aic_and_bic(self, month, picked_tfidf_param, picked_k):
        # locale.setlocale(locale.LC_NUMERIC, 'English')
        f_ols = month + '_' + picked_tfidf_param + '_' + str(picked_k) + '_AIC_BIC.xlsx'
        q_ols = self.ols_results / 'model_selection' / f_ols
        df = pd.read_excel(q_ols)
        # filter the lowest AIC and BIC within each step for each y variable, and keep other columns
        min_aicbic_df = df.sort_values(['AIC', 'BIC']).groupby(['im', 'k', 'y', 'step'], as_index=False).first()
        f_ols = month + '_' + picked_tfidf_param + '_' + str(picked_k) + '_minAICBIC.xlsx'
        q_ols = self.ols_results / 'model_selection' / f_ols
        min_aicbic_df.to_excel(q_ols)
        # create dissertation tables
        for ss in ['FULL', 'MF', 'ML']:
            f_ols = month + '_' + ss + '_' + '_best_models.xlsx'
            q_ols = self.ols_results / 'model_selection' / f_ols
            y_vars = self.imputed_y_vars[ss]
            numeric_cols = [p + '_' + y + '_' + month for y in y_vars for p in ['AIC', 'BIC']]
            model_col = ['CONTROLS_' + y + '_' + month for y in y_vars]
            newdf = pd.DataFrame(columns=['MODEL-STEP'] + model_col + numeric_cols)
            newdf['MODEL-STEP'] = [0, 1, 2, 3, 4, 5]
            for y in [s + '_' + month for s in y_vars]:
                df2 = min_aicbic_df.loc[(min_aicbic_df['y'] == y) & (min_aicbic_df['k'] == ss)].copy()
                for step in [0, 1, 2, 3, 4, 5]:
                    df3 = df2.loc[df2['step'] == step].copy()
                    newdf.loc[newdf['MODEL-STEP'] == step, 'CONTROLS_' + y] = df3['control_vars'].values[0]
                    newdf.loc[newdf['MODEL-STEP'] == step, 'AIC_' + y] = df3['AIC'].values[0]
                    newdf.loc[newdf['MODEL-STEP'] == step, 'BIC_' + y] = df3['BIC'].values[0]
            newdf[numeric_cols] = newdf[numeric_cols].apply(pd.to_numeric)
            newdf[numeric_cols] = np.round(newdf[numeric_cols], 0)
            # since the MF sub-sample does not contain any to_tier_1 data points by definition
            # the estimated AIC and BIC will be -inf, so I will first replace them with NA
            newdf.replace([np.inf, -np.inf], 0, inplace=True)
            newdf[numeric_cols] = newdf[numeric_cols].fillna(0).astype(np.int64)
            newdf[numeric_cols] = newdf[numeric_cols].apply(lambda series: series.apply(lambda value: '&' + f'{value:,}'))
            newdf.to_excel(q_ols)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def _create_cat_interactions(self, df, cats, month):
        for i in cats:
            n = 'continuous_niche_X_' + i + '_' + month
            df[n] = df['continuous_niche_' + month] * df[i]
        return df

    def _add_astericks_to_pvalue(self, pvalue, coef):
        if abs(pvalue) <= 0.01:
            return '&' + str(coef) + '***'
        elif abs(pvalue) <= 0.05:
            return '&' + str(coef) + '**'
        elif abs(pvalue) <= 0.1:
            return '&' + str(coef) + '*'
        else:
            return '&' + str(coef)

    def _extract_niche_estimates(self, smmodel, panel, sample=None, cat=False, col=None, month=None):
        table_as_html = smmodel.summary().tables[1].as_html()
        resultobject = pd.read_html(table_as_html, header=0, index_col=0)[0]
        print(resultobject)
        if panel is True:
            slice_col = col
        else:
            if cat is False:
                slice_col = 'continuous_niche_' + month
            else:
                if col in self.sub_sample_dummies[sample]:
                    slice_col = col
                else:
                    slice_col = col + '_' + month
        niche_coef = np.round(resultobject.at[slice_col, 'coef'].squeeze(), 2)
        niche_std_err = np.round(resultobject.at[slice_col, 'std err'].squeeze(), 2)
        niche_std_err_str = '&(' + str(niche_std_err) + ')'
        niche_p_value = np.round(resultobject.at[slice_col, 'P>|t|'].squeeze(), 2)
        niche_coef_str = self._add_astericks_to_pvalue(pvalue=niche_p_value, coef=niche_coef)
        return niche_coef_str, niche_std_err_str

    def OLS_after_AIC_BIC_selection(self, picked_tfidf_param, picked_k, month):
        print('----------------------------- OLS_after_AIC_BIC_selection ---------------------------------')
        # one set of regression use continuous niche (or interaction if stacked is True) as independent var, the other set uses dummy niche
        f_name = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_merged_kmeans_data.pickle'
        q = self.des_stats / f_name
        self.ss_data_dict = pickle.load(open(q, 'rb'))
        for k in self.sub_sample_d:
            ys = self.imputed_y_vars[k]
            df = self.ss_data_dict['imputed'][k].copy()
            ldf = self._convert_to_long(df=df, sample=k, chosen_step=self.chosen_step_for_reg[k])
            print(list(df.columns))
            index_names = []
            for y in ys:
                std_err = y + '_std_err'
                index_names.append(y)
                index_names.append(std_err)
            panel_df_cols = ['continuous_niche'] + ['period_' + str(i) + '_continuous_niche' for i in [0,1,2,3]]
            niche_and_cat_interactions = ['continuous_niche_X_' + i for i in self.sub_sample_dummies[k]]
            cat_df_cols = ['continuous_niche'] + niche_and_cat_interactions + self.sub_sample_dummies[k]
            step_ols_cols = [0, 1, 2, 3, 4, 5]
            cdf = self._create_cat_interactions(df=df, cats=self.sub_sample_dummies[k], month=month)
            olsres = pd.DataFrame(columns=step_ols_cols, index=index_names)
            olscatres = pd.DataFrame(columns=cat_df_cols, index=index_names)
            panelres = pd.DataFrame(columns=panel_df_cols, index=index_names)
            for y in ys:
                # ----------- OLS with categories (only need the chosen STEP with categorical dummy interactions)------
                y_m = y + '_' + month
                xs = self.step_models_imputed_control_vars[k][y][self.chosen_step_for_reg[k]] + [
                    'continuous_niche'] + niche_and_cat_interactions
                xs_m = [i + '_' + month for i in xs] + self.sub_sample_dummies[k] + self.unstacked_ols_categorical_control_vars[k]
                xs_m = list(set(xs_m))
                o_x_str = ' + '.join(xs_m)
                formula = y_m + ' ~ ' + o_x_str
                print(y + ' -- ' + k + '-- formula --')
                print(formula)
                olscatmod = smf.ols(formula, data=cdf).fit()  # this is OLS regression result object
                for col in cat_df_cols:
                    olscatres.at[y, col] = self._extract_niche_estimates(
                        smmodel=olscatmod, panel=False, sample=k, cat=True, col=col, month='202107')[0]
                    olscatres.at[y + '_std_err', col] = self._extract_niche_estimates(
                        smmodel=olscatmod, panel=False, sample=k, cat=True, col=col, month='202107')[1]
                # ----------- Pooled OLS PANEL reg (only need the chosen STEP with time dummy interactions) -----
                ydf = ldf[[y]]
                time_dummies_and_interactions = [i for i in list(ldf.columns) if 'period_' in i]
                xs = ['continuous_niche'] + time_dummies_and_interactions + \
                     self.step_models_imputed_control_vars[k][y][self.chosen_step_for_reg[k]] + \
                     self.unstacked_ols_categorical_control_vars[k]
                xsdf = sm.add_constant(ldf[xs])
                print(k + '---- fit model with ' + y + ' ~ ' + ' + '.join(xs))
                mod = sm.OLS(endog=ydf, exog=xsdf)
                panelmod = mod.fit()
                for col in panel_df_cols:
                    panelres.at[y, col] = self._extract_niche_estimates(smmodel=panelmod, panel=True, sample=k, col=col)[0]
                    panelres.at[y + '_std_err', col] = \
                    self._extract_niche_estimates(smmodel=panelmod, panel=True, sample=k, col=col)[1]
                # ----------- OLS reg (need to get estimates for all STEP models to compare) --------------------
                for step in [0, 1, 2, 3, 4, 5]:
                    xs = self.step_models_imputed_control_vars[k][y][step] + ['continuous_niche']
                    xs_m = [i + '_' + month for i in xs] + self.unstacked_ols_categorical_control_vars[k]
                    o_x_str = ' + '.join(xs_m)
                    formula = y_m + ' ~ ' + o_x_str
                    print(y + ' -- ' + k + '-- formula --')
                    print(formula)
                    olsmod = smf.ols(formula, data=df).fit() # this is OLS regression result object
                    olsres.at[y, step] = self._extract_niche_estimates(smmodel=olsmod, panel=False, sample=k, month='202107')[0]
                    olsres.at[y + '_std_err', step] = self._extract_niche_estimates(smmodel=olsmod, panel=False, sample=k, month='202107')[1]
            f_ols = month + '_' + picked_tfidf_param + '_' + str(
                picked_k) + '_IMPUTED_' + k + '_BEST_FIT_STEP_REGMODELS.xlsx'
            q_ols = self.ols_results / 'ols_aic_bic_best_fit_step_regs' / f_ols
            olsres.to_excel(q_ols)
            f_ols = month + '_' + picked_tfidf_param + '_' + str(
                picked_k) + '_IMPUTED_' + k + '_BEST_FIT_CAT_OLS.xlsx'
            q_ols = self.ols_results / 'ols_aic_bic_best_fit_step_regs' / f_ols
            olscatres.to_excel(q_ols)
            f_ols = month + '_' + picked_tfidf_param + '_' + str(
                picked_k) + '_IMPUTED_' + k + '_BEST_FIT_PANEL.xlsx'
            q_ols = self.ols_results / 'ols_aic_bic_best_fit_step_regs' / f_ols
            panelres.to_excel(q_ols)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)


    def _create_time_dummies_and_interactions(self, df):
        """
        https://www.littler.com/publication-press/publication/stay-top-stay-home-list-statewide
        https://www.usatoday.com/storytelling/coronavirus-reopening-america-map/
        :param df:
        :return:
        The event study time dummy is 1 only for that period after stay-at-home order in Mar 2020.
        What I'm suggesting is turning your diff-in-diff into an event study.
        Leah's email 2022 Apr 28.
        You were estimating
        outcome_i,t = beta_0 + beta_1 treatment_i + beta_2 after_t + beta_3 treatment_i * after_t + e_i,t  (1)

        Assume there are two post-treatment periods after1 and after2 and that you have indicator variables for each period with the same name. You could instead estimate
        outcome_i,t = beta_0 + beta_1 treatment_i + beta_2 after1_t + beta_3 after2_t + beta_4 treatment_i * after1_t + beta_5 treatment_i * after2_t + e_i,t (2)

        This breaks up the beta_3 term from (1) into 2 parts in (2).  beta_3 in (1) is the weighted average of beta_4 and beta_5.

        You could be even more flexible and include indicators for each period, even before treatment.
        This allows you to assess whether there are pre-treatment changes in the treated group relative to the control.

        Since too many time dummies affect the interpretation, I am going to divide time dummies into 4-periods intervals, and the pre-covid periods are not represented by time dummies to prevent multi-collinearity
        """
        print('----------------------------- _create_time_dummies_and_interactions ---------------------------------')
        # the baseline (pre-covid-stay at home) has already been left out
        period_0 = ['202003', '202004']
        period_1 = ['202009', '202010', '202011', '202012']
        period_2 = ['202101', '202102', '202103', '202104']
        period_3 = ['202105', '202106', '202107']
        periods_after_covid = {'period_0': period_0,
                               'period_1': period_1,
                               'period_2': period_2,
                               'period_3': period_3}
        # print(periods_after_events)
        for time_dummy_name, periods in periods_after_covid.items():
            df[time_dummy_name] = df.apply(lambda row: 1 if row['month'] in periods else 0, axis=1)
            df[time_dummy_name + '_continuous_niche'] = df[time_dummy_name] * df['continuous_niche']
            # print(df[[time_dummy_name, interaction]].describe())
        return df

    def summarize_ols_results(self, balanced, stacked, picked_tfidf_param, picked_k, month):
        """
        :param balanced:
        :return:
        """
        print('----------------------------- summarize_ols_results ---------------------------------')
        ys = ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'imputed_offersIAPdummy', 'imputed_containsAdsdummy',
              'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA']
        if stacked is True:
            stack = '_STACKED_'
            res = pd.DataFrame(columns=['FULL', 'FULL_X_ML', 'ML', 'ML_X_ML_GAME', 'ML_X_ML_SOCIAL', 'ML_X_ML_BUSINESS', 'ML_X_ML_MEDICAL',
                                        'MF', 'MF_X_MF_GAME', 'MF_X_MF_SOCIAL', 'MF_X_MF_BUSINESS', 'MF_X_MF_MEDICAL'], index=ys)
            # FULL indicate regression on niche in the full sample ML indicates regression on niche in market leading sample and so on
            # _X_ indicates the interaction variable between niche and indicator
            suffixes = {'FULL': ['', '_X_ML'],
                        'ML':  ['', '_X_ML_GAME', '_X_ML_SOCIAL', '_X_ML_BUSINESS', '_X_ML_MEDICAL'],
                        'MF':  ['', '_X_MF_GAME', '_X_MF_SOCIAL', '_X_MF_BUSINESS', '_X_MF_MEDICAL']}
        else:
            stack = ''
            res = pd.DataFrame(columns=['FULL', 'ML', 'MF'], index=ys)
            suffixes = {'FULL': [''], 'ML':  [''], 'MF':  ['']}
            # FULL indicate regression on niche in the full sample ML indicates regression on niche in market leading sample and so on
        f_ols = month + '_' + picked_tfidf_param + '_' + str(picked_k) + stack + '_OLS_RESULTS.xlsx'
        q_ols = self.ols_results / 'ols_raw_results' / balanced / f_ols
        # -----------------------------------------------------------------------
        for k in ['FULL', 'ML', 'MF']:
            for y in ys:
                y_var = 'imputed_' + k + '_' + y + '_' + month
                df = pd.read_excel(q_ols, sheet_name=y_var)
                # print(df)
                for suffix in suffixes[k]:
                    niche_coef = np.round(df.loc[df['Unnamed: 0']=='continuous_niche' + suffix + '_' + month, 'coef'].squeeze(), 2)
                    p_value = df.loc[df['Unnamed: 0']=='continuous_niche' + suffix + '_' + month, 'P>|t|'].squeeze()
                    if abs(p_value) <= 0.01:
                        niche_str = '&' + str(niche_coef) + '***'
                    elif abs(p_value) <= 0.05:
                        niche_str = '&' + str(niche_coef) + '**'
                    elif abs(p_value) <= 0.1:
                        niche_str = '&' + str(niche_coef) + '*'
                    else:
                        niche_str = '&' + str(niche_coef)
                    col_name = k + suffix
                    res.at[y, col_name] = niche_str
        f_ols_res = month + '_' + picked_tfidf_param + '_' + str(picked_k) + stack + '_OLS_SUMMARY.xlsx'
        q_ols_res = self.des_stats_tables / balanced / 'table_in_dissertation' / f_ols_res
        res.to_excel(q_ols_res)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def summarize_panel_results(self, balanced, type, picked_tfidf_param, picked_k):
        print('----------------------------- summarize_panel_results ---------------------------------')
        ys = ['nlog_imputed_price', 'nlog_imputed_minInstalls', 'imputed_offersIAPdummy', 'imputed_containsAdsdummy',
              'noisy_death', 'T_TO_TIER1_minInstalls', 'T_TO_top_firm', 'MA']
        f_panel = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_PANEL_RESULTS.xlsx'
        folder_name = type + '_raw_results'
        q_panel = self.panel_results / folder_name / balanced / f_panel
        coef_vars = ['continuous_niche', 'period_0_continuous_niche', 'period_1_continuous_niche',
                     'period_2_continuous_niche', 'period_3_continuous_niche']
        res = pd.DataFrame(
            columns=[k + '_' + i for k in ['FULL', 'ML', 'MF'] for i in coef_vars], index=ys)
        # -----------------------------------------------------------------------
        for k in ['FULL', 'ML', 'MF']:
            for y in ys:
                if y == 'T_TO_TIER1_minInstalls' and k == 'MF':
                    pass
                else:
                    y_var = 'imputed_' + k + '_' + y
                    df = pd.read_excel(q_panel, sheet_name=y_var)
                    for i in coef_vars:
                        niche_coef = np.round(df.loc[df['Unnamed: 0'] == i, 'parameter'].squeeze(), 2)
                        p_value = df.loc[df['Unnamed: 0'] == i, 'pvalue'].squeeze()
                        if abs(p_value) <= 0.01:
                            niche_str = '&' + str(niche_coef) + '***'
                        elif abs(p_value) <= 0.05:
                            niche_str = '&' + str(niche_coef) + '**'
                        elif abs(p_value) <= 0.1:
                            niche_str = '&' + str(niche_coef) + '*'
                        else:
                            niche_str = '&' + str(niche_coef)
                        res.at[y, k + '_' + i] = niche_str
        f_panel_res = self.initial_panel + '_' + picked_tfidf_param + '_' + str(picked_k) + '_PANEL_SUMMARY.xlsx'
        q_panel_res = self.des_stats_tables / balanced / 'table_in_dissertation' / f_panel_res
        res.to_excel(q_panel_res)
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)

    def summarize_ols_results_in_graph(self, balanced):
        """
        should first run self.summarize_ols_results
        :param balanced:
        :return:
        """
        print('----------------------------- summarize_ols_results_in_graph ---------------------------------')
        for niche_v in self.niche_vars:
            for im in ['original', 'imputed']:
                imputed_y_core_vars = self.core_scaled_continuous_y_vars_d[im] + self.core_dummy_y_vars_d[im]
                f_name = b + '_' + niche_v + '_' + im + '_ols_cross_sectional_results.csv'
                df = pd.read_csv(self.ols_results / 'summary_results' / f_name)
                df.reset_index(inplace=True)
                df.rename(columns={
                    'Unnamed: 0': 'Month',
                    'Unnamed: 1': 'Sub-sample'}, inplace=True)
                for k, s in self.sub_sample_d.items():
                    ss_list = list(s.keys())
                    fig, axes = plt.subplots(nrows=len(imputed_y_core_vars),
                                             ncols=1,
                                             figsize=(10, 24),
                                             sharey='row', sharex='col')
                    sns.set_style("whitegrid")
                    for i in range(len(imputed_y_core_vars)):
                        y = imputed_y_core_vars[i]
                        df2 = df.loc[df['Sub-sample'].isin(ss_list), ['Month', 'Sub-sample', y]].copy()
                        df2[y] = df2[y].apply(lambda x: x.replace('*', '') if isinstance(x, str) else x)
                        df2[y] = pd.to_numeric(df2[y])
                        df2['Month'] = pd.to_datetime(df2['Month'], format='%Y%m')
                        df2 = df2.sort_values(by=['Month'])
                        sns.lineplot(data=df2, x='Month', y=y, hue='Sub-sample', ax=axes[i])
                        axes[i].set(xticks=df2.Month.values)
                        axes[i].xaxis.set_major_formatter(dates.DateFormatter("%Y%m"))
                        axes[i].tick_params(axis='x', labelrotation=90)
                        handles, labels = axes[i].get_legend_handles_labels()
                        axes[i].get_legend().remove()
                    fig.legend(handles, labels, loc='lower left', ncol=1)
                    f_name = k + '_beta_time_trend.png'
                    q = self.ols_results / 'ols_beta_graph' / balanced / im / niche_v / f_name
                    fig.savefig(q, facecolor='w', edgecolor='w', dpi=300, bbox_inches='tight')
        return stats_and_regs(
                        initial_panel=self.initial_panel,
                        all_panels=self.all_panels,
                        df=self.df,
                        ss_data_dict=self.ss_data_dict,
                        reg_results = self.reg_results)





    # ==================================================================================================
    # COMPARE the coefficient from different samples
    # ==================================================================================================
    """
    Apr 2, 2022
    https://www.theanalysisfactor.com/compare-regression-coefficients/
    Simply include an interaction term between Sex (male/female) and any predictor whose coefficient you want to compare.  
    If you want to compare all of them because you believe that all predictors have different effects for men and women, 
    then include an interaction term between sex and each predictor.  If you have 6 predictors, that means 6 interaction terms.
    In such a model, if Sex is a dummy variable (and it should be), two things happen:
    1.the coefficient for each predictor becomes the coefficient for that variable ONLY for the reference group.
    2. the interaction term between sex and each predictor represents the DIFFERENCE in the coefficients between 
    the reference group and the comparison group.  If you want to know the coefficient for the comparison group, 
    you have to add the coefficients for the predictor alone and that predictor’s interaction with Sex.
    The beauty of this approach is that the p-value for each interaction term gives you a significance 
    test for the difference in those coefficients.
    """



