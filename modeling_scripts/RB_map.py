import pandas as pd
import numpy as np

RB_targets = ['ATT', 'RusYDS', 'RusTD', 'TGT', 'REC', 'RecYDS', 'RecTD']
RB_counting = RB_targets + ['Snaps', 'Routes', 'In5ATT', 'In5REC', 'In5TG']
RB_lag1s = ['Snaps/G', 'significant_injury_lag1','RusY/A', 'RYOE/ATT', 'RYOE%', 'Rush%', 'Routes/G', 'Open',
            'Catch', 'YAC', 'Overall', 'Tgt%', 'Y/RR', 'RusTD%', 'xRusTD%',]
RB_lag2s = ['In5ATT', 'In5Rush%', 'In5REC/G', 'In5TG/G']

RB_cross_features = ['Snaps/G', 'Snaps/G_lag1', 'significant_injury_lag1', 
    'draft_round_filled', 'draft_pick_filled', 'age', 'age_sq', 
    'years', 'inflated_apy', 'inflated_guaranteed', 'pct_gtd_sign', 
    'years_exp', 'years_since_full', 'team_change', 'games_missed_rate']

# for att/rusyds
RB_rush_base_features = ['ATT/G', 'ATT/G_lag1', 'ATT/G_lag2', 'ATT/G_lag3', 'ATT/G_std3',
    'RusYDS/G', 'RusYDS/G_lag1', 'RusYDS/G_lag2', 'RusYDS/G_lag3',
    'RusY/A', 'RusY/A_lag1', 'TmRBWR', 'RYOE/ATT', 'RYOE/ATT_lag1',
    'RYOE%', 'RYOE%_lag1', 'Rush%', 'Rush%_lag1', 'pct_peak_att']

# for all receiving statistics
RB_rec_base_features = ['TGT/G', 'TGT/G_lag1', 'TGT/G_lag2', 'TGT/G_lag3', 'TGT/G_std3',
    'REC/G', 'REC/G_lag1', 'REC/G_lag2', 'REC/G_lag3',
    'RecYDS/G', 'RecYDS/G_lag1', 'RecYDS/G_lag2', 'RecYDS/G_lag3',
    'Routes/G', 'Routes/G_lag1', 'Open', 'Open_lag1', 'Catch', 'Catch_lag1', 
    'YAC', 'YAC_lag1', 'Overall', 'Overall_lag1',
    'Tgt%', 'Tgt%_lag1', 'pct_peak_tgt', 'Y/RR', 'Y/RR_lag1']

RB_target_feature_map = {
    'ATT': RB_cross_features + RB_rush_base_features,
    'RusYDS': RB_cross_features + RB_rush_base_features,
    'RusTD': ['RusTD/G', 'RusTD/G_lag1', 'RusTD/G_lag2', 'RusTD/G_lag3', 
        'In5ATT/G', 'In5ATT/G_lag1', 'In5ATT/G_lag2', 'In5Rush%', 'In5Rush%_lag1', 'In5Rush%_lag2',
        'RusTD%', 'RusTD%_lag1', 'xRusTD%', 'xRusTD%_lag1'] + RB_cross_features + RB_rush_base_features,
    'TGT': RB_cross_features + RB_rec_base_features,
    'REC': RB_cross_features + RB_rec_base_features,
    'RecYDS': RB_cross_features + RB_rec_base_features,
    'RecTD': ['RecTD/G', 'RecTD/G_lag1', 'RecTD/G_lag2', 'RecTD/G_lag3',
        'RecTD%', 'RecTD%_lag1', 'xRecTD%', 'xRecTD%_lag1', 
        'In5REC/G', 'In5REC/G_lag1', 'In5REC/G_lag2', 
        'In5TG/G', 'In5TG/G_lag1', 'In5TG/G_lag2'] + RB_cross_features + RB_rec_base_features
}

target_cols = [f'target_{c}/G' for c in RB_targets]
