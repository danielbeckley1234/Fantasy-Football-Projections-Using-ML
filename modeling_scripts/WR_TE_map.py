WRTE_targets = ['TGT', 'REC', 'RecYDS', 'RecTD', 'ATT', 'RusYDS', 'RusTD']
WRTE_counting = WRTE_targets + ['RZREC', 'RZTGT', 'RZATT', 'Routes', 'CATCHABLE_TGT', 'Drops']
WRTE_lag1s = ['RecTD%', 'RuTD%', 'xRecTD%', 'xRuTD%', 'significant_injury', 'Snaps/G',
              'RUSH %', 'TGT %', 'TOUCH %', 'YPRR', 'Open', 'Catch', 'YAC', 'Overall',
              'CUSH', 'SEP', 'TAY', 'TAY%', 'CTCH%', 'YAC/R', 'xYAC/R', '+/-', 'Routes/G', 'CATCHABLE_TGT/G']
WRTE_lag2s = ['RZREC/G', 'RZTGT/G', 'RZTGT%', 'Drops/G']

WRTE_cross_features = ['Snaps/G', 'Snaps/G_lag1', 'significant_injury_lag1', 
    'draft_round_filled', 'draft_pick_filled', 'age', 'age_sq', 
    'years', 'inflated_apy', 'inflated_guaranteed', 'pct_gtd_sign', 
    'years_exp', 'years_since_prod', 'years_since_full', 'team_change', 'games_missed_rate', 'pct_peak_vol']

WRTE_rec_base_features = ['TGT/G', 'TGT/G_lag1', 'TGT/G_lag2', 'TGT/G_lag3', 'TGT/G_std3',
    'REC/G', 'REC/G_lag1', 'REC/G_lag2', 'REC/G_lag3', 'CATCHABLE_TGT/G', 'CATCHABLE_TGT/G_lag1',
    'RecYDS/G', 'RecYDS/G_lag1', 'RecYDS/G_lag2', 'RecYDS/G_lag3', 'Drops/G', 'Drops/G_lag1', 'Drops/G_lag2',
    'Routes/G', 'Routes/G_lag1', 'Open', 'Open_lag1', 'Catch', 'Catch_lag1', 
    'YAC', 'YAC_lag1', 'Overall', 'Overall_lag1', 
    'TGT %', 'TGT %_lag1', 'YPRR', 'YPRR_lag1',
    'CUSH', 'CUSH_lag1', 'SEP', 'SEP_lag1', 'YAC/R', 'YAC/R_lag1',
    'TAY', 'TAY_lag1', 'TAY%', 'TAY%_lag1', 'CTCH%', 'CTCH%_lag1', 'xYAC/R', 'xYAC/R_lag1', '+/-', '+/-_lag1']

WRTE_rush_base_features = ['ATT/G', 'ATT/G_lag1', 'RusYDS/G', 'RusYDS/G_lag1', 'RUSH %', 'RUSH %_lag1']

WR_TE_target_feature_map = {
    'REC': WRTE_cross_features + WRTE_rec_base_features,
    'TGT': WRTE_cross_features + WRTE_rec_base_features,
    'RecYDS': WRTE_cross_features + WRTE_rec_base_features,
    'RecTD': ['RecTD/G', 'RecTD/G_lag1', 'RecTD/G_lag2', 'RecTD/G_lag3',
        'RecTD%', 'RecTD%_lag1', 'xRecTD%', 'xRecTD%_lag1', 
        'RZREC/G', 'RZREC/G_lag1', 'RZREC/G_lag2', 'RZTGT/G', 'RZTGT/G_lag1', 
        'RZTGT/G_lag2', 'RZTGT%', 'RZTGT%_lag1'] + WRTE_cross_features + WRTE_rec_base_features,
    'ATT': WRTE_cross_features + WRTE_rush_base_features,
    'RusYDS': WRTE_cross_features + WRTE_rush_base_features,
    'RusTD': ['RusTD/G', 'RusTD/G_lag1', 'RZATT/G',
        'RuTD%', 'RuTD%_lag1', 'xRuTD%', 'xRuTD%_lag1'] + WRTE_cross_features + WRTE_rush_base_features,
}

WRTE_target_cols = [f'target_{c}/G' for c in WRTE_targets]