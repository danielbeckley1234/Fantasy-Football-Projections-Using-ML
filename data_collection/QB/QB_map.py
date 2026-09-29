QB_targets = ['CMP', 'pasATT', 'pasYDS', 'pasTD', 'INT', 'ATT', 'rusYDS', 'rusTD']
QB_counting = QB_targets + ['SACKS', 'RZCMP', 'RZPasATT', 'RZPasTD', 
    'RZRusYDS', 'RZRusTD', 'HRRY', 'POOR', 'DROP']
QB_lag1s = ['CMP%', 'Y/A', 'PaTD%', 'xPaTD%', 'RuTD%', 'xRuTD%',
    'RZCMP/G', 'RZPasATT/G', 'RZCMP%', 'significant_injury', 'DROP/G',
    'HRRY/G', 'POOR/G', 'CAY', 'IAY', 'AYD', 'AGG%', 'xCMP%', '+/-']
QB_lag2s = ['RZATT/G', 'RZRusTD/G', 'SACKS/G']

QB_cross_features = ['significant_injury_lag1', 'draft_round_filled', 'draft_pick_filled', 'age', 'age_sq', 
    'years', 'inflated_apy', 'inflated_guaranteed', 'pct_gtd_sign', 
    'years_exp', 'years_since_prod', 'years_since_full', 'team_change', 'games_missed_rate', 'pct_peak_vol']

QB_pass_base_features = ['CMP/G', 'CMP/G_lag1', 'CMP/G_lag2', 'CMP/G_lag3',
    'pasATT/G', 'pasATT/G_lag1', 'pasATT/G_lag2', 'pasATT/G_lag3',
    'pasYDS/G', 'pasYDS/G_lag1', 'pasYDS/G_lag2', 'pasYDS/G_lag3',
    'INT/G', 'INT/G_lag1', 'INT/G_lag2', 'INT/G_lag3', 
    'CMP%', 'CMP%_lag1', 'SACKS/G', 'SACKS/G_lag1', 'SACKS/G_lag2', 'HRRY/G', 'HRRY/G_lag1', 
    'POOR/G', 'POOR/G_lag1', 'DROP/G', 'DROP/G_lag1', 'DROP/G_lag2', 'AGG%', 'xCMP%', '+/-']

QB_rush_base_features = ['ATT/G', 'ATT/G_lag1', 'rusYDS/G', 'rusYDS/G_lag1']

QB_target_feature_map = {
    'CMP': QB_cross_features + QB_pass_base_features + ['CAY'],
    'pasATT': QB_cross_features + QB_pass_base_features + ['IAY', 'AYD'],
    'pasTD': ['PaTD%', 'PaTD%_lag1', 'xPaTD%', 'xPaTD%_lag1', 'RZCMP/G', 'RZCMP/G_lag1',
              'RZPasATT/G', 'RZPasATT/G_lag1', 'RZCMP%', 'RZCMP%_lag1'] + QB_cross_features + QB_pass_base_features,
    'INT': QB_cross_features + QB_pass_base_features,
    'ATT': QB_cross_features + QB_rush_base_features,
    'rusYDS': QB_cross_features + QB_rush_base_features,
    'rusTD': QB_cross_features + QB_rush_base_features + ['RuTD%', 'RuTD%_lag1', 'xRuTD%', 'xRuTD%_lag1'],
}

QB_target_cols = [f'target_{c}/G' for c in QB_targets]