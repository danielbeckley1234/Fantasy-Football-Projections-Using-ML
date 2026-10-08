QB_targets = ['CMP', 'pasATT', 'pasYDS', 'pasTD', 'INT', 'ATT', 'rusYDS', 'rusTD']
QB_counting = QB_targets + ['SACKS', 'HRRY', 'POOR', 'DROP',
    'RZATT', 'RZRusYDS', 'RZRusTD', 'RZCMP', 'RZPasATT', 'RZPasTD', 'RZINT']
QB_lag1s = ['CMP%', 'xCMP%', '+/-', 'Y/A', 'CAY', 'IAY', 'AYD', 'AGG%',
    'PaTD%', 'xPaTD%', 'RuTD%', 'xRuTD%', 'significant_injury',
    'RZCMP/G', 'RZPasATT/G', 'RZCMP%', 'SACKS/G', 'HRRY/G', 'POOR/G']
QB_lag2s = ['RZATT/G', 'RZRusTD/G', 'RZINT/G', 'SACKS/G', 'DROP/G', 'INT%']

QB_cross_features = ['significant_injury_lag1', 'draft_round_filled', 'draft_pick_filled', 
    'age', 'age_sq', 'years', 'inflated_apy', 'inflated_guaranteed', 'pct_gtd_sign', 
    'years_exp', 'years_since_prod', 'years_since_full', 'team_change', 'pct_peak_vol']

QB_pass_base_features = ['CMP/G', 'CMP/G_lag1', 'CMP/G_lag2', 'CMP/G_lag3',
    'pasATT/G', 'pasATT/G_lag1', 'pasATT/G_lag2', 'pasATT/G_lag3', 'pasATT/G_std3', 
    'pasYDS/G', 'pasYDS/G_lag1', 'pasYDS/G_lag2', 'pasYDS/G_lag3', 'AGG%',
    'SACKS/G', 'SACKS/G_lag1', 'SACKS/G_lag2', 'HRRY/G', 'HRRY/G_lag1', 
    'POOR/G', 'POOR/G_lag1', 'DROP/G', 'DROP/G_lag1', 'DROP/G_lag2']

QB_rush_base_features = ['ATT/G', 'ATT/G_lag1', 'ATT/G_lag2', 'ATT/G_lag3', 'ATT/G_std3', 
    'rusYDS/G', 'rusYDS/G_lag1', 'rusYDS/G_lag2', 'rusYDS/G_lag3']

QB_target_feature_map = {
    'CMP': QB_cross_features + QB_pass_base_features + ['CAY', 'CAY_lag1', 'AYD', 'AYD_lag1',
            'CMP%', 'CMP%_lag1', 'xCMP%', 'xCMP%_lag1', '+/-', '+/-_lag1',],
    'pasATT': QB_cross_features + QB_pass_base_features + ['IAY', 'IAY_lag1',
            'CMP%', 'CMP%_lag1', 'xCMP%', 'xCMP%_lag1', '+/-', '+/-_lag1',],
    'pasYDS': QB_cross_features + QB_pass_base_features + ['CAY', 'CAY_lag1', 'AYD', 'AYD_lag1'],
    'pasTD': QB_cross_features + QB_pass_base_features + ['PaTD%', 'PaTD%_lag1', 'xPaTD%', 'xPaTD%_lag1', 
            'RZCMP/G', 'RZCMP/G_lag1', 'RZINT/G', 'RZINT/G_lag1', 'RZINT/G_lag2',
            'RZPasATT/G', 'RZPasATT/G_lag1', 'RZCMP%', 'RZCMP%_lag1',
            'pasTD/G', 'pasTD/G_lag1', 'pasTD/G_lag2', 'pasTD/G_lag3'],
    'INT': ['INT%', 'INT%_lag1', 'INT%_lag2', 'INT/G', 'INT/G_lag1', 'INT/G_lag2', 'INT/G_lag3', 
            'years_since_full', 'POOR/G', 'POOR/G_lag1', 'HRRY/G', 'HRRY/G_lag1',
            'xCMP%', 'xCMP%_lag1', '+/-', '+/-_lag1', 'AGG%', 'AGG%_lag1',  
            'pasATT/G', 'pasATT/G_lag1', 'pasATT/G_lag2', 'pasATT/G_lag3'],
    'ATT': QB_cross_features + QB_rush_base_features,
    'rusYDS': QB_cross_features + QB_rush_base_features,
    'rusTD': QB_cross_features + QB_rush_base_features + ['RuTD%', 'RuTD%_lag1', 'xRuTD%', 'xRuTD%_lag1',
    'RZATT/G', 'RZATT/G_lag1', 'RZRusTD/G', 'RZRusTD/G_lag1',
    'rusTD/G', 'rusTD/G_lag1', 'rusTD/G_lag2', 'rusTD/G_lag3'],
}

QB_target_cols = [f'target_{c}/G' for c in QB_targets]