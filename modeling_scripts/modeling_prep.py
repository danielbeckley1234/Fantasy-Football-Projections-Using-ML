import pandas as pd
import numpy as np


targets = {'ATT', 'RusYDS', 'RusTD', 'TGT', 'REC', 'RecYDS', 'RecTD'}
full_season_thresh = 14

def build_features(raw: pd.DataFrame) ->pd.DataFrame:
    df = raw.copy()
    df = df.sort_values(['gsis_id', 'Year']).reset_index(drop=True)
    games = df['G'].where(df['G'] > 0)

    # convert counting stats to per game
    for col in targets:
        df[f'{col}/G'] = df[col] / games

    counting_stats = ['Routes', 'In5ATT', 'In5REC', 'In5TG']
    for col in counting_stats:
        df[f'{col}/G'] = df[col] / games

    g = df.groupby('gsis_id', group_keys=False)

    # lag and weigh target features three years
    for col in targets:
        df[f'{col}/G_lag1'] = g[f'{col}/G'].shift(1)
        df[f'{col}/G_lag2'] = g[f'{col}/G'].shift(2)
        df[f'{col}/G_lag3'] = g[f'{col}/G'].shift(3)

    for col in targets:
        l1, l2, l3 = df[f'{col}/G_lag1'], df[f'{col}/G_lag2'], df[f'{col}/G_lag3']
        w1, w2, w3, = 0.5, 0.3, 0.2
        weights = np.where(l1.notna(), w1, 0) + np.where(l2.notna(), w2, 0) + np.where(l3.notna(), w3, 0)
        weighted_sum = l1.fillna(0) * w1 + l2.fillna(0) * w2 + l3.fillna(0) * w3
        with np.errstate(invalid='ignore', divide='ignore'):
            df[f'{col}/G_weighted'] = np.where(weights > 0, weighted_sum / weights, np.nan)

    # workload volatility: std of att/g and tgt/g 
    for col in ['ATT', 'TGT']:
        df[f'{col}/G_std'] = g[f'{col}/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())

    # lag other features
    role_share = ['Rush%', 'Tgt%', 'Snaps/G']
    for col in role_share:
        df[f'{col}_lag1'] = g[col].shift(1)

    df[f'Routes/G_lag1'] = g['Routes/G'].shift(1)

    efficiency_metrics = ['RusY/A', 'RYOE/ATT', 'Y/RR']
    for col in efficiency_metrics:
        df[f'{col}_lag1'] = g[col].shift(1)

    TD_rates = ['RusTD%', 'RecTD%', 'xRusTD%', 'xRecTD%']
    for col in TD_rates:
        df[f'{col}_lag1'] = g[col].shift(1)

    redzone_stats = ['In5Rush%', 'In5ATT/G', 'In5TG/G']
    for col in redzone_stats:
        df[f'{col}_lag1'] = g[col].shift(1)
        df[f'{col}_lag2'] = g[col].shift(2)

    df['years_exp'] = df['Year'] - df['rookie_season']

    # stage/trajectory/health in career evaluator
    def pct_of_peak_role(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values('Year')
        prior_att, prior_tgt = [], []
        pct_peak_att, pct_peak_tgt = [], []
        for _, row in sub.iterrows():
            att_val = row['ATT'] if pd.notna(row['ATT']) else 0.0
            tgt_val = row['TGT'] if pd.notna(row['TGT']) else 0.0

            # rookies/first observed season
            if len(prior_att) == 0:
                pa, pt = np.nan, np.nan
            else:
                peak_att = max(prior_att)
                peak_tgt = max(prior_tgt)
                pa = att_val / peak_att if peak_att > 0 else np.nan
                pt = tgt_val / peak_tgt if peak_tgt > 0 else np.nan
            pct_peak_att.append(pa)
            pct_peak_tgt.append(pt)
            prior_att.append(att_val)
            prior_tgt.append(tgt_val)
        return pd.DataFrame({'pct_peak_att': pct_peak_att, 'pct_peak_tgt': pct_peak_tgt}, index=sub.index)

    peak_df = g.apply(pct_of_peak_role)
    if isinstance(peak_df.index, pd.MultiIndex):
        peak_df = peak_df.reset_index(level=0, drop=True)
    df[['pct_peak_att', 'pct_peak_tgt']] = peak_df
    g = df.groupby('gsis_id', group_keys=False)

    def years_since_full(sub: pd.DataFrame) -> pd.Series:
        sub = sub.sort_values('Year')
        last_full_year = np.nan
        out = []
        for _, row in sub.iterrows():
            if pd.notna(last_full_year):
                out.append(row['Year'] - last_full_year)
            else:
                out.append(np.nan)
            curr_g = row['G'] if pd.notna(row['G']) else 0
            if curr_g >= full_season_thresh:
                last_full_year = row['Year']
        return pd.Series(out, index=sub.index)
 
    df['years_since_full'] = g.apply(years_since_full).reset_index(level=0, drop=True)
    data_start_year = df['Year'].min()
    never_full_mask = df['years_since_full'].isna()
    career_in_window = df['rookie_season'] >= data_start_year
    apply_sentinel_mask = never_full_mask & career_in_window
    df.loc[apply_sentinel_mask, 'years_since_full'] = df.loc[apply_sentinel_mask, 'years_exp'] + 2

    df['significant_injury_lag1'] = g['significant_injury'].shift(1)
    df['team_change'] = (df['TM'] != g['TM'].shift(1)).astype(int)
    df.loc[g.cumcount() == 0, 'team_change'] = 0

    season_length = np.where(df['Year'] >= 2021, 17, 16)
    df['games_missed_rate'] = 1 - (df['G'].fillna(0) / season_length)

    df['draft_round_filled'] = df['draft_round'].fillna(8)
    df['draft_pick_filled'] = df['draft_pick'].fillna(300)

    df['age_sq'] = df['age'] ** 2

    for col in targets:
        df[f'target_{col}'] = df[col].shift(-1)

    return df