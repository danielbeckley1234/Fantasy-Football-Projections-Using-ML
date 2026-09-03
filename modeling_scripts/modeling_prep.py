import pandas as pd
import numpy as np


targets = {'ATT', 'RusYDS', 'RusTD', 'TGT', 'REC', 'RecYDS', 'RecTD', 'FL'}
full_season_thresh = 14

def build_features(raw: pd.DataFrame) ->pd.DataFrame:
    df = raw.copy()
    df = df.sort_values(['gsis_id', 'Year']).reset_index(drop=True)
    games = df['G'].where(df['G'] > 0)

    for col in targets:
        df[f'{col}/G'] = df[col] / games

    g = df.groupby('gsis_id', group_keys=False)

    # lag features three years
    for col in targets:
        df[f'{col}/G_lag1'] = g[f'{col}/G'].shift(1)
        df[f'{col}/G_lag2'] = g[f'{col}/G'].shift(2)
        df[f'{col}/G_lag3'] = g[f'{col}/G'].shift(3)

    # weigh years 0.5/0.3/0.2
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

    # efficiency lags
    for col in ['RusY/A', 'RYOE/ATT', 'Y/RR', 'Y/R']:
        df[f'{col}_lag1'] = g[col].shift(1)
        df[f'{col}_lag2'] = g[col].shift(2)
        df[f'{col}_lag3'] = g[col].shift(3)

    df['years_exp'] = df['Year'] - df['rookie_season']

    # stage in career
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
