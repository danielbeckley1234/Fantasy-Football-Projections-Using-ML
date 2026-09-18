import pandas as pd
import numpy as np


def build_features(
    raw: pd.DataFrame,
    targets: list,
    counting_stats: list,
    lag1s: list,
    lag2s: list,
    pos: str,
) ->pd.DataFrame:
    
    df = raw.copy()
    df = df.sort_values(['gsis_id', 'Year']).reset_index(drop=True)
    games = df['G'].where(df['G'] > 0)
    for col in counting_stats:
        df[f'{col}/G'] = df[col] / games

    # lag respective statistics
    g = df.groupby('gsis_id', group_keys=False)
    for col in lag1s:
        df[f'{col}/G_lag1'] = g[f'{col}/G'].shift(1) 

    for col in lag2s:
        df[f'{col}/G_lag1'] = g[f'{col}/G'].shift(1)
        df[f'{col}/G_lag2'] = g[f'{col}/G'].shift(2)

    for col in targets:
        df[f'{col}/G_lag1'] = g[f'{col}/G'].shift(1)
        df[f'{col}/G_lag2'] = g[f'{col}/G'].shift(2)
        df[f'{col}/G_lag3'] = g[f'{col}/G'].shift(3)   

# stage/trajectory/health in career evaluator
    def pct_of_peak_role(sub: pd.DataFrame, pos: str) -> pd.DataFrame:
        sub = sub.sort_values('Year')
        if pos == 'RB':
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

    # specific position work
    if pos == 'RB':
        for col in ['ATT', 'TGT']:
                df[f'{col}/G_std3'] = g[f'{col}/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    elif pos.isin['WR', 'TE']:
        df[f'TGT/G_std3'] = g[f'TGT/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    # elif pos == 'QB':
    #     df[f'PATT/G_std3'] = g[f'PATT/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    # elif pos == 'K':
    #     df[f'FgATT/G_std3'] = g[f'FgATT/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())

    # universal feature engineered columns
    df['years_exp'] = df['Year'] - df['rookie_season']
    df['team_change'] = (df['TM'] != g['TM'].shift(1)).astype(int)
    df.loc[g.cumcount() == 0, 'team_change'] = 0
    season_length = np.where(df['Year'] >= 2021, 17, 16)
    df['games_missed_rate'] = 1 - (df['G'].fillna(0) / season_length)
    df['draft_round_filled'] = df['draft_round'].fillna(8)
    df['draft_pick_filled'] = df['draft_pick'].fillna(300)
    df['age_sq'] = df['age'] ** 2

    for col in targets:
        df[f'target_{col}/G'] = g[f'{col}/G'].shift(-1)

    return df
  

