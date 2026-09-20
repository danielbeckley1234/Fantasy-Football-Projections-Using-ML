# base_prep.py: feature engineering, lagged features
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
        df[f'{col}_lag1'] = g[f'{col}'].shift(1) 
    for col in lag2s:
        df[f'{col}_lag1'] = g[f'{col}'].shift(1)
        df[f'{col}_lag2'] = g[f'{col}'].shift(2)
    for col in targets:
        df[f'{col}/G_lag1'] = g[f'{col}/G'].shift(1)
        df[f'{col}/G_lag2'] = g[f'{col}/G'].shift(2)
        df[f'{col}/G_lag3'] = g[f'{col}/G'].shift(3) 

        # universal feature engineered columns
    df['years_exp'] = df['Year'] - df['rookie_season']
    df['team_change'] = (df['TM'] != g['TM'].shift(1)).astype(int)
    df.loc[g.cumcount() == 0, 'team_change'] = 0
    season_length = np.where(df['Year'] >= 2021, 17, 16)
    df['games_missed_rate'] = 1 - (df['G'].fillna(0) / season_length)
    df['draft_round_filled'] = df['draft_round'].fillna(8)
    df['draft_pick_filled'] = df['draft_pick'].fillna(300)
    df['age_sq'] = df['age'] ** 2

# stage/trajectory/health in career evaluator
    def pct_of_peak_role(sub: pd.DataFrame) -> pd.DataFrame:
        sub = sub.sort_values('Year')
        prior_vol, pct_peak_vol = [], []
        for _, row in sub.iterrows():
            if 'touches' in df.columns:
                vol_val = row['touches'] if pd.notna(row['touches']) else 0.0
            # elif 'pATT' in df.columns():
            #     vol_val = row['pATT'] if pd.notna(row['pATT']) else 0.0
            # elif 'FG+XP' in df.columns():
            #     vol_val = row['FG+XP'] if pd.notna(row['FG+XP']) else 0.0

            # rookies/first observed season
            if len(prior_vol) == 0:
                pv = np.nan
            else:
                peak_vol = max(prior_vol)
                pv = vol_val / peak_vol if peak_vol > 0 else np.nan
            pct_peak_vol.append(pv)
            prior_vol.append(vol_val)
        return pd.DataFrame({'pct_peak_vol': pct_peak_vol}, index=sub.index)

    peak_df = g.apply(pct_of_peak_role)
    if isinstance(peak_df.index, pd.MultiIndex):
        peak_df = peak_df.reset_index(level=0, drop=True)
    df[['pct_peak_vol']] = peak_df
    g = df.groupby('gsis_id', group_keys=False)

    # positional volume stds
    if pos == 'RB':
        for col in ['ATT', 'TGT']:
                df[f'{col}/G_std3'] = g[f'{col}/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    elif pos in ['WR', 'TE']:
        df[f'TGT/G_std3'] = g[f'TGT/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    # elif pos == 'QB':
    #     df[f'pATT/G_std3'] = g[f'pATT/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
    # elif pos == 'K':
    #     df[f'FGA/G_std3'] = g[f'FGA/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())


    for col in targets:
        df[f'target_{col}/G'] = g[f'{col}/G'].shift(-1)

    return df
  

