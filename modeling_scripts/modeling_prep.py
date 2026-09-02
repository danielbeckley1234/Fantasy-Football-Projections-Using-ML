import pandas as pd
import numpy as np


targets = {'ATT', 'RusYDS', 'RusTD', 'TGT', 'REC', 'RecYDS', 'RecTD', 'FL'}

prod_att_thresh = 50
prod_rec_thresh = 20
prod_g_thresh = 13

def build_features(raw: pd.DataFrame) ->pd.DataFrame:
    df = raw.copy()
    df = df.sort_values(['gsis_id', 'Year']).reset_index(drop=True)

    for col in targets:
        df[f'{col}/G'] = df[col] / df['G']

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

    # std of att/g and tgt/3 
    for col in ['ATT', 'TGT']:
        df[f'{col}/G_std'] = g[f'{col}/G'].apply(lambda s: s.shift(1).rolling(3, min_periods=2).std())
