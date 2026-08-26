import pandas as pd
import numpy as np


df = pd.read_csv("data_collection/RB/RB_master.csv")

# convert target stats to per game
target_rates = {
    'ATT': 'ATT/G', 'RusYDS': 'RusYDS/G', 'RusTD': 'RusTD/G', 'FL': 'FL/G',
    'REC': 'REC/G', 'Tar': 'Tar/G', 'RecYDS': 'RecYDS/G', 'RecTD': 'RecTD/G',
    'RYOE': 'RYOE/G', 'Routes': 'Routes/G',
}
for stat, conv in target_rates.items():
    df[conv] = df[stat] / df['G'].replace(0, np.nan)

# feature engineering
df['Exp'] = df['Year'] - df['rookie_season']
df['draft_round'] = df['draft_round'].fillna(8)
df['draft_pick'] = df['draft_pick'].fillna(300)

feature_cols = {
    'ATT/G', 'RusYDS/G', 'RusTD/G', 'FL/G', 'REC/G', 'Tar/G', 'RecYDS/G', 'RecTD/G',
    'TmRBWR', 'RYOE/G', 'RYOE/ATT', 'RYOE%', 'Y/R', 'Routes/G', 'Y/RR',
    'Open', 'Catch', 'YAC', 'Overall', 'significant_injury', 'Snaps/G', 'Rush%', 'Tgt%', 'Touch %', 'Util%',
    'age', 'years', 'inflated_value', 'inflated_guaranteed', 'inflated_apy', 'pct_gtd_sign',
}

