import sys
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path.parent
sys.path.append(str(data_path))
from utils import vol_check, apply_vol_check, normalize_player

## prep and merge data
# read data and drop/fix necessary columns
QB_base = pd.read_excel(base_path / "QB_base.xlsx")
QB_base = QB_base.drop(columns=['Player (TM)'])

QB_td = pd.read_excel(base_path / "QB_TD.xlsx")
QB_td = QB_td.drop(columns=['Player (TM)'])
QB_td = QB_td.rename(columns={'xRuTD': 'xRuTD%'})

QB_redzone = pd.read_excel(base_path / "QB_redzone.xlsx")
QB_redzone = QB_redzone.drop(columns=['Player (TM)'])

QB_injuries = pd.read_excel(base_path / "QB_injuries.xlsx")

QB_nextgen = pd.read_excel(base_path / "QB_nextgen.xlsx")
QB_nextgen = QB_nextgen.drop(columns=['TM']) # different naming conventions, wont hurt to drop and use backup merge keys

QB_adv = pd.read_excel(base_path / 'QB_adv.xlsx')
QB_adv = QB_adv.drop(columns={'Player (TM)'})
QB_adv = QB_adv.drop(columns=['SACK']) # already in base

misc = pd.read_csv(data_path / "misc_data.csv")
QB_misc = misc[misc['position'] == 'QB']

starters = pd.read_csv(data_path / "starters.csv")
QB_starters = set(starters.loc[starters['Pos'] == 'QB', 'gsis_id'].dropna())

renames = {'Cameron Ward': 'Cam Ward'}
for df in [QB_base, QB_td, QB_redzone, QB_injuries, QB_nextgen, QB_misc]:
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].replace(renames)
    df['Player'] = df['Player'].apply(normalize_player)
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()
        df['TM'] = df['TM'].replace({'OAK': 'LV'})

merge_keys = ['Player', 'TM', 'Year']
backup_merge = ['Player', 'Year']

QB_master = QB_base.merge(QB_td, on=merge_keys, how="outer")
QB_master = QB_master.merge(QB_redzone, on=backup_merge, how="outer")
QB_master = QB_master.merge(QB_injuries, on=backup_merge, how="outer")
QB_master['significant_injury'] = QB_master['significant_injury'].fillna(0)
QB_master = QB_master.merge(QB_adv, on=merge_keys, how="outer")
QB_master = QB_master.merge(QB_nextgen, on=backup_merge, how="outer")
QB_master = QB_master.merge(QB_misc, on=backup_merge, how="left")

QB_master['pasATT/G'] = round(QB_master['pasATT'] / QB_master['G'], 3)

QB_tailoff_df, QB_insuff_df, QB_df = vol_check(QB_master, vol_cols=["pasATT/G", "pasATT"], prod_thresholds=[10, 100],
    manual_keep_ids=QB_starters)
QB_master, QB_insuff_dropped, QB_tailoff_dropped = apply_vol_check(QB_df, QB_tailoff_df, QB_insuff_df)
QB_insuff_dropped.to_csv(base_path / "QB_dropped_insuff.csv", index=False)
QB_tailoff_dropped.to_csv(base_path / "QB_dropped_tailoff.csv", index=False)

QB_master.to_csv(base_path / "QB_MASTER.csv", index=False)
