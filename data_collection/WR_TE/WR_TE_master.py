import sys
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path.parent
sys.path.append(str(data_path))
from utils import vol_check, apply_vol_check, normalize_player

## prep and merge data
# read data and drop/fix necessary columns
WR_base = pd.read_excel(base_path / "WR_base.xlsx")
WR_base = WR_base.drop(columns=['Player (TM)', 'TGT %']) # tgt% redundant with other dataset
TE_base = pd.read_excel(base_path / "TE_base.xlsx")
TE_base = TE_base.drop(columns=['Player (TM)', 'TGT %'])


WR_td = pd.read_excel(base_path / "WR_TD.xlsx")
WR_td = WR_td.drop(columns=['Player (TM)'])
TE_td = pd.read_excel(base_path / "TE_TD.xlsx")
TE_td = TE_td.drop(columns=['Player (TM)'])

WR_redzone = pd.read_excel(base_path / "WR_redzone.xlsx")
WR_redzone = WR_redzone.drop(columns=['Player (TM)', 'TM']) # TM current to date not respective to season
TE_redzone = pd.read_excel(base_path / "TE_redzone.xlsx")
TE_redzone = TE_redzone.drop(columns=['Player (TM)', 'TM'])

WR_injuries = pd.read_excel(base_path / "WR_injuries.xlsx")
TE_injuries = pd.read_excel(base_path / "TE_injuries.xlsx")

WR_shares = pd.read_excel(base_path / "WR_shares.xlsx")
WR_shares = WR_shares.rename(columns={'SNAPS/GM': 'Snaps/G'})
TE_shares = pd.read_excel(base_path / "TE_shares.xlsx")

espn = pd.read_excel(base_path / "WR_TE_ESPN.xlsx")
WR_espn = espn[espn['Pos'] == 'WR']
TE_espn = espn[espn['Pos'] == 'TE']

nextgen = pd.read_excel(base_path / "WR_TE_nextgen.xlsx")
WR_nextgen = nextgen[nextgen['POS'] == 'WR']
TE_nextgen = nextgen[nextgen['POS'] == 'TE']

WR_adv = pd.read_excel(base_path / 'WR_adv.xlsx')
WR_adv = WR_adv.drop(columns={'Player (TM)'})
TE_adv = pd.read_excel(base_path / 'TE_adv.xlsx')
TE_adv = TE_adv.drop(columns={'Player (TM)'})

misc = pd.read_csv(data_path / "misc_data.csv")
WR_misc = misc[misc['position'] == 'WR']
TE_misc = misc[misc['position'] == 'TE']

# renames
renames = {'Deonte Harris': 'Deonte Harty', 'Josh Palmer': 'Joshua Palmer', 'Scott Miller': 'Scotty Miller', 
           'William Fuller V': 'Will Fuller', 'Marquise Brown': 'Hollywood Brown', 'Robbie Anderson': 'Robbie Chosen',
           'Gabriel Davis': 'Gabe Davis'}

for df in [WR_base, TE_base, WR_td, TE_td, WR_redzone, TE_redzone, WR_injuries, TE_injuries, WR_shares, TE_shares,
        WR_espn, TE_espn, WR_nextgen, TE_nextgen, WR_misc, TE_misc]:
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].replace(renames)
    df['Player'] = df['Player'].apply(normalize_player)
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()
        df['TM'] = df['TM'].replace({'OAK': 'LV'})

merge_keys = ['Player', 'TM', 'Year']
backup_merge = ['Player', 'Year']
WR_master = WR_base.merge(WR_td, on=merge_keys, how="outer")
WR_master = WR_master.merge(WR_redzone, on=backup_merge, how="outer")
WR_master = WR_master.merge(WR_injuries, on=backup_merge, how="outer")
WR_master['significant_injury'] = WR_master['significant_injury'].fillna(0)
WR_master = WR_master.merge(WR_shares, on=merge_keys, how="outer")
WR_master = WR_master.merge(WR_espn, on=merge_keys, how="outer")
WR_master = WR_master.merge(WR_nextgen, on=merge_keys, how="outer")
WR_master = WR_master.merge(WR_misc, on=backup_merge, how="left")

TE_master = TE_base.merge(TE_td, on=merge_keys, how="outer")
TE_master = TE_master.merge(TE_redzone, on=backup_merge, how="outer")
TE_master = TE_master.merge(TE_injuries, on=backup_merge, how="outer")
TE_master['significant_injury'] = TE_master['significant_injury'].fillna(0)
TE_master = TE_master.merge(TE_shares, on=merge_keys, how="outer")
TE_master = TE_master.merge(TE_espn, on=merge_keys, how="outer")
TE_master = TE_master.merge(TE_nextgen, on=merge_keys, how="outer")
TE_master = TE_master.merge(TE_misc, on=backup_merge, how="left")

WR_master['TGT/G'] = round(WR_master['TGT'] / WR_master['G'], 3)
TE_master['TGT/G'] = round(TE_master['TGT'] / TE_master['G'], 3)
WR_master['touches'] = WR_master['ATT'] + WR_master['REC']
TE_master['touches'] = TE_master['ATT'] + TE_master['REC']

## data cleaning
WR_manual_drops = {
'Antonio Callaway': {2018, 2019}, 'DJ Chark': {2024}
}
drop_pairs = pd.DataFrame(
    [(player, year) for player, years in WR_manual_drops.items() for year in years], columns=["Player", "Year"]
)
drop_mask = WR_master.set_index(["Player", "Year"]).index.isin(drop_pairs.set_index(["Player", "Year"]).index)
WR_master = WR_master[~drop_mask]
manual_keeps = {'Cedric Wilson', 'Tim Patrick'}

WR_tailoff_df, WR_insuff_df, WR_df = vol_check(WR_master, vol_cols=["TGT/G", "TGT"], prod_thresholds=[2.5, 28], manual_keep=manual_keeps)
WR_master, WR_insuff_dropped, WR_tailoff_dropped = apply_vol_check(WR_df, WR_tailoff_df, WR_insuff_df)
WR_insuff_dropped.to_csv(base_path / "WR_dropped_insuff.csv", index=False)
WR_tailoff_dropped.to_csv(base_path / "WR_dropped_tailoff.csv", index=False)


TE_tailoff_df, TE_insuff_df, TE_df = vol_check(TE_master, vol_cols=["TGT/G", "TGT"], prod_thresholds=[2.5, 20])
TE_master, TE_insuff_dropped, TE_tailoff_dropped = apply_vol_check(TE_df, TE_tailoff_df, TE_insuff_df)
TE_insuff_dropped.to_csv(base_path / "TE_dropped_insuff.csv", index=False)
TE_tailoff_dropped.to_csv(base_path / "TE_dropped_tailoff.csv", index=False)

WR_master = WR_master.sort_values(['Player', 'Year']).reset_index(drop=True)
TE_master = TE_master.sort_values(['Player', 'Year']).reset_index(drop=True)
WR_master.to_csv(base_path / "WR_MASTER.csv", index=False)
TE_master.to_csv(base_path / "TE_MASTER.csv", index=False)