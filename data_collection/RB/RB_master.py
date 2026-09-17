import sys
import pandas as pd
from pathlib import Path

RB_path = Path(__file__).resolve().parent
data_path = RB_path.parent
sys.path.append(str(data_path))
from utils import vol_check, apply_vol_check, normalize_player

## prep and merge data
# read data and drop/fix repetitive columns
rush = pd.read_excel(RB_path / "RB_rush.xlsx")
rush = rush.drop(columns=['Player (TM)'])

rec = pd.read_excel(RB_path / "RB_rec.xlsx")
rec = rec.drop(columns=['Player (TM)', 'G', 'FL']) # decided not to project FL

shares = pd.read_excel(RB_path / "RB_shares.xlsx")
shares = shares.rename(columns={'Tm':'TM'})
shares = shares.drop(columns=['G', 'Snap%']) # faulty data from FantasyPros for Snap% (exceeds 100% for several player seasons)

injuries = pd.read_excel(RB_path / "RB_injuries.xlsx")
injuries = injuries.drop(columns=['TM']) # inconsistencies with misc
tds = pd.read_excel(RB_path / "RB_advTD.xlsx")
tds = tds.drop(columns=['Player (TM)'])

misc = pd.read_csv(data_path / "misc_data.csv")
misc = misc[misc['position'] == 'RB']

# rename signficantly varied names of players
renames = {'Nyheim Miller-Hines': 'Nyheim Hines', 'Nathan Carter': 'Nate Carter', 'Bo Scarborough': 'Bo Scarbrough',
           'Rodney Smith': 'Rod Smith'}

for df in [rush, rec, shares, injuries, tds, misc]:
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].replace(renames)
    df['Player'] = df['Player'].apply(normalize_player)
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()
        df['TM'] = df['TM'].replace({'OAK': 'LV'})

merge_keys = ['Player', 'Year', 'TM']
backup_merge = ['Player', 'Year']
master = rush.merge(rec, on=merge_keys, how='outer')
master = master.merge(shares, on=merge_keys, how='outer')
master = master.merge(tds, on=merge_keys, how='outer')
master = master.merge(injuries, on=backup_merge, how='outer')
master['significant_injury'] = master['significant_injury'].fillna(0)
master.loc[(master['Player'] == 'Damien Harris') & (master['Year'] == 2019), 'significant_injury'] = 1 # missed in original injuries
master = master.merge(misc, on=backup_merge, how='left')
master = master[master['position'] == 'RB']

# audric estime was the only RB of interest who did not appear in the misc dataset
estime_manual = {
    'position': 'RB',
    'gsis_id': 'ESTIME',
    'birth_date': '2003-09-06',
    'draft_round': 5,
    'draft_pick': 147,
    'rookie_season': 2024,
    'last_season': 2025,
}

estime_birth = pd.Timestamp(estime_manual['birth_date'])
for year in [2024, 2025]:
    rep = (master['Player'] == 'Audric Estime') & (master['Year'] == year)
    for col, val in estime_manual.items():
        master.loc[rep, col] = val
    season_ref = pd.Timestamp(f'{year}-09-01')
    master.loc[rep, 'age'] = round((season_ref - estime_birth).days / 365.25, 2)


## data cleaning
# manual_drops = {

# }
# drop_pairs = pd.DataFrame(
#     [(player, year) for player, years in manual_drops.items() for year in years],
#     columns=["Player", "Year"]
# )
# drop_mask = master.set_index(["Player", "Year"]).index.isin(
#     drop_pairs.set_index(["Player", "Year"]).index
# )
# master = master[~drop_mask]
master = master[master['gsis_id'] != '00-0035957'] # duplicate rod smith
master['ATT/G'] = round(master['ATT'] / master['G'], 3)
master['TGT/G'] = round(master['TGT'] / master['G'], 3)

tailoff_df, insuff_df, RB_df = vol_check(
    master, vol_cols=["ATT/G", "TGT/G", "ATT", "TGT"], prod_thresholds=[6, 3, 25, 15], manual_keep={"Braelon Allen"}
)
RB_master, insuff_dropped, tailoff_dropped = apply_vol_check(RB_df, tailoff_df, insuff_df)

insuff_dropped.to_csv(RB_path / "RB_dropped_insuff.csv", index=False)
tailoff_dropped.to_csv(RB_path / "RB_dropped_tailoff.csv", index=False)
RB_master.to_csv(RB_path / "RB_MASTER.csv", index=False)