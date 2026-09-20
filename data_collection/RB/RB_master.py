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
RB_master = rush.merge(rec, on=merge_keys, how='outer')
RB_master = RB_master.merge(shares, on=merge_keys, how='outer')
RB_master = RB_master.merge(tds, on=merge_keys, how='outer')
RB_master = RB_master.merge(injuries, on=backup_merge, how='outer')
RB_master['significant_injury'] = RB_master['significant_injury'].fillna(0)
RB_master.loc[(RB_master['Player'] == 'Damien Harris') & (RB_master['Year'] == 2019), 'significant_injury'] = 1 # missed in original injuries
RB_master = RB_master.merge(misc, on=backup_merge, how='left')
RB_master = RB_master[RB_master['position'] == 'RB'] # some fullbacks made it into the FantasyPros data

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
    rep = (RB_master['Player'] == 'Audric Estime') & (RB_master['Year'] == year)
    for col, val in estime_manual.items():
        RB_master.loc[rep, col] = val
    season_ref = pd.Timestamp(f'{year}-09-01')
    RB_master.loc[rep, 'age'] = round((season_ref - estime_birth).days / 365.25, 2)


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
RB_master = RB_master[RB_master['gsis_id'] != '00-0035957'] # duplicate rod smith
RB_master['ATT/G'] = round(RB_master['ATT'] / RB_master['G'], 3)
RB_master['TGT/G'] = round(RB_master['TGT'] / RB_master['G'], 3)
RB_master['touches'] = RB_master['ATT'] + RB_master['REC']

tailoff_df, insuff_df, RB_df = vol_check(
    RB_master, vol_cols=["ATT/G", "TGT/G", "ATT", "TGT"], prod_thresholds=[6, 3, 25, 15], manual_keep={"Braelon Allen"}
)
RB_master, insuff_dropped, tailoff_dropped = apply_vol_check(RB_df, tailoff_df, insuff_df)

insuff_dropped.to_csv(RB_path / "RB_dropped_insuff.csv", index=False)
tailoff_dropped.to_csv(RB_path / "RB_dropped_tailoff.csv", index=False)
RB_master.to_csv(RB_path / "RB_MASTER.csv", index=False)