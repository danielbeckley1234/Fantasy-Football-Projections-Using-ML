import pandas as pd
from pathlib import Path
import re

base_path = Path(__file__).resolve().parent
data_path = base_path.parent

# prep and merge data
rush = pd.read_excel(base_path / "RB_rush.xlsx")
rush = rush.drop(columns=['Player (TM)'])

rec = pd.read_excel(base_path / "RB_rec.xlsx")
rec = rec.drop(columns=['Player (TM)', 'G', 'FL'])

shares = pd.read_excel(base_path / "RB_shares.xlsx")
shares = shares.rename(columns={'Tm':'TM'})
shares = shares.drop(columns=['G'])

misc = pd.read_csv(data_path / "misc_data.csv")
pos = ['RB', 'FB']
misc = misc[misc['position'].isin(pos)].copy()

renames = {'Nyheim Miller-Hines': 'Nyheim Hines', 'Nathan Carter': 'Nate Carter'}
manual_drops = {'LeGarrette Blount', 'Chris Ivory'}
estime_manual = {
    'position': 'RB',
    'gsis_id': 'ESTIME',
    'birth_date': '2003-09-06',
    'draft_round': 5,
    'draft_pick': 147,
    'rookie_season': 2024,
    'last_season': 2025,
}

def normalize_player(name):
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name)
    name = re.sub(r"['.]", '', name)
    return name.strip()

for df in [rush, rec, shares]:
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].replace(renames)
    df['TM'] = df['TM'].str.strip()

merge_keys = ['Player', 'Year', 'TM']
master = rush.merge(rec, on=merge_keys, how='outer')
master = master.merge(shares, on=merge_keys, how='outer')

master['norm_name'] = master['Player'].apply(normalize_player)
misc['norm_name'] = misc['Player'].apply(normalize_player)
master = master.merge(misc, on=['norm_name', 'Year'], how='left', suffixes=('', '_misc'))
master = master.drop(columns=['norm_name', 'Player_misc'])
master = master[~master['Player'].isin(manual_drops)]

estime_birth = pd.Timestamp(estime_manual['birth_date'])
for year in [2024, 2025]:
    rep = (master['Player'] == 'Audric Estime') & (master['Year'] == year)
    for col, val in estime_manual.items():
        master.loc[rep, col] = val
    season_ref = pd.Timestamp(f'{year}-09-01')
    master.loc[rep, 'age'] = round((season_ref - estime_birth).days / 365.25, 2)

## data cleaning
# drop players who have never had 40+ carries or 15+ targets in a season
qualifies = master.groupby('Player')[['ATT', 'TGT']].transform('max')
master['qualified'] = (qualifies['ATT'] >= 40) | (qualifies['TGT'] >= 15)
before_vol = master['Player'].nunique()
dropped_vol = master.loc[~master['qualified']].drop(columns=['qualified'])

print(f"Original number of players: {before_vol} and seasons: {master.shape[0]}")
master = master[master['qualified']].drop(columns=['qualified'])
post_vol = master['Player'].nunique()

print(f"Number of players after filtering by volume: {post_vol} and seasons: {master.shape[0]}")
print(f"Number of players dropped: {dropped_vol['Player'].nunique()} and seasons: {dropped_vol.shape[0]}")
dropped_vol.to_excel(base_path / "RB_droppedVol.xlsx", index=False)

# drop players with one year of data besides 2025
seasons = master.groupby('Player')['Year'].nunique()
years_played = master.groupby('Player')['Year'].apply(set)

def keep_player(player):
    if seasons[player] > 1:
        return True
    return years_played[player] == {2025}

master['keep'] = master['Player'].map(keep_player)
before_years = master['Player'].nunique()
dropped_years = master.loc[~master['keep']].drop(columns=['keep'])

print(f"Number of players before filtering single seasons: {before_years} and seasons: {master.shape[0]}")
master = master[master['keep']].drop(columns=['keep'])
post_years = master['Player'].nunique()


print(f"Number of players after filtering single seasons: {post_years} and seasons: {master.shape[0]}")
print(f"Number of players dropped: {dropped_years['Player'].nunique()} and seasons: {dropped_years.shape[0]}")
dropped_years.to_excel(base_path / "RB_droppedYears.xlsx", index=False)

master.to_csv(base_path / "RB_MASTER.csv", index=False)
master.to_excel(base_path / "RB_MASTER.xlsx", index=False)
