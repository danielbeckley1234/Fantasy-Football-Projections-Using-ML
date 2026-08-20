import pandas as pd
from pathlib import Path

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
misc = misc[misc['position'] == 'RB']

for df in [rec, rush, shares]:
    df['Player'] = df['Player'].str.strip()
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()

merge_keys = ['Player', 'Year', 'TM']
master = rush.merge(rec, on=merge_keys, how='outer')
master = master.merge(shares, on=merge_keys, how='outer')
master = master.merge(misc, on=['Player', 'Year'], how='left')


## data cleaning

# drop players who have never had 80+ carries or 15+ targets in a season
qualifies = master.groupby('Player')[['ATT', 'TGT']].transform('max')
master['qualified'] = (qualifies['ATT'] >= 40) | (qualifies['TGT'] >= 15)

before_vol = master['Player'].nunique()
dropped_vol = master.loc[~master['qualified']].drop(columns=['qualified'])

master = master[master['qualified']].drop(columns=['qualified'])
post_vol = master['Player'].nunique()

print(f"Original number of players: {before_vol}")
print(f"Number of players after filtering by volume: {post_vol}")
print(f"Number of players dropped: {dropped_vol['Player'].nunique()}")
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
master = master[master['keep']].drop(columns=['keep'])
post_years = master['Player'].nunique()

print(f"Number of players before filtering single seasons: {before_years}")
print(f"Number of players after filtering single seasons: {post_years}")
print(f"Number of players dropped: {dropped_years['Player'].nunique()}")
dropped_years.to_excel(base_path / "RB_droppedYears.xlsx", index=False)


# master.to_csv(base_path / "RB_MASTER.csv", index=False)
# master.to_excel(base_path / "RB_MASTER.xlsx", index=False)
