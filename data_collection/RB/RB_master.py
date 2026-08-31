import pandas as pd
from pathlib import Path
import re

RB_path = Path(__file__).resolve().parent
data_path = RB_path.parent

## prep and merge data
# read data and drop/fix repetitive columns
rush = pd.read_excel(RB_path / "RB_rush.xlsx")
rush = rush.drop(columns=['Player (TM)'])

rec = pd.read_excel(RB_path / "RB_rec.xlsx")
rec = rec.drop(columns=['Player (TM)', 'G', 'FL'])

shares = pd.read_excel(RB_path / "RB_shares.xlsx")
shares = shares.rename(columns={'Tm':'TM'})
shares = shares.drop(columns=['G', 'Snap%']) # faulty data from FantasyPros for Snap% (exceeds 100% for several player seasons)

injuries = pd.read_excel(RB_path / "RB_injuries.xlsx")
tds = pd.read_excel(RB_path / "RB_advTD.xlsx")
tds = tds.drop(columns=['Player (TM)'])

misc = pd.read_csv(data_path / "misc_data.csv")
pos = ['RB']
misc = misc[misc['position'].isin(pos)].copy()

# normalize player names (particularly for merger between misc and rest)
def normalize_player(name):
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name)
    name = re.sub(r"['.]", '', name)
    return name.strip()

renames = {'Nyheim Miller-Hines': 'Nyheim Hines', 'Nathan Carter': 'Nate Carter', 'Bo Scarborough': 'Bo Scarbrough',
           'Rodney Smith': 'Rod Smith'}

for df in [rush, rec, shares, injuries, tds]:
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].replace(renames)
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()

# merge excel data (FantasyPros base)
merge_keys = ['Player', 'Year', 'TM']
master = rush.merge(rec, on=merge_keys, how='outer')
master = master.merge(shares, on=merge_keys, how='outer')
master = master.merge(tds, on=merge_keys, how='outer')
master = master.merge(injuries, on=['Player', 'Year'], how='outer', suffixes=('','_inj'))
master['TM'] = master['TM'].fillna(master['TM_inj'])
master = master.drop(columns='TM_inj')
master['significant_injury'] = master['significant_injury'].fillna(0)

# merge with data pulled from nflreadpy
master['norm_name'] = master['Player'].apply(normalize_player)
misc['norm_name'] = misc['Player'].apply(normalize_player)
master = master.merge(misc, on=['norm_name', 'Year'], how='left', suffixes=('', '_misc'))
master = master.drop(columns=['norm_name', 'Player_misc'])

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
# volume filters: at least two productive seasons, drop low-volume seasons trailing at least two real seasons
def vol_check(
        df: pd.DataFrame,
        player_col: str = "Player",
        year_col: str = "Year",
        last_season_col: str = "last_season",
        vol_cols: list = ("ATT", "REC"), 
        prod_thresholds: list = (40, 10),
        dead_thresholds: list = (20, 10),
        min_prod: int = 2,
        min_real: int = 2,
        curr_year_exempt: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame]:

    vol_cols = list(vol_cols)
    prod_thresholds = list(prod_thresholds)
    dead_thresholds = list(dead_thresholds)
    max_year = df[year_col].max()
    proj_year = max_year + 1

    # mark productive seasons
    prod_ind = pd.Series(False, index=df.index)
    for col, threshold in zip(vol_cols, prod_thresholds):
        prod_ind |= df[col] >= threshold

    # mark dead (not real) seasons
    dead_ind = pd.Series(True, index=df.index)
    for col, threshold in zip(vol_cols, dead_thresholds):
        dead_ind &= df[col] < threshold
    dead_ind |= df[vol_cols].isna().all(axis=1)
    real_ind = ~dead_ind

    temp = df[[player_col, year_col, last_season_col]].copy()
    temp["_prod"] = prod_ind
    temp["_dead"] = dead_ind
    temp["_real"] = real_ind

    insuff_prod = []
    tailoff = []

    for player, g in temp.sort_values(year_col).groupby(player_col):
        g = g.reset_index(drop=True)
        prod = g["_prod"].tolist()
        dead = g["_dead"].tolist()
        real = g["_real"].tolist()
        years = g[year_col].tolist()
        n = len(g)

        # insufficient real seasons check
        real_seasons = sum(real)
        if real_seasons < min_real:
            rookie_exempt = curr_year_exempt and years == [max_year]
            if not rookie_exempt:
                insuff_prod.append(
                    {
                        player_col: player,
                        "real_seasons": real_seasons,
                        "total_seasons": n,
                        "years": years,
                    }
                )
            continue # no real seasons

        # tailoff (trailing dead seasons) check
        dead_trail = 0
        for i in reversed(dead):
            if i:
                dead_trail += 1
            else:
                break

        if dead_trail == 0 or dead_trail == n:
            continue

        prior_prod = prod[:n - dead_trail]
        prod_seasons = sum(prior_prod)
        if prod_seasons < min_prod:
            continue

        dead_years = years[n - dead_trail:]
        last_year = g[last_season_col].iloc[0]
        end_dead = last_year < proj_year
        tailoff.append(
            {
                player_col: player,
                "prod_seasons": prod_seasons,
                "dead_years": dead_years,
                "end_on_dead": end_dead,
            }
        )

    tailoff_df = pd.DataFrame(tailoff)
    if not tailoff_df.empty:
        tailoff_df = tailoff_df.sort_values(by="end_on_dead", ascending=False).reset_index(drop=True)

    insuff_df = pd.DataFrame(insuff_prod)
    if not insuff_df.empty:
        insuff_df = insuff_df.sort_values(by="real_seasons").reset_index(drop=True)

    return tailoff_df, insuff_df

tailoff_df, insuff_df = vol_check(master)
insuff_dropped = master[master["Player"].isin(insuff_df["Player"])]
insuff_dropped.to_excel("RB_dropped_insuff.xlsx", index=False)

tail_off_pairs = tailoff_df[["Player", "dead_years"]].explode("dead_years")
tail_off_pairs = tail_off_pairs.rename(columns={"dead_years": "Year"}).drop_duplicates()
tailoff_dropped = master.merge(tail_off_pairs, on=["Player", "Year"], how="inner")
tailoff_dropped.to_excel("RB_dropped_tailoff.xlsx", index=False)

print(f"Before insufficient seasons filter: "
      f"{master['Player'].nunique()} players and {len(master)} seasons.")
master = master[~master["Player"].isin(insuff_df["Player"])].copy()

print(f"Before tailoff filter: "
      f"{master['Player'].nunique()} players and {len(master)} seasons.")
drop_mask = master.set_index(["Player", "Year"]).index.isin(tail_off_pairs.set_index(["Player", "Year"]).index)
master = master[~drop_mask]

print(f"After tailoff filter: "
      f"{master['Player'].nunique()} players and {len(master)} seasons.")

master.to_excel(RB_path / "RB_MASTER.xlsx", index=False)
