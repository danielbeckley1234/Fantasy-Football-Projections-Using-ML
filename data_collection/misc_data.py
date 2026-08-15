import nflreadpy as nfl
import pandas as pd


# filters
contract_keep = ['player', 'gsis_id', 'position', 'team', 'year_signed', 'years', 'inflated_value', 
    'inflated_guaranteed', 'inflated_apy']
pos_keep = ['QB', 'RB', 'WR', 'TE', 'K']
player_keep = ['gsis_id', 'birth_date', 'draft_round', 'draft_pick']
inc_years = list(range(2018, 2027))


# load data
contracts = nfl.load_contracts().to_pandas()
print(f"Contracts shape: {contracts.shape}")
print(contracts.head())

players = nfl.load_players().to_pandas()
print(f"Players shape: {players.shape}")
print(players.head())


# brief filtering and explode seasons for each year of interest
contracts = contracts[contract_keep]
contracts = contracts[contracts['position'].isin(pos_keep)]
contracts = contracts[contracts['year_signed'] > 0]
contracts = contracts[contracts['years'] > 0]
contracts["pct_gtd_sign"] = round(contracts["inflated_guaranteed"] / contracts["inflated_value"], 3)
contracts["season"] = contracts.apply(
    lambda r: list(range(int(r["year_signed"]), int(r["year_signed"]) + int(r["years"]))), axis=1)
contracts = contracts.explode("season").reset_index(drop=True)
contracts["season"] = contracts["season"].astype(int)

players = players[player_keep]
players = players[players['gsis_id'].isin(contracts['gsis_id'])]
players = players.dropna(subset=['birth_date'])
players['birth_date'] = pd.to_datetime(players['birth_date'])
players = players.assign(season=[inc_years] * len(players)).explode('season').reset_index(drop=True)
players['season'] = players['season'].astype(int)
season_ref = pd.to_datetime(players['season'].astype(str) + '-09-01')
players['age'] = ((season_ref - players['birth_date']).dt.days / 365.25).round(2)


# merge players with contracts, expand across years of contract
misc = contracts.merge(players, on=['gsis_id', 'season'], how='left')

# for overlapping deals we keep the most recent contract for each player-season combination

misc = (misc.sort_values("year_signed", ascending=False)
             .drop_duplicates(subset=["gsis_id", "season"], keep="first"))

# filter years to include contracts signed before 2018 but still active in 2018 and beyond
misc = misc[misc['season'] >= 2018].copy()

print(f"\ndraft and contract data: {misc.shape}")
print(misc.columns.tolist())

misc.to_csv("RB_misc.csv", index=False)