import nflreadpy as nfl
import pandas as pd


# filters
contract_keep = ['gsis_id', 'year_signed', 'years', 'inflated_value', 'inflated_guaranteed', 'inflated_apy']
pos_keep = ['QB', 'RB', 'WR', 'TE', 'K']
player_keep = ['display_name', 'position', 'gsis_id', 'birth_date', 'draft_round', 'draft_pick']
inc_years = list(range(2018, 2027))


# load data
players = nfl.load_players().to_pandas()
print(f"Players shape: {players.shape}")
print(players.head())

contracts = nfl.load_contracts().to_pandas()
print(f"Contracts shape: {contracts.shape}")
print(contracts.head())


# brief filtering and explode seasons for each year of interest
players = players[player_keep]
players = players[players['position'].isin(pos_keep)]
players = players.dropna(subset=['birth_date'])
players['birth_date'] = pd.to_datetime(players['birth_date'])
players = players.assign(season=[inc_years] * len(players)).explode('season').reset_index(drop=True)
players['season'] = players['season'].astype(int)
season_ref = pd.to_datetime(players['season'].astype(str) + '-09-01')
players['age'] = ((season_ref - players['birth_date']).dt.days / 365.25).round(2)

contracts = contracts[contract_keep]
contracts = contracts[contracts['gsis_id'].isin(players['gsis_id'])]
contracts = contracts[contracts['year_signed'] > 0]
contracts = contracts[contracts['years'] > 0]
contracts["pct_gtd_sign"] = round(contracts["inflated_guaranteed"] / contracts["inflated_value"], 3)
contracts["season"] = contracts.apply(
    lambda r: list(range(int(r["year_signed"]), int(r["year_signed"]) + int(r["years"]))), axis=1)
contracts = contracts.explode("season").reset_index(drop=True)
contracts["season"] = contracts["season"].astype(int)
contracts = (contracts.sort_values("year_signed", ascending=False)
             .drop_duplicates(subset=["gsis_id", "season"], keep="first"))


# merge players with contracts, expand across years of contract
misc = players.merge(contracts, on=['gsis_id', 'season'], how='left')
misc = misc[misc['season'] >= 2018]

print(f"\nmisc data: {misc.shape}")
print(misc.columns.tolist())

misc.to_csv("RB_misc.csv", index=False)