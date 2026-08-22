import nflreadpy as nfl
import pandas as pd


# filters
contract_keep = ['gsis_id', 'year_signed', 'years', 'inflated_value', 'inflated_guaranteed', 'inflated_apy']
pos_keep = ['QB', 'RB', 'FB', 'WR', 'TE', 'K']
player_keep = ['display_name', 'position', 'gsis_id', 'birth_date', 'draft_round', 'draft_pick', 'rookie_season', 'last_season']
inc_years = list(range(2018, 2027))


# load data
players = nfl.load_players().to_pandas()
print(f"Players shape: {players.shape}")
print(players.head())

contracts = nfl.load_contracts().to_pandas()
print(f"Contracts shape: {contracts.shape}")
print(contracts.head())

    
# initial filtering and explode seasons for each year of interest
players = players[player_keep]
players = players.rename(columns={'display_name': 'Player'})
players.loc[players['Player'] == 'Ty Montgomery', 'position'] = 'RB'  # correct position for Ty Montgomery
players = players[players['position'].isin(pos_keep)]

players = players.dropna(subset=['birth_date'])
players['birth_date'] = pd.to_datetime(players['birth_date'])
players = players.assign(Year=[inc_years] * len(players)).explode('Year').reset_index(drop=True)
players['Year'] = players['Year'].astype(int)
season_ref = pd.to_datetime(players['Year'].astype(str) + '-09-01')
players['age'] = ((season_ref - players['birth_date']).dt.days / 365.25).round(2)
players = players[players['Year'] >= players['rookie_season']]
players = players[players['Year'] <= players['last_season']]

contracts = contracts[contract_keep]
contracts = contracts[contracts['gsis_id'].isin(players['gsis_id'])]
contracts = contracts[contracts['year_signed'] > 0]
contracts = contracts[contracts['years'] > 0]
contracts["pct_gtd_sign"] = round(contracts["inflated_guaranteed"] / contracts["inflated_value"], 3)
contracts["Year"] = contracts.apply(
    lambda r: list(range(int(r["year_signed"]), int(r["year_signed"]) + int(r["years"]))), axis=1)
contracts = contracts.explode("Year").reset_index(drop=True)
contracts["Year"] = contracts["Year"].astype(int)
contracts = (contracts.sort_values("year_signed", ascending=False)
             .drop_duplicates(subset=["gsis_id", "Year"], keep="first"))


# merge players with contracts, expand across years of contract
misc = players.merge(contracts, on=['gsis_id', 'Year'], how='left')
misc = misc[misc['Year'] >= 2018]

print(f"\nmisc data: {misc.shape}")
print(misc.columns.tolist())

misc.to_csv("misc_data.csv", index=False)