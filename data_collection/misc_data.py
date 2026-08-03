import nflreadpy as nfl
import pandas as pd


# filters
contract_col_keep = ['player', 'gsis_id', 'position', 'team', 'year_signed', 'years', 'inflated_value', 
    'inflated_guaranteed', 'inflated_apy']
pos_keep = ['QB', 'RB', 'WR', 'TE', 'K']

contracts = nfl.load_contracts().to_pandas()
print(f"Contracts shape: {contracts.shape}")
print(contracts.head())

draft = nfl.load_draft_picks().to_pandas()
print(f"Draft Picks shape: {draft.shape}")
print(draft.head())

# drop observations missing important values
ccontracts = contracts[contract_col_keep]
ccontracts = ccontracts[ccontracts['position'].isin(pos_keep)]
ccontracts = ccontracts[ccontracts['year_signed'] > 0]
ccontracts = ccontracts[ccontracts['years'] > 0]
ccontracts["pct_gtd_sign"] = ccontracts["inflated_guaranteed"] / ccontracts["inflated_value"]

draft_picks = draft[draft['gsis_id'].isin(ccontracts['gsis_id'])]
draft_picks = draft_picks[['gsis_id', 'round', 'pick']].drop_duplicates(subset=['gsis_id'], keep='first')

# merge draft picks with contracts, expand across years of contract
drcon = ccontracts.merge(draft_picks, on='gsis_id', how='left')
drcon["season"] = drcon.apply(
    lambda r: list(range(int(r["year_signed"]), int(r["year_signed"]) + int(r["years"]))), axis=1)

# for overlapping deals we keep the most recent contract for each player-season combination
drcon = drcon.explode("season").reset_index(drop=True)
drcon = (drcon.sort_values("year_signed", ascending=False)
             .drop_duplicates(subset=["gsis_id", "season"], keep="first"))

# filter years to include contracts signed before 2018 but still active in 2018 and beyond
drcon = drcon[drcon['season'] >= 2018].copy()

print(f"\ndraft and contract data: {drcon.shape}")
print(drcon.columns.tolist())

drcon.to_csv("FFdrcon.csv", index=False)