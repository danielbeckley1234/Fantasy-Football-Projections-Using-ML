import nflreadpy as nfl
import pandas as pd


# filters
contract_col_drop = ['apy', 'inflated_value', 'is_active','inflated_guaranteed', 'player_page', 'date_of_birth', 'height', 'weight',
 'college', 'draft_year', 'draft_team', 'cols']
pos_keep = ['QB', 'RB', 'WR', 'TE', 'K']

contracts = nfl.load_contracts().to_pandas()
print(f"Contracts shape: {contracts.shape}")
print(contracts.head())

draft = nfl.load_draft_picks().to_pandas()
print(f"Draft Picks shape: {draft.shape}")
print(draft.head())

ccontracts = contracts.drop(columns=contract_col_drop)
ccontracts = ccontracts[ccontracts['position'].isin(pos_keep)]
ccontracts = ccontracts[ccontracts['year_signed'] > 0]
ccontracts = ccontracts[ccontracts['years'] > 0]
ccontracts["pct_gtd_sign"] = ccontracts["guaranteed"] / ccontracts["value"]

draft_picks = draft[draft['gsis_id'].isin(ccontracts['gsis_id'])]
draft_picks = draft_picks[draft_picks['gsis_id', 'round', 'pick']].drop_duplicates(subset=['gsis_id'], keep='first')

ccontracts = ccontracts.merge (draft_picks, on='gsis_id', how='left')
ccontracts[season] = ccontracts.apply(
    lambda r: list(range(int(r["year_signed"]), int(r["year_signed"]) + int(r["years"]))), axis=1)

ccontracts = ccontracts.explode("season").reset_index(drop=True)
ccontracts = (ccontracts.sort_values("year_signed", ascending=False)
             .drop_duplicates(subset=["gsis_id", "season"], keep="first"))

ccontracts = ccontracts[ccontracts['season'] >= 2018].copy()

print(f"\nmisc_data: {ccontracts.shape}")
print(ccontracts.columns.tolist())