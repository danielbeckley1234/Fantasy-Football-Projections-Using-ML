import sys
import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path.parent
sys.path.append(str(data_path))
from utils import vol_check, apply_vol_check, normalize_player

## prep and merge data
# read data and drop/fix repetitive columns
WR_base = pd.read_excel(base_path / "WR_Base.xlsx")
WR_base = WR_base.drop(columns=['Player (TM)'])

WR_TD = pd.read_excel(base_path / "WR_TD.xlsx")
WR_TD = WR_TD.drop(columns=['Player (TM)', 'GAMES'])

WR_shares = pd.read_excel(base_path / "WR_shares.xlsx")
WR_shares = WR_shares.drop(columns=['GAMES'])

WR_redzone = pd.read_excel(base_path / "WR_redzone.xlsx")
WR_redzone = WR_redzone.drop(columns=['Player TM'])

WR_injuries = pd.read_excel(base_path / "WR_injuries.xlsx")

espn = pd.read_excel(base_path / "WR_TE_ESPN.xlsx")
espn = espn.drop(columns=['Targets', 'Yds'])
WR_espn = espn[espn['Pos'] == 'WR']
TE_espn = espn[espn['Pos'] == 'TE']

nextgen = pd.read_excel(base_path / "WR_TE_nextgen.xlsx")
nextgen = nextgen.drop(columns='TD')
WR_nextgen = nextgen[nextgen['POS'] == 'WR']
TE_nextgen = nextgen[nextgen['POS'] == 'TE']

misc = pd.read_csv(data_path / "misc_data.csv")
WR_misc = misc[misc['position'] == 'WR']
TE_misc = misc[misc['position'] == 'TE']

# renames
renames = {'Deonte Harris': 'Deonte Harty', 'Josh Palmer': 'Joshua Palmer', 'Scott Miller': 'Scotty Miller', 'Will Fuller': 'William Fuller'}

for df in [WR_base, WR_TD, WR_shares, WR_redzone, WR_injuries, WR_espn, TE_espn, WR_misc, TE_misc]:
    df['Player'] = df['Player'].str.strip()
    df['Player'] = df['Player'].replace(renames)
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()