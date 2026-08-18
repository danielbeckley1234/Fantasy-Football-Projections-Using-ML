import pandas as pd
from pathlib import Path

base_path = Path(__file__).resolve().parent
data_path = base_path.parent

# read in data
rec = pd.read_excel(base_path / "RB_rec.xlsx")
rush = pd.read_excel(base_path / "RB_rush.xlsx")
shares = pd.read_excel(base_path / "RB_shares.xlsx")
misc = pd.read_csv(data_path / "misc_data.csv")


# column manipulation
shares = shares.rename(columns={'Tm':'TM'})
rec = rec.drop(columns=['Player (TM)', 'G', 'FL'])
rush = rush.drop(columns=['Player (TM)'])

for df in [rec, rush, shares]:
    df['Player'] = df['Player'].str.strip()
    if 'TM' in df.columns:
        df['TM'] = df['TM'].str.strip()