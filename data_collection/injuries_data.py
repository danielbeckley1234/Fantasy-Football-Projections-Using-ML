import nflreadpy as nfl
import pandas as pd


years = [2018, 2019, 2020, 2021, 2022, 2023, 2024, 2025]
# injury report weekly 2018-2025
injuries = nfl.load_injuries(years).to_pandas()

# weekly player stats 2018-2025 (determines whether player played or not that week)
weekly_stats = nfl.load_player_stats(years).to_pandas()

schedules = nfl.load_schedules(years).to_pandas()

# to-do
## risk of missing games due to injury
## risk of production decline due to injury