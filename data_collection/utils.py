# utils.py: includes universal functions for each position's master dataset contruction construction.
import re
import pandas as pd


# normalize player names (particularly for merger between misc and rest)
def normalize_player(name):
    name = re.sub(r'\s+(Jr\.?|Sr\.?|II|III|IV|V)$', '', name)
    name = re.sub(r"['.]", '', name)
    return name.strip()

# volume filters: at least two productive seasons per player, drop low-volume seasons trailing at least two real seasons
def vol_check(
        df: pd.DataFrame,
        vol_cols: list, 
        prod_thresholds: list,
        player_col: str = "Player",
        year_col: str = "Year",
        last_season_col: str = "last_season",
        min_prod: int = 2,
        curr_year_exempt: bool = True,
        manual_keep: set = frozenset(),
) -> tuple[pd.DataFrame, pd.DataFrame]:

    vol_cols = list(vol_cols)
    prod_thresholds = list(prod_thresholds)
    max_year = df[year_col].max()
    proj_year = max_year + 1

    # mark productive and dead seasons
    prod_ind = pd.Series(False, index=df.index)
    for col, threshold in zip(vol_cols, prod_thresholds):
        prod_ind |= df[col] >= threshold
    dead_ind = ~prod_ind

    temp = df[[player_col, year_col, last_season_col]].copy()
    temp["_prod"] = prod_ind
    temp["_dead"] = dead_ind

    insuff_prod = []
    tailoff = []

    for player, g in temp.sort_values(year_col).groupby(player_col):
        g = g.reset_index(drop=True)
        prod = g["_prod"].tolist()
        dead = g["_dead"].tolist()
        years = g[year_col].tolist()
        n = len(g)

        # insufficient productive seasons check
        prod_seasons = sum(prod)
        if prod_seasons < min_prod:
            rookie = curr_year_exempt and years == [max_year] 
            prod_rookie = rookie and prod_seasons >= 1
            manual_exempt = player in manual_keep
            if not (prod_rookie or manual_exempt):
                insuff_prod.append(
                    {
                        player_col: player,
                        "prod_seasons": prod_seasons,
                        "total_seasons": n,
                        "years": years,
                    }
                )
            continue

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
        tailoff_df = tailoff_df.sort_values(by="prod_seasons", ascending=False).reset_index(drop=True)

    insuff_df = pd.DataFrame(insuff_prod)
    if not insuff_df.empty:
        insuff_df = insuff_df.sort_values(by="prod_seasons").reset_index(drop=True)

    return tailoff_df, insuff_df

def apply_vol_check(
    master: pd.DataFrame,
    tailoff_df: pd.DataFrame,
    insuff_df: pd.DataFrame,
    player_col: str = "Player",
    year_col: str = "Year",
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    insuff_dropped = master[master[player_col].isin(insuff_df[player_col])]

    tailoff_pairs = tailoff_df[[player_col, "dead_years"]].explode("dead_years")
    tailoff_pairs = tailoff_pairs.rename(columns={"dead_years": year_col}).drop_duplicates()
    tailoff_dropped = master.merge(tailoff_pairs, on=[player_col, year_col], how="inner")

    print(f"Before insufficient seasons filter: "
      f"{master[player_col].nunique()} players and {len(master)} seasons.")
    cleaned = master[~master[player_col].isin(insuff_df[player_col])]

    print(f"Before tailoff filter: "
      f"{cleaned[player_col].nunique()} players and {len(cleaned)} seasons.")
    cleaned = cleaned.merge(tailoff_pairs, on=[player_col, year_col], how="left", indicator=True)
    cleaned = cleaned[cleaned["_merge"] == "left_only"].drop(columns=["_merge"])

    print(f"After tailoff filter: "
      f"{cleaned[player_col].nunique()} players and {len(cleaned)} seasons.")

    return cleaned.reset_index(drop=True), insuff_dropped, tailoff_dropped