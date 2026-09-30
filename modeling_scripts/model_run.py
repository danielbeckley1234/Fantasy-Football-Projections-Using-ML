# model_run.py: outputs 2026 projections for each position
import sys
from pathlib import Path
project_path = Path(__file__).resolve().parent.parent
if str(project_path) not in sys.path:
    sys.path.insert(0, str(project_path))

import pandas as pd
from base_prep import build_features
from fit_and_pred import eval_pred
from data_collection.QB.QB_map import QB_targets, QB_counting, QB_lag1s, QB_lag2s, QB_target_feature_map, QB_target_cols
from data_collection.RB.RB_map import RB_targets, RB_counting, RB_lag1s, RB_lag2s, RB_target_feature_map, RB_target_cols
from data_collection.WR_TE.WR_TE_map import WRTE_targets, WRTE_counting, WRTE_lag1s, WRTE_lag2s, WR_TE_target_feature_map, WRTE_target_cols

starters = pd.read_csv(project_path / 'data_collection/starters.csv')
starter_ids = set(starters['gsis_id'])

QB_df = pd.read_csv(project_path / 'data_collection/QB/QB_MASTER.csv')
QB_df = build_features(QB_df, QB_targets, QB_counting, QB_lag1s, QB_lag2s, 'QB')
QB_df = QB_df[QB_df['gsis_id'].isin(starter_ids)]
print("Engineered feature rows:", QB_df.shape)

RB_df = pd.read_csv(project_path / 'data_collection/RB/RB_MASTER.csv')
RB_df = build_features(RB_df, RB_targets, RB_counting, RB_lag1s, RB_lag2s, 'RB')
RB_df = RB_df[RB_df['gsis_id'].isin(starter_ids)]
print("Engineered feature rows:", RB_df.shape)

WR_df = pd.read_csv(project_path / 'data_collection/WR_TE/WR_MASTER.csv')
WR_df = build_features(WR_df, WRTE_targets, WRTE_counting, WRTE_lag1s, WRTE_lag2s, 'WR')
WR_df = WR_df[WR_df['gsis_id'].isin(starter_ids)]
print("Engineered feature rows:", WR_df.shape)

TE_df = pd.read_csv(project_path / 'data_collection/WR_TE/TE_MASTER.csv')
TE_df = build_features(TE_df, WRTE_targets, WRTE_counting, WRTE_lag1s, WRTE_lag2s, 'TE')
TE_df = TE_df[TE_df['gsis_id'].isin(starter_ids)]
print("Engineered feature rows:", TE_df.shape)

QB_comp_df, QB_best_df, QB_pred_out = eval_pred(
    QB_df, QB_target_cols, QB_targets, QB_target_feature_map, 'QB', holdout_year=2024, predict_from_year=2025)
QB_comp_df.to_csv('QB_model_comps.csv', index=False)
QB_best_df.to_csv('QB_best_model_by_stat.csv')
QB_pred_out.sort_values('FPTS', ascending=False).to_csv('QB_2026_projections.csv', index=False)

RB_comp_df, RB_best_df, RB_pred_out = eval_pred(
    RB_df, RB_target_cols, RB_targets, RB_target_feature_map, 'RB', holdout_year=2024, predict_from_year=2025)
RB_comp_df.to_csv('RB_model_comps.csv', index=False)
RB_best_df.to_csv('RB_best_model_by_stat.csv')
RB_pred_out.sort_values('FPTS', ascending=False).to_csv('RB_2026_projections.csv', index=False)

WR_comp_df, WR_best_df, WR_pred_out = eval_pred(
    WR_df, WRTE_target_cols, WRTE_targets, WR_TE_target_feature_map, 'WR', holdout_year=2024, predict_from_year=2025)
WR_comp_df.to_csv('WR_model_comps.csv', index=False)
WR_best_df.to_csv('WR_best_model_by_stat.csv')
WR_pred_out.sort_values('FPTS', ascending=False).to_csv('WR_2026_projections.csv', index=False)

TE_comp_df, TE_best_df, TE_pred_out = eval_pred(
    TE_df, WRTE_target_cols, WRTE_targets, WR_TE_target_feature_map, 'TE', holdout_year=2024, predict_from_year=2025)
TE_comp_df.to_csv('TE_model_comps.csv', index=False)
TE_best_df.to_csv('TE_best_model_by_stat.csv')
TE_pred_out.sort_values('FPTS', ascending=False).to_csv('TE_2026_projections.csv', index=False)

print("\n 2026 Projections (top 10 QBs):")
print(QB_pred_out.sort_values('FPTS', ascending=False).head(10).to_string(index=False))
print("\n 2026 Projections (top 10 RBs):")
print(RB_pred_out.sort_values('FPTS', ascending=False).head(10).to_string(index=False))
print("\n 2026 Projections (top 10 WRs):")
print(WR_pred_out.sort_values('FPTS', ascending=False).head(10).to_string(index=False))
print("\n 2026 Projections (top 10 TEs):")
print(TE_pred_out.sort_values('FPTS', ascending=False).head(10).to_string(index=False))