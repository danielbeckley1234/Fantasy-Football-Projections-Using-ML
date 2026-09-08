import pandas as pd
from pathlib import Path
from fit_and_pred import load_engineer, eval_pred

model_path = Path(__file__).resolve().parent
gen_path = model_path.parent

df = load_engineer(gen_path / 'data_collection/RB/RB_MASTER.csv')
print("Engineered feature rows:", df.shape)

comparison_df, best_df, pred_out = eval_pred(df, holdout_year=2024, predict_from_year=2025)
comparison_df.to_csv('model_comparisons.csv', index=False)
best_df.to_csv('best_model_by_stat.csv')
pred_out.sort_values('FPTs', ascending=False).to_csv('RB_2026_projections.csv', index=False)

print("\n Model Comparison (holdout: 2024-2025):")
print(comparison_df.to_string(index=False))
print("\n Best Model Per Stat:")
print(best_df.to_string(index=False))
print("\n 2026 Projections (top 10 RBs):")
print(pred_out.sort_values('FPTs', ascending=False).head(10).to_string(index=False))