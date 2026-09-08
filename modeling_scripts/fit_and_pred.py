import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV, LassoCV
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error
import lightgbm as lgb
import statsmodels.api as sm

from modeling_prep import build_features, target_feature_map, target_cols, targets

RANDOM_STATE = 42


## fitting and predicting functions
def load_engineer(path: str) -> pd.DataFrame:
    raw = pd.read_csv(path)
    df = build_features(raw)
    return df

def fit_ridge(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    model = RidgeCV(alphas=np.logspace(-2, 3, 30), cv=5).fit(scaler.transform(X_tr), y_tr)
    return {'scaler': scaler, 'model': model}

def pred_ridge(fit, X):
    return fit['model'].predict(fit['scaler'].transform(X))

def fit_lasso(X_tr, y_tr):
    scaler = StandardScaler().fit(X_tr)
    model = LassoCV(alphas=np.logspace(-3, 1, 30), cv=5, max_iter=20000).fit(scaler.transform(X_tr), y_tr)
    return {'scaler': scaler, 'model': model}

def pred_lasso(fit, X):
    return fit['model'].predict(fit['scaler'].transform(X))

def fit_gbm(X_tr_raw, y_tr, X_val_raw=None, y_val=None):
    model = lgb.LGBMRegressor(
        n_estimators=500, max_depth=4, num_leaves=15, learning_rate=0.03,
        min_child_samples=10, subsample=0.8, subsample_freq=1, colsample_bytree=0.8,
        random_state=RANDOM_STATE, verbosity=-1
    )
    if X_val_raw is not None:
        model.fit(X_tr_raw, y_tr, eval_set=[(X_val_raw, y_val)], callbacks=[lgb.early_stopping(30, verbose=False)])
    else:
        model.fit(X_tr_raw, y_tr)
    return{'model': model}

def fit_mixed(X_tr_imp, y_tr, groups_tr):
    scaler = StandardScaler().fit(X_tr_imp)
    Xs = sm.add_constant(scaler.transform(X_tr_imp))
    try: 
        model = sm.MixedLM(y_tr.values, Xs, groups=groups_tr.values)
        result = model.fit(reml=True, method='lbfgs', maxiter=200)
    except Exception:
        result = None
    return {'scaler': scaler, 'result': result}

def pred_mixed(fit, X_imp, groups):
    if fit['result'] is None:
        return np.full(len(X_imp), np.nan)
    Xs = sm.add_constant(fit['scaler'].transform(X_imp), has_constant='add')
    fe = fit['result'].fe_params
    fe_vals = fe.values if hasattr(fe, 'values') else np.asarray(fe)
    base = Xs @ fe_vals
    re_dict = fit['result'].random_effects
    adj = np.array([re_dict[g].values[0] if g in re_dict else 0.0 for g in groups])
    return base + adj


## apply/compare model errors (mae, rmse), refit on data
def eval_pred(df: pd.DataFrame, holdout_year: int=2024, predict_from_year: int=2025):
    transitions = df[df[target_cols].notna().any(axis=1)].copy()
    comparison_rows = []
    best_model_per_target = {}

    preds = {stat: {} for stat in targets}
    pred_rows = df[df['Year'] == predict_from_year].copy()
    pred_rows = pred_rows.loc[pred_rows['last_season'] >= 2026].copy()


    for stat in targets:
        tgt_col = f'target_{stat}/G'
        feature_cols = target_feature_map[stat]
        sub = transitions[transitions[tgt_col].notna()].copy()

        train_mask = sub['Year'] < holdout_year
        val_mask = sub['Year'] == holdout_year

        X_train_raw = sub.loc[train_mask, feature_cols]
        y_train = sub.loc[train_mask, tgt_col]
        X_val_raw = sub.loc[val_mask, feature_cols]
        y_val = sub.loc[val_mask, tgt_col]
        groups_train = sub.loc[train_mask, 'gsis_id']
        groups_val = sub.loc[val_mask, 'gsis_id']

        imputer = SimpleImputer(strategy='median').fit(X_train_raw)
        X_train_imp = pd.DataFrame(imputer.transform(X_train_raw), columns=feature_cols, index=X_train_raw.index)
        X_val_imp = pd.DataFrame(imputer.transform(X_val_raw), columns=feature_cols, index=X_val_raw.index)

        results = {}

        ridge_fit = fit_ridge(X_train_imp, y_train)
        pred = pred_ridge(ridge_fit, X_val_imp)
        results['Ridge'] = (ridge_fit, pred)

        lasso_fit = fit_lasso(X_train_imp, y_train)
        pred = pred_lasso(lasso_fit, X_val_imp)
        results['Lasso'] = (lasso_fit, pred)

        gbm_fit = fit_gbm(X_train_raw, y_train, X_val_raw, y_val)
        pred = gbm_fit['model'].predict(X_val_raw)
        results['GBM'] = (gbm_fit, pred)

        mixed_fit = fit_mixed(X_train_imp, y_train, groups_train)
        pred = pred_mixed(mixed_fit, X_val_imp, groups_val)
        results['MixedLM'] = (mixed_fit, pred)

        best_mae = np.inf
        best_name = None
        for name, (fit, pred) in results.items():
            valid = ~np.isnan(pred)
            if valid.sum() == 0:
                continue
            mae = mean_absolute_error(y_val[valid], pred[valid])
            rmse = np.sqrt(mean_squared_error(y_val[valid], pred[valid]))
            comparison_rows.append({'Stat': stat, 'Model': name, 'MAE/G': round(mae, 4), 
                'RMSE/G': round(rmse, 4), 'N_holdout': int(valid.sum())})
            if mae < best_mae:
                best_mae = mae
                best_name = name

        best_model_per_target[stat] = best_name

        X_all_raw = sub[feature_cols]
        y_all = sub[tgt_col]
        groups_all = sub['gsis_id']
        imputer_all = SimpleImputer(strategy='median').fit(X_all_raw)
        X_all_imp = pd.DataFrame(imputer_all.transform(X_all_raw), columns=feature_cols, index=X_all_raw.index)

        X_pred_raw = pred_rows[feature_cols]
        X_pred_imp = pd.DataFrame(imputer_all.transform(X_pred_raw), columns=feature_cols, index=X_pred_raw.index)
        groups_pred = pred_rows['gsis_id']

        if best_name == 'Ridge':
            fit = fit_ridge(X_all_imp, y_all)
            final_pred = pred_ridge(fit, X_pred_imp)
        elif best_name == 'Lasso':
            fit = fit_lasso(X_all_imp, y_all)
            final_pred = pred_lasso(fit, X_pred_imp)
        elif best_name == 'GBM':
            fit = fit_gbm(X_all_raw, y_all)
            final_pred = fit['model'].predict(X_pred_raw)
        elif best_name == 'MixedLM':
            fit = fit_mixed(X_all_imp, y_all, groups_all)
            final_pred = pred_mixed(fit, X_pred_imp, groups_pred)
        else:
            final_pred = np.full(len(pred_rows), np.nan)

        final_pred = np.clip(final_pred, 0, None)
        preds[stat] = dict(zip(pred_rows['Player'], final_pred))

    comparison_df = pd.DataFrame(comparison_rows).sort_values(['Stat', 'MAE/G'])
    best_df = pd.DataFrame([{'Stat': k, 'Best_Model': v} for k, v in best_model_per_target.items()])

    pred_out = pred_rows[['Player', 'TM', 'age', 'years_exp']].copy()
    pred_out['years_exp'] = pred_out['years_exp'] + 1
    for stat in targets:
        pred_out[f'{stat}'] = round(pred_out['Player'].map(preds[stat]) * 17, 0)

    def to_fantasy_points(rusyds, rustd, rec, recyds, rectd):
        return 0.1 * (rusyds + recyds) + 6 * (rustd + rectd) + rec

    pred_out['FPTs'] = to_fantasy_points(
        pred_out['RusYDS'], pred_out['RusTD'], pred_out['REC'], pred_out['RecYDS'], pred_out['RecTD'])
    pred_out['FPT/G'] = round(pred_out['FPTs'] / 17, 1)

    return comparison_df, best_df, pred_out