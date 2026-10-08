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
RANDOM_STATE = 42


## fitting and predicting functions
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
def eval_pred(
        df: pd.DataFrame, 
        target_cols: list,
        targets: list,
        target_feature_map: list,
        pos: str,
        holdout_year: int=2024, 
        predict_from_year: int=2025,
        ):
    transitions = df[df[target_cols].notna().any(axis=1)].copy()
    comparison_rows = []
    best_model_per_target = {}
    coefficients = {}

    preds = {stat: {} for stat in targets}
    pred_rows = df[df['Year'] == predict_from_year].copy()
    pred_rows = pred_rows.loc[pred_rows['last_season'] >= 2026].copy()

    def get_coefs(fit, feature_cols, model_name):
        if model_name in ['Ridge', 'Lasso']:
            coef_scaled = fit['model'].coef_
            coef_raw = coef_scaled / fit['scaler'].scale_
            return pd.DataFrame({
                'feature': feature_cols,
                'standardized_coef': coef_scaled,
                'raw_coef': coef_raw,
            }).sort_values('standardized_coef', key=abs, ascending=False)
        elif model_name == 'GBM':
            return pd.DataFrame({
                'feature': feature_cols,
                'importance': fit['model'].feature_importances_,
            }).sort_values('importance', ascending=False)
        elif model_name == 'MixedLM':
            if fit['result'] is None:
                return pd.DataFrame(columns=['feature', 'standardized_coef'])
            fe = fit['result'].fe_params
            fe_vals = fe.values if hasattr(fe, 'values') else np.asarray(fe)
            return pd.DataFrame({
                'feature': ['intercept'] + list(feature_cols),
                'standardized_coef': fe_vals,
            }).sort_values('standardized_coef', key=abs, ascending=False)
        else:
            return pd.DataFrame()
            
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
        second_mae = np.inf
        second_name = None
        for name, (fit, pred) in results.items():
            valid = ~np.isnan(pred)
            if valid.sum() == 0:
                continue
            mae = mean_absolute_error(y_val[valid], pred[valid])
            rmse = np.sqrt(mean_squared_error(y_val[valid], pred[valid]))
            comparison_rows.append({'Stat': stat, 'Model': name, 'MAE/G': round(mae, 4), 
                'RMSE/G': round(rmse, 4), 'N_holdout': int(valid.sum())})
            if mae < best_mae:
                second_mae, second_name = best_mae, best_name
                best_mae, best_name = mae, name
            elif mae < second_mae:
                second_mae, second_name = mae, name

        best_model_per_target[stat] = best_name

        def fit_and_pred(name, X_all_imp, X_all_raw, y_all, groups_all, X_pred_imp, X_pred_raw, groups_pred):
            if name == 'Ridge':
                fit = fit_ridge(X_all_imp, y_all)
                return fit, pred_ridge(fit, X_pred_imp)
            elif name == 'Lasso':
                fit = fit_lasso(X_all_imp, y_all)
                return fit, pred_lasso(fit, X_pred_imp)
            elif name == 'GBM':
                fit = fit_gbm(X_all_raw, y_all)
                return fit, fit['model'].predict(X_pred_raw)
            elif name == 'MixedLM':
                fit = fit_mixed(X_all_imp, y_all, groups_all)
                return fit, pred_mixed(fit, X_pred_imp, groups_pred)
            return fit, np.full(len(X_pred_imp), np.nan)
        
        X_all_raw = sub[feature_cols]
        y_all = sub[tgt_col]
        groups_all = sub['gsis_id']
        imputer_all = SimpleImputer(strategy='median').fit(X_all_raw)
        X_all_imp = pd.DataFrame(imputer_all.transform(X_all_raw), columns=feature_cols, index=X_all_raw.index)
        X_pred_raw = pred_rows[feature_cols]
        X_pred_imp = pd.DataFrame(imputer_all.transform(X_pred_raw), columns=feature_cols, index=X_pred_raw.index)
        groups_pred = pred_rows['gsis_id']

        fit, final_pred = fit_and_pred(best_name, X_all_imp, X_all_raw, y_all, groups_all, X_pred_imp, X_pred_raw, groups_pred)
        if np.all(np.isnan(final_pred)) and second_name is not None:
            print(f" [{pos}] stat={stat}: {best_name} produce all NaNs, falling back to {second_name}")
            fit, final_pred = fit_and_pred(second_name, X_all_imp, X_all_raw, y_all, groups_all, X_pred_imp, X_pred_raw, groups_pred)
        
        final_pred = np.clip(final_pred, 0, None)
        preds[stat] = dict(zip(pred_rows['Player'], final_pred))
        coefficients[stat] = get_coefs(fit, feature_cols, best_name)

    comparison_df = pd.DataFrame(comparison_rows).sort_values(['Stat', 'MAE/G'])
    best_df = pd.DataFrame([{'Stat': k, 'Best_Model': v} for k, v in best_model_per_target.items()])

    pred_out = pred_rows[['Player', 'TM', 'age', 'years_exp']].copy()
    pred_out['years_exp'] = pred_out['years_exp'] + 1
    for stat in targets:
        pred_out[f'{stat}'] = round(pred_out['Player'].map(preds[stat]) * 17, 0)

    def flex_to_fpts(rusyds, rustd, rec, recyds, rectd):
        return 0.1 * (rusyds + recyds) + 6 * (rustd + rectd) + rec
    def qb_to_fpts(pasyds, pastd, int, rusyds, rustd):
        return 0.04 * (pasyds) + 4 * (pastd) - 2 * (int) + 0.1 * (rusyds) + 6 * (rustd)

    if pos in ['RB', 'TE', 'WR']:
        pred_out['FPTS'] = flex_to_fpts(
            pred_out['RusYDS'], pred_out['RusTD'], pred_out['REC'], pred_out['RecYDS'], pred_out['RecTD'])
        pred_out['FPTS/G'] = round(pred_out['FPTS'] / 17, 1)
    elif pos == 'QB':
        pred_out['FPTS'] = qb_to_fpts(
            pred_out['pasYDS'], pred_out['pasTD'], pred_out['INT'], pred_out['rusYDS'], pred_out['rusTD'])
        pred_out['FPTS/G'] = round(pred_out['FPTS'] / 17, 1) 

    return comparison_df, best_df, pred_out, coefficients