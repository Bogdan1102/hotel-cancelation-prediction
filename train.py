"""
train.py — Train the final model and save it (cap. 5.1.2)

Usage:
    python train.py

Reads:  data/hotel_bookings.csv
Writes: model.bin   (DictVectorizer + XGBoost model + metadata)
"""

import pickle
import warnings
import numpy as np
import pandas as pd
import xgboost as xgb

from sklearn.model_selection import train_test_split, KFold
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score

warnings.filterwarnings('ignore')
RANDOM_STATE = 42

DATA_PATH = 'data/hotel_bookings.csv'
OUTPUT_PATH = 'model.bin'


def load_and_clean(path):
    """Load CSV, normalize columns, drop leakage + sparse cols, subsample."""
    df = pd.read_csv(path)
    # Normalize column names (cap. 2.2.2)
    df.columns = df.columns.str.lower().str.replace(' ', '_')

    # Drop leakage columns (they encode the target)
    df = df.drop(columns=['reservation_status', 'reservation_status_date'])

    # Drop company (94% missing)
    df = df.drop(columns=['company'])

    # Subsample to 15k rows (still >10k requirement, faster training)
    df = df.sample(n=15000, random_state=RANDOM_STATE).reset_index(drop=True)
    return df


def split_features_target(df, target='is_canceled'):
    """Identify numerical/categorical columns and split target."""
    numerical = [c for c in df.select_dtypes(include='number').columns if c != target]
    categorical = list(df.select_dtypes(exclude='number').columns)
    return numerical, categorical


def impute_missing(df, numerical, categorical):
    """Fill missing values: numerical -> 0, categorical -> 'NA'."""
    df = df.copy()
    for c in numerical:
        df[c] = df[c].fillna(0)
    for c in categorical:
        df[c] = df[c].fillna('NA').astype(str)
    return df


def vectorize(df_train, df_val, df_test, categorical, numerical):
    """One-hot encode with DictVectorizer(sparse=True) (cap. 3.2.1, 6.1.3)."""
    dv = DictVectorizer(sparse=True)
    dv.fit(df_train[categorical + numerical].to_dict(orient='records'))
    X_train = dv.transform(df_train[categorical + numerical].to_dict(orient='records'))
    X_val   = dv.transform(df_val[categorical + numerical].to_dict(orient='records'))
    X_test  = dv.transform(df_test[categorical + numerical].to_dict(orient='records'))
    return dv, X_train, X_val, X_test


def train_logreg(X_train, y_train):
    """Baseline logistic regression (cap. 3.3.2)."""
    m = LogisticRegression(solver='liblinear', C=1.0, max_iter=1000,
                          random_state=RANDOM_STATE)
    m.fit(X_train, y_train)
    return m


def tune_decision_tree(X_train, y_train, X_val, y_val):
    """Tune max_depth + min_samples_leaf (cap. 6.2.3)."""
    best_auc, best_params, best_model = 0, None, None
    for depth in [5, 10, 15]:
        for ms_leaf in [10, 50, 100]:
            dt = DecisionTreeClassifier(max_depth=depth, min_samples_leaf=ms_leaf,
                                        random_state=RANDOM_STATE)
            dt.fit(X_train, y_train)
            auc = roc_auc_score(y_val, dt.predict_proba(X_val)[:, 1])
            if auc > best_auc:
                best_auc = auc
                best_params = (depth, ms_leaf)
                best_model = dt
    return best_model, best_params, best_auc


def tune_random_forest(X_train, y_train, X_val, y_val):
    """Tune n_estimators + max_depth (cap. 6.3.2)."""
    best_auc, best_params, best_model = 0, None, None
    for depth in [10, 15, 20]:
        for n_est in [50, 100]:
            rf = RandomForestClassifier(n_estimators=n_est, max_depth=depth,
                                        random_state=RANDOM_STATE, n_jobs=-1)
            rf.fit(X_train, y_train)
            auc = roc_auc_score(y_val, rf.predict_proba(X_val)[:, 1])
            if auc > best_auc:
                best_auc = auc
                best_params = (depth, n_est)
                best_model = rf
    return best_model, best_params, best_auc


def tune_xgboost(X_train, y_train, X_val, y_val, feature_names):
    """Tune eta + max_depth (cap. 6.4)."""
    dtrain = xgb.DMatrix(X_train, label=y_train, feature_names=feature_names)
    dval   = xgb.DMatrix(X_val,   label=y_val,   feature_names=feature_names)

    best_auc, best_params, best_model = 0, None, None
    for eta in [0.1, 0.3]:
        for depth in [3, 6]:
            params = {
                'eta': eta,
                'max_depth': depth,
                'min_child_weight': 1,
                'objective': 'binary:logistic',
                'nthread': 8,
                'seed': RANDOM_STATE,
                'verbosity': 0,
            }
            m = xgb.train(params, dtrain, num_boost_round=100,
                          evals=[(dtrain, 'train'), (dval, 'val')],
                          verbose_eval=False)
            auc = roc_auc_score(y_val, m.predict(dval))
            if auc > best_auc:
                best_auc = auc
                best_params = params
                best_model = m
    return best_model, best_params, best_auc


def kfold_cv_xgb(df_full, target, categorical, numerical, xgb_params, k=5):
    """K-fold cross-validation (cap. 4.4.1)."""
    df_full = df_full.reset_index(drop=True).copy()
    kfold = KFold(n_splits=k, shuffle=True, random_state=RANDOM_STATE)
    aucs = []
    for tr_idx, vl_idx in kfold.split(df_full):
        df_tr = df_full.iloc[tr_idx].reset_index(drop=True)
        df_vl = df_full.iloc[vl_idx].reset_index(drop=True)
        y_tr = df_tr[target].values
        y_vl = df_vl[target].values
        del df_tr[target]; del df_vl[target]

        dv_cv = DictVectorizer(sparse=True)
        X_tr = dv_cv.fit_transform(df_tr[categorical + numerical].to_dict(orient='records'))
        X_vl = dv_cv.transform(df_vl[categorical + numerical].to_dict(orient='records'))

        dtr = xgb.DMatrix(X_tr, label=y_tr, feature_names=dv_cv.feature_names_)
        dvl = xgb.DMatrix(X_vl, label=y_vl, feature_names=dv_cv.feature_names_)
        m = xgb.train(xgb_params, dtr, num_boost_round=100, verbose_eval=False)
        aucs.append(roc_auc_score(y_vl, m.predict(dvl)))
    return aucs


def main():
    print("=" * 60)
    print("Hotel Booking Cancellation Prediction — train.py")
    print("=" * 60)

    # 1) Load + clean
    df = load_and_clean(DATA_PATH)
    print(f"Loaded: {df.shape}")

    # 2) Identify columns + impute missing
    target = 'is_canceled'
    numerical, categorical = split_features_target(df, target)
    df = impute_missing(df, numerical, categorical)
    print(f"Numerical ({len(numerical)}): {numerical}")
    print(f"Categorical ({len(categorical)}): {categorical}")

    # 3) Split (cap. 3.1.2)
    df_full_train, df_test = train_test_split(df, test_size=0.2, random_state=RANDOM_STATE)
    df_train, df_val = train_test_split(df_full_train, test_size=0.25, random_state=RANDOM_STATE)
    df_train = df_train.reset_index(drop=True)
    df_val   = df_val.reset_index(drop=True)
    df_test  = df_test.reset_index(drop=True)

    y_train = df_train[target].values
    y_val   = df_val[target].values
    y_test  = df_test[target].values
    for d in (df_train, df_val, df_test):
        if target in d.columns:
            del d[target]
    print(f"Train: {len(df_train)}, Val: {len(df_val)}, Test: {len(df_test)}")

    # 4) One-hot encoding
    dv, X_train, X_val, X_test = vectorize(df_train, df_val, df_test,
                                            categorical, numerical)
    print(f"Features after one-hot: {X_train.shape[1]}")

    # 5) Train + tune 4 models
    print("\n--- Training Logistic Regression ---")
    lr = train_logreg(X_train, y_train)
    auc_lr = roc_auc_score(y_val, lr.predict_proba(X_val)[:, 1])
    print(f"LogReg AUC = {auc_lr:.4f}")

    print("\n--- Tuning Decision Tree ---")
    dt, dt_params, auc_dt = tune_decision_tree(X_train, y_train, X_val, y_val)
    print(f"Best DT params: max_depth={dt_params[0]}, min_samples_leaf={dt_params[1]}, AUC={auc_dt:.4f}")

    print("\n--- Tuning Random Forest ---")
    rf, rf_params, auc_rf = tune_random_forest(X_train, y_train, X_val, y_val)
    print(f"Best RF params: max_depth={rf_params[0]}, n_estimators={rf_params[1]}, AUC={auc_rf:.4f}")

    print("\n--- Tuning XGBoost ---")
    xgb_model, xgb_params, auc_xgb = tune_xgboost(X_train, y_train, X_val, y_val,
                                                   dv.feature_names_)
    print(f"Best XGB params: eta={xgb_params['eta']}, max_depth={xgb_params['max_depth']}, AUC={auc_xgb:.4f}")

    # 6) Summary
    print("\n" + "=" * 60)
    print("MODEL COMPARISON (AUC on validation):")
    print("=" * 60)
    print(f"  LogReg         : {auc_lr:.4f}")
    print(f"  DecisionTree   : {auc_dt:.4f}")
    print(f"  RandomForest   : {auc_rf:.4f}")
    print(f"  XGBoost        : {auc_xgb:.4f}")
    print("=" * 60)
    print(f"Best model: XGBoost")

    # 7) K-fold CV (cap. 4.4.1)
    print(f"\n--- K-fold CV (XGBoost) ---")
    aucs_cv = kfold_cv_xgb(df_full_train, target, categorical, numerical, xgb_params, k=5)
    for i, a in enumerate(aucs_cv, 1):
        print(f"  Fold {i}: AUC = {a:.4f}")
    print(f"\nK-fold CV: AUC = {np.mean(aucs_cv):.4f} ± {np.std(aucs_cv):.4f}")

    # 8) Train final model on full_train + evaluate on test (cap. 6.4.4)
    print("\n--- Training final model on full_train ---")
    df_full_train = df_full_train.reset_index(drop=True)
    y_full = df_full_train[target].values
    df_full_features = df_full_train.drop(columns=[target])

    dv_final = DictVectorizer(sparse=True)
    X_full_train = dv_final.fit_transform(df_full_features[categorical + numerical].to_dict(orient='records'))
    X_test_final = dv_final.transform(df_test[categorical + numerical].to_dict(orient='records'))

    dfull = xgb.DMatrix(X_full_train, label=y_full, feature_names=dv_final.feature_names_)
    dtest = xgb.DMatrix(X_test_final, feature_names=dv_final.feature_names_)
    final_model = xgb.train(xgb_params, dfull, num_boost_round=100, verbose_eval=False)
    y_pred_test = final_model.predict(dtest)

    auc_test = roc_auc_score(y_test, y_pred_test)
    print(f"Test AUC: {auc_test:.4f}")

    # 9) Save model (cap. 5.1.2)
    with open(OUTPUT_PATH, 'wb') as f_out:
        pickle.dump({
            'dv': dv_final,
            'model': final_model,
            'model_name': 'XGBoost',
            'features': categorical + numerical,
            'best_params': xgb_params,
            'metrics': {
                'val_auc': float(auc_xgb),
                'cv_auc_mean': float(np.mean(aucs_cv)),
                'cv_auc_std': float(np.std(aucs_cv)),
                'test_auc': float(auc_test),
            }
        }, f_out)
    print(f"\nModel saved to {OUTPUT_PATH}")
    print("=" * 60)
    print("DONE.")


if __name__ == '__main__':
    main()
