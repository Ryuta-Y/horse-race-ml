# scripts/train_oddsless_lgbm.py
from __future__ import annotations
import argparse, os, joblib
import numpy as np
import lightgbm as lgb
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.metrics import log_loss, roc_auc_score

try:
    from scripts.utils.io_utils import load_records, build_features
except ModuleNotFoundError:
    from utils.io_utils import load_records, build_features

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-glob", default="data/raw/record_data_*.csv")
    ap.add_argument("--out", default="models/oddsless_lgbm.pkl")
    ap.add_argument("--folds", type=int, default=5)
    ap.add_argument("--early_stopping", type=int, default=50)
    args = ap.parse_args()

    rec = load_records(args.records_glob)
    X, y, groups, feat_cols = build_features(rec, use_odds=False)

    sgkf = StratifiedGroupKFold(n_splits=args.folds, shuffle=True, random_state=42)
    oof = np.zeros(len(y))
    models = []
    for fold, (tr, va) in enumerate(sgkf.split(X, y, groups), 1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y[tr], y[va]
        train_set = lgb.Dataset(Xtr, label=ytr)
        valid_set = lgb.Dataset(Xva, label=yva, reference=train_set)

        params = dict(
            objective="binary",
            metric="binary_logloss",
            learning_rate=0.05,
            num_leaves=63,
            feature_fraction=0.9,
            bagging_fraction=0.9,
            bagging_freq=1,
            min_data_in_leaf=20,
            verbose=-1,
        )
        gbm = lgb.train(
            params,
            train_set,
            valid_sets=[valid_set],
            num_boost_round=2000,
            callbacks=[lgb.early_stopping(stopping_rounds=args.early_stopping, verbose=False)],
        )  # early stopping callback 

        p = gbm.predict(Xva, num_iteration=gbm.best_iteration)
        oof[va] = p
        print(f"[lgbm-oddsless] fold{fold}: logloss={log_loss(yva, p):.4f}, auc={roc_auc_score(yva, p):.4f}")
        models.append(gbm)

    print(f"[lgbm-oddsless] OOF logloss={log_loss(y, oof):.4f}, auc={roc_auc_score(y, oof):.4f}")

    # 全データで学習（best_iter の中央値あたりを採用）
    best_its = [m.best_iteration or 100 for m in models]
    final_iter = int(np.median(best_its))
    final_set = lgb.Dataset(X, label=y)
    final_model = lgb.train(params, final_set, num_boost_round=final_iter)
    bundle = {"model": final_model, "scaler": None, "columns": list(X.columns)}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[lgbm-oddsless] features: {list(X.columns)}")
    print(f"[lgbm-oddsless] saved: {args.out}")

if __name__ == "__main__":
    main()