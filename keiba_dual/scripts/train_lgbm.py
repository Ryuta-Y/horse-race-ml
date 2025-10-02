#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
LightGBM ベースの win 確率モデル。
- --use-odds 0/1 で oddsless / oddsaware を切替
- 5-fold OOF ログ
- --calibrate で CalibratedClassifierCV（sigmoid）
- LightGBM が無い環境では HistGradientBoostingClassifier に自動フォールバック
"""

from __future__ import annotations
import argparse, os
from typing import Any, Optional, Tuple, List

import joblib, numpy as np, pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold

# lgbm or fallback
_LGBM_AVAILABLE = True
try:
    from lightgbm import LGBMClassifier
except Exception:
    _LGBM_AVAILABLE = False
    from sklearn.ensemble import HistGradientBoostingClassifier

def _import_io_utils():
    for path in ("scripts.utils.io_utils", "utils.io_utils", "io_utils"):
        try:
            mod = __import__(path, fromlist=["load_records", "build_features"])
            return getattr(mod, "load_records"), getattr(mod, "build_features")
        except Exception:
            pass
    raise ModuleNotFoundError("io_utils が見つかりません。scripts/utils/io_utils.py を配置してください。")

load_records, build_features = _import_io_utils()

def _is_1d_array_like(x): 
    try: arr=np.asarray(x); return arr.ndim==1 and arr.size>0
    except Exception: return False

def _split_feats_obj(o: Any) -> Tuple[Optional[pd.DataFrame], Optional[np.ndarray]]:
    X_df=None; y=None
    if isinstance(o,pd.DataFrame): X_df=o
    elif isinstance(o,(list,tuple)):
        for el in o:
            if isinstance(el,pd.DataFrame) and X_df is None: X_df=el
        for el in o:
            if isinstance(el,pd.Series) and y is None: y=el.to_numpy()
            elif _is_1d_array_like(el) and y is None: y=np.asarray(el)
    elif isinstance(o,dict):
        for k in ("X","df","features"):
            if k in o and isinstance(o[k],pd.DataFrame): X_df=o[k]; break
        for k in ("y","target"):
            if k in o and _is_1d_array_like(o[k]): y=np.asarray(o[k]); break
    return X_df, y

def _coerce_y(rec: pd.DataFrame, X: Optional[pd.DataFrame], y_hint: Optional[np.ndarray]) -> np.ndarray:
    if y_hint is not None:
        y = pd.to_numeric(pd.Series(y_hint), errors="coerce").fillna(0).astype(float).to_numpy()
        return (y==1).astype(int)
    if X is not None and "y_win" in X.columns:
        y = pd.to_numeric(X["y_win"], errors="coerce").fillna(0).astype(float).to_numpy()
        return (y==1).astype(int)
    if "rank" in rec.columns:
        return (pd.to_numeric(rec["rank"], errors="coerce")==1).astype(int).to_numpy()
    raise RuntimeError("目的変数が作れませんでした。")

_DROP_ALWAYS = {
    "y_win", "rank", "race_id", "race_id_full", "race_id_odds", "race_id_bet",
    "id", "horse", "jockey", "jockey_id", "father", "mother",
    "error_code", "weather", "state", "sex", "leg", "place", "daily",
    "class_name", "track_name", "time"
}

def _build_features(rec_df: pd.DataFrame, X_df_in: Optional[pd.DataFrame], use_odds: bool) -> pd.DataFrame:
    def to_num(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce")
        return s
    X = (X_df_in.copy() if X_df_in is not None else pd.DataFrame(index=rec_df.index))
    for c in list(X.columns): X[c] = to_num(X[c])

    for c in rec_df.columns:
        if c in _DROP_ALWAYS: 
            continue
        if (not use_odds) and ("odds" in c.lower()): 
            continue
        if c not in X.columns:
            X[c] = to_num(rec_df[c])

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    keep = [c for c in X.columns if (X[c] != 0).sum()>0 and c not in _DROP_ALWAYS]
    if not keep: raise RuntimeError("有効な数値特徴量が1列も作れませんでした。")
    X = X[keep]
    tag = "oddsaware" if use_odds else "oddsless"
    print(f"[lgbm-{tag}] feature columns used: {keep[:12]}{'...' if len(keep)>12 else ''} (total {len(keep)})")
    return X

def _make_estimator(use_odds: bool, random_state: int):
    if _LGBM_AVAILABLE:
        return LGBMClassifier(
            objective="binary",
            n_estimators=2000,
            learning_rate=0.05,
            num_leaves=63,
            min_child_samples=20,
            max_depth=-1,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.0,
            reg_lambda=0.0,
            random_state=random_state,
            n_jobs=-1,
        ), "lightgbm"
    else:
        # フォールバック（scikit-learn 標準）
        return HistGradientBoostingClassifier(
            loss="log_loss",
            max_depth=None,
            max_leaf_nodes=63,
            learning_rate=0.05,
            l2_regularization=0.0,
            max_iter=500,
            random_state=random_state,
        ), "sklearn-hgbt"

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--records-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--use-odds", type=int, choices=[0,1], default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    args=ap.parse_args()

    rec_df=load_records(args.records_glob)
    feats_obj=build_features(rec_df)
    X_in, y_hint = _split_feats_obj(feats_obj)
    y = _coerce_y(rec_df, X_in, y_hint)
    if len(np.unique(y))<2: raise RuntimeError("y が単一クラスです。学習できません。")

    X = _build_features(rec_df, X_in, use_odds=bool(args.use_odds))
    Xv = X.values

    # OOF
    skf=StratifiedKFold(n_splits=args.kfolds,shuffle=True,random_state=args.random_state)
    oof=np.zeros(len(y),dtype=float)
    tag = "oddsaware" if args.use_odds else "oddsless"
    for i,(tr,va) in enumerate(skf.split(Xv,y), start=1):
        base, fw = _make_estimator(bool(args.use_odds), args.random_state+i)
        est = CalibratedClassifierCV(base, method="sigmoid", cv=3) if args.calibrate else base
        est.fit(Xv[tr], y[tr])
        p = est.predict_proba(Xv[va])[:,1]
        oof[va]=p
        ll=log_loss(y[va],p,labels=[0,1])
        try: auc=roc_auc_score(y[va],p)
        except Exception: auc=float("nan")
        print(f"[lgbm-{tag}] fold{i}: logloss={ll:.4f}, auc={auc:.4f} ({fw})")
    ll=log_loss(y,oof,labels=[0,1])
    try: auc=roc_auc_score(y,oof)
    except Exception: auc=float("nan")
    print(f"[lgbm-{tag}] OOF logloss={ll:.4f}, auc={auc:.4f}")

    # 全量学習
    base, fw = _make_estimator(bool(args.use_odds), args.random_state)
    final = CalibratedClassifierCV(base, method="sigmoid", cv=args.kfolds) if args.calibrate else base
    final.fit(Xv, y)

    bundle = {
        "model": final,
        "columns": list(X.columns),
        "meta": {
            "framework": fw,
            "use_odds": bool(args.use_odds),
            "calibrated": bool(args.calibrate),
            "model_type": f"{fw}({'oddsaware' if args.use_odds else 'oddsless'})",
            "cv_folds": int(args.kfolds),
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[lgbm-{tag}] saved: {args.out}")

if __name__=="__main__":
    main()