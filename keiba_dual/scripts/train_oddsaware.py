#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
odds を“使う” win 確率モデル（ロジスティック回帰）。
- 5-fold OOF（logloss/AUC）
- --calibrate で isotonic
- 特徴量は数値列＋odds列を積極採用（rank/y_win/ID系は除外）
"""

from __future__ import annotations
import argparse, os
from typing import Any, Optional, Tuple, List

import joblib, numpy as np, pandas as pd
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

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
    try:
        arr = np.asarray(x); return arr.ndim==1 and arr.size>0
    except Exception: 
        return False

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

def _build_feature_df_oddsaware(rec_df: pd.DataFrame, X_df_in: Optional[pd.DataFrame]) -> pd.DataFrame:
    def to_num(s: pd.Series) -> pd.Series:
        if not pd.api.types.is_numeric_dtype(s): return pd.to_numeric(s, errors="coerce")
        return s

    X = (X_df_in.copy() if X_df_in is not None else pd.DataFrame(index=rec_df.index))
    for c in list(X.columns):
        X[c] = to_num(X[c])

    # rec_df から補完（数値列は全部採用、odds も含む）
    for c in rec_df.columns:
        if c in _DROP_ALWAYS: 
            continue
        if c not in X.columns:
            X[c] = to_num(rec_df[c])

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    if X.shape[1]==0:
        raise RuntimeError("有効な数値特徴量が1列も作れませんでした（oddsaware）。")

    keep = [c for c in X.columns if (X[c] != 0).sum() > 0 and c not in _DROP_ALWAYS]
    X = X[keep]
    print(f"[oddsaware] feature columns used: {keep[:12]}{'...' if len(keep)>12 else ''} (total {len(keep)})")
    return X

def _fit_oof(X: pd.DataFrame, y: np.ndarray, calibrate: bool) -> np.ndarray:
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    oof = np.zeros(len(X), dtype=float)
    for i,(tr,va) in enumerate(kf.split(X,y), start=1):
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", LogisticRegression(max_iter=1000, class_weight="balanced", C=1.0, solver="lbfgs")),
        ])
        model = CalibratedClassifierCV(estimator=clone(base), cv=3, method="isotonic") if calibrate else clone(base)
        model.fit(X.iloc[tr], y[tr])
        p = model.predict_proba(X.iloc[va])[:,1]
        oof[va]=p
        ll=log_loss(y[va],p,labels=[0,1])
        try: auc=roc_auc_score(y[va],p)
        except Exception: auc=float("nan")
        print(f"[oddsaware] fold{i}: logloss={ll:.4f}, auc={auc:.4f}")
    ll_all=log_loss(y,oof,labels=[0,1])
    try: auc_all=roc_auc_score(y,oof)
    except Exception: auc_all=float("nan")
    print(f"[oddsaware] OOF logloss={ll_all:.4f}, auc={auc_all:.4f}")
    return oof

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--records-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--calibrate", action="store_true")
    args=ap.parse_args()

    rec_df=load_records(args.records_glob)
    feats_obj=build_features(rec_df)
    X_df_in, y_hint = _split_feats_obj(feats_obj)
    y = _coerce_y(rec_df, X_df_in, y_hint)
    if len(np.unique(y)) < 2:
        raise RuntimeError("y が単一クラスです。学習できません。")

    X = _build_feature_df_oddsaware(rec_df, X_df_in)

    _fit_oof(X, y, calibrate=args.calibrate)

    base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", LogisticRegression(max_iter=2000, class_weight="balanced", solver="lbfgs")),
    ])
    final_model = CalibratedClassifierCV(estimator=base, cv=5, method="isotonic") if args.calibrate else base
    final_model.fit(X, y)

    bundle = {
        "model": final_model,
        "columns": list(X.columns),
        "meta": {
            "calibrated": bool(args.calibrate),
            "model_type": "logreg(oddsaware)",
            "cv_folds": 5,
        },
    }
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[oddsaware] saved: {args.out}")

if __name__ == "__main__":
    main()