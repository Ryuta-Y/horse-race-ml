# keiba_dual_ready_full/scripts/train_nn.py
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Tuple, Union, Optional, Any, List

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import log_loss, roc_auc_score
from sklearn.calibration import CalibratedClassifierCV
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline

# ---- io_utils インポート（堅牢） ----
_THIS = Path(__file__).resolve()
ROOT = _THIS.parents[1]
for p in {str(ROOT), str(ROOT / "scripts"), str(ROOT / "scripts" / "utils")}:
    if p not in sys.path: sys.path.append(p)

def _import_io_utils():
    for mod in ("scripts.utils.io_utils", "utils.io_utils", "io_utils"):
        try:
            io_utils = __import__(mod, fromlist=["load_records", "build_features"])
            return getattr(io_utils, "load_records"), getattr(io_utils, "build_features")
        except Exception:
            pass
    raise ImportError("io_utils の import に失敗しました。scripts/utils/io_utils.py を確認してください。")
load_records, build_features = _import_io_utils()

# ---- PyTorch ----
try:
    import torch
    import torch.nn as nn
    from torch.utils.data import TensorDataset, DataLoader
except Exception as e:
    raise ImportError("PyTorch が見つかりません。`pip install torch` を実行してください。") from e

# ---- 便利関数 ----
_FALLBACK_COLS_ORDERED = [
    "year","month","day","times","race_num","waku_num","horse_num",
    "class_code","track_code","corner_num","dist","age","rank_last",
    "corner1_rank","corner2_rank","corner3_rank","corner4_rank",
    "last_3F_time","last_3F_rank","PCI","Ave_3F","time_diff",
    "pop","prize","win_odds","place_odds",
]

def _as_dataframe(obj: Any) -> pd.DataFrame:
    if isinstance(obj, pd.DataFrame): return obj.copy()
    if isinstance(obj, (list, tuple)) and obj and isinstance(obj[0], pd.DataFrame): return obj[0].copy()
    if isinstance(obj, dict):
        if "X" in obj and isinstance(obj["X"], pd.DataFrame): return obj["X"].copy()
        if "df" in obj and isinstance(obj["df"], pd.DataFrame): return obj["df"].copy()
    raise RuntimeError("build_features の戻りに DataFrame を含む形式が必要です。")

def _as_target(obj: Any, rec_df: pd.DataFrame) -> np.ndarray:
    if isinstance(obj, (list, tuple)) and len(obj) >= 2:
        return np.asarray(obj[1]).astype(int)
    if isinstance(obj, dict) and "y" in obj:
        return np.asarray(obj["y"]).astype(int)
    try:
        df = _as_dataframe(obj)
        if "y_win" in df.columns:
            return pd.to_numeric(df["y_win"], errors="coerce").fillna(0).astype(int).values
    except Exception:
        pass
    if "rank" in rec_df.columns:
        return (pd.to_numeric(rec_df["rank"], errors="coerce") == 1).astype(int).values
    raise RuntimeError("目的変数 y を特定できませんでした。")

def _ensure_dataframe_features(rec_df: pd.DataFrame, feats_obj: Any, use_odds: bool):
    feats_df = _as_dataframe(feats_obj)
    y = _as_target(feats_obj, rec_df)
    for c in feats_df.columns:
        if feats_df[c].dtype == object:
            feats_df[c] = pd.to_numeric(feats_df[c], errors="coerce")
    drop_cols = {"y_win","rank","time","race_id","race_id_full","race_id_odds","race_id_bet",
                 "horse","jockey","jockey_id","father","mother","id","error_code","weather","state",
                 "sex","leg","place","daily","class_name","track_name"}
    odds_cols = {c for c in feats_df.columns if "odds" in c.lower()}
    if not use_odds: drop_cols |= odds_cols
    num_cols = [c for c in feats_df.columns if c not in drop_cols and pd.api.types.is_numeric_dtype(feats_df[c])]
    X_df = feats_df[num_cols].copy()
    if X_df.shape[1] == 0:
        df2 = rec_df.copy()
        for c in df2.columns:
            if df2[c].dtype == object:
                df2[c] = pd.to_numeric(df2[c], errors="coerce")
        fb = []
        for c in _FALLBACK_COLS_ORDERED:
            if c in df2.columns:
                if (not use_odds) and ("odds" in c.lower()): continue
                if pd.api.types.is_numeric_dtype(df2[c]): fb.append(c)
        if not fb:
            for c in df2.columns:
                if c in drop_cols: continue
                if (not use_odds) and ("odds" in c.lower()): continue
                if pd.api.types.is_numeric_dtype(df2[c]): fb.append(c)
        if not fb:
            raise RuntimeError("有効な数値特徴量が見つかりませんでした。")
        X_df = df2[fb].copy()
    X_df = X_df.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    print(f"[nn-odds{'aware' if use_odds else 'less'}] feature columns used: {list(X_df.columns)[:12]}{'...' if X_df.shape[1]>12 else ''} (total {X_df.shape[1]})")
    return X_df.astype(np.float32), y.astype(int)

def _get_device():
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available(): return torch.device("mps")
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ---- NN 本体 ----
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden=(256,128,64), drop=0.1):
        super().__init__()
        layers: List[nn.Module] = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, int(h)), nn.ReLU(), nn.Dropout(float(drop))]
            d = int(h)
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(-1)

class TorchNNClassifier(BaseEstimator, ClassifierMixin):
    def __init__(self, input_dim, hidden=(256,128,64), drop=0.1,
                 lr=1e-3, weight_decay=1e-5, batch_size=512, epochs=60, patience=10,
                 random_state=42, device=None):
        # ★sklearn準拠：__init__では値を一切変形しない（clone互換）
        self.input_dim = input_dim
        self.hidden = hidden
        self.drop = drop
        self.lr = lr
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.epochs = epochs
        self.patience = patience
        self.random_state = random_state
        self.device = device         # cloneが参照するので公開属性で保持
        self._estimator_type = "classifier"
        self.model_ = None
        self.classes_ = None

    def _to_tensor(self, X, y=None, device=None):
        X = torch.tensor(np.asarray(X, dtype=np.float32), device=device)
        y = None if y is None else torch.tensor(np.asarray(y, dtype=np.float32), device=device)
        return X, y

    def fit(self, X, y):
        # ここで初めて型を確定（__init__では変換しない）
        input_dim = int(self.input_dim)
        if isinstance(self.hidden, (list, tuple, np.ndarray)):
            hidden = tuple(int(h) for h in self.hidden)
        else:
            hidden = (int(self.hidden),)
        drop = float(self.drop)
        lr = float(self.lr)
        weight_decay = float(self.weight_decay)
        batch_size = int(self.batch_size)
        epochs = int(self.epochs)
        patience = int(self.patience)
        random_state = int(self.random_state)
        device = self.device or _get_device()

        torch.manual_seed(random_state); np.random.seed(random_state)
        model = MLP(input_dim, hidden, drop).to(device)
        self.model_ = model
        self.classes_ = np.array([0, 1], dtype=int)

        pos = float(np.sum(y)); neg = float(len(y) - pos)
        pos_weight = torch.tensor(max(1e-6, neg / max(1.0, pos)), device=device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)

        X_t, y_t = self._to_tensor(X, y, device=device)
        ds = TensorDataset(X_t, y_t)
        dl = DataLoader(ds, batch_size=batch_size, shuffle=True)

        best = float("inf"); noimp = 0; best_state = None
        for ep in range(epochs):
            model.train(); run = 0.0
            for xb, yb in dl:
                opt.zero_grad()
                logit = model(xb)
                loss = criterion(logit, yb)
                loss.backward(); opt.step()
                run += float(loss.detach().cpu()) * len(xb)
            eloss = run / len(ds)
            if eloss + 1e-6 < best:
                best = eloss; noimp = 0
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            else:
                noimp += 1
                if noimp >= patience:
                    break
        if best_state is not None:
            model.load_state_dict({k: v.to(device) for k, v in best_state.items()})
        return self

    def predict_proba(self, X):
        assert self.model_ is not None, "Call fit() first."
        device = next(self.model_.parameters()).device
        X_t, _ = self._to_tensor(X, None, device=device)
        self.model_.eval()
        with torch.no_grad():
            p1 = torch.sigmoid(self.model_(X_t)).float().cpu().numpy()
        p1 = np.clip(p1, 1e-7, 1 - 1e-7)
        p0 = 1.0 - p1
        return np.stack([p0, p1], axis=1)

    def predict(self, X):
        p1 = self.predict_proba(X)[:, 1]
        return (p1 >= 0.5).astype(int)

# ---- メイン ----
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-glob", default="data/raw/record_data_*.csv")
    ap.add_argument("--out", default="models/nn_oddsless.pkl")
    ap.add_argument("--use-odds", type=int, choices=[0,1], default=0)
    ap.add_argument("--calibrate", action="store_true")
    ap.add_argument("--kfolds", type=int, default=5)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--hidden", type=str, default="256,128,64")
    ap.add_argument("--drop", type=float, default=0.1)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--epochs", type=int, default=60)
    ap.add_argument("--patience", type=int, default=10)
    args = ap.parse_args()

    rec_df = load_records(args.records_glob)
    feats_obj = build_features(rec_df)
    X_df, y = _ensure_dataframe_features(rec_df, feats_obj, use_odds=bool(args.use_odds))
    X = X_df.values
    feature_cols = list(X_df.columns)
    hidden_tuple = tuple(int(x) for x in args.hidden.split(",") if x.strip())

    skf = StratifiedKFold(n_splits=args.kfolds, shuffle=True, random_state=args.random_state)
    oof = np.zeros(len(y), dtype=np.float32)
    fold = 1
    for tr, va in skf.split(X, y):
        base = Pipeline([
            ("scaler", StandardScaler()),
            ("clf", TorchNNClassifier(
                input_dim=X.shape[1],
                hidden=hidden_tuple,
                drop=args.drop,
                lr=args.lr,
                weight_decay=args.weight_decay,
                batch_size=args.batch_size,
                epochs=args.epochs,
                patience=args.patience,
                random_state=args.random_state + fold,
                device=None
            ))
        ])
        est = CalibratedClassifierCV(base, method="sigmoid", cv=3) if args.calibrate else base
        est.fit(X[tr], y[tr])
        p = est.predict_proba(X[va])[:, 1]
        oof[va] = p
        ll = log_loss(y[va], p, labels=[0,1])
        try:
            auc = roc_auc_score(y[va], p)
        except Exception:
            auc = float("nan")
        print(f"[nn-odds{'aware' if args.use_odds else 'less'}] fold{fold}: logloss={ll:.4f}, auc={auc:.4f}")
        fold += 1
    ll = log_loss(y, oof, labels=[0,1])
    try:
        auc = roc_auc_score(y, oof)
    except Exception:
        auc = float("nan")
    print(f"[nn-odds{'aware' if args.use_odds else 'less'}] OOF logloss={ll:.4f}, auc={auc:.4f}")

    final_base = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", TorchNNClassifier(
            input_dim=X.shape[1],
            hidden=hidden_tuple,
            drop=args.drop,
            lr=args.lr,
            weight_decay=args.weight_decay,
            batch_size=args.batch_size,
            epochs=args.epochs,
            patience=args.patience,
            random_state=args.random_state,
            device=None
        ))
    ])
    final_est = CalibratedClassifierCV(final_base, method="sigmoid", cv=args.kfolds) if args.calibrate else final_base
    final_est.fit(X, y)

    bundle = {"model": final_est, "scaler": None, "columns": feature_cols,
              "meta": {"framework": "pytorch+sklearn-pipeline",
                       "use_odds": bool(args.use_odds),
                       "calibrated": bool(args.calibrate),
                       "kfolds": args.kfolds}}
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    joblib.dump(bundle, args.out)
    if len(feature_cols) > 12:
        print(f"[nn] features: {feature_cols[:12]}... (+{len(feature_cols)-12} more)")
    else:
        print(f"[nn] features: {feature_cols}")
    print(f"[nn] saved: {args.out}")

if __name__ == "__main__":
    main()