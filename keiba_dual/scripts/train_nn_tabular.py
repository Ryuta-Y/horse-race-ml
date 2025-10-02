# scripts/train_nn_tabular.py
# 使い方例:
#   (venv) $ python -m scripts.train_nn_tabular --records-glob "data/raw/record_data_*.csv" \
#                 --out models/nn_oddsless.pkl --mode oddsless
#   (venv) $ python -m scripts.train_nn_tabular --records-glob "data/raw/record_data_*.csv" \
#                 --out models/nn_oddsaware.pkl --mode oddsaware
#
# 特徴:
# - Apple Silicon(MPS) 対応
# - K-fold CV + 早期終了 + 学習曲線CSV/PNG保存
# - oddsless/oddsaware の両対応（--mode）
# - bet_policy.py がそのまま使える model_bundle を保存（predict_proba 実装）

from __future__ import annotations
import os, glob, math, argparse, json, warnings
from typing import List, Tuple
import numpy as np
import pandas as pd

import joblib
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import log_loss

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# ------------------ データ読込（ヘッダ無し/文字コード安全） ------------------
DEFAULT_COLS = [
    "race_id","year","month","day","times","place","daily","race_num","horse","jockey_id",
    "horse_N","waku_num","horse_num","class_code","track_code","corner_num","dist","state",
    "weather","age_code","sex","age","basis_weight","blinker","weight","inc_dec","weight_code",
    "win_odds","rank","time_diff","time","corner1_rank","corner2_rank","corner3_rank","corner4_rank",
    "last_3F_time","last_3F_rank","Ave_3F","PCI","last_3F_time_diff","leg","pop","prize","error_code",
    "father","mother","id"
]

def _read_csv_any(path: str) -> pd.DataFrame:
    for enc in ("cp932","shift_jis","utf-8"):
        try:
            df = pd.read_csv(path, encoding=enc, names=DEFAULT_COLS, low_memory=False)
            return df
        except Exception:
            continue
    # 最後にヘッダありの可能性も一応見る
    try:
        return pd.read_csv(path, low_memory=False)
    except Exception as e:
        raise e

def load_records(glob_pat: str) -> pd.DataFrame:
    paths = sorted(glob.glob(glob_pat))
    if not paths:
        raise RuntimeError(f"no files matched: {glob_pat}")
    df = pd.concat([_read_csv_any(p) for p in paths], axis=0, ignore_index=True)
    # 型整理
    for c in ["race_id","id","rank","horse_num","waku_num","race_num","year","month","day","dist",
              "class_code","track_code","basis_weight","weight","inc_dec","weight_code","corner1_rank",
              "corner2_rank","corner3_rank","corner4_rank","last_3F_time","last_3F_rank","Ave_3F","PCI",
              "time_diff","pop","prize","times","age","win_odds"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    # 前走順位 rank_last を付与（id->race_id 時系列）
    if "id" in df.columns and "race_id" in df.columns:
        df = df.sort_values(["id","race_id"]).copy()
        df["rank_last"] = df.groupby("id")["rank"].shift(1)
    else:
        df["rank_last"] = np.nan
    return df

# ------------------ 特徴量構築 ------------------
BASE_FEATS = [
    "age","rank_last","waku_num","horse_num","race_num","dist","class_code","track_code",
    "basis_weight","weight","inc_dec","weight_code","corner1_rank","corner2_rank","corner3_rank",
    "corner4_rank","last_3F_time","last_3F_rank","PCI","Ave_3F","time_diff","pop","prize",
    "times","month","year","day"
]
AWARE_EXTRA = ["win_odds"]           # oddsaware だけが使う列
AWARE_FLAGS = ["has_win_odds"]       # 実オッズがあるかのフラグ

def build_features(df: pd.DataFrame, mode: str) -> Tuple[pd.DataFrame, pd.Series]:
    df = df.copy()
    # 目的変数: 1着
    y = (pd.to_numeric(df.get("rank", np.nan), errors="coerce") == 1).astype(int)
    # 欠損/無効な win_odds を処理
    if "win_odds" in df.columns:
        df["has_win_odds"] = (~df["win_odds"].isna()) & (df["win_odds"] > 0)
        df.loc[~df["has_win_odds"], "win_odds"] = np.nan  # 0 や負値は欠損扱い
    # 特徴リスト
    feats = BASE_FEATS.copy()
    if mode == "oddsaware":
        feats = feats + AWARE_EXTRA + AWARE_FLAGS
    X = df.reindex(columns=feats).astype(float)
    # rank_last の欠損は0埋め（=前走情報なし）
    if "rank_last" in X.columns:
        X["rank_last"] = X["rank_last"].fillna(0.0)
    # 欠損の一般処理
    X = X.fillna(X.median(numeric_only=True))
    return X, y

# ------------------ NN モデル ------------------
class MLP(nn.Module):
    def __init__(self, in_dim: int, hidden: List[int]=(256,128,64), dropout: float=0.2):
        super().__init__()
        layers = []
        d = in_dim
        for h in hidden:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(inplace=True), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)
    def forward(self, x):
        return self.net(x).squeeze(1)  # logits

class FocalLoss(nn.Module):
    # 2-class focal loss (binary)
    def __init__(self, alpha=0.25, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        p = torch.sigmoid(logits)
        pt = torch.where(targets==1, p, 1-p)
        loss = self.alpha * (1-pt)**self.gamma * bce
        return loss.mean() if self.reduction=="mean" else loss.sum()

class TabularDataset(Dataset):
    def __init__(self, X: np.ndarray, y: np.ndarray):
        self.X = X.astype(np.float32)
        self.y = y.astype(np.float32)
    def __len__(self): return len(self.X)
    def __getitem__(self, i):
        return self.X[i], self.y[i]

class NNPredictor:
    """bet_policy の _predict_win_prob から呼ばれる predict_proba を提供。"""
    def __init__(self, in_dim: int, hidden: List[int], dropout: float, state_dict: dict, device: str="cpu"):
        self.in_dim, self.hidden, self.dropout = in_dim, hidden, dropout
        self.device = torch.device(device)
        self.model = MLP(in_dim, hidden, dropout)
        self.model.load_state_dict(state_dict)
        self.model.to(self.device)
        self.model.eval()
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        with torch.no_grad():
            t = torch.tensor(X.astype(np.float32), device=self.device)
            logits = self.model(t)
            p = torch.sigmoid(logits).cpu().numpy()
        # 返すのは shape (n, 2) ではなく (n,) の正例確率でも OK（_predict_win_prob 側で [:,1] を想定していない）
        # 互換のため (n,2) を返す
        p1 = p.reshape(-1,1)
        p0 = 1.0 - p1
        return np.hstack([p0, p1])

# ------------------ 学習ループ ------------------
def train_fold(Xtr, ytr, Xva, yva, in_dim, device, args, run_dir, fold_idx):
    train_ds = TabularDataset(Xtr, ytr)
    val_ds   = TabularDataset(Xva, yva)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, num_workers=0)

    model = MLP(in_dim, hidden=args.hidden, dropout=args.dropout).to(device)
    if args.loss == "focal":
        criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma)
    else:
        criterion = nn.BCEWithLogitsLoss()

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=max(10, args.epochs))

    best_loss = 1e9
    patience = args.patience
    no_improve = 0
    history = []

    for epoch in range(1, args.epochs+1):
        model.train()
        tr_losses = []
        for xb, yb in train_loader:
            xb = xb.to(device)
            yb = yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits = model(xb)
            loss = criterion(logits, yb)
            loss.backward()
            opt.step()
            tr_losses.append(loss.item())
        scheduler.step()

        # validation
        model.eval()
        va_losses = []
        with torch.no_grad():
            for xb, yb in val_loader:
                xb = xb.to(device); yb = yb.to(device)
                logits = model(xb)
                loss = criterion(logits, yb)
                va_losses.append(loss.item())
        tr = float(np.mean(tr_losses)) if tr_losses else np.nan
        va = float(np.mean(va_losses)) if va_losses else np.nan
        history.append((epoch, tr, va))
        print(f"[fold{fold_idx}] epoch {epoch:03d} train {tr:.4f} val {va:.4f}")

        if va < best_loss - 1e-5:
            best_loss = va
            no_improve = 0
            torch.save(model.state_dict(), os.path.join(run_dir, f"best_fold{fold_idx}.pt"))
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"[fold{fold_idx}] early stop at epoch {epoch}")
                break

    # 学習曲線保存
    hist_df = pd.DataFrame(history, columns=["epoch","train_loss","val_loss"])
    hist_df.to_csv(os.path.join(run_dir, f"curve_fold{fold_idx}.csv"), index=False)
    try:
        import matplotlib.pyplot as plt
        plt.figure()
        plt.plot(hist_df["epoch"], hist_df["train_loss"], label="train")
        plt.plot(hist_df["epoch"], hist_df["val_loss"], label="val")
        plt.xlabel("epoch"); plt.ylabel("loss"); plt.legend(); plt.tight_layout()
        plt.savefig(os.path.join(run_dir, f"curve_fold{fold_idx}.png"))
        plt.close()
    except Exception:
        pass

    # best を読み戻し
    state = torch.load(os.path.join(run_dir, f"best_fold{fold_idx}.pt"), map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    # 検証 logloss （スカラー）も返す
    with torch.no_grad():
        t = torch.tensor(Xva.astype(np.float32))
        logits = model(t)
        p = torch.sigmoid(logits).numpy()
        val_ll = log_loss(yva, p, eps=1e-7)
    return state, val_ll

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--records-glob", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--mode", choices=["oddsless","oddsaware"], default="oddsless")
    ap.add_argument("--epochs", type=int, default=120)
    ap.add_argument("--batch-size", type=int, default=512)
    ap.add_argument("--lr", type=float, default=3e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--hidden", type=int, nargs="+", default=[256,128,64])
    ap.add_argument("--dropout", type=float, default=0.2)
    ap.add_argument("--kfold", type=int, default=5)
    ap.add_argument("--patience", type=int, default=16)
    ap.add_argument("--loss", choices=["bce","focal"], default="bce")
    ap.add_argument("--focal-alpha", type=float, default=0.25)
    ap.add_argument("--focal-gamma", type=float, default=2.0)
    args = ap.parse_args()

    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    print(f"[nn] device={device}")

    rec = load_records(args.records_glob)
    X_raw, y = build_features(rec, mode=args.mode)
    feature_cols = list(X_raw.columns)

    # スケーラは fold ごとに作り、最後に全データで作り直して保存
    skf = StratifiedKFold(n_splits=args.kfold, shuffle=True, random_state=42)
    run_dir = os.path.join("runs","nn", os.path.splitext(os.path.basename(args.out))[0])
    os.makedirs(run_dir, exist_ok=True)

    fold_losses = []
    states = []
    for f, (tr_idx, va_idx) in enumerate(skf.split(X_raw, y), start=1):
        scaler = StandardScaler()
        Xtr = scaler.fit_transform(X_raw.iloc[tr_idx].values)
        Xva = scaler.transform( X_raw.iloc[va_idx].values)
        ytr = y.iloc[tr_idx].values.astype(np.float32)
        yva = y.iloc[va_idx].values.astype(np.float32)
        state, val_ll = train_fold(Xtr, ytr, Xva, yva, Xtr.shape[1], device, args, run_dir, f)
        fold_losses.append(val_ll); states.append(state)
        print(f"[fold{f}] val logloss = {val_ll:.4f}")

    print(f"[cv] mean val logloss = {np.mean(fold_losses):.4f} (+/- {np.std(fold_losses):.4f})")

    # ---- 全データで再学習 -> バンドル保存（bet_policy互換） ----
    scaler = StandardScaler().fit(X_raw.values)
    X_all = scaler.transform(X_raw.values).astype(np.float32)
    ds = TabularDataset(X_all, y.values.astype(np.float32))
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=True)

    model = MLP(X_all.shape[1], hidden=args.hidden, dropout=args.dropout).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    criterion = FocalLoss(alpha=args.focal_alpha, gamma=args.focal_gamma) if args.loss=="focal" else nn.BCEWithLogitsLoss()

    model.train()
    for epoch in range(1, max(20, args.epochs//2)+1):
        losses=[]
        for xb,yb in loader:
            xb=xb.to(device); yb=yb.to(device)
            opt.zero_grad(set_to_none=True)
            logits=model(xb); loss=criterion(logits,yb)
            loss.backward(); opt.step()
            losses.append(loss.item())
        if epoch%10==0:
            print(f"[final-train] epoch {epoch} loss {np.mean(losses):.4f}")

    state = model.state_dict()
    predictor = NNPredictor(
        in_dim=X_all.shape[1],
        hidden=args.hidden,
        dropout=args.dropout,
        state_dict=state,
        device="mps" if torch.backends.mps.is_available() else "cpu"
    )
    bundle = {
        "model": predictor,
        "scaler": scaler,
        "columns": feature_cols,
    }
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    joblib.dump(bundle, args.out)
    print(f"[nn] saved: {args.out}")
    # CV結果の概要も保存
    with open(os.path.join(run_dir, "cv_summary.json"), "w", encoding="utf-8") as f:
        json.dump({"val_logloss_mean":float(np.mean(fold_losses)),
                   "val_logloss_std":float(np.std(fold_losses)),
                   "fold_logloss":list(map(float, fold_losses))}, f, ensure_ascii=False, indent=2)

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=UserWarning)
    main()