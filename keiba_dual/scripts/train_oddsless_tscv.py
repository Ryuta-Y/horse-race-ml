# keiba_dual_ready_full/scripts/train_oddsless_tscv.py
# ----------------------------------------------------
# 時系列CV(TimeSeriesSplit)で評価しつつ、全期間で再学習して保存。
# 保存: models/oddsless_tscv.pkl / 履歴: models/oddsless_tscv_history.csv, .png
# 実行: (venv) keiba_dual_ready_full $ python -m scripts.train_oddsless_tscv

import os, glob, math
import numpy as np
import pandas as pd
import joblib

from conf import MODEL_DIR
from conf import ODDSLESS_MODEL_PATH as _DUMMYPATH  # 使わないが依存を満たす
ODDSLESS_TSCV_MODEL_PATH = os.path.join(MODEL_DIR, "oddsless_tscv.pkl")

# ===== 調整パラメータ =====
INPUT_GLOBS = ["./data/raw/record_data_*.csv", "./input_data/record_data_*.csv"]
RANDOM_STATE = 17
N_SPLITS = 5
EPOCHS = 20
BATCH_SIZE = 4096
LEARNING_RATE = 0.01
CALIB_BUCKETS = 10
# ========================

# 既存学習の共通関数を軽く流用（列定義や加工は oddsless と同様）
from scripts.train_oddsless import (
    NUM_COLS, CAT_COLS,
    logloss, calibration_error,
)

def load_many_csv(patterns):
    paths=[]
    for p in patterns: paths+=glob.glob(p)
    if not paths: raise FileNotFoundError("学習CSVが見つかりません")
    col_list = ["race_id","year","month","day","times","place","daily","race_num",
                "horse","jockey_id","horse_N","waku_num","horse_num","class_code",
                "track_code","corner_num","dist","state","weather","age_code",
                "sex","age","basis_weight","blinker","weight","inc_dec","weight_code",
                "win_odds","rank","time_diff","time","corner1_rank","corner2_rank",
                "corner3_rank","corner4_rank","last_3F_time","last_3F_rank","Ave_3F",
                "PCI","last_3F_time_diff","leg","pop","prize","error_code","father","mother","id"]
    dfs = [pd.read_csv(p, encoding="shift_jis", names=col_list, low_memory=False) for p in paths]
    return pd.concat(dfs, axis=0, ignore_index=True)

def add_rank_last(df):
    dat = df.sort_values(["id","race_id"]).copy()
    dat["rank_last"] = dat.groupby("id")[["rank"]].shift(1)
    dat["rank_last"] = dat["rank_last"].fillna(0)
    return dat

def make_features(df):
    y = (df["rank"]==1).astype(int)
    Xnum = df[[c for c in NUM_COLS if c in df.columns]].apply(pd.to_numeric, errors="coerce").fillna(0.0)
    Xnum["rank_last"] = pd.to_numeric(df["rank_last"], errors="coerce").fillna(0.0)
    Xcat = []
    for col in [col for col in CAT_COLS if col in df.columns]:
        m = df.groupby(col)["rank"].apply(lambda s: (s==1).mean()).to_dict()
        Xcat.append(df[col].map(m).fillna(0.0).rename(f"{col}_mean1"))
    X = pd.concat([Xnum]+Xcat, axis=1).fillna(0.0)
    return X, y

def sgd_fit(X, y):
    from sklearn.preprocessing import StandardScaler
    from sklearn.linear_model import SGDClassifier
    scaler = StandardScaler()
    Xs = scaler.fit_transform(X.values)
    cls = SGDClassifier(loss="log_loss", learning_rate="constant", eta0=LEARNING_RATE, random_state=RANDOM_STATE)
    n = Xs.shape[0]; idx = np.arange(n)
    # ざっくりEPOCHS回転
    for ep in range(EPOCHS):
        np.random.shuffle(idx)
        for st in range(0, n, BATCH_SIZE):
            ed = min(st+BATCH_SIZE, n)
            Xb = Xs[idx[st:ed]]; yb = y.values[idx[st:ed]]
            if ep==0 and st==0: cls.partial_fit(Xb, yb, classes=np.array([0,1]))
            else: cls.partial_fit(Xb, yb)
    return {"model": cls, "scaler": scaler, "columns": list(X.columns)}

def main():
    os.makedirs(MODEL_DIR, exist_ok=True)
    df = load_many_csv(INPUT_GLOBS)
    df = add_rank_last(df)

    # 時系列K分割（年×月×日で昇順）
    df = df.sort_values(["year","month","day","race_id"]).reset_index(drop=True)
    X, y = make_features(df)

    # TimeSeriesSplit
    from sklearn.model_selection import TimeSeriesSplit
    tscv = TimeSeriesSplit(n_splits=N_SPLITS)
    rows=[]
    for k, (tr, va) in enumerate(tscv.split(X), start=1):
        Xtr, Xva = X.iloc[tr], X.iloc[va]
        ytr, yva = y.iloc[tr], y.iloc[va]
        bundle = sgd_fit(Xtr, ytr)
        # 評価
        m, sc = bundle["model"], bundle["scaler"]
        pva = m.predict_proba(sc.transform(Xva.values))[:,1]
        from sklearn.metrics import roc_auc_score
        row = {
            "fold": k,
            "valid_logloss": logloss(yva.values, pva),
            "valid_auc": roc_auc_score(yva.values, pva) if len(np.unique(yva.values))>1 else np.nan,
            "calibration_err_valid": calibration_error(yva.values, pva, bins=CALIB_BUCKETS),
        }
        rows.append(row)

    hist = pd.DataFrame(rows)
    hist_path = os.path.join(MODEL_DIR, "oddsless_tscv_history.csv")
    hist.to_csv(hist_path, index=False)

    # 全期間で再学習→保存
    bundle = sgd_fit(X, y)
    joblib.dump(bundle, ODDSLESS_TSCV_MODEL_PATH)

    # グラフ（任意）
    try:
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(7,4))
        ax.plot(hist["fold"], hist["valid_logloss"], marker="o", label="valid_logloss")
        ax.set_xlabel("fold"); ax.set_ylabel("logloss"); ax.grid(True); ax.legend()
        fig.tight_layout(); fig.savefig(os.path.join(MODEL_DIR,"oddsless_tscv_history.png")); plt.close(fig)
    except Exception as e:
        print("plot skipped:", e)

    print(f"saved: {ODDSLESS_TSCV_MODEL_PATH}")
    print(f"history csv: {hist_path}")

if __name__=="__main__":
    main()