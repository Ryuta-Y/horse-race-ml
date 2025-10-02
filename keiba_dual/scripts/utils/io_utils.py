# io_utils.py
# record_data_*.csv（ヘッダ無し, Shift_JIS/CP932想定）を安全に読み込み、
# 学習・推論で使う素性に加工します。

from __future__ import annotations
import glob
import os
from typing import List, Tuple, Dict

import pandas as pd
import numpy as np

# 仕様準拠の列（先頭行にヘッダは無い想定）
COLS = [
    "race_id", "year", "month", "day", "times",
    "place", "daily", "race_num", "horse", "jockey_id",
    "horse_N", "waku_num", "horse_num", "class_code", "track_code",
    "corner_num", "dist", "state", "weather", "age_code",
    "sex", "age", "basis_weight", "blinker", "weight",
    "inc_dec", "weight_code", "win_odds", "rank", "time_diff",
    "time", "corner1_rank", "corner2_rank", "corner3_rank", "corner4_rank",
    "last_3F_time", "last_3F_rank", "Ave_3F", "PCI", "last_3F_time_diff",
    "leg", "pop", "prize", "error_code",
    "father", "mother", "id"
]

# 数値化を強制したい列
_NUMERIC_COLS = {
    "year", "month", "day", "times", "daily",
    "race_num", "jockey_id", "horse_N", "waku_num", "horse_num",
    "class_code", "track_code", "corner_num", "dist",
    "age_code", "age", "basis_weight", "weight", "inc_dec", "weight_code",
    "win_odds", "rank", "time_diff", "corner1_rank", "corner2_rank",
    "corner3_rank", "corner4_rank", "last_3F_time", "last_3F_rank",
    "Ave_3F", "PCI", "last_3F_time_diff", "pop", "prize", "error_code", "id"
}

# 文字列として保持したい列（日本語など）
_STR_COLS = {"race_id", "place", "horse", "sex", "blinker", "state", "weather", "father", "mother", "time", "leg"}


def _read_one_csv(path: str) -> pd.DataFrame:
    # エンコーディングはCP932優先（Windows版Shift_JIS相当）、だめならUTF-8-SIGにフォールバック
    for enc in ("cp932", "shift_jis", "utf-8-sig"):
        try:
            df = pd.read_csv(path, names=COLS, header=None, encoding=enc, low_memory=False)
            return df
        except Exception:
            continue
    # 最後の手段：エンコ未指定
    return pd.read_csv(path, names=COLS, header=None, low_memory=False)


def load_records(records_glob: str) -> pd.DataFrame:
    """
    例: records_glob="data/raw/record_data_*.csv"
    すべて縦結合し、列型を揃えて返す。
    """
    paths = sorted(glob.glob(records_glob))
    if not paths:
        raise FileNotFoundError(f"No CSV matched: {records_glob}")

    dfs = []
    for p in paths:
        df = _read_one_csv(p)
        dfs.append(df)
    data = pd.concat(dfs, axis=0, ignore_index=True)

    # 型整形
    for c in data.columns:
        if c in _STR_COLS:
            data[c] = data[c].astype(str)
        elif c in _NUMERIC_COLS:
            data[c] = pd.to_numeric(data[c], errors="coerce")
        else:
            # 未指定列は可能な限り数値化、それ以外は文字列化
            try:
                data[c] = pd.to_numeric(data[c], errors="coerce")
            except Exception:
                data[c] = data[c].astype(str)

    # ありえない値の正規化
    # 例: rank=0 を欠損扱い、win_odds<=0 も欠損
    data.loc[(data["rank"] <= 0) | (data["rank"].isna()), "rank"] = np.nan
    data.loc[(data["win_odds"] <= 0) | (data["win_odds"].isna()), "win_odds"] = np.nan

    # id（馬ID）が欠損なら一意キーを生成（本来は提供されているはず）
    if data["id"].isna().any():
        # 馬名ベースの簡易ID（重複注意だが最後の手段）
        data["id"] = data["id"].fillna(
            pd.factorize(data["horse"].astype(str))[0] + 2_000_000_000
        )

    return data


def build_features(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    学習用の特徴量とターゲット（単勝的中=rank==1）を返す。
    - 馬ごとの直近レース指標（順位、上がり、PCIなど）のラグ特徴
    - 枠/距離/クラス/年齢などの基本特徴
    """
    df = data.copy()

    # ターゲット: 1着=1, それ以外=0（学習時のみ）
    y = (df["rank"] == 1).astype(float).fillna(0.0)

    # 時系列順（race_idは文字列なのでゼロ埋め長さで比較できるよう一旦数値化）
    # 仕様上18桁相当なのでそのまま文字列ソートでも概ね大丈夫だが、厳密に numeric に
    def _rid_to_int(x: str) -> int:
        try:
            return int(str(x).strip()[:18])
        except Exception:
            return 0

    df["_rid"] = df["race_id"].apply(_rid_to_int)

    # 馬ID単位で並べてラグ特徴
    df = df.sort_values(["id", "_rid"]).copy()

    def lag(name: str, k: int = 1):
        return df.groupby("id")[name].shift(k)

    # 直近指標
    df["rank_last1"] = lag("rank", 1)
    df["rank_last2"] = lag("rank", 2)
    df["last3F_last1"] = lag("last_3F_time", 1)
    df["last3F_last2"] = lag("last_3F_time", 2)
    df["pop_last1"] = lag("pop", 1)
    df["pop_last2"] = lag("pop", 2)
    df["PCI_last1"] = lag("PCI", 1)

    # 派生：平均順位(過去3走)
    df["rank_mean3"] = df.groupby("id")["rank"].rolling(3, min_periods=1).mean().reset_index(level=0, drop=True).shift(1)

    # 現レース特徴
    base_cols = [
        "year", "month", "day", "times", "place", "race_num",
        "waku_num", "horse_num", "class_code", "track_code",
        "corner_num", "dist", "age", "age_code",
        "basis_weight", "weight", "weight_code",
        "pop", "PCI", "Ave_3F",
    ]
    # 数値化（place などカテゴリは factorize）
    out = pd.DataFrame(index=df.index)

    # カテゴリ→数値ID
    for c in base_cols:
        if c not in df.columns:
            out[c] = 0.0
            continue
        if df[c].dtype.kind in "biufc":  # number-like
            out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)
        else:
            out[c] = pd.factorize(df[c].astype(str))[0].astype(float)

    # ラグ特徴
    for c in ["rank_last1", "rank_last2", "last3F_last1", "last3F_last2", "pop_last1", "pop_last2", "PCI_last1", "rank_mean3"]:
        out[c] = pd.to_numeric(df[c], errors="coerce").fillna(0.0)

    # 目的変数
    y = y.reindex(out.index).fillna(0.0)

    # 予測時に必要なキーを残す（呼び出し側で結合用に使うことがある）
    out["race_id"] = df["race_id"].astype(str).values
    out["id"] = pd.to_numeric(df["id"], errors="coerce").fillna(0).astype(int).values
    out["horse_num"] = pd.to_numeric(df["horse_num"], errors="coerce").fillna(0).astype(int).values

    return out, y