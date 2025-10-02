#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
generate_bets.py
- offline: features.csv の末尾レース（または --race-id-offline 指定）で馬単位の賭け金JSONを出力
- online : APIから当日データを取得し、--date/--place/--race を指定して賭け金JSONを出力
          ※ --allow-no-odds を付けると当日のオッズ無し（出走表のみ）でも実行可能
"""

import argparse
import os
import json
import numpy as np
import pandas as pd
import joblib

# API ユーティリティ（scripts パッケージから）
from scripts.api_utils import get_racecards, get_odds

PLACE_TO_CODE = {
    "札幌": "01", "函館": "02", "福島": "03", "新潟": "04", "東京": "05",
    "中山": "06", "中京": "07", "京都": "08", "阪神": "09", "小倉": "10"
}


def kelly_fraction(p: float, d: float) -> float:
    """ケリーの式（単勝オッズ d に対する最適比率）。"""
    if d <= 1.0:
        return 0.0
    edge = p * d - 1.0
    return 0.0 if edge <= 0 else edge / (d - 1.0)


def round_to_unit(x: float, unit: int = 100) -> int:
    """賭け金を unit（例:100円）単位に丸める。"""
    return int(np.floor(max(0.0, x) / unit) * unit)


def choose_stakes(
    df: pd.DataFrame,
    kelly_frac: float = 0.10,
    min_p: float = 0.05,
    stake_cap: int = 1000,
    unit: int = 100,
) -> pd.DataFrame:
    """
    予測確率 pred_prob と単勝オッズ win_odds から賭け金 stake を決める。
    総額が stake_cap を超える場合はスケールダウンする。
    """
    out = df.copy()
    stakes = []

    for _, r in out.iterrows():
        p = float(r["pred_prob"])
        d = float(r["win_odds"]) if not np.isnan(r["win_odds"]) else 10.0
        if p < min_p:
            stakes.append(0)
            continue
        raw = kelly_frac * kelly_fraction(p, d) * stake_cap
        stakes.append(round_to_unit(raw, unit))

    out["stake"] = stakes
    tot = int(out["stake"].sum())
    if tot > stake_cap and tot > 0:
        scale = stake_cap / tot
        out["stake"] = out["stake"].apply(lambda v: round_to_unit(v * scale, unit))

    return out


def _ensure_cols(df: pd.DataFrame, need_cols: list[str]) -> None:
    """必須列の存在確認。rank_last は無ければ 0.0 で補完。"""
    for col in need_cols:
        if col not in df.columns:
            if col == "rank_last":
                df[col] = 0.0
            else:
                raise SystemExit(f"features に必須列 '{col}' がありません。前処理を確認してください。")


def _normalize_odds_response(resp) -> list:
    """
    get_odds の戻り値が dict の時も list の時もあるため、レコードの list に正規化する。
    単勝（odds_type == "1"）のレコード抽出は呼び出し側で実施。
    """
    if isinstance(resp, list):
        return resp
    if isinstance(resp, dict):
        # 代表的な形: {"odds_rt": [...]} / {"data": {"odds_rt": [...]}} / {"data": [...]}
        if "odds_rt" in resp and isinstance(resp["odds_rt"], list):
            return resp["odds_rt"]
        data = resp.get("data")
        if isinstance(data, dict) and isinstance(data.get("odds_rt"), list):
            return data["odds_rt"]
        if isinstance(data, list):
            return data
    return []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["offline", "online"], required=True)
    ap.add_argument("--model", default="models/win_model.pkl")
    ap.add_argument("--export-dir", default="data/api_out")
    ap.add_argument("--unit", type=int, default=100)
    ap.add_argument("--stake-cap", type=int, default=1000)

    # しきい値・強度（調整できるように引数化）
    ap.add_argument("--min-p", type=float, default=0.02, help="最低予測確率（既定 0.02）")
    ap.add_argument("--kelly-frac", type=float, default=0.10, help="ケリー比率（既定 0.10）")

    # オッズなし運用
    ap.add_argument("--allow-no-odds", action="store_true",
                    help="当日のオッズが取得できなくても出走表のみで計算を続行する")
    ap.add_argument("--default-odds", type=float, default=10.0,
                    help="オッズ欠損時に用いる仮オッズ（既定 10.0）")

    # offline 用
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--race-id-offline", help="投票ID（未指定なら features.csv 末尾の race_id を使用）")

    # online 用
    ap.add_argument("--date")
    ap.add_argument("--place")
    ap.add_argument("--race")
    args = ap.parse_args()

    # モデルロード & 出力先
    clf = joblib.load(args.model)
    os.makedirs(args.export_dir, exist_ok=True)

    # -------- offline --------
    if args.mode == "offline":
        df = pd.read_csv(args.features)
        if df.empty:
            raise SystemExit(f"features が空です: {args.features}")

        # race_id の決定（指定が無ければ末尾を採用）
        race_id = str(args.race_id_offline) if args.race_id_offline else str(df["race_id"].iloc[-1])

        # 型ずれ対策（str 同士で比較）
        sub = df[df["race_id"].astype(str) == race_id].copy()
        if sub.empty:
            sample_ids = df["race_id"].dropna().astype(str).tail(5).unique().tolist()
            raise SystemExit(f"race_id={race_id} に一致する行がありません。直近の race_id 例: {sample_ids}")

        # 必須列の確認・補完
        _ensure_cols(sub, ["age", "rank_last"])

        # 予測
        X = sub[["age", "rank_last"]].fillna(0.0).values
        sub["pred_prob"] = clf.predict_proba(X)[:, 1]
        sub["win_odds"] = np.nan  # offline はオッズ未取得

        # 賭け金計算
        bet_df = choose_stakes(
            sub[["horse_num", "pred_prob", "win_odds"]],
            kelly_frac=args.kelly_frac, min_p=args.min_p,
            stake_cap=args.stake_cap, unit=args.unit
        )

        # 出力 JSON
        rid_vote = args.race_id_offline if args.race_id_offline else race_id
        mark = {str(int(r.horse_num)).zfill(2): 1 for _, r in bet_df.iterrows() if int(r.stake) > 0}
        bet = [
            {"bet_id": f"b1_c0_{str(int(r.horse_num)).zfill(2)}", "money": str(int(r.stake))}
            for _, r in bet_df.iterrows() if int(r.stake) > 0
        ]

        out_path = os.path.join(args.export_dir, f"{rid_vote}_OFFLINE.json")
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump({"race_id": str(rid_vote), "mark": mark, "bet": bet}, f, ensure_ascii=False, indent=2)
        print("Wrote", out_path)
        return

    # -------- online --------
    if not (args.date and args.place and args.race):
        raise SystemExit("--date/--place/--race が必要です（online）")

    jj = PLACE_TO_CODE.get(args.place)
    if not jj:
        raise SystemExit(f"Unknown place: {args.place}")

    # レースカード取得
    data = get_racecards(args.date, verify=False)
    runtable = pd.DataFrame(data.get("runtable", []))
    if runtable.empty:
        raise SystemExit("runtable が空です。日付や API 応答を確認してください。")

    # 場所・レース番号で抽出
    rsub = runtable[
        (runtable["place"] == args.place) &
        (runtable["race_num"].astype(str) == str(int(args.race)))
    ].copy()
    if rsub.empty:
        raise SystemExit("指定のレースが見つかりません（runtable フィルタ結果が空）")

    # 必須列の確認・補完
    _ensure_cols(rsub, ["age", "rank_last"])

    # 予測
    X = rsub[["age", "rank_last"]].fillna(0.0).values
    rsub["pred_prob"] = clf.predict_proba(X)[:, 1]

    # --- オッズ取得（失敗しても進められるようにフォールバック） ---
    win_map: dict[int, float] = {}
    odds_ok = False
    try:
        race_key = f"{args.date}{jj}{str(args.race).zfill(2)}"
        resp = get_odds(race_key, verify=False)
        odds_list = _normalize_odds_response(resp)
        for o in odds_list:
            try:
                if str(o.get("odds_type", "")) == "1" and "comb" in o and "odds" in o:
                    win_map[int(o["comb"])] = float(o["odds"])
            except Exception:
                continue
        odds_ok = len(win_map) > 0
    except Exception:
        odds_ok = False

    if odds_ok:
        rsub["win_odds"] = rsub["horse_num"].astype(int).map(win_map).astype(float)
        rsub["win_odds"] = rsub["win_odds"].fillna(args.default_odds)
    else:
        if not args.allow_no_odds:
            raise SystemExit("当日のオッズ取得に失敗しました。--allow-no-odds を付けると出走表のみで続行します。")
        # 出走表のみ：全頭に仮オッズを入れる
        rsub["win_odds"] = float(args.default_odds)

    # 賭け金計算
    bet_df = choose_stakes(
        rsub[["horse_num", "pred_prob", "win_odds"]],
        kelly_frac=args.kelly_frac, min_p=args.min_p,
        stake_cap=args.stake_cap, unit=args.unit
    )

    # 出力 JSON
    race_id_vote = str(rsub["race_id"].iloc[0])
    mark = {str(int(r.horse_num)).zfill(2): 1 for _, r in bet_df.iterrows() if int(r.stake) > 0}
    bet = [
        {"bet_id": f"b1_c0_{str(int(r.horse_num)).zfill(2)}", "money": str(int(r.stake))}
        for _, r in bet_df.iterrows() if int(r.stake) > 0
    ]

    out_path = os.path.join(args.export_dir, f"{race_id_vote}.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump({"race_id": race_id_vote, "mark": mark, "bet": bet}, f, ensure_ascii=False, indent=2)
    print("Wrote", out_path)


if __name__ == "__main__":
    main()