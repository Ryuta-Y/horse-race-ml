from __future__ import annotations
import math
import numpy as np
import pandas as pd
from typing import Callable

from conf import (
    PER_RACE_BUDGET_YEN, BASE_BET_UNIT, BOTH_AGREE_BOOST,
    THRESH_WIN_PROB_MIN, THRESH_WIN_EV_MIN, THRESH_PLACE_BY_WINP,
)

# ================= 調整ブロック =================
DEFAULT_ALLOC_RATIO = {
    "place": 0.40,
    "tansho": 0.30,
    "umaren": 0.10,
    "wide":   0.10,
    "sanrenpuku": 0.07,
    "sanrentan":  0.03,
}

def get_alloc_ratio(n_horses: int) -> dict[str, float]:
    ratio = DEFAULT_ALLOC_RATIO.copy()
    if n_horses <= 6:
        ratio.update({"sanrenpuku": 0.15, "sanrentan": 0.10, "umaren": 0.15,
                      "wide": 0.15, "place": 0.20, "tansho": 0.25})
    elif n_horses <= 8:
        ratio.update({"sanrenpuku": 0.10, "sanrentan": 0.05, "umaren": 0.12,
                      "wide": 0.12, "place": 0.31, "tansho": 0.30})
    total = sum(ratio.values())
    for k in ratio:
        ratio[k] /= total
    return ratio

TOPK_TANSHO = 3
TOPK_PLACE  = 4
MAX_POINTS_EACH_TYPE = 10
MIN_AMOUNT_PER_POINT = 100

PSEUDO_MARGIN = {
    "tansho": 1.20, "place": 1.05,
    "umaren": 1.25, "wide": 1.22, "umatan": 1.30,
    "sanrenpuku": 1.35, "sanrentan": 1.45, "wakuren": 1.25
}

SAFE_FALLBACK_KIND = "place"
SAFE_FALLBACK_AMOUNT = 100

# --------- ユーティリティ ---------
def _to_int100(x: float) -> int:
    return int(max(MIN_AMOUNT_PER_POINT, math.floor(x / BASE_BET_UNIT) * BASE_BET_UNIT))

def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(p.astype(float), 1e-6, 1.0 - 1e-6)
    s = float(np.sum(p))
    return p / s if s > 0 else np.full_like(p, 1.0 / len(p))

def _predict_win_prob(model_bundle, feature_df: pd.DataFrame) -> np.ndarray:
    if model_bundle is None:
        return np.full(len(feature_df), 1.0 / max(1, len(feature_df)))
    cols = model_bundle["columns"]
    X = feature_df.reindex(columns=cols, fill_value=0.0).values
    if "scaler" in model_bundle and model_bundle["scaler"] is not None:
        X = model_bundle["scaler"].transform(X)
    m = model_bundle["model"]
    if hasattr(m, "predict_proba"):
        return m.predict_proba(X)[:, 1]
    f = getattr(m, "decision_function", None)
    if f is None:
        return np.full(len(feature_df), 1.0 / len(feature_df))
    z = f(X)
    return 1.0 / (1.0 + np.exp(-z))

def _make_feature_df(race_df: pd.DataFrame, model_bundle) -> pd.DataFrame:
    cols = model_bundle["columns"] if model_bundle else []
    df = pd.DataFrame(index=race_df.index)
    for c in cols:
        df[c] = pd.to_numeric(race_df[c], errors="coerce") if c in race_df.columns else 0.0
    return df.fillna(0.0)

def _pseudo_odds(prob: float, kind: str) -> float:
    prob = max(1e-9, float(prob))
    return float(PSEUDO_MARGIN.get(kind, 1.25) / prob)

def _fetch_odds_safe(get_odds_func: Callable[[str], dict] | None, race_id: str) -> dict:
    if get_odds_func is None:
        return {}
    try:
        data = get_odds_func(str(race_id))
        return data.get("data", data) if isinstance(data, dict) else {}
    except Exception:
        return {}

def _parse_odds_table(data: dict, kind: str) -> dict:
    if not isinstance(data, dict):
        return {}
    keys = {
        "tansho": ["tansho", "win", "単勝"],
        "place":  ["place", "複勝"],
        "umaren": ["umaren", "馬連"],
        "wide":   ["wide", "ワイド"],
        "umatan": ["umatan", "馬単"],
        "sanrenpuku": ["sanrenpuku", "三連複"],
        "sanrentan":  ["sanrentan",  "三連単"],
        "wakuren": ["wakuren", "枠連"],
    }.get(kind, [])
    out = {}
    for k in keys:
        if k in data and isinstance(data[k], dict):
            for kk, vv in data[k].items():
                try:
                    out[str(kk)] = float(vv)
                except Exception:
                    pass
            if out:
                return out
    return {}

# --------- PL近似 ---------
def _p_umatan(p: np.ndarray, i: int, j: int) -> float:
    if i == j:
        return 0.0
    denom = max(1e-12, 1.0 - p[i])
    return float(p[i] * (p[j] / denom))

def _p_umaren(p: np.ndarray, i: int, j: int) -> float:
    if i == j:
        return 0.0
    return _p_umatan(p, i, j) + _p_umatan(p, j, i)

def _p_sanrentan(p: np.ndarray, i: int, j: int, k: int) -> float:
    if len({i, j, k}) < 3:
        return 0.0
    denom1 = max(1e-12, 1.0 - p[i])
    denom2 = max(1e-12, denom1 - p[j])
    return float(p[i] * (p[j] / denom1) * (p[k] / denom2))

def _p_sanrenpuku(p: np.ndarray, i: int, j: int, k: int) -> float:
    if len({i, j, k}) < 3:
        return 0.0
    return float(
        _p_sanrentan(p, i, j, k) + _p_sanrentan(p, i, k, j) +
        _p_sanrentan(p, j, i, k) + _p_sanrentan(p, j, k, i) +
        _p_sanrentan(p, k, i, j) + _p_sanrentan(p, k, j, i)
    )

def _p_place_top3(p: np.ndarray, i: int) -> float:
    n = len(p)
    p1 = p[i]
    p2 = sum(_p_umatan(p, j, i) for j in range(n) if j != i)
    p3 = 0.0
    for j in range(n):
        if j == i:
            continue
        for k in range(n):
            if k == i or k == j:
                continue
            p3 += _p_sanrentan(p, j, k, i)
    return float(min(1.0, p1 + p2 + p3))

# --------- 候補生成 ---------
def _tansho_candidates(race_df: pd.DataFrame, p: np.ndarray, odds_map: dict) -> list[dict]:
    horses = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int).values
    cand = []
    for i, h in enumerate(horses):
        odds = odds_map.get(str(h), 0.0) or _pseudo_odds(p[i], "tansho")
        ev = p[i] * odds - 1.0
        if p[i] >= THRESH_WIN_PROB_MIN and ev >= THRESH_WIN_EV_MIN:
            cand.append({"sel": [int(h)], "p": float(p[i]), "odds": float(odds), "ev": float(ev)})
    cand.sort(key=lambda x: x["ev"], reverse=True)
    return cand[:TOPK_TANSHO]

def _place_candidates(race_df: pd.DataFrame, p: np.ndarray, odds_map: dict) -> list[dict]:
    horses = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int).values
    cand = []
    for i, h in enumerate(horses):
        p3 = _p_place_top3(p, i)
        odds = odds_map.get(str(h), 0.0) or _pseudo_odds(p3, "place")
        ev = p3 * odds - 1.0
        if p3 >= max(THRESH_PLACE_BY_WINP, 0.15) and ev >= -0.02:
            cand.append({"sel": [int(h)], "p": float(p3), "odds": float(odds), "ev": float(ev)})
    cand.sort(key=lambda x: (x["ev"], x["p"]), reverse=True)
    return cand[:TOPK_PLACE]

def _pairs_candidates(p: np.ndarray, horses: np.ndarray, odds_map: dict, kind: str) -> list[dict]:
    cand = []
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            pr = _p_umaren(p, i, j)
            odds = odds_map.get(f"{horses[i]}-{horses[j]}", 0.0) or _pseudo_odds(pr, kind)
            ev = pr * odds - 1.0
            cand.append({"sel": [int(horses[i]), int(horses[j])],
                         "p": float(pr), "odds": float(odds), "ev": float(ev)})
    cand.sort(key=lambda x: x["ev"], reverse=True)
    return cand[:MAX_POINTS_EACH_TYPE]

def _triples_candidates(p: np.ndarray, horses: np.ndarray, odds_map: dict, kind: str) -> list[dict]:
    cand = []
    n = len(p)
    if kind == "sanrenpuku":
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    pr = _p_sanrenpuku(p, i, j, k)
                    odds = odds_map.get(f"{horses[i]}-{horses[j]}-{horses[k]}", 0.0) or _pseudo_odds(pr, "sanrenpuku")
                    ev = pr * odds - 1.0
                    cand.append({"sel": [int(horses[i]), int(horses[j]), int(horses[k])],
                                 "p": float(pr), "odds": float(odds), "ev": float(ev)})
    else:  # sanrentan
        for i in range(n):
            for j in range(n):
                if j == i:
                    continue
                for k in range(n):
                    if k == i or k == j:
                        continue
                    pr = _p_sanrentan(p, i, j, k)
                    odds = odds_map.get(f"{horses[i]}>{horses[j]}>{horses[k]}", 0.0) or _pseudo_odds(pr, "sanrentan")
                    ev = pr * odds - 1.0
                    cand.append({"sel": [int(horses[i]), int(horses[j]), int(horses[k])],
                                 "p": float(pr), "odds": float(odds), "ev": float(ev)})
    cand.sort(key=lambda x: x["ev"], reverse=True)
    return cand[:MAX_POINTS_EACH_TYPE]

# --------- 配分支援 ---------
def _allocate(budget: int, n: int) -> list[int]:
    if n <= 0 or budget <= 0:
        return []
    base = _to_int100(budget / n)
    amounts = [base] * n
    rest = budget - base * n
    i = 0
    while rest > 0:
        add = min(rest, BASE_BET_UNIT)
        amounts[i] += add
        rest -= add
        i = (i + 1) % n
    return amounts

# --------- Payload生成 ---------
def _payload_mark_bet(race_row: pd.Series,
                      bet_items: list[dict],
                      mark: dict) -> dict:
    """
    Bet API用payloadを生成する。
    mark: {"馬番": 印番号} の辞書を渡す。印は1=◎, 2=◯, 3=▲, 4=△。
    bet_items: [{kind, sel, amount, p, odds, ev}, ...]
    """
    kind_code = {
        "tansho": 1, "place": 2, "wakuren": 3, "umaren": 4,
        "wide": 5, "umatan": 6, "sanrenpuku": 7, "sanrentan": 8
    }
    bet_list = []
    for item in bet_items:
        k = item["kind"]
        sel = item["sel"]
        amt = item["amount"]
        sel_str = "_".join(str(x) for x in sel)
        bet_id = f"b{kind_code[k]}_c0_{sel_str}"
        bet_list.append({
            "bet_id": bet_id,
            "money": str(int(amt))
        })
    return {
        "race_id": str(race_row.get("race_id", "")),
        "mark": mark,
        "bet": bet_list,
    }

# --------- 1レース計画 ---------
def _plan_one_race(race_df: pd.DataFrame, odds_maps: dict[str, dict],
                   p: np.ndarray, budget: int) -> list[dict]:
    horses = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int).values
    plan = []

    cand_place  = _place_candidates(race_df, p, odds_maps.get("place", {}))
    cand_tan    = _tansho_candidates(race_df, p, odds_maps.get("tansho", {}))
    cand_umaren = _pairs_candidates(p, horses, odds_maps.get("umaren", {}), "umaren")
    cand_wide   = _pairs_candidates(p, horses, odds_maps.get("wide",   {}), "wide")
    cand_srpk   = _triples_candidates(p, horses, odds_maps.get("sanrenpuku", {}), "sanrenpuku")
    cand_srtan  = _triples_candidates(p, horses, odds_maps.get("sanrentan",  {}), "sanrentan")

    alloc_ratio = get_alloc_ratio(len(horses))
    sub = {
        "place": int(budget * alloc_ratio.get("place", 0.0)),
        "tansho": int(budget * alloc_ratio.get("tansho", 0.0)),
        "umaren": int(budget * alloc_ratio.get("umaren", 0.0)),
        "wide":   int(budget * alloc_ratio.get("wide",   0.0)),
        "sanrenpuku": int(budget * alloc_ratio.get("sanrenpuku", 0.0)),
        "sanrentan":  int(budget * alloc_ratio.get("sanrentan",  0.0)),
    }

    candidates_by_kind = {
        "place": cand_place,
        "tansho": cand_tan,
        "umaren": cand_umaren,
        "wide": cand_wide,
        "sanrenpuku": cand_srpk,
        "sanrentan": cand_srtan,
    }

    bet_items = []
    for kind, cands in candidates_by_kind.items():
        budget_k = sub.get(kind, 0)
        if budget_k <= 0 or not cands:
            continue
        weights = []
        for c in cands:
            ev = c.get("ev", 0.0)
            weights.append((ev if ev > 0 else 0.0) + 1e-3)
        sum_w = sum(weights)
        if sum_w > 0:
            raw_amounts = [budget_k * w / sum_w for w in weights]
            amounts = [_to_int100(a) for a in raw_amounts]
            total = sum(amounts)
            if total > budget_k:
                amounts = _allocate(budget_k, len(cands))
            else:
                diff = budget_k - total
                i = 0
                # diffが100円以上のときのみ100円単位で追加配分
                while diff >= BASE_BET_UNIT and amounts:
                    amounts[i] += BASE_BET_UNIT
                    diff -= BASE_BET_UNIT
                    i = (i + 1) % len(amounts)
                # diff < 100円は余剰金として later へ回す
        else:
            amounts = _allocate(budget_k, len(cands))

        for c, a in zip(cands, amounts):
            bet_items.append({
                "kind": kind,
                "sel": c["sel"],
                "amount": a,
                "p": c["p"],
                "odds": c["odds"],
                "ev": c["ev"],
            })

    # フォールバック：候補がなければ複勝1点100円
    if not bet_items:
        best = int(np.argmax(p))
        h = int(horses[best])
        bet_items = [{
            "kind": SAFE_FALLBACK_KIND,
            "sel": [h],
            "amount": SAFE_FALLBACK_AMOUNT,
            "p": _p_place_top3(p, best) if SAFE_FALLBACK_KIND == "place" else p[best],
            "odds": _pseudo_odds(_p_place_top3(p, best) if SAFE_FALLBACK_KIND == "place" else p[best], SAFE_FALLBACK_KIND),
            "ev": 0.0,
        }]

    # markを期待値順に割り当て：印は1=◎、2=◯、3=▲、それ以外は4=△
    mark_order = []
    for item in sorted(bet_items, key=lambda x: x["ev"], reverse=True):
        for h in item["sel"]:
            if h not in mark_order:
                mark_order.append(h)
    mark = {}
    for i, h in enumerate(mark_order):
        if i == 0:
            mark[str(h)] = 1  # ◎
        elif i == 1:
            mark[str(h)] = 2  # ◯
        elif i == 2:
            mark[str(h)] = 3  # ▲
        else:
            mark[str(h)] = 4  # △

    race_row = race_df.iloc[0]
    payload = _payload_mark_bet(race_row, bet_items, mark)

    out = []
    for item in bet_items:
        out.append({
            "race_id": race_row["race_id"],
            "bet_type_name": item["kind"],
            "selection": item["sel"],
            "amount": item["amount"],
            "p": float(item["p"]),
            "odds": float(item["odds"]),
            "ev": float(item["ev"]),
            "api_payload": payload,
        })
    return out

# --------- デイリープラン ---------
def make_plan_for_day(timetable: pd.DataFrame, runtable: pd.DataFrame,
                      oddsless, oddsaware,
                      get_odds_func: Callable[[str], dict] | None = None,
                      date_yyyymmdd: str | None = None, verbose: int = 1) -> pd.DataFrame:
    if "race_id" not in runtable.columns:
        raise RuntimeError("runtable に race_id 列が必要です")
    plans = []
    for race_id, race_df in runtable.groupby("race_id"):
        race_df = race_df.copy()
        # oddsless と oddsaware の予測平均を使用し、一致時はブーストする
        p1 = _predict_win_prob(oddsless,  _make_feature_df(race_df, oddsless))
        p2 = _predict_win_prob(oddsaware, _make_feature_df(race_df, oddsaware))
        agree = (p1 >= THRESH_WIN_PROB_MIN) & (p2 >= THRESH_WIN_PROB_MIN)
        p = (p1 + p2) / 2.0
        p = np.where(agree, p * BOTH_AGREE_BOOST, p)
        p = _normalize_probs(p)

        rid_odds = race_df.iloc[0].get("race_id_odds") if "race_id_odds" in race_df.columns else None
        odds_id = str(rid_odds) if rid_odds else str(race_id)
        odds_raw = _fetch_odds_safe(get_odds_func, odds_id)
        odds_maps = {
            "tansho": _parse_odds_table(odds_raw, "tansho"),
            "place":  _parse_odds_table(odds_raw, "place"),
            "umaren": _parse_odds_table(odds_raw, "umaren"),
            "wide":   _parse_odds_table(odds_raw, "wide"),
            "umatan": _parse_odds_table(odds_raw, "umatan"),
            "sanrenpuku": _parse_odds_table(odds_raw, "sanrenpuku"),
            "sanrentan":  _parse_odds_table(odds_raw, "sanrentan"),
        }

        race_plan = _plan_one_race(race_df, odds_maps, p, int(PER_RACE_BUDGET_YEN))
        if verbose >= 2:
            print(f"[plan] race_id={race_id}, points={len(race_plan)}")
        plans.extend(race_plan)
    return pd.DataFrame(plans)