# keiba_dual_ready_full/odds_policy.py
from __future__ import annotations
import math
from typing import Callable, Dict, Any, Iterable, Optional, Tuple

import numpy as np
import pandas as pd

from conf import (
    PER_RACE_BUDGET_YEN, BASE_BET_UNIT,
    THRESH_WIN_PROB_MIN, THRESH_WIN_EV_MIN, THRESH_PLACE_BY_WINP,
    ENSEMBLE_VOTE_TOPK, ENSEMBLE_VOTE_BONUS_ALPHA, ENSEMBLE_PRIORITY_BONUS_ALPHA, ROUND_BUDGET_OVERRIDES,
)

# ========= 可調パラメータ（ベースライン） =========
DEFAULT_ALLOC_RATIO = {
    "place": 0.38,
    "tansho": 0.12,
    "umaren": 0.05,
    "wide":   0.37,
    "sanrenpuku": 0.05,
    "sanrentan":  0.03,
}

TOPK_TANSHO = 3
TOPK_PLACE  = 3
MAX_POINTS_EACH_TYPE = 5
MIN_AMOUNT_PER_POINT = 100

# --- Wide 分散 & Favorite(◎)抑制・Place 単点時の上限 ---
TOPK_WIDE = 6                 # ワイドの候補上限（ワイドだけ独立に制御）
MAX_WIDE_PER_HORSE = 3       # 1頭が絡むワイド候補の最大本数（散らす上限）
WIDE_FAVORITE_PENALTY = 0.85  # ◎を含むワイド候補のEV減衰係数（<1で抑制）
WIDE_PLACE_BONUS = 1.15       # 複勝候補が絡むワイドを優先させる倍率（>1で強化）
PLACE_SOLO_BUDGET_CAP_RATIO = 0.20  # Place候補が1本のときの、その券種予算の上限（レース総予算比）
WIDE_REFILL_MIN_SCORE_RATIO = 0.75   # 補充で採用する最低スコア比（best_scoreに対する割合）
WIDE_REFILL_STAKE_RATIO     = 0.20   # 補充採用分にかける金額の係数（小さめに）
HORSE_CAP_BY_KIND = {         # 種別ごとの 1頭あたり露出上限（レース総予算比）
    "wide":  0.18,            # ワイドはより分散させる
    "place": 0.15,            # 複勝は単点集中を弱める
    # 未指定は従来の 0.25 を使用
}

PSEUDO_MARGIN = {
    "tansho": 1.20, "place": 1.05,
    "umaren": 1.25, "wide": 1.22, "umatan": 1.30,
    "sanrenpuku": 1.35, "sanrentan": 1.45, "wakuren": 1.25
}

SAFE_FALLBACK_KIND = "place"
SAFE_FALLBACK_AMOUNT = 0

# 人気薄混入
LONGSHOT_TOP_K = 1
FAVORITE_SET_K = 4
FAVORITE_CLOSE_RATIO = 1.5

# ========= ユーティリティ =========
def _to_int100(x: float) -> int:
    return int(max(MIN_AMOUNT_PER_POINT, math.floor(x / BASE_BET_UNIT) * BASE_BET_UNIT))

def _normalize_probs(p: np.ndarray) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), 1e-9, 1.0 - 1e-9)
    s = float(np.sum(p))
    return p / s if s > 0 else np.full_like(p, 1.0 / max(1, len(p)))

def _entropy(p: np.ndarray) -> float:
    p = np.clip(p, 1e-12, 1.0)
    return float(-np.sum(p * np.log(p)))

def _make_feature_df(race_df: pd.DataFrame, model_bundle: dict) -> pd.DataFrame:
    cols = model_bundle.get("columns", [])
    df = pd.DataFrame(index=race_df.index)
    for c in cols:
        df[c] = pd.to_numeric(race_df[c], errors="coerce") if c in race_df.columns else 0.0
    return df.fillna(0.0)

# 確率的トップK（Gumbel-Top-K）: スコアのsoftmaxに比例してサンプルしつつK本選ぶ
def _stochastic_topk(cands, score_fn, k, tau, rng):
    """
    cands: 候補リスト
    score_fn: 候補 -> 正の重み（0不可、pやp×平均単体pなど）
    k: 取り出す本数
    tau: 温度(>0)。大きいほどランダム性↑
    rng: np.random.Generator
    """
    if not cands or k <= 0:
        return []
    s = np.array([max(1e-12, float(score_fn(c))) for c in cands], dtype=float)
    logw = np.log(s)
    g = rng.gumbel(size=len(s))                # Gumbel(0,1)
    y = logw / max(1e-6, float(tau)) + g       # 温度でノイズ/鋭さを調整
    idx = np.argsort(y)[::-1][:k]
    return [cands[i] for i in idx]

def _predict_win_prob_generic(model_bundle: dict, race_df: pd.DataFrame) -> np.ndarray:
    # scikit/Calibrated/Pipeline/lightgbm/torch（簡易）を吸収。失敗時は等確率。
    try:
        X = _make_feature_df(race_df, model_bundle).values
        scaler = model_bundle.get("scaler")
        if scaler is not None:
            try: X = scaler.transform(X)
            except Exception: pass
        model = model_bundle.get("model")
        if model is not None:
            if hasattr(model, "predict_proba"):
                p = model.predict_proba(X)[:, 1]
                return _normalize_probs(p)
            if hasattr(model, "decision_function"):
                z = model.decision_function(X)
                p = 1.0 / (1.0 + np.exp(-z))
                return _normalize_probs(p)
            try:
                import lightgbm as lgb  # noqa
                if str(type(model)).startswith("<class 'lightgbm"):
                    p = model.predict(X)
                    return _normalize_probs(p)
            except Exception:
                pass
    except Exception:
        pass

    if isinstance(model_bundle, dict) and "torch_state_dict" in model_bundle:
        try:
            import torch
            cols = model_bundle.get("columns", [])
            X = race_df.reindex(columns=cols, fill_value=0.0).astype(float).values
            mean = np.array(model_bundle.get("scaler_mean", np.zeros(X.shape[1])))
            scale= np.array(model_bundle.get("scaler_scale", np.ones(X.shape[1])))
            X = (X - mean) / np.clip(scale, 1e-9, None)
            X = torch.tensor(X, dtype=torch.float32)
            class M(torch.nn.Module):
                def __init__(self, in_dim, hidden):
                    super().__init__()
                    self.net = torch.nn.Sequential(
                        torch.nn.Linear(in_dim, hidden), torch.nn.ReLU(),
                        torch.nn.Linear(hidden, hidden), torch.nn.ReLU(),
                        torch.nn.Linear(hidden, 1)
                    )
                def forward(self, x): return self.net(x).squeeze(-1)
            m = M(X.shape[1], int(model_bundle.get("hidden_dim", 64)))
            m.load_state_dict(model_bundle["torch_state_dict"])
            m.eval()
            with torch.no_grad():
                p = torch.sigmoid(m(X)).cpu().numpy()
            return _normalize_probs(p)
        except Exception:
            pass

    return np.full(len(race_df), 1.0 / max(1, len(race_df)))

def _pseudo_odds(prob: float, kind: str) -> float:
    prob = max(1e-9, float(prob))
    return float(PSEUDO_MARGIN.get(kind, 1.25) / prob)

def _fetch_odds_safe(get_odds_func: Callable[[str], dict] | None, race_id: str) -> dict:
    if get_odds_func is None: return {}
    try:
        data = get_odds_func(str(race_id))
        return data.get("data", data) if isinstance(data, dict) else {}
    except Exception:
        return {}

def _parse_odds_table(data: dict, kind: str) -> dict:
    if not isinstance(data, dict): return {}
    # まず既存の辞書型 {"tansho": {...}} に対応（互換）
    keys_map = {
        "tansho": ["tansho", "win", "単勝"],
        "place":  ["place", "複勝"],
        "umaren": ["umaren", "馬連"],
        "wide":   ["wide", "ワイド"],
        "umatan": ["umatan", "馬単"],
        "sanrenpuku": ["sanrenpuku", "三連複"],
        "sanrentan":  ["sanrentan",  "三連単"],
        "wakuren": ["wakuren", "枠連"],
    }
    out = {}
    for k in keys_map.get(kind, []):
        if k in data and isinstance(data[k], dict):
            for kk, vv in data[k].items():
                try: out[str(kk)] = float(vv)
                except: pass
            if out: return out

    # 次に "odds_rt"（list）に対応（あなたのAPI例）
    if "odds_rt" in data and isinstance(data["odds_rt"], list):
        # 1:単勝 2:複勝 3:枠連 4:馬連 5:ワイド 6:馬単 7:三連複 8:三連単
        type2kind = {1:"tansho",2:"place",3:"wakuren",4:"umaren",5:"wide",6:"umatan",7:"sanrenpuku",8:"sanrentan"}
        def comb_key(s: str, knd: str) -> str:
            s = str(s)                           # "100904"等
            nums = [int(s[i:i+2]) for i in range(0, len(s), 2)]  # -> [10,9,4]
            if knd in ("tansho","place"): return str(nums[0])
            if knd in ("umaren","wide","wakuren","sanrenpuku"):  # 順不同
                return "-".join(str(x) for x in sorted(nums))
            if knd in ("umatan","sanrentan"):                    # 順序あり
                return ">".join(str(x) for x in nums)
            return "-".join(str(x) for x in nums)

        out = {}
        for row in data["odds_rt"]:
            if not isinstance(row, dict): continue
            knd = type2kind.get(int(row.get("odds_type", 0)))
            if knd != kind: continue
            comb = row.get("comb"); odd = row.get("odds")
            if comb is None or odd is None: continue
            try:
                out[comb_key(comb, knd)] = float(odd)
            except: pass
        return out

    return {}

# ========= PL近似 =========
def _p_umatan(p, i, j):
    if i == j: return 0.0
    denom = max(1e-12, 1.0 - p[i])
    return float(p[i] * (p[j] / denom))

def _p_umaren(p, i, j):
    if i == j: return 0.0
    return _p_umatan(p, i, j) + _p_umatan(p, j, i)

def _p_sanrentan(p, i, j, k):
    if len({i, j, k}) < 3: return 0.0
    denom1 = max(1e-12, 1.0 - p[i])
    denom2 = max(1e-12, denom1 - p[j])
    return float(p[i] * (p[j] / denom1) * (p[k] / denom2))

def _p_sanrenpuku(p, i, j, k):
    if len({i, j, k}) < 3: return 0.0
    return float(
        _p_sanrentan(p, i, j, k) + _p_sanrentan(p, i, k, j) +
        _p_sanrentan(p, j, i, k) + _p_sanrentan(p, j, k, i) +
        _p_sanrentan(p, k, i, j) + _p_sanrentan(p, k, j, i)
    )

def _p_place_top3(p, i):
    n = len(p); p1 = p[i]
    p2 = sum(_p_umatan(p, j, i) for j in range(n) if j != i)
    p3 = 0.0
    for j in range(n):
        if j == i: continue
        for k in range(n):
            if k == i or k == j: continue
            p3 += _p_sanrentan(p, j, k, i)
    return float(min(1.0, p1 + p2 + p3))

# ========= 人気薄/人気ユーティリティ =========
def _build_horse_odds(horses, p, tansho_map):
    out = {}
    for idx, h in enumerate(horses):
        h = int(h)
        if tansho_map and str(h) in tansho_map and tansho_map[str(h)] > 0:
            out[h] = float(tansho_map[str(h)])
        else:
            out[h] = _pseudo_odds(p[idx], "tansho")
    return out

def _pick_longshots(horse_odds, k=LONGSHOT_TOP_K):
    return [h for h, _ in sorted(horse_odds.items(), key=lambda x: (-x[1], x[0]))][:max(0, k)]

def _pick_favorites(horse_odds, k=FAVORITE_SET_K):
    return [h for h, _ in sorted(horse_odds.items(), key=lambda x: (x[1], x[0]))][:max(0, k)]

def _favorites_close(horse_odds):
    favs = _pick_favorites(horse_odds, FAVORITE_SET_K)
    if len(favs) < 3: return False
    o1, o3 = (horse_odds[favs[0]], horse_odds[favs[2]])
    return (o3 / max(1e-9, o1)) <= FAVORITE_CLOSE_RATIO

# ========= 候補生成 =========
def _tansho_candidates(race_df, p, odds_map):
    horses = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int).values
    cand = []
    for i, h in enumerate(horses):
        odds = odds_map.get(str(h), 0.0) or _pseudo_odds(p[i], "tansho")
        ev = float(p[i] * odds - 1.0)
        if p[i] >= THRESH_WIN_PROB_MIN:
            cand.append({"sel": [int(h)], "p": float(p[i]), "odds": float(odds), "ev": ev})
    cand.sort(key=lambda x: x["p"], reverse=True)
    return cand[:TOPK_TANSHO]

def _place_candidates(race_df, p, odds_map):
    horses = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int).values
    cand = []
    for i, h in enumerate(horses):
        p3 = _p_place_top3(p, i)
        odds = odds_map.get(str(h), 0.0) or _pseudo_odds(p3, "place")
        ev = float(p3 * odds - 1.0)
        if p3 >= max(THRESH_PLACE_BY_WINP, 0.15):
            cand.append({"sel": [int(h)], "p": float(p3), "odds": float(odds), "ev": ev})
    cand.sort(key=lambda x: x["p"], reverse=True)
    return cand[:TOPK_PLACE]

def _pairs_candidates(p, horses, odds_map, kind):
    cand = []
    n = len(p)
    for i in range(n):
        for j in range(i + 1, n):
            pr = _p_umaren(p, i, j)
            odds = odds_map.get(f"{horses[i]}-{horses[j]}", 0.0) or _pseudo_odds(pr, kind)
            ev = float(pr * odds - 1.0)
            cand.append({"sel": [int(horses[i]), int(horses[j])],
                         "p": float(pr), "odds": float(odds), "ev": ev})
    cand.sort(key=lambda x: x["p"], reverse=True)
    # Wideだけは後段の制約/補充に備えてプールを拡張して返す
    limit = MAX_POINTS_EACH_TYPE
    if kind == "wide":
        extra = (TOPK_WIDE or 7) * 3
        limit = max(limit, int(extra))
    return cand[:limit]

def _triples_candidates(p, horses, odds_map, kind):
    cand = []
    n = len(p)
    if kind == "sanrenpuku":
        for i in range(n):
            for j in range(i + 1, n):
                for k in range(j + 1, n):
                    pr = _p_sanrenpuku(p, i, j, k)
                    odds = odds_map.get(f"{horses[i]}-{horses[j]}-{horses[k]}", 0.0) or _pseudo_odds(pr, "sanrenpuku")
                    ev = float(pr * odds - 1.0)
                    cand.append({"sel": [int(horses[i]), int(horses[j]), int(horses[k])],
                                 "p": float(pr), "odds": float(odds), "ev": ev})
    else:
        for i in range(n):
            for j in range(n):
                if j == i: continue
                for k in range(n):
                    if k == i or k == j: continue
                    pr = _p_sanrentan(p, i, j, k)
                    odds = odds_map.get(f"{horses[i]}>{horses[j]}>{horses[k]}", 0.0) or _pseudo_odds(pr, "sanrentan")
                    ev = float(pr * odds - 1.0)
                    cand.append({"sel": [int(horses[i]), int(horses[j]), int(horses[k])],
                                 "p": float(pr), "odds": float(odds), "ev": ev})
    cand.sort(key=lambda x: x["ev"], reverse=True)
    return cand[:MAX_POINTS_EACH_TYPE]

# ========= リスク適応：配分補正 =========
def _risk_adjust_alloc_ratio(p_ens: np.ndarray) -> dict[str, float]:
    # エントロピーで不確実性（0〜logN）→ 0〜1 正規化
    H = _entropy(p_ens)
    Hmax = math.log(len(p_ens) + 1e-9)
    u = float(H / max(1e-9, Hmax))  # 大：不確実（守り）
    # オッズ歪み：上位の単勝オッズ比（強い=攻め）
    # 勝率の集中度：上位2頭の確率比（p2/p1）。小さいほど一強＝攻め/大きいほど拮抗＝守り
    idx = np.argsort(p_ens)[::-1]
    p1 = float(p_ens[idx[0]]) if len(idx) > 0 else 0.0
    p2 = float(p_ens[idx[1]]) if len(idx) > 1 else 0.0
    comp = p2 / max(1e-9, p1)  # 0〜1
    guard = np.clip(0.5 * u + 0.5 * comp, 0.0, 1.0)

    ratio = dict(DEFAULT_ALLOC_RATIO)
    # 守り：place/wide up、攻め：sanrentan/sanrenpuku up
    ratio["place"]      *= (1.0 + 0.8 * guard)
    ratio["wide"]       *= (1.0 + 0.5 * guard)
    ratio["sanrentan"]  *= (1.0 + 0.6 * (1.0 - guard))
    ratio["sanrenpuku"] *= (1.0 + 0.4 * (1.0 - guard))
    s = sum(ratio.values())
    for k in ratio: ratio[k] /= s
    cap = {"sanrenpuku": 0.04, "sanrentan": 0.04}
    for k, v in cap.items():
        if k in ratio:
            ratio[k] = min(ratio[k], v)
    # 正規化
    s = sum(ratio.values())
    for k in ratio:
        ratio[k] /= s
    return ratio

# ========= モデル信頼度適応 =========
def _model_weight_adapt(base_weight: float, model_name: str, bundle: dict,
                        race_df: pd.DataFrame, odds_available: bool) -> float:
    cols = bundle.get("columns", [])
    if cols:
        present = sum(1 for c in cols if c in race_df.columns)
        cover = present / max(1, len(cols))
    else:
        cover = 0.7  # 列リストが無いモデルは控えめに
    w = base_weight * (0.5 + 0.5 * cover)  # 欠損が多いと減衰
    if odds_available and ("oddsaware" in model_name.lower() or "odds_aware" in model_name.lower()):
        w *= 1.15  # 当日オッズが使えるなら aware をややブースト
    return float(w)

# ========= アンサンブル =========
def _ensemble_probs(models: dict[str, dict], race_df: pd.DataFrame, odds_available: bool
                   ) -> Tuple[np.ndarray, dict[str, np.ndarray], dict[str, float]]:
    per: dict[str, np.ndarray] = {}
    eff_w: dict[str, float] = {}
    acc = None; tw = 0.0
    for name, spec in models.items():
        bundle = spec.get("bundle")
        base_w = float(spec.get("weight", 0.0))
        if not bundle or base_w <= 0: continue
        w = _model_weight_adapt(base_w, name, bundle, race_df, odds_available)
        p = _predict_win_prob_generic(bundle, race_df)
        per[name] = p; eff_w[name] = w
        acc = (p * w) if acc is None else (acc + p * w)
        tw += w
    if not per:
        p_ens = np.full(len(race_df), 1.0 / max(1, len(race_df)))
        return p_ens, per, eff_w
    p_ens = _normalize_probs(acc / max(1e-12, tw))
    return p_ens, per, eff_w

# ========= 票ブースト（優先度込み） =========
def _vote_bonus(priorities: dict[str, int], per_model_p: dict[str, np.ndarray],
                horses: np.ndarray) -> dict[int, float]:
    votes_cnt = {int(h): 0 for h in horses}
    votes_pri = {int(h): 0 for h in horses}
    n_models = max(1, len(per_model_p))
    for name, p in per_model_p.items():
        idx = np.argsort(p)[::-1][:ENSEMBLE_VOTE_TOPK]
        s = set(int(horses[i]) for i in idx)
        for h in s:
            votes_cnt[h] += 1
            votes_pri[h] += int(priorities.get(name, 0))
    bonus = {}
    pri_total = max(1, sum(priorities.values()) or 1)
    for h in horses:
        v = votes_cnt[int(h)]
        if v <= 0:
            bonus[int(h)] = 1.0
        else:
            base = 1.0 + ENSEMBLE_VOTE_BONUS_ALPHA * (v / n_models)
            pri_norm = votes_pri[int(h)] / pri_total
            bonus[int(h)] = base * (1.0 + ENSEMBLE_PRIORITY_BONUS_ALPHA * pri_norm)
    return bonus

# ========= 相関制約付きの配分（LP） =========
def _opt_allocate_with_constraints(budget: int, items: list[dict],
                                  horse_cap_ratio: float = 0.35) -> list[int]:
    """
    目的： Σ w_i * a_i を最大化 s.t.
      Σ a_i ≤ budget, 各馬 h の露出 Σ_{i: h∈sel_i} a_i ≤ horse_cap_ratio * budget
    依存: scipy があれば LP、無ければ貪欲にフォールバック
    """
    if budget <= 0 or not items:
        return []
    try:
        import itertools
        from scipy.optimize import linprog  # type: ignore
        n = len(items)
        w = np.array([max(1e-9, float(x.get("p", 0.0))) for x in items])
        # 係数は負にして最小化
        c = -w
        # 総額制約
        A = [np.ones(n)]
        b = [float(budget)]
        # 馬ごとキャップ
        horses = sorted({h for it in items for h in it["sel"]})
        for h in horses:
            row = [1.0 if h in it["sel"] else 0.0 for it in items]
            A.append(row); b.append(float(horse_cap_ratio * budget))
        bounds = [(0, float(budget))] * n
        res = linprog(c, A_ub=np.array(A), b_ub=np.array(b), bounds=bounds, method="highs")
        if res.success:
            cont = np.clip(res.x, 0.0, float(budget))
            # 100円単位へ丸め直し
            amounts = [_to_int100(x) for x in cont]
            # 合計を合わせる
            diff = budget - sum(amounts); i = 0
            while diff >= BASE_BET_UNIT and amounts:
                amounts[i] += BASE_BET_UNIT; diff -= BASE_BET_UNIT; i = (i + 1) % len(amounts)
            return amounts
        # 失敗時はフォールバックへ
    except Exception:
        pass

    # フォールバック：EV重みの確率的配分＋馬別キャップで貪欲に調整
    weights = [max(1e-9, float(x.get("p", 0.0))) for x in items]
    S = sum(weights); amounts = []
    if S > 0:
        raw = [budget * w / S for w in weights]
        amounts = [_to_int100(a) for a in raw]
    else:
        # 均等
        base = _to_int100(budget / len(items))
        amounts = [base] * len(items)
    # 馬別キャップ適用
    cap = int(horse_cap_ratio * budget)
    expo = {}
    for idx, it in enumerate(items):
        for h in it["sel"]:
            expo[h] = expo.get(h, 0) + amounts[idx]
    # キャップ超過分を徐々に削る
    changed = True
    while changed:
        changed = False
        for h, s in list(expo.items()):
            if s > cap:
                over = s - cap
                for idx, it in enumerate(items):
                    if over <= 0: break
                    if h in it["sel"] and amounts[idx] > 0:
                        dec = min(amounts[idx], over, BASE_BET_UNIT)
                        amounts[idx] -= dec; over -= dec; expo[h] -= dec; changed = True
    # 余りを他へ再分配
    rest = budget - sum(amounts); i = 0
    while rest >= BASE_BET_UNIT and amounts:
        amounts[i] += BASE_BET_UNIT; rest -= BASE_BET_UNIT; i = (i + 1) % len(amounts)
    return amounts

# ========= オンライン校正（任意） =========
def _online_platt_update(p: np.ndarray, recent_outcomes: Optional[Iterable[int]]) -> np.ndarray:
    """
    直近の (y in {0,1}) を使って簡易に a,b を更新し、 sigmoid(a*z + b) を適用。
    ここでは z = logit(p) を用い、最急降下 3 step 程度の軽量更新（データが無ければそのまま）。
    """
    if not recent_outcomes:
        return p
    y = np.array(list(recent_outcomes), dtype=float)
    if y.size < 10:  # データ少なら何もしない
        return p
    z = np.log(np.clip(p, 1e-6, 1-1e-6)) - np.log(1 - np.clip(p, 1e-6, 1-1e-6))
    a, b = 1.0, 0.0
    lr = 0.05
    for _ in range(3):
        q = 1.0 / (1.0 + np.exp(-(a*z + b)))
        da = np.sum((q - y) * z) / y.size
        db = np.sum(q - y) / y.size
        a -= lr * da; b -= lr * db
    p2 = 1.0 / (1.0 + np.exp(-(a*z + b)))
    return _normalize_probs(p2)

# ========= 1レース計画（発展版） =========
def _plan_one_race_advanced(race_df: pd.DataFrame, odds_maps: dict,
                            models: dict[str, dict],
                            recent_outcomes: Optional[Iterable[int]] = None) -> list[dict]:
    horses = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int).values
    odds_available = bool(odds_maps.get("tansho"))
    # odds_available = bool(odds_maps.get("tansho")) の直後に
    if odds_available:
        # 単勝オッズを馬番で race_df に張る（モデルが columns に含めていれば自動で使われる）
        win_map = odds_maps.get("tansho", {})
        hnums = pd.to_numeric(race_df["horse_num"], errors="coerce").fillna(0).astype(int)
        race_df["win_odds"] = hnums.map(lambda h: float(win_map.get(str(int(h)), 0.0))).astype(float)
        # もし学習時に log/逆数などを入れているなら任意で派生列も用意（存在すれば拾われる）
        if "log_win_odds" in {c for spec in models.values() for c in (spec.get("bundle",{}).get("columns",[]))}:
            race_df["log_win_odds"] = np.log(np.where(race_df["win_odds"]>0, race_df["win_odds"], np.nan)).fillna(0.0)
        if "imp_win_prob" in {c for spec in models.values() for c in (spec.get("bundle",{}).get("columns",[]))}:
            race_df["imp_win_prob"] = 1.0 / np.clip(race_df["win_odds"], 1e-9, None)

    # 1) アンサンブル（信頼度適応）
    p_ens, per_model_p, eff_w = _ensemble_probs(models, race_df, odds_available)
    # 1.5) オンライン校正（任意）
    p_ens = _online_platt_update(p_ens, recent_outcomes)

    # 2) リスク適応の配分
    alloc_ratio = _risk_adjust_alloc_ratio(p_ens)
    
     # 単体 p の参照用（馬番 -> p）
    p_map = {int(horses[i]): float(p_ens[i]) for i in range(len(horses))}

    # 3) 票ブーストを単勝/複勝の重みに
    priorities = {name: int(spec.get("priority", 0)) for name, spec in models.items() if spec.get("bundle")}
    bonus_map = _vote_bonus(priorities, per_model_p, horses)

    # 4) 候補生成
    cand_place  = _place_candidates(race_df, p_ens, odds_maps.get("place", {}))
    place_horses = {int(c["sel"][0]) for c in cand_place}  # 複勝候補の馬番集合
    cand_tan    = _tansho_candidates(race_df, p_ens, odds_maps.get("tansho", {}))
    cand_umaren = _pairs_candidates(p_ens, horses, odds_maps.get("umaren", {}), "umaren")
    cand_wide   = _pairs_candidates(p_ens, horses, odds_maps.get("wide",   {}), "wide")
        # 単体 p マップ（馬番 -> p）
    p_map = {int(horses[i]): float(p_ens[i]) for i in range(len(horses))}

    # ランダム性の強さ(温度)は不確実性に応じて可変：u = H/Hmax
    H = _entropy(p_ens)
    Hmax = math.log(len(p_ens) + 1e-9)
    u = float(H / max(1e-9, Hmax))               # 0〜1（大きいほど拮抗＝ランダム強め）

    # レース単位で再現性を持たせるための乱数生成器（race_id種）
    race_id_seed = int(pd.to_numeric(race_df.iloc[0].get("race_id", 0), errors="coerce") or 0) & 0xFFFFFFFF
    rng = np.random.default_rng(race_id_seed ^ 0x9E3779B9)

    # ワイドの「スコア」＝ 組のp × 構成馬の単体p平均（高pを含むほど上がる）
    def _score_wide(c):
        h1, h2 = c["sel"]
        avg_single_p = (p_map.get(h1, 0.0) + p_map.get(h2, 0.0)) / 2.0
        s = float(c["p"]) * float(avg_single_p)
        # 複勝候補が含まれるペアはスコアをブースト
        if h1 in place_horses or h2 in place_horses:
            s *= float(WIDE_PLACE_BONUS)
        return s

    # 温度tauは 0.8〜1.6 目安で可変（新規定数は作らずローカルで決定）
    tau = 0.8 + 0.8 * u
    # 動的K: 荒れなら K を拡張（最大9）
    base_k = TOPK_WIDE if TOPK_WIDE else 7
    top1 = float(np.max(p_ens)) if len(p_ens) else 0.0
    is_areru = (u >= 0.80) or (top1 <= 0.22)
    wide_target_k = min(9, (base_k + 2) if is_areru else base_k)

    # 確率的Top-K（余裕を持って 2×K をサンプル）
    cand_wide_all = list(cand_wide)
    if cand_wide:
        pool_k = min(len(cand_wide), max(wide_target_k, base_k) * 2)
        cand_wide_pool = _stochastic_topk(cand_wide, _score_wide, pool_k, tau, rng)
        cand_wide = cand_wide_pool[:wide_target_k]
    else:
        wide_target_k = 0

    # （必要なら）この後に既存の散らし・上限制約を適用
    if MAX_WIDE_PER_HORSE and cand_wide:
        from collections import defaultdict
        used = defaultdict(int)
        pruned = []
        # まずキャップ適用（ここで入った分は“補充ではない”）
        for c in cand_wide:
            if MAX_WIDE_PER_HORSE and any(used[h] >= MAX_WIDE_PER_HORSE for h in c["sel"]):
                continue
            pruned.append(c)
            for h in c["sel"]:
                used[h] += 1
        # 目標本数まで補充（スコアが十分なものだけ・補充フラグを付ける）
        if wide_target_k:
            chosen = {tuple(sorted(c["sel"])) for c in pruned}
            # ベースとなるスコア（補充の最低基準を決める）
            def _best_score(pool_lists):
                best = 0.0
                for pl in pool_lists:
                    if not pl: 
                        continue
                    s = max((_score_wide(x) for x in pl), default=0.0)
                    if s > best:
                        best = s
                return best
            base_best = _best_score([pruned, cand_wide])  # 現採用・候補双方から
            thr = float(base_best) * float(WIDE_REFILL_MIN_SCORE_RATIO)
            def refill_from(pool):
                added = 0
                for c in sorted(pool, key=_score_wide, reverse=True):
                    key = tuple(sorted(c["sel"]))
                    if key in chosen:
                        continue
                    if MAX_WIDE_PER_HORSE and any(used[h] >= MAX_WIDE_PER_HORSE for h in c["sel"]):
                        continue
                    # スコアがしきい値未満なら補充しない（= 無理に本数を埋めない）
                    if _score_wide(c) < thr:
                        continue
                    d = dict(c)
                    d["_refill"] = True     # ← 補充フラグ
                    pruned.append(d); chosen.add(key)
                    for h in d["sel"]:
                        used[h] += 1
                    added += 1
                    if len(pruned) >= wide_target_k:
                        break
                return added

            if 'cand_wide_pool' in locals():
                refill_from(cand_wide_pool)
            if len(pruned) < wide_target_k:
                refill_from(cand_wide_all)

        cand_wide = pruned
    cand_srpk   = _triples_candidates(p_ens, horses, odds_maps.get("sanrenpuku", {}), "sanrenpuku")
    cand_srtan  = _triples_candidates(p_ens, horses, odds_maps.get("sanrentan",  {}), "sanrentan")

    # 4.5) ラウンド別予算の上書き（race_num ベース）
    try:
        race_no = int(pd.to_numeric(race_df["race_num"], errors="coerce").iloc[0])
    except Exception:
        race_no = None
    ACTUAL_RACE_BUDGET_YEN = int(ROUND_BUDGET_OVERRIDES.get(race_no, PER_RACE_BUDGET_YEN))
    # 5) 予算配分（相関制約込み）
    sub = {k: int(ACTUAL_RACE_BUDGET_YEN * alloc_ratio.get(k, 0.0))
           for k in ["place","tansho","umaren","wide","sanrenpuku","sanrentan"]}
    by_kind = {"place":cand_place,"tansho":cand_tan,"umaren":cand_umaren,"wide":cand_wide,
               "sanrenpuku":cand_srpk,"sanrentan":cand_srtan}

    bet_items = []
    for kind, cands in by_kind.items():
        if kind == "wide":
            kfinal = (wide_target_k if 'wide_target_k' in locals() and wide_target_k else (TOPK_WIDE or len(cands)))
            cands = cands[:kfinal]
        budget_k = sub.get(kind, 0)
        if kind == "place" and len(cands) == 1:
            cap_yen = int(ACTUAL_RACE_BUDGET_YEN * PLACE_SOLO_BUDGET_CAP_RATIO)
            budget_k = min(budget_k, cap_yen)
        
        if budget_k <= 0 or not cands: 
            continue
        # EV × 票ブースト（単/複のみ）を重みとして LP で最適化
        items = []
        for c in cands:
            # 基本は「組の p」
            w = float(c.get("p", 0.0))

            # 単体pによる“構成ボーナス”を追加（既存変数のみ・新規定数なし）
            if kind in ("umaren", "wide"):
                sel = c["sel"]  # [h1, h2]
                avg_single_p = (p_map.get(sel[0], 0.0) + p_map.get(sel[1], 0.0)) / 2.0
                w *= float(avg_single_p)
            elif kind in ("sanrenpuku", "sanrentan"):
                sel = c["sel"]  # [h1, h2, h3]
                avg_single_p = sum(p_map.get(h, 0.0) for h in sel) / 3.0
                w *= float(avg_single_p)
            else:  # 単/複は既存の票ボーナスをそのまま活用
                if kind in ("tansho", "place"):
                    h = int(c["sel"][0])
                    w *= float(bonus_map.get(h, 1.0))

            items.append({"kind":kind, "sel":c["sel"], "p":c["p"], "odds":c["odds"], "ev":c["ev"]})
        cap_ratio = HORSE_CAP_BY_KIND.get(kind, 0.25)
        
        amounts = _opt_allocate_with_constraints(budget_k, items, horse_cap_ratio=cap_ratio)
        for c, a in zip(items, amounts):
            # 補充で採用されたワイドは金額を控えめに
            if kind == "wide" and c.get("_refill"):
                a = max(MIN_AMOUNT_PER_POINT, int(a * WIDE_REFILL_STAKE_RATIO))
            bet_items.append({"kind":c["kind"], "sel":c["sel"], "amount":int(a),
                              "p":float(c["p"]), "odds":float(c["odds"]), "ev":float(c["ev"])})
        # --- 同一の賭け（種別・選択が同じ）の二重登録を統合 ---
    if bet_items:
        merged = {}
        for b in bet_items:
            key = (b["kind"], tuple(b["sel"]))
            if key not in merged:
                merged[key] = dict(b)
            else:
                merged[key]["amount"] += int(b["amount"])
                # 表示用に p / odds / ev は大きい方を残す（集計には影響なし）
                merged[key]["p"] = max(merged[key]["p"], float(b["p"]))
                merged[key]["odds"] = max(merged[key]["odds"], float(b["odds"]))
                merged[key]["ev"] = max(merged[key]["ev"], float(b["ev"]))
        bet_items = list(merged.values())

    if SAFE_FALLBACK_AMOUNT > 0 and not bet_items:
        best = int(np.argmax(p_ens)); h = int(horses[best])
        pk = SAFE_FALLBACK_KIND
        pv = _p_place_top3(p_ens, best) if pk=="place" else p_ens[best]
        bet_items = [{"kind":pk, "sel":[h], "amount":SAFE_FALLBACK_AMOUNT,
                      "p":float(pv), "odds":_pseudo_odds(pv, pk), "ev":0.0}]

    # 印
    mark_order = []
    for item in sorted(bet_items, key=lambda x: x["p"], reverse=True):
        for h in item["sel"]:
            if h not in mark_order: mark_order.append(h)
    mark = {}
    for i, h in enumerate(mark_order):
        mark[str(h)] = 1 if i==0 else 2 if i==1 else 3 if i==2 else 4

    race_row = race_df.iloc[0]
    payload = _payload_mark_bet(race_row, bet_items, mark)
    out = []
    for item in bet_items:
        out.append({
            "race_id": race_row.get("race_id"),
            "bet_type_name": item["kind"],
            "selection": item["sel"],
            "amount": item["amount"],
            "p": float(item["p"]), "odds": float(item["odds"]), "ev": float(item["ev"]),
            "api_payload": payload,
        })
    return out

def _payload_mark_bet(race_row: pd.Series, bet_items: list[dict], mark: dict) -> dict:
    kind_code = {"tansho":1,"place":2,"wakuren":3,"umaren":4,"wide":5,"umatan":6,"sanrenpuku":7,"sanrentan":8}
    bet_list = []
    for item in bet_items:
        sel_str = "_".join(str(x) for x in item["sel"])
        bet_id = f"b{kind_code[item['kind']]}_c0_{sel_str}"
        bet_list.append({"bet_id": bet_id, "money": str(int(item["amount"]))})
    return {"race_id": str(race_row.get("race_id","")), "mark": mark, "bet": bet_list}

# ========= デイリープラン（公開） =========
def make_plan_for_day(timetable: pd.DataFrame, runtable: pd.DataFrame,
                      get_odds_func: Callable[[str], dict] | None = None,
                      date_yyyymmdd: str | None = None, verbose: int = 1,
                      models: dict[str, dict] | None = None,
                      recent_outcomes_map: Optional[dict[str, Iterable[int]]] = None) -> pd.DataFrame:
    """
    models: { name: {"bundle": model_bundle, "weight": float, "priority": int}, ... }
    recent_outcomes_map: { race_id(str): iterable of 0/1 }  # 当日の先行レース結果があれば簡易オンライン校正に使用
    """
    if "race_id" not in runtable.columns:
        raise RuntimeError("runtable に race_id 列が必要です")
    if not models:
        # モデル未指定なら等確率で進める（最低限の安全運転）
        models = {}

    plans = []
    for race_id, race_df in runtable.groupby("race_id"):
        race_df = race_df.copy()
        odds_raw = _fetch_odds_safe(get_odds_func, str(race_df.iloc[0].get("race_id_odds", "")) or str(race_id))
        odds_maps = {k: _parse_odds_table(odds_raw, k) for k in
                     ["tansho","place","umaren","wide","sanrenpuku","sanrentan","wakuren"]}

        recents = None
        if recent_outcomes_map and str(race_id) in recent_outcomes_map:
            recents = recent_outcomes_map[str(race_id)]

        bet_list = _plan_one_race_advanced(race_df, odds_maps, models, recent_outcomes=recents)
        plans.extend(bet_list)

    return pd.DataFrame(plans)

# ========= （任意）ベイジアン最適化 / 代替サーチ =========
def tune_ensemble(models: dict[str, dict],
                  days: Iterable[str],
                  backtest_provider: Callable[[str], Tuple[pd.DataFrame, pd.DataFrame, dict]],
                  n_trials: int = 40) -> dict[str, dict]:
    """
    backtest_provider(date) -> (timetable_df, runtable_df, outcomes_map)
      outcomes_map: {race_id: iterable of 0/1（勝ち=1）}
    目的：総回収率（=総払い戻し / 総投資）最大化。
    戻り値：チューニング後の models（weight/priority/alpha を上書き）
    """
    names = [k for k in models.keys()]
    base = {k: {"weight": float(models[k].get("weight", 0.5)),
                "priority": int(models[k].get("priority", 100))}
            for k in names}

    # 目的関数
    def objective(xw: dict) -> float:
        # xw = {name: (weight, priority)}
        trial_models = {}
        for n in names:
            m = dict(models[n])
            m["weight"] = float(xw[n][0])
            m["priority"] = int(round(xw[n][1]))
            trial_models[n] = m
        invest = 0; payout = 0
        for d in days:
            tt, rt, out_map = backtest_provider(d)
            df = make_plan_for_day(tt, rt, get_odds_func=None, models=trial_models,
                                   recent_outcomes_map=out_map)
            invest += int(df["amount"].sum()) if not df.empty else 0
            # 簡易：的中は ev>0 かつ抽選（実際は結果照合が必要。backtest_provider で補えばOK）
            # ここでは outcomes_map を馬単位に使う場合は適宜拡張してください。
            # 今回は安全に「投資額を小さく、ev重いほど当たりやすい」として proxy を置く。
            payout += float((df["amount"] * df["p"]).sum()) if not df.empty else 0.0
        if invest <= 0:
            return -0.0
        roi = payout / invest
        return -roi  # 最小化 → ROI最大化

    # できれば optuna / skopt を使う
    used_bo = False
    try:
        import optuna  # type: ignore
        used_bo = True
        def _opt_objective(trial: "optuna.Trial"):
            xw = {}
            for n in names:
                w = trial.suggest_float(f"w_{n}", 0.05, 1.5)
                p = trial.suggest_int(f"p_{n}", 50, 200)
                xw[n] = (w, p)
            return objective(xw)
        study = optuna.create_study(direction="minimize")
        study.optimize(_opt_objective, n_trials=n_trials, show_progress_bar=False)
        best = study.best_params
        for n in names:
            models[n]["weight"] = float(best[f"w_{n}"])
            models[n]["priority"] = int(best[f"p_{n}"])
        return models
    except Exception:
        pass
    if not used_bo:
        try:
            from skopt import gp_minimize  # type: ignore
            used_bo = True
            space = []
            low, high = [], []
            for n in names:
                low.append(0.05); high.append(1.5)  # weight
                low.append(50);   high.append(200)  # priority
            bounds = list(zip(low, high))
            def f(vec):
                xw = {}
                it = iter(vec)
                for n in names:
                    w = float(next(it)); p = float(next(it))
                    xw[n] = (w, p)
                return objective(xw)
            res = gp_minimize(f, bounds, n_calls=n_trials, random_state=42)
            vec = res.x
            it = iter(vec)
            for n in names:
                models[n]["weight"] = float(next(it))
                models[n]["priority"] = int(round(next(it)))
            return models
        except Exception:
            pass

    # フォールバック：座標降下 + ランダムリスタート
    best_models = {k: dict(models[k]) for k in names}
    best_score = objective({n:(best_models[n]["weight"], best_models[n]["priority"]) for n in names})
    rng = np.random.default_rng(42)
    for _ in range(max(5, n_trials//5)):
        cur = {k: dict(best_models[k]) for k in names}
        for _step in range(15):
            for n in names:
                for scale in [0.8, 1.2]:
                    trial = {k: dict(cur[k]) for k in names}
                    trial[n]["weight"] = float(np.clip(cur[n]["weight"] * scale, 0.05, 1.5))
                    trial[n]["priority"] = int(np.clip(cur[n]["priority"] + rng.integers(-10, 11), 50, 200))
                    sc = objective({k:(trial[k]["weight"], trial[k]["priority"]) for k in names})
                    if sc < best_score:
                        best_score = sc; best_models = trial
                        cur = {k: dict(trial[k]) for k in names}
        # random restart
        for n in names:
            best_models[n]["weight"] = float(rng.uniform(0.05, 1.5))
            best_models[n]["priority"] = int(rng.integers(50, 201))
    return best_models