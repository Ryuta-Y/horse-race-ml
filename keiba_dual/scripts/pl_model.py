# keiba_dual_ready_full/scripts/pl_model.py
# ========================================
from typing import Dict, List, Tuple
from itertools import permutations, combinations

def normalize(d: Dict[int, float]) -> Dict[int, float]:
    s = sum(max(v, 0.0) for v in d.values())
    if s <= 0: s = 1.0
    return {k: max(v,0.0)/s for k,v in d.items()}

def pl_topk_probs(strength: Dict[int, float], k: int=3) -> Dict[Tuple[int,...], float]:
    """
    Plackett–Luce: 順序確率 P(i1->i2->...->ik) = ∏_{t=1..k} s_{it} / (∑_{j∈残} s_j)
    strength: {horse_num: s_i}（勝率推定などを非負に変換）
    """
    s = normalize(strength)
    keys = list(s.keys())
    probs: Dict[Tuple[int,...], float] = {}
    for order in permutations(keys, k):
        rem = set(keys)
        p = 1.0
        for it in order:
            denom = sum(s[j] for j in rem)
            if denom <= 0: 
                p = 0.0; break
            p *= s[it] / denom
            rem.remove(it)
        probs[order] = p
    return probs

def pair_unordered_from_pl(strength: Dict[int,float]) -> Dict[Tuple[int,int], float]:
    """
    馬連/ワイド用の“順不同2頭が上位2or3着内に入る”確率の近似。
    まずはTop-2の順序あり確率を足し合わせて順不同化。
    """
    top2 = pl_topk_probs(strength, k=2)
    pair = {}
    for (a,b), p in top2.items():
        key = tuple(sorted((a,b)))
        pair[key] = pair.get(key, 0.0) + p
    return pair

def trifecta_from_pl(strength: Dict[int,float]) -> Dict[Tuple[int,int,int], float]:
    """三連単（順序あり）確率分布"""
    return pl_topk_probs(strength, k=3)

def trifecta_box_from_pl(strength: Dict[int,float]) -> Dict[Tuple[int,int,int], float]:
    """三連複（順不同）確率：三連単の順序確率を順不同で足し込み"""
    tri = pl_topk_probs(strength, k=3)
    box = {}
    for (a,b,c), p in tri.items():
        key = tuple(sorted((a,b,c)))
        box[key] = box.get(key, 0.0) + p
    return box