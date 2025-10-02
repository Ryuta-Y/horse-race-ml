# scripts/watch_vote_2.py
# 使い方:
#   (venv) $ python -m scripts.watch_vote_2
#
# 概要:
# - conf.py の TARGET_DATE / TARGET_PLACE / TARGET_RACE を対象に、
#   タイムテーブルの「発走5分前(厳密には4分10秒前)」で自動的に
#   計画作成(odds_policy.make_plan_for_day)→投票API送信を行う常駐スクリプト。
# - odds_policy の出力契約は bet_policy と互換（payload形式は同じ）。
#
# 注意:
# - トークン有効期限は5分。毎レース直前にログイン→投票→ログアウト。
# - 金額はポリシー側で100円単位に丸め済み。ここでは再丸めしない。

from __future__ import annotations
import os
import time
import json
from datetime import datetime
import pandas as pd
import joblib
import requests

from conf import (
    TARGET_DATE, TARGET_PLACE, TARGET_RACE,
    VERIFY_TLS, RACEDAY_API_BASE, RACEDAY_API_KEY,
    BET_API_LOGIN, BET_API_BET, BET_API_LOGOUT,
    LOGIN_ID, LOGIN_PW,
    ODDSLESS_MODEL_PATH, ODDSAWARE_MODEL_PATH,
    PLAN_DIR, DRY_RUN, LOG_VERBOSITY,
    ENSEMBLE_MODEL_SPECS,  # ← 追加
)

# ===== APIレスポンス正規化（dict/list揺れの吸収） =========================
def _unwrap_api_json(obj) -> dict:
    """
    常に {'data': {...}} 形式に寄せる。
      - {'data': {...}}
      - {'timetable': [...], 'runtable': [...]}
      - [{'data': {...}}]
      - [{'timetable': [...], 'runtable': [...]}]
    """
    if isinstance(obj, dict):
        if "data" in obj:
            if isinstance(obj["data"], dict):
                return obj
            if isinstance(obj["data"], list):
                for el in obj["data"]:
                    if isinstance(el, dict) and (("timetable" in el) or ("runtable" in el)):
                        return {"data": el}
                return {"data": {}}
        if ("timetable" in obj) or ("runtable" in obj):
            return {"data": obj}
        return {"data": {}}
    if isinstance(obj, list):
        for el in obj:
            if isinstance(el, dict) and "data" in el and isinstance(el["data"], dict):
                return el
        for el in obj:
            if isinstance(el, dict) and (("timetable" in el) or ("runtable" in el)):
                return {"data": el}
        return {"data": {}}
    return {"data": {}}

# ============== API ============================================
def api_get_racecards(date_yyyymmdd: str) -> dict:
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Racecards", "id": date_yyyymmdd}
    res = requests.get(url, headers=headers, verify=VERIFY_TLS, timeout=20)
    res.raise_for_status()
    return _unwrap_api_json(res.json())

def api_get_odds(race_id_jjrr: str) -> dict:
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Odds", "id": race_id_jjrr}
    res = requests.get(url, headers=headers, verify=VERIFY_TLS, timeout=20)
    res.raise_for_status()
    return _unwrap_api_json(res.json())

def api_login(login_id: str, password: str) -> str:
    payload = {"login_id": login_id, "password": password}
    res = requests.post(BET_API_LOGIN, json=payload, timeout=20)
    if res.status_code == 200:
        print("[login] ログイン成功")
        return res.json()["data"]["access_token"]
    print(f"[login] ログイン失敗: {res.status_code} {res.text}")
    return ""

def api_bet(access_token: str, bet_data_list: list[dict]) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"bet_data": bet_data_list}
    res = requests.post(BET_API_BET, headers=headers, json=payload, timeout=60)
    if res.status_code == 200:
        print("[bet] ベット成功")
        return res.json()
    print(f"[bet] ベット失敗: {res.status_code} {res.text}")
    return {}

def api_logout(access_token: str) -> None:
    headers = {"Authorization": f"Bearer {access_token}"}
    res = requests.post(BET_API_LOGOUT, headers=headers, timeout=20)
    print("[logout] ログアウト成功" if res.status_code == 200 else f"[logout] ログアウト失敗: {res.status_code}")

# ============== 列名正規化 =====================================
def normalize_runtable_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for cand in ["horse_num", "umaban", "馬番"]:
        if cand in df.columns: colmap[cand] = "horse_num"; break
    for cand in ["waku_num", "枠番"]:
        if cand in df.columns: colmap[cand] = "waku_num"; break
    for cand in ["race_id", "raceId", "レースID"]:
        if cand in df.columns: colmap[cand] = "race_id"; break
    for cand in ["place", "場名"]:
        if cand in df.columns: colmap[cand] = "place"; break
    for cand in ["race_num", "R", "レース番号"]:
        if cand in df.columns: colmap[cand] = "race_num"; break
    for cand in ["win_odds", "単勝オッズ", "odds_win"]:
        if cand in df.columns: colmap[cand] = "win_odds"; break

    df = df.rename(columns=colmap).copy()
    if "race_id" in df.columns:
        df["race_id_full"] = df["race_id"].astype(str)
        def to_odds_id(rid: str) -> str:
            s = str(rid)
            if len(s) >= 16:
                return s[:8] + s[8:10] + s[14:16]  # YYYYMMDD + place + race
            return s
        def to_bet_id(rid: str) -> str:
            s = str(rid)
            if len(s) >= 16:
                return s[:4] + s[8:16]            # YYYY + place + kaiji + nichi + race
            return s
        df["race_id_odds"] = df["race_id_full"].apply(to_odds_id)
        df["race_id_bet"]  = df["race_id_full"].apply(to_bet_id)
        df["race_id"] = df["race_id_bet"]

    if "horse_num" not in df.columns:
        df["horse_num"] = (df.groupby("race_id").cumcount() + 1) if "race_id" in df.columns else (df.index + 1)
    if "waku_num" not in df.columns:
        df["waku_num"] = 1
    return df

# ============== モデル読み込み（存在するもの全部） ============================
def load_models_for_ensemble() -> dict:
    models = {}
    for spec in ENSEMBLE_MODEL_SPECS:
        try:
            name, path, weight, priority = spec
        except ValueError:
            name, path, weight = spec
            priority = 0
        if os.path.exists(path):
            try:
                b = joblib.load(path)
                models[name] = {"bundle": b, "weight": float(weight), "priority": int(priority)}
                print(f"[model] loaded: {name} from {path} (w={weight}, prio={priority})")
            except Exception as e:
                print(f"[model] skip {name}: load error: {e}")
        else:
            print(f"[model] missing: {name} path={path}")

    if not models:
        # 旧2本のフォールバック
        if os.path.exists(ODDSAWARE_MODEL_PATH):
            try:
                b = joblib.load(ODDSAWARE_MODEL_PATH)
                models["oddsaware"] = {"bundle": b, "weight": 0.60, "priority": 100}
                print(f"[model] fallback loaded: oddsaware from {ODDSAWARE_MODEL_PATH}")
            except Exception as e:
                print(f"[model] fallback skip oddsaware: {e}")
        if os.path.exists(ODDSLESS_MODEL_PATH):
            try:
                b = joblib.load(ODDSLESS_MODEL_PATH)
                models["oddsless"]  = {"bundle": b, "weight": 0.40, "priority": 90}
                print(f"[model] fallback loaded: oddsless from {ODDSLESS_MODEL_PATH}")
            except Exception as e:
                print(f"[model] fallback skip oddsless: {e}")
    if not models:
        print("[warn] 有効なモデルが見つかりません（均等確率モードで進行）")
    return models

# ============== ポリシー（odds_policy推奨 / 互換フォールバック付） ============
from odds_policy import make_plan_for_day

LEAD_SEC = 4 * 60 + 10   # 発走の 4分10秒前
POLL_INTERVAL = 5

def _list_target_slots(timetable: pd.DataFrame) -> list[tuple[str, int, str]]:
    if timetable.empty: return []
    df = timetable.copy()
    if TARGET_PLACE:
        df = df[df.get("place") == TARGET_PLACE]
    if TARGET_RACE:
        df = df[df.get("race_num").astype(str) == str(int(TARGET_RACE))]
    df = df.sort_values("start_time")
    return [(r.place, int(r.race_num), r.start_time) for _, r in df.iterrows()]

def _log_countdown(wait_sec: int, start_hhmm: str):
    left = max(wait_sec, 0)
    print(f"[{start_hhmm}] 5分前トリガーまで {left} 秒待機...")
    last = left
    while left > 0:
        time.sleep(min(POLL_INTERVAL, left))
        left = max(0, left - POLL_INTERVAL)
        if left == 0 or left // 20 != last // 20:
            print(f"  ... 残り {left} 秒")
        last = left

def _build_and_bet_for(place: str, race_num: int, date_str: str):
    print(f"[plan] 対象: {place}{race_num}R / 日付={date_str}")

    rc = api_get_racecards(date_str)
    data = rc.get("data", {})
    timetable = pd.DataFrame(data.get("timetable", []))
    runtable  = pd.DataFrame(data.get("runtable",  []))
    if runtable.empty:
        print("[plan] runtable空: 中止"); return

    runtable = normalize_runtable_columns(runtable)
    try:
        mask = (runtable.get("place") == place) & (runtable.get("race_num").astype(int) == int(race_num))
        runtable = runtable[mask]
    except Exception:
        print("[plan] レースフィルタに失敗: 中止"); return
    if runtable.empty:
        print("[plan] 指定レースの出走表なし: 中止"); return

    # モデル（存在するもの全部）
    models = load_models_for_ensemble()
    mode = "+".join(models.keys()) if models else "uniform"
    print(f"[plan] 使用モデル: {mode}")

    # odds_policy の新旧シグネチャ吸収（models=優先・ダメなら旧式）
    try:
        plan_df = make_plan_for_day(
            timetable=timetable, runtable=runtable,
            get_odds_func=api_get_odds, date_yyyymmdd=date_str,
            verbose=LOG_VERBOSITY, models=models,
        )
    except TypeError:
        # 旧 bet_policy 互換
        oddsless  = models.get("oddsless", {}).get("bundle")  if models else None
        oddsaware = models.get("oddsaware", {}).get("bundle") if models else None
        plan_df = make_plan_for_day(
            timetable=timetable, runtable=runtable,
            oddsless=oddsless, oddsaware=oddsaware,
            get_odds_func=api_get_odds, date_yyyymmdd=date_str,
            verbose=LOG_VERBOSITY,
        )

    print(f"[plan] 生成件数: {len(plan_df)}")
    if plan_df.empty:
        print("[plan] プランなし（しきい値/データ要確認）"); return

    # 保存（監査用）
    os.makedirs(PLAN_DIR, exist_ok=True)
    fname = f"{date_str}_{place}_{race_num:02d}_plan.json"
    fpath = os.path.join(PLAN_DIR, fname)
    plan_df.to_json(fpath, orient="records", force_ascii=False)
    print(f"[plan] 保存: {fpath}")

    if DRY_RUN:
        print("[bet] DRY_RUN=True のため送信しません")
        return

    # 投票：直前ログイン→送信→ログアウト
    token = api_login(LOGIN_ID or os.getenv("KEIBA_LOGIN_ID", ""),
                      LOGIN_PW  or os.getenv("KEIBA_PASSWORD", ""))
    if not token:
        print("[bet] トークン取得失敗: 中止"); return

    bet_data_list = [row["api_payload"] for _, row in plan_df.iterrows() if "api_payload" in row]
    if not bet_data_list:
        print("[bet] API payloadなし: 中止")
        api_logout(token); return

    print(f"[bet] 送信件数: {len(bet_data_list)} / 例: {bet_data_list[0] if bet_data_list else 'N/A'}")
    res = api_bet(token, bet_data_list)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if isinstance(res, dict) and "remaining_money" in res:
        print(f"[bet] remaining money: {res['remaining_money']}")
    api_logout(token)

def main():
    date_str = TARGET_DATE if TARGET_DATE else datetime.now().strftime("%Y%m%d")
    print(f"[watch_vote_2] TARGET_DATE={date_str}, TARGET_PLACE={TARGET_PLACE}, TARGET_RACE={TARGET_RACE}")

    try:
        rc = api_get_racecards(date_str)
    except Exception as e:
        print(f"[watch] 当日データ取得失敗: {e}")
        return

    data = rc.get("data", {})
    timetable = pd.DataFrame(data.get("timetable", []))
    if timetable.empty:
        print("[watch] timetable空: 終了")
        return

    slots = _list_target_slots(timetable)
    if not slots:
        print("[watch] 指定条件に一致するレースなし: 終了")
        return

    for place, race_num, start_hhmm in slots:
        # 「発走の4分10秒前」まで待機
        today = datetime.now()
        try:
            target = datetime.strptime(start_hhmm, "%H:%M").replace(year=today.year, month=today.month, day=today.day)
        except Exception:
            print(f"[watch] 時刻解析失敗({start_hhmm}): スキップ"); continue
        wait_sec = int((target - datetime.now()).total_seconds()) - (4 * 60 + 10)
        if wait_sec <= 0:
            print(f"[watch] [{start_hhmm}] 5分前取得時刻を過ぎています: スキップ"); continue

        print(f"[watch] 次の対象: {place}{race_num}R (発走 {start_hhmm})")
        _log_countdown(wait_sec, start_hhmm)

        try:
            _build_and_bet_for(place, race_num, date_str)
        except KeyboardInterrupt:
            print("\n[watch] 中断されました。終了。"); return
        except Exception as e:
            print(f"[watch] 例外によりこのレースをスキップ: {e}")

    print("[watch] すべての対象レース処理が完了しました。")

if __name__ == "__main__":
    # TLS警告を抑制したい場合:
    # import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()