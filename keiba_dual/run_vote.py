# keiba_dual_ready_full/run_vote.py
# 実行例: (venv) $ python -m run_vote

import os
import json
from datetime import datetime
import pandas as pd
import joblib
import requests

from conf import (
    TARGET_DATE, VERIFY_TLS,
    MODEL_DIR, ODDSLESS_MODEL_PATH, ODDSAWARE_MODEL_PATH,
    PLAN_DIR, DRY_RUN, LOG_VERBOSITY,
    RACEDAY_API_BASE, RACEDAY_API_KEY,
    BET_API_LOGIN, BET_API_BET, BET_API_LOGOUT,
    LOGIN_ID, LOGIN_PW,
    TARGET_PLACE, TARGET_RACE,
    ENSEMBLE_MODEL_SPECS,  # ← 追加：ここからモデル一覧を読む
)

from odds_policy import make_plan_for_day   # 発展版（推奨）
# from bet_policy import make_plan_for_day  # 既存

# ---- APIレスポンスの形を正規化（dict/listどちらでもOKにする） -----------------
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

# ---- API ------------------------------------------------------
def api_get_racecards(date_yyyymmdd: str) -> dict:
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Racecards", "id": date_yyyymmdd}
    res = requests.get(url, headers=headers, verify=VERIFY_TLS, timeout=15)
    res.raise_for_status()
    return _unwrap_api_json(res.json())

def api_get_odds(race_id_jjrr: str) -> dict:
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Odds", "id": race_id_jjrr}
    res = requests.get(url, headers=headers, verify=VERIFY_TLS, timeout=15)
    res.raise_for_status()
    return _unwrap_api_json(res.json())

def api_login(login_id: str, password: str) -> str:
    payload = {"login_id": login_id, "password": password}
    res = requests.post(BET_API_LOGIN, json=payload, timeout=15)
    if res.status_code == 200:
        print("ログイン成功")
        return res.json()["data"]["access_token"]
    print(f"ログイン失敗: {res.status_code} {res.text}")
    return ""

def api_bet(access_token: str, bet_data_list: list[dict]) -> dict:
    headers = {"Authorization": f"Bearer {access_token}"}
    payload = {"bet_data": bet_data_list}
    res = requests.post(BET_API_BET, headers=headers, json=payload, timeout=30)
    if res.status_code == 200:
        print("ベット成功")
        return res.json()
    print(f"ベット失敗: {res.status_code} {res.text}")
    return {}

def api_logout(access_token: str) -> None:
    headers = {"Authorization": f"Bearer {access_token}"}
    res = requests.post(BET_API_LOGOUT, headers=headers, timeout=15)
    print("ログアウト成功" if res.status_code == 200 else f"ログアウト失敗: {res.status_code}")

# ---- 列名マッピング（runtableの揺れ吸収） -----------------------------------
def normalize_runtable_columns(df: pd.DataFrame) -> pd.DataFrame:
    colmap = {}
    for cand in ["horse_num", "umaban", "馬番"]:
        if cand in df.columns:
            colmap[cand] = "horse_num"; break
    for cand in ["waku_num", "枠番"]:
        if cand in df.columns:
            colmap[cand] = "waku_num"; break
    for cand in ["race_id", "raceId", "レースID"]:
        if cand in df.columns:
            colmap[cand] = "race_id"; break
    for cand in ["place", "場名"]:
        if cand in df.columns:
            colmap[cand] = "place"; break
    for cand in ["race_num", "R", "レース番号"]:
        if cand in df.columns:
            colmap[cand] = "race_num"; break
    for cand in ["win_odds", "単勝オッズ", "odds_win"]:
        if cand in df.columns:
            colmap[cand] = "win_odds"; break

    df = df.rename(columns=colmap).copy()

    # --- race_id の正規化 ---
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
                return s[:4] + s[8:16]  # YYYY + place + kaiji + nichi + race
            return s
        df["race_id_odds"] = df["race_id_full"].apply(to_odds_id)
        df["race_id_bet"]  = df["race_id_full"].apply(to_bet_id)
        df["race_id"] = df["race_id_bet"]

    if "horse_num" not in df.columns:
        df["horse_num"] = (df.groupby("race_id").cumcount() + 1) if "race_id" in df.columns else (df.index + 1)
    if "waku_num" not in df.columns:
        df["waku_num"] = 1
    return df

# ---- モデル読み込み（存在するものだけ採用、oddsless/awareは互換フォールバック） ----
def load_models_for_ensemble() -> dict:
    """
    conf.ENSEMBLE_MODEL_SPECS = [(name, path, weight, priority), ...]
    戻り値: { name: {"bundle": joblib_loaded, "weight": float, "priority": int} }
    """
    models = {}
    for spec in ENSEMBLE_MODEL_SPECS:
        try:
            name, path, weight, priority = spec
        except ValueError:
            # 旧式 (name, path, weight) でも動かす
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

    # 何も無ければ旧 conf の2本をフォールバック（存在する方だけ）
    if not models:
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

# ---- メイン ----------------------------------------------------
def main():
    date_str = TARGET_DATE if TARGET_DATE else datetime.now().strftime("%Y%m%d")
    print(f"[run_vote] TARGET_DATE={date_str}")

    # 1) 当日データ取得
    rc = api_get_racecards(date_str)
    if "data" not in rc:
        print("Racecardsが空です。終了。")
        return
    timetable = pd.DataFrame(rc["data"].get("timetable", []))
    runtable  = pd.DataFrame(rc["data"].get("runtable",  []))
    if runtable.empty:
        print("runtableが空です。終了。")
        return

    # 2) 列名整形＆race_id変換
    runtable = normalize_runtable_columns(runtable)

    # 2.5) 開催地・レース番号フィルタ
    if TARGET_PLACE:
        runtable = runtable[runtable.get("place") == TARGET_PLACE]
    if TARGET_RACE:
        runtable = runtable[runtable.get("race_num").astype(str) == str(int(TARGET_RACE))]
    if runtable.empty:
        print("指定条件に該当するレースがありません。終了。")
        return

    # 3) 状況ログ
    n_races = runtable["race_id"].nunique() if "race_id" in runtable.columns else 0
    by_race = runtable.groupby("race_id")["horse_num"].count().reset_index(name="n_horses")
    print(f"[run_vote] races={n_races}, head:")
    print(by_race.head(10).to_string(index=False))

    # 4) モデル読み込み（存在するものを全部）
    models = load_models_for_ensemble()

    # 5) 計画作成（odds_policy / bet_policy どちらのシグネチャでも動くアダプタ）
    try:
        plan_df = make_plan_for_day(
            timetable=timetable,
            runtable=runtable,
            get_odds_func=api_get_odds,
            date_yyyymmdd=date_str,
            verbose=LOG_VERBOSITY,
            models=models,  # ← odds_policy想定
        )
    except TypeError:
        # 旧シグネチャ（bet_policy互換）
        oddsless  = models.get("oddsless", {}).get("bundle")  if models else None
        oddsaware = models.get("oddsaware", {}).get("bundle") if models else None
        plan_df = make_plan_for_day(
            timetable=timetable,
            runtable=runtable,
            oddsless=oddsless,
            oddsaware=oddsaware,
            get_odds_func=api_get_odds,
            date_yyyymmdd=date_str,
            verbose=LOG_VERBOSITY,
        )

    print(f"[run_vote] plan rows = {len(plan_df)}")
    if plan_df.empty:
        print("[warn] plan が空です。しきい値やデータ取得を確認してください。")

    # 6) プラン保存
    os.makedirs(PLAN_DIR, exist_ok=True)
    plan_path = os.path.join(PLAN_DIR, f"{date_str}_plan.json")
    plan_df.to_json(plan_path, orient="records", force_ascii=False)
    print(f"bet plan saved: {plan_path}")

    # 7) DRY_RUN
    if DRY_RUN:
        print("DRY_RUN=True のためAPI送信しません。")
        return

    # 8) 投票
    login_id = LOGIN_ID or os.getenv("KEIBA_LOGIN_ID", "")
    password = LOGIN_PW or os.getenv("KEIBA_PASSWORD", "")
    token = api_login(login_id, password)
    if not token:
        print("ログイン失敗のため、計画保存のみで終了。")
        return

    bet_data_list = [row["api_payload"] for _, row in plan_df.iterrows() if "api_payload" in row]
    if not bet_data_list:
        print("API送信用のpayloadが空です。")
        api_logout(token)
        return

    res = api_bet(token, bet_data_list)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if isinstance(res, dict) and "remaining_money" in res:
        print(f"remaining money: {res['remaining_money']}")
    api_logout(token)

if __name__ == "__main__":
    # TLS警告が気になる場合:
    # import urllib3; urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    main()