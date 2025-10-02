# scripts/watch_vote.py
# 使い方:
#   (venv) $ python -m scripts.watch_vote
#
# 概要:
# - conf.py の TARGET_DATE / TARGET_PLACE / TARGET_RACE を対象に、
#   タイムテーブルの「発走5分前(厳密には4分10秒前)」で
#   計画作成(make_plan_for_day)→投票API送信を行う常駐スクリプト。
# - この版は「対象の全レース」を、発走順に
#     ① 5分前まで待機 → ② oddsaware が利用可なら投票
#   を繰り返し、**全対象が終わったら終了**します。
# - ONLY_IF_ODDSAWARE=1 のときは、oddsaware.pkl が存在する場合のみ投票します。
# - ログを詳細に出力。残ポイントも表示。
#
# 注意:
# - トークン有効期限は5分なので、各レースごとに直前でログイン→投票→ログアウトします。
# - 金額は bet_policy 側で 100円単位に丸められている想定。念のためここでも最終チェックします。

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
    BASE_BET_UNIT,
)

# ===== 追加オプション（このファイル内だけで完結） =====
# oddsaware.pkl がある時だけ投票する（既定=1=有効）
ONLY_IF_ODDSAWARE = bool(int(os.getenv("ONLY_IF_ODDSAWARE", "1")))

LEAD_SEC = 4 * 60 + 10   # 発走の 4分10秒前に取得/投票
POLL_INTERVAL = 5        # 待機時の進捗ログ間隔(秒)

# -------------------- API --------------------
def api_get_racecards(date_yyyymmdd: str) -> dict:
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Racecards", "id": date_yyyymmdd}
    res = requests.get(url, headers=headers, verify=VERIFY_TLS, timeout=20)
    res.raise_for_status()
    return res.json()

def api_get_odds(race_id_jjrr: str) -> dict:
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Odds", "id": race_id_jjrr}
    res = requests.get(url, headers=headers, verify=VERIFY_TLS, timeout=20)
    res.raise_for_status()
    return res.json()

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

# -------------------- util --------------------
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
    if "race_id" in df.columns:
        df["race_id_full"] = df["race_id"].astype(str)
        def to_odds_id(rid: str) -> str:
            s = str(rid)
            if len(s) >= 16:
                return s[:8] + s[8:10] + s[14:16]
            return s
        def to_bet_id(rid: str) -> str:
            s = str(rid)
            if len(s) >= 16:
                return s[:4] + s[8:16]
            return s
        df["race_id_odds"] = df["race_id_full"].apply(to_odds_id)
        df["race_id_bet"]  = df["race_id_full"].apply(to_bet_id)
        df["race_id"] = df["race_id_bet"]

    if "horse_num" not in df.columns:
        df["horse_num"] = (df.groupby("race_id").cumcount() + 1) if "race_id" in df.columns else (df.index + 1)
    if "waku_num" not in df.columns:
        df["waku_num"] = 1
    return df

def _round_bet_amounts(plan_df: pd.DataFrame) -> pd.DataFrame:
    df = plan_df.copy()
    if "amount" in df.columns:
        df["amount"] = (df["amount"] // BASE_BET_UNIT) * BASE_BET_UNIT
        df = df[df["amount"] > 0]
    return df

def _filter_target_race(runtable: pd.DataFrame) -> pd.DataFrame:
    df = runtable.copy()
    if TARGET_PLACE:
        df = df[df.get("place") == TARGET_PLACE]
    if TARGET_RACE:
        df = df[df.get("race_num").astype(str) == str(int(TARGET_RACE))]
    return df

def _list_target_slots(timetable: pd.DataFrame) -> list[tuple[str, int, str]]:
    """(place, race_num, start_time 'HH:MM') のリスト。TARGET_* で絞り、時刻昇順。"""
    if timetable.empty: return []
    df = timetable.copy()
    if TARGET_PLACE:
        df = df[df.get("place") == TARGET_PLACE]
    if TARGET_RACE:
        df = df[df.get("race_num").astype(str) == str(int(TARGET_RACE))]
    df = df.sort_values("start_time")
    return [(r.place, int(r.race_num), r.start_time) for _, r in df.iterrows()]

def _compute_wait_seconds(start_hhmm: str) -> int:
    # 今日のカレンダー上の start_hhmm をターゲットとして扱う
    now = datetime.now()
    try:
        target = datetime.strptime(start_hhmm, "%H:%M").replace(year=now.year, month=now.month, day=now.day)
    except Exception:
        return -1
    return int((target - datetime.now()).total_seconds() - LEAD_SEC)

def _log_countdown(wait_sec: int, start_hhmm: str):
    left = max(wait_sec, 0)
    print(f"[{start_hhmm}] 5分前トリガーまで {left} 秒待機...")
    last = left
    while left > 0:
        step = min(POLL_INTERVAL, left)
        time.sleep(step)
        left = max(0, left - step)
        if left == 0 or left // 20 != last // 20:
            print(f"  ... 残り {left} 秒")
        last = left

# bet_policy からプラン作成を呼ぶ
from bet_policy import make_plan_for_day

def _build_and_bet_for(place: str, race_num: int, date_str: str) -> None:
    """対象1レースのプラン作成→投票（必要条件を満たさなければ投票スキップ）。"""
    print(f"[plan] 対象: {place}{race_num}R / 日付={date_str}")
    # 最新の当日データ（直前に取得）
    rc = api_get_racecards(date_str)
    if "data" not in rc:
        print("[plan] Racecards空: 中止")
        return
    timetable = pd.DataFrame(rc["data"].get("timetable", []))
    runtable  = pd.DataFrame(rc["data"].get("runtable",  []))
    if runtable.empty:
        print("[plan] runtable空: 中止"); return

    runtable = normalize_runtable_columns(runtable)
    runtable = _filter_target_race(runtable)
    if runtable.empty:
        print("[plan] 指定条件に一致する出走表なし: 中止"); return

    # モデル状態（このタイミングで再確認：朝→昼に oddsaware が出現するケースに対応）
    oddsless_exists  = os.path.exists(ODDSLESS_MODEL_PATH)
    oddsaware_exists = os.path.exists(ODDSAWARE_MODEL_PATH)
    if ONLY_IF_ODDSAWARE and not oddsaware_exists:
        print("[plan] ONLY_IF_ODDSAWARE=True だが oddsaware.pkl が見つからないため、このレースは投票スキップ。")
        return

    oddsless  = joblib.load(ODDSLESS_MODEL_PATH)  if oddsless_exists  else None
    oddsaware = joblib.load(ODDSAWARE_MODEL_PATH) if oddsaware_exists else None
    mode = ("oddsaware+oddsless" if (oddsaware and oddsless) else
            "oddsaware" if oddsaware else
            "oddsless"  if oddsless  else "uniform")
    print(f"[plan] 使用モデル: {mode}")

    # プラン作成（内部で get_odds_func=api_get_odds を使い直近オッズ反映）
    plan_df = make_plan_for_day(
        timetable=timetable, runtable=runtable,
        oddsless=oddsless, oddsaware=oddsaware,
        get_odds_func=api_get_odds, date_yyyymmdd=date_str,
        verbose=LOG_VERBOSITY,
    )
    plan_df = _round_bet_amounts(plan_df)
    print(f"[plan] 生成件数: {len(plan_df)}")
    if plan_df.empty:
        print("[plan] プランなし（しきい値/データ要確認）")
        return

    # 保存（監査用）
    os.makedirs(PLAN_DIR, exist_ok=True)
    fname = f"{date_str}_{place}_{race_num:02d}_plan.json"
    fpath = os.path.join(PLAN_DIR, fname)
    plan_df.to_json(fpath, orient="records", force_ascii=False)
    print(f"[plan] 保存: {fpath}")

    if DRY_RUN:
        print("[bet] DRY_RUN=True のため送信しません")
        return

    # 投票（直前ログイン：トークン寿命5分）
    token = api_login(LOGIN_ID or os.getenv("KEIBA_LOGIN_ID", ""),
                      LOGIN_PW  or os.getenv("KEIBA_PASSWORD", ""))
    if not token:
        print("[bet] トークン取得失敗: 中止")
        return

    bet_data_list = [row["api_payload"] for _, row in plan_df.iterrows() if "api_payload" in row]
    if not bet_data_list:
        print("[bet] API payloadなし: 中止")
        api_logout(token)
        return

    print(f"[bet] 送信件数: {len(bet_data_list)} / 例: {bet_data_list[0] if bet_data_list else 'N/A'}")
    res = api_bet(token, bet_data_list)
    print(json.dumps(res, ensure_ascii=False, indent=2))
    if isinstance(res, dict) and "remaining_money" in res:
        print(f"[bet] remaining money: {res['remaining_money']}")
    api_logout(token)

def main():
    date_str = TARGET_DATE if TARGET_DATE else datetime.now().strftime("%Y%m%d")
    print(f"[watch_vote] TARGET_DATE={date_str}, TARGET_PLACE={TARGET_PLACE}, TARGET_RACE={TARGET_RACE}, ONLY_IF_ODDSAWARE={ONLY_IF_ODDSAWARE}")

    # 当日データからタイムテーブルを取得し、対象スロットを決定
    try:
        rc = api_get_racecards(date_str)
    except Exception as e:
        print(f"[watch] 当日データ取得失敗: {e}")
        return

    timetable = pd.DataFrame(rc.get("data", {}).get("timetable", []))
    if timetable.empty:
        print("[watch] timetable空: 終了")
        return

    slots = _list_target_slots(timetable)
    if not slots:
        print("[watch] 指定条件に一致するレースなし: 終了")
        return

    # すべての対象レースについて、発走順に処理
    for idx, (place, race_num, start_hhmm) in enumerate(slots, start=1):
        wait_sec = _compute_wait_seconds(start_hhmm)
        if wait_sec < 0:
            print(f"[watch] 時刻解析失敗({start_hhmm}): スキップ")
            continue
        if wait_sec <= 0:
            print(f"[watch] [{start_hhmm}] 5分前取得時刻を過ぎています: スキップ")
            continue

        print(f"[watch] ({idx}/{len(slots)}) 次の対象: {place}{race_num}R (発走 {start_hhmm})")
        _log_countdown(wait_sec, start_hhmm)

        try:
            _build_and_bet_for(place, race_num, date_str)
        except KeyboardInterrupt:
            print("\n[watch] 中断されました。終了。")
            return
        except Exception as e:
            print(f"[watch] 例外によりこのレースをスキップ: {e}")

    print("[watch] すべての対象レース処理が完了しました。")

if __name__ == "__main__":
    main()