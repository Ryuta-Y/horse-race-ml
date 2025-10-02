# keiba_dual_ready_full/scripts/api_client.py
# ===========================================
import os
import json
import requests
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime
from ..conf import (
    VERIFY_TLS, RACEDAY_API_BASE, RACEDAY_API_KEY,
    BET_API_LOGIN, BET_API_BET, BET_API_LOGOUT,
    LOGIN_ID, LOGIN_PW
)

def _raceday_get(data_type: str, id_value: str) -> Dict[str, Any]:
    """
    data_type: "Racecards" or "Odds"
    id_value : Racecards -> "YYYYMMDD"
               Odds      -> "YYYYMMDDJJRR"
    """
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": data_type, "id": id_value}
    r = requests.get(url, headers=headers, verify=VERIFY_TLS)
    r.raise_for_status()
    return r.json()

def get_racecards(yyyymmdd: str) -> Tuple[List[dict], List[dict]]:
    """
    Returns (timetable, runtable)
    """
    res = _raceday_get("Racecards", yyyymmdd)
    data = res["data"]
    return data["timetable"], data["runtable"]  # 資料の返却形式に準拠
    # 参照：Racecards取得例。 [oai_citation:6‡vertopal.com_Sample_API.txt](file-service://file-AL4asFd4BArrbhW19aX8hR)

def get_odds(yyyymmddjjrr: str) -> List[dict]:
    """
    Returns odds_rt list
    """
    res = _raceday_get("Odds", yyyymmddjjrr)
    return res["data"]["odds_rt"]
    # 参照：Odds取得例。 [oai_citation:7‡vertopal.com_Sample_API.txt](file-service://file-AL4asFd4BArrbhW19aX8hR)

def vote_login(login_id: Optional[str]=None, password: Optional[str]=None) -> str:
    login_id = login_id if login_id is not None else LOGIN_ID
    password = password if password is not None else LOGIN_PW
    headers = {"Content-Type": "application/json"}
    payload = {"login_id": login_id, "password": password}
    r = requests.post(BET_API_LOGIN, headers=headers, json=payload)
    r.raise_for_status()
    print("ログイン成功")
    return r.json()["data"]["access_token"]

def vote_bet(access_token: str, bet_data: List[dict]) -> dict:
    headers = {"Content-Type": "application/json", "Authorization": f"Bearer {access_token}"}
    body = {"bet_data": bet_data}
    r = requests.post(BET_API_BET, headers=headers, data=json.dumps(body))
    r.raise_for_status()
    print("ベット成功")
    return r.json()  # {"status":"OK","data":{"list_bet_race":[...],"success_count":...,"error_count":...}, "remaining_money":"..."}
    # 参照：vote.py の例と同様の戻り値。 [oai_citation:8‡vertopal.com_02_Running.txt](file-service://file-DDZS1SUcTUDguuB4mNVodW)

def vote_logout(access_token: str) -> None:
    headers = {"Authorization": f"Bearer {access_token}"}
    r = requests.post(BET_API_LOGOUT, headers=headers)
    r.raise_for_status()
    print("ログアウト成功")