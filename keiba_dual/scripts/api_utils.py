import os, json, requests
from typing import Optional
from dotenv import load_dotenv

load_dotenv()

RACEDAY_API_BASE = os.getenv("RACEDAY_API_BASE", "https://172.192.40.114").rstrip("/")
RACEDAY_API_KEY  = os.getenv("RACEDAY_API_KEY", "AI_Keiba_2025")

def get_racecards(date_yyyymmdd: str, verify: bool=False):
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Racecards", "id": date_yyyymmdd}
    r = requests.get(url, headers=headers, verify=verify)
    r.raise_for_status()
    return r.json()["data"]

def get_odds(race_id_yyyymmddjjrr: str, verify: bool=False):
    url = f"{RACEDAY_API_BASE}/data"
    headers = {"api-key": RACEDAY_API_KEY, "type_of_data": "Odds", "id": race_id_yyyymmddjjrr}
    r = requests.get(url, headers=headers, verify=verify)
    r.raise_for_status()
    return r.json()["data"]

BET_API_ENDPOINT = os.getenv("BET_API_ENDPOINT","").strip()
BET_API_TOKEN    = os.getenv("BET_API_TOKEN","").strip()

LOGIN_API_ENDPOINT  = os.getenv("LOGIN_API_ENDPOINT","").strip()
LOGOUT_API_ENDPOINT = os.getenv("LOGOUT_API_ENDPOINT","").strip()
LOGIN_ID            = os.getenv("LOGIN_ID","").strip()
LOGIN_PASSWORD      = os.getenv("LOGIN_PASSWORD","").strip()

def _login_get_token() -> str:
    if not LOGIN_API_ENDPOINT or not LOGIN_ID or not LOGIN_PASSWORD:
        raise RuntimeError("LOGIN_* env not set; cannot login. Provide BET_API_TOKEN instead or set login envs.")
    headers = {"Content-Type":"application/json"}
    payload = {"login_id": LOGIN_ID, "password": LOGIN_PASSWORD}
    r = requests.post(LOGIN_API_ENDPOINT, headers=headers, data=json.dumps(payload))
    if r.status_code != 200:
        raise RuntimeError(f"login failed {r.status_code}: {r.text}")
    data = r.json()
    token = data.get("data",{}).get("access_token")
    if not token:
        raise RuntimeError(f"login did not return access_token: {data}")
    return token

def _logout(token: str):
    if not LOGOUT_API_ENDPOINT:
        return
    try:
        requests.post(LOGOUT_API_ENDPOINT, headers={"Authorization": f"Bearer {token}"})
    except Exception:
        pass

def submit_vote_single(bet_payload: dict) -> dict:
    if not BET_API_ENDPOINT:
        raise RuntimeError("BET_API_ENDPOINT is not set")
    headers = {"Content-Type": "application/json"}

    token = BET_API_TOKEN if BET_API_TOKEN else _login_get_token()
    headers["Authorization"] = f"Bearer {token}"

    body = {"bet_data": [bet_payload]}
    r = requests.post(BET_API_ENDPOINT, headers=headers, data=json.dumps(body))
    if r.status_code != 200:
        if not BET_API_TOKEN and token:
            _logout(token)
        raise RuntimeError(f"bet failed {r.status_code}: {r.text}")
    res = r.json()

    if not BET_API_TOKEN and token:
        _logout(token)
    return res
