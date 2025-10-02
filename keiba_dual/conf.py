# keiba_dual_ready_full/conf.py
# ============================
# 実行は keiba_dual_ready_full 直下を想定（例: python -m run_vote）

# ----------------------------------------------------------------------
# 実行日設定
# ----------------------------------------------------------------------
# None ならローカルJST当日で実行。固定したい場合は "YYYYMMDD" を指定。
#TARGET_DATE: str | None = None
TARGET_DATE = "20250921"

# ----------------------------------------------------------------------
# レースフィルタ（任意）
# ----------------------------------------------------------------------
TARGET_PLACE: str | None = "中山"
TARGET_RACE: int | None = 12

# ----------------------------------------------------------------------
# 予算・配分・しきい値
# ----------------------------------------------------------------------
PER_RACE_BUDGET_YEN: int = 30000
BASE_BET_UNIT: int = 100

# 単勝候補の最低勝率・最低期待値
THRESH_WIN_PROB_MIN: float = 0.15
THRESH_WIN_EV_MIN: float   = 0.05
# 複勝の最低ライン（win確率由来の目安）
THRESH_PLACE_BY_WINP: float = 0.19

# 投票・ブースト関連
BOTH_AGREE_BOOST: float = 1.30           # コア2モデル一致時のブースト
ENSEMBLE_VOTE_TOPK: int = 4             # 各モデルのTOP-K馬を“票”とする
ENSEMBLE_VOTE_BONUS_ALPHA: float = 0.15  # 票によるブースト係数（0〜0.3程度）
ENSEMBLE_PRIORITY_BONUS_ALPHA: float = 0.30  # 同票数のタイは優先度合計でブースト

# ----------------------------------------------------------------------
# モデル入出力
# ----------------------------------------------------------------------
MODEL_DIR: str = "models"

# 既存との後方互換（oddsless/oddsaware）
ODDSLESS_MODEL_PATH: str  = f"{MODEL_DIR}/lr_oddsless_cal.pkl"
ODDSAWARE_MODEL_PATH: str = f"{MODEL_DIR}/lr_oddsaware_cal.pkl"

# アンサンブルに使うモデル一覧：
#   (name, path, weight, priority)
# weight: アンサンブル平均の重み
# priority: 同票タイ時の優先度（大きいほど強い）
ENSEMBLE_MODEL_SPECS: list[tuple[str, str, float, int]] = [
    ("oddsaware_lr",   f"{MODEL_DIR}/lr_oddsaware_cal.pkl", 0.50, 900),
    ("oddsaware_lgbm", f"{MODEL_DIR}/lgbm_oddsaware.pkl",   0.80, 120),
    ("oddsless_lgbm",  f"{MODEL_DIR}/lgbm_oddsless.pkl",    0.40,  70),
    ("oddsaware_nn",   f"{MODEL_DIR}/nn_oddsaware.pkl",     0.70, 105),
    ("oddsless_nn",    f"{MODEL_DIR}/nn_oddsless.pkl",      0.50,  92),
]

# “一致ブースト”のコアモデル名（上の name を指定）
CORE_AGREE_PAIRS: list[tuple[str, str]] = [
    ("oddsaware_lr", "oddsless_lr"),
    ("oddsaware_lgbm", "oddsless_lgbm"),
]

# ----------------------------------------------------------------------
# 当日データAPI（レースカード/オッズ）
# ----------------------------------------------------------------------
RACEDAY_API_BASE: str = "https://172.192.40.114"
RACEDAY_API_KEY:  str = "AI_Keiba_2025"
VERIFY_TLS: bool  = False
HTTP_TIMEOUT_SEC: int = 20
RETRY_TIMES: int = 10
RETRY_SLEEP_SEC: float = 10

# ----------------------------------------------------------------------
# 投票API
# ----------------------------------------------------------------------
BET_API_LOGIN:  str = "https://masters.netkeiba.com/ai2025_student/api/login"
BET_API_BET:    str = "https://masters.netkeiba.com/ai2025_student/api/bet"
BET_API_LOGOUT: str = "https://masters.netkeiba.com/ai2025_student/api/logout"

# 認証情報（環境変数で上書きを強く推奨）
import os
#LOGIN_ID = os.getenv("KEIBA_LOGIN_ID", "")
#LOGIN_PW = os.getenv("KEIBA_PASSWORD", "")

LOGIN_ID = "ryuta.yamamoto@icloud.com"
LOGIN_PW = "Mottei0608"

# ----------------------------------------------------------------------
# 計画ファイル
# ----------------------------------------------------------------------
PLAN_DIR: str = "plans"
PLAN_BACKUP_GLOB: str = f"{PLAN_DIR}/*_plan.json"

# ----------------------------------------------------------------------
# 実行モード/ログ
# ----------------------------------------------------------------------
DRY_RUN: bool = False
LOG_VERBOSITY: int = 1

# 学習レポート出力先（学習スクリプトで使用）
REPORT_DIR: str = "reports"

# ----------------------------------------------------------------------
# レース別予算の上書き（race_num → 総予算[円]）
# 空のままなら上書きなし。例: {3: 3000, 5: 5000, 9: 8000, 11: 10000}
# 使い方: ここに必要なレース番号と予算を入れると、そのレースだけ PER_RACE_BUDGET_YEN を上書きします。
# ----------------------------------------------------------------------
ROUND_BUDGET_OVERRIDES: dict[int, int] = {3: 14000, 5: 24000, 9: 50000, 11: 300000, 4: 3000, 7: 10000, 8: 10000, 12: 280000}