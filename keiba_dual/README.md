#（当日オッズなし／あり両対応）— 即運用版

## 0) セットアップ
```bash
python -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
```

## 1) 前処理 → 学習
```bash
python scripts/make_dataset.py --records-glob "data/raw/record_data_*.csv"

python scripts/train_model.py   --features data/processed/features.csv   --out models/win_model.pkl
```

## 2-A) 当日オッズなし（OFFLINE）
```bash
python scripts/generate_bets.py   --mode offline   --model models/win_model.pkl   --export-dir data/api_out
# または投票IDが既知なら
python scripts/generate_bets.py   --mode offline   --model models/win_model.pkl   --race-id-offline 2025081701010101
```

## 2-B) 当日オッズあり（ONLINE）
```bash
python scripts/generate_bets.py   --mode online   --date 20250817 --place 札幌 --race 01   --model models/win_model.pkl   --export-dir data/api_out
```

## 3) 提出
```bash
python scripts/submit_bet.py --json data/api_out/<race_id>.json
```

### ヒント
- verify=False の警告は無視可（資料の通り“形だけ証明書”）。
- すべて賭け金 0 の場合は stake-cap, unit, min_p, kelly_frac を調整。
