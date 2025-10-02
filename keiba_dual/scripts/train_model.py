#!/usr/bin/env python
import argparse, os, joblib, pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--features", default="data/processed/features.csv")
    ap.add_argument("--out", default="models/win_model.pkl")
    args = ap.parse_args()
    df = pd.read_csv(args.features)
    X = df[["age","rank_last"]].fillna(0.0).values
    y = df["y_win"].astype(int).values
    Xtr, Xva, ytr, yva = train_test_split(X,y,test_size=0.2,random_state=42,stratify=y)
    clf = LogisticRegression(max_iter=1000).fit(Xtr,ytr)
    auc = roc_auc_score(yva, clf.predict_proba(Xva)[:,1]) if len(set(yva))>1 else 0.5
    print("Validation AUC:", round(auc,4))
    os.makedirs("models", exist_ok=True)
    joblib.dump(clf, args.out)
    print("Saved", args.out)

if __name__ == "__main__":
    main()
