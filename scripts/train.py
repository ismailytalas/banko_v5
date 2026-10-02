# -*- coding: utf-8 -*-
"""Modeli eğit: python scripts/train.py  (data/raw altındaki tüm parquet'leri kullanır)"""
import sys
import json
import pathlib

import pandas as pd

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))
from tjk.features import build_features, add_label_from_results
from tjk.history import add_history_features
from tjk.model import train, save

files = sorted(pathlib.Path("data/raw").glob("results_*.parquet"))
if not files:
    sys.exit("Önce scripts/collect_history.py ile veri indirin.")

raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
print(f"{len(raw)} ham satır yüklendi.")

raw = add_label_from_results(raw)
df = build_features(raw)
df["kazandi"] = raw["kazandi"].values
df["bitis"] = raw["bitis"].values

# Geçmiş-tabanlı özellikler (jokey/antrenör/sahip/at) — sızıntısız
print("Geçmiş özellikleri üretiliyor...")
df = add_history_features(df)

booster, metrics = train(df)
save(booster, metrics)
print(json.dumps(metrics, indent=2, ensure_ascii=False))
print("Model kaydedildi -> models/model.txt")
