# -*- coding: utf-8 -*-
"""
LightGBM tabanlı kazanma olasılığı modeli.

Yaklaşım:
  - Her at-satırı için binary hedef: kazandi (1/0)
  - LightGBM binary sınıflandırıcı ham skor üretir
  - Skorlar yarış (tarih, sehir, kosu_no) içinde softmax ile normalize edilir
    -> her koşuda olasılıklar 1'e toplanır (kazanan tek at olduğu için doğru çerçeve)

Değerlendirme: zaman bazlı bölme (son %20 gün test), log-loss + isabet oranı.
"""
from __future__ import annotations

import json
import pathlib

import numpy as np
import pandas as pd
import lightgbm as lgb

from .features import FEATURE_COLS as BASE_FEATURES
from .history import HISTORY_FEATURES
from .raceclass import RACE_CLASS_FEATURES

FEATURE_COLS = BASE_FEATURES + HISTORY_FEATURES + RACE_CLASS_FEATURES

RACE_KEY = ["tarih", "sehir", "kosu_no"]


def softmax_by_race(df: pd.DataFrame, score_col: str = "score",
                    out_col: str = "p_win") -> pd.DataFrame:
    d = df.copy()
    def _sm(s):
        e = np.exp(s - s.max())
        return e / e.sum()
    d[out_col] = d.groupby(RACE_KEY, dropna=False)[score_col].transform(_sm)
    return d


def time_split(df: pd.DataFrame, test_frac: float = 0.2):
    days = sorted(df["tarih"].unique())
    cut = days[int(len(days) * (1 - test_frac))]
    return df[df["tarih"] < cut], df[df["tarih"] >= cut]


DEFAULT_PARAMS = dict(
    objective="binary", metric="binary_logloss",
    learning_rate=0.05, num_leaves=31, min_data_in_leaf=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1,
    lambda_l2=1.0, verbosity=-1,
)


def train(df: pd.DataFrame, params: dict | None = None,
          num_rounds: int = 2000, early_stopping: int = 100):
    """df: build_features + add_label_from_results uygulanmış eğitim verisi."""
    feats = [c for c in FEATURE_COLS if c in df.columns]
    df = df.dropna(subset=["kazandi"])
    tr, va = time_split(df)

    dtr = lgb.Dataset(tr[feats], label=tr["kazandi"])
    dva = lgb.Dataset(va[feats], label=va["kazandi"], reference=dtr)
    booster = lgb.train(
        {**DEFAULT_PARAMS, **(params or {})}, dtr, num_rounds,
        valid_sets=[dva],
        callbacks=[lgb.early_stopping(early_stopping, verbose=False)],
    )

    # Değerlendirme: yarış-içi normalize olasılıkla kazanan isabeti
    va = va.copy()
    va["score"] = booster.predict(va[feats])
    va = softmax_by_race(va)
    top1 = (va.sort_values("p_win", ascending=False)
              .groupby(RACE_KEY).head(1))
    metrics = {
        "n_races_test": int(top1.shape[0]),
        "top1_hit_rate": float(top1["kazandi"].mean()),
        "top2_hit_rate": float(
            va.sort_values("p_win", ascending=False)
              .groupby(RACE_KEY).head(2)
              .groupby(RACE_KEY)["kazandi"].max().mean()),
        "features": feats,
    }
    return booster, metrics


def save(booster: lgb.Booster, metrics: dict, path: str = "models/model"):
    p = pathlib.Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    booster.save_model(str(p) + ".txt")
    (p.parent / "metrics.json").write_text(json.dumps(metrics, indent=2))


def load(path: str = "models/model") -> tuple[lgb.Booster, list[str]]:
    booster = lgb.Booster(model_file=str(path) + ".txt")
    metrics = json.loads((pathlib.Path(path).parent / "metrics.json").read_text())
    return booster, metrics["features"]


def predict_program(program_feats: pd.DataFrame, booster: lgb.Booster,
                    feats: list[str]) -> pd.DataFrame:
    """Günlük program özellik tablosuna p_win kolonu ekler."""
    d = program_feats.copy()
    for c in feats:
        if c not in d.columns:
            d[c] = np.nan
    d["score"] = booster.predict(d[feats])
    d = softmax_by_race(d)
    return d


def market_baseline(program_feats: pd.DataFrame) -> pd.DataFrame:
    """Model yokken kullanılabilecek temel: sabah ganyanı + AGF karışımı.

    Elinizde henüz eğitilmiş model yoksa bu fonksiyon makul bir p_win üretir;
    Streamlit uygulaması modeli bulamazsa otomatik buna düşer.
    """
    d = program_feats.copy()
    imp = d["gny_implied"].fillna(0)
    agf = d["agf"].fillna(0)
    d["score_raw"] = 0.6 * imp + 0.4 * agf
    grp = d.groupby(RACE_KEY, dropna=False)["score_raw"]
    d["p_win"] = d["score_raw"] / grp.transform("sum").replace(0, np.nan)
    d["p_win"] = d["p_win"].fillna(1.0 / grp.transform("count"))
    return d
