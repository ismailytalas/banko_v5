# -*- coding: utf-8 -*-
"""Farklı modelleme yaklaşımlarını AYNI test setinde karşılaştırır.

    python scripts/compare_models.py

Denenen yaklaşımlar:
  1. BINARY      : mevcut yöntem — "bu at kazanır mı" (0/1)
  2. LAMBDARANK  : yarış içi SIRALAMAYI doğrudan optimize eder (LightGBM ranker)
  3. DERECE      : atın yarış içi normalize edilmiş derecesini tahmin eder,
                   sonra sıralar. Kaybetme MİKTARINI da kullanır.
  4. PIYASA      : referans — sadece AGF

Neden derece tahmini farklı: "yarım boy farkla ikinci" ile "11 boy farkla
ikinci" ikili etikette aynı görünür. Derece bu bilgiyi korur.

DİKKAT: Derece yarış SONUCUDUR, özellik olarak KULLANILAMAZ — sadece hedef
olarak. Ayrıca yarış içinde standartlaştırılır, yoksa model mesafeyi öğrenip
yüksek R² üretir ama sıralama için işe yaramaz.
"""
import sys
import pathlib
import re

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.features import build_features, add_label_from_results  # noqa: E402
from tjk.history import add_history_features  # noqa: E402
from tjk.model import FEATURE_COLS, time_split, RACE_KEY  # noqa: E402

RAW = ROOT / "data" / "raw"
TIME_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")


def derece_saniye(s):
    """'1.26.88' -> 86.88 saniye"""
    if s is None or (isinstance(s, float) and pd.isna(s)):
        return np.nan
    m = TIME_RE.search(str(s))
    if not m:
        return np.nan
    return int(m.group(1)) * 60 + int(m.group(2)) + int(m.group(3)) / 100


def yaris_ici_z(df: pd.DataFrame, col: str) -> pd.Series:
    """Yarış içinde standartlaştır: mesafe/pist etkisini tamamen siler."""
    g = df.groupby(RACE_KEY, dropna=False)[col]
    return (df[col] - g.transform("mean")) / g.transform("std").replace(0, np.nan)


def olcum(test: pd.DataFrame, skor_col: str, ad: str,
          buyuk_iyi: bool = True) -> dict:
    d = test.dropna(subset=[skor_col]).copy()
    d = d.sort_values(skor_col, ascending=not buyuk_iyi)
    ilk = d.groupby(RACE_KEY, dropna=False).head(1)
    ilk2 = d.groupby(RACE_KEY, dropna=False).head(2)
    getiri = np.where(ilk["kazandi"] == 1, ilk["oran"], 0.0)
    return {
        "Yöntem": ad,
        "Top1 %": round(100 * ilk["kazandi"].mean(), 1),
        "Top2 %": round(100 * ilk2.groupby(RACE_KEY)["kazandi"].max().mean(), 1),
        "ROI %": round(100 * (getiri.mean() - 1), 1),
    }


def main():
    files = sorted(RAW.glob("results_*.parquet"))
    if not files:
        sys.exit("Veri yok.")
    print(f"{len(files)} dosya yükleniyor...")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    raw = add_label_from_results(raw)

    df = build_features(raw)
    df["kazandi"] = raw["kazandi"].values
    df["bitis"] = raw["bitis"].values
    df["derece_ham"] = raw["Derece"].map(derece_saniye) if "Derece" in raw else np.nan
    print("Geçmiş özellikleri üretiliyor...")
    df = add_history_features(df)

    # Ganyan ölçeği
    g = pd.to_numeric(df["gny"], errors="coerce")
    df["oran"] = g / 100.0 if g.median(skipna=True) > 50 else g

    feats = [c for c in FEATURE_COLS if c in df.columns]
    tr, te = time_split(df)
    tr, te = tr.copy(), te.copy()
    print(f"Eğitim {len(tr):,} satır | Test {len(te):,} satır "
          f"({te.groupby(RACE_KEY).ngroups:,} koşu)\n")

    sonuclar = []

    # --- 1) BINARY -------------------------------------------------------
    m1 = lgb.train(dict(objective="binary", metric="binary_logloss",
                        learning_rate=.05, num_leaves=31, min_data_in_leaf=50,
                        feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                        verbosity=-1),
                   lgb.Dataset(tr[feats], label=tr["kazandi"]), 800)
    te["s_binary"] = m1.predict(te[feats])
    sonuclar.append(olcum(te, "s_binary", "1. Binary (mevcut)"))

    # --- 2) LAMBDARANK ---------------------------------------------------
    # Yarış = "sorgu grubu". Etiket: iyi bitiş = yüksek alaka puanı.
    def hazirla_rank(d):
        d = d.dropna(subset=["bitis"]).sort_values(RACE_KEY).copy()
        # bitiş 1 -> 4 puan, 2 -> 3, 3 -> 2, 4+ -> 0
        d["alaka"] = np.select(
            [d["bitis"] == 1, d["bitis"] == 2, d["bitis"] == 3],
            [4, 3, 2], default=0).astype(int)
        gruplar = d.groupby(RACE_KEY, sort=False).size().values
        return d, gruplar

    tr_r, grup_tr = hazirla_rank(tr)
    m2 = lgb.train(dict(objective="lambdarank", metric="ndcg", ndcg_eval_at=[1, 2],
                        learning_rate=.05, num_leaves=31, min_data_in_leaf=50,
                        feature_fraction=.8, label_gain=[0, 1, 3, 7, 15],
                        verbosity=-1),
                   lgb.Dataset(tr_r[feats], label=tr_r["alaka"], group=grup_tr), 800)
    te["s_rank"] = m2.predict(te[feats])
    sonuclar.append(olcum(te, "s_rank", "2. LambdaRank (sıralama)"))

    # --- 3) DERECE (hız figürü) ------------------------------------------
    for d in (tr, te):
        d["derece_z"] = yaris_ici_z(d, "derece_ham")
    tr_d = tr.dropna(subset=["derece_z"])
    if len(tr_d) > 1000:
        m3 = lgb.train(dict(objective="regression", metric="l2",
                            learning_rate=.05, num_leaves=31, min_data_in_leaf=50,
                            feature_fraction=.8, verbosity=-1),
                       lgb.Dataset(tr_d[feats], label=tr_d["derece_z"]), 800)
        # düşük z = hızlı = iyi  -> ters çevir
        te["s_derece"] = -m3.predict(te[feats])
        sonuclar.append(olcum(te, "s_derece", "3. Derece tahmini"))

    # --- 4) ENSEMBLE ------------------------------------------------------
    def rank_pct(col):
        return te.groupby(RACE_KEY, dropna=False)[col].rank(pct=True)
    parcalar = [rank_pct("s_binary"), rank_pct("s_rank")]
    if "s_derece" in te:
        parcalar.append(rank_pct("s_derece"))
    te["s_ens"] = sum(parcalar) / len(parcalar)
    sonuclar.append(olcum(te, "s_ens", "4. Üçünün ortalaması"))

    # --- 5) PIYASA referansı ---------------------------------------------
    sonuclar.append(olcum(te, "agf", "5. AGF (referans)"))
    sonuclar.append(olcum(te, "oran", "6. Ganyan (referans)", buyuk_iyi=False))

    print("=== YÖNTEM KARŞILAŞTIRMASI (aynı test seti) ===")
    print(pd.DataFrame(sonuclar).to_string(index=False))

    # AGF ile ne kadar örtüşüyor?
    print("\n=== AGF'DEN BAĞIMSIZLIK ===")
    for col, ad in (("s_binary", "Binary"), ("s_rank", "LambdaRank"),
                    ("s_derece", "Derece"), ("s_ens", "Ensemble")):
        if col not in te:
            continue
        a = te.sort_values(col, ascending=False).groupby(RACE_KEY).head(1)
        b = te.sort_values("agf", ascending=False).groupby(RACE_KEY).head(1)
        birlesik = a[RACE_KEY + ["at"]].merge(
            b[RACE_KEY + ["at"]], on=RACE_KEY, suffixes=("_m", "_agf"))
        ayni = (birlesik["at_m"] == birlesik["at_agf"]).mean()
        print(f"  {ad:<12} AGF ile aynı seçim: %{100*ayni:.1f}")


if __name__ == "__main__":
    main()
