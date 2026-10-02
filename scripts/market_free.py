# -*- coding: utf-8 -*-
"""PİYASASIZ TEMEL MODEL deneyi.

    python scripts/market_free.py

Sorun: Modele AGF ve ganyanı verince model piyasaya yakınsıyor (%78-87 aynı
seçim) ve diğer 60 değişken hiç konuşamıyor. Piyasayı yenmek için piyasadan
BAĞIMSIZ bir görüş gerekir.

Yöntem (iki aşamalı, klasik değer bahsi yaklaşımı):
  A) TAM MODEL      : piyasa dahil (mevcut durum, referans)
  B) TEMEL MODEL    : AGF ve ganyan YOK — sadece at/jokey/form/sınıf bilgisi
  C) KARŞILAŞTIRMA  : B'nin olasılıkları piyasadan sapıyorsa orası değerdir
  D) HARMAN         : B + piyasa birlikte

Beklenti: B'nin tek başına isabeti A'dan DÜŞÜK olacak — bu normal ve sorun
değil. Önemli olan B'nin piyasanın görmediği bir şey görüp görmediği; bunu
C bölümündeki değer analizi söyler.
"""
import sys
import pathlib

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.features import build_features, add_label_from_results  # noqa: E402
from tjk.history import add_history_features  # noqa: E402
from tjk.model import FEATURE_COLS, time_split, RACE_KEY  # noqa: E402

RAW = ROOT / "data" / "raw"

# Piyasa kaynaklı kolonlar — TEMEL MODELDEN ÇIKARILACAK
PIYASA_KOLONLARI = {
    "gny", "agf", "gny_implied",
    "gny_implied_rel", "gny_implied_rank",
    "agf_rel", "agf_rank",
    "sabah_gny", "agf_prg", "agf_sira_prg",
}

PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=.05,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1)


def kosu_ici_olasilik(df, skor_col, out_col):
    """Ham skorları koşu içinde 1'e toplanacak şekilde normalize eder."""
    d = df.copy()
    t = d[skor_col].clip(1e-9, 1 - 1e-9)
    d["_l"] = np.log(t / (1 - t))

    def _sm(s):
        e = np.exp(s - s.max())
        return e / e.sum()
    d[out_col] = d.groupby(RACE_KEY, dropna=False)["_l"].transform(_sm)
    return d.drop(columns=["_l"])


def olcum(te, col, ad, buyuk_iyi=True):
    d = te.dropna(subset=[col, "oran"])
    d = d.sort_values(col, ascending=not buyuk_iyi)
    ilk = d.groupby(RACE_KEY, dropna=False).head(1)
    ilk2 = d.groupby(RACE_KEY, dropna=False).head(2)
    getiri = np.where(ilk["kazandi"] == 1, ilk["oran"], 0.0)
    return {"Model": ad,
            "Top1 %": round(100 * ilk["kazandi"].mean(), 1),
            "Top2 %": round(100 * ilk2.groupby(RACE_KEY)["kazandi"].max().mean(), 1),
            "ROI %": round(100 * (getiri.mean() - 1), 1)}


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
    print("Geçmiş özellikleri üretiliyor...")
    df = add_history_features(df)

    g = pd.to_numeric(df["gny"], errors="coerce")
    df["oran"] = g / 100.0 if g.median(skipna=True) > 50 else g
    df = df.dropna(subset=["oran", "agf", "kazandi"])

    tam_feats = [c for c in FEATURE_COLS if c in df.columns]
    temel_feats = [c for c in tam_feats if c not in PIYASA_KOLONLARI]
    print(f"\nTam model  : {len(tam_feats)} özellik")
    print(f"Temel model: {len(temel_feats)} özellik "
          f"({len(tam_feats)-len(temel_feats)} piyasa kolonu çıkarıldı)")

    tr, te = time_split(df)
    tr, te = tr.copy(), te.copy()
    print(f"Eğitim {len(tr):,} | Test {len(te):,} "
          f"({te.groupby(RACE_KEY).ngroups:,} koşu)\n")

    # A) TAM MODEL
    mA = lgb.train(PARAMS, lgb.Dataset(tr[tam_feats], label=tr["kazandi"]), 800)
    te["sA"] = mA.predict(te[tam_feats])
    te = kosu_ici_olasilik(te, "sA", "pA")

    # B) TEMEL MODEL (piyasasız)
    mB = lgb.train(PARAMS, lgb.Dataset(tr[temel_feats], label=tr["kazandi"]), 800)
    te["sB"] = mB.predict(te[temel_feats])
    te = kosu_ici_olasilik(te, "sB", "pB")

    # Piyasa olasılığı (marj çıkarılmış)
    te["p_piyasa"] = 1.0 / te["oran"]
    te["p_piyasa"] = te["p_piyasa"] / te.groupby(RACE_KEY)["p_piyasa"].transform("sum")

    # D) HARMAN: temel model + piyasa (log-olasılık ortalaması)
    te["sD"] = np.sqrt(te["pB"] * te["p_piyasa"])
    te["pD"] = te["sD"] / te.groupby(RACE_KEY)["sD"].transform("sum")

    print("=== 1. MODEL KARŞILAŞTIRMASI ===")
    tablo = [olcum(te, "pA", "A) Tam model (piyasa dahil)"),
             olcum(te, "pB", "B) TEMEL model (piyasasız)"),
             olcum(te, "pD", "D) Temel + piyasa harmanı"),
             olcum(te, "agf", "   AGF (referans)"),
             olcum(te, "oran", "   Ganyan (referans)", buyuk_iyi=False)]
    print(pd.DataFrame(tablo).to_string(index=False))

    # Bağımsızlık
    print("\n=== 2. PİYASADAN BAĞIMSIZLIK ===")
    agf_ilk = te.sort_values("agf", ascending=False).groupby(RACE_KEY).head(1)
    for col, ad in (("pA", "Tam model"), ("pB", "TEMEL model"), ("pD", "Harman")):
        m = te.sort_values(col, ascending=False).groupby(RACE_KEY).head(1)
        bir = m[RACE_KEY + ["at"]].merge(agf_ilk[RACE_KEY + ["at"]], on=RACE_KEY,
                                         suffixes=("_m", "_a"))
        print(f"  {ad:<14} AGF ile aynı seçim: %{100*(bir['at_m']==bir['at_a']).mean():.1f}")

    # C) DEĞER ANALİZİ — asıl soru
    print("\n=== 3. DEĞER ANALİZİ (temel model vs piyasa) ===")
    print("Temel modelin olasılığı / piyasa olasılığı")
    te["deger"] = te["pB"] / te["p_piyasa"].replace(0, np.nan)
    te["kova"] = pd.cut(te["deger"], [0, .5, .8, 1.0, 1.25, 1.6, 2.5, 100])
    sat = []
    for k, grp in te.groupby("kova", observed=True):
        getiri = np.where(grp["kazandi"] == 1, grp["oran"], 0.0)
        sat.append({"Değer aralığı": str(k), "Bahis": len(grp),
                    "İsabet %": round(100 * grp["kazandi"].mean(), 1),
                    "Ort. oran": round(grp["oran"].mean(), 2),
                    "ROI %": round(100 * (getiri.mean() - 1), 1)})
    print(pd.DataFrame(sat).to_string(index=False))

    # Sadece güçlü değer sinyalleri + makul oran aralığı
    print("\n=== 4. SEÇİCİ STRATEJİLER ===")
    for esik, alt, ust in [(1.25, 1, 100), (1.6, 1, 100), (1.25, 2, 10),
                           (1.6, 2, 15), (2.5, 1, 100)]:
        sec = te[(te["deger"] >= esik) & (te["oran"] >= alt) & (te["oran"] <= ust)]
        if len(sec) < 100:
            continue
        getiri = np.where(sec["kazandi"] == 1, sec["oran"], 0.0)
        print(f"  değer>={esik:<4} oran {alt}-{ust}: {len(sec):6,} bahis | "
              f"isabet %{100*sec['kazandi'].mean():5.1f} | "
              f"ROI %{100*(getiri.mean()-1):+6.1f}")

    print("\n=== 5. TEMEL MODELDE EN ETKİLİ ÖZELLİKLER ===")
    imp = pd.Series(mB.feature_importance("gain"), index=temel_feats)
    print(imp.sort_values(ascending=False).head(12).round(0).to_string())


if __name__ == "__main__":
    main()
