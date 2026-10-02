# -*- coding: utf-8 -*-
"""SÜRPRİZ AT ANALİZİ — halkın gözden kaçırdığı atları arar.

    python scripts/anomaly.py

Mantık:
Favoriler verimli fiyatlanır; para orada değildir. Asıl kazanç, AGF'nin düşük
gösterdiği ama bizim değişkenlerimizin güçlü dediği atlardadır.

Bu script üç şey yapar:
  1. SÜRPRİZ SIKLIĞI  : Halkın ilk 2 favorisi dışından kazanan ne kadar sık?
  2. AYIRT EDİCİ NELER: Sürpriz kazananlar, kazanamayan diğer düşük-AGF
                        atlardan hangi değişkenlerde ayrışıyor?
  3. SÜRPRİZ MODELİ   : Sadece düşük-AGF atlar üzerinde eğitilmiş özel model.
                        Bu modelin seçimleri kâr getiriyor mu?

NOT: Klasik "favori-sürpriz yanlılığı" literatürde iyi bilinir — bahisçiler
yüksek oranlı atlara fazla oynar, dolayısıyla sürprizler genelde PAHALIDIR.
Bu analiz o yanlılığı yenip yenemediğimizi dürüstçe ölçer.
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
    df = df.dropna(subset=["agf", "oran", "kazandi"])

    # Halkın sıralaması: AGF'ye göre koşu içi sıra (1 = en çok oynanan)
    df["agf_sira"] = df.groupby(RACE_KEY, dropna=False)["agf"].rank(
        ascending=False, method="min")
    df["halk_favorisi"] = (df["agf_sira"] <= 2).astype(int)

    # --- 1) SÜRPRİZ SIKLIĞI ----------------------------------------------
    print("\n=== 1. SÜRPRİZ SIKLIĞI ===")
    kazananlar = df[df["kazandi"] == 1]
    dagilim = kazananlar["agf_sira"].value_counts(normalize=True).sort_index()
    print("Kazanan atın AGF sırası:")
    for sira, oran in dagilim.head(8).items():
        etiket = "  (halk favorisi)" if sira <= 2 else ""
        print(f"  AGF {int(sira)}. sıra: %{100*oran:5.1f}{etiket}")
    surpriz_oran = (kazananlar["agf_sira"] > 2).mean()
    print(f"\n>>> Koşuların %{100*surpriz_oran:.1f}'ini halkın ilk 2 favorisi "
          f"DIŞINDAN bir at kazanıyor.")

    # Bu sürprizler ne ödüyor?
    surpriz_kazanan = kazananlar[kazananlar["agf_sira"] > 2]
    print(f">>> Sürpriz kazananların ortalama oranı: "
          f"{surpriz_kazanan['oran'].mean():.2f} "
          f"(medyan {surpriz_kazanan['oran'].median():.2f})")

    # --- 2) SÜRPRİZ KAZANANLAR NEDE AYRIŞIYOR? ---------------------------
    print("\n=== 2. SÜRPRİZ KAZANANLARI AYIRT EDEN DEĞİŞKENLER ===")
    print("(düşük-AGF atlar içinde: kazananlar vs kazanamayanlar)")
    dusuk = df[df["agf_sira"] > 2].copy()
    feats = [c for c in FEATURE_COLS if c in dusuk.columns]
    satirlar = []
    for c in feats:
        k = dusuk.loc[dusuk["kazandi"] == 1, c].dropna()
        n = dusuk.loc[dusuk["kazandi"] == 0, c].dropna()
        if len(k) < 100 or len(n) < 100:
            continue
        std = dusuk[c].std()
        if not std or np.isnan(std):
            continue
        etki = (k.mean() - n.mean()) / std        # Cohen's d benzeri
        satirlar.append({"Değişken": c, "Kazanan ort": round(k.mean(), 3),
                         "Diğer ort": round(n.mean(), 3),
                         "Etki (std)": round(etki, 3)})
    tab = pd.DataFrame(satirlar).reindex(
        pd.DataFrame(satirlar)["Etki (std)"].abs().sort_values(
            ascending=False).index)
    print(tab.head(15).to_string(index=False))
    print("\n(Etki: pozitif = sürpriz kazananlarda daha yüksek. "
          "|0.2| üstü anlamlı sayılır.)")

    # --- 3) SÜRPRİZ MODELİ -------------------------------------------------
    print("\n=== 3. SÜRPRİZ MODELİ (sadece düşük-AGF atlar) ===")
    tr, te = time_split(dusuk)
    if len(tr) < 5000:
        print("Yetersiz veri.")
        return
    m = lgb.train(dict(objective="binary", metric="binary_logloss",
                       learning_rate=.05, num_leaves=31, min_data_in_leaf=100,
                       feature_fraction=.8, bagging_fraction=.8, bagging_freq=1,
                       lambda_l2=2.0, verbosity=-1),
                  lgb.Dataset(tr[feats], label=tr["kazandi"]), 600)
    te = te.copy()
    te["surpriz_skor"] = m.predict(te[feats])

    print(f"Test: {len(te):,} düşük-AGF at, "
          f"{te.groupby(RACE_KEY).ngroups:,} koşu")

    # Skor dilimlerine göre kârlılık
    te["dilim"] = pd.qcut(te["surpriz_skor"], 10, labels=False, duplicates="drop")
    rapor = []
    for d, grp in te.groupby("dilim"):
        getiri = np.where(grp["kazandi"] == 1, grp["oran"], 0.0)
        rapor.append({"Dilim": int(d) + 1, "At": len(grp),
                      "Kazanma %": round(100 * grp["kazandi"].mean(), 1),
                      "Ort. oran": round(grp["oran"].mean(), 1),
                      "ROI %": round(100 * (getiri.mean() - 1), 1)})
    print("\nSürpriz skoruna göre 10 dilim (10 = model en çok beğendikleri):")
    print(pd.DataFrame(rapor).to_string(index=False))

    # En üst dilimde koşu başına 1 seçim yapsak
    en_iyi = te.sort_values("surpriz_skor", ascending=False).groupby(
        RACE_KEY, dropna=False).head(1)
    getiri = np.where(en_iyi["kazandi"] == 1, en_iyi["oran"], 0.0)
    print(f"\nHer koşuda EN İYİ sürpriz adayına 1 TL:")
    print(f"  isabet: %{100*en_iyi['kazandi'].mean():.1f} | "
          f"ort. oran: {en_iyi['oran'].mean():.2f} | "
          f"ROI: %{100*(getiri.mean()-1):.1f}")

    # Sadece güçlü sinyal olduğunda oyna
    esik = te["surpriz_skor"].quantile(0.98)
    sec = te[te["surpriz_skor"] >= esik]
    getiri2 = np.where(sec["kazandi"] == 1, sec["oran"], 0.0)
    print(f"\nSadece en güçlü %2'lik sinyaller ({len(sec)} bahis):")
    print(f"  isabet: %{100*sec['kazandi'].mean():.1f} | "
          f"ort. oran: {sec['oran'].mean():.2f} | "
          f"ROI: %{100*(getiri2.mean()-1):.1f}")
    print("\n>>> Pozitif ROI varsa gerçek kenar bulunmuş demektir.")
    print(">>> Negatifse: sürprizler pahalı, halkın yanlılığı yenilememiş.")


if __name__ == "__main__":
    main()
