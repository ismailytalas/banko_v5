# -*- coding: utf-8 -*-
"""HİPODROM ANALİZİ — sürpriz sıklığı ve hipodrom bazlı model denemesi.

    python scripts/track_analysis.py

İki soru:
  1. Hangi hipodromlarda sürpriz daha sık ve daha pahalı çıkıyor?
  2. Her hipodroma AYRI model eğitmek, tek genel modelden iyi mi?

2. soruda dikkat: ayrı model = daha az veri. Elazığ'a özel model sadece
Elazığ verisiyle eğitilir; hipodroma özgü örüntüyü yakalar ama örneklem
küçüldüğü için gürültüye de açıktır. Bu ödünleşmeyi ölçüyoruz.
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
# market_free'den import etmek yerine burada tanımlıyoruz (klasör bağımsız çalışsın)
PIYASA_KOLONLARI = {"gny", "agf", "gny_implied", "gny_implied_rel",
                    "gny_implied_rank", "agf_rel", "agf_rank",
                    "sabah_gny", "agf_prg", "agf_sira_prg"}

PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=.05,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1)


def kosu_ici_olasilik(df, skor_col, out_col):
    d = df.copy()
    t = d[skor_col].clip(1e-9, 1 - 1e-9)
    d["_l"] = np.log(t / (1 - t))
    d[out_col] = d.groupby(RACE_KEY, dropna=False)["_l"].transform(
        lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum())
    return d.drop(columns=["_l"])

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
    df = df.dropna(subset=["oran", "agf", "kazandi"])
    df["agf_sira"] = df.groupby(RACE_KEY, dropna=False)["agf"].rank(
        ascending=False, method="min")

    # --- 1) Hipodrom bazlı sürpriz istatistikleri -------------------------
    print("\n=== 1. HİPODROM BAZINDA SÜRPRİZ ===")
    kaz = df[df["kazandi"] == 1]
    sat = []
    for sehir, grp in kaz.groupby("sehir"):
        surpriz = grp["agf_sira"] > 2
        sat.append({
            "Hipodrom": sehir,
            "Koşu": len(grp),
            "Sürpriz %": round(100 * surpriz.mean(), 1),
            "Fav. kazanma %": round(100 * (grp["agf_sira"] == 1).mean(), 1),
            "Sürpriz ort. oran": round(grp.loc[surpriz, "oran"].mean(), 1),
            "Ort. at sayısı": round(grp["alan_sayisi"].mean(), 1),
        })
    tab = pd.DataFrame(sat).sort_values("Sürpriz %", ascending=False)
    print(tab.to_string(index=False))
    print("\n(Sürpriz = kazanan at, AGF'ye göre halkın ilk 2 favorisi değil)")

    # --- 2) Hipodrom bazlı model vs genel model ---------------------------
    print("\n=== 2. AYRI MODEL vs GENEL MODEL ===")
    feats = [c for c in FEATURE_COLS
             if c in df.columns and c not in PIYASA_KOLONLARI]
    tr_all, te_all = time_split(df)

    print("Genel model eğitiliyor...")
    m_genel = lgb.train(PARAMS, lgb.Dataset(tr_all[feats],
                                            label=tr_all["kazandi"]), 800)

    sonuc = []
    for sehir in sorted(df["sehir"].unique()):
        tr = tr_all[tr_all["sehir"] == sehir]
        te = te_all[te_all["sehir"] == sehir].copy()
        if len(tr) < 8000 or te.groupby(RACE_KEY).ngroups < 150:
            continue

        # Genel modelin bu hipodromdaki performansı
        te["s_genel"] = m_genel.predict(te[feats])
        te = kosu_ici_olasilik(te, "s_genel", "p_genel")

        # Hipodroma özel model
        m_ozel = lgb.train(PARAMS, lgb.Dataset(tr[feats], label=tr["kazandi"]), 600)
        te["s_ozel"] = m_ozel.predict(te[feats])
        te = kosu_ici_olasilik(te, "s_ozel", "p_ozel")

        satir = {"Hipodrom": sehir, "Eğitim": len(tr),
                 "Test koşu": te.groupby(RACE_KEY).ngroups}
        for col, ad in (("p_genel", "Genel"), ("p_ozel", "Özel")):
            ilk = te.sort_values(col, ascending=False).groupby(RACE_KEY).head(1)
            getiri = np.where(ilk["kazandi"] == 1, ilk["oran"], 0.0)
            satir[f"{ad} Top1%"] = round(100 * ilk["kazandi"].mean(), 1)
            satir[f"{ad} ROI%"] = round(100 * (getiri.mean() - 1), 1)
        satir["Fark(Top1)"] = round(satir["Özel Top1%"] - satir["Genel Top1%"], 1)
        sonuc.append(satir)

    if sonuc:
        s = pd.DataFrame(sonuc)
        print(s.to_string(index=False))
        ort = s["Fark(Top1)"].mean()
        print(f"\n>>> Özel modellerin ortalama farkı: {ort:+.2f} puan")
        if ort > 0.5:
            print(">>> Hipodrom bazlı model işe yarıyor.")
        elif ort < -0.5:
            print(">>> Ayrı model ZARARLI — veri bölünmesi gürültü yaratıyor.")
        else:
            print(">>> Anlamlı fark yok; genel model yeterli. "
                  "Hipodrom zaten bir özellik olarak modelde mevcut.")


if __name__ == "__main__":
    main()
