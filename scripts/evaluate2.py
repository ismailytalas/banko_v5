# -*- coding: utf-8 -*-
"""Modeli PİYASA ile karşılaştırır ve kârlılığı ölçer.

Kullanım:
    python scripts/evaluate.py

Cevap aradığımız sorular:
  1. Model piyasa favorisinden (en düşük ganyan) daha iyi mi?
  2. AGF favorisinden daha iyi mi?
  3. Düz bahis oynansa uzun vadede kâr mı zarar mı? (ROI)
  4. Ürettiği olasılıklar kalibre mi? (%20 dediğinde gerçekten %20 mi kazanıyor)
  5. İkili tahminleri ne kadar tutuyor?

NOT: Isabet oranı yüksek olması kâr anlamına GELMEZ. Favoriler sık kazanır
ama düşük öder. Asıl ölçüt ROI'dir.
"""
import sys
import pathlib

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.features import build_features, add_label_from_results  # noqa: E402
from tjk.history import add_history_features  # noqa: E402
from tjk.model import load, softmax_by_race, time_split, RACE_KEY  # noqa: E402
from tjk.bets import harville_top2  # noqa: E402

RAW = ROOT / "data" / "raw"


def decimal_odds(gny: pd.Series) -> pd.Series:
    """TJK ganyanı 445.0 gibi gelir = 4.45 TL. Ölçeği otomatik tespit eder."""
    g = pd.to_numeric(gny, errors="coerce")
    if g.median(skipna=True) > 50:
        g = g / 100.0
    return g


def top_pick(df: pd.DataFrame, col: str, ascending: bool = False) -> pd.DataFrame:
    """Her koşuda verilen kolona göre en iyi atı seçer."""
    return (df.sort_values(col, ascending=ascending)
              .groupby(RACE_KEY, dropna=False).head(1))


def rapor(ad: str, sec: pd.DataFrame) -> dict:
    n = len(sec)
    isabet = sec["kazandi"].mean()
    # Düz bahis: her koşuya 1 TL. Kazanırsa oran kadar geri döner.
    getiri = np.where(sec["kazandi"] == 1, sec["oran"], 0.0)
    roi = getiri.mean() - 1.0
    return {"strateji": ad, "kosu": n, "isabet": isabet,
            "ort_oran": sec.loc[sec["kazandi"] == 1, "oran"].mean(),
            "roi": roi}


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

    booster, feats = load(str(ROOT / "models" / "model"))
    for c in feats:
        if c not in df.columns:
            df[c] = np.nan

    # Eğitimdekiyle AYNI zaman bölmesi — test seti modeli hiç görmedi
    _, test = time_split(df)
    test = test.copy()
    test["score"] = booster.predict(test[feats])
    test = softmax_by_race(test)
    test["oran"] = decimal_odds(test["gny"])
    test = test.dropna(subset=["oran", "kazandi"])

    # Sonuç sayfasında "N" (program numarası) yok; at kimliği için sırayla
    # no -> start_no -> at ismi kullanılır.
    if "no" not in test.columns or test["no"].isna().all():
        if "start_no" in test.columns and test["start_no"].notna().any():
            test["no"] = test["start_no"]
        else:
            test["no"] = test["at"]

    print(f"\nTest dönemi: {test['tarih'].min()} → {test['tarih'].max()}")
    print(f"Koşu sayısı: {test.groupby(RACE_KEY).ngroups:,}\n")

    # --- 1) Strateji karşılaştırması -------------------------------------
    stratejiler = [
        ("MODEL (en yüksek p_win)", top_pick(test, "p_win")),
        ("PİYASA (en düşük ganyan)", top_pick(test, "oran", ascending=True)),
        ("AGF favorisi", top_pick(test, "agf")),
        ("HP favorisi", top_pick(test, "hp")),
    ]
    sonuc = pd.DataFrame([rapor(ad, s) for ad, s in stratejiler])
    sonuc["isabet"] = (100 * sonuc["isabet"]).round(1)
    sonuc["roi"] = (100 * sonuc["roi"]).round(1)
    sonuc["ort_oran"] = sonuc["ort_oran"].round(2)
    sonuc.columns = ["Strateji", "Koşu", "İsabet %", "Ort. Kazanan Oran", "ROI %"]
    print("=== 1. STRATEJİ KARŞILAŞTIRMASI ===")
    print(sonuc.to_string(index=False))
    print("\n(ROI: -%X = her 100 TL'de X TL kayıp. TJK kesintisi ~%50 olduğu için")
    print(" -%20 civarı bile piyasadan iyi sayılır; pozitif ROI çok zordur.)")

    # --- 2) Model piyasadan farklı mı? -----------------------------------
    m = top_pick(test, "p_win")[RACE_KEY + ["no"]].rename(columns={"no": "model_no"})
    p = top_pick(test, "oran", ascending=True)[RACE_KEY + ["no"]].rename(
        columns={"no": "piyasa_no"})
    birlesik = m.merge(p, on=RACE_KEY)
    ayni = (birlesik["model_no"] == birlesik["piyasa_no"]).mean()
    print(f"\n=== 2. MODEL PİYASADAN FARKLI MI? ===")
    print(f"Model ve piyasa aynı atı seçme oranı: %{100*ayni:.1f}")
    if ayni > 0.85:
        print("!! Model neredeyse piyasayı taklit ediyor — ek özellikler katkı vermiyor.")
    else:
        print("Model piyasadan bağımsız seçimler yapıyor.")

    # --- 3) Kalibrasyon ---------------------------------------------------
    print("\n=== 3. KALİBRASYON (tahmin vs gerçek) ===")
    test["kova"] = pd.cut(test["p_win"], [0, .05, .1, .15, .2, .3, .5, 1.0])
    kal = test.groupby("kova", observed=True).agg(
        n=("kazandi", "size"), tahmin=("p_win", "mean"), gercek=("kazandi", "mean"))
    kal["tahmin"] = (100 * kal["tahmin"]).round(1)
    kal["gercek"] = (100 * kal["gercek"]).round(1)
    print(kal.to_string())

    # --- 4) İkili başarısı -------------------------------------------------
    print("\n=== 4. İKİLİ TAHMİNİ ===")
    ilk2_gercek = test[test["bitis"] <= 2].groupby(RACE_KEY)["no"].apply(
        lambda s: frozenset(s))
    tut = 0, 0
    isabet_sayisi = 0
    toplam = 0
    for anahtar, grp in test.groupby(RACE_KEY):
        if anahtar not in ilk2_gercek.index or len(ilk2_gercek[anahtar]) != 2:
            continue
        g = grp.dropna(subset=["p_win"]).set_index("no")["p_win"]
        if len(g) < 2:
            continue
        en_iyi = harville_top2(g).iloc[0]
        toplam += 1
        if frozenset([en_iyi["at1"], en_iyi["at2"]]) == ilk2_gercek[anahtar]:
            isabet_sayisi += 1
    if toplam:
        print(f"En olası ikili kombinasyonu tutma oranı: "
              f"%{100*isabet_sayisi/toplam:.1f}  ({isabet_sayisi}/{toplam} koşu)")

    deger_analizi(test)


def deger_analizi(test: pd.DataFrame):
    """Model piyasadan SAPTIĞI yerlerde haklı mı?

    Değer bahsi mantığı: modelin verdiği olasılık, ganyanın ima ettiği
    olasılıktan yüksekse o at 'ucuz' demektir. Kâr ancak buradan gelir;
    favoriyi herkesle birlikte oynamaktan değil.
    """
    d = test.copy()
    d["piyasa_p"] = 1.0 / d["oran"]
    # Piyasa olasılıklarını koşu içinde 1'e normalize et (marjı çıkar)
    d["piyasa_p"] = d["piyasa_p"] / d.groupby(RACE_KEY)["piyasa_p"].transform("sum")
    d["deger"] = d["p_win"] / d["piyasa_p"]        # 1'den büyük = model ucuz diyor

    print("\n=== 5. DEĞER BAHSİ ANALİZİ ===")
    print("(model olasılığı / piyasa olasılığı oranına göre gruplar)")
    d["deger_kova"] = pd.cut(d["deger"], [0, .8, 1.0, 1.2, 1.5, 2.0, 100])
    tab = d.groupby("deger_kova", observed=True).agg(
        bahis=("kazandi", "size"), isabet=("kazandi", "mean"),
        ort_oran=("oran", "mean"))
    tab["roi"] = [100 * (np.where(g["kazandi"] == 1, g["oran"], 0).mean() - 1)
                  for _, g in d.groupby("deger_kova", observed=True)]
    tab["isabet"] = (100 * tab["isabet"]).round(1)
    tab["ort_oran"] = tab["ort_oran"].round(2)
    tab["roi"] = tab["roi"].round(1)
    tab.columns = ["Bahis", "İsabet %", "Ort. Oran", "ROI %"]
    print(tab.to_string())
    print("\nPozitif ROI'li bir grup varsa gerçek kenar orada olabilir.")


if __name__ == "__main__":
    main()
