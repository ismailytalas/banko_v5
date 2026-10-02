# -*- coding: utf-8 -*-
"""
Geçmişe dayalı ("prior") özellikler — SIZINTISIZ.

Temel kural: bir yarış satırı için hesaplanan her istatistik, YALNIZCA o yarıştan
ÖNCEKİ GÜNLERDE koşulmuş yarışlara dayanır. Aynı günün sonuçları asla kullanılmaz.
Aksi halde model "geleceği bilerek" öğrenir ve gerçek hayatta çöker.

Üretilen özellik grupları:
  1) Kişi/kurum: jokey, antrenör, at sahibi  -> geçmiş kazanma & tabela oranı
  2) At: genel form, pist uyumu, mesafe uyumu, hipodrom uyumu
  3) Dinlenme: son yarıştan bu yana geçen gün
  4) Kombinasyon: jokey x antrenör uyumu

Küçük örneklem sorunu Bayesçi düzleştirme ile çözülür:
    oran = (kazanma + alpha * genel_ortalama) / (yarış_sayisi + alpha)
Yani 3 yarışta 2 kazanan bir jokey %67'ye değil, genel ortalamaya yakın çıkar.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _prior_counts(df: pd.DataFrame, keys: list[str], target: str) -> pd.DataFrame:
    """Verilen anahtar(lar) için ÖNCEKİ GÜNLERE ait toplam ve adet döndürür.

    Gün bazında toplayıp kaydırdığımız için aynı gün içindeki yarışlar
    birbirinin bilgisini kullanamaz.
    """
    daily = (df.groupby(keys + ["tarih"], dropna=False)[target]
               .agg(["sum", "count"]).reset_index()
               .sort_values("tarih"))
    g = daily.groupby(keys, dropna=False)
    # cumsum sonra shift(1): o güne kadar olan, o gün HARİÇ birikim
    daily["prior_sum"] = g["sum"].cumsum() - daily["sum"]
    daily["prior_n"] = g["count"].cumsum() - daily["count"]
    return daily[keys + ["tarih", "prior_sum", "prior_n"]]


def add_prior_rate(df: pd.DataFrame, keys: list[str], prefix: str,
                   target: str = "kazandi", alpha: float = 20.0,
                   add_count: bool = True) -> pd.DataFrame:
    """df'e '<prefix>_rate' (düzleştirilmiş geçmiş oran) ve '<prefix>_n' ekler."""
    if target not in df.columns or any(k not in df.columns for k in keys):
        return df
    d = df.copy()
    prior = _prior_counts(d, keys, target)
    d = d.merge(prior, on=keys + ["tarih"], how="left")

    global_mean = d[target].mean()
    d[f"{prefix}_rate"] = ((d["prior_sum"] + alpha * global_mean) /
                           (d["prior_n"] + alpha))
    # Hiç geçmişi olmayanlar genel ortalamaya oturur (yukarıdaki formül bunu sağlar)
    if add_count:
        d[f"{prefix}_n"] = d["prior_n"].fillna(0)
    return d.drop(columns=["prior_sum", "prior_n"])


def add_days_since_last_run(df: pd.DataFrame, horse_col: str = "at") -> pd.DataFrame:
    """Atın son yarışından bu yana geçen gün sayısı (dinlenme süresi)."""
    d = df.copy()
    if horse_col not in d.columns:
        return d
    d["_t"] = pd.to_datetime(d["tarih"], errors="coerce")
    d = d.sort_values("_t")
    prev = d.groupby(horse_col, dropna=False)["_t"].shift(1)
    d["dinlenme_gun"] = (d["_t"] - prev).dt.days
    d["ilk_yaris_mi"] = prev.isna().astype(int)
    return d.drop(columns=["_t"])


def _distance_band(m) -> str:
    """Mesafeyi kabaca sınıflandırır (sprint / mil / uzun)."""
    if pd.isna(m):
        return "bilinmiyor"
    m = float(m)
    if m <= 1300:
        return "kisa"
    if m <= 1700:
        return "orta"
    return "uzun"


def add_history_features(df: pd.DataFrame, alpha: float = 20.0) -> pd.DataFrame:
    """Tüm geçmiş-tabanlı özellikleri ekler.

    Girdi: build_features + add_label_from_results uygulanmış, 'kazandi' ve
    'tarih' kolonları olan tam veri (eğitim seti veya eğitim+bugünün programı).
    """
    d = df.copy()
    d = d.sort_values("tarih").reset_index(drop=True)

    if "kazandi" not in d.columns:
        # Tahmin edilecek günün programında etiket yoktur; geçmiş veriyle
        # birleştirilmeden çağrılırsa istatistik üretilemez.
        return d

    if "ilk3" not in d.columns and "bitis" in d.columns:
        d["ilk3"] = (d["bitis"] <= 3).astype(int)

    d["mesafe_bandi"] = d.get("mesafe", np.nan).map(_distance_band)
    d["pist_adi"] = d.get("pist", "").astype(str).str.lower()

    # --- 1) Kişi/kurum istatistikleri -------------------------------------
    for col, pfx in (("Jokey", "jokey"), ("Antrenör", "antrenor"),
                     ("Sahip", "sahip")):
        if col in d.columns:
            d = add_prior_rate(d, [col], pfx, "kazandi", alpha)
            if "ilk3" in d.columns:
                d = add_prior_rate(d, [col], f"{pfx}_ilk3", "ilk3", alpha,
                                   add_count=False)

    # Jokey x Antrenör uyumu (belirli ahırlarla çalışan jokeyler daha başarılı)
    if "Jokey" in d.columns and "Antrenör" in d.columns:
        d = add_prior_rate(d, ["Jokey", "Antrenör"], "jok_ant", "kazandi",
                           alpha=10)

    # Jokeyin bu hipodromdaki geçmişi
    if "Jokey" in d.columns and "sehir" in d.columns:
        d = add_prior_rate(d, ["Jokey", "sehir"], "jokey_sehir", "kazandi",
                           alpha=15, add_count=False)

    # --- 2) At istatistikleri ---------------------------------------------
    if "at" in d.columns:
        d = add_prior_rate(d, ["at"], "at_genel", "kazandi", alpha=5)
        if "ilk3" in d.columns:
            d = add_prior_rate(d, ["at"], "at_ilk3", "ilk3", alpha=5,
                               add_count=False)
        # Pist uyumu: kum atı mı çim atı mı?
        d = add_prior_rate(d, ["at", "pist_adi"], "at_pist", "kazandi",
                           alpha=4, add_count=False)
        # Mesafe uyumu
        d = add_prior_rate(d, ["at", "mesafe_bandi"], "at_mesafe", "kazandi",
                           alpha=4, add_count=False)
        # Hipodrom uyumu (senin dediğin gibi atlar saha değiştiriyor)
        if "sehir" in d.columns:
            d = add_prior_rate(d, ["at", "sehir"], "at_sehir", "kazandi",
                               alpha=4, add_count=False)

    # --- 2b) SOY (pedigri) istatistikleri ---------------------------------
    # Neden önemli: ilk kez koşan atın kendi geçmişi YOKTUR; modelin elinde
    # sadece soy bilgisi vardır. Ayrıca babadan gelen pist/mesafe yatkınlığı
    # tecrübeli atlarda da açıklayıcıdır.
    # alpha yüksek tutuldu: çoğu babanın az sayıda tayı var, ham oran gürültülü.
    if "baba" in d.columns:
        d = add_prior_rate(d, ["baba"], "baba", "kazandi", alpha=25)
        if "ilk3" in d.columns:
            d = add_prior_rate(d, ["baba"], "baba_ilk3", "ilk3", alpha=25,
                               add_count=False)
        # Babanın bu pist tipindeki tayları (kum atı mı çim atı mı üretiyor?)
        d = add_prior_rate(d, ["baba", "pist_adi"], "baba_pist", "kazandi",
                           alpha=20, add_count=False)
        # Babanın bu mesafe bandındaki tayları (sprinter mı stayer mı?)
        d = add_prior_rate(d, ["baba", "mesafe_bandi"], "baba_mesafe",
                           "kazandi", alpha=20, add_count=False)

    if "anne_babasi" in d.columns:
        d = add_prior_rate(d, ["anne_babasi"], "annebaba", "kazandi", alpha=25,
                           add_count=False)
    if "anne" in d.columns:
        # Anne bazında az veri olur (bir kısrağın az tayı olur) -> alpha yüksek
        d = add_prior_rate(d, ["anne"], "anne", "kazandi", alpha=8,
                           add_count=False)

    # --- 3) Dinlenme -------------------------------------------------------
    d = add_days_since_last_run(d)

    # --- 4) Koşu sınıfı ve sınıf değişimi ---------------------------------
    from .raceclass import (add_race_class_features, add_relative_prize,
                            add_class_move_features)
    d = add_race_class_features(d)
    d = add_relative_prize(d)      # enflasyon düzeltmesi (değişimden ÖNCE)
    d = add_class_move_features(d)

    # Atın geçmişte koştuğu en yüksek sınıf ve o sınıftaki başarısı
    if "at" in d.columns and "sinif_kalite" in d.columns:
        gg = d.sort_values("tarih").groupby("at", dropna=False)["sinif_kalite"]
        d["at_max_sinif"] = gg.transform(lambda x: x.shift(1).cummax())
        d["at_ort_sinif"] = gg.transform(lambda x: x.shift(1).expanding().mean())

    return d


# Model özellik listesine eklenecek yeni kolonlar
HISTORY_FEATURES = [
    "jokey_rate", "jokey_n", "jokey_ilk3_rate",
    "antrenor_rate", "antrenor_n", "antrenor_ilk3_rate",
    "sahip_rate", "sahip_n", "sahip_ilk3_rate",
    "jok_ant_rate", "jokey_sehir_rate",
    "at_genel_rate", "at_genel_n", "at_ilk3_rate",
    "at_pist_rate", "at_mesafe_rate", "at_sehir_rate",
    "dinlenme_gun", "ilk_yaris_mi", "at_max_sinif", "at_ort_sinif",
    # soy (pedigri)
    "baba_rate", "baba_n", "baba_ilk3_rate", "baba_pist_rate",
    "baba_mesafe_rate", "annebaba_rate", "anne_rate",
]
