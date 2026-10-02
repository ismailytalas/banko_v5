# -*- coding: utf-8 -*-
"""
Koşu şartı ayrıştırma — atın kalite seviyesini belirleyen kritik bilgi.

TJK koşu başlığı örnekleri:
    "ŞARTLI 5 , 3 ve Yukarı İngilizler, 55.50 kg, 60 kg, 1500 Kum"
    "Handikap 16/DHÖW /H1/Y2 , 4 ve Yukarı Araplar, 1600 Çim"
    "Maiden , 2 Yaşlı İngilizler, 57 kg, 1600 Çim"
    "MEHMETÇİK KOŞUSU KV-8/DHT , 4 Yaşlı Araplar, 58 kg, 1400 Çim"
    "SATIŞ 2/Dişi , 3 ve Yukarı İngilizler, 1900 Çim"

Çıkarılanlar:
    kosu_turu   : Handikap / Şartlı / Maiden / Satış / KV / Grup / Diğer
    sinif_no    : tür yanındaki numara (Şartlı 5 -> 5, Handikap 16 -> 16)
    cins        : İngiliz / Arap
    min_yas     : yaş şartının alt sınırı
    yas_acik    : "ve Yukarı" ise 1 (üst sınır yok)
    disi_mi     : sadece dişilere açıksa 1
    isimli_kosu : "GAZİ KOŞUSU" gibi isimli koşu ise 1
    grup_no     : G1/G2/G3 varsa derece

ÖNEMLİ TASARIM NOTU — sınıf sıralaması:
Şartlı 5 mi Şartlı 3 mü daha kaliteli sorusunun cevabını KODA GÖMMÜYORUZ.
Bunun yerine kalite ölçüsünü veriden ampirik olarak çıkarıyoruz: her sınıfın
ikramiyesi (varsa) ve o sınıfta koşan atların ortalama handikap puanı.
Böylece yanlış bir varsayım modeli yanıltmaz.
"""
from __future__ import annotations

import re

import numpy as np
import pandas as pd

TUR_DESENLERI = [
    ("Handikap", re.compile(r"\bHand[iı]kap\s*(\d+)?", re.IGNORECASE)),
    ("Şartlı",   re.compile(r"\bŞARTLI\s*(\d+)?", re.IGNORECASE)),
    ("Maiden",   re.compile(r"\bMaiden\b()", re.IGNORECASE)),
    ("Satış",    re.compile(r"\bSATI[ŞS]\s*(\d+)?", re.IGNORECASE)),
    ("KV",       re.compile(r"\bKV\s*-?\s*(\d+)?", re.IGNORECASE)),
    ("Deneme",   re.compile(r"\bDeneme\s*(\d+)?", re.IGNORECASE)),
]

GRUP_RE = re.compile(r"\b(?:GRUP|Gr\.?|G)\s*([123])\b")
CINS_ARAP_RE = re.compile(r"Arap", re.IGNORECASE)
CINS_ING_RE = re.compile(r"İngiliz|Ingiliz", re.IGNORECASE)
YAS_RE = re.compile(r"(\d+)\s*(ve\s*Yukar[ıi]|Ya[şs]l[ıi])", re.IGNORECASE)
DISI_RE = re.compile(r"\bDi[şs]i\b|\bKısrak\b", re.IGNORECASE)
ISIMLI_RE = re.compile(r"([A-ZÇĞİÖŞÜ][A-ZÇĞİÖŞÜ\s\.']{3,})\s*KO[ŞS]USU")
# Handikap alt kodları: H1, H2, Y2, DHÖ, DHÖW, DHT ...
KOD_RE = re.compile(r"/\s*(H\d|Y\d|DH[ÖO]W?|DHT|Dişi)", re.IGNORECASE)
IKRAMIYE_RE = re.compile(r"Ikramiye\s*:?\s*1\.\)\s*([\d\.]+)", re.IGNORECASE)


def parse_race_class(text) -> dict:
    """Koşu başlığı metnini yapılandırılmış alanlara çevirir."""
    out = {
        "kosu_turu": None, "sinif_no": np.nan, "cins": None,
        "min_yas": np.nan, "yas_acik": 0, "disi_mi": 0,
        "isimli_kosu": 0, "grup_no": np.nan, "ikramiye_1": np.nan,
        "handikap_kodu": None,
    }
    if text is None or (isinstance(text, float) and pd.isna(text)):
        return out
    t = str(text)

    # Koşu numarası/saat kısmını at ("1. Koşu 14.00")
    t = re.sub(r"^\s*\d+\.\s*Ko[şs]u\s*\d{1,2}[.:]\d{2}", "", t).strip()

    # Tür ve sınıf numarası
    for ad, rx in TUR_DESENLERI:
        m = rx.search(t)
        if m:
            out["kosu_turu"] = ad
            if m.lastindex and m.group(1):
                out["sinif_no"] = float(m.group(1))
            break

    # Grup koşusu (G1/G2/G3) — en yüksek kalite
    g = GRUP_RE.search(t)
    if g:
        out["grup_no"] = float(g.group(1))
        if out["kosu_turu"] is None:
            out["kosu_turu"] = "Grup"

    # İsimli koşu
    if ISIMLI_RE.search(t):
        out["isimli_kosu"] = 1

    # Cins
    if CINS_ARAP_RE.search(t):
        out["cins"] = "Arap"
    elif CINS_ING_RE.search(t):
        out["cins"] = "İngiliz"

    # Yaş şartı
    y = YAS_RE.search(t)
    if y:
        out["min_yas"] = float(y.group(1))
        out["yas_acik"] = 1 if "yukar" in y.group(2).lower() else 0

    # Dişi kısıtlaması
    if DISI_RE.search(t):
        out["disi_mi"] = 1

    # Handikap alt kodları
    kodlar = KOD_RE.findall(t)
    if kodlar:
        out["handikap_kodu"] = "/".join(sorted(set(k.upper() for k in kodlar)))

    # İkramiye (sınıf kalitesinin en objektif ölçüsü)
    ik = IKRAMIYE_RE.search(t)
    if ik:
        try:
            out["ikramiye_1"] = float(ik.group(1).replace(".", ""))
        except ValueError:
            pass

    return out


def add_race_class_features(df: pd.DataFrame,
                            text_col: str = "kosu_cinsi") -> pd.DataFrame:
    """Koşu şartı alanlarını + ampirik sınıf kalite endeksini ekler."""
    d = df.copy()
    if text_col not in d.columns:
        return d

    parsed = pd.DataFrame(list(d[text_col].map(parse_race_class)),
                          index=d.index)
    for c in parsed.columns:
        d[c] = parsed[c]

    # Kategorik alanları modele uygun sayısala çevir
    d["arap_mi"] = (d["cins"] == "Arap").astype(int)
    d["maiden_mi"] = (d["kosu_turu"] == "Maiden").astype(int)
    d["handikap_mi"] = (d["kosu_turu"] == "Handikap").astype(int)
    d["satis_mi"] = (d["kosu_turu"] == "Satış").astype(int)
    d["sartli_mi"] = (d["kosu_turu"] == "Şartlı").astype(int)
    d["log_ikramiye"] = np.log1p(d["ikramiye_1"])

    # --- Ampirik sınıf kalite endeksi -----------------------------------
    # Sıralamayı varsaymak yerine veriden öğreniyoruz: bir sınıfta koşan
    # atların ortalama handikap puanı o sınıfın kalitesini gösterir.
    if "hp" in d.columns:
        d["sinif_kod"] = (d["kosu_turu"].fillna("?") + "_" +
                          d["sinif_no"].fillna(-1).astype(int).astype(str) + "_" +
                          d["cins"].fillna("?"))
        endeks = d.groupby("sinif_kod")["hp"].transform("mean")
        d["sinif_kalite"] = endeks
    return d


# ---------------------------------------------------------------------------
# Enflasyon düzeltmesi
# ---------------------------------------------------------------------------
def add_relative_prize(df: pd.DataFrame, window_days: int = 90) -> pd.DataFrame:
    """İkramiyeyi enflasyondan arındırır.

    SORUN: 2021'de 100.000 TL'lik bir koşu, 2026'daki 100.000 TL'lik koşudan
    çok daha yüksek sınıftır. Ham TL değeri sınıfı değil, takvimi ölçer.

    ÇÖZÜM: Her koşunun ikramiyesini, o tarihe yakın dönemde (varsayılan 90 gün)
    koşulan TÜM yarışların medyan ikramiyesine bölüyoruz. Sonuç "bu koşu
    dönemin ortalama koşusunun kaç katı değerinde" anlamına gelir ve
    zaman içinde karşılaştırılabilir.

    Bu yaklaşım TÜFE verisi gerektirmez ve TJK'nın tarife güncellemelerini
    (enflasyondan bağımsız sıçramalar) da otomatik olarak emer.

    Sızıntı riski YOK: ikramiye yarıştan önce ilan edilir, sonuç değildir.
    """
    d = df.copy()
    if "ikramiye_1" not in d.columns:
        return d

    d["_t"] = pd.to_datetime(d["tarih"], errors="coerce")

    # Koşu bazına indir (aynı koşudaki her at aynı ikramiyeyi taşır)
    kosu = (d.dropna(subset=["ikramiye_1"])
              .groupby(["_t", "sehir", "kosu_no"], dropna=False)["ikramiye_1"]
              .first().reset_index()
              .sort_values("_t"))
    if kosu.empty:
        d["ikramiye_rel"] = np.nan
        d["log_ikramiye_rel"] = np.nan
        return d.drop(columns=["_t"])

    # Kayan pencerede medyan ikramiye = dönemin "normal" koşusu
    seri = kosu.set_index("_t")["ikramiye_1"]
    medyan = (seri.rolling(f"{window_days}D", min_periods=5).median()
                  .rename("donem_medyan"))
    kosu = kosu.join(medyan.reset_index(drop=True).rename("donem_medyan"))
    # Pencerenin başındaki boşluklar için geriye doğru doldur
    kosu["donem_medyan"] = kosu["donem_medyan"].bfill()

    d = d.merge(kosu[["_t", "sehir", "kosu_no", "donem_medyan"]],
                on=["_t", "sehir", "kosu_no"], how="left")

    d["ikramiye_rel"] = d["ikramiye_1"] / d["donem_medyan"].replace(0, np.nan)
    d["log_ikramiye_rel"] = np.log(d["ikramiye_rel"].clip(lower=0.01))
    return d.drop(columns=["_t"])


def add_class_move_features(df: pd.DataFrame,
                            horse_col: str = "at") -> pd.DataFrame:
    """Atın bir önceki koşusuna göre sınıf değişimi (SIZINTISIZ).

    Sınıf yükseltmek/düşürmek güçlü bir sinyaldir: antrenör atı daha zayıf
    rakiplere karşı koşturuyorsa kazanma şansı artar.
    """
    d = df.copy()
    if horse_col not in d.columns:
        return d
    d = d.sort_values("tarih")
    g = d.groupby(horse_col, dropna=False)

    # DİKKAT: ikramiye değişimi için ham log_ikramiye KULLANILMAZ. At 6 ay ara
    # verdiyse aradaki enflasyon "sınıf yükseltti" yanılsaması yaratır.
    # Enflasyondan arındırılmış log_ikramiye_rel doğru ölçüdür.
    for kaynak, hedef in (("sinif_kalite", "sinif_degisim"),
                          ("log_ikramiye_rel", "ikramiye_degisim"),
                          ("mesafe", "mesafe_degisim")):
        if kaynak in d.columns:
            onceki = g[kaynak].shift(1)          # bir önceki yarış
            d[hedef] = d[kaynak] - onceki
            d[f"onceki_{kaynak}"] = onceki

    # Pist değişimi: kum -> çim geçişi
    if "pist" in d.columns:
        onceki_pist = g["pist"].shift(1)
        d["pist_degisti"] = (d["pist"].astype(str).str.lower() !=
                             onceki_pist.astype(str).str.lower()).astype(int)
        d.loc[onceki_pist.isna(), "pist_degisti"] = np.nan
    return d


RACE_CLASS_FEATURES = [
    "sinif_no", "min_yas", "yas_acik", "disi_mi", "isimli_kosu", "grup_no",
    "arap_mi", "maiden_mi", "handikap_mi", "satis_mi", "sartli_mi",
    "log_ikramiye_rel", "ikramiye_rel", "sinif_kalite",
    "sinif_degisim", "ikramiye_degisim", "mesafe_degisim", "pist_degisti",
    "onceki_sinif_kalite", "onceki_log_ikramiye_rel",
]
