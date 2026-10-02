# -*- coding: utf-8 -*-
"""
Özellik mühendisliği: TJK program/sonuç satırlarını model girdisine çevirir.

Girdi DataFrame'de beklenen kolonlar (scraper üretir, isimler TJK sayfasındaki gibi):
  N, At İsmi, Yaş, Sıklet, Jokey, Sahip, Antrenör, St, HP, Son 6 Y., KGS, s20,
  En İyi D., Gny, AGF  + meta: tarih, sehir, kosu_no, mesafe, pist, kosu_cinsi

Sonuç verisinde ek olarak bitiş sırası (sonuç sayfasında ilk kolon) bulunur.
"""
from __future__ import annotations

import re
import numpy as np
import pandas as pd

SON6_DIGIT_RE = re.compile(r"\d")
WEIGHT_RE = re.compile(r"(\d+(?:[.,]\d+)?)")
TIME_RE = re.compile(r"(\d+)\.(\d+)\.(\d+)")
AGF_RE = re.compile(r"%\s*(\d+(?:[.,]\d+)?)")
# "POWERSCOURT (GB)- SİYAH BEYAZ / DISTANT RELATIVE (IRE)"
#   -> baba: POWERSCOURT (GB) | anne: SİYAH BEYAZ | anne babası: DISTANT RELATIVE
ORIJIN_RE = re.compile(r"^(.*?)\s*-\s*(.*?)(?:\s*/\s*(.*))?$")


def parse_orijin(v) -> dict:
    """Soy bilgisini baba / anne / anne babası olarak ayırır."""
    out = {"baba": None, "anne": None, "anne_babasi": None}
    if v is None or (isinstance(v, float) and pd.isna(v)):
        return out
    m = ORIJIN_RE.match(str(v).strip())
    if not m:
        return out
    out["baba"] = (m.group(1) or "").strip() or None
    out["anne"] = (m.group(2) or "").strip() or None
    out["anne_babasi"] = (m.group(3) or "").strip() or None
    return out


def _to_float(x) -> float:
    if pd.isna(x):
        return np.nan
    s = str(x).strip().replace(",", ".")
    m = WEIGHT_RE.search(s)
    return float(m.group(1)) if m else np.nan


def parse_son6(s) -> dict:
    """'483611' benzeri son 6 yarış dizisini istatistiklere çevirir.
    0 = 10 ve üzeri sıralama olarak kabul edilir."""
    out = {"son6_adet": 0, "son6_ort": np.nan, "son6_kazanma": 0.0,
           "son6_tabela": 0.0, "son_yaris": np.nan}
    if pd.isna(s):
        return out
    digits = [int(d) for d in SON6_DIGIT_RE.findall(str(s))][:6]
    if not digits:
        return out
    vals = [10 if d == 0 else d for d in digits]
    out["son6_adet"] = len(vals)
    out["son6_ort"] = float(np.mean(vals))
    out["son6_kazanma"] = float(np.mean([v == 1 for v in vals]))
    out["son6_tabela"] = float(np.mean([v <= 3 for v in vals]))
    out["son_yaris"] = float(vals[-1])  # dizide en sağdaki en yeni yarış
    return out


def parse_derece(s) -> float:
    """'1.33.51' -> saniye."""
    if pd.isna(s):
        return np.nan
    m = TIME_RE.search(str(s))
    if not m:
        return np.nan
    mm, ss, ff = (int(m.group(i)) for i in (1, 2, 3))
    return mm * 60 + ss + ff / 100


def parse_agf(s) -> float:
    if pd.isna(s):
        return np.nan
    m = AGF_RE.search(str(s))
    return float(m.group(1).replace(",", ".")) / 100 if m else np.nan


def parse_yas(s) -> tuple[float, str]:
    """'4y a a' -> (4, cins/cinsiyet kodu)."""
    if pd.isna(s):
        return np.nan, ""
    txt = str(s)
    m = re.search(r"(\d+)\s*y", txt)
    return (float(m.group(1)) if m else np.nan), txt


def build_features(df: pd.DataFrame,
                   jockey_stats: pd.DataFrame | None = None,
                   trainer_stats: pd.DataFrame | None = None) -> pd.DataFrame:
    """Ham satırlardan model özellikleri üretir. Orijinal df'i bozmaz."""
    d = df.copy()
    d.columns = [str(c).strip() for c in d.columns]

    # Kolon adlarını normalize et
    ren = {}
    for c in d.columns:
        lc = c.lower()
        if lc.startswith("at ") or lc == "at i̇smi" or "at ismi" in lc or "at isim" in lc:
            ren[c] = "at"
        elif lc in ("n", "no"):
            ren[c] = "no"
        elif "sıklet" in lc or "siklet" in lc:
            ren[c] = "siklet_raw"
        elif lc.startswith("son 6"):
            ren[c] = "son6_raw"
        elif "en iyi" in lc:
            ren[c] = "derece_raw"
        elif lc == "gny" or "ganyan" in lc:
            ren[c] = "gny_raw"
        elif lc == "agf":
            ren[c] = "agf_raw"
        elif lc in ("yaş", "yas"):
            ren[c] = "yas_raw"
        elif lc == "st":
            ren[c] = "st_raw"
        elif lc.startswith("antrenör"):
            ren[c] = "Antrenör"
    d = d.rename(columns=ren)

    # Temel dönüşümler
    # İsim temizliği scraper.clean_name içinde yapılıyor; burada tekrar
    # regex uygulamak "GÜÇLÜ TÜRK" gibi isimlerin son harfini yiyordu.
    d["at"] = d.get("at", "").astype(str).str.strip()
    d["hp"] = pd.to_numeric(d.get("HP"), errors="coerce")
    d["kgs"] = pd.to_numeric(d.get("KGS"), errors="coerce")
    d["s20"] = pd.to_numeric(d.get("s20"), errors="coerce")
    d["siklet"] = d.get("siklet_raw", np.nan).map(_to_float) if "siklet_raw" in d else np.nan
    d["derece_sn"] = d.get("derece_raw", np.nan).map(parse_derece) if "derece_raw" in d else np.nan
    d["gny"] = d.get("gny_raw", np.nan).map(_to_float) if "gny_raw" in d else np.nan
    d["agf"] = d.get("agf_raw", np.nan).map(parse_agf) if "agf_raw" in d else np.nan
    d["start_no"] = pd.to_numeric(
        d.get("st_raw", np.nan).astype(str).str.extract(r"(\d+)")[0], errors="coerce") \
        if "st_raw" in d else np.nan

    if "yas_raw" in d:
        parsed = d["yas_raw"].map(parse_yas)
        d["yas"] = [p[0] for p in parsed]
        d["cins_kodu"] = [p[1] for p in parsed]
        d["arap_mi"] = d["cins_kodu"].str.contains(r"\ba\b", regex=True).fillna(False).astype(int) \
            if d["cins_kodu"].notna().any() else 0
    else:
        d["yas"] = np.nan

    # Soy bilgisi — özellikle İLK KEZ koşan atlar için tek bilgi kaynağı.
    # Atın geçmişi yoktur ama babasının kum/çim performansı bilinir.
    orj_kolonu = next((c for c in d.columns if "rijin" in str(c)), None)
    if orj_kolonu is not None:
        soy = pd.DataFrame(list(d[orj_kolonu].map(parse_orijin)), index=d.index)
        for c in soy.columns:
            d[c] = soy[c]

    if "son6_raw" in d:
        son6 = pd.DataFrame(list(d["son6_raw"].map(parse_son6)))
        d = pd.concat([d.reset_index(drop=True), son6.reset_index(drop=True)], axis=1)

    # Oran tabanlı özellikler (piyasa bilgisi çok güçlü sinyaldir)
    d["gny_implied"] = 1.0 / d["gny"].replace(0, np.nan)
    d["mesafe"] = pd.to_numeric(d.get("mesafe"), errors="coerce")
    d["pist_kum"] = (d.get("pist", "").astype(str).str.lower() == "kum").astype(int)
    d["pist_cim"] = (d.get("pist", "").astype(str).str.lower() == "çim").astype(int)

    # Yarış içi göreli özellikler (aynı koşudaki rakiplere göre)
    grp = d.groupby(["tarih", "sehir", "kosu_no"], dropna=False)
    for col in ("hp", "agf", "gny_implied", "son6_ort", "derece_sn", "siklet"):
        if col in d:
            d[f"{col}_rel"] = d[col] - grp[col].transform("mean")
            d[f"{col}_rank"] = grp[col].rank(pct=True)
    d["alan_sayisi"] = grp["at"].transform("count")

    # Jokey / antrenör tarihsel istatistikleri (train aşamasında hesaplanıp verilir)
    if jockey_stats is not None and "Jokey" in d:
        d = d.merge(jockey_stats, how="left", left_on="Jokey", right_index=True)
    if trainer_stats is not None and "Antrenör" in d:
        d = d.merge(trainer_stats, how="left", left_on="Antrenör", right_index=True)

    return d


# Program sayfasında olup sonuç sayfasında OLMAYAN kolonlar. Eğitim verisi
# sonuç sayfalarından geldiği için bunlar eğitimde boş kalır; modele koymanın
# faydası yok. Karşılıkları history.py'de geçmişten hesaplanıyor:
#   Son 6 Y.  -> at_genel_rate / at_ilk3_rate
#   KGS       -> dinlenme_gun
#   En İyi D. -> at_hiz_* (geçmiş derecelerden)
PROGRAM_ONLY_COLS = ["kgs", "s20", "son6_adet", "son6_ort", "son6_kazanma",
                     "son6_tabela", "son_yaris", "derece_sn"]

FEATURE_COLS = [
    "hp", "siklet", "yas", "gny", "agf",
    "gny_implied", "start_no", "mesafe", "pist_kum", "pist_cim",
    "hp_rel", "agf_rel", "gny_implied_rel", "siklet_rel",
    "hp_rank", "agf_rank", "gny_implied_rank", "siklet_rank", "alan_sayisi",
]


def compute_person_stats(results: pd.DataFrame, col: str, prefix: str,
                         min_races: int = 10) -> pd.DataFrame:
    """Sonuç verisinden jokey/antrenör kazanma oranlarını çıkarır."""
    if col not in results or "kazandi" not in results:
        return pd.DataFrame()
    g = results.groupby(col)["kazandi"].agg(["mean", "count"])
    g.columns = [f"{prefix}_kazanma", f"{prefix}_yaris"]
    g.loc[g[f"{prefix}_yaris"] < min_races, f"{prefix}_kazanma"] = np.nan
    return g


def add_label_from_results(results: pd.DataFrame) -> pd.DataFrame:
    """Sonuç sayfasında ilk kolon bitiş sırasıdır; 'kazandi' ve 'ilk2' etiketi üretir."""
    d = results.copy()
    # sonuç tablolarında sıralama genelde ilk kolonda veya 'S' başlıklı kolonda gelir
    order_col = None
    for cand in ("S", "S.", "Sıra", "Sira", d.columns[0]):
        if cand in d.columns:
            vals = pd.to_numeric(d[cand].astype(str).str.extract(r"(\d+)")[0],
                                 errors="coerce")
            if vals.notna().mean() > 0.5:
                order_col = cand
                d["bitis"] = vals
                break
    if order_col is None:
        d["bitis"] = np.nan
    d["kazandi"] = (d["bitis"] == 1).astype(int)
    d["ilk2"] = (d["bitis"] <= 2).astype(int)
    return d
