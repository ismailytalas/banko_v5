# -*- coding: utf-8 -*-
"""Altılı kupon laboratuvarı — farklı bütçe × strateji kombinasyonlarında kupon üretir.

Her kupon için:
  isabet % (model) = Π_ayak Σ seçilen atların p_win'i
  halk %           = Π_ayak Σ seçilen atların AGF'si
  halka göre avantaj = model / halk  (>1: kupon halkın küçümsediği atları içeriyor,
                                       isabet ederse ikramiye ortalamanın üstünde olur)

Stratejiler at ekleme sırasını belirler (isabet hep model p_win ile hesaplanır):
  Olasılık : en olası atlar önce
  Dengeli  : p_win × değer^0.5
  Değer    : p_win × değer^1   (halkın geride bıraktığı atları öne alır)
"""
from __future__ import annotations

import math

import numpy as np
import pandas as pd

STRATEJILER = {"Olasılık": 0.0, "Dengeli": 0.5, "Değer": 1.0}


def ayak_hazirla(leg: pd.DataFrame) -> pd.DataFrame:
    d = leg.dropna(subset=["no", "p_win"]).copy()
    d["no"] = d["no"].astype(int)
    d["p"] = d["p_win"] / d["p_win"].sum()
    if "agf" in d and d["agf"].notna().any():
        d["h"] = d["agf"].fillna(d["agf"].min() / 2)
        d["h"] = d["h"] / d["h"].sum()
    else:
        d["h"] = np.nan
    if "deger" not in d:
        d["deger"] = np.nan
    return d.sort_values("p", ascending=False).reset_index(drop=True)


def _sira(d: pd.DataFrame, alfa: float) -> list[int]:
    deg = d["deger"].fillna(1.0).clip(0.3, 4.0)
    skor = d["p"] * deg ** alfa
    return d.assign(_s=skor).sort_values("_s", ascending=False).index.tolist()


def kupon_kur(ayaklar: list[pd.DataFrame], butce: float, misli: float,
              alfa: float = 0.0, min_at: int = 1, max_at: int = 8) -> dict | None:
    """Greedy: her adımda (log isabet artışı)/(log maliyet artışı) en yüksek at eklenir."""
    sira = [_sira(d, alfa) for d in ayaklar]
    sec = [s[:min(min_at, len(s))] for s in sira]

    def maliyet(s):
        return math.prod(len(x) for x in s) * misli

    if maliyet(sec) > butce:
        return None
    while True:
        en_iyi, en_skor = None, 0.0
        for i, d in enumerate(ayaklar):
            n = len(sec[i])
            if n >= min(max_at, len(sira[i])):
                continue
            aday = sira[i][n]
            if maliyet(sec) / n * (n + 1) > butce + 1e-9:
                continue
            # değer stratejisinde kazanç, ağırlıklı olasılıkla ölçülür
            w = d["p"] * d["deger"].fillna(1.0).clip(0.3, 4.0) ** alfa
            eski = w.loc[sec[i]].sum()
            kazanc = math.log((eski + w.loc[aday]) / eski)
            skor = kazanc / math.log((n + 1) / n)
            if skor > en_skor:
                en_iyi, en_skor = (i, aday), skor
        if en_iyi is None:
            break
        sec[en_iyi[0]].append(en_iyi[1])
    return _ozet(ayaklar, sec, misli)


def _ozet(ayaklar, sec, misli) -> dict:
    p = math.prod(d.loc[s, "p"].sum() for d, s in zip(ayaklar, sec))
    h_var = all(d["h"].notna().all() for d in ayaklar)
    h = math.prod(d.loc[s, "h"].sum() for d, s in zip(ayaklar, sec)) if h_var else np.nan
    komb = math.prod(len(s) for s in sec)
    return {"secim": [sorted(d.loc[s, "no"].tolist()) for d, s in zip(ayaklar, sec)],
            "idx": sec, "kombinasyon": komb, "maliyet": komb * misli,
            "p_isabet": p, "p_halk": h,
            "avantaj": p / h if h and h > 0 else np.nan,
            "tek_ayak": sum(len(s) == 1 for s in sec)}


def butce_izgarasi(en_az: float, en_cok: float, adim: int = 7) -> list[float]:
    return sorted({round(x, 0) for x in np.geomspace(en_az, en_cok, adim)})


def varyasyonlar(ayaklar: list[pd.DataFrame], misli: float, butceler: list[float],
                 stratejiler: dict = STRATEJILER, min_at: int = 1,
                 max_at: int = 8) -> list[dict]:
    out, gorulen = [], set()
    for ad, alfa in stratejiler.items():
        for b in butceler:
            k = kupon_kur(ayaklar, b, misli, alfa, min_at, max_at)
            if k is None:
                continue
            anahtar = tuple(tuple(s) for s in k["secim"])
            if anahtar in gorulen:
                continue
            gorulen.add(anahtar)
            k.update({"strateji": ad, "butce": b})
            out.append(k)
    return sorted(out, key=lambda k: (k["maliyet"], -k["p_isabet"]))


def ozet_tablosu(kuponlar: list[dict], kosu_nolar: list[int]) -> pd.DataFrame:
    s = []
    for i, k in enumerate(kuponlar, 1):
        s.append({"#": i, "Strateji": k["strateji"], "Kombinasyon": k["kombinasyon"],
                  "Maliyet (TL)": round(k["maliyet"], 2),
                  "İsabet %": round(100 * k["p_isabet"], 2),
                  "Halk %": None if pd.isna(k["p_halk"]) else round(100 * k["p_halk"], 2),
                  "Halka göre avantaj": None if pd.isna(k["avantaj"]) else round(k["avantaj"], 2),
                  "Tek at ayak": k["tek_ayak"],
                  "Kupon": "  /  ".join(",".join(map(str, x)) for x in k["secim"])})
    return pd.DataFrame(s)


def kupon_detay(ayaklar, kupon, kosu_nolar) -> pd.DataFrame:
    s = []
    for d, idx, k in zip(ayaklar, kupon["idx"], kosu_nolar):
        sec = d.loc[idx]
        s.append({"Koşu": k, "At sayısı": len(idx),
                  "Seçimler": ", ".join(f"{r.no} {r.at}" for r in sec.itertuples()),
                  "Ayak isabet %": round(100 * sec["p"].sum(), 1),
                  "Halk %": None if sec["h"].isna().any() else round(100 * sec["h"].sum(), 1)})
    return pd.DataFrame(s)


def tek_at_alternatifleri(ayaklar, kupon, kosu_nolar, n: int = 4) -> pd.DataFrame:
    """Tek at bırakılan ayaklarda bir sonraki adaylar ve eklenirse kuponun yeni hali."""
    s = []
    for d, idx, k in zip(ayaklar, kupon["idx"], kosu_nolar):
        if len(idx) != 1:
            continue
        banko = d.loc[idx[0]]
        p_ayak = banko["p"]
        for r in d.drop(index=idx).head(n).itertuples():
            yeni_p = kupon["p_isabet"] / p_ayak * (p_ayak + r.p)
            s.append({"Koşu": k, "Banko": f"{banko['no']} {banko['at']} (%{100*p_ayak:.0f})",
                      "Alternatif": f"{r.no} {r.at}",
                      "Kazanma %": round(100 * r.p, 1),
                      "AGF sıra": None if pd.isna(getattr(r, "agf_sira", np.nan))
                      else int(r.agf_sira),
                      "Değer": None if pd.isna(r.deger) else round(r.deger, 2),
                      "Eklenirse isabet %": round(100 * yeni_p, 2),
                      "Eklenirse maliyet": round(kupon["maliyet"] * 2, 2)})
    return pd.DataFrame(s)


def yedek_at_tablosu(ayaklar, kosu_nolar, n: int = 6) -> pd.DataFrame:
    """Her ayakta ilk n at: kümülatif olasılık — kaç at alınırsa ayağın yüzde kaçı kapanır."""
    s = []
    for d, k in zip(ayaklar, kosu_nolar):
        kum = 0.0
        for i, r in enumerate(d.head(n).itertuples(), 1):
            kum += r.p
            s.append({"Koşu": k, "Sıra": i, "No": r.no, "At": r.at,
                      "Kazanma %": round(100 * r.p, 1), "Kümülatif %": round(100 * kum, 1),
                      "AGF sıra": None if pd.isna(getattr(r, "agf_sira", np.nan))
                      else int(r.agf_sira),
                      "Değer": None if pd.isna(r.deger) else round(r.deger, 2)})
    return pd.DataFrame(s)
