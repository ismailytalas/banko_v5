# -*- coding: utf-8 -*-
"""
Tahmin çekirdeği — hem Streamlit önyüzü hem komut satırı bunu kullanır.

İçerik:
  - Geçmiş veri + günün programını birleştirip özellik üretme
  - Piyasasız (temel) iki model: kazanma ve ilk-2
  - Sürpriz işaretleme
  - İkili tek kolon kurma
  - Altılı Ganyan: dar / orta / geniş kapsam
"""
from __future__ import annotations

import pathlib
import datetime as dt
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb

from . import scraper
from .features import build_features, add_label_from_results
from .history import add_history_features
from .model import FEATURE_COLS, RACE_KEY

ROOT = pathlib.Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

# Modele piyasa bilgisi VERİLMEZ: amaç piyasadan bağımsız bir görüş üretmek,
# sonra o görüşü piyasayla karşılaştırıp değer aramak.
PIYASA_KOLONLARI = {"gny", "agf", "gny_implied", "gny_implied_rel",
                    "gny_implied_rank", "agf_rel", "agf_rank",
                    "sabah_gny", "agf_prg", "agf_sira_prg"}

PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=.05,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1)


# ---------------------------------------------------------------------------
# Veri ve model
# ---------------------------------------------------------------------------

def kosu_ici_olasilik(df: pd.DataFrame, col: str, out: str) -> pd.DataFrame:
    """Skorları koşu içinde 1'e toplanacak şekilde normalize eder."""
    d = df.copy()
    t = d[col].clip(1e-9, 1 - 1e-9)
    d["_l"] = np.log(t / (1 - t))
    d[out] = d.groupby(RACE_KEY, dropna=False)["_l"].transform(
        lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum())
    return d.drop(columns=["_l"])


def gecmisi_yukle() -> pd.DataFrame:
    files = sorted(RAW.glob("results_*.parquet"))
    if not files:
        raise FileNotFoundError("data/raw altında geçmiş veri yok.")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return add_label_from_results(raw)


def gunun_programi(tarih: dt.date, sehir: str | None = None) -> pd.DataFrame:
    parcalar = []
    for sid, ad in scraper.get_cities(tarih):
        if not scraper.is_domestic(ad):
            continue
        if sehir and sehir.lower() not in ad.lower():
            continue
        try:
            d = scraper.get_program(tarih, sid, ad)
            if not d.empty:
                parcalar.append(d)
        except Exception:
            continue
    return pd.concat(parcalar, ignore_index=True) if parcalar else pd.DataFrame()


def ozellik_uret(hist: pd.DataFrame, prog: pd.DataFrame,
                 tarih: dt.date) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Geçmiş + bugünü birlikte işler; (bugün, geçmiş) döndürür.

    Birlikte işlemek şart: atın/jokeyin geçmiş istatistikleri ancak tüm
    tarihçeye bakılarak hesaplanır. Bugünün satırlarında sonuç yok; etiketi
    0 veriyoruz — prior hesabı aynı günü zaten dışladığı için sızıntı olmaz.
    """
    prog = prog.copy()
    prog["S"] = np.nan
    birlesik = add_label_from_results(pd.concat([hist, prog], ignore_index=True))
    birlesik["kazandi"] = birlesik["kazandi"].fillna(0)

    df = build_features(birlesik)
    df["kazandi"] = birlesik["kazandi"].values
    df["bitis"] = birlesik["bitis"].values
    df["ilk2"] = (df["bitis"] <= 2).astype(float)
    df.loc[df["bitis"].isna(), "ilk2"] = np.nan
    df = add_history_features(df)

    bugun = df[df["tarih"] == str(tarih)].copy()
    gecmis = df[(df["tarih"] < str(tarih)) & df["bitis"].notna()].copy()
    return bugun, gecmis


def modelleri_hazirla(gecmis: pd.DataFrame, feats: list[str],
                      yeniden: bool = False):
    MODELS.mkdir(exist_ok=True)
    y_win, y_ilk2 = MODELS / "temel_win.txt", MODELS / "temel_ilk2.txt"
    if yeniden or not y_win.exists() or not y_ilk2.exists():
        d1 = gecmis.dropna(subset=["kazandi"])
        d2 = gecmis.dropna(subset=["ilk2"])
        m_win = lgb.train(PARAMS, lgb.Dataset(d1[feats], label=d1["kazandi"]), 800)
        m_ilk2 = lgb.train(PARAMS, lgb.Dataset(d2[feats], label=d2["ilk2"]), 800)
        m_win.save_model(str(y_win))
        m_ilk2.save_model(str(y_ilk2))
    else:
        m_win = lgb.Booster(model_file=str(y_win))
        m_ilk2 = lgb.Booster(model_file=str(y_ilk2))
    return m_win, m_ilk2


def tahmin_et(bugun: pd.DataFrame, m_win, m_ilk2,
              feats: list[str]) -> pd.DataFrame:
    d = bugun.copy()
    for c in feats:
        if c not in d.columns:
            d[c] = np.nan
    d["s_win"] = m_win.predict(d[feats])
    d = kosu_ici_olasilik(d, "s_win", "p_win")
    d["p_ilk2"] = m_ilk2.predict(d[feats])

    o = pd.to_numeric(d["gny"], errors="coerce")
    d["oran"] = o / 100.0 if o.median(skipna=True) > 50 else o
    d["p_piyasa"] = 1.0 / d["oran"]
    d["p_piyasa"] = d["p_piyasa"] / d.groupby(RACE_KEY)["p_piyasa"].transform("sum")
    d["deger"] = d["p_win"] / d["p_piyasa"].replace(0, np.nan)

    d["agf_sira"] = d.groupby(RACE_KEY)["agf"].rank(ascending=False, method="min")
    d["model_sira"] = d.groupby(RACE_KEY)["p_win"].rank(ascending=False, method="min")
    d["ilk2_sira"] = d.groupby(RACE_KEY)["p_ilk2"].rank(ascending=False, method="min")

    if "no" not in d.columns or d["no"].isna().all():
        d["no"] = d.get("start_no", d["at"])
    d["no"] = pd.to_numeric(d["no"], errors="coerce")
    return d


# ---------------------------------------------------------------------------
# Sürpriz işaretleme
# ---------------------------------------------------------------------------

def surpriz_notlari(rd: pd.DataFrame, agf_esigi: int = 6,
                    model_esigi: int = 4, deger_esigi: float = 1.6) -> list[str]:
    """Her at için not listesi üretir.

    SÜRPRİZ tanımı: model atı ilk `model_esigi` sırada görüyor AMA halkın AGF
    sıralamasında `agf_esigi` veya daha geride. Varsayılan 6, yani halkın ilk
    5 favorisi sürpriz sayılmaz.
    """
    notlar = []
    for _, r in rd.iterrows():
        n = []
        if (pd.notna(r.get("model_sira")) and pd.notna(r.get("agf_sira"))
                and r["model_sira"] <= model_esigi and r["agf_sira"] >= agf_esigi):
            n.append("🔥 SÜRPRİZ")
        if pd.notna(r.get("deger")) and r["deger"] >= deger_esigi:
            n.append("değerli")
        if pd.notna(r.get("dinlenme_gun")) and 20 <= r["dinlenme_gun"] <= 45:
            n.append("form aralığı")
        if r.get("ilk_yaris_mi") == 1:
            n.append("ilk yarış")
        if pd.notna(r.get("at_pist_rate")) and r["at_pist_rate"] > 0.25:
            n.append("pist uyumlu")
        if pd.notna(r.get("dinlenme_gun")) and r["dinlenme_gun"] > 120:
            n.append("uzun ara")
        notlar.append(", ".join(n))
    return notlar


# ---------------------------------------------------------------------------
# İkili tek kolon
# ---------------------------------------------------------------------------

def ikili_kolon(rd: pd.DataFrame, n_at: int) -> tuple[pd.DataFrame, int, float]:
    """İlk-2 olasılığına göre en iyi N atı seçer.

    İkili tek kolonda N at yazılır, C(N,2) kombinasyon oluşur ve gerçek ilk
    ikinin İKİSİ de kolonda ise kazanılır.
    """
    d = rd.dropna(subset=["p_ilk2"]).sort_values("p_ilk2", ascending=False)
    sec = d.head(n_at)
    p = sec["p_ilk2"].values
    tut = sum(p[i] * p[j] / 2 for i, j in combinations(range(len(p)), 2))
    return sec, len(list(combinations(range(len(sec)), 2))), float(min(tut, 1.0))


# ---------------------------------------------------------------------------
# Altılı Ganyan — dar / orta / geniş
# ---------------------------------------------------------------------------

KAPSAM_HEDEFLERI = {"Dar": 48, "Orta": 432, "Geniş": 2592}


def altili_kur(legs: list[pd.DataFrame], hedef_kombinasyon: int,
               max_ayak: int = 6) -> dict:
    """Hedef kombinasyon sayısına ulaşana kadar ayakları genişletir.

    Her adımda, eklendiğinde ayağın toplam olasılığını en çok artıran atı
    seçer. Böylece belirsiz ayaklar genişler, net ayaklar tek geçilir.
    """
    sirali = []
    for leg in legs:
        d = leg.dropna(subset=["p_win"]).sort_values("p_win", ascending=False)
        sirali.append(list(zip(d["no"], d["p_win"], d["at"])))

    secim = [lst[:1] for lst in sirali]

    def kombinasyon(s):
        n = 1
        for leg in s:
            n *= max(len(leg), 1)
        return n

    while True:
        en_iyi = None
        mevcut = kombinasyon(secim)
        for i, lst in enumerate(sirali):
            k = len(secim[i])
            if k >= min(max_ayak, len(lst)):
                continue
            aday = list(secim[i]) + [lst[k]]
            deneme = secim[:i] + [aday] + secim[i + 1:]
            yeni = kombinasyon(deneme)
            if yeni > hedef_kombinasyon:
                continue
            eski_p = sum(p for _, p, _ in secim[i])
            yeni_p = sum(p for _, p, _ in aday)
            kazanc = (yeni_p / max(eski_p, 1e-9)) - 1
            oran = kazanc / max(yeni - mevcut, 1)
            if en_iyi is None or oran > en_iyi[0]:
                en_iyi = (oran, i, aday)
        if en_iyi is None:
            break
        secim[en_iyi[1]] = en_iyi[2]

    p_isabet = float(np.prod([sum(p for _, p, _ in leg) for leg in secim]))
    return {
        "secim": secim,
        "kombinasyon": kombinasyon(secim),
        "p_isabet": p_isabet,
    }


def altili_tablosu(legs: list[pd.DataFrame], kosu_nolari: list[int],
                   misli: float = 3.0) -> dict:
    """Dar / Orta / Geniş üç kupon birden üretir."""
    sonuc = {}
    for ad, hedef in KAPSAM_HEDEFLERI.items():
        t = altili_kur(legs, hedef)
        satirlar = []
        for i, leg in enumerate(t["secim"]):
            satirlar.append({
                "Ayak": f"{kosu_nolari[i]}. Koşu",
                "At sayısı": len(leg),
                "Seçimler": " - ".join(
                    f"{int(no)}" if pd.notna(no) else "?" for no, _, _ in leg),
                "İsimler": ", ".join(ad2 for _, _, ad2 in leg),
            })
        sonuc[ad] = {
            "tablo": pd.DataFrame(satirlar),
            "kombinasyon": t["kombinasyon"],
            "maliyet": t["kombinasyon"] * misli,
            "p_isabet": t["p_isabet"],
        }
    return sonuc
