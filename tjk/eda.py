# -*- coding: utf-8 -*-
"""Günlük sonuçlar için EDA tabloları ve Altair grafikleri."""
from __future__ import annotations

import altair as alt
import numpy as np
import pandas as pd

RENK_1 = "#2a6f97"   # kazanma
RENK_2 = "#a9cce3"   # 2-3.
RENK_S = "#d35400"   # sürpriz


def hazirla(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "hipodrom" not in df.columns:
        df["hipodrom"] = ""
    df["kosu_etiket"] = (df["hipodrom"].astype(str) + " " +
                         df["kosu_no"].astype(str) + ". koşu").str.strip()
    grp = ["hipodrom", "kosu_no"] if "hipodrom" in df.columns else ["kosu_no"]
    if "agf" in df.columns and df["agf"].notna().any():
        df["agf_sira"] = df.groupby(grp)["agf"].rank(ascending=False, method="min")
    if "ganyan" in df.columns and df["ganyan"].notna().any():
        df["ganyan_sira"] = df.groupby(grp)["ganyan"].rank(method="min")
    return df


def kazananlar(df: pd.DataFrame, surpriz_ganyan=10.0, agf_esigi=6) -> pd.DataFrame:
    w = df[df["sira"] == 1].copy()
    surpriz = pd.Series(False, index=w.index)
    if "ganyan" in w:
        surpriz |= w["ganyan"].fillna(0) >= surpriz_ganyan
    if "agf_sira" in w:
        surpriz |= w["agf_sira"].fillna(0) >= agf_esigi
    w["surpriz"] = surpriz
    kol = [c for c in ["hipodrom", "kosu_no", "saat", "mesafe", "pist", "no", "at",
                       "jokey", "antrenor", "sahip", "ganyan", "agf", "agf_sira",
                       "derece", "surpriz"] if c in w.columns]
    return w[kol].sort_values([c for c in ["hipodrom", "kosu_no"] if c in kol])


def ozet(df: pd.DataFrame, w: pd.DataFrame) -> dict:
    grp = ["hipodrom", "kosu_no"] if "hipodrom" in df.columns else ["kosu_no"]
    fav_kazandi = np.nan
    if "ganyan_sira" in df:
        fav_kazandi = (df[df["sira"] == 1]["ganyan_sira"] == 1).mean()
    return {
        "kosu": df.groupby(grp).ngroups,
        "at": len(df),
        "fav_kazanma": fav_kazandi,
        "ort_ganyan": w["ganyan"].median() if "ganyan" in w else np.nan,
        "surpriz": int(w["surpriz"].sum()),
    }


def kisi_tablosu(df: pd.DataFrame, kolon: str) -> pd.DataFrame:
    if kolon not in df.columns:
        return pd.DataFrame()
    t = (df.dropna(subset=[kolon])
           .groupby(kolon)
           .agg(binis=("sira", "size"), birinci=("kazandi", "sum"), ilk3=("ilk3", "sum"))
           .reset_index())
    t["ikinci_ucuncu"] = t["ilk3"] - t["birinci"]
    t["kazanma_%"] = (100 * t["birinci"] / t["binis"]).round(0)
    return t.sort_values(["birinci", "ilk3", "binis"], ascending=[False, False, True])


def kisi_grafik(t: pd.DataFrame, kolon: str, baslik: str, n=15) -> alt.Chart:
    t = t[t["ilk3"] > 0].head(n)
    uzun = t.melt(id_vars=[kolon, "binis"], value_vars=["birinci", "ikinci_ucuncu"],
                  var_name="derece", value_name="adet")
    uzun["derece"] = uzun["derece"].map({"birinci": "1.", "ikinci_ucuncu": "2.–3."})
    sira = t[kolon].tolist()
    return (alt.Chart(uzun, title=baslik)
            .mark_bar()
            .encode(y=alt.Y(f"{kolon}:N", sort=sira, title=None),
                    x=alt.X("adet:Q", title="Adet", axis=alt.Axis(tickMinStep=1)),
                    color=alt.Color("derece:N", scale=alt.Scale(
                        domain=["1.", "2.–3."], range=[RENK_1, RENK_2]),
                        legend=alt.Legend(orient="bottom", title=None)),
                    order=alt.Order("derece:N", sort="ascending"),
                    tooltip=[kolon, "derece", "adet", alt.Tooltip("binis", title="Biniş")])
            .properties(height=max(160, 22 * len(t))))


def kazanan_ganyan_grafik(w: pd.DataFrame) -> alt.Chart | None:
    if "ganyan" not in w or w["ganyan"].isna().all():
        return None
    w = w.copy()
    hip = w["hipodrom"].astype(str) if "hipodrom" in w else ""
    w["etiket"] = (hip + " " + w["kosu_no"].astype(str)).str.strip()
    w["tip"] = np.where(w["surpriz"], "Sürpriz", "Beklenen")
    return (alt.Chart(w, title="Kazananın ganyanı (koşu bazında)")
            .mark_bar()
            .encode(x=alt.X("etiket:N", sort=None, title=None),
                    y=alt.Y("ganyan:Q", scale=alt.Scale(type="log"), title="Ganyan (log)"),
                    color=alt.Color("tip:N", scale=alt.Scale(domain=["Beklenen", "Sürpriz"],
                                                             range=[RENK_1, RENK_S]),
                                    legend=alt.Legend(orient="bottom", title=None)),
                    tooltip=["etiket", "at", "jokey", "antrenor", "sahip", "ganyan"])
            .properties(height=260))


def ganyan_sira_grafik(df: pd.DataFrame) -> alt.Chart | None:
    if "ganyan" not in df or df["ganyan"].isna().all():
        return None
    d = df.dropna(subset=["ganyan", "sira"]).copy()
    d["sira"] = d["sira"].astype(int)
    return (alt.Chart(d, title="Ganyan oranı vs bitiriş sırası")
            .mark_circle(size=55, opacity=0.6)
            .encode(x=alt.X("ganyan:Q", scale=alt.Scale(type="log"), title="Ganyan (log)"),
                    y=alt.Y("sira:Q", scale=alt.Scale(reverse=True), title="Sıra"),
                    color=alt.condition(alt.datum.sira == 1, alt.value(RENK_S),
                                        alt.value(RENK_1)),
                    tooltip=["kosu_etiket", "at", "jokey", "ganyan", "sira"])
            .properties(height=280))


def model_vs_sonuc(sonuc: pd.DataFrame, pred: pd.DataFrame | None) -> pd.DataFrame:
    """Aynı gün/hipodrom tahminleri varsa: modelin 1 numarası ne yaptı?"""
    if pred is None or pred.empty or "no" not in sonuc:
        return pd.DataFrame()
    p = pred[["kosu_no", "no", "at", "p_win"]].copy()
    p["no"] = pd.to_numeric(p["no"], errors="coerce").astype("Int64")
    p["kosu_no"] = pd.to_numeric(p["kosu_no"], errors="coerce").astype("Int64")
    p["model_sira"] = p.groupby("kosu_no")["p_win"].rank(ascending=False, method="first")
    s = sonuc[["kosu_no", "no", "sira"]].copy()
    s["kosu_no"] = pd.to_numeric(s["kosu_no"], errors="coerce").astype("Int64")
    m = p.merge(s, on=["kosu_no", "no"], how="inner")
    if m.empty:
        return pd.DataFrame()
    satir = []
    for k, g in m.groupby("kosu_no"):
        top = g.sort_values("p_win", ascending=False).iloc[0]
        kaz = g[g["sira"] == 1]
        satir.append({
            "Koşu": int(k),
            "Model 1. tercihi": f"{int(top['no'])} {top['at']}",
            "Kazanma %": round(100 * top["p_win"], 1),
            "Bitirdiği sıra": int(top["sira"]) if pd.notna(top["sira"]) else None,
            "Kazanan": (f"{int(kaz.iloc[0]['no'])} {kaz.iloc[0]['at']}" if len(kaz) else "-"),
            "Kazananın model sırası": (int(kaz.iloc[0]["model_sira"]) if len(kaz) else None),
        })
    return pd.DataFrame(satir)
