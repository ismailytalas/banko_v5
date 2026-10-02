# -*- coding: utf-8 -*-
"""Koşu başına İkili ve Çifte önerileri — NVIDIA NIM üzerinden gpt-oss.

Tüm rakamlar burada, bizim tahmin tablosundan hesaplanır:
  p_win (Kazanma %), p_ilk2 (İlk2 %), agf (halk), agf_sira, deger
  İkili olasılığı = Harville(a,b) + Harville(b,a)  (model ve halk için ayrı)
  İkili değer     = model ikili % / halk ikili %   (>1: halk bu ikiliyi küçümsüyor)
  Çifte olasılığı = p_k(a) × p_{k+1}(b)
LLM yalnızca bu adayları okuyup yorumlar; at, oran veya yüzde uydurmaz.
API anahtarı yoksa aynı adaylar şablon metinle gösterilir.

Anahtar: st.secrets["NVIDIA_API_KEY"] veya ortam değişkeni NVIDIA_API_KEY
"""
from __future__ import annotations

import itertools
import json
import os
import re

import pandas as pd
import requests

NIM_URL = "https://integrate.api.nvidia.com/v1/chat/completions"
VARSAYILAN_MODEL = "openai/gpt-oss-120b"
ANAHTARLAR = ("ikili_guvenli", "ikili_surpriz", "ikili_kolon", "cifte")


# --- yardımcılar -----------------------------------------------------------------
def _hazirla(rd: pd.DataFrame) -> pd.DataFrame:
    rd = rd.dropna(subset=["no", "p_win"]).copy()
    rd["no"] = rd["no"].astype(int)
    rd["p"] = rd["p_win"] / rd["p_win"].sum()
    if "agf" in rd and rd["agf"].notna().any():
        rd["h"] = rd["agf"].fillna(rd["agf"].min() / 2)
        rd["h"] = rd["h"] / rd["h"].sum()
    else:
        rd["h"] = float("nan")
    return rd.sort_values("p", ascending=False)


def _harville_ikili(pa, pb):
    return pa * pb / max(1e-9, 1 - pa) + pb * pa / max(1e-9, 1 - pb)


def _yuzde(x, n=1):
    return None if x is None or pd.isna(x) else round(100 * float(x), n)


def _ikililer(rd: pd.DataFrame) -> pd.DataFrame:
    p = dict(zip(rd["no"], rd["p"]))
    h = dict(zip(rd["no"], rd["h"]))
    ad = dict(zip(rd["no"], rd["at"].astype(str)))
    s = []
    for a, b in itertools.combinations(sorted(p), 2):
        pm = _harville_ikili(p[a], p[b])
        ph = _harville_ikili(h[a], h[b]) if pd.notna(h[a]) else float("nan")
        s.append({"komb": f"{a}-{b}", "atlar": f"{ad[a]} / {ad[b]}", "p": pm,
                  "halk": ph, "deger": pm / ph if ph and ph > 0 else float("nan")})
    return pd.DataFrame(s)


def _kolon_p(rd: pd.DataFrame, nolar: list[int]) -> tuple[float, float]:
    sec = rd[rd["no"].isin(nolar)]
    pm = ph = 0.0
    for (a, pa, ha), (b, pb, hb) in itertools.combinations(
            sec[["no", "p", "h"]].itertuples(index=False), 2):
        pm += _harville_ikili(pa, pb)
        ph += _harville_ikili(ha, hb) if pd.notna(ha) else 0.0
    return pm, ph


def _kolon(rd, nolar, etiket):
    pm, ph = _kolon_p(rd, nolar)
    n = len(nolar)
    return {"tur": etiket, "atlar": "-".join(map(str, nolar)), "at_sayisi": n,
            "kombinasyon": n * (n - 1) // 2, "model_%": _yuzde(pm),
            "halk_%": _yuzde(ph) if ph else None}


# --- deterministik adaylar ----------------------------------------------------------
def adaylar(rd: pd.DataFrame, rd_sonraki: pd.DataFrame | None = None,
            agf_esigi: int = 6, min_p_surpriz: float = 0.025) -> dict:
    rd = _hazirla(rd)
    atlar = []
    for r in rd.head(7).to_dict("records"):
        atlar.append({"no": r["no"], "at": r["at"], "kazanma_%": _yuzde(r["p"]),
                      "ilk2_%": _yuzde(r.get("p_ilk2")),
                      "agf_sira": None if pd.isna(r.get("agf_sira")) else int(r["agf_sira"]),
                      "deger": None if pd.isna(r.get("deger")) else round(float(r["deger"]), 2)})

    ik = _ikililer(rd)
    guvenli = ik.sort_values("p", ascending=False).head(3)
    sur = ik[(ik["p"] >= min_p_surpriz) & ~ik["komb"].isin(guvenli["komb"])]
    if ik["halk"].notna().any():
        sur = sur[(sur["deger"] >= 1.3) & (sur["halk"] < 0.08)]
        sur = sur.sort_values("deger", ascending=False).head(3)
    else:
        sur = sur.head(0)

    def _kayit(d):
        return [{"komb": r["komb"], "atlar": r["atlar"], "model_%": _yuzde(r["p"]),
                 "halk_%": _yuzde(r["halk"]),
                 "deger": None if pd.isna(r["deger"]) else round(r["deger"], 2)}
                for r in d.to_dict("records")]

    sirali = rd["no"].tolist()
    kolonlar = [_kolon(rd, sirali[:n], f"ilk {n} at") for n in (3, 4, 5) if len(sirali) >= n]
    # değer kolonu: modelin ilk 3'ü + değeri en yüksek 2 at (kazanma ≥ %5)
    if "deger" in rd and len(sirali) >= 5:
        ek = (rd[~rd["no"].isin(sirali[:3]) & (rd["p"] >= 0.05)]
              .sort_values("deger", ascending=False)["no"].head(2).tolist())
        if len(ek) == 2 and set(ek) != set(sirali[3:5]):
            kolonlar.append(_kolon(rd, sirali[:3] + ek, "değer kolonu (5 at)"))

    cifte = None
    if rd_sonraki is not None and not rd_sonraki.empty:
        r2 = _hazirla(rd_sonraki)
        a1, a2 = rd.head(4), r2.head(4)
        ciftler = []
        for x in a1.to_dict("records"):
            for y in a2.to_dict("records"):
                pm, ph = x["p"] * y["p"], x["h"] * y["h"]
                ciftler.append({"komb": f"{x['no']}/{y['no']}",
                                "atlar": f"{x['at']} / {y['at']}",
                                "model_%": _yuzde(pm), "halk_%": _yuzde(ph),
                                "deger": round(pm / ph, 2) if ph and ph > 0 else None})
        cf = pd.DataFrame(ciftler)
        g = cf.sort_values("model_%", ascending=False).head(2)
        s = cf[(cf["model_%"] >= 100 * min_p_surpriz / 2) & cf["deger"].notna()]
        s = s[(s["deger"] >= 1.3) & (s["halk_%"] < 5)].sort_values("deger", ascending=False).head(2)
        p22 = rd["p"].head(2).sum() * r2["p"].head(2).sum()
        h22 = rd["h"].head(2).sum() * r2["h"].head(2).sum()
        cifte = {"ayaklar": f"{int(rd['kosu_no'].iloc[0])}+{int(r2['kosu_no'].iloc[0])}",
                 "guvenli": g.to_dict("records"), "surpriz": s.to_dict("records"),
                 "kolon_2x2": {"atlar": f"{'-'.join(map(str, rd['no'].head(2)))} / "
                                        f"{'-'.join(map(str, r2['no'].head(2)))}",
                               "kombinasyon": 4, "model_%": _yuzde(p22),
                               "halk_%": _yuzde(h22)}}

    return {"atlar": atlar, "ikili_guvenli": _kayit(guvenli),
            "ikili_surpriz": _kayit(sur), "ikili_kolonlar": kolonlar,
            "cifte": cifte, "surpriz_agf_esigi": agf_esigi}


# --- şablon (LLM yokken) ------------------------------------------------------------
def _ikili_satir(x):
    s = f"{x['komb']} ({x['atlar']}) model %{x['model_%']}"
    if x.get("halk_%") is not None:
        s += f", halk %{x['halk_%']}, değer {x['deger']}"
    return s


def sablon_metin(a: dict) -> dict:
    out = {"ikili_guvenli": "; ".join(_ikili_satir(x) for x in a["ikili_guvenli"][:2]),
           "ikili_surpriz": ("; ".join(_ikili_satir(x) for x in a["ikili_surpriz"][:2])
                             or "Halkın küçümsediği belirgin ikili yok.")}
    kol = []
    for k in a["ikili_kolonlar"]:
        t = f"{k['tur']}: {k['atlar']} ({k['kombinasyon']} komb.) tutma %{k['model_%']}"
        if k.get("halk_%"):
            t += f" (halk %{k['halk_%']})"
        kol.append(t)
    out["ikili_kolon"] = " | ".join(kol)
    c = a.get("cifte")
    if c:
        parca = [f"Güvenli {x['komb']} %{x['model_%']}" for x in c["guvenli"][:1]]
        parca.append(f"2x2 {c['kolon_2x2']['atlar']} %{c['kolon_2x2']['model_%']}")
        parca += [f"Sürpriz {x['komb']} %{x['model_%']} (değer {x['deger']})"
                  for x in c["surpriz"][:1]]
        out["cifte"] = f"{c['ayaklar']}. koşular — " + "; ".join(parca)
    else:
        out["cifte"] = "Son koşu — çifte yok."
    return out


# --- LLM ------------------------------------------------------------------------------
SISTEM = (
    "Sen TJK at yarışı tahmin tablosunu yorumlayan, Türkçe yazan bir analistsin. "
    "Her koşu için hesaplanmış adaylar JSON olarak verilir. Alanlar: model_% = bizim "
    "modelin olasılığı, halk_% = AGF'den türetilen halk olasılığı, deger = model_%/halk_% "
    "(1'in üstü: halk küçümsüyor, isabet ederse ödeme yüksek olur). "
    "KURALLAR: 1) Yalnızca verilen kombinasyonları ve rakamları kullan; yeni at, yüzde "
    "veya oran uydurma. 2) Her koşu için 4 alan yaz, her biri en fazla 2 cümle:\n"
    " - ikili_guvenli: en olası 1-2 İkili kombinasyonu ve yüzdesi (düşük ödeme beklenir).\n"
    " - ikili_surpriz: değeri yüksek İkili(ler); neden sürpriz olduğunu halk_% ile açıkla. "
    "Aday yoksa açıkça 'belirgin sürpriz yok' de.\n"
    " - ikili_kolon: 3/4/5 atlı kolonları karşılaştır (kombinasyon sayısı vs tutma %); "
    "örn. '5 atı oynarsan 10 kombinasyonla %X tutar' gibi; değer kolonu varsa farkını söyle.\n"
    " - cifte: güvenli çifte, 2x2 kolon ve varsa sürpriz çifte; cifte null ise 'çifte yok'.\n"
    "3) Kazanç vaat etme, 'kesin' deme; yüzdeleri mutlaka yaz. "
    "Çıktı yalnızca geçerli JSON: {\"<kosu_no>\": {\"ikili_guvenli\": \"...\", "
    "\"ikili_surpriz\": \"...\", \"ikili_kolon\": \"...\", \"cifte\": \"...\"}}"
)


def api_anahtari(st_secrets=None) -> str | None:
    if st_secrets is not None:
        try:
            k = st_secrets.get("NVIDIA_API_KEY")
            if k:
                return k
        except Exception:
            pass
    return os.environ.get("NVIDIA_API_KEY")


def llm_yorumla(kosu_adaylari: dict, api_key: str, model: str | None = None,
                baglam: str = "", timeout: int = 180) -> dict:
    """kosu_adaylari: {kosu_no: adaylar(...)} -> {kosu_no: {ikili_guvenli, ...}}"""
    model = model or os.environ.get("NVIDIA_MODEL", VARSAYILAN_MODEL)
    girdi = json.dumps({str(k): v for k, v in kosu_adaylari.items()},
                       ensure_ascii=False, default=str)
    govde = {
        "model": model,
        "messages": [{"role": "system", "content": SISTEM},
                     {"role": "user", "content": f"{baglam}\n\nAdaylar:\n{girdi}"}],
        "temperature": 0.3, "top_p": 0.9, "max_tokens": 8192,
    }
    r = requests.post(NIM_URL, json=govde, timeout=timeout,
                      headers={"Authorization": f"Bearer {api_key}",
                               "Accept": "application/json"})
    r.raise_for_status()
    icerik = r.json()["choices"][0]["message"].get("content") or ""
    m = re.search(r"\{.*\}", icerik, re.S)
    if not m:
        raise ValueError("LLM JSON döndürmedi: " + icerik[:200])
    veri = json.loads(m.group())
    return {int(k): v for k, v in veri.items() if str(k).isdigit()}
