# -*- coding: utf-8 -*-
"""Günlük yarış sonuçları — EDA için tek günlük çekici.

Kaynak sırası:
  1) TJK CDN CSV  (medya-cdn.tjk.org/.../GunlukYarisSonuclari/...csv)
  2) tjk.org HTML sonuç sayfası (SehirId gerekiyorsa)

Çıktı: koşu başına bir satır = bir at. Normalize kolonlar:
  kosu_no, sira, no, at, yas, jokey, sahip, antrenor, derece, ganyan, agf,
  fark, hp, baba, anne, + koşu bilgisi (saat, mesafe, pist, kosu_bilgi)
"""
from __future__ import annotations

import csv
import datetime as dt
import io
import re
import unicodedata
from urllib.parse import quote

import pandas as pd
import requests

HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                   "(KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept-Language": "tr-TR,tr;q=0.9",
}

CDN_URL = ("https://medya-cdn.tjk.org/raporftp/TJKPDF/{y}/{iso}/CSV/"
           "GunlukYarisSonuclari/{dmy}-{sehir}-GunlukYarisSonuclari-TR.csv")
HTML_URL = ("https://www.tjk.org/TR/YarisSever/Info/Sehir/GunlukYarisSonuclari"
            "?SehirId={sid}&QueryParameter_Tarih={tarih}&SehirAdi={sehir}&Era=past")

KOSU_RE = re.compile(r"^\W*(\d{1,2})\s*\.?\s*ko[sş]u", re.I)


# --- yardımcılar -------------------------------------------------------------
def fold(s: str) -> str:
    """Türkçe karakterleri ASCII'ye indirger, küçük harf yapar."""
    s = str(s).replace("İ", "I").replace("ı", "i")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).lower().strip()


def sehir_sade(ad: str) -> str:
    """'Bursa (17. Y.G.)' -> 'Bursa'"""
    return re.sub(r"\s*\(.*?\)\s*", "", str(ad)).strip()


def _sayi(x):
    if x is None:
        return None
    s = str(x).strip().replace("%", "")
    if not s or s in {"-", "--"}:
        return None
    s = s.replace(".", "").replace(",", ".") if s.count(",") == 1 else s
    try:
        return float(s)
    except ValueError:
        m = re.search(r"\d+(?:[.,]\d+)?", s)
        return float(m.group().replace(",", ".")) if m else None


KOLON_ESLE = [  # (hedef, anahtar kelimeler - fold edilmiş, ilk eşleşen kazanır)
    ("sira", ["sira", "derece sira", "kum."]),
    ("no", ["at no", "no"]),
    ("at", ["at ismi", "at adi", "at"]),
    ("yas", ["yas"]),
    ("baba", ["baba", "orijin"]),
    ("anne", ["anne"]),
    ("siklet", ["siklet", "kilo"]),
    ("jokey", ["jokey"]),
    ("sahip", ["sahib", "sahip"]),
    ("antrenor", ["antren"]),
    ("derece", ["derece"]),
    ("ganyan", ["gny", "ganyan"]),
    ("agf", ["agf"]),
    ("start", ["st", "start"]),
    ("fark", ["fark"]),
    ("gec_cikis", ["g.cik", "g. cik", "gec"]),
    ("hp", ["hp"]),
]


def kolonlari_normalize(df: pd.DataFrame) -> pd.DataFrame:
    yeni, kullanilan = {}, set()
    for c in df.columns:
        f = fold(c)
        hedef = None
        for h, anahtarlar in KOLON_ESLE:
            if h in kullanilan:
                continue
            for a in anahtarlar:
                # kısa anahtarlar tam eşleşmeli, uzunlar içerme
                if (len(a) <= 3 and f == a) or (len(a) > 3 and a in f):
                    hedef = h
                    break
            if hedef:
                break
        # "Derece" kolonu sıra değil zaman; "S" ise sıra
        if f in {"s", "s."}:
            hedef = "sira" if "sira" not in kullanilan else None
        if hedef:
            yeni[c] = hedef
            kullanilan.add(hedef)
    out = df.rename(columns=yeni)
    return out.loc[:, ~out.columns.duplicated()]


def kosu_bilgisi(satir: str) -> dict:
    d = {"kosu_bilgi": satir.strip(" ;,")}
    m = re.search(r"(\d{1,2}[.:]\d{2})", satir)
    if m:
        d["saat"] = m.group(1).replace(".", ":")
    m = re.search(r"(\d{3,4})\s*m?\b.*?\b(çim|cim|kum|sentetik)", satir, re.I)
    if m:
        d["mesafe"] = int(m.group(1))
        d["pist"] = m.group(2).capitalize()
    return d


def _temizle(df: pd.DataFrame) -> pd.DataFrame:
    df = kolonlari_normalize(df)
    for c in ("ganyan", "agf", "hp", "yas", "siklet"):
        if c in df.columns:
            df[c] = df[c].map(_sayi)
    for c in ("no", "sira"):
        if c in df.columns:
            df[c] = pd.to_numeric(df[c].map(_sayi), errors="coerce").astype("Int64")
    if "at" in df.columns:
        # at isminin sonuna eklenen ekipman/satış notlarını at: "KARAYEL (KG) DB"
        df["at"] = (df["at"].astype(str)
                    .str.replace(r"\s*\(.*?\)", "", regex=True)
                    .str.replace(r"\s+(KG|DB|SK|SKG|GKR|ÖG|BB|DS|KBG|TK)(\s|$).*$", "",
                                 regex=True)
                    .str.strip())
    for c in ("jokey", "sahip", "antrenor"):
        if c in df.columns:
            df[c] = df[c].astype(str).str.strip().replace({"nan": None, "": None})
    # sıra kolonu yoksa: sonuç tablosu bitiş sırasıyla gelir
    if "sira" not in df.columns or df["sira"].isna().all():
        df["sira"] = df.groupby("kosu_no").cumcount() + 1
    df["kazandi"] = (df["sira"] == 1).astype(int)
    df["ilk3"] = (df["sira"] <= 3).astype(int)
    return df


# --- CSV ---------------------------------------------------------------------
def _coz(b: bytes) -> str:
    for enc in ("utf-8-sig", "cp1254", "utf-16", "iso-8859-9"):
        try:
            t = b.decode(enc)
            if "Jokey" in t or "JOKEY" in t or "jokey" in t:
                return t
        except UnicodeDecodeError:
            continue
    return b.decode("cp1254", errors="replace")


def csv_parse(metin: str) -> pd.DataFrame:
    satirlar = [s for s in metin.splitlines() if s.strip()]
    if not satirlar:
        return pd.DataFrame()
    ayrac = ";" if sum(s.count(";") for s in satirlar[:50]) >= \
        sum(s.count(",") for s in satirlar[:50]) else ","
    kayitlar, baslik, kosu, bilgi = [], None, None, {}
    for satir in satirlar:
        m = KOSU_RE.match(satir)
        if m:
            kosu, bilgi, baslik = int(m.group(1)), kosu_bilgisi(satir), None
            continue
        hucre = [h.strip() for h in next(csv.reader([satir], delimiter=ayrac))]
        f = fold(satir)
        if "jokey" in f and ("at" in f):
            baslik = hucre
            continue
        if baslik and kosu is not None:
            dolu = [h for h in hucre if h]
            if len(dolu) >= max(4, len(baslik) // 2):
                kayit = dict(zip(baslik, hucre + [""] * (len(baslik) - len(hucre))))
                kayit.update({"kosu_no": kosu, **bilgi})
                kayitlar.append(kayit)
    if not kayitlar:
        return pd.DataFrame()
    return _temizle(pd.DataFrame(kayitlar))


def _cdn_cek(tarih: dt.date, sehir: str, timeout=20) -> pd.DataFrame:
    sade = sehir_sade(sehir)
    adaylar = dict.fromkeys([sade, sade.title(), fold(sade).title(),
                             sade.replace("İ", "I")])
    for ad in adaylar:
        url = CDN_URL.format(y=tarih.year, iso=tarih.isoformat(),
                             dmy=tarih.strftime("%d.%m.%Y"), sehir=quote(ad))
        try:
            r = requests.get(url, headers=HEADERS, timeout=timeout)
        except requests.RequestException:
            continue
        if r.status_code == 200 and len(r.content) > 200:
            df = csv_parse(_coz(r.content))
            if not df.empty:
                df["kaynak"] = "cdn_csv"
                return df
    return pd.DataFrame()


# --- HTML ---------------------------------------------------------------------
def html_parse(html: str) -> pd.DataFrame:
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html, "lxml")
    parcalar = []
    for i, tablo in enumerate(soup.find_all("table")):
        basliklar = " ".join(th.get_text(" ", strip=True) for th in tablo.find_all("th"))
        if "jokey" not in fold(basliklar):
            continue
        # koşu numarası: tablodan önceki en yakın "N. Koşu" metni
        kosu, bilgi = None, {}
        for onceki in tablo.find_all_previous(
                string=re.compile(r"\d{1,2}\s*\.?\s*Ko[sş]u", re.I), limit=1):
            blok = onceki.find_parent().get_text(" ", strip=True)
            m = re.search(r"(\d{1,2})\s*\.?\s*Ko[sş]u", blok, re.I)
            if m:
                kosu, bilgi = int(m.group(1)), kosu_bilgisi(blok)
        try:
            df = pd.read_html(io.StringIO(str(tablo)), thousands=None)[0]
        except ValueError:
            continue
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[-1] for c in df.columns]
        df["kosu_no"] = kosu if kosu is not None else i + 1
        for k, v in bilgi.items():
            df[k] = v
        parcalar.append(df)
    if not parcalar:
        return pd.DataFrame()
    return _temizle(pd.concat(parcalar, ignore_index=True))


def _html_cek(tarih: dt.date, sehir: str, sid, timeout=25) -> pd.DataFrame:
    if sid is None:
        return pd.DataFrame()
    url = HTML_URL.format(sid=sid, tarih=tarih.strftime("%d/%m/%Y"),
                          sehir=quote(sehir_sade(sehir)))
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
    except requests.RequestException:
        return pd.DataFrame()
    df = html_parse(r.text)
    if not df.empty:
        df["kaynak"] = "html"
    return df


# --- dış arayüz ---------------------------------------------------------------
def gunluk_sonuclar(tarih: dt.date, sehir: str, sid=None) -> pd.DataFrame:
    """Tek hipodromun o günkü sonuçları. Boş DF = sonuç yok / henüz koşulmadı."""
    df = _cdn_cek(tarih, sehir)
    if df.empty:
        df = _html_cek(tarih, sehir, sid)
    if not df.empty:
        df.insert(0, "hipodrom", sehir_sade(sehir))
        df.insert(0, "tarih", pd.Timestamp(tarih))
    return df
