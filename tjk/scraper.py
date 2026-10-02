# -*- coding: utf-8 -*-
"""
TJK veri çekme modülü.

İki kaynak desteklenir:
  1) medya-cdn.tjk.org üzerindeki günlük CSV dosyaları (tercih edilen, hızlı)
  2) tjk.org HTML sayfaları (CSV yoksa yedek)

Ana fonksiyonlar:
  get_cities(date)            -> o gün koşu olan hipodromlar [(sehir_id, sehir_adi), ...]
  get_program(date, city)     -> günlük yarış programı DataFrame (at bazında satırlar)
  get_results(date, city)     -> günlük yarış sonuçları DataFrame (bitiş sırası dahil)
  collect_range(start, end)   -> tarih aralığı için tüm veriyi indirip birleştirir
"""
from __future__ import annotations

import io
import re
import time
import datetime as dt
from urllib.parse import quote

import pandas as pd
import requests
from bs4 import BeautifulSoup

BASE = "https://www.tjk.org"
CDN = "https://medya-cdn.tjk.org/raporftp/TJKPDF"

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                  "(KHTML, like Gecko) Chrome/124.0 Safari/537.36",
    "Accept-Language": "tr-TR,tr;q=0.9",
}

SESSION = requests.Session()
SESSION.headers.update(HEADERS)

# ---------------------------------------------------------------------------
# Yurt içi hipodrom filtresi
# ---------------------------------------------------------------------------
# ÖNEMLİ: Ülke adı kara listesi güvenilir değil (yeni ülke eklenince sızıyor).
# Bunun yerine bilinen Türkiye hipodromlarından oluşan BEYAZ LİSTE kullanıyoruz.
# Listede olmayan her şey yurt dışı sayılır ve atlanır; bilinmeyen isimler
# uyarı olarak yazdırılır ki gerçek bir Türkiye sahası gözden kaçmasın.
YURTICI_HIPODROMLAR = {
    "adana", "ankara", "antalya", "bursa", "diyarbakır", "elazığ",
    "istanbul", "izmir", "kocaeli", "mardin", "şanlıurfa",
}

_bilinmeyen_uyarildi: set[str] = set()


def _normalize_city(name: str) -> str:
    """'İstanbul (50. Y.G.)' -> 'istanbul' gibi sadeleştirir."""
    s = re.sub(r"\(.*?\)", "", str(name))          # parantez içi (50. Y.G.)
    s = s.strip().lower()
    s = s.replace("i̇", "i").replace("İ", "i")      # Türkçe nokta sorunları
    return s.strip()


def is_domestic(city_name: str) -> bool:
    """Şehir adı bir Türkiye hipodromu mu?"""
    norm = _normalize_city(city_name)
    if norm in YURTICI_HIPODROMLAR:
        return True
    # "Karma" birden fazla sahanın koşularını birleştiren sanal listedir;
    # dahil edilirse aynı yarışlar iki kez sayılır.
    if norm not in ("karma",) and norm not in _bilinmeyen_uyarildi:
        _bilinmeyen_uyarildi.add(norm)
        print(f"  [bilgi] '{city_name}' yurt dışı sayıldı ve atlandı. "
              f"Türkiye sahasıysa YURTICI_HIPODROMLAR listesine ekleyin.")
    return False


# Geriye dönük uyumluluk için bırakıldı (artık is_domestic kullanılıyor)
YURTDISI_PATTERN = re.compile(
    r"(YD|ABD|Krallık|Fransa|Afrika|Karma|Avustralya|İrlanda|Hong\s*Kong)",
    re.IGNORECASE)


def _tr_date(date: dt.date) -> str:
    return date.strftime("%d/%m/%Y")


def _dot_date(date: dt.date) -> str:
    return date.strftime("%d.%m.%Y")


def _iso_date(date: dt.date) -> str:
    return date.strftime("%Y-%m-%d")


def _get(url: str, **kw) -> requests.Response:
    r = SESSION.get(url, timeout=30, **kw)
    r.raise_for_status()
    return r


# ---------------------------------------------------------------------------
# Şehir listesi
# ---------------------------------------------------------------------------

def get_cities(date: dt.date) -> list[tuple[int, str]]:
    """O gün koşu olan hipodromları (SehirId, SehirAdi) döndürür."""
    url = f"{BASE}/TR/yarissever/Info/Page/GunlukYarisProgrami?QueryParameter_Tarih={quote(_tr_date(date))}"
    html = _get(url).text
    cities = []
    for m in re.finditer(r"SehirId=(\d+)[^\"']*SehirAdi=([^&\"']+)", html):
        sid = int(m.group(1))
        name = requests.utils.unquote(m.group(2)).replace("+", " ").strip()
        if (sid, name) not in cities:
            cities.append((sid, name))
    return cities


# ---------------------------------------------------------------------------
# CSV kaynağı
# ---------------------------------------------------------------------------

def _csv_url(date: dt.date, city_name: str, kind: str) -> str:
    # kind: "GunlukYarisProgrami" | "GunlukYarisSonuclari"
    return (f"{CDN}/{date.year}/{_iso_date(date)}/CSV/{kind}/"
            f"{_dot_date(date)}-{city_name}-{kind}-TR.csv")


def _read_tjk_csv(content: bytes) -> pd.DataFrame | None:
    """TJK CSV'leri genelde windows-1254 + ';' ayraçlıdır; esnek dener."""
    for enc in ("windows-1254", "iso-8859-9", "utf-8-sig", "utf-8"):
        for sep in (";", ","):
            try:
                df = pd.read_csv(io.BytesIO(content), encoding=enc, sep=sep,
                                 engine="python", on_bad_lines="skip")
                if df.shape[1] >= 5:
                    return df
            except Exception:
                continue
    return None


def fetch_csv(date: dt.date, city_name: str, kind: str) -> pd.DataFrame | None:
    try:
        r = _get(_csv_url(date, city_name, kind))
        if r.status_code == 200 and len(r.content) > 200:
            return _read_tjk_csv(r.content)
    except requests.RequestException:
        pass
    return None


# ---------------------------------------------------------------------------
# HTML kaynağı (asıl güvenilir yol — kolonlar sabit)
# ---------------------------------------------------------------------------

RACE_HEADER_RE = re.compile(
    r"(\d+)\.\s*Koşu\s+(\d{2}[.:]\d{2})", re.IGNORECASE)
DIST_SURFACE_RE = re.compile(r"(\d{3,4})\s*(Çim|Kum|Sentetik)", re.IGNORECASE)
# "Ikramiye:1.)730.000 t" -> 730.000  (sayfa genelinden sırayla toplanır)
IKRAMIYE_SAYFA_RE = re.compile(r"[Iİ]kramiye\s*:?\s*1\.\)\s*([\d.,]+)")


def _parse_city_page(html: str, date: dt.date, city_name: str,
                     is_result: bool) -> pd.DataFrame:
    """Bir hipodromun günlük program/sonuç sayfasını at-satırlarına çevirir."""
    soup = BeautifulSoup(html, "lxml")
    rows: list[dict] = []

    # --- Koşu başlıklarını topla ---------------------------------------
    # DİKKAT: Sayfada her koşu için birden çok eşleşme var. Menüdeki kısa
    # link ("1. Koşu 14.00") mesafe/pist İÇERMEZ; asıl başlık ise
    # "... 1500 Kum" bilgisini taşır. Bu yüzden koşu numarası başına
    # EN UZUN metni seçiyoruz, ilkini değil.
    aday: dict[int, str] = {}
    for el in soup.find_all(["h2", "h3", "div", "span", "td", "p"]):
        if el.find("table") is not None:
            continue
        txt = el.get_text(" ", strip=True)
        if not txt or len(txt) > 800:
            continue
        m = RACE_HEADER_RE.search(txt)
        if not m:
            continue
        no = int(m.group(1))
        if no not in aday or len(txt) > len(aday[no]):
            aday[no] = txt

    race_meta = []
    for no in sorted(aday):
        txt = aday[no]
        d = DIST_SURFACE_RE.search(txt)
        saat = RACE_HEADER_RE.search(txt)
        race_meta.append({
            "kosu_no": no,
            "saat": saat.group(2).replace(".", ":") if saat else None,
            "mesafe": int(d.group(1)) if d else None,
            "pist": d.group(2).title() if d else None,
            "kosu_cinsi": txt[:800],  # şart+ikramiye sığsın
        })

    # --- İkramiyeleri topla ----------------------------------------------
    # Ödül bilgisi başlıkla aynı elemanda DEĞİL; sayfa genelinden sırayla
    # toplayıp koşulara indeks bazında eşliyoruz.
    sayfa_metni = soup.get_text(" ", strip=True)
    ikramiyeler = IKRAMIYE_SAYFA_RE.findall(sayfa_metni)

    # --- At tablolarını ayrıştır ----------------------------------------
    horse_tables = []
    for t in soup.find_all("table"):
        head = t.get_text(" ", strip=True)[:400]
        if "At İsmi" in head or "At Ismi" in head:
            horse_tables.append(t)

    for idx, table in enumerate(horse_tables):
        meta = race_meta[idx] if idx < len(race_meta) else {
            "kosu_no": idx + 1, "saat": None, "mesafe": None,
            "pist": None, "kosu_cinsi": None}
        try:
            df = pd.read_html(io.StringIO(str(table)))[0]
        except ValueError:
            continue
        df.columns = [str(c).strip() for c in df.columns]

        ikramiye = None
        if len(ikramiyeler) == len(horse_tables) and idx < len(ikramiyeler):
            try:
                ikramiye = float(ikramiyeler[idx].replace(".", "").replace(",", "."))
            except ValueError:
                ikramiye = None

        for _, r in df.iterrows():
            row = dict(meta)
            row["ikramiye_1"] = ikramiye
            row.update({str(k): r[k] for k in df.columns})
            row["tarih"] = _iso_date(date)
            row["sehir"] = city_name
            row["yurtdisi"] = not is_domestic(city_name)
            row["sonuc_mu"] = is_result
            rows.append(row)

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    # --- Kolon adlarını normalize et ------------------------------------
    # Program sayfası "Antrenör", sonuç sayfası "Antrenörü" diyor. Aynı
    # varlık iki farklı isimle gelirse antrenör istatistikleri hesaplanamaz.
    yeniden = {}
    for c in out.columns:
        lc = str(c).strip().lower()
        if lc.startswith("antrenör"):
            yeniden[c] = "Antrenör"
        elif "at ismi" in lc or "at i̇smi" in lc:
            yeniden[c] = "At İsmi"
    out = out.rename(columns=yeniden)

    # --- İsimleri temizle -------------------------------------------------
    for c in ("At İsmi", "Jokey", "Antrenör", "Sahip"):
        if c in out.columns:
            out[c] = out[c].map(clean_name)

    return out


def _city_page_url(date: dt.date, sehir_id: int, city_name: str, kind: str) -> str:
    return (f"{BASE}/TR/yarissever/Info/Sehir/{kind}?SehirId={sehir_id}"
            f"&QueryParameter_Tarih={quote(_tr_date(date))}"
            f"&SehirAdi={quote(city_name)}&Era=today")


def get_program(date: dt.date, sehir_id: int, city_name: str) -> pd.DataFrame:
    html = _get(_city_page_url(date, sehir_id, city_name, "GunlukYarisProgrami")).text
    df = _parse_city_page(html, date, city_name, is_result=False)
    return df


def get_results_html(date: dt.date, sehir_id: int, city_name: str) -> str:
    """Sonuç sayfasının ham HTML'i. Bahis ödemeleri tablo içinde DEĞİL,
    div.bahisSonucCard elemanlarında olduğu için ham HTML gerekir."""
    return _get(_city_page_url(date, sehir_id, city_name,
                               "GunlukYarisSonuclari")).text


def get_results(date: dt.date, sehir_id: int, city_name: str) -> pd.DataFrame:
    html = _get(_city_page_url(date, sehir_id, city_name, "GunlukYarisSonuclari")).text
    df = _parse_city_page(html, date, city_name, is_result=True)
    return df


# ---------------------------------------------------------------------------
# Toplu indirme
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Metin temizleme
# ---------------------------------------------------------------------------
# TJK tablolarında at/jokey isimlerine ekipman açıklamaları yapışık geliyor:
#   "TLOS(5) KKulaklık takılacağını ifade eder."  -> "TLOS"
#   "S.ÇELİK APApranti"                            -> "S.ÇELİK"
# Bu temizlik kritik: isim tutarsız olursa atın/jokeyin geçmişi birleşmez.

_EK_METINLER = [
    "Kapalı gözlük", "Kulaklık", "Dilinin", "Ring mahalinden", "Start makinasına",
    "Göz koruyucu", "Yanak peluşu", "Tek gözü", "Tercihli Start", "Apranti",
    "Koşmaz", "Satılık", "Fazla Kilo", "Satış formu", "eküridir",
]
_EK_RE = re.compile("|".join(re.escape(x) for x in _EK_METINLER))
# Açıklama metni kesildikten sonra geriye kalan yapışık kodlar
# Not: tek harflik "K" kodu listede YOK — "GÜÇLÜ TÜRK" gibi isimlerin
# sonundaki K harfini yiyordu. Parantezli biçim zaten o vakayı çözüyor.
_KOD_SON_RE = re.compile(r"\s*(SKG|GKR|TGK|KG|SK|DB|ÖG|YP|AP|DS)$")


def clean_name(value) -> str:
    """At / jokey / antrenör / sahip isimlerini yapışık açıklamalardan arındırır."""
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    txt = str(value).replace("\xa0", " ").strip()
    # Satış koşularında fiyat isme yapışıyor: "BEŞİRHAT t1.500.000,00 TL"
    txt = re.sub(r"\s*t?\s*\d[\d.,]*\s*TL.*$", "", txt)
    # "AT ADI(3)" -> parantezli start numarasını ve sonrasını at
    txt = re.split(r"\(", txt)[0]
    # Ekipman/durum açıklaması varsa oradan kes
    m = _EK_RE.search(txt)
    if m:
        txt = txt[:m.start()]
    # Geriye kalan yapışık kodu temizle ("MEYDAN OKUKG" -> "MEYDAN OKU").
    # SADECE açıklama metni kesildiyse ve tek geçişte: aksi halde gerçek
    # isimlerin son harfleri kurban gider.
    txt = txt.strip()
    if m:
        txt = _KOD_SON_RE.sub("", txt).strip()
    return re.sub(r"\s+", " ", txt).strip()


def sanitize_for_parquet(df: pd.DataFrame) -> pd.DataFrame:
    """Karma tipli kolonları parquet'in yazabileceği hale getirir.

    TJK sayfalarında aynı kolon bazı günler metin ('59'), bazı günler sayı (59)
    olarak gelebiliyor. Aylık birleştirmede kolon 'object' tipine düşüyor ve
    pyarrow "Expected bytes, got a 'int' object" hatası veriyor.

    Çözüm: object tipli kolonları metne çeviriyoruz. Sayısal ayrıştırma zaten
    features.py içinde yapıldığı için bilgi kaybı olmaz.
    """
    d = df.copy()
    for c in d.columns:
        if d[c].dtype == object:
            d[c] = d[c].map(lambda v: None if pd.isna(v) else str(v))
    return d


def collect_range(start: dt.date, end: dt.date, kind: str = "results",
                  only_domestic: bool = True, sleep: float = 1.0,
                  progress: bool = True) -> pd.DataFrame:
    """Tarih aralığındaki tüm hipodromların verisini indirir ve birleştirir.

    kind: "results" (model eğitimi için) veya "program"
    only_domestic: True ise yurt dışı ayakları atlar
    """
    frames = []
    day = start
    fetch = get_results if kind == "results" else get_program
    while day <= end:
        try:
            cities = get_cities(day)
        except Exception as e:
            if progress:
                print(f"{day}: şehir listesi alınamadı ({e})")
            day += dt.timedelta(days=1)
            continue
        for sid, name in cities:
            if only_domestic and not is_domestic(name):
                continue
            try:
                df = fetch(day, sid, name)
                if not df.empty:
                    frames.append(df)
                    if progress:
                        print(f"{day} {name}: {len(df)} satır")
            except Exception as e:
                if progress:
                    print(f"{day} {name}: hata ({e})")
            time.sleep(sleep)  # siteye nazik davran
        day += dt.timedelta(days=1)
    if not frames:
        return pd.DataFrame()
    return sanitize_for_parquet(pd.concat(frames, ignore_index=True))
