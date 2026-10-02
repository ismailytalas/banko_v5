# -*- coding: utf-8 -*-
"""vhs.tjk.org/muhtemeller — İkili, Sıralı İkili, Çifte (ve Ganyan) muhtemel oranları.

Sayfa JavaScript ile çizildiği için düz `requests` boş kabuk döner. Bu yüzden
Playwright (headless Chromium) ile açılır; sayfadaki tablolar + görünür metin
okunur. Streamlit'in olay döngüsüyle çakışmaması için tarayıcı ayrı bir
**alt süreçte** çalışır:

    python -m tjk.muhtemeller 02/10/2026 BURSA 1 2 3 --debug

`--debug` ham tablo/metin/JSON çıktısını data/muhtemeller_debug/ altına yazar;
ayrıştırma bir gün bozulursa bu dosyaya bakarak düzeltilir.

Çıktı (her koşu için): {"ikili": [...], "sirali_ikili": [...], "cifte": [...],
"ganyan": [...]} — her kayıt {"a": int, "b": int|None, "oran": float}.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import unicodedata
from pathlib import Path

import pandas as pd

URL = "https://vhs.tjk.org/muhtemeller/?date={tarih}&hipodromkey={key}&no={no}"
BAHISLER = ("ikili", "sirali_ikili", "cifte", "ganyan")
DEBUG_DIR = Path("data/muhtemeller_debug")


def _fold(s: str) -> str:
    s = str(s).replace("İ", "I").replace("ı", "i")
    s = unicodedata.normalize("NFKD", s)
    return "".join(c for c in s if not unicodedata.combining(c)).upper().strip()


def hipodrom_key(sehir: str) -> str:
    """'İzmir (40. Y.G.)' -> 'IZMIR', 'Şanlıurfa' -> 'SANLIURFA'"""
    sade = re.sub(r"\s*\(.*?\)\s*", "", str(sehir)).strip()
    return re.sub(r"[^A-Z]", "", _fold(sade))


# --- ayrıştırma (tarayıcıdan bağımsız, test edilebilir) ------------------------
def bahis_turu(baslik: str) -> str | None:
    f = _fold(baslik)
    if "SIRALI" in f and "IKILI" in f:
        return "sirali_ikili"
    if "IKILI" in f:
        return "ikili"
    if "CIFTE" in f:
        return "cifte"
    if "GANYAN" in f and "ALTILI" not in f and "BESLI" not in f:
        return "ganyan"
    return None


def _oran(x: str):
    s = str(x).strip()
    if not re.fullmatch(r"\d{1,6}(?:[.,]\d{1,2})?", s):
        return None
    s = s.replace(",", ".")
    try:
        v = float(s)
    except ValueError:
        return None
    return v if v >= 1.0 else None


def _int(x):
    s = str(x).strip()
    return int(s) if re.fullmatch(r"\d{1,2}", s) else None


KOMB_RE = re.compile(r"(?<!\d)(\d{1,2})\s*[-/]\s*(\d{1,2})(?!\d)")


def tablo_parse(satirlar: list[list[str]], tur: str) -> list[dict]:
    """Bir tabloyu kayıtlara çevirir. İki düzen desteklenir:
    - liste:  ['1-2', '12,50']  veya ['1', '2', '12,50']
    - matris: üst satır at no'ları, sol kolon at no'ları, hücreler oran
    """
    out = []
    if not satirlar:
        return out
    ust = [_int(h) for h in satirlar[0][1:]]
    sol = [_int(r[0]) for r in satirlar[1:] if r]
    matris = (len(satirlar) > 2 and sum(u is not None for u in ust) >= 2
              and sum(s is not None for s in sol) >= 2 and tur != "ganyan")
    if matris:
        for r in satirlar[1:]:
            a = _int(r[0]) if r else None
            if a is None:
                continue
            for b, h in zip(ust, r[1:]):
                o = _oran(h)
                if b is not None and o is not None and a != b:
                    out.append({"a": a, "b": b, "oran": o})
        return out
    for r in satirlar:
        hucre = [str(h).strip() for h in r if str(h).strip()]
        if not hucre:
            continue
        if tur == "ganyan":
            a = _int(hucre[0])
            oranlar = [_oran(h) for h in hucre[1:]]
            oranlar = [o for o in oranlar if o is not None]
            if a is not None and oranlar:
                out.append({"a": a, "b": None, "oran": oranlar[-1]})
            continue
        birlesik = " ".join(hucre)
        m = KOMB_RE.search(birlesik)
        if m:
            sonrasi = birlesik[m.end():].split()
            o = next((_oran(t) for t in sonrasi if _oran(t) is not None), None)
            if o is not None:
                out.append({"a": int(m.group(1)), "b": int(m.group(2)), "oran": o})
            continue
        if len(hucre) >= 3 and _int(hucre[0]) is not None and _int(hucre[1]) is not None:
            o = _oran(hucre[2])
            if o is not None:
                out.append({"a": _int(hucre[0]), "b": _int(hucre[1]), "oran": o})
    return out


def metin_parse(metin: str) -> dict[str, list[dict]]:
    """Tablo bulunamazsa: görünür metni başlıklara göre bölüp 'a-b oran' ara."""
    sonuc = {b: [] for b in BAHISLER}
    tur = None
    for satir in metin.splitlines():
        t = bahis_turu(satir)
        if t and len(satir) < 40:
            tur = t
            continue
        if tur is None or tur == "ganyan":
            continue
        for m in re.finditer(KOMB_RE.pattern + r"\s*[:=]?\s*(\d{1,6}(?:[.,]\d{1,2})?)",
                             satir):
            o = _oran(m.group(3))
            if o:
                sonuc[tur].append({"a": int(m.group(1)), "b": int(m.group(2)), "oran": o})
    return sonuc


def sayfa_parse(tablolar: list[dict], metin: str) -> dict[str, list[dict]]:
    """tablolar: [{'baslik': str, 'satirlar': [[hücre,...],...]}, ...]"""
    sonuc = {b: [] for b in BAHISLER}
    for t in tablolar:
        tur = bahis_turu(t.get("baslik", ""))
        if tur is None and t.get("satirlar"):
            tur = bahis_turu(" ".join(t["satirlar"][0]))
        if tur:
            sonuc[tur].extend(tablo_parse(t.get("satirlar", []), tur))
    if not any(sonuc[b] for b in ("ikili", "sirali_ikili", "cifte")):
        yedek = metin_parse(metin)
        for b in BAHISLER:
            sonuc[b] = sonuc[b] or yedek[b]
    # İkili sırasız: (a,b) ve (b,a) tekilleştir
    gorulen, tek = set(), []
    for r in sonuc["ikili"]:
        k = tuple(sorted((r["a"], r["b"])))
        if k not in gorulen:
            gorulen.add(k)
            tek.append({"a": k[0], "b": k[1], "oran": r["oran"]})
    sonuc["ikili"] = tek
    return sonuc


# --- tarayıcı (alt süreçte çalışır) --------------------------------------------
_JS_TABLOLAR = r"""
() => {
  const baslikBul = (el) => {
    let n = el;
    for (let i = 0; i < 8 && n; i++) {
      let p = n.previousElementSibling;
      while (p) {
        const t = (p.innerText || '').trim();
        if (t && t.length < 60) return t;
        p = p.previousElementSibling;
      }
      n = n.parentElement;
      if (n) {
        const cap = n.querySelector(':scope > h1,:scope > h2,:scope > h3,:scope > h4,:scope > .title,:scope > .header');
        if (cap && (cap.innerText||'').trim().length < 60) return cap.innerText.trim();
      }
    }
    return '';
  };
  return Array.from(document.querySelectorAll('table')).map(t => ({
    baslik: (t.caption ? t.caption.innerText : '') || baslikBul(t),
    satirlar: Array.from(t.rows).map(r => Array.from(r.cells).map(c => c.innerText.trim()))
  }));
}
"""


def _chromium_yolu():
    for aday in (os.environ.get("TJK_CHROMIUM"), shutil.which("chromium"),
                 shutil.which("chromium-browser"), "/opt/pw-browsers/chromium"):
        if aday and os.path.isfile(aday):
            return aday
    return None


def _tarayici_cek(tarih: str, key: str, kosular: list[int], debug=False) -> dict:
    import asyncio
    if sys.platform.startswith("win"):
        asyncio.set_event_loop_policy(asyncio.WindowsProactorEventLoopPolicy())
    from playwright.sync_api import sync_playwright

    sonuc = {}
    with sync_playwright() as p:
        kw = {"headless": True}
        yol = _chromium_yolu()
        if yol:
            kw["executable_path"] = yol
        tarayici = p.chromium.launch(**kw)
        sayfa = tarayici.new_page(locale="tr-TR")
        jsonlar = []

        def _yakala(resp):
            try:
                if "json" in (resp.headers.get("content-type") or ""):
                    jsonlar.append({"url": resp.url, "veri": resp.json()})
            except Exception:
                pass

        sayfa.on("response", _yakala)
        for no in kosular:
            jsonlar.clear()
            url = URL.format(tarih=tarih, key=key, no=no)
            try:
                sayfa.goto(url, wait_until="networkidle", timeout=30000)
                sayfa.wait_for_timeout(1500)
                tablolar = sayfa.evaluate(_JS_TABLOLAR)
                metin = sayfa.inner_text("body")
            except Exception as e:  # tek koşu bozuksa diğerlerine devam
                sonuc[str(no)] = {"hata": str(e)}
                continue
            kayit = sayfa_parse(tablolar, metin)
            kayit["url"] = url
            sonuc[str(no)] = kayit
            if debug:
                DEBUG_DIR.mkdir(parents=True, exist_ok=True)
                (DEBUG_DIR / f"{key}_{tarih.replace('/', '-')}_{no}.json").write_text(
                    json.dumps({"url": url, "tablolar": tablolar, "metin": metin,
                                "xhr_json": jsonlar}, ensure_ascii=False, indent=1),
                    encoding="utf-8")
        tarayici.close()
    return sonuc


def cek(tarih_dmy: str, sehir: str, kosular: list[int], timeout=240,
        debug=False) -> dict:
    """Streamlit'ten çağrılır. Alt süreçte tarayıcıyı çalıştırır.
    Dönüş: {"1": {"ikili": [...], ...}, ...} veya {"_hata": "..."}"""
    key = hipodrom_key(sehir)
    komut = [sys.executable, "-m", "tjk.muhtemeller", tarih_dmy, key,
             *map(str, kosular)] + (["--debug"] if debug else [])
    kok = Path(__file__).resolve().parent.parent
    try:
        r = subprocess.run(komut, capture_output=True, text=True, timeout=timeout,
                           cwd=kok, encoding="utf-8")
    except subprocess.TimeoutExpired:
        return {"_hata": "Zaman aşımı (muhtemeller sayfası yanıt vermedi)"}
    if r.returncode != 0:
        son = (r.stderr or "").strip().splitlines()[-3:]
        return {"_hata": "Tarayıcı çalıştırılamadı: " + " | ".join(son)}
    try:
        return json.loads(r.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return {"_hata": "Çıktı okunamadı: " + r.stdout[-300:]}


# --- model ile birleştirme -----------------------------------------------------
def _harville_sirali(p: dict, a: int, b: int) -> float:
    pa, pb = p.get(a, 0.0), p.get(b, 0.0)
    return pa * pb / max(1e-9, 1 - pa)


def deger_tablosu(odds: dict, pred: pd.DataFrame, kosu_no: int) -> pd.DataFrame:
    """Muhtemel oranları model olasılıklarıyla birleştirir.

    p_model: İkili = H(a,b)+H(b,a); Sıralı = H(a,b); Çifte = p_k(a)*p_{k+1}(b)
    beklenen = p_model * oran  (>1 ise model fiyatı ucuz buluyor)
    """
    rd = pred[pred["kosu_no"] == kosu_no]
    p = dict(zip(rd["no"].astype(int), rd["p_win"].astype(float)))
    sonraki = pred[pred["kosu_no"] == kosu_no + 1]
    p2 = dict(zip(sonraki["no"].astype(int), sonraki["p_win"].astype(float)))
    adlar = dict(zip(rd["no"].astype(int), rd["at"]))
    adlar2 = dict(zip(sonraki["no"].astype(int), sonraki["at"]))
    satirlar = []
    for tur in ("ikili", "sirali_ikili", "cifte"):
        for r in odds.get(tur, []) or []:
            a, b, o = r["a"], r["b"], r["oran"]
            if tur == "ikili":
                pm = _harville_sirali(p, a, b) + _harville_sirali(p, b, a)
                ad = f"{adlar.get(a, '?')} / {adlar.get(b, '?')}"
            elif tur == "sirali_ikili":
                pm = _harville_sirali(p, a, b)
                ad = f"{adlar.get(a, '?')} → {adlar.get(b, '?')}"
            else:
                if not p2:
                    continue
                pm = p.get(a, 0.0) * p2.get(b, 0.0)
                ad = f"{adlar.get(a, '?')} | {adlar2.get(b, '?')}"
            satirlar.append({"Bahis": {"ikili": "İkili", "sirali_ikili": "Sıralı İkili",
                                        "cifte": "Çifte"}[tur],
                             "Kombinasyon": f"{a}-{b}", "Atlar": ad, "Oran": o,
                             "Piyasa %": 100 / o, "Model %": 100 * pm,
                             "Beklenen": pm * o})
    df = pd.DataFrame(satirlar)
    if df.empty:
        return df
    return df.sort_values(["Bahis", "Beklenen"], ascending=[True, False])


# --- CLI ----------------------------------------------------------------------
if __name__ == "__main__":
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    if len(args) < 3:
        print("kullanım: python -m tjk.muhtemeller DD/MM/YYYY HIPODROMKEY NO [NO...] "
              "[--debug]", file=sys.stderr)
        sys.exit(2)
    tarih, key, *nolar = args
    cikti = _tarayici_cek(tarih, key, [int(n) for n in nolar], "--debug" in sys.argv)
    print(json.dumps(cikti, ensure_ascii=False))
