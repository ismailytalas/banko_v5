# -*- coding: utf-8 -*-
"""Günlük güncelleme: eksik yarış günlerini bulur, indirir, aylık dosyalara ekler.

Kullanım:
    python scripts/update_daily.py              # dünden geriye eksikleri tamamlar
    python scripts/update_daily.py --today      # bugünün sonuçlarını da dene
    python scripts/update_daily.py --train      # güncelleme sonrası modeli de eğit

Neden "dün"? Yarış bittikten sonra komiser kararı, diskalifiye veya itiraz
sonucu sıralama değişebilir. Koşudan hemen sonra çekilen sonuç yanlış kalabilir.
Ertesi sabah çekmek hem güvenli hem yeterlidir.
"""
import sys
import subprocess
import pathlib
import datetime as dt

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.scraper import collect_range, sanitize_for_parquet  # noqa: E402

RAW = ROOT / "data" / "raw"
KEY = ["tarih", "sehir", "kosu_no"]


def last_downloaded_date() -> dt.date | None:
    files = sorted(RAW.glob("results_*.parquet"))
    if not files:
        return None
    son = pd.read_parquet(files[-1], columns=["tarih"])
    if son.empty:
        return None
    return dt.date.fromisoformat(str(son["tarih"].max())[:10])


def merge_into_month(df: pd.DataFrame) -> list[str]:
    """Gelen veriyi ilgili aylık dosyalara ekler, tekrarları temizler."""
    yazilan = []
    df = df.copy()
    df["_ay"] = df["tarih"].astype(str).str[:7]

    for ay, chunk in df.groupby("_ay"):
        chunk = chunk.drop(columns=["_ay"])
        path = RAW / f"results_{ay}.parquet"
        if path.exists():
            eski = pd.read_parquet(path)
            # Aynı güne ait eski kayıtları at, yenisini koy (düzeltmeler için)
            yeni_gunler = set(chunk["tarih"].astype(str))
            eski = eski[~eski["tarih"].astype(str).isin(yeni_gunler)]
            chunk = pd.concat([eski, chunk], ignore_index=True)
        chunk = sanitize_for_parquet(chunk)
        chunk.to_parquet(path)
        yazilan.append(f"{path.name} ({len(chunk):,} satır)")
    return yazilan


def main():
    bugun = dt.date.today()
    dahil_bugun = "--today" in sys.argv
    egit = "--train" in sys.argv

    end = bugun if dahil_bugun else bugun - dt.timedelta(days=1)

    son = last_downloaded_date()
    if son is None:
        sys.exit("Hiç veri yok. Önce: python scripts/collect_history.py "
                 "2021-01-01 2026-08-12")

    start = son + dt.timedelta(days=1)
    print(f"Son indirilen gün : {son}")
    print(f"İndirilecek aralık: {start} → {end}")

    if start > end:
        print("\nVeri zaten güncel, yapılacak bir şey yok.")
    else:
        df = collect_range(start, end, kind="results", only_domestic=True)
        if df.empty:
            print("\nBu aralıkta yarış bulunamadı (tatil/ara olabilir).")
        else:
            print(f"\n{len(df):,} satır indirildi.")
            for w in merge_into_month(df):
                print(f"  güncellendi -> {w}")

    if egit:
        print("\n" + "=" * 50)
        print("Model yeniden eğitiliyor...")
        subprocess.run([sys.executable, str(ROOT / "scripts" / "train.py")],
                       cwd=str(ROOT))


if __name__ == "__main__":
    main()
