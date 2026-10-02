# -*- coding: utf-8 -*-
"""Geçmiş yarış sonuçlarını indirir.

Kullanım:
    python scripts/collect_history.py 2021-01-01 2026-08-12

Özellikler:
  - Nereden çalıştırılırsa çalıştırılsın veriyi PROJE KÖKÜNDEKİ data/raw'a yazar
  - Ay ay kaydeder: kesinti olursa baştan başlamaz, kaldığı yerden devam eder
  - Tamamlanmış aylar atlanır (yeniden çalıştırmak güvenlidir)

Not: Şehre göre filtreleme YAPILMAZ. Atlar hipodrom değiştirdiği için bir atın
form geçmişini kurabilmek adına tüm yurt içi hipodromlar toplanır.
"""
import sys
import pathlib
import datetime as dt

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.scraper import collect_range  # noqa: E402

OUT_DIR = ROOT / "data" / "raw"


def month_ranges(start: dt.date, end: dt.date):
    """Aralığı aylık dilimlere böler."""
    cur = start.replace(day=1)
    while cur <= end:
        nxt = (cur.replace(day=28) + dt.timedelta(days=4)).replace(day=1)
        yield max(cur, start), min(nxt - dt.timedelta(days=1), end)
        cur = nxt


def main():
    if len(sys.argv) < 3:
        sys.exit("Kullanim: python scripts/collect_history.py BASLANGIC BITIS "
                 "(or. 2021-01-01 2026-08-12)")

    start = dt.date.fromisoformat(sys.argv[1])
    end = dt.date.fromisoformat(sys.argv[2])
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    chunks = list(month_ranges(start, end))
    print(f"{len(chunks)} aylik dilim indirilecek. Cikti: {OUT_DIR}\n")

    for i, (a, b) in enumerate(chunks, 1):
        path = OUT_DIR / f"results_{a:%Y-%m}.parquet"
        if path.exists():
            print(f"[{i}/{len(chunks)}] {a:%Y-%m} zaten var, atlaniyor.")
            continue

        print(f"[{i}/{len(chunks)}] {a} -> {b} indiriliyor...")
        try:
            df = collect_range(a, b, kind="results", only_domestic=True)
        except KeyboardInterrupt:
            print("\nKullanici durdurdu. Tamamlanan aylar korundu, "
                  "ayni komutu tekrar calistirarak devam edebilirsiniz.")
            return
        except Exception as e:
            print(f"  !! {a:%Y-%m} atlandi: {e}")
            continue

        if df.empty:
            print(f"  {a:%Y-%m}: veri yok.")
            continue

        df.to_parquet(path)
        print(f"  OK {len(df):,} satir -> {path.name}\n")

    files = sorted(OUT_DIR.glob("results_*.parquet"))
    total = sum(len(pd.read_parquet(f)) for f in files)
    print(f"\nTamamlandi. {len(files)} dosya, toplam {total:,} satir.")
    print("Siradaki adim:  python scripts/train.py")


if __name__ == "__main__":
    main()
