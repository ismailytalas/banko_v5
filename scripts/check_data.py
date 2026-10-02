# -*- coding: utf-8 -*-
"""İndirilen veriyi kalite kontrolden geçirir.

Kullanım:
    python scripts/check_data.py                 # tüm indirilmiş aylar
    python scripts/check_data.py 2021-01         # tek ay
"""
import sys
import pathlib

import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.features import add_label_from_results, build_features  # noqa: E402
from tjk.scraper import is_domestic  # noqa: E402

RAW = ROOT / "data" / "raw"

pd.set_option("display.width", 200)
pd.set_option("display.max_columns", 50)


def main():
    pattern = f"results_{sys.argv[1]}.parquet" if len(sys.argv) > 1 else "results_*.parquet"
    files = sorted(RAW.glob(pattern))
    if not files:
        sys.exit(f"{RAW} içinde dosya bulunamadı ({pattern}).")

    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"=== {len(files)} dosya, {len(raw):,} satır ===\n")

    # 1) Hangi hipodromlar geldi?
    print("--- 1. HİPODROMLAR ---")
    sehirler = raw["sehir"].value_counts()
    print(sehirler.to_string())
    yabanci = [s for s in sehirler.index if not is_domestic(s)]
    print(f"\n{'!! YABANCI SIZMIŞ: ' + str(yabanci) if yabanci else 'OK: hepsi yurt içi.'}\n")

    # 2) Kolonlar dolu mu?
    print("--- 2. KOLON DOLULUK ORANLARI ---")
    onemli = ["At İsmi", "Jokey", "Antrenör", "Sahip", "HP", "Gny", "AGF",
              "Son 6 Y.", "KGS", "s20", "Sıklet", "Yaş", "En İyi D.",
              "mesafe", "pist", "kosu_no", "S"]
    for c in onemli:
        if c in raw.columns:
            oran = raw[c].notna().mean()
            isaret = "OK " if oran > 0.5 else "!! "
            print(f"{isaret}{c:<12}: %{100*oran:5.1f} dolu")
        else:
            print(f"!! {c:<12}: KOLON YOK")
    print()

    # 3) Etiketler doğru mu? (her koşuda tam 1 kazanan olmalı)
    print("--- 3. ETİKET KONTROLÜ (en kritik) ---")
    d = add_label_from_results(raw)
    print(f"Bitiş sırası okunabilen satır: %{100*d['bitis'].notna().mean():.1f}")

    per_race = d.groupby(["tarih", "sehir", "kosu_no"])["kazandi"].sum()
    dagilim = per_race.value_counts().sort_index()
    print("\nKoşu başına kazanan sayısı (1 olmalı):")
    for k, v in dagilim.items():
        isaret = "OK " if k == 1 else "!! "
        print(f"  {isaret}{int(k)} kazanan: {v:,} koşu (%{100*v/len(per_race):.1f})")

    saglikli = 100 * (per_race == 1).mean()
    print(f"\n>>> Sağlıklı koşu oranı: %{saglikli:.1f}")
    if saglikli > 90:
        print(">>> SONUÇ: Etiketler güvenilir, eğitime geçilebilir.")
    else:
        print(">>> SONUÇ: PROBLEM VAR — sonuç sayfası parser'ı düzeltilmeli.")

    # 4) Örnek bir koşu
    print("\n--- 4. ÖRNEK KOŞU (gözle kontrol) ---")
    ornek = d[(d["kosu_no"] == 1)].head(12)
    goster = [c for c in ["tarih", "sehir", "kosu_no", "bitis", "At İsmi",
                          "Jokey", "HP", "Gny", "AGF", "kazandi"]
              if c in ornek.columns]
    print(ornek[goster].to_string(index=False))

    # 5) Özellik üretimi çalışıyor mu?
    print("\n--- 5. ÖZELLİK ÜRETİMİ ---")
    try:
        f = build_features(d)
        kritik = ["hp", "gny", "agf", "son6_ort", "siklet", "yas", "derece_sn"]
        for c in kritik:
            if c in f.columns:
                print(f"OK {c:<12}: %{100*f[c].notna().mean():5.1f} dolu")
            else:
                print(f"!! {c:<12}: ÜRETİLEMEDİ")
    except Exception as e:
        print(f"!! Özellik üretimi hata verdi: {e}")


if __name__ == "__main__":
    main()
