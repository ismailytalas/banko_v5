# -*- coding: utf-8 -*-
"""Tüm parquet'leri birleştirir, model değişkenlerini üretir, Excel'e aktarır.

Kullanım:
    python scripts/export_dataset.py                    # son 3 ay -> Excel
    python scripts/export_dataset.py --ay 12            # son 12 ay
    python scripts/export_dataset.py --tumu             # tamamı (sadece CSV)
    python scripts/export_dataset.py --tarih 2026-08-01 # bu tarihten sonrası

Çıktılar (data/export klasörüne):
    tjk_veri_YYYYMMDD.xlsx   -> Excel: Veri + Sözlük + Özet sayfaları
    tjk_veri_tam.csv         -> tüm satırlar (Excel'e sığmayan durumlar için)

NOT: Excel 1 milyon satır sınırına sahip ve büyük dosyalarda çok yavaşlar.
5,5 yıllık veri ~320 bin satır olduğu için varsayılan olarak SON 3 AY
aktarılır. Tamamını istersen --tumu ile CSV alırsın.
"""
import sys
import argparse
import pathlib
import datetime as dt

import numpy as np
import pandas as pd

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk.features import build_features, add_label_from_results  # noqa: E402
from tjk.history import add_history_features  # noqa: E402
from tjk.model import FEATURE_COLS  # noqa: E402

RAW = ROOT / "data" / "raw"
OUT = ROOT / "data" / "export"

# Kolon açıklamaları — Excel'de "Sözlük" sayfası olarak yazılır
SOZLUK = {
    "tarih": "Yarış tarihi",
    "sehir": "Hipodrom",
    "kosu_no": "Koşu numarası",
    "saat": "Koşu saati",
    "at": "At ismi (temizlenmiş)",
    "Jokey": "Jokey", "Antrenör": "Antrenör", "Sahip": "At sahibi",
    "bitis": "Bitiş sırası (1 = birinci)",
    "kazandi": "HEDEF: 1 = kazandı",
    "ilk3": "1 = ilk üçe girdi",
    "Derece": "Bitirme derecesi (ham)",
    "Fark": "Öndekiyle arasındaki boy farkı",
    # --- program/piyasa bilgileri ---
    "hp": "Handikap puanı",
    "siklet": "Taşıdığı kilo",
    "yas": "Yaş",
    "gny": "Ganyan (sonuç sayfasındaki kesinleşmiş oran)",
    "gny_implied": "Ganyandan türetilen zımni olasılık (1/oran)",
    "agf": "AGF oranı (bahis dağılımı)",
    "start_no": "Start yeri",
    "mesafe": "Mesafe (m)",
    "pist_kum": "1 = kum pist", "pist_cim": "1 = çim pist",
    "alan_sayisi": "Koşudaki at sayısı",
    # --- yarış içi göreli ---
    "hp_rel": "HP'nin koşu ortalamasından farkı",
    "agf_rel": "AGF'nin koşu ortalamasından farkı",
    "gny_implied_rel": "Zımni olasılığın koşu ortalamasından farkı",
    "siklet_rel": "Sikletin koşu ortalamasından farkı",
    "hp_rank": "HP'nin koşu içi yüzdelik sırası",
    "agf_rank": "AGF'nin koşu içi yüzdelik sırası",
    "gny_implied_rank": "Zımni olasılığın koşu içi sırası",
    "siklet_rank": "Sikletin koşu içi sırası",
    # --- geçmiş (sızıntısız: sadece önceki günler) ---
    "jokey_rate": "Jokeyin geçmiş kazanma oranı (düzleştirilmiş)",
    "jokey_n": "Jokeyin geçmiş yarış sayısı",
    "jokey_ilk3_rate": "Jokeyin geçmiş ilk-3 oranı",
    "jokey_sehir_rate": "Jokeyin bu hipodromdaki geçmiş oranı",
    "antrenor_rate": "Antrenörün geçmiş kazanma oranı",
    "antrenor_n": "Antrenörün geçmiş yarış sayısı",
    "antrenor_ilk3_rate": "Antrenörün geçmiş ilk-3 oranı",
    "sahip_rate": "At sahibinin geçmiş kazanma oranı",
    "sahip_n": "At sahibinin geçmiş yarış sayısı",
    "sahip_ilk3_rate": "At sahibinin geçmiş ilk-3 oranı",
    "jok_ant_rate": "Jokey+antrenör ikilisinin geçmiş oranı",
    "at_genel_rate": "Atın geçmiş kazanma oranı",
    "at_genel_n": "Atın geçmiş yarış sayısı",
    "at_ilk3_rate": "Atın geçmiş ilk-3 oranı",
    "at_pist_rate": "Atın bu pist tipindeki geçmiş oranı",
    "at_mesafe_rate": "Atın bu mesafe bandındaki geçmiş oranı",
    "at_sehir_rate": "Atın bu hipodromdaki geçmiş oranı",
    "dinlenme_gun": "Son yarışından bu yana geçen gün",
    "ilk_yaris_mi": "1 = kariyerinin ilk yarışı",
    "at_max_sinif": "Geçmişte koştuğu en yüksek sınıf kalitesi",
    "at_ort_sinif": "Geçmişte koştuğu ortalama sınıf kalitesi",
    # --- koşu şartı ---
    "kosu_turu": "Handikap / Şartlı / Maiden / KV / Satış / Vasıf / Grup",
    "sinif_no": "Sınıf numarası (Şartlı 5 -> 5)",
    "cins": "İngiliz / Arap",
    "min_yas": "Yaş şartının alt sınırı",
    "yas_acik": "1 = 've yukarı' (üst sınır yok)",
    "disi_mi": "1 = sadece dişilere açık",
    "isimli_kosu": "1 = isimli koşu (ör. GAZİ KOŞUSU)",
    "grup_no": "Grup derecesi (1/2/3)",
    "grup_mu": "1 = grup koşusu",
    "arap_mi": "1 = Arap atı koşusu",
    "maiden_mi": "1 = Maiden", "handikap_mi": "1 = Handikap",
    "satis_mi": "1 = Satış", "sartli_mi": "1 = Şartlı", "vasif_mi": "1 = A/B/C vasıf",
    "sinif_kalite": "Sınıfın ampirik kalitesi (o sınıftaki ort. HP)",
    "sinif_degisim": "Önceki koşusuna göre sınıf değişimi (+ = yükseldi)",
    "onceki_sinif_kalite": "Bir önceki koşusunun sınıf kalitesi",
    "mesafe_degisim": "Önceki koşusuna göre mesafe değişimi (m)",
    "pist_degisti": "1 = önceki koşusundan farklı pist tipi",
    "ikramiye_1": "Birinciye ödenen ikramiye (TL)",
    "ikramiye_rel": "İkramiyenin dönem medyanına oranı (enflasyondan arındırılmış)",
    "log_ikramiye_rel": "ikramiye_rel'in logaritması",
    "ikramiye_degisim": "Önceki koşusuna göre (arındırılmış) ikramiye değişimi",
}

KIMLIK = ["tarih", "sehir", "kosu_no", "saat", "at", "Jokey", "Antrenör",
          "Sahip", "bitis", "kazandi", "Derece", "Fark", "kosu_turu", "cins"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ay", type=int, default=3, help="son kaç ay (varsayılan 3)")
    ap.add_argument("--tarih", type=str, help="bu tarihten sonrası (YYYY-MM-DD)")
    ap.add_argument("--tumu", action="store_true", help="tüm veri (CSV)")
    args = ap.parse_args()

    files = sorted(RAW.glob("results_*.parquet"))
    if not files:
        sys.exit(f"{RAW} içinde parquet yok.")

    print(f"{len(files)} dosya birleştiriliyor...")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    print(f"  {len(raw):,} ham satır")

    # ÖNEMLİ: Geçmiş özellikleri TÜM veriden hesaplanmalı. Önce kesip sonra
    # hesaplarsak atların/jokeylerin geçmişi kaybolur ve değerler bozulur.
    print("Etiketler ve özellikler üretiliyor...")
    raw = add_label_from_results(raw)
    df = build_features(raw)
    df["kazandi"] = raw["kazandi"].values
    df["bitis"] = raw["bitis"].values
    for c in ("Derece", "Fark"):
        if c in raw.columns:
            df[c] = raw[c].values

    print("Geçmiş özellikleri üretiliyor (birkaç dakika sürebilir)...")
    df = add_history_features(df)
    print(f"  {df.shape[0]:,} satır x {df.shape[1]} kolon")

    # Kolon sırası: kimlik bilgileri önce, sonra model değişkenleri
    kimlik = [c for c in KIMLIK if c in df.columns]
    ozellik = [c for c in FEATURE_COLS if c in df.columns and c not in kimlik]
    dislenmis = df[kimlik + ozellik].copy()
    dislenmis_sirali = dislenmis.sort_values(["tarih", "sehir", "kosu_no", "bitis"])

    OUT.mkdir(parents=True, exist_ok=True)
    bugun = dt.date.today().strftime("%Y%m%d")

    # --- Tam CSV -----------------------------------------------------------
    csv_yolu = OUT / "tjk_veri_tam.csv"
    dislenmis_sirali.to_csv(csv_yolu, index=False, encoding="utf-8-sig", sep=";")
    print(f"\nCSV yazıldı: {csv_yolu}  ({len(dislenmis_sirali):,} satır)")

    if args.tumu:
        print("(--tumu verildi, Excel atlandı — bu boyut Excel'de çok yavaşlar)")
        return

    # --- Excel için filtrele ----------------------------------------------
    if args.tarih:
        kesim = args.tarih
    else:
        son = pd.to_datetime(dislenmis_sirali["tarih"]).max()
        kesim = (son - pd.DateOffset(months=args.ay)).strftime("%Y-%m-%d")
    excel_df = dislenmis_sirali[dislenmis_sirali["tarih"] >= kesim]
    print(f"Excel için filtre: {kesim} sonrası -> {len(excel_df):,} satır")

    if len(excel_df) > 300_000:
        print("!! Çok büyük, Excel yavaş açılacak. --ay ile daralt.")

    xlsx_yolu = OUT / f"tjk_veri_{bugun}.xlsx"
    with pd.ExcelWriter(xlsx_yolu, engine="openpyxl") as w:
        excel_df.to_excel(w, sheet_name="Veri", index=False)

        # Sözlük sayfası
        soz = pd.DataFrame(
            [{"Kolon": c, "Açıklama": SOZLUK.get(c, ""),
              "Tip": "Kimlik" if c in kimlik else "Model değişkeni",
              "Dolu %": round(100 * excel_df[c].notna().mean(), 1)}
             for c in excel_df.columns])
        soz.to_excel(w, sheet_name="Sözlük", index=False)

        # Özet sayfası
        sayisal = excel_df.select_dtypes(include=[np.number])
        ozet = sayisal.describe().T.reset_index().rename(columns={"index": "Kolon"})
        ozet = ozet.round(3)
        ozet.to_excel(w, sheet_name="Özet", index=False)

        # Kolon genişlikleri ve font
        from openpyxl.styles import Font
        for ad in ("Veri", "Sözlük", "Özet"):
            ws = w.book[ad]
            for cell in ws[1]:
                cell.font = Font(name="Arial", bold=True)
            ws.freeze_panes = "A2"
            for col in ws.columns:
                harf = col[0].column_letter
                uzunluk = max((len(str(c.value)) for c in col[:200] if c.value), default=8)
                ws.column_dimensions[harf].width = min(max(uzunluk + 2, 10), 42)

    print(f"Excel yazıldı: {xlsx_yolu}")
    print("\nSayfalar: Veri (satırlar) | Sözlük (kolon açıklamaları) | Özet (istatistikler)")


if __name__ == "__main__":
    main()
