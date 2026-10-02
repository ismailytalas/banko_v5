# -*- coding: utf-8 -*-
"""GÜNLÜK SÜRPRİZ RAPORU — her koşu için at listesi + sürpriz notları.

    python scripts/daily_picks.py                    # bugün, tüm hipodromlar
    python scripts/daily_picks.py --tarih 2026-08-15
    python scripts/daily_picks.py --sehir Ankara
    python scripts/daily_picks.py --kolon 5          # İkili kolonunda 5 at
    python scripts/daily_picks.py --retrain          # modeli yeniden eğit

Ne yapar:
  1. Geçmiş veriyle PİYASASIZ iki model eğitir (kazanma + ilk2)
  2. Bugünün programını çeker, aynı özellikleri üretir
  3. Her koşu için atları listeler: model görüşü, piyasa görüşü, değer oranı
  4. SÜRPRİZ adaylarını işaretler (model beğeniyor ama halk görmüyor)
  5. İkili tek kolon için N atlık kolon önerir

NOT: Nihai karar senin. Bu liste modelin görüşüdür, garanti değildir.
Ganyan bazlı geçmiş testlerde model piyasayı geçemedi; İkili havuzunda
durum farklı olabilir ama bu HENÜZ ÖLÇÜLMEDİ.
"""
import sys
import argparse
import pathlib
import datetime as dt
from itertools import combinations

import numpy as np
import pandas as pd
import lightgbm as lgb

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from tjk import scraper  # noqa: E402
from tjk.features import build_features, add_label_from_results  # noqa: E402
from tjk.history import add_history_features  # noqa: E402
from tjk.model import FEATURE_COLS, RACE_KEY  # noqa: E402

RAW = ROOT / "data" / "raw"
MODELS = ROOT / "models"

PIYASA_KOLONLARI = {"gny", "agf", "gny_implied", "gny_implied_rel",
                    "gny_implied_rank", "agf_rel", "agf_rank",
                    "sabah_gny", "agf_prg", "agf_sira_prg"}

PARAMS = dict(objective="binary", metric="binary_logloss", learning_rate=.05,
              num_leaves=31, min_data_in_leaf=50, feature_fraction=.8,
              bagging_fraction=.8, bagging_freq=1, lambda_l2=1.0, verbosity=-1)


def kosu_ici(df, col, out):
    d = df.copy()
    t = d[col].clip(1e-9, 1 - 1e-9)
    d["_l"] = np.log(t / (1 - t))
    d[out] = d.groupby(RACE_KEY, dropna=False)["_l"].transform(
        lambda s: np.exp(s - s.max()) / np.exp(s - s.max()).sum())
    return d.drop(columns=["_l"])


def gecmisi_yukle():
    files = sorted(RAW.glob("results_*.parquet"))
    if not files:
        sys.exit("Geçmiş veri yok. Önce collect_history.py çalıştırın.")
    raw = pd.concat([pd.read_parquet(f) for f in files], ignore_index=True)
    return add_label_from_results(raw)


def programi_cek(tarih, sehir_filtre=None):
    """Verilen günün programını tüm yurt içi hipodromlar için çeker."""
    sehirler = scraper.get_cities(tarih)
    parcalar = []
    for sid, ad in sehirler:
        if not scraper.is_domestic(ad):
            continue
        if sehir_filtre and sehir_filtre.lower() not in ad.lower():
            continue
        try:
            d = scraper.get_program(tarih, sid, ad)
            if not d.empty:
                parcalar.append(d)
                print(f"  {ad}: {len(d)} at")
        except Exception as e:
            print(f"  {ad}: alınamadı ({e})")
    return pd.concat(parcalar, ignore_index=True) if parcalar else pd.DataFrame()


def egit(df, feats, hedef, tur=800):
    d = df.dropna(subset=[hedef])
    return lgb.train(PARAMS, lgb.Dataset(d[feats], label=d[hedef]), tur)


def ikili_kolon(rd, n_at, no_col="no"):
    """İkili tek kolon için en iyi N atı seçer.

    Ölçüt: atın ilk-2'ye girme olasılığı (p_ilk2). Kolonda N at varsa
    C(N,2) kombinasyon oluşur ve ikisi de ilk 2'ye girerse kazanılır.
    """
    d = rd.dropna(subset=["p_ilk2"]).sort_values("p_ilk2", ascending=False)
    sec = d.head(n_at)
    # Kolonun tutma olasılığı (kaba tahmin): seçilen atlardan en az ikisi ilk2'de
    p = sec["p_ilk2"].values
    # ilk2'de tam 2 at var; hepsinin kolonda olma olasılığını yaklaşık hesapla
    tut = 0.0
    for i, j in combinations(range(len(p)), 2):
        tut += p[i] * p[j] / 2
    return sec, min(tut, 1.0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tarih", type=str)
    ap.add_argument("--sehir", type=str)
    ap.add_argument("--kolon", type=int, default=5, help="İkili kolonunda at sayısı")
    ap.add_argument("--retrain", action="store_true")
    args = ap.parse_args()

    tarih = (dt.date.fromisoformat(args.tarih) if args.tarih else dt.date.today())

    print("Geçmiş veri yükleniyor...")
    hist = gecmisi_yukle()

    print(f"\n{tarih} programı çekiliyor...")
    prog = programi_cek(tarih, args.sehir)
    if prog.empty:
        sys.exit("Program bulunamadı (o gün koşu olmayabilir).")

    # Geçmiş + bugün birlikte: geçmiş özellikleri bugünün atları için de
    # hesaplansın diye. Bugünün satırlarında sonuç yok, etiketleri 0 veriyoruz;
    # prior hesabı aynı günü zaten dışlıyor, sızıntı olmaz.
    prog = prog.copy()
    prog["S"] = np.nan
    birlesik = pd.concat([hist, prog], ignore_index=True)
    birlesik = add_label_from_results(birlesik)
    birlesik["kazandi"] = birlesik["kazandi"].fillna(0)

    print("Özellikler üretiliyor (birkaç dakika)...")
    df = build_features(birlesik)
    df["kazandi"] = birlesik["kazandi"].values
    df["bitis"] = birlesik["bitis"].values
    df["ilk2"] = (df["bitis"] <= 2).astype(float)
    df.loc[df["bitis"].isna(), "ilk2"] = np.nan
    df = add_history_features(df)

    bugun = df[df["tarih"] == str(tarih)].copy()
    gecmis = df[(df["tarih"] < str(tarih)) & df["bitis"].notna()].copy()
    if bugun.empty:
        sys.exit("Bugünün satırları özellik tablosunda bulunamadı.")

    feats = [c for c in FEATURE_COLS
             if c in df.columns and c not in PIYASA_KOLONLARI]

    MODELS.mkdir(exist_ok=True)
    yol_win = MODELS / "temel_win.txt"
    yol_ilk2 = MODELS / "temel_ilk2.txt"
    if args.retrain or not yol_win.exists() or not yol_ilk2.exists():
        print("Modeller eğitiliyor (piyasasız)...")
        m_win = egit(gecmis, feats, "kazandi")
        m_ilk2 = egit(gecmis, feats, "ilk2")
        m_win.save_model(str(yol_win))
        m_ilk2.save_model(str(yol_ilk2))
    else:
        print("Kayıtlı modeller kullanılıyor (--retrain ile yenileyin).")
        m_win = lgb.Booster(model_file=str(yol_win))
        m_ilk2 = lgb.Booster(model_file=str(yol_ilk2))

    bugun["s_win"] = m_win.predict(bugun[feats])
    bugun = kosu_ici(bugun, "s_win", "p_win")
    bugun["p_ilk2"] = m_ilk2.predict(bugun[feats])

    # Piyasa görüşü: sabah ganyanı ve AGF
    o = pd.to_numeric(bugun["gny"], errors="coerce")
    bugun["oran"] = o / 100.0 if o.median(skipna=True) > 50 else o
    bugun["p_piyasa"] = 1.0 / bugun["oran"]
    bugun["p_piyasa"] = (bugun["p_piyasa"] /
                         bugun.groupby(RACE_KEY)["p_piyasa"].transform("sum"))
    bugun["deger"] = bugun["p_win"] / bugun["p_piyasa"].replace(0, np.nan)
    bugun["agf_sira"] = bugun.groupby(RACE_KEY)["agf"].rank(
        ascending=False, method="min")
    bugun["model_sira"] = bugun.groupby(RACE_KEY)["p_win"].rank(
        ascending=False, method="min")

    if "no" not in bugun.columns or bugun["no"].isna().all():
        bugun["no"] = bugun.get("start_no", bugun["at"])

    # --- RAPOR ------------------------------------------------------------
    for (t, sehir, kno), rd in bugun.groupby(RACE_KEY, sort=True):
        rd = rd.sort_values("p_win", ascending=False)
        meta = rd.iloc[0]
        baslik = f"{sehir} — {int(kno)}. Koşu"
        if pd.notna(meta.get("saat")):
            baslik += f" ({meta['saat']})"
        if pd.notna(meta.get("mesafe")):
            baslik += f" — {int(meta['mesafe'])}m {meta.get('pist') or ''}"
        if pd.notna(meta.get("kosu_turu")):
            baslik += f" — {meta['kosu_turu']}"
        print("\n" + "=" * 78)
        print(baslik)
        print("=" * 78)

        goster = rd[["no", "at", "Jokey", "oran", "agf", "agf_sira",
                     "p_win", "p_ilk2", "deger"]].copy()
        goster["agf"] = (100 * goster["agf"]).round(0)
        goster["p_win"] = (100 * goster["p_win"]).round(1)
        goster["p_ilk2"] = (100 * goster["p_ilk2"]).round(1)
        goster["deger"] = goster["deger"].round(2)

        # Sürpriz işareti: model beğeniyor (ilk 4) ama halk görmüyor (AGF 5+)
        notlar = []
        for _, r in rd.iterrows():
            n = []
            if r["model_sira"] <= 4 and r["agf_sira"] >= 5:
                n.append("SÜRPRİZ")
            if pd.notna(r["deger"]) and r["deger"] >= 1.6:
                n.append("değerli")
            if pd.notna(r.get("dinlenme_gun")) and 20 <= r["dinlenme_gun"] <= 45:
                n.append("form aralığı")
            if r.get("ilk_yaris_mi") == 1:
                n.append("ilk yarış")
            if pd.notna(r.get("at_pist_rate")) and r["at_pist_rate"] > 0.25:
                n.append("pist uyumlu")
            notlar.append(", ".join(n))
        goster["NOT"] = notlar
        goster.columns = ["No", "At", "Jokey", "Ganyan", "AGF%", "AGF sıra",
                          "Model kazanma%", "Model ilk2%", "Değer", "NOT"]
        print(goster.to_string(index=False))

        sec, tut = ikili_kolon(rd, args.kolon)
        if len(sec) >= 2:
            nolar = [str(int(x)) if str(x).replace(".", "").isdigit() else str(x)
                     for x in sec["no"]]
            kombin = len(list(combinations(nolar, 2)))
            print(f"\n  İKİLİ TEK KOLON ({args.kolon} at): {' - '.join(nolar)}")
            print(f"  {kombin} kombinasyon | tahmini tutma olasılığı ~%{100*tut:.0f}")
            surprizli = sec[sec["agf_sira"] >= 5]
            if not surprizli.empty:
                print(f"  Kolondaki sürpriz adayları: "
                      f"{', '.join(str(x) for x in surprizli['no'])}")

    print("\n" + "=" * 78)
    print("Bu liste modelin görüşüdür; nihai karar senin.")
    print("Model geçmiş testlerde ganyan piyasasını GEÇEMEDİ. İkili havuzunda")
    print("performansı henüz ölçülmedi (ödeme verisi indiriliyor).")


if __name__ == "__main__":
    main()
