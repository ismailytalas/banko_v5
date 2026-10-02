# 🐎 TJK Tahmin

TJK günlük yarış programını çeker, LightGBM ile her at için kazanma olasılığı üretir,
**İkili** kombinasyonları ve bütçeye göre optimize edilmiş **Altılı Ganyan** kuponu önerir.
Streamlit önyüzü ile günlük kullanım için tasarlanmıştır.

## Kurulum

```bash
pip install -r requirements.txt
```

## Hızlı başlangıç (model olmadan)

Model eğitmeden de uygulama çalışır; bu durumda sabah ganyanı + AGF karışımından
türetilen piyasa-temelli olasılıklar gösterilir:

```bash
streamlit run app.py
```

## Model eğitimi

1. **Geçmiş sonuçları indir** (ör. son 1,5 yıl — siteye saniyede ~1 istek atar,
   bir gecede biter):

   ```bash
   python scripts/collect_history.py 2025-01-01 2026-08-12
   ```

2. **Eğit:**

   ```bash
   python scripts/train.py
   ```

   Çıktıda zaman-bazlı test kümesinde `top1_hit_rate` (favori tahmininin kazanma
   oranı) ve `top2_hit_rate` görürsünüz. Model `models/model.txt` olarak kaydedilir
   ve Streamlit otomatik kullanır.

## Mimari

```
tjk/scraper.py    TJK HTML + CDN CSV kaynaklarından program/sonuç çekme
tjk/features.py   HP, Son6, KGS, s20, derece, AGF, ganyan, yarış-içi göreli özellikler
tjk/model.py      LightGBM binary + yarış içi softmax -> p_win; zaman-bazlı doğrulama
tjk/bets.py       Harville ile İkili olasılıkları; greedy bütçeli Altılı kupon kurucu
app.py            Streamlit önyüzü
```

## Modelleme notları

- **Hedef:** her at için `kazandı` (binary). Ham skorlar koşu içinde softmax ile
  normalize edilir; böylece bir koşudaki olasılıklar 1'e toplanır.
- **Sızıntıya dikkat:** Jokey/antrenör istatistikleri yalnızca eğitim döneminden
  hesaplanır; test bölmesi zamana göredir (rastgele değil).
- **En güçlü sinyaller** genelde piyasa kaynaklıdır (sabah ganyanı, AGF). Model
  bunların üzerine HP, form (Son 6), KGS, sınıf ve pist uyumu ekleyerek katkı arar.
- **İkili:** kazanma olasılıklarından Harville formülüyle ilk-2 olasılığı türetilir.
- **Altılı:** ayak başına at sayısı, bütçe sınırı içinde marjinal olasılık
  kazancı / maliyet oranı en yüksek ekleme yapılarak (greedy) genişletilir.

## Yol haritası fikirleri

- Sonuç sayfalarından bitirme derecesi + ganyan ödemelerini toplayıp **ROI bazlı
  backtest** (isabet oranı tek başına kârlılığı göstermez).
- At bazında geçmiş yarış zinciri (AtId üzerinden) ile pist/mesafe uyum özellikleri.
- Sıralı İkili ve Üçlü Bahis için Harville genişletmesi.
- Yurt dışı ayakları için ayrı model.

## Yeni özellikler (app_picks.py)

| Özellik | Nerede | Not |
|---|---|---|
| 🔄 Değişkenleri yeniden üret | Kenar çubuğu | Programı/AGF'yi taze çeker; özellik, tahmin ve Altılı yeniden üretilir. Değişenler tabloda listelenir. |
| 🤖 İkili & Çifte önerisi | 1. sekme, her koşunun altında | Güvenli / sürpriz İkili, 3-4-5 atlı kolon karşılaştırması, değer kolonu, Çifte (güvenli, 2x2, sürpriz). Rakamlar kodda (Harville, model % / halk %) hesaplanır; NVIDIA gpt-oss yalnızca yorumlar. |
| 🧪 Kupon laboratuvarı | Altılı sekmesi | Bütçe aralığı × strateji (Olasılık / Dengeli / Değer) ile çok sayıda kupon; halka göre avantaj; tek at bırakılan ayaklar için alternatif atlar; ayak başı en az at seçeneği. |
| Günün Sonuçları (EDA) | 3. sekme | Kazanan at, jokey, antrenör, sahip; ilk-3 grafikleri; model vs sonuç. |

NVIDIA anahtarı: `.streamlit/secrets.toml` → `NVIDIA_API_KEY = "nvapi-..."`
(Streamlit Cloud: Settings → Secrets).

Yeni modüller: `tjk/sonuclar.py`, `tjk/eda.py`, `tjk/llm_yorum.py`, `tjk/altili_plus.py`.

## Uyarı

Bu yazılım istatistiksel tahmin aracıdır; kazanç garantisi yoktur ve hiçbir model
uzun vadede kesin kâr sağlayamaz. Yalnızca yasal TJK bahis kanallarında ve
kaybetmeyi göze alabileceğiniz bütçeyle kullanın.
