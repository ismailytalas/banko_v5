# -*- coding: utf-8 -*-
"""TJK Sürpriz Radarı — Streamlit önyüzü.

    streamlit run app_picks.py
"""
import datetime as dt

import pandas as pd
import streamlit as st

from tjk import scraper
from tjk.picks import (gecmisi_yukle, gunun_programi, ozellik_uret,
                       modelleri_hazirla, tahmin_et, surpriz_notlari,
                       ikili_kolon, altili_tablosu, PIYASA_KOLONLARI)
from tjk.model import FEATURE_COLS, RACE_KEY
from tjk import sonuclar, eda, llm_yorum, altili_plus

st.set_page_config(page_title="TJK Sürpriz Radarı", page_icon="🐎", layout="wide")
st.title("🐎 TJK Sürpriz Radarı")
st.caption("Model piyasadan bağımsız çalışır (AGF ve ganyan modele verilmez). "
           "Bu bir tahmin aracıdır, kazanç garantisi yoktur.")

ss = st.session_state
ss.setdefault("yenile", 0)          # program/özellik önbellek anahtarı
ss.setdefault("son_guncelleme", None)

# --- Ayarlar ---------------------------------------------------------------
with st.sidebar:
    st.header("Ayarlar")
    tarih = st.date_input("Tarih", dt.date.today(), format="DD/MM/YYYY")

    st.subheader("Güncelleme")
    if st.button("🔄 Değişkenleri yeniden üret",
                 help="Programı (AGF, koşmaz, jokey değişikliği) TJK'dan yeniden "
                      "çeker, özellikleri ve tahminleri baştan üretir. Geçmiş veri "
                      "ve model yeniden yüklenmez."):
        if "son_pred" in ss:
            ss["onceki_pred"] = ss["son_pred"]   # (anahtar, pred)
        ss.yenile += 1
    if ss.son_guncelleme:
        st.caption(f"Son üretim: {ss.son_guncelleme:%H:%M:%S}")

    st.subheader("Sürpriz tanımı")
    agf_esigi = st.slider("Halkın AGF sırası şundan geride olsun", 3, 12, 6,
                          help="6 = halkın ilk 5 favorisi sürpriz sayılmaz")
    model_esigi = st.slider("Model sıralamasında ilk kaça girsin", 2, 8, 4)
    deger_esigi = st.slider("'Değerli' eşiği (model/piyasa oranı)",
                            1.0, 3.0, 1.6, 0.1)

    st.subheader("İkili tek kolon")
    kolon_at = st.slider("Kolondaki at sayısı", 2, 10, 5)
    st.caption(f"{kolon_at} at = {kolon_at*(kolon_at-1)//2} kombinasyon")

    st.subheader("Altılı Ganyan")
    misli = st.number_input("Misli birim fiyatı (TL)", 0.5, 100.0, 3.0, 0.5)
    ilk_ayak = st.number_input("Altılı başlangıç koşusu", 1, 12, 1)

    st.subheader("🤖 LLM önerisi (NVIDIA)")
    nv_key = llm_yorum.api_anahtari(st.secrets if hasattr(st, "secrets") else None)
    if not nv_key:
        nv_key = st.text_input("NVIDIA API anahtarı", type="password",
                               help="build.nvidia.com → Get API Key. Kalıcı için "
                                    ".streamlit/secrets.toml içine NVIDIA_API_KEY yazın.")
    nv_model = st.selectbox("Model", ["openai/gpt-oss-20b"],
                            help="20b daha hızlı, 120b daha tutarlı")

    yeniden = st.checkbox("Modeli yeniden eğit", value=False,
                          help="İlk çalıştırmada veya yeni veri indirince işaretleyin")


@st.cache_data(ttl=3600, show_spinner="Hipodromlar yükleniyor…")
def _sehirler(t: dt.date):
    return [(sid, ad) for sid, ad in scraper.get_cities(t)
            if scraper.is_domestic(ad)]


try:
    sehirler = _sehirler(tarih)
except Exception as e:
    st.error(f"TJK'ya ulaşılamadı: {e}")
    st.stop()

if not sehirler:
    st.warning("Bu tarihte yurt içi koşu bulunamadı.")
    st.stop()

secili = st.selectbox("Hipodrom", sehirler, format_func=lambda x: x[1])
sehir_id, sehir_adi = secili


@st.cache_resource(show_spinner="Geçmiş veri yükleniyor (ilk seferde uzun sürer)…")
def _gecmis():
    return gecmisi_yukle()


@st.cache_data(ttl=900, show_spinner="Program çekiliyor…")
def _program(t: dt.date, sehir: str, token: int):
    # token değişince önbellek atlanır -> TJK'dan taze program (güncel AGF)
    return gunun_programi(t, sehir)


@st.cache_resource(show_spinner="Özellikler ve modeller hazırlanıyor…",
                   max_entries=6)
def _tahminler(t_str: str, sehir: str, yeniden_egit: bool, token: int):
    t = dt.date.fromisoformat(t_str)
    hist = _gecmis()
    prog = _program(t, sehir, token)
    if prog.empty:
        return pd.DataFrame()
    bugun, gecmis = ozellik_uret(hist, prog, t)
    feats = [c for c in FEATURE_COLS
             if c in gecmis.columns and c not in PIYASA_KOLONLARI]
    m_win, m_ilk2 = modelleri_hazirla(gecmis, feats, yeniden_egit)
    out = tahmin_et(bugun, m_win, m_ilk2, feats)
    return out, dt.datetime.now()


try:
    sonuc_tahmin = _tahminler(tarih.isoformat(), sehir_adi, yeniden, ss.yenile)
except FileNotFoundError as e:
    st.error(f"{e}\n\nÖnce `python scripts/collect_history.py` çalıştırın.")
    st.stop()

if isinstance(sonuc_tahmin, pd.DataFrame) or sonuc_tahmin is None:
    pred, uretim = pd.DataFrame(), None
else:
    pred, uretim = sonuc_tahmin
    ss.son_guncelleme = uretim

if pred is None or pred.empty:
    st.warning("Bu hipodrom için program verisi alınamadı.")
    st.stop()

pred = pred.copy()
gun_anahtari = (tarih.isoformat(), sehir_adi)
kosular = sorted(pred["kosu_no"].dropna().unique())


# --- Yenileme farkları -----------------------------------------------------------
def _farklar(eski: pd.DataFrame, yeni: pd.DataFrame) -> pd.DataFrame:
    k = ["kosu_no", "no"]
    kol = [c for c in ["at", "Jokey", "agf", "agf_sira", "p_win"] if c in yeni.columns]
    e = eski[k + kol].copy()
    y = yeni[k + kol].copy()
    m = y.merge(e, on=k, how="outer", suffixes=("", "_eski"), indicator=True)
    satir = []
    for r in m.to_dict("records"):
        neden = []
        if r["_merge"] == "left_only":
            neden.append("yeni eklendi")
        elif r["_merge"] == "right_only":
            neden.append("listeden çıktı (koşmaz?)")
        else:
            if "agf_sira" in kol and r.get("agf_sira") != r.get("agf_sira_eski") \
                    and pd.notna(r.get("agf_sira")):
                neden.append(f"AGF sıra {_i(r.get('agf_sira_eski'))} → {_i(r.get('agf_sira'))}")
            if "Jokey" in kol and r.get("Jokey") != r.get("Jokey_eski"):
                neden.append(f"jokey {r.get('Jokey_eski')} → {r.get('Jokey')}")
            if "p_win" in kol and pd.notna(r.get("p_win")) and pd.notna(r.get("p_win_eski")) \
                    and abs(r["p_win"] - r["p_win_eski"]) >= 0.01:
                neden.append(f"model %{100*r['p_win_eski']:.1f} → %{100*r['p_win']:.1f}")
        if neden:
            satir.append({"Koşu": _i(r["kosu_no"]), "No": _i(r["no"]),
                          "At": r.get("at") or r.get("at_eski"),
                          "Değişiklik": "; ".join(neden)})
    return pd.DataFrame(satir)


def _i(x):
    return None if x is None or pd.isna(x) else int(x)


onceki = ss.get("onceki_pred")
if onceki is not None and onceki[0] == gun_anahtari and onceki[1] is not pred:
    fark = _farklar(onceki[1], pred)
    with st.expander(f"🔄 Son yenilemede değişenler ({len(fark)})",
                     expanded=not fark.empty):
        if fark.empty:
            st.write("Programda veya AGF sıralamasında değişiklik yok.")
        else:
            st.dataframe(fark, hide_index=True)
ss["son_pred"] = (gun_anahtari, pred)


llm_anahtar = f"llm|{'|'.join(gun_anahtari)}|{ss.yenile}"
yorumlar = ss.get(llm_anahtar, {})

sekme1, sekme2, sekme4 = st.tabs(
    ["Koşular ve İkili", "Altılı Ganyan", "Günün Sonuçları (EDA)"])


# --- Koşu bazlı ------------------------------------------------------------
with sekme1:
    c_a, c_b = st.columns([1, 3])
    if c_a.button("🤖 LLM önerilerini üret"):
        aday = {}
        for k in kosular:
            sonraki = pred[pred["kosu_no"] == k + 1]
            aday[int(k)] = llm_yorum.adaylar(pred[pred["kosu_no"] == k],
                                            sonraki if not sonraki.empty else None,
                                            agf_esigi=agf_esigi)
        if nv_key:
            try:
                with st.spinner(f"{nv_model} yorumluyor…"):
                    yorumlar = llm_yorum.llm_yorumla(
                        aday, nv_key, nv_model,
                        baglam=f"{sehir_adi} {tarih:%d.%m.%Y}. Model piyasadan "
                               f"bağımsızdır; İkili olasılığı Harville ile hesaplandı.")
            except Exception as e:
                st.warning(f"LLM çağrısı başarısız ({e}); şablon öneriler gösteriliyor.")
                yorumlar = {k: llm_yorum.sablon_metin(a) for k, a in aday.items()}
        else:
            yorumlar = {k: llm_yorum.sablon_metin(a) for k, a in aday.items()}
        ss[llm_anahtar] = yorumlar
        ss[llm_anahtar + "|aday"] = aday
    c_b.caption("İkili ve Çifte önerileri tahmin tablosundaki Kazanma %, İlk2 %, AGF "
                "ve Değer'den hesaplanır (Harville). Değer = model % / halk %; 1'in "
                "üstü halkın küçümsediği, isabet ederse yüksek ödeyen kombinasyon. "
                "LLM yalnızca bu rakamları yorumlar.")

    surpriz_sayaci = 0
    for k in kosular:
        rd = pred[pred["kosu_no"] == k].sort_values("p_win", ascending=False)
        meta = rd.iloc[0]
        baslik = f"{int(k)}. Koşu"
        if pd.notna(meta.get("saat")):
            baslik += f" — {meta['saat']}"
        if pd.notna(meta.get("mesafe")):
            baslik += f" — {int(meta['mesafe'])}m {meta.get('pist') or ''}"
        if pd.notna(meta.get("kosu_turu")):
            baslik += f" — {meta['kosu_turu']}"

        notlar = surpriz_notlari(rd, agf_esigi, model_esigi, deger_esigi)
        var_surpriz = any("SÜRPRİZ" in n for n in notlar)
        surpriz_sayaci += sum("SÜRPRİZ" in n for n in notlar)

        with st.expander(("🔥 " if var_surpriz else "") + baslik,
                         expanded=var_surpriz):
            g = rd[["no", "at", "Jokey", "oran", "agf", "agf_sira",
                    "p_win", "p_ilk2", "deger"]].copy()
            g["no"] = g["no"].astype("Int64")
            g["agf_sira"] = g["agf_sira"].astype("Int64")
            g["agf"] = (100 * g["agf"]).round(0)
            g["p_win"] = (100 * g["p_win"]).round(1)
            g["p_ilk2"] = (100 * g["p_ilk2"]).round(1)
            g["deger"] = g["deger"].round(2)
            g["Not"] = notlar
            g.columns = ["No", "At", "Jokey", "Ganyan", "AGF %", "AGF sıra",
                         "Kazanma %", "İlk2 %", "Değer", "Not"]
            st.dataframe(g, hide_index=True)

            sec, kombin, tut = ikili_kolon(rd, kolon_at)
            nolar = " - ".join(str(int(n)) for n in sec["no"] if pd.notna(n))
            c1, c2, c3 = st.columns([2, 1, 1])
            c1.markdown(f"**İkili kolon:** `{nolar}`")
            c2.metric("Kombinasyon", kombin)
            c3.metric("Tutma olasılığı", f"%{100*tut:.0f}")
            kolon_surpriz = sec[sec["agf_sira"] >= agf_esigi]
            if not kolon_surpriz.empty:
                isimler = ", ".join(
                    f"{int(r.no)} {r.at}" for r in kolon_surpriz.itertuples()
                    if pd.notna(r.no))
                st.info(f"Kolondaki sürpriz adayları: {isimler}")

            y = yorumlar.get(int(k))
            if y:
                st.markdown(
                    f"🛡️ **İkili — güvenli:** {y.get('ikili_guvenli', '-')}  \n"
                    f"🎯 **İkili — sürpriz:** {y.get('ikili_surpriz', '-')}  \n"
                    f"🧮 **İkili — kolon:** {y.get('ikili_kolon', '-')}  \n"
                    f"🔗 **Çifte:** {y.get('cifte', '-')}")
                a = ss.get(llm_anahtar + "|aday", {}).get(int(k))
                if a and a.get("ikili_kolonlar"):
                    if st.checkbox("Hesaplanan adayları göster", key=f"aday_{int(k)}"):
                        st.dataframe(pd.DataFrame(a["ikili_guvenli"] + a["ikili_surpriz"]),
                                     hide_index=True)
                        st.dataframe(pd.DataFrame(a["ikili_kolonlar"]), hide_index=True)
                        if a.get("cifte"):
                            st.dataframe(pd.DataFrame(a["cifte"]["guvenli"]
                                                      + a["cifte"]["surpriz"]),
                                         hide_index=True)

    if surpriz_sayaci == 0:
        st.info("Bugün sürpriz adayı bulunamadı — model halkla hemfikir. "
                "Eşikleri gevşetmeyi deneyebilirsiniz.")

# --- Altılı ----------------------------------------------------------------
with sekme2:
    ayaklar = [k for k in kosular if k >= ilk_ayak][:6]
    if len(ayaklar) < 6:
        st.warning(f"{ilk_ayak}. koşudan itibaren 6 ayak yok "
                   f"(mevcut: {len(ayaklar)}).")
    else:
        kosu_nolar = [int(k) for k in ayaklar]
        legs = [pred[pred["kosu_no"] == k] for k in ayaklar]
        kuponlar = altili_tablosu(legs, kosu_nolar, misli)
        st.markdown(f"**Ayaklar:** {' → '.join(str(k) for k in kosu_nolar)}. koşular"
                    + (f"  ·  _üretim {uretim:%H:%M}_" if uretim else ""))

        st.subheader("Hazır kuponlar")
        kolonlar = st.columns(len(kuponlar))
        for kol, (ad, veri) in zip(kolonlar, kuponlar.items()):
            with kol:
                st.markdown(f"**{ad}**")
                st.metric("Kombinasyon", f"{veri['kombinasyon']:,}")
                st.metric("Maliyet", f"{veri['maliyet']:,.0f} TL")
                st.metric("İsabet olasılığı", f"%{100*veri['p_isabet']:.2f}")
                st.dataframe(veri["tablo"][["Ayak", "At sayısı", "Seçimler"]],
                             hide_index=True)

        # --- Kupon laboratuvarı: bütçe × strateji ---------------------------------
        st.divider()
        st.subheader("🧪 Kupon laboratuvarı")
        st.caption("Farklı bütçe ve stratejilerde kupon üretir. Halka göre avantaj = "
                   "model isabet % / halk (AGF) isabet %. 1'in üstü: kupon halkın "
                   "küçümsediği atları içeriyor, tutarsa ikramiye ortalamadan yüksek olur.")
        l1, l2, l3, l4 = st.columns(4)
        b_min = l1.number_input("En düşük bütçe (TL)", 10.0, 100000.0, 50.0, 10.0)
        b_max = l2.number_input("En yüksek bütçe (TL)", 20.0, 500000.0, 3000.0, 50.0)
        adim = l3.slider("Bütçe adımı", 3, 12, 7)
        min_at = l4.slider("Ayak başı en az at", 1, 4, 1,
                           help="2 seçilirse hiç tek atlı (banko) ayak bırakılmaz")
        s1, s2 = st.columns([3, 1])
        stratejiler = s1.multiselect("Stratejiler", list(altili_plus.STRATEJILER),
                                     default=list(altili_plus.STRATEJILER),
                                     help="Olasılık: en olası atlar · Dengeli: olasılık × "
                                          "√değer · Değer: halkın geride bıraktığı atlar önce")
        max_at = s2.slider("Ayak başı en çok at", 2, 12, 8)

        ayak_df = [altili_plus.ayak_hazirla(l) for l in legs]
        izgara = altili_plus.butce_izgarasi(b_min, max(b_max, b_min + 1), adim)
        lab = altili_plus.varyasyonlar(
            ayak_df, misli, izgara,
            {k: v for k, v in altili_plus.STRATEJILER.items() if k in stratejiler},
            min_at=min_at, max_at=max_at)
        if not lab:
            st.warning("Bu bütçe aralığında kupon kurulamadı — en düşük bütçeyi artırın "
                       "veya 'en az at' değerini düşürün.")
        else:
            ozet = altili_plus.ozet_tablosu(lab, kosu_nolar)
            st.dataframe(ozet, hide_index=True, column_config={
                "Halka göre avantaj": st.column_config.ProgressColumn(
                    "Halka göre avantaj", min_value=0.0, max_value=3.0, format="%.2f")})
            secim = st.selectbox("Kupon detayı", ozet["#"].tolist(),
                                 format_func=lambda i: (
                                     f"#{i} · {lab[i-1]['strateji']} · "
                                     f"{lab[i-1]['maliyet']:,.0f} TL · "
                                     f"%{100*lab[i-1]['p_isabet']:.2f}"))
            k_sec = lab[secim - 1]
            st.dataframe(altili_plus.kupon_detay(ayak_df, k_sec, kosu_nolar),
                         hide_index=True)
            alt = altili_plus.tek_at_alternatifleri(ayak_df, k_sec, kosu_nolar)
            if not alt.empty:
                st.markdown("**Tek at bırakılan ayaklarda olası diğer atlar**")
                st.dataframe(alt, hide_index=True)
            st.caption(
                f"Bu kupon isabet ederse dağıtılan ikramiyenin "
                f"{k_sec['maliyet']/max(k_sec['p_isabet'],1e-9):,.0f} TL üzerinde "
                f"olması uzun vadeli kârlılık için gerekir.")

        with st.expander("Her ayakta yedek atlar (kümülatif olasılık)"):
            st.dataframe(altili_plus.yedek_at_tablosu(ayak_df, kosu_nolar),
                         hide_index=True)


# --- Günün sonuçları (EDA) -----------------------------------------------------
with sekme4:
    @st.cache_data(ttl=600, show_spinner="Sonuçlar çekiliyor…")
    def _sonuclar(t: dt.date, liste: tuple):
        parca = [sonuclar.gunluk_sonuclar(t, ad, sid) for sid, ad in liste]
        parca = [p for p in parca if not p.empty]
        return pd.concat(parca, ignore_index=True) if parca else pd.DataFrame()

    e1, e2 = st.columns([1, 2])
    tum = e2.checkbox("Tüm yurt içi hipodromlar", value=False)
    eda_anahtar = f"eda|{'|'.join(gun_anahtari)}|{tum}"
    if e1.button("📊 Sonuçları getir"):
        ss[eda_anahtar] = True

    if ss.get(eda_anahtar):
        liste = tuple(sehirler) if tum else (secili,)
        sdf = _sonuclar(tarih, liste)
        if sdf.empty:
            st.info("Sonuç bulunamadı — koşular henüz bitmemiş ya da TJK sonuç "
                    "dosyası yayınlanmamış olabilir.")
        else:
            sdf = eda.hazirla(sdf)
            w = eda.kazananlar(sdf, agf_esigi=agf_esigi)
            oz = eda.ozet(sdf, w)
            k1, k2, k3, k4 = st.columns(4)
            k1.metric("Koşu", oz["kosu"])
            k2.metric("Favori kazandı",
                      "-" if pd.isna(oz["fav_kazanma"]) else f"%{100*oz['fav_kazanma']:.0f}")
            k3.metric("Kazanan ganyan (medyan)",
                      "-" if pd.isna(oz["ort_ganyan"]) else f"{oz['ort_ganyan']:.2f}")
            k4.metric("Sürpriz kazanan", oz["surpriz"])

            st.subheader("Kazananlar — at, jokey, antrenör, sahip")
            st.dataframe(w.rename(columns={
                "hipodrom": "Hipodrom", "kosu_no": "Koşu", "saat": "Saat",
                "mesafe": "Mesafe", "pist": "Pist", "no": "No", "at": "At",
                "jokey": "Jokey", "antrenor": "Antrenör", "sahip": "Sahip",
                "ganyan": "Ganyan", "agf": "AGF", "agf_sira": "AGF sıra",
                "derece": "Derece", "surpriz": "Sürpriz"}),
                hide_index=True)

            g1 = eda.kazanan_ganyan_grafik(w)
            if g1 is not None:
                st.altair_chart(g1)

            sekmeler = st.tabs(["Jokey", "Antrenör", "Sahip"])
            for sek, (kol, ad) in zip(sekmeler, [("jokey", "Jokey"),
                                                 ("antrenor", "Antrenör"),
                                                 ("sahip", "Sahip")]):
                with sek:
                    t = eda.kisi_tablosu(sdf, kol)
                    if t.empty:
                        st.caption(f"{ad} bilgisi sonuç dosyasında yok.")
                        continue
                    a, b = st.columns([3, 2])
                    a.altair_chart(eda.kisi_grafik(t, kol, f"{ad} — günün ilk 3'leri"))
                    b.dataframe(t.rename(columns={kol: ad, "binis": "Biniş",
                                                  "birinci": "1.", "ilk3": "İlk 3",
                                                  "kazanma_%": "Kazanma %"})
                                .drop(columns="ikinci_ucuncu"),
                                hide_index=True, height=420)

            g2 = eda.ganyan_sira_grafik(sdf)
            if g2 is not None:
                st.altair_chart(g2)

            if not tum:
                mv = eda.model_vs_sonuc(sdf, pred)
                if not mv.empty:
                    st.subheader("Model tahmini vs sonuç")
                    isabet = (mv["Bitirdiği sıra"] == 1).mean()
                    st.caption(f"Modelin 1 numarası {len(mv)} koşunun "
                               f"%{100*isabet:.0f}'ünü kazandı.")
                    st.dataframe(mv, hide_index=True)

            with st.expander("Tüm sonuç satırları"):
                st.dataframe(sdf, hide_index=True)
            st.download_button("CSV indir", sdf.to_csv(index=False).encode("utf-8-sig"),
                               file_name=f"sonuclar_{tarih:%Y%m%d}.csv", mime="text/csv")
    else:
        st.info("Seçili günün sonuçlarını görmek için **Sonuçları getir**'e basın.")

st.divider()
st.caption("Model geçmiş testlerde ganyan piyasasını geçemedi. İkili ve Altılı "
           "havuzlarındaki performansı henüz ölçülmedi. Nihai karar sizin.")
