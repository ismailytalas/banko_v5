# -*- coding: utf-8 -*-
"""TJK Tahmin — Streamlit önyüzü.

Çalıştırma:  streamlit run app.py
"""
import datetime as dt

import pandas as pd
import streamlit as st

from tjk import scraper
from tjk.features import build_features
from tjk.model import load, predict_program, market_baseline, RACE_KEY
from tjk.bets import suggest_ikili, build_pick6, expected_value_note

st.set_page_config(page_title="TJK Tahmin", page_icon="🐎", layout="wide")
st.title("🐎 TJK Günlük Yarış Tahminleri")
st.caption("Tahminler istatistiksel modeldir, kazanç garantisi yoktur. "
           "Sorumlu oyun: kaybetmeyi göze alabileceğiniz bütçeyle oynayın.")

# --- Kenar çubuğu -----------------------------------------------------------
with st.sidebar:
    date = st.date_input("Tarih", dt.date.today(), format="DD/MM/YYYY")
    st.divider()
    st.subheader("Altılı Ganyan Ayarları")
    budget = st.number_input("Bütçe (TL)", 20.0, 100000.0, 300.0, step=20.0)
    unit = st.number_input("Misli birim fiyatı (TL)", 0.5, 100.0, 3.0, step=0.5)
    max_per_leg = st.slider("Ayak başı en fazla at", 1, 8, 5)
    first_leg = st.number_input("Altılı başlangıç koşusu", 1, 12, 1)
    st.divider()
    st.session_state.setdefault("yenile", 0)
    if st.button("🔄 Değişkenleri yeniden üret",
                 help="Programı ve AGF'yi TJK'dan yeniden çekip tahminleri günceller"):
        st.session_state.yenile += 1


@st.cache_data(ttl=1800, show_spinner="Hipodromlar yükleniyor…")
def cities_for(d: dt.date):
    return scraper.get_cities(d)

try:
    cities = cities_for(date)
except Exception as e:
    st.error(f"TJK'ya ulaşılamadı: {e}")
    st.stop()

if not cities:
    st.warning("Bu tarihte koşu bulunamadı.")
    st.stop()

domestic = [(sid, n) for sid, n in cities
            if not scraper.YURTDISI_PATTERN.search(n)]
label = st.selectbox("Hipodrom", domestic + [c for c in cities if c not in domestic],
                     format_func=lambda c: c[1])
sid, city = label


@st.cache_data(ttl=900, show_spinner="Program indiriliyor…")
def program_for(d: dt.date, sid: int, city: str, token: int):
    return scraper.get_program(d, sid, city)

raw = program_for(date, sid, city, st.session_state.yenile)
st.caption(f"Program sürümü: {st.session_state.yenile} · "
           f"{dt.datetime.now():%H:%M:%S}")
if raw.empty:
    st.warning("Program verisi alınamadı.")
    st.stop()

feats = build_features(raw)

# Model varsa kullan, yoksa piyasa (ganyan+AGF) temeline düş
try:
    booster, fcols = load("models/model")
    pred = predict_program(feats, booster, fcols)
    st.success("Eğitilmiş model kullanılıyor.")
except Exception:
    pred = market_baseline(feats)
    st.info("Eğitilmiş model bulunamadı — sabah ganyanı + AGF temelli "
            "tahmin gösteriliyor. `python scripts/train.py` ile model eğitin.")

pred["no"] = pd.to_numeric(pred.get("no"), errors="coerce")
pred = pred.dropna(subset=["no", "p_win"])
pred["no"] = pred["no"].astype(int)

races = sorted(pred["kosu_no"].dropna().unique())

# --- Koşu bazlı görünüm -----------------------------------------------------
tab1, tab2 = st.tabs(["Koşu Tahminleri & İkili", "Altılı Ganyan Kuponu"])

with tab1:
    for k in races:
        rd = pred[pred["kosu_no"] == k].sort_values("p_win", ascending=False)
        meta = rd.iloc[0]
        st.subheader(f"{int(k)}. Koşu"
                     + (f" – {meta['saat']}" if pd.notna(meta.get('saat')) else "")
                     + (f" – {int(meta['mesafe'])}m {meta.get('pist') or ''}"
                        if pd.notna(meta.get('mesafe')) else ""))
        show = rd[["no", "at", "Jokey", "hp", "gny", "agf", "p_win"]].copy()
        show["p_win"] = (100 * show["p_win"]).round(1)
        show["agf"] = (100 * show["agf"]).round(1)
        show.columns = ["No", "At", "Jokey", "HP", "Ganyan", "AGF %", "Kazanma %"]
        st.dataframe(show, hide_index=True)

        ikili = suggest_ikili(rd, top_n=5)
        if not ikili.empty:
            ikili["p_ikili"] = (100 * ikili["p_ikili"]).round(1)
            ikili.columns = ["İkili Kombinasyonu", "Olasılık %"]
            st.markdown("**En olası İkili kombinasyonları**")
            st.dataframe(ikili, hide_index=True)
        st.divider()

with tab2:
    legs_no = [k for k in races if k >= first_leg][:6]
    if len(legs_no) < 6:
        st.warning(f"{first_leg}. koşudan itibaren 6 ayak bulunamadı "
                   f"(mevcut: {len(legs_no)}).")
    else:
        legs = [pred[pred["kosu_no"] == k] for k in legs_no]
        ticket = build_pick6(legs, budget=budget, unit_price=unit,
                             max_per_leg=max_per_leg)
        c1, c2, c3 = st.columns(3)
        c1.metric("Kombinasyon", ticket["kombinasyon"])
        c2.metric("Maliyet", f"{ticket['maliyet']:,.0f} TL")
        c3.metric("İsabet Olasılığı", f"%{100*ticket['p_isabet']:.2f}")
        st.table(ticket["detay"])
        st.caption(expected_value_note(ticket["p_isabet"], ticket["maliyet"]))
