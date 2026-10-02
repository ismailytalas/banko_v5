# -*- coding: utf-8 -*-
"""
Bahis kurgusu: İkili ve Altılı Ganyan.

İkili (quinella): Bir koşuda ilk iki atı sırasız bilmek.
  Harville modeli: P(i birinci, j ikinci) = p_i * p_j / (1 - p_i)
  P(ikili{i,j}) = p_i*p_j/(1-p_i) + p_j*p_i/(1-p_j)

Altılı Ganyan (pick-6): Ardışık 6 koşunun kazananını bilmek.
  Her ayakta seçilen atların p_win toplamı o ayağın isabet olasılığıdır;
  kupon isabeti = ayak olasılıklarının çarpımı.
  Kupon maliyeti = kombinasyon sayısı x birim fiyat.
  Greedy genişletme: bütçe dolana kadar "en çok olasılık kazandıran" atı ekle.
"""
from __future__ import annotations

from itertools import combinations

import numpy as np
import pandas as pd


def harville_top2(p: pd.Series) -> pd.DataFrame:
    """p: at_no -> p_win. Tüm {i,j} ikilileri için sırasız ilk-2 olasılığı."""
    p = p / p.sum()
    rows = []
    for i, j in combinations(p.index, 2):
        pi, pj = p[i], p[j]
        prob = pi * pj / max(1 - pi, 1e-9) + pj * pi / max(1 - pj, 1e-9)
        rows.append({"at1": i, "at2": j, "p_ikili": prob})
    return (pd.DataFrame(rows)
            .sort_values("p_ikili", ascending=False)
            .reset_index(drop=True))


def suggest_ikili(race_df: pd.DataFrame, top_n: int = 5,
                  no_col: str = "no", p_col: str = "p_win") -> pd.DataFrame:
    """Bir koşu için en olası ikili kombinasyonları (at isimleriyle) döndürür."""
    d = race_df.dropna(subset=[p_col]).set_index(no_col)
    if len(d) < 2:
        return pd.DataFrame()
    combos = harville_top2(d[p_col]).head(top_n)
    name = d["at"].to_dict() if "at" in d else {}
    def _fmt(x):
        try:
            return str(int(x))
        except (TypeError, ValueError):
            return str(x)
    combos["kombinasyon"] = combos.apply(
        lambda r: f"{_fmt(r.at1)}-{_fmt(r.at2)}  "
                  f"({name.get(r.at1,'?')} / {name.get(r.at2,'?')})", axis=1)
    return combos[["kombinasyon", "p_ikili"]]


def build_pick6(legs: list[pd.DataFrame], budget: float, unit_price: float,
                no_col: str = "no", p_col: str = "p_win",
                min_per_leg: int = 1, max_per_leg: int = 6) -> dict:
    """Altılı Ganyan kuponu kur.

    legs: 6 koşunun her biri için (no, at, p_win) DataFrame listesi
    budget / unit_price: toplam bütçe ve kombinasyon başı misli fiyatı

    Dönen: {"secimler": [[no,...] x6], "kombinasyon": int, "maliyet": float,
            "p_isabet": float, "detay": DataFrame}
    """
    # Her ayakta olasılığa göre sıralı at listesi
    sorted_legs = []
    for leg in legs:
        d = leg.dropna(subset=[p_col]).sort_values(p_col, ascending=False)
        sorted_legs.append(list(zip(d[no_col], d[p_col],
                                    d["at"] if "at" in d else d[no_col])))

    picks = [lst[:min_per_leg] for lst in sorted_legs]

    def cost(pk):
        n = 1
        for leg in pk:
            n *= len(leg)
        return n * unit_price, n

    # Greedy: her adımda "eklenecek atın marjinal olasılık katkısı / maliyet artışı"
    # oranı en yüksek ayağı genişlet
    while True:
        best = None
        cur_cost, _ = cost(picks)
        for li, lst in enumerate(sorted_legs):
            k = len(picks[li])
            if k >= min(max_per_leg, len(lst)):
                continue
            cand = list(picks[li]) + [lst[k]]
            trial = picks[:li] + [cand] + picks[li + 1:]
            new_cost, _ = cost(trial)
            if new_cost > budget:
                continue
            leg_p_old = sum(p for _, p, _ in picks[li])
            leg_p_new = sum(p for _, p, _ in cand)
            gain = (leg_p_new / max(leg_p_old, 1e-9))  # çarpımsal kazanç
            ratio = (gain - 1) / max(new_cost - cur_cost, 1e-9)
            if best is None or ratio > best[0]:
                best = (ratio, li, cand)
        if best is None:
            break
        picks[best[1]] = best[2]

    total_cost, n_comb = cost(picks)
    p_hit = float(np.prod([sum(p for _, p, _ in leg) for leg in picks]))
    detay = pd.DataFrame({
        f"{i+1}. Ayak": [", ".join(f"{no} ({name})" for no, _, name in leg)]
        for i, leg in enumerate(picks)
    }).T.rename(columns={0: "Seçimler"})
    return {
        "secimler": [[no for no, _, _ in leg] for leg in picks],
        "kombinasyon": n_comb,
        "maliyet": total_cost,
        "p_isabet": p_hit,
        "detay": detay,
    }


def expected_value_note(p: float, cost: float) -> str:
    """Basit bilgilendirme metni."""
    return (f"Kupon isabet olasılığı ≈ %{100*p:.2f}. Uzun vadede kârlılık, "
            f"isabet halinde dağıtılan ikramiyenin {cost/max(p,1e-9):,.0f} TL "
            f"üzerinde olmasına bağlıdır.")
