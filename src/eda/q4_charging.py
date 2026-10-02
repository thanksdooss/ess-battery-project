"""Q4. 충전 조건 (C-rate)과 수명의 관계는?

노션 하위 질문
  (a) 충전 프로토콜별 평균 수명 비교
  (b) 고속 충전 셀이 정말 수명이 짧은가?
  (c) 충전 전류 패턴과 열화 속도 상관 분석
  (d) 전략 연결(추가 점검): 충전 피처의 dQ_logvar 대비 추가 정보, 반복 셀 편차, Qcc_init 배치 이동(Q5-(d) 교차 확인)

DAY 1 EDA 전용 — 모델 학습/튜닝 없음. 상관·기술통계만 계산한다.
특징 선택 근거는 Batch 1(학습셋) + 도메인 논리, Batch 2/3 는 일반화 위험 확인용으로만 인용.

실행: cd ess-battery-project && python src/eda/q4_charging.py
산출: reports/figures/q4_*.png (q4_pdf_* = 보고서 PDF 용 단순화 그림), results/eda/q4_results.json, q4_cell_features.csv
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
from data import *  # noqa: E402,F401,F403
from data import LABEL_THRESHOLD, clean_summary  # noqa: E402
import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

RAW_ROWS = (2, 5, 10)   # cell['raw'] 키 = 사이클 행 인덱스 → cycle 3·6·11. 기술자는 세 사이클 중앙값 사용
RAW_ROW = 10            # 그림 예시 I(t) 에 쓰는 행 (cycle 11)
RAW_ROW_CHECK = 2       # 재현성 확인용 (행 2 = cycle 3 vs 행 10 = cycle 11)
HI_C = 1.5          # '고속 충전 단계' 판정 전류 [C]. 이후 단계는 1C CC-CV 또는 휴지(0)
T_VALID = (0.0, 80.0)   # 원시/요약 온도 센서 이상값(예: 400°C) 제외 범위
GRAY, MUTED = "#9CA3AF", "#6B7280"
NS_MARK, FC_MARK = "D", "o"   # newstructure / fastcharge 마커


# ───────────────────────── 유틸 ─────────────────────────
def sp(x, y):
    """Spearman ρ, p, n (결측 제외)."""
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    if m.sum() < 4 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
        return {"rho": None, "p": None, "n": int(m.sum())}
    r, p = stats.spearmanr(x[m], y[m])
    return {"rho": round(float(r), 3), "p": float(f"{p:.3g}"), "n": int(m.sum())}


def pr(x, y):
    x, y = np.asarray(x, float), np.asarray(y, float)
    m = np.isfinite(x) & np.isfinite(y)
    r, p = stats.pearsonr(x[m], y[m])
    return {"r": round(float(r), 3), "p": float(f"{p:.3g}"), "n": int(m.sum())}


def base_policy(p: str) -> str:
    return p.replace("-newstructure", "")


def iqw_policy(c1, q1, c2):
    """정책 문자열로 계산한 충전량 가중 평균 전류 = Σ C_k·ΔSOC_k / 0.8 (0→80%).

    노션 정의는 'Q1%까지 C1, 이후 나머지는 C2' 이지만, 원시 I(t) 확인 결과 대부분 셀은 80% 에서
    1C CC-CV(3.6 V)로 전환된다(원논문 프로토콜). 그래서 C2 단계는 0→80% 구간까지만 계산한다.
    """
    return (c1 * q1 + c2 * max(80 - q1, 0)) / 80


# ─────────────── 원시 I(t) → 충전 전류 패턴 기술자 ───────────────
def current_descriptors(r: pd.DataFrame) -> dict:
    """한 사이클 원시 시계열에서 충전 구간(I>0, 방전 시작 전) 전류 패턴 기술자.

    I 열은 C-rate 단위(1C = 1.1 A): dQc/dt ≈ 1.1·I 로 확인 (qc_per_c 로 검증).
    """
    t, i_c = r["t"].to_numpy(float), r["I"].to_numpy(float)
    temp = r["T"].to_numpy(float).copy()
    temp[(temp <= T_VALID[0]) | (temp > T_VALID[1])] = np.nan
    qc = r["Qc"].to_numpy(float)
    idx = np.arange(len(i_c))
    st = int(np.argmax(i_c > 0.1))                                  # 충전 시작
    dis_cand = idx[(idx > st) & (i_c < -0.5)]
    dis = int(dis_cand[0]) if len(dis_cand) else len(i_c) - 1      # 방전 시작
    seg = slice(st, dis + 1)
    ts, Is = t[seg], np.clip(i_c[seg], 0, None)
    # 누적 충전량(정격 대비 분율, C·h): 0.8 도달 시점 = 실측 t80
    cum = np.concatenate([[0], np.cumsum(np.diff(ts) * (Is[1:] + Is[:-1]) / 2)]) / 60
    # 0.80 직전에 고속 단계가 끝나는 정책은 적분 오차(≈0.1%)만으로 0.80 도달이 다음 1C 단계로 밀릴 수 있어
    # 0.79 도달 시점에서 그 순간 전류로 남은 0.01 을 외삽 → t80 (정책 정의와 동일한 기준)
    if cum[-1] >= 0.8:
        t79 = float(np.interp(0.79, cum, ts))
        t80_abs = t79 + 0.01 / max(float(np.interp(t79, ts, Is)), 1e-6) * 60
        t80 = t80_abs - ts[0]
    else:
        t80_abs, t80 = ts[-1], np.nan
    w = ts <= t80_abs
    tw, Iw = ts[w], Is[w]
    i_pk = float(np.percentile(Is[Is >= HI_C], 99)) if (Is >= HI_C).any() else np.nan
    dt = np.diff(ts)
    imid = (Is[1:] + Is[:-1]) / 2
    vmid = r["V"].to_numpy(float)[seg][:-1]
    q_hi = float(np.sum(dt[imid >= HI_C] * imid[imid >= HI_C]) / 60)
    q_w = np.trapezoid(Iw, tw)
    # 80% 이후 단계 확인용: 누적 충전량 0.80~0.95 구간의 충전량 가중 평균 전류, 고전류 단계 종료 전압
    w2 = (cum >= 0.80) & (cum <= 0.95)
    i_post = (float(np.trapezoid(Is[w2] ** 2, ts[w2]) / np.trapezoid(Is[w2], ts[w2]))
              if w2.sum() > 3 and cum[-1] >= 0.9 else np.nan)
    hi_idx = np.where(Is >= HI_C)[0]
    v_hi_end = float(r["V"].to_numpy(float)[seg][hi_idx[-1]]) if len(hi_idx) else np.nan
    return {
        "I_peak": i_pk,                                            # 피크 전류 [C]
        "t_peak": float(dt[imid >= 0.95 * i_pk].sum()),            # 피크 전류(≥95%) 유지 시간 [분]
        "t80_meas": t80,                                           # 0→80% 실측 충전시간 [분]
        "I_avg80": 0.8 / (t80 / 60) if np.isfinite(t80) else np.nan,   # 실측 평균 C-rate [C]
        "I_qw80": float(np.trapezoid(Iw ** 2, tw) / q_w),           # 충전량 가중 평균 전류 [C] = ∫I²dt/∫Idt
        "t_chg": float(ts[-1] - ts[0]),                             # 충전 단계 총 시간(휴지·CV 포함) [분]
        "t_rest_chg": float(dt[imid < 0.01].sum()),                 # 충전 단계 내 휴지(I≈0) 시간 [분]
        "t_cv": float(dt[(imid >= 0.01) & (vmid >= 3.595)].sum()),  # 3.6 V 정전압(CV) 유지 시간 [분]
        "Q_hi": q_hi,                                               # ≥1.5C 로 넣은 충전량 [정격 대비 분율]
        "I_post80": i_post,                                         # 누적 0.80~0.95 구간 가중 평균 전류 [C]
        "V_hi_end": v_hi_end,                                       # 고전류(≥1.5C) 단계 종료 시점 전압 [V]
        "dT_chg": float(np.nanmax(temp[seg][w]) - temp[st]) if np.isfinite(temp[st]) else np.nan,
        "qc_per_c": float(np.interp(t80 + ts[0], t, qc) / 0.8) if np.isfinite(t80) else np.nan,
    }


# ─────────────── summary → 열화 속도 지표 ───────────────
def degradation_measures(c: dict) -> dict:
    s = clean_summary(c["summary"])
    for col in ("Tmax", "Tavg", "Tmin"):
        s.loc[s[col] > T_VALID[1], col] = np.nan
    cyc = s["cycle"]

    def med(col, a, b):
        v = s.loc[(cyc >= a) & (cyc <= b), col]
        return float(np.nanmedian(v)) if v.notna().any() else np.nan

    m = (cyc >= 2) & (cyc <= 100) & s["QD"].notna()
    slope = stats.theilslopes(s.loc[m, "QD"], cyc[m])[0] if m.sum() > 10 else np.nan
    # 중기 열화 기울기(사이클 100–300): 수명 값으로 계산하지 않는 '측정된' 열화 속도 — 모든 셀이 ≥392 사이클이라
    # 중도절단 셀도 계산 가능. 100 사이클 이후 정보라 모델 입력으로는 쓸 수 없고 EDA(열화 속도 분석) 전용.
    m2 = (cyc >= 100) & (cyc <= 300) & s["QD"].notna()
    slope2 = stats.theilslopes(s.loc[m2, "QD"], cyc[m2])[0] if m2.sum() > 20 else np.nan
    qd0 = med("QD", 2, 11)
    life = c["cycle_life"]
    ct = s.loc[(cyc >= 2) & (cyc <= 100), "chargetime"]
    return {
        # 공통 규약: 용량 기울기 [mAh/100 cycle], 음수 = 감소, 추정량 Theil–Sen
        "slope_2_100": slope * 1000 * 100,                          # 초기 용량 기울기 [mAh/100cyc] (음수 = 감소)
        "slope_100_300": slope2 * 1000 * 100,                       # 중기 용량 기울기 [mAh/100cyc] (음수 = 감소)
        "QD_init": qd0,
        "fade_life": (qd0 - EOL_AH) / life * 1000 if np.isfinite(life) else np.nan,  # 전체 열화율 [mAh/cyc]
        "dTmax": med("Tmax", 91, 100) - med("Tmax", 6, 15),          # Tmax 상승 [°C]
        "dIR": (med("IR", 91, 100) - med("IR", 2, 11)) * 1000,      # IR 증가 [mΩ]
        "ct_med": float(np.nanmedian(ct)) if ct.notna().any() else np.nan,   # summary chargetime 중앙값 [분]
        "ct_removed": int(((c["summary"]["chargetime"] > 100) | (c["summary"]["chargetime"] <= 0)).sum()),
    }


# ───────────────────────── 데이터 구성 ─────────────────────────
cells = load_all()
by_key = {c["cell_key"]: c for c in cells}
tab = cell_table(cells).set_index("cell_key")
rows = []
for c in cells:
    d = {"cell_key": c["cell_key"]}
    per_row = {row: current_descriptors(c["raw"][row]) for row in RAW_ROWS if row in c["raw"]}
    for k in per_row[RAW_ROW]:
        d[k] = float(np.nanmedian([v[k] for v in per_row.values()]))   # cycle 3·6·11 중앙값 (단일 사이클 글리치 완화)
        d[f"{k}_r10"] = per_row[RAW_ROW][k]
        d[f"{k}_chk"] = per_row[RAW_ROW_CHECK][k] if RAW_ROW_CHECK in per_row else np.nan
    d.update(degradation_measures(c))
    rows.append(d)
D = pd.DataFrame(rows).set_index("cell_key").join(tab)
D["I_qw_policy"] = [iqw_policy(a, b, c_) for a, b, c_ in zip(D.C1, D.Q1, D.C2)]
D["base_policy"] = D.policy.map(base_policy)
# 중도절단·EOL 미도달 셀의 수명 하한값 (B1 censored: 기록된 cycle_life, B3 NaN: 기록 사이클 수)
D["life_lb"] = np.where(D.labeled, np.nan,
                        np.where(D.cycle_life.notna(), D.cycle_life, D.n_cycles))
LIFE_GROUPS = ("fastcharge", "newstructure")
D_ALL = D.copy()                                   # chargetime 제외 개수 등 배치 전체 기준 집계용
D = D[D.group.isin(LIFE_GROUPS) | D.group.isna()]  # varcharge·slowcycle(수명 시험 아님) 제외
R: dict = {"meta": {
    "raw_rows_used": list(RAW_ROWS), "raw_cycles_used": [r + 1 for r in RAW_ROWS],
    "descriptor_aggregation": "cycle 3·6·11 기술자 중앙값",
    "cells_by_batch_group": {f"{b}-{g}": int(n) for (b, g), n in D.groupby(["batch", "group"]).size().items()},
    "note": "varcharge/slowcycle(Batch 2 8셀)은 수명 라벨이 없어 제외",
}}
LAB = D[D.labeled]
b1, b1_all = LAB[LAB.batch == "batch1"], D[D.batch == "batch1"]
b2, b2_fc = LAB[LAB.batch == "batch2"], LAB[(LAB.batch == "batch2") & (LAB.group == "fastcharge")]
b3 = LAB[LAB.batch == "batch3"]

# ═════════════════ (a) 충전 프로토콜별 평균 수명 ═════════════════
pol_rows = []
for (b, p), g in D.groupby(["batch", "policy"]):
    lab = g[g.labeled]
    pol_rows.append({
        "batch": b, "policy": p, "group": g.group.iloc[0],
        "n_labeled": len(lab), "n_censored_or_nan": int((~g.labeled).sum()),
        "mean": lab.cycle_life.mean() if len(lab) else np.nan,
        "std": lab.cycle_life.std(ddof=1) if len(lab) > 1 else np.nan,
        "min": lab.cycle_life.min() if len(lab) else np.nan,
        "max": lab.cycle_life.max() if len(lab) else np.nan,
        "lower_bounds": sorted(g.life_lb.dropna().astype(int).tolist()),
        "t80_policy": g.t80_min.iloc[0], "avgC_policy": g.avgC_80.iloc[0],
        "I_qw_policy": g.I_qw_policy.iloc[0], "t80_meas": g.t80_meas.mean(), "I_avg80_meas": g.I_avg80.mean(),
        "I_qw_meas": g.I_qw80.mean(),
    })
P = pd.DataFrame(pol_rows)


def omega_sq(df):
    """정책(그룹)이 설명하는 수명 분산 비율 — 셀 1~2개 그룹이 많아 η² 는 과대추정되므로 ω² 를 같이 보고."""
    g = df.groupby("policy").cycle_life
    n, k = len(df), g.ngroups
    sst = ((df.cycle_life - df.cycle_life.mean()) ** 2).sum()
    ssb = (g.size() * (g.mean() - df.cycle_life.mean()) ** 2).sum()
    msw = (sst - ssb) / (n - k)
    kw = stats.kruskal(*[v.values for _, v in g])
    return {"n_cells": n, "n_policies": k, "eta2": round(ssb / sst, 3),
            "omega2": round((ssb - (k - 1) * msw) / (sst + msw), 3),
            "eta2_null_expect": round((k - 1) / (n - 1), 3),
            "kruskal_H": round(float(kw.statistic), 2), "kruskal_p": float(f"{kw.pvalue:.3g}")}


same_name = {}
for bp in ("4.8C(80%)-4.8C", "5.2C(58%)-4C", "5.6C(26%)-4.5C", "6C(60%)-3C"):
    s = LAB[LAB.base_policy == bp]
    same_name[bp] = {f"{b}-{g}": {"n": int(len(v)), "mean": round(v.cycle_life.mean(), 1),
                                  "min": int(v.cycle_life.min()), "max": int(v.cycle_life.max()),
                                  "I_peak_meas_median": round(float(v.I_peak.median()), 2)}
                     for (b, g), v in s.groupby(["batch", "group"])}
# 같은 정책명 중 실측 전류도 같은(피크 ≈ 4.8C) 그룹만 비교 — Batch 3 해당 셀은 실측 4.36C 라 제외
_sn48 = same_name["4.8C(80%)-4.8C"]
_same_cur = {k: v for k, v in _sn48.items() if abs(v["I_peak_meas_median"] - 4.8) < 0.1}
same_name_same_current = {
    "groups": _same_cur,
    "ratio_max_over_min": round(max(v["mean"] for v in _same_cur.values()) / min(v["mean"] for v in _same_cur.values()), 2),
    "ratio_incl_batch3_name_only": round(max(v["mean"] for v in _sn48.values()) / min(v["mean"] for v in _sn48.values()), 2),
}
# B2: 같은 정책명 fastcharge vs newstructure — NS 최솟값 vs FC 최댓값
b2_pairs = {bp: {"fc_max": int(b2[(b2.base_policy == bp) & (b2.group == "fastcharge")].cycle_life.max()),
                 "ns_min": int(b2[(b2.base_policy == bp) & (b2.group == "newstructure")].cycle_life.min())}
            for bp in ("4.8C(80%)-4.8C", "5.2C(58%)-4C", "5.6C(26%)-4.5C")}
R["a_policy_life"] = {
    "table": P.round(3).replace({np.nan: None}).to_dict(orient="records"),
    "policy_effect": {b: omega_sq(LAB[LAB.batch == b]) for b in BATCH_NAMES},
    "policy_effect_b2_fastcharge_only": omega_sq(b2_fc),
    "same_policy_name_across_batches": same_name,
    "same_policy_name_4.8C_same_measured_current": same_name_same_current,
    "b2_fastcharge_vs_newstructure_same_name": b2_pairs,
    "b1_fully_censored_policies": P[(P.batch == "batch1") & (P.n_labeled == 0)].policy.tolist(),
}

# ═════════════════ (b) 고속 충전 셀이 정말 수명이 짧은가? ═════════════════
vars_b = ["C1", "Q1", "C2", "t80_min", "avgC_80", "I_qw_policy", "I_qw80", "I_peak"]
corr_b1 = {v: {"spearman": sp(b1[v], b1.cycle_life), "pearson": pr(b1[v], b1.cycle_life)} for v in vars_b}
pl = b1.groupby("policy").agg(life=("cycle_life", "mean"), **{v: (v, "mean") for v in vars_b})
corr_b1_policy = {v: sp(pl[v], pl.life) for v in vars_b}
# 민감도: 중도절단 10셀을 하한값(cycle_life) 그대로 포함 — 실제 수명은 더 길므로 보수적 확인
b1_lb = b1_all.assign(life_any=b1_all.cycle_life)
corr_b1_incl_cens = {v: sp(b1_lb[v], b1_lb.life_any) for v in ["C1", "avgC_80", "t80_min", "I_qw80"]}
# C1 단일 숫자의 비단조성: 같은 C1=8 에서 수명 차이, 낮은 C1 정책보다 긴 수명
c8 = b1[b1.C1 == 8].groupby("policy").cycle_life.agg(["mean", "size"])
by_c1 = b1.groupby("C1").cycle_life.agg(["mean", "min", "max", "size"])
iso10 = lambda df: df[(df.t80_min - 10).abs() < 0.05]  # noqa: E731  정책상 10분 충전
R["b_crate_vs_life"] = {
    "batch1_labeled_n": int(len(b1)),
    "batch1_avgC_range_labeled": [round(b1.avgC_80.min(), 3), round(b1.avgC_80.max(), 3)],
    "batch1_avgC_range_all": [round(b1_all.avgC_80.min(), 3), round(b1_all.avgC_80.max(), 3)],
    "batch1_censored_avgC": sorted(b1_all[~b1_all.labeled].avgC_80.round(2).tolist()),
    "corr_batch1_cell": corr_b1, "corr_batch1_policy_level": corr_b1_policy,
    "corr_batch1_incl_censored_lowerbound": corr_b1_incl_cens,
    "c1_8_policies": c8.round(1).to_dict(orient="index"),
    "life_by_C1": by_c1.round(1).to_dict(orient="index"),
    "avgC_is_48_over_t80": "avgC_80 = 48/t80_min (단조 변환) → Spearman |ρ| 동일",
    "iqw_policy_vs_meas_b1_maxabsdiff": round(float((b1_all.I_qw_policy - b1_all.I_qw80).abs().max()), 4),
    "batch23_iso_time": {
        b: {"avgC_policy_min": round(D[D.batch == b].avgC_80.min(), 3),
            "avgC_policy_max": round(D[D.batch == b].avgC_80.max(), 3),
            "avgC_policy_std": round(D[D.batch == b].avgC_80.std(), 4),
            "t80_policy_min": round(D[D.batch == b].t80_min.min(), 3),
            "t80_policy_max": round(D[D.batch == b].t80_min.max(), 3),
            "I_qw80_meas_min": round(D[D.batch == b].I_qw80.min(), 3),
            "I_qw80_meas_max": round(D[D.batch == b].I_qw80.max(), 3),
            "I_avg80_meas_min": round(D[D.batch == b].I_avg80.min(), 3),
            "I_avg80_meas_max": round(D[D.batch == b].I_avg80.max(), 3)} for b in BATCH_NAMES},
    "b1_I_qw80_range_labeled": [round(b1.I_qw80.min(), 3), round(b1.I_qw80.max(), 3)],
    "corr_other_batches": {
        "batch2_all": {v: sp(b2[v], b2.cycle_life) for v in ["avgC_80", "I_avg80", "I_qw80", "I_peak", "C1", "Q1"]},
        "batch2_fastcharge": {v: sp(b2_fc[v], b2_fc.cycle_life) for v in ["avgC_80", "I_avg80", "I_qw80", "I_peak", "C1", "Q1"]},
        "batch3": {v: sp(b3[v], b3.cycle_life) for v in ["avgC_80", "I_avg80", "I_qw80", "I_peak", "C1", "Q1"]},
        # 정책명 '4.8C(80%)-4.8C' 8셀(실측 4.36C) 중 라벨 셀은 6개 → 제외 후 n=38
        "batch3_excl_measured_4.36C_labeled6": {
            "n_excluded_labeled": int((b3.I_peak <= 4.5).sum()),
            **{v: sp(b3[b3.I_peak > 4.5][v], b3[b3.I_peak > 4.5].cycle_life)
               for v in ["I_qw80", "I_qw_policy", "I_peak"]}},
    },
    "life_at_policy_t80_10min": {
        "batch1": {"n": int(len(iso10(b1))), "mean": round(iso10(b1).cycle_life.mean(), 1),
                   "min": int(iso10(b1).cycle_life.min()), "max": int(iso10(b1).cycle_life.max()),
                   "policies": sorted(iso10(b1).policy.unique().tolist())},
        **{f"{b}-{g}": {"n": int(len(v)), "mean": round(v.cycle_life.mean(), 1),
                        "min": int(v.cycle_life.min()), "max": int(v.cycle_life.max())}
           for (b, g), v in iso10(LAB[LAB.batch != "batch1"]).groupby(["batch", "group"])},
    },
}
# 같은 I_qw 범위 비교: B2 fastcharge 의 실측 I_qw80 범위 안에 드는 B1 라벨 셀 vs B2 fastcharge
_lo, _hi = float(b2_fc.I_qw80.min()), float(b2_fc.I_qw80.max())
_in = b1[(b1.I_qw80 >= _lo) & (b1.I_qw80 <= _hi)]
_in_loose = b1[(b1.I_qw80 >= round(_lo, 3) - 0.0005) & (b1.I_qw80 <= _hi)]   # 경계 0.0005C 완화(batch1-27 4.7981 포함)
R["b_crate_vs_life"]["b1_within_b2fc_iqw_range"] = {
    "b2fc_I_qw80_range": [round(_lo, 4), round(_hi, 4)],
    "b2fc_mean_life": round(float(b2_fc.cycle_life.mean()), 1),
    "strict": {"n": int(len(_in)), "mean_life": round(float(_in.cycle_life.mean()), 1),
               "b2fc_shorter_pct": round(100 * (1 - b2_fc.cycle_life.mean() / _in.cycle_life.mean()), 1)},
    "loose_0.0005C": {"n": int(len(_in_loose)), "mean_life": round(float(_in_loose.cycle_life.mean()), 1),
                      "b2fc_shorter_pct": round(100 * (1 - b2_fc.cycle_life.mean() / _in_loose.cycle_life.mean()), 1)},
}

# ── (b) 노션 임계값으로 직접 답하기: '고속 충전 셀'을 두 가지로 정의하고 장수명(>1,000)/단수명(<500)/라벨(<550) 개수 비교
IQW_CUTS = [float(v) for v in b1.I_qw80.quantile([1 / 3, 2 / 3])]   # Batch 1 라벨 36셀 기준 3분위 경계
TIER_LBL = ["하위 1/3", "중위 1/3", "상위 1/3"]
TIME_LBL = [">10분 (느린 쪽)", "≤10분 (고속)"]


def iqw_tier(x):
    return pd.cut(x, [-np.inf, *IQW_CUTS, np.inf], labels=TIER_LBL)


def time_grp(df):
    return pd.Series(np.where(df.t80_min <= 10.05, TIME_LBL[1], TIME_LBL[0]), index=df.index)


def notion_counts(lab_df, cen_df):
    v = lab_df.cycle_life
    return {"n_labeled": int(len(v)), "mean": round(float(v.mean()), 1) if len(v) else None,
            "min": int(v.min()) if len(v) else None, "max": int(v.max()) if len(v) else None,
            "n_gt1000": int((v > 1000).sum()), "n_550_1000": int(((v >= 550) & (v <= 1000)).sum()),
            "n_500_550": int(((v >= 500) & (v < 550)).sum()), "n_lt500": int((v < 500).sum()),
            "n_lt550": int((v < LABEL_THRESHOLD).sum()),
            "n_censored": int(len(cen_df)),
            "censored_lower_bounds": sorted(cen_df.cycle_life.astype(int).tolist()),
            "n_censored_lb_gt1000": int((cen_df.cycle_life > 1000).sum())}


b1_cen = b1_all[~b1_all.labeled]
tier_l, tier_c = iqw_tier(b1.I_qw80), iqw_tier(b1_cen.I_qw80)
tg_l, tg_c = time_grp(b1), time_grp(b1_cen)
NOTION_TBL = {
    "def_t80": {g: notion_counts(b1[tg_l == g], b1_cen[tg_c == g]) for g in TIME_LBL},
    "def_iqw_tertile": {g: notion_counts(b1[tier_l == g], b1_cen[tier_c == g]) for g in TIER_LBL},
}
R["b_notion_thresholds"] = {
    "definitions": {
        "def_t80": "정책 t80(0→80% 충전시간) ≤ 10분 = 고속 충전 (평균 C-rate ≥ 4.8C)",
        "def_iqw_tertile": f"실측 충전량 가중 평균 전류 3분위 (Batch 1 라벨 36셀 기준 경계 {IQW_CUTS[0]:.4f}C, {IQW_CUTS[1]:.4f}C; "
                           "정책값 4.80 동률 6셀은 실측 미세 차이로 하위 4 / 중위 2 로 갈림 → tertile_tie_sensitivity 참고)",
        "classes": "장수명 >1,000 / 단수명 <500 (노션 Q1 정의), 라벨 기준 550 (노션 분류 정의). 중도절단 셀은 하한값이라 개수에서 분리",
    },
    "iqw_tertile_cuts": [round(v, 3) for v in IQW_CUTS],
    "table": NOTION_TBL,
    "tests": {
        "t80_mannwhitney_p": float(f"{stats.mannwhitneyu(b1[tg_l == TIME_LBL[1]].cycle_life, b1[tg_l == TIME_LBL[0]].cycle_life).pvalue:.3g}"),
        "tertile_kruskal_p": float(f"{stats.kruskal(*[b1[tier_l == g].cycle_life for g in TIER_LBL]).pvalue:.3g}"),
        "tertile_low_vs_high_mannwhitney_p": float(f"{stats.mannwhitneyu(b1[tier_l == TIER_LBL[0]].cycle_life, b1[tier_l == TIER_LBL[2]].cycle_life).pvalue:.3g}"),
    },
    "b1_cells_gt1000_labeled": b1[b1.cycle_life > 1000][["policy", "cycle_life", "I_qw80", "t80_min"]]
    .round(3).reset_index().to_dict(orient="records"),
    "b1_cells_lt550_labeled": b1[b1.cycle_life < LABEL_THRESHOLD][["policy", "cycle_life", "I_qw80", "t80_min"]]
    .round(3).reset_index().to_dict(orient="records"),
}


# ── 3분위 경계의 동률(tie) 민감도: 정책값 I_qw 가 같은 셀들이 실측 미세 차이(<0.001C)로 경계 양쪽에 갈리는지 확인
def tied_policy_values(cut):
    """정책 I_qw 값이 같은 셀들 중 실측값이 경계 cut 의 양쪽에 걸친 정책값 목록."""
    g = b1.groupby(b1.I_qw_policy.round(3)).I_qw80
    return [float(k) for k, v in g if v.min() <= cut < v.max()]


def tier_by_policy(df, lo_max, hi_min):
    """정책값 기준 3그룹: 하위 = 정책 I_qw ≤ lo_max, 상위 = 정책 I_qw ≥ hi_min (동률 셀은 같은 그룹)."""
    p = df.I_qw_policy.round(3)
    return pd.Series(np.select([p <= lo_max + 1e-9, p >= hi_min - 1e-9], [TIER_LBL[0], TIER_LBL[2]], TIER_LBL[1]),
                     index=df.index)


def tier_summary(tl, tc):
    out = {g: notion_counts(b1[tl == g], b1_cen[tc == g]) for g in TIER_LBL}
    out["kruskal_p"] = float(f"{stats.kruskal(*[b1[tl == g].cycle_life for g in TIER_LBL]).pvalue:.3g}")
    out["low_vs_high_mannwhitney_p"] = float(
        f"{stats.mannwhitneyu(b1[tl == TIER_LBL[0]].cycle_life, b1[tl == TIER_LBL[2]].cycle_life).pvalue:.3g}")
    return out


TIES = {i: tied_policy_values(cut) for i, cut in enumerate(IQW_CUTS)}
pol_vals = np.sort(b1_all.I_qw_policy.round(3).unique())
# 경계 1 의 동률값(4.80)을 하위로 / 중위로 보내는 두 변형. 경계 2 는 동률 없음 → 실측 경계 그대로(정책값으로 환산)
hi_min = float(pol_vals[pol_vals > IQW_CUTS[1]].min())
tie_lo = TIES[0][0] if TIES[0] else float(pol_vals[pol_vals <= IQW_CUTS[0]].max())
lo_max_mid = float(pol_vals[pol_vals < tie_lo].max())
tied_cells = b1[b1.I_qw_policy.round(3).isin(TIES[0])]
R["b_notion_thresholds"]["tertile_tie_sensitivity"] = {
    "note": "실측 3분위 경계 1 (4.799C) 에서 정책 I_qw = 4.80 인 셀 6개(3개 정책 × 2셀)가 실측 차이 <0.001C 로 "
            "하위 4 / 중위 2 로 갈림 (방법 자체의 실측–정책 차이 최대 0.009C 보다 작음). 경계 2 (5.032C) 에는 동률 없음",
    "tied_policy_values_at_cut1": TIES[0], "tied_policy_values_at_cut2": TIES[1],
    "tied_cells": tied_cells[["policy", "I_qw_policy", "I_qw80", "cycle_life"]].assign(
        measured_tier=iqw_tier(tied_cells.I_qw80).astype(str)).round(4).reset_index().to_dict(orient="records"),
    "measured_tertile_n": {g: int((tier_l == g).sum()) for g in TIER_LBL},
    "ties_to_low (정책 I_qw ≤ 4.80 → 하위)": tier_summary(tier_by_policy(b1, tie_lo, hi_min),
                                                       tier_by_policy(b1_cen, tie_lo, hi_min)),
    "ties_to_mid (정책 I_qw = 4.80 → 중위)": tier_summary(tier_by_policy(b1, lo_max_mid, hi_min),
                                                       tier_by_policy(b1_cen, lo_max_mid, hi_min)),
    "policy_cut_values": {"tie_value": tie_lo, "lo_max_if_ties_to_mid": lo_max_mid, "hi_min": hi_min},
}

# ═════════════════ (c) 충전 전류 패턴 ↔ 열화 속도 ═════════════════
DESC = {"I_peak": "피크 전류 (C)", "t_peak": "피크 유지 시간 (분)", "t80_meas": "0→80% 충전시간 (분)",
        "I_avg80": "평균 C-rate (C)", "I_qw80": "충전량 가중 평균 전류 (C)",
        "t_chg": "충전 단계 총 시간 (분)", "dT_chg": "충전 중 온도 상승 (°C)"}
# 열화 속도 지표: 수명 값으로 계산하지 않는 '측정된' 지표만 사용.
# (전체 열화율 (QD초기−0.88)/수명 은 수명의 역수와 거의 같아(B1 ρ=−0.96) 수명 상관을 반복할 뿐이므로 참고용 JSON 에만 둔다)
DEG = {"slope_2_100": "초기 용량\n기울기\n(2–100)", "slope_100_300": "중기 용량\n기울기\n(100–300)",
       "dTmax": "Tmax\n상승", "dIR": "IR\n증가"}
SETS = {"batch1": b1, "batch2_fastcharge": b2_fc, "batch2_all": b2, "batch3": b3,
        "batch1_all46_incl_censored": b1_all, "batch3_all46": D[D.batch == "batch3"]}
corr_c = {name: {d: {g: sp(df[d], df[g]) for g in list(DEG) + ["cycle_life", "fade_life"]} for d in DESC}
          for name, df in SETS.items()}
# 열화 지표 자체와 수명의 관계 (지표가 '수명과 연결된 열화 속도'인지 확인)
deg_vs_life = {name: {g: sp(df[g], df.cycle_life) for g in list(DEG) + ["fade_life"]}
               for name, df in SETS.items() if name not in ("batch1_all46_incl_censored", "batch3_all46")}
# 원시 사이클 선택(행 10 vs 행 2) 재현성
chk = {k: {"spearman": sp(D[f"{k}_r10"], D[f"{k}_chk"])["rho"],
           "median_abs_diff": round(float(np.nanmedian(np.abs(D[f"{k}_r10"] - D[f"{k}_chk"]))), 4)} for k in DESC}
# summary chargetime ↔ 정책 t80 정합성
D["ct_ratio"] = D.ct_med / D.t80_min
ct_tbl = D.groupby(["batch", "group"]).ct_ratio.agg(["median", "min", "max", "size"]).round(4)
dev = D[D.ct_ratio > 1.05]
R["c_current_pattern"] = {
    "descriptor_definitions": {
        "I_peak": "충전 구간(I≥1.5C) 전류 99백분위 [C]", "t_peak": "I ≥ 0.95·I_peak 유지 시간 [분]",
        "t80_meas": "∫I dt/60 이 0.8(정격 80%) 도달까지 시간 [분]", "I_avg80": "0.8/(t80_meas/60) [C]",
        "I_qw80": "0→80% 구간 ∫I²dt/∫Idt = Σ C_k·ΔSOC_k/0.8 (충전량 가중 평균 전류) [C]",
        "t_chg": "충전 시작~방전 시작(휴지·1C CC-CV 포함) [분]", "dT_chg": "0→80% 구간 최고 T − 시작 T [°C]",
        "slope_2_100": "clean QD 사이클 2–100 Theil–Sen 기울기 × 1e5 [mAh/100cyc], 음수 = 감소",
        "slope_100_300": "clean QD 사이클 100–300 Theil–Sen 기울기 × 1e5 [mAh/100cyc], 음수 = 감소 (100 사이클 이후 정보 → EDA 전용)",
        "fade_life": "(QD_init − 0.88)/cycle_life × 1000 [mAh/cyc], QD_init = 사이클 2–11 QD 중앙값",
        "dTmax": "Tmax 중앙값(사이클 91–100) − Tmax 중앙값(사이클 6–15) [°C]",
        "dIR": "IR 중앙값(91–100) − IR 중앙값(2–11) [mΩ]",
    },
    "I_column_unit_check_qc_per_c_median": round(float(D.qc_per_c.median()), 4),
    "spearman": corr_c,
    "degradation_measure_vs_life": deg_vs_life,
    "key_iqw_vs_initial_slope": {k: corr_c[k]["I_qw80"]["slope_2_100"] for k in SETS},
    "key_iqw_vs_mid_slope": {k: corr_c[k]["I_qw80"]["slope_100_300"] for k in SETS},
    "b1_dIR_dTmax_vs_life": {"dIR": sp(b1.dIR, b1.cycle_life), "dTmax": sp(b1.dTmax, b1.cycle_life)},
    "row10_vs_row2_spearman": chk,
    "fade_life_vs_life_b1": sp(b1.fade_life, b1.cycle_life),
    "chargetime_vs_t80": {
        "ratio_by_batch_group": {f"{b}-{g}": v for (b, g), v in ct_tbl.to_dict(orient="index").items()},
        "deviating_cells_ratio_gt_1.05": dev[["policy", "t80_min", "ct_med", "t80_meas", "I_peak", "cycle_life"]]
        .round(3).replace({np.nan: None}).reset_index().to_dict(orient="records"),
        "chargetime_values_removed_gt100_or_le0_fastcharge_newstructure": D.groupby("batch").ct_removed.sum().to_dict(),
        "chargetime_values_removed_gt100_or_le0_all_cells": D_ALL.groupby("batch").ct_removed.sum().to_dict(),
        "chargetime_values_removed_by_batch_group_all_cells":
            {f"{b}-{g}": int(v) for (b, g), v in D_ALL.groupby(["batch", "group"]).ct_removed.sum().items()},
        "spearman_ct_med_vs_t80_meas_all": sp(D.ct_med, D.t80_meas),
    },
    "charge_structure_by_group_median": D.groupby(["batch", "group"])[["t_chg", "t_rest_chg", "t_cv", "Q_hi",
                                                                      "I_post80", "V_hi_end"]]
    .median().round(3).rename(index=str).reset_index().to_dict(orient="records"),
    "b2_q_hi_gt_0.85_policies": sorted(D[D.Q_hi > 0.85].policy.unique().tolist()),
    # 노션 정의('Q1%까지 C1, 이후 나머지는 C2') 대비 실제 80% 이후 전류
    "post80_protocol_check": {
        "n_cells_hi_phase_ends_at_80pct": int(((D.Q_hi >= 0.79) & (D.Q_hi <= 0.81)).sum()),
        "n_cells_hi_phase_beyond_80pct": int((D.Q_hi > 0.85).sum()),
        "I_post80_median_others": round(float(D[D.Q_hi <= 0.85].I_post80.median()), 3),
        "I_post80_range_others": [round(float(D[D.Q_hi <= 0.85].I_post80.min()), 3),
                                  round(float(D[D.Q_hi <= 0.85].I_post80.max()), 3)],
        "beyond80_cells": D[D.Q_hi > 0.85][["policy", "Q_hi", "I_post80", "V_hi_end"]].round(3)
        .reset_index().to_dict(orient="records"),
    },
    "b3_measured_4.36C": {
        "n_cells": int((D[D.batch == "batch3"].I_peak < 4.5).sum()),
        "life_labeled": b3[b3.I_peak < 4.5].cycle_life.astype(int).tolist(),
        "lower_bounds_unlabeled": D[(D.batch == "batch3") & (D.I_peak < 4.5) & ~D.labeled].n_cycles.tolist(),
        "others_mean_life": round(b3[b3.I_peak >= 4.5].cycle_life.mean(), 1),
        "mannwhitney_p": float(f"{stats.mannwhitneyu(b3[b3.I_peak < 4.5].cycle_life, b3[b3.I_peak >= 4.5].cycle_life).pvalue:.3g}"),
    },
}

# ═════════════ (d) 전략 연결: 충전 피처의 추가 정보 & 반복 셀 편차 (Batch 1, 기술통계만) ═════════════
# 주 피처 dQ_logvar = log10 var(ΔQ_{100-10}(V)) (Q3/Q5 와 같은 정의). Qcc_init = cycle 2–6 Qdlin 2.0 V 끝점 중앙값
# (Q5-(d) 후보 정의와 동일, ≤0·>1.2 Ah 는 NaN). 모델 학습 없음 — 편상관·반복쌍 비교·라벨 없는 분포 이동만 계산.
def _qcc_init(c, a=2, b=6):
    v = np.array([qdlin(c, k)[-1] for k in range(a, b + 1)], float)
    v[(v <= 0) | (v > 1.2)] = np.nan
    return float(np.nanmedian(v))


D["dQ_logvar"] = [float(np.log10(np.var(delta_q(by_key[k], 100, 10)))) for k in D.index]
D["Qcc_init"] = [_qcc_init(by_key[k]) for k in D.index]
b1 = D[(D.batch == "batch1") & D.labeled].copy()      # (위 b1 과 같은 36셀, 새 열 포함)
_ly = np.log10(b1.cycle_life.to_numpy(float))


def _resid(v, ctrl):
    X = np.column_stack([np.ones(len(v))] + ctrl)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def partial_r(f, ctrl_cols):
    """log10(수명) 과 f 의 편상관(Pearson, ctrl 선형 통제), 자유도 n−2−k 의 t 검정."""
    ctrl = [b1[c].to_numpy(float) for c in ctrl_cols]
    r = float(np.corrcoef(_resid(b1[f].to_numpy(float), ctrl), _resid(_ly, ctrl))[0, 1])
    n, k = len(_ly), len(ctrl)
    t_ = r * np.sqrt((n - 2 - k) / (1 - r ** 2))
    return {"r": round(r, 3), "p": float(f"{2 * stats.t.sf(abs(t_), n - 2 - k):.3g}"), "n": n, "controls": ctrl_cols}


# 반복 셀(같은 정책 2셀) 쌍: 정책 평균 단위 vs 정책 내부(정책 평균 제거) 상관
_d = b1.assign(ly=_ly)
_pair = _d.groupby("policy").cycle_life.transform("size") == 2
_w = {c: (_d[c] - _d.groupby("policy")[c].transform("mean"))[_pair] for c in ("dQ_logvar", "Qcc_init", "I_qw80", "ly")}
_pm = _d.groupby("policy")[["dQ_logvar", "Qcc_init", "ly"]].mean()
_pairs = []
for pol, g in _d[_pair].groupby("policy"):
    a_, b_ = g.iloc[0], g.iloc[1]
    _pairs.append({"policy": pol, "life": [int(a_.cycle_life), int(b_.cycle_life)],
                   "spread_pct": round(100 * abs(a_.cycle_life - b_.cycle_life) / min(a_.cycle_life, b_.cycle_life), 1),
                   "dly": a_.ly - b_.ly, "dQcc": a_.Qcc_init - b_.Qcc_init, "dlv": a_.dQ_logvar - b_.dQ_logvar,
                   "dIqw": a_.I_qw80 - b_.I_qw80})
_PP = pd.DataFrame(_pairs)
_k_qcc = int((np.sign(_PP.dQcc) == np.sign(_PP.dly)).sum())        # 기대 방향: Qcc 큰 셀이 더 오래
_k_lv = int((np.sign(_PP.dlv) == -np.sign(_PP.dly)).sum())         # 기대 방향: logvar 큰 셀이 더 짧게
# Qcc_init 라벨 없는 배치 이동 (B1 46셀 중앙값 대비) 과 B1 기술 회귀 계수로 환산한 예측 배율
_ref = float(D[D.batch == "batch1"].Qcc_init.median())
_X = np.column_stack([np.ones(len(b1)), b1.dQ_logvar, b1.Qcc_init])
_coef = np.linalg.lstsq(_X, _ly, rcond=None)[0]
_shift = {f"{b}-{g}": round(1000 * (float(v.Qcc_init.median()) - _ref), 1)
          for (b, g), v in D[D.batch != "batch1"].groupby(["batch", "group"])}
R["d_strategy_link"] = {
    "note": "Batch 1 라벨 36셀 기술통계(모델 학습·CV 아님). Qcc_init 관련 값은 Q5-(d) 교차 확인용.",
    "iqw_vs_dQ_logvar_spearman": sp(b1.I_qw80, b1.dQ_logvar),
    "partial_iqw_given_logvar": partial_r("I_qw80", ["dQ_logvar"]),
    "partial_iqw_given_logvar_qcc": partial_r("I_qw80", ["dQ_logvar", "Qcc_init"]),
    "partial_avgC_given_logvar": partial_r("avgC_80", ["dQ_logvar"]),
    "partial_qcc_given_logvar": partial_r("Qcc_init", ["dQ_logvar"]),
    "replicate_pairs": {
        "n_pairs": int(len(_PP)), "n_cells": int(_pair.sum()),
        "spread_pct_max": float(_PP.spread_pct.max()), "spread_pct_median": float(_PP.spread_pct.median()),
        "within_policy_I_qw80_sd_C": round(float(_w["I_qw80"].std()), 4),
        "within_policy_r_logvar": round(float(np.corrcoef(_w["dQ_logvar"], _w["ly"])[0, 1]), 3),
        "within_policy_r_qcc": round(float(np.corrcoef(_w["Qcc_init"], _w["ly"])[0, 1]), 3),
        "policy_mean_r_logvar": round(float(np.corrcoef(_pm.dQ_logvar, _pm.ly)[0, 1]), 3),
        "policy_mean_r_qcc": round(float(np.corrcoef(_pm.Qcc_init, _pm.ly)[0, 1]), 3),
        "n_policies": int(len(_pm)),
        "pairs_expected_direction_qcc": _k_qcc, "pairs_expected_direction_logvar": _k_lv,
        "sign_test_p_qcc": float(f"{stats.binomtest(_k_qcc, len(_PP)).pvalue:.3g}"),
        "sign_test_p_logvar": float(f"{stats.binomtest(_k_lv, len(_PP)).pvalue:.3g}"),
    },
    "qcc_init_label_free_shift_mAh_vs_b1_median": _shift,
    "b1_descriptive_coef_log10_per_Ah": round(float(_coef[2]), 3),
    "implied_prediction_factor": {k: round(float(10 ** (_coef[2] * v / 1000)), 3) for k, v in _shift.items()},
}

# ═════════════════════════ 그림 ═════════════════════════
BC, BL = ps.BATCH_COLOR, ps.BATCH_LABEL
SHORT_LBL = {"batch1": "Batch 1", "batch2": "Batch 2", "batch3": "Batch 3"}


def fmt_p(p):
    return "p<0.001" if p < 1e-3 else f"p={p:.3f}"


# ── 그림 1: (a) 정책별 평균 수명 ──
def fig_policy_life():
    counts = [len(P[P.batch == b]) for b in BATCH_NAMES]
    fig = plt.figure(figsize=(11.5, 0.27 * sum(counts) + 2.6))
    gs = GridSpec(3, 1, height_ratios=counts, hspace=0.3, figure=fig)
    axes = []
    for bi, b in enumerate(BATCH_NAMES):
        ax = fig.add_subplot(gs[bi], sharex=axes[0] if axes else None)
        axes.append(ax)
        pb = P[P.batch == b].copy()
        pb["key"] = pb["mean"].fillna(pb.lower_bounds.map(lambda v: np.mean(v) if v else np.nan))
        pb = pb.sort_values("key").reset_index(drop=True)
        col = BC[b]
        lbls = []
        for y, rw in pb.iterrows():
            mk = NS_MARK if rw.group == "newstructure" else FC_MARK
            g = D[(D.batch == b) & (D.policy == rw.policy)]
            lab = g[g.labeled]
            ax.scatter(lab.cycle_life, np.full(len(lab), y), s=14, color=col, alpha=0.35, lw=0, zorder=2)
            if np.isfinite(rw["mean"]):
                if np.isfinite(rw["std"]):
                    ax.errorbar(rw["mean"], y, xerr=rw["std"], fmt="none", ecolor=col, elinewidth=1.6,
                                capsize=3, zorder=3)
                ax.scatter(rw["mean"], y, s=46, marker=mk, color=col, edgecolor="white", lw=0.8, zorder=4)
            lb = g.life_lb.dropna()
            ax.scatter(lb, np.full(len(lb), y), s=34, marker="^", facecolor="white", edgecolor=col,
                       lw=1.2, zorder=4)
            name = base_policy(rw.policy) + (" [NS]" if (b == "batch2" and rw.group == "newstructure") else "")
            flag = " ※실측 다름" if abs(rw.t80_meas - rw.t80_policy) > 0.5 else ""
            ncell = f"n={rw.n_labeled}" + (f"+{len(lb)}△" if len(lb) else "")
            lbls.append(f"{name}{flag}  (가중 {rw.I_qw_meas:.2f}C · t80 {rw.t80_meas:.1f}분, {ncell})")
        ax.set_yticks(range(len(pb)))
        ax.set_yticklabels(lbls, fontsize=8.2)
        ax.set_ylim(-0.7, len(pb) - 0.3)
        ax.axvline(LABEL_THRESHOLD, color=ps.SHORT, ls="--", lw=1, alpha=0.7, zorder=1)
        ax.grid(axis="y", alpha=0)
        st = R["a_policy_life"]["policy_effect"][b]
        ax.set_title(f"{BL[b]} — 정책 {st['n_policies']}개, 라벨 셀 {st['n_cells']}개 · "
                     f"정책이 설명하는 수명 분산 ω² = {st['omega2']:.2f}", loc="left", fontsize=11, color=col)
        if bi < 2:
            plt.setp(ax.get_xticklabels(), visible=False)
    axes[0].text(LABEL_THRESHOLD + 12, len(P[P.batch == "batch1"]) - 0.6, "라벨 기준 550", fontsize=8,
                 color=ps.SHORT, va="top")
    pl_rho = R["b_crate_vs_life"]["corr_batch1_policy_level"]["I_qw80"]
    axes[0].text(0.985, 0.97, "가중 평균 전류가 낮은 정책일수록 위쪽(장수명)\n"
                 f"정책 단위 Spearman ρ = {pl_rho['rho']:+.2f} (n={pl_rho['n']}, 라벨 셀 있는 정책)",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8.8,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"))
    axes[-1].set_xlim(150, 2350)
    axes[-1].set_xlabel("수명 (사이클, EOL = 0.88 Ah 도달)")
    handles = [Line2D([], [], marker=FC_MARK, ls="-", color="#374151", ms=7, label="정책 평균 ± 표준편차 (fastcharge)"),
               Line2D([], [], marker=NS_MARK, ls="", color="#374151", ms=6, label="정책 평균 (newstructure, [NS])"),
               Line2D([], [], marker="o", ls="", color="#374151", alpha=0.35, ms=4, label="개별 라벨 셀"),
               Line2D([], [], marker="^", ls="", mfc="white", mec="#374151", ms=6,
                      label="중도절단/EOL 미도달 셀 (기록 사이클 = 수명 하한)")]
    axes[0].legend(handles=handles, loc="lower right", fontsize=8.5, frameon=True, framealpha=0.95,
                   edgecolor="#D1D5DB")
    sn = R["a_policy_life"]["same_policy_name_across_batches"]["4.8C(80%)-4.8C"]
    sc = R["a_policy_life"]["same_policy_name_4.8C_same_measured_current"]
    fig.suptitle("Q4-(a) 충전 프로토콜별 평균 수명 — 배치 안에서는 정책과 수명이 함께 움직이지만, "
                 "같은 정책도 배치·시험 구조에 따라 수명 수준이 다름", y=0.997, fontsize=12.5, fontweight="bold")
    fig.text(0.5, 0.978,
             f"같은 '4.8C(80%)-4.8C'(실측 4.8C)의 평균 수명: B2 fastcharge {sn['batch2-fastcharge']['mean']:.0f} · "
             f"B1 {sn['batch1-fastcharge']['mean']:.0f} · B2 newstructure {sn['batch2-newstructure']['mean']:.0f} "
             f"({sc['ratio_max_over_min']:.1f}배)  |  B3 의 같은 이름 셀({sn['batch3-newstructure']['mean']:.0f})은 "
             f"실측 4.36C·11.0분이라 같은 프로토콜이 아님", ha="center", va="top", fontsize=10, color="#374151")
    fig.subplots_adjust(top=0.935)
    return ps.save(fig, "q4_1_policy_life")


# ── 그림 2: (b) Batch 1 — 정책명 C-rate vs 평균/가중 전류 ──
HL = {"8C(15%)-3.6C": ps.LONG, "8C(35%)-3.6C": ps.SHORT}


def fig_batch1_crate():
    fig = plt.figure(figsize=(14.5, 10.6))
    gs = GridSpec(2, 2, hspace=0.36, wspace=0.16, figure=fig)
    axes = [fig.add_subplot(gs[0, 0])]
    axes += [fig.add_subplot(gs[0, 1], sharey=axes[0]), fig.add_subplot(gs[1, 0], sharey=axes[0])]
    ax4 = fig.add_subplot(gs[1, 1])
    specs = [("C1", "정책명 첫 단계 C-rate, C1 [C]", "① 정책명 속 'C-rate 숫자' (C1)"),
             ("avgC_80", "평균 C-rate (0→80%) [C]  = 48 / t80(분)", "② 평균 C-rate (= 충전시간의 역수)"),
             ("I_qw80", "충전량 가중 평균 전류 ∫I²dt / ∫Idt [C]", "③ 충전량 가중 평균 전류 (실측 I(t))")]
    rng = np.random.default_rng(RANDOM_STATE)
    YL = (420, 1330)
    for ax, (v, xl, tt) in zip(axes, specs):
        jit = rng.uniform(-0.06, 0.06, len(b1_all)) if v == "C1" else np.zeros(len(b1_all))
        x = b1_all[v].to_numpy() + jit
        lab = b1_all.labeled.to_numpy()
        hl = b1_all.policy.isin(list(HL)).to_numpy()
        base = lab & ~hl
        # 노션 기준선: 단수명 영역(<500) 음영, 라벨 550, 장수명 1,000
        ax.axhspan(YL[0], 500, color=ps.SHORT, alpha=0.07, lw=0, zorder=0)
        ax.axhline(LABEL_THRESHOLD, color=ps.SHORT, ls=":", lw=1.1, alpha=0.8, zorder=1)
        ax.axhline(1000, color=ps.LONG, ls=":", lw=1.1, alpha=0.8, zorder=1)
        ax.scatter(x[base], b1_all.cycle_life[base], s=36, color=GRAY, alpha=0.85,
                   edgecolor="white", lw=0.6, zorder=2, label="라벨 셀 (n=36)")
        cens = ~lab
        ax.scatter(x[cens], b1_all.cycle_life[cens], s=42, marker="^", facecolor="white",
                   edgecolor=MUTED, lw=1.2, zorder=2, label="중도절단 (기록 사이클 = 수명 하한, n=10)")
        for p, colr in HL.items():
            m = (b1_all.policy == p).to_numpy()
            ax.scatter(x[m], b1_all.cycle_life[m], s=72, color=colr, edgecolor="white", lw=0.8, zorder=4, label=p)
        if v != "C1":
            k, a = np.polyfit(b1[v], b1.cycle_life, 1)
            xs = np.linspace(b1[v].min(), b1[v].max(), 50)
            ax.plot(xs, k * xs + a, color="#374151", lw=1.2, ls="--", zorder=1)
        s, s_c = corr_b1[v]["spearman"], corr_b1_incl_cens.get(v)
        txt = f"Spearman ρ = {s['rho']:+.2f} ({fmt_p(s['p'])}, n={s['n']})"
        if s_c:
            txt += f"\n중도절단 하한 포함 ρ = {s_c['rho']:+.2f} (n={s_c['n']})"
        ax.text(0.97, 0.97, txt, transform=ax.transAxes, ha="right", va="top", fontsize=9,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"), zorder=6)
        ax.set_xlabel(xl)
        ax.set_title(tt, loc="left")
        ax.set_ylabel("수명 (사이클)")
    axes[0].set_ylim(*YL)
    # ① 같은 C1 = 8C 두 정책: 점 옆에 직접 표시
    m15, m35 = c8.loc["8C(15%)-3.6C", "mean"], c8.loc["8C(35%)-3.6C", "mean"]
    axes[0].set_xlim(3.3, 9.75)
    # 각 정책의 두 셀을 대괄호로 묶어 표시 (화살표가 빈 곳을 가리키지 않도록)
    for pol, mval, colr, txt, dy in [("8C(15%)-3.6C", m15, ps.LONG, "15%까지만 8C", 75),
                                     ("8C(35%)-3.6C", m35, ps.SHORT, "35%까지 8C", 95)]:
        v = b1[b1.policy == pol].cycle_life
        lo, hi = float(v.min()) - 12, float(v.max()) + 12
        axes[0].plot([8.16, 8.22, 8.22, 8.16], [lo, lo, hi, hi], color=colr, lw=1.1, zorder=5)
        axes[0].annotate(f"{pol}\n{txt}\n평균 {mval:.0f} ({int(v.min())}, {int(v.max())})", xy=(8.22, mval),
                         xytext=(8.32, mval + dy), fontsize=8.6, color=colr, va="center",
                         arrowprops=dict(arrowstyle="-", color=colr, lw=0.9))
    axes[0].text(8.3, 1145, "같은 C1 = 8C", fontsize=8.8, color="#111827", fontweight="bold")
    axes[0].text(3.4, 1000 + 12, "장수명 기준 1,000", fontsize=8, color=ps.LONG, va="bottom")
    axes[0].text(3.4, LABEL_THRESHOLD + 12, "라벨 기준 550", fontsize=8, color=ps.SHORT, va="bottom")
    axes[0].text(3.4, 460, "단수명 영역 (<500): Batch 1 해당 셀 0개", fontsize=8, color=ps.SHORT, va="center")
    h_, l_ = axes[0].get_legend_handles_labels()
    fig.legend(h_, l_, loc="upper center", bbox_to_anchor=(0.5, 0.957), ncol=4, fontsize=9, frameon=False)
    # ③ 3분위 경계 표시 (④ 와 연결)
    for cut in IQW_CUTS:
        axes[2].axvline(cut, color=MUTED, lw=0.9, ls="-.", alpha=0.7, zorder=1)
    for xpos, lbl in zip([(b1_all.I_qw80.min() + IQW_CUTS[0]) / 2, sum(IQW_CUTS) / 2,
                          (IQW_CUTS[1] + b1.I_qw80.max()) / 2], ["하위 1/3", "중위 1/3", "상위 1/3"]):
        axes[2].text(xpos, 445, lbl, ha="center", fontsize=8.3, color=MUTED)
    axes[2].text(IQW_CUTS[0] - 0.02, 1150, f"경계 {IQW_CUTS[0]:.3f}C 에 정책값 4.80C 6셀이 걸침\n"
                 "(실측 차 <0.001C → 하위 4 · 중위 2)", ha="right", va="center", fontsize=7.8, color=MUTED)
    # ④ 노션 기준(장수명 >1,000 / 단수명 <500 / 라벨 550) 개수 — 두 가지 '고속 충전' 정의
    tbl = R["b_notion_thresholds"]["table"]
    groups = [("def_t80", TIME_LBL[0], "t80 >10분"), ("def_t80", TIME_LBL[1], "t80 ≤10분\n(고속)"),
              ("def_iqw_tertile", TIER_LBL[0], "가중 전류\n하위 1/3"), ("def_iqw_tertile", TIER_LBL[1], "가중 전류\n중위 1/3"),
              ("def_iqw_tertile", TIER_LBL[2], "가중 전류\n상위 1/3")]
    xs_ = [0, 1, 2.5, 3.5, 4.5]
    cls = [("n_gt1000", "장수명 >1,000", ps.LONG), ("n_550_1000", "550~1,000", "#CBD5E1"),
           ("n_500_550", "500~550 (라벨 단수명)", "#FCA5A5"), ("n_lt500", "단수명 <500", ps.SHORT)]
    for xx, (dk, gk, _) in zip(xs_, groups):
        row = tbl[dk][gk]
        bottom = 0
        for key, _, colr in cls:
            h = row[key]
            if h:
                ax4.bar(xx, h, bottom=bottom, color=colr, width=0.72, edgecolor="white", lw=0.8, zorder=3)
                ax4.text(xx, bottom + h / 2, str(h), ha="center", va="center", fontsize=8.6,
                         color="white" if colr in (ps.LONG, ps.SHORT) else "#111827", zorder=4)
                bottom += h
        if row["n_censored"]:
            ax4.bar(xx, row["n_censored"], bottom=bottom, color="white", width=0.72, edgecolor=MUTED,
                    hatch="///", lw=0.8, zorder=3)
            ax4.text(xx, bottom + row["n_censored"] / 2,
                     f"△{row['n_censored']}\n(>1,000: {row['n_censored_lb_gt1000']})",
                     ha="center", va="center", fontsize=7.6, color="#111827", zorder=4,
                     bbox=dict(boxstyle="round,pad=0.12", fc="white", ec="none", alpha=0.85))
            bottom += row["n_censored"]
        ax4.text(xx, bottom + 0.6, f"평균 {row['mean']:.0f}\n<500: {row['n_lt500']} · <550: {row['n_lt550']}",
                 ha="center", va="bottom", fontsize=8.2, color="#111827")
    ax4.set_xticks(xs_)
    ax4.set_xticklabels([g[2] for g in groups], fontsize=8.8)
    ax4.axvline(1.75, color="#D1D5DB", lw=1)
    tests = R["b_notion_thresholds"]["tests"]
    ax4.text(0.5, -0.215, f"충전시간 기준: 평균 {tbl['def_t80'][TIME_LBL[1]]['mean']:.0f} vs "
             f"{tbl['def_t80'][TIME_LBL[0]]['mean']:.0f} (Mann–Whitney p={tests['t80_mannwhitney_p']:.2f})",
             transform=ax4.get_xaxis_transform(), ha="center", fontsize=8.2, color=MUTED)
    ax4.text(3.5, -0.215, f"가중 전류 3분위: Kruskal–Wallis p={tests['tertile_kruskal_p']:.4f}",
             transform=ax4.get_xaxis_transform(), ha="center", fontsize=8.2, color=MUTED)
    ax4.text(0.5, -0.27, "막대 위 평균·<500·<550 개수는 라벨 셀 기준 (중도절단 △ 은 수명 하한이라 평균에서 제외) · "
             f"가중 전류 3분위 실측 경계 {IQW_CUTS[0]:.3f}C / {IQW_CUTS[1]:.3f}C",
             transform=ax4.transAxes, ha="center", fontsize=8.2, color=MUTED)
    ts = R["b_notion_thresholds"]["tertile_tie_sensitivity"]["ties_to_low (정책 I_qw ≤ 4.80 → 하위)"]
    ts_n = "/".join(str(ts[g]["n_labeled"]) for g in TIER_LBL)
    ts_mean = " → ".join(f"{ts[g]['mean']:.0f}" for g in TIER_LBL)
    ts_gt = "/".join(str(ts[g]["n_gt1000"]) for g in TIER_LBL)
    ax4.text(0.5, -0.30,
             "민감도: 정책값 4.80C 동률 6셀이 실측 차이 <0.001C 로 하위 4 · 중위 2 로 갈림\n"
             f"→ 동률을 하위로 묶어도 {ts_n}셀, 평균 {ts_mean}, >1,000 = {ts_gt}, "
             f"KW p={ts['kruskal_p']:.4f} (결론 동일)",
             transform=ax4.transAxes, ha="center", va="top", fontsize=8.2, color=MUTED, linespacing=1.5)
    ax4.set_ylim(0, 41)
    ax4.set_ylabel("셀 수 (Batch 1)")
    ax4.grid(axis="x", alpha=0)
    ax4.set_title("④ 노션 기준으로 본 '고속 충전 → 단수명' (Batch 1, 라벨 36 + 중도절단 10)", loc="left")
    h4 = [Rectangle((0, 0), 1, 1, color=c_, label=l_) for _, l_, c_ in cls] + [
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="///", label="중도절단 △ (수명 하한)")]
    ax4.legend(handles=h4, loc="upper right", fontsize=8, frameon=True, framealpha=0.95, edgecolor="#E5E7EB")
    fig.suptitle("Q4-(b) Batch 1 (학습셋): 노션 정의의 단수명(<500) 셀은 없고, 충전량 가중 평균 전류가 높을수록 "
                 "수명이 상대적으로 짧은 경향", y=0.995, fontsize=13, fontweight="bold")
    return ps.save(fig, "q4_2_batch1_crate_vs_life")


# ── 그림 3: (b)+(c) 배치 간 — 등시간(10분) 정책 & chargetime 정합성 ──
def fig_cross_batch():
    fig, axes = plt.subplots(1, 3, figsize=(16, 5.0), gridspec_kw={"width_ratios": [1, 1, 0.9]})
    for ax, v, xl in [(axes[0], "I_avg80", "실측 평균 C-rate (0→80%) [C]"),
                      (axes[1], "I_qw80", "실측 충전량 가중 평균 전류 [C]")]:
        for b in BATCH_NAMES:
            for g, mk in (("fastcharge", FC_MARK), ("newstructure", NS_MARK)):
                s = D[(D.batch == b) & (D.group == g)]
                if s.empty:
                    continue
                lab = s[s.labeled]
                ax.scatter(lab[v], lab.cycle_life, s=30 if mk == FC_MARK else 26, marker=mk, color=BC[b],
                           alpha=0.7, edgecolor="white", lw=0.5, zorder=3)
                un = s[~s.labeled]
                ax.scatter(un[v], un.life_lb, s=40, marker="^", facecolor="white", edgecolor=BC[b], lw=1.2,
                           zorder=3)
        k, a = np.polyfit(b1[v], b1.cycle_life, 1)
        xs = np.linspace(b1[v].min(), b1[v].max(), 50)
        ax.plot(xs, k * xs + a, color=BC["batch1"], lw=1.3, ls="--", zorder=2)
        ax.axhline(LABEL_THRESHOLD, color=ps.SHORT, ls=":", lw=1, alpha=0.6)
        ax.set_xlabel(xl)
        ax.set_ylim(300, 2350)
    axes[0].set_ylabel("수명 (사이클)")
    axes[0].text(3.52, LABEL_THRESHOLD - 15, "라벨 기준 550", fontsize=8, color=ps.SHORT, va="top")
    axes[0].set_title("① 평균 C-rate: Batch 2·3 는 4.8C(10분)에 몰려 있음", loc="left")
    oc = R["b_crate_vs_life"]["corr_other_batches"]
    ex6 = oc["batch3_excl_measured_4.36C_labeled6"]
    axes[1].set_title("② 가중 평균 전류: 배치 안 방향은 같지만 수준이 다름", loc="left")
    axes[1].text(0.98, 0.97,
                 "배치 내 Spearman ρ (가중 평균 전류 vs 수명)\n"
                 f"B1 {corr_b1['I_qw80']['spearman']['rho']:+.2f} · B2 fastcharge {oc['batch2_fastcharge']['I_qw80']['rho']:+.2f}"
                 f" · B3 {oc['batch3']['I_qw80']['rho']:+.2f}\n"
                 f"(B3 에서 실측 4.36C 라벨 6셀 제외 시 ρ≈{ex6['I_qw80']['rho']:+.2f}, "
                 f"p≈0.05(경계), n={ex6['I_qw80']['n']})",
                 transform=axes[1].transAxes, ha="right", va="top", fontsize=8.3,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"))
    ex = R["c_current_pattern"]["b3_measured_4.36C"]
    axes[0].annotate("Batch 3 '4.8C(80%)-4.8C' 8셀\n실측 4.36C(11.0분) · 평균 수명\n"
                     f"{np.mean(ex['life_labeled']):.0f} (+2셀 EOL 미도달)",
                     xy=(4.37, 1700), xytext=(3.58, 1560), fontsize=8.5,
                     arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    iso = R["b_crate_vs_life"]["life_at_policy_t80_10min"]
    axes[0].text(0.98, 0.97,
                 "정책상 10분 충전(t80 = 10±0.05분) 셀 평균 수명\n"
                 f"B1 {iso['batch1']['mean']:.0f} (n={iso['batch1']['n']}) · "
                 f"B2 fastcharge {iso['batch2-fastcharge']['mean']:.0f} (n={iso['batch2-fastcharge']['n']})\n"
                 f"B2 newstructure {iso['batch2-newstructure']['mean']:.0f} (n={iso['batch2-newstructure']['n']}) · "
                 f"B3 {iso['batch3-newstructure']['mean']:.0f} (n={iso['batch3-newstructure']['n']})",
                 transform=axes[0].transAxes, ha="right", va="top", fontsize=8.3,
                 bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#D1D5DB"))
    # 오른쪽: summary chargetime vs 정책 t80
    ax = axes[2]
    for b in BATCH_NAMES:
        for g, mk in (("fastcharge", FC_MARK), ("newstructure", NS_MARK)):
            s = D[(D.batch == b) & (D.group == g)]
            if not s.empty:
                ax.scatter(s.t80_min, s.ct_med, s=28, marker=mk, color=BC[b], alpha=0.7, edgecolor="white", lw=0.5)
    lim = [8.6, 13.6]
    ax.plot(lim, lim, color="#374151", lw=1, ls="--")
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("정책 문자열로 계산한 t80 [분]")
    ax.set_ylabel("summary chargetime 중앙값 (사이클 2–100) [분]")
    ax.set_title("③ chargetime vs 정책 t80 정합성", loc="left")
    ok = D[D.ct_ratio <= 1.05].ct_ratio
    dv = D[D.ct_ratio > 1.05].ct_ratio
    ax.annotate(f"+{(dv.median() - 1) * 100:.1f}%: Batch 3\n'4.8C(80%)-4.8C' {len(dv)}셀", xy=(10.0, 11.04),
                xytext=(10.55, 12.4), fontsize=8.5, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.text(11.6, 10.4, f"나머지 {len(ok)}셀:\n정책 대비 +{(ok.min() - 1) * 100:.1f}~{(ok.max() - 1) * 100:.1f}%",
            fontsize=8.5, color=MUTED)
    handles = [Line2D([], [], marker="s", ls="", color=BC[b], ms=7, label=BL[b]) for b in BATCH_NAMES] + [
        Line2D([], [], marker=FC_MARK, ls="", color="#374151", ms=6, label="fastcharge"),
        Line2D([], [], marker=NS_MARK, ls="", color="#374151", ms=5, label="newstructure"),
        Line2D([], [], marker="^", ls="", mfc="white", mec="#374151", ms=6, label="수명 하한 (중도절단/EOL 미도달)"),
        Line2D([], [], ls="--", color=BC["batch1"], label="Batch 1 라벨 셀 선형 추세")]
    fig.legend(handles=handles, loc="lower center", ncol=4, fontsize=8.5, bbox_to_anchor=(0.5, -0.1))
    fig.suptitle("Q4-(b·c) 배치 간 비교: Batch 2·3 는 '10분 충전' 등시간 정책 → 평균 C-rate 신호가 사실상 상수",
                 y=1.03, fontsize=13, fontweight="bold")
    fig.tight_layout()
    return ps.save(fig, "q4_3_cross_batch_iso_time")


# ── 그림 4: (c) 전류 패턴 예시 + 배치별 상관 히트맵 ──
def fig_current_pattern():
    fig = plt.figure(figsize=(16.5, 11.0))
    gs = GridSpec(2, 7, height_ratios=[1, 1.3], width_ratios=[1, 1, 1, 1, 1, 1, 0.1], hspace=0.62,
                  wspace=0.5, figure=fig)
    # (좌상) Batch 1 대표 정책 I(t)
    ax = fig.add_subplot(gs[0, :3])
    ex = [("4.8C(80%)-4.8C", "#374151"), ("8C(35%)-3.6C", ps.SHORT), ("8C(15%)-3.6C", ps.LONG)]
    for p, colr in ex:
        s = b1[b1.policy == p]
        r = by_key[s.index[0]]["raw"][RAW_ROW]
        st = int(np.argmax(r["I"].to_numpy() > 0.1))
        tt, ii = r["t"].to_numpy()[st:] - r["t"].to_numpy()[st], r["I"].to_numpy()[st:]
        m = tt <= 16
        ax.plot(tt[m], ii[m], color=colr, lw=1.8,
                label=f"{p}: 가중 평균 {s.I_qw80.mean():.2f}C · t80 {s.t80_min.iloc[0]:.1f}분 · 평균 수명 {s.cycle_life.mean():.0f}")
    ax.set_xlim(0, 16)
    ax.set_ylim(0, 9.6)
    ax.set_xlabel("충전 시작 후 시간 [분]")
    ax.set_ylabel("전류 I [C]  (1C = 1.1 A)")
    ax.set_title("① Batch 1 충전 전류 패턴 (cycle 11): 고전류 충전량 비중이 큰 정책일수록 평균 수명이 짧은 경향",
                 loc="left", fontsize=10.5)
    ax.legend(fontsize=8, loc="upper right")
    # (우상) 같은 정책명 4.8C(80%)-4.8C, 배치·구조별 실제 프로토콜
    ax = fig.add_subplot(gs[0, 3:6])
    cs = {f"{b}-{g}": v for b, g, v in
          [(x["batch"], x["group"], x) for x in R["c_current_pattern"]["charge_structure_by_group_median"]]}
    for b, g, ls in [("batch1", "fastcharge", "-"), ("batch2", "fastcharge", "-"), ("batch2", "newstructure", "--"),
                     ("batch3", "newstructure", "--")]:
        s = D[(D.batch == b) & (D.group == g) & (D.base_policy == "4.8C(80%)-4.8C")]
        r = by_key[s.index[0]]["raw"][RAW_ROW]
        I_ = r["I"].to_numpy()
        st = int(np.argmax(I_ > 0.1))
        dis = st + int(np.argmax(I_[st:] < -0.5))
        tt, ii = r["t"].to_numpy()[st:dis] - r["t"].to_numpy()[st], I_[st:dis]
        lab = s[s.labeled]
        # Batch 1 은 0~10분 4.8C 구간이 Batch 2 fastcharge 와 겹치므로 굵은 반투명 선으로 아래에 그림
        is_b1 = b == "batch1"
        ax.plot(tt, ii, color=BC[b], lw=5.0 if is_b1 else 1.6, alpha=0.45 if is_b1 else 1.0, ls=ls,
                zorder=2 if is_b1 else 3, solid_capstyle="butt",
                label=f"{SHORT_LBL[b]} {g}{' (굵은 선)' if is_b1 else ''}: 충전 단계 {tt[-1]:.0f}분 · "
                      f"평균 수명 {lab.cycle_life.mean():.0f}")
    ax.set_xlim(0, 46)
    ax.set_ylim(0, 6.2)
    ax.set_xlabel("충전 시작 후 시간 [분]")
    ax.set_ylabel("전류 I [C]")
    ax.set_title("② 같은 '4.8C(80%)-4.8C'라도 실제 프로토콜이 다름 (전류 크기·휴지·CV)", loc="left", fontsize=11)
    ax.legend(fontsize=7.9, loc="upper right")
    ax.annotate("Batch 3: 실측 4.36C\n(명목 4.8C 아님, 0→80% 11.0분)", xy=(5, 4.36), xytext=(13, 3.3), fontsize=8.3,
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    ax.annotate("Batch 1·2 fastcharge 는 0~10분\n4.8C 구간이 겹침", xy=(3, 4.9), xytext=(1.0, 5.55), fontsize=8.3,
                color=BC["batch1"], arrowprops=dict(arrowstyle="->", color=BC["batch1"], lw=0.9))
    b2fc = cs["batch2-fastcharge"]
    ax.annotate(f"Batch 2 fastcharge: 80% 후 휴지 ~{b2fc['t_rest_chg']:.0f}분,\n"
                f"3.6 V CV ~{b2fc['t_cv']:.0f}분 (newstructure ~{cs['batch2-newstructure']['t_cv']:.0f}분)",
                xy=(30, 0.08), xytext=(25, 1.6), fontsize=8.3, arrowprops=dict(arrowstyle="->", color=MUTED, lw=0.9))
    # (하) 배치별 Spearman 히트맵
    cols = list(DEG) + ["cycle_life"]
    col_lbl = list(DEG.values()) + ["수명\n(참고)"]
    i_key, k_key = list(DESC).index("I_qw80"), cols.index("slope_2_100")
    sets = [("batch1", f"Batch 1 (라벨 셀, n={len(b1)})"), ("batch2_fastcharge", f"Batch 2 fastcharge (n={len(b2_fc)})"),
            ("batch3", f"Batch 3 (라벨 셀, n={len(b3)})")]
    im, hm_axes = None, []
    for j, (key, ttl) in enumerate(sets):
        ax = fig.add_subplot(gs[1, 2 * j:2 * j + 2])
        hm_axes.append(ax)
        M = np.array([[corr_c[key][d][g]["rho"] if corr_c[key][d][g]["rho"] is not None else np.nan
                       for g in cols] for d in DESC], float)
        Pm = np.array([[corr_c[key][d][g]["p"] if corr_c[key][d][g]["p"] is not None else np.nan
                        for g in cols] for d in DESC], float)
        im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
        ax.axvline(len(DEG) - 0.5, color="white", lw=3)
        # 배치 안에서 사실상 상수인 기술자(변동폭 <1%)는 빗금 — ρ 가 순위 동률·미세 오차로만 결정됨
        dfk = SETS[key]
        for i, d in enumerate(DESC):
            v = dfk[d].dropna()
            if len(v) and (v.max() - v.min()) / abs(v.median()) < 0.01:
                ax.add_patch(Rectangle((-0.5, i - 0.5), len(cols), 1, fill=False, hatch="////",
                                       edgecolor="#6B7280", lw=0, zorder=5))
        for i in range(M.shape[0]):
            for k in range(M.shape[1]):
                if np.isfinite(M[i, k]):
                    sig = Pm[i, k] < 0.05
                    ax.text(k, i, f"{M[i, k]:+.2f}" + ("*" if sig else ""), ha="center", va="center",
                            fontsize=8.2, fontweight="bold" if sig else "normal",
                            color="white" if abs(M[i, k]) > 0.6 else "#111827")
        # 핵심 셀(충전량 가중 평균 전류 × 초기 용량 기울기) 테두리
        ax.add_patch(Rectangle((k_key - 0.5, i_key - 0.5), 1, 1, fill=False, edgecolor="#111827", lw=2.2, zorder=6))
        ax.set_xticks(range(len(cols)))
        ax.set_xticklabels(col_lbl, fontsize=7.9)
        ax.set_yticks(range(len(DESC)))
        ax.set_yticklabels(list(DESC.values()) if j == 0 else [], fontsize=8.6)
        ax.tick_params(length=0)
        ax.grid(False)
        for sp_ in ax.spines.values():
            sp_.set_visible(False)
        ax.set_title(ttl, fontsize=10.5, color=BC[key.split("_")[0]])
    cax = fig.add_subplot(gs[1, 6])
    cb = fig.colorbar(im, cax=cax)
    cb.set_label("Spearman ρ  (* p<0.05)")
    fig.text(0.07, 0.0, "열 정의 — 초기 용량 기울기: 사이클 2–100 QD Theil–Sen 기울기 [mAh/100cycle, 음수 = 감소] · 중기 용량 기울기: 사이클 100–300 "
             "(100 사이클 이후 정보라 모델 입력 불가, 열화 속도 확인용) · Tmax 상승: 사이클 91–100 − 6–15 중앙값 [°C] · "
             "IR 증가: 91–100 − 2–11 중앙값 [mΩ]\n"
             f"'전체 열화율 (QD초기 − 0.88)/수명'은 수명의 역수와 거의 같아(Batch 1 ρ = {R['c_current_pattern']['fade_life_vs_life_b1']['rho']:+.2f}) "
             "별도 열로 두지 않음 → 수명(참고) 열이 같은 정보\n"
             "빗금 행 = 해당 배치 안 변동폭 < 1% (사실상 상수 → ρ 는 정책별 미세한 장비 오버헤드 차이를 반영할 뿐, 물리적 해석 불가) · "
             "검은 테두리 = 핵심 비교 칸",
             ha="left", va="bottom", fontsize=8.4, color=MUTED)
    top = hm_axes[0].get_position().y1
    fig.text(0.07, top + 0.045, "③ 충전 전류 패턴 기술자(행, cycle 3·6·11 원시 I(t) 중앙값) × 열화 속도 지표(열, summary) — Spearman ρ",
             ha="left", fontsize=11.5, fontweight="bold")
    kq = R["c_current_pattern"]["key_iqw_vs_initial_slope"]

    def _r(d):
        return f"{d['rho']:+.2f}" + ("" if d["p"] < 0.05 else " (비유의)")
    fig.suptitle("Q4-(c) 충전 전류 패턴과 열화 속도 — 충전량 가중 평균 전류 ↔ 초기 용량 기울기(사이클 2–100, 음수=감소): "
                 f"Batch 1 {_r(kq['batch1'])} → Batch 2 fastcharge {_r(kq['batch2_fastcharge'])} · Batch 3 {_r(kq['batch3'])}",
                 y=0.975, fontsize=12.5, fontweight="bold")
    return ps.save(fig, "q4_4_current_pattern_vs_degradation")


# ── PDF 용 단순화 그림 (본문 폭 ~16 cm 에서 읽히도록 실제 크기로 그림, 상세 수치는 본문이 담당) ──
def _save_pdf(fig, name):
    p = ps.FIG / f"{name}.png"
    fig.savefig(p, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p


def _small(ax):
    ax.tick_params(labelsize=7.2, length=2.5, pad=2)
    ax.xaxis.label.set_size(7.8)
    ax.yaxis.label.set_size(7.8)


def fig_pdf_batch1():
    """PDF 그림 1: Batch 1 — 가중 평균 전류 vs 수명 + 노션 기준 셀 수."""
    fig, (ax, ax4) = plt.subplots(1, 2, figsize=(7.0, 3.05), gridspec_kw={"width_ratios": [1.12, 1], "wspace": 0.26})
    YL = (430, 1290)
    ax.axhspan(YL[0], 500, color=ps.SHORT, alpha=0.08, lw=0, zorder=0)
    ax.axhline(LABEL_THRESHOLD, color=ps.SHORT, ls=":", lw=0.9, zorder=1)
    ax.axhline(1000, color=ps.LONG, ls=":", lw=0.9, zorder=1)
    for cut in IQW_CUTS:
        ax.axvline(cut, color=MUTED, lw=0.7, ls="-.", alpha=0.8, zorder=1)
    hl = b1_all.policy.isin(list(HL))
    lab = b1_all.labeled
    ax.scatter(b1_all.I_qw80[lab & ~hl], b1_all.cycle_life[lab & ~hl], s=14, color=GRAY, edgecolor="white", lw=0.4,
               zorder=2, label="라벨 셀 (n=36)")
    ax.scatter(b1_all.I_qw80[~lab], b1_all.cycle_life[~lab], s=18, marker="^", facecolor="white", edgecolor=MUTED,
               lw=0.9, zorder=2, label="중도절단 (수명 하한, n=10)")
    for pol, colr in HL.items():
        m = b1_all.policy == pol
        ax.scatter(b1_all.I_qw80[m], b1_all.cycle_life[m], s=24, color=colr, edgecolor="white", lw=0.5, zorder=4)
    k, a = np.polyfit(b1.I_qw80, b1.cycle_life, 1)
    xs = np.linspace(b1.I_qw80.min(), b1.I_qw80.max(), 20)
    ax.plot(xs, k * xs + a, color="#374151", lw=0.9, ls="--", zorder=1)
    ax.text(4.42, 1120, "8C(15%)-3.6C", fontsize=6.8, color=ps.LONG, ha="center", va="center")
    ax.text(5.47, 675, "8C(35%)-3.6C", fontsize=6.8, color=ps.SHORT, ha="center", va="bottom")
    ax.text(3.57, 1000 + 8, "장수명 1,000", fontsize=6.8, color=ps.LONG, va="bottom")
    ax.text(3.57, LABEL_THRESHOLD + 8, "라벨 550", fontsize=6.8, color=ps.SHORT, va="bottom")
    ax.text(3.57, 465, "단수명 <500: 0셀", fontsize=6.8, color=ps.SHORT, va="center")
    for xpos, lbl in zip([4.45, sum(IQW_CUTS) / 2, 5.3], ["하위 1/3", "중위", "상위 1/3"]):
        ax.text(xpos, 1262, lbl, ha="center", va="top", fontsize=6.6, color=MUTED)
    s_iqw, s_c1, s_av = corr_b1["I_qw80"]["spearman"], corr_b1["C1"]["spearman"], corr_b1["avgC_80"]["spearman"]
    ax.text(3.575, 945, f"수명과 Spearman ρ\n가중 평균 전류 {s_iqw['rho']:+.2f}\n평균 C-rate {s_av['rho']:+.2f}\n정책명 C1 {s_c1['rho']:+.2f}",
            ha="left", va="top", fontsize=6.8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#D1D5DB", lw=0.6), zorder=6)
    ax.set_ylim(*YL)
    ax.set_xlim(3.52, 5.6)
    ax.set_xlabel("충전량 가중 평균 전류, 0→80% [C]")
    ax.set_ylabel("수명 (사이클)")
    ax.set_title("① 가중 전류가 높을수록 수명이 짧은 경향", loc="left", fontsize=8.6)
    ax.legend(loc="lower left", bbox_to_anchor=(0.0, 0.21), fontsize=6.6, handletextpad=0.2, borderaxespad=0.3)
    _small(ax)
    # ② 노션 기준 셀 수
    tbl = R["b_notion_thresholds"]["table"]
    groups = [("def_t80", TIME_LBL[0], "t80\n>10분"), ("def_t80", TIME_LBL[1], "t80\n≤10분"),
              ("def_iqw_tertile", TIER_LBL[0], "하위\n1/3"), ("def_iqw_tertile", TIER_LBL[1], "중위\n1/3"),
              ("def_iqw_tertile", TIER_LBL[2], "상위\n1/3")]
    xs_ = [0, 1, 2.4, 3.4, 4.4]
    cls = [("n_gt1000", "장수명 >1,000", ps.LONG), ("n_550_1000", "550~1,000", "#CBD5E1"),
           ("n_500_550", "500~550", "#FCA5A5"), ("n_lt500", "단수명 <500", ps.SHORT)]
    for xx, (dk, gk, _) in zip(xs_, groups):
        row, bottom = tbl[dk][gk], 0
        for key, _, colr in cls:
            h = row[key]
            if h:
                ax4.bar(xx, h, bottom=bottom, color=colr, width=0.7, edgecolor="white", lw=0.5, zorder=3)
                if h >= 3:
                    ax4.text(xx, bottom + h / 2, str(h), ha="center", va="center", fontsize=6.6,
                             color="white" if colr == ps.LONG else "#111827", zorder=4)
                bottom += h
        if row["n_censored"]:
            ax4.bar(xx, row["n_censored"], bottom=bottom, color="white", width=0.7, edgecolor=MUTED, hatch="////",
                    lw=0.5, zorder=3)
            bottom += row["n_censored"]
        ax4.text(xx, bottom + 0.6, f"{row['mean']:.0f}", ha="center", va="bottom", fontsize=7.0, color="#111827")
    tests = R["b_notion_thresholds"]["tests"]
    ax4.text(0.5, 40.5, f"충전시간 기준\np={tests['t80_mannwhitney_p']:.2f}", ha="center", va="top", fontsize=6.8,
             color=MUTED)
    ax4.text(3.4, 40.5, f"가중 전류 3분위\nKW p={tests['tertile_kruskal_p']:.4f}", ha="center", va="top", fontsize=6.8,
             color=MUTED)
    ax4.axvline(1.7, color="#D1D5DB", lw=0.8)
    ax4.set_xticks(xs_)
    ax4.set_xticklabels([g[2] for g in groups])
    ax4.set_ylim(0, 41)
    ax4.set_xlim(-0.55, 4.95)
    ax4.set_ylabel("셀 수 (막대 위 = 라벨 셀 평균 수명)")
    ax4.grid(axis="x", alpha=0)
    ax4.set_title("② 노션 기준(>1,000 / <500)으로 센 셀 수", loc="left", fontsize=8.6)
    h4 = [Rectangle((0, 0), 1, 1, color=c_, label=l_) for _, l_, c_ in cls] + [
        Rectangle((0, 0), 1, 1, facecolor="white", edgecolor=MUTED, hatch="////", lw=0.5, label="중도절단")]
    ax4.legend(handles=h4, loc="upper right", bbox_to_anchor=(1.0, 0.86), fontsize=6.6, ncol=1, handlelength=1.2,
               handletextpad=0.4, borderaxespad=0.2)
    _small(ax4)
    return _save_pdf(fig, "q4_pdf_1_batch1_iqw_vs_life")


def fig_pdf_cross_batch():
    """PDF 그림 2: 배치 간 — 같은 가중 전류에서도 수명 수준이 다르고, 배치 안 상관도 약해짐."""
    fig, (ax, axr) = plt.subplots(1, 2, figsize=(7.0, 3.0), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.3})
    rng_ = R["b_crate_vs_life"]["b1_within_b2fc_iqw_range"]
    lo, hi = rng_["b2fc_I_qw80_range"]
    ax.axvspan(lo, hi, color="#F3F4F6", lw=0, zorder=0)
    for b in BATCH_NAMES:
        for g, mk in (("fastcharge", FC_MARK), ("newstructure", NS_MARK)):
            s = D[(D.batch == b) & (D.group == g)]
            if s.empty:
                continue
            l_ = s[s.labeled]
            ax.scatter(l_.I_qw80, l_.cycle_life, s=11 if mk == FC_MARK else 10, marker=mk, color=BC[b], alpha=0.75,
                       edgecolor="white", lw=0.3, zorder=3)
            u_ = s[~s.labeled]
            ax.scatter(u_.I_qw80, u_.life_lb, s=13, marker="^", facecolor="white", edgecolor=BC[b], lw=0.8, zorder=3)
    k, a = np.polyfit(b1.I_qw80, b1.cycle_life, 1)
    xs = np.linspace(b1.I_qw80.min(), b1.I_qw80.max(), 20)
    ax.plot(xs, k * xs + a, color=BC["batch1"], lw=0.9, ls="--", zorder=2)
    m1, m2 = rng_["strict"]["mean_life"], rng_["b2fc_mean_life"]
    ax.hlines(m1, lo, hi, color=BC["batch1"], lw=1.6, zorder=4)
    ax.hlines(m2, lo, hi, color=BC["batch2"], lw=1.6, zorder=4)
    ax.annotate(f"회색 띠 = B2 fastcharge 범위\n평균 수명 B1 {m1:.0f}\nvs B2 fastcharge {m2:.0f}\n"
                f"({rng_['strict']['b2fc_shorter_pct']:.0f}% 짧음)",
                xy=(5.15, (m1 + m2) / 2), xytext=(5.58, 1390), fontsize=6.6, ha="right", va="bottom",
                arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.annotate("B3 정책명 4.8C(80%)-4.8C\n= 실측 4.36C", xy=(4.37, 1850), xytext=(3.58, 2080), fontsize=6.6,
                va="center", arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    ax.axhline(LABEL_THRESHOLD, color=ps.SHORT, ls=":", lw=0.8, zorder=1)
    ax.set_ylim(300, 2350)
    ax.set_xlim(3.52, 5.6)
    ax.set_xlabel("충전량 가중 평균 전류, 0→80% [C]")
    ax.set_ylabel("수명 (사이클)")
    ax.set_title("① 배치별 가중 전류 vs 수명 (점선 = B1 추세)", loc="left", fontsize=8.6)
    hd = [Line2D([], [], marker="o", ls="", color=BC["batch1"], ms=3.5, label="B1"),
          Line2D([], [], marker="o", ls="", color=BC["batch2"], ms=3.5, label="B2 fastcharge"),
          Line2D([], [], marker="D", ls="", color=BC["batch2"], ms=3, label="B2 newstructure"),
          Line2D([], [], marker="D", ls="", color=BC["batch3"], ms=3, label="B3"),
          Line2D([], [], marker="^", ls="", mfc="white", mec="#374151", ms=3.5, label="수명 하한")]
    ax.legend(handles=hd, loc="lower left", fontsize=6.6, ncol=1, handletextpad=0.2, borderaxespad=0.3,
              frameon=True, facecolor="white", edgecolor="none", framealpha=0.9)
    _small(ax)
    # ② 배치 안 Spearman ρ: 가중 전류 ↔ 수명 / ↔ 초기 용량 기울기
    oc = R["b_crate_vs_life"]["corr_other_batches"]
    rel = {"수명": [corr_b1["I_qw80"]["spearman"], oc["batch2_fastcharge"]["I_qw80"], oc["batch3"]["I_qw80"]],
           "초기 용량 기울기\n(2–100, 음수=감소)": [corr_c[k]["I_qw80"]["slope_2_100"]
                                       for k in ("batch1", "batch2_fastcharge", "batch3")]}
    names = ["B1", "B2 fastcharge", "B3"]
    cols_ = [BC["batch1"], BC["batch2"], BC["batch3"]]
    w = 0.26
    for gi, (gname, vals) in enumerate(rel.items()):
        for j, v in enumerate(vals):
            x = gi + (j - 1) * w
            sig = v["p"] < 0.05
            axr.bar(x, v["rho"], width=w * 0.92, color=cols_[j] if sig else "white", edgecolor=cols_[j], lw=0.9,
                    hatch=None if sig else "////", zorder=3, label=names[j] if gi == 0 else None)
            axr.text(x, v["rho"] - 0.03, f"{v['rho']:+.2f}" + ("" if sig else "\n(ns)"), ha="center", va="top",
                     fontsize=6.8, color="#111827")
    ex6 = oc["batch3_excl_measured_4.36C_labeled6"]["I_qw80"]
    axr.scatter([w], [ex6["rho"]], marker="_", s=120, color="#111827", lw=1.4, zorder=5)
    axr.annotate(f"B3: 4.36C 6셀 제외 시\n{ex6['rho']:+.2f} (p≈0.05, 경계)", xy=(w + 0.12, ex6["rho"]),
                 xytext=(0.45, -0.86), fontsize=6.6, ha="center", va="center",
                 arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.6))
    axr.axhline(0, color="#374151", lw=0.7)
    axr.set_xticks(range(len(rel)))
    axr.set_xticklabels([f"가중 전류 ↔ {k}" for k in rel], fontsize=7)
    axr.set_ylim(-1.0, 0.2)
    axr.set_ylabel("배치 안 Spearman ρ")
    axr.grid(axis="x", alpha=0)
    axr.set_title("② B1 의 관계가 B2·B3 에서 약해짐", loc="left", fontsize=8.6)
    axr.legend(loc="upper center", ncol=3, fontsize=6.6, handlelength=1.0, borderaxespad=0.2, columnspacing=0.8)
    _small(axr)
    axr.tick_params(axis="x", labelsize=6.8)
    return _save_pdf(fig, "q4_pdf_2_cross_batch")


figs = [fig_policy_life(), fig_batch1_crate(), fig_cross_batch(), fig_current_pattern(),
        fig_pdf_batch1(), fig_pdf_cross_batch()]
R["figures"] = [str(Path(p).resolve().relative_to(ROOT)) for p in figs]   # 저장소 기준 상대 경로

# 셀 단위 표도 저장 (검증용)
cols = ["batch", "idx", "policy", "group", "cycle_life", "labeled", "censored", "C1", "Q1", "C2", "t80_min",
        "avgC_80", "I_qw_policy", "I_peak", "t_peak", "t80_meas", "I_avg80", "I_qw80", "t_chg", "t_rest_chg", "t_cv",
        "Q_hi", "I_post80", "V_hi_end", "dT_chg", "slope_2_100", "slope_100_300", "QD_init", "fade_life", "dTmax", "dIR",
        "ct_med", "ct_ratio", "dQ_logvar", "Qcc_init"]
D[cols].round(4).to_csv(OUT / "q4_cell_features.csv")


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f"{float(o):.5g}")   # 유효숫자 5자리 (작은 p 값 보존)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.bool_,)):
        return bool(o)
    return o


with open(OUT / "q4_results.json", "w", encoding="utf-8") as fh:
    json.dump(_clean(R), fh, ensure_ascii=False, indent=2)

# 콘솔 요약
print("[a] 정책 효과:", R["a_policy_life"]["policy_effect"])
print("[b] B1 Spearman:", {k: v["spearman"]["rho"] for k, v in corr_b1.items()})
print("[b] B1 정책 수준:", {k: v["rho"] for k, v in corr_b1_policy.items()})
print("[b] 중도절단 포함:", corr_b1_incl_cens)
print("[b] 10분 정책 수명:", R["b_crate_vs_life"]["life_at_policy_t80_10min"])
print("[c] chargetime/t80:", R["c_current_pattern"]["chargetime_vs_t80"]["ratio_by_batch_group"])
print("[c] row10 vs row2:", chk)
print("[d] 전략 연결:", json.dumps(_clean(R["d_strategy_link"]), ensure_ascii=False))
print("saved:", *figs, OUT / "q4_results.json", sep="\n  ")
