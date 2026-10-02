"""Q1. Cycle Life 분포는 어떻게 생겼는가?  (DAY 1 EDA — 모델 학습 없음)

노션 하위 항목
  (a) 150 ~ 2,300 사이클 Histogram
  (b) 장수명(>1,000)/단수명(<500) 비율 확인   (+ 분류 라벨 기준 550 분할)
  (c) 이상치 셀 식별 - 왜 유독 짧은가?
  (+) 노션 문장 'Batch 1/2는 유사하지만, Batch 3는 분포 차이 있음' 데이터 검증

실행:  python src/eda/q1_cycle_life.py
산출:  reports/figures/q1_*.png,  results/eda/q1_results.json
(c)에는 수명 '값'의 통계적 이상치(Tukey·robust z)와 별개로 QD 점프·IR 스파이크 기반 데이터 품질 점검을 포함.
(c)에는 Batch 1 복제셀 산포 분해(dQ_logvar·Qcc_init의 정책 평균 vs 정책 내부 상관, 복제쌍 16개)와
    Qcc_init(2.0 V Qdlin 절대 수준)의 라벨 없는 배치 이동량도 포함 → results['c_replicate_spread_batch1'].
그림 q1_hist_150_2300 · q1_short_cells_evidence 는 PDF(16 cm 폭)용 간결판, 세부 수치는 q1_results.json.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from scipy import stats  # noqa: E402

import plot_style as ps  # noqa: E402
from data import (BATCH_NAMES, EOL_AH, LABEL_THRESHOLD, RANDOM_STATE,  # noqa: E402,F401
                  cell_table, clean_summary, delta_q, load_all)

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

# 노션 정의 그대로
HIST_RANGE = (150, 2300)          # "150 ~ 2,300 사이클 Histogram"
BIN_W = 50                         # 2,150 / 50 = 43 bins (모든 배치 동일)
LONG_T, SHORT_T = 1000, 500        # "장수명(>1,000)/단수명(<500)"
LBL_T = LABEL_THRESHOLD            # 550 (분류 라벨)
BL = {"batch1": "Batch 1", "batch2": "Batch 2", "batch3": "Batch 3"}
GL = {"fastcharge": "fastcharge", "newstructure": "newstructure"}


def tint(hex_color: str, w: float) -> tuple:
    """배치 색을 흰색과 섞은 연한 색 (w=0 원색, w=1 흰색)."""
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * w, g + (1 - g) * w, b + (1 - b) * w)


def jsonable(o):
    if isinstance(o, dict):
        return {str(k): jsonable(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [jsonable(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else round(float(o), 6)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


# ════════════════════════════════════════════════════════════════
# 0. 데이터
# ════════════════════════════════════════════════════════════════
cells = load_all()
CELL = {c["cell_key"]: c for c in cells}
T = cell_table(cells)
T["base_policy"] = T["policy"].str.replace("-newstructure", "", regex=False)
# 수명 시험 셀 = fastcharge/newstructure (Batch 2 VarCharge·SLOWCYCLE 8개는 수명 시험이 아니라 라벨 NaN → 분포에서 제외)
T["life_test"] = T["group"].isin(["fastcharge", "newstructure"])
# 하한값(lower bound): 중도절단(Batch 1) = 기록된 cycle_life, EOL 미도달 NaN(Batch 3 23·32) = 기록 사이클 수
T["lower_bound"] = np.where(T["censored"], T["cycle_life"],
                            np.where(T["life_test"] & T["cycle_life"].isna(), T["n_cycles"], np.nan))
LAB = T[T["labeled"]].copy()                      # 실제 EOL 도달 셀만 = 분포 통계의 대상
LB = T[T["lower_bound"].notna()].copy()           # 하한값만 아는 셀


def early_features(c: dict) -> dict:
    """'왜 짧은가' 근거용 초기 사이클 지표 (clean_summary 기준, 사이클 번호 = summary['cycle'])."""
    s = clean_summary(c["summary"]).set_index("cycle")
    w = s.loc[2:100]                                  # Batch 1 cycle 1 = 더미 → 2~100
    f = {
        "QD_10": s.loc[10, "QD"],
        "dQD_100_10": s.loc[100, "QD"] - s.loc[10, "QD"],           # Ah, 음수 = 감소
        "IR_mean_2_100": w["IR"].mean() * 1000,                      # mΩ
        "Tmax_mean_2_100": w["Tmax"].mean(),                         # °C
        "chargetime_2_6": s.loc[2:6, "chargetime"].mean(),           # 분
    }
    dq = delta_q(c, 100, 10)                                         # 노션: cycle 100 − cycle 10
    f["log10_var_dQ"] = float(np.log10(np.nanvar(dq)))
    return f


F = pd.DataFrame([{"cell_key": c["cell_key"], **early_features(c)} for c in cells])
T = T.merge(F, on="cell_key", how="left")
LAB = LAB.merge(F, on="cell_key", how="left")

results: dict = {"definitions": {
    "labeled": "cycle_life 존재 & 중도절단 아님 (data.cell_table 'labeled')",
    "censored_lower_bound": "Batch 1 중도절단 10개 = cycle_life(하한), Batch 3 23·32 = n_cycles(하한, EOL 미도달)",
    "excluded": "Batch 2 VarCharge 4 + SLOWCYCLE 4 (수명 시험 아님, 라벨 NaN)",
    "hist_bins": f"np.arange({HIST_RANGE[0]}, {HIST_RANGE[1]}+1, {BIN_W}) → 43 bins",
    "skewness": "scipy.stats.skew(x, bias=False) (보정된 Fisher-Pearson G1, 양수 = 오른쪽 긴 꼬리)",
    "std": "표본표준편차 ddof=1",
}}

# ════════════════════════════════════════════════════════════════
# (a) 기술통계 + Histogram
# ════════════════════════════════════════════════════════════════

def describe(x: pd.Series) -> dict:
    x = x.dropna().astype(float)
    q1, q3 = x.quantile([0.25, 0.75])
    return {"n": len(x), "mean": x.mean(), "median": x.median(), "std": x.std(ddof=1),
            "skew": stats.skew(x, bias=False) if len(x) > 2 else np.nan,
            "skew_log10": stats.skew(np.log10(x), bias=False) if len(x) > 2 else np.nan,
            "min": x.min(), "q1": q1, "q3": q3, "max": x.max(), "cv": x.std(ddof=1) / x.mean()}


def km_median(events: np.ndarray, censored: np.ndarray) -> float:
    """Kaplan–Meier 중앙 수명 (중도절단 하한값 반영). S(t) ≤ 0.5 가 되는 최소 t."""
    t = np.r_[events, censored]
    e = np.r_[np.ones(len(events)), np.zeros(len(censored))]
    s = 1.0
    for u in np.unique(t[e == 1]):
        n_risk = (t >= u).sum()
        d = ((t == u) & (e == 1)).sum()
        s *= 1 - d / n_risk
        if s <= 0.5:
            return float(u)
    return float("nan")


desc = {}
for b in BATCH_NAMES:
    g = LAB[LAB.batch == b]
    desc[b] = {"all_labeled": describe(g.cycle_life)}
    for grp, gg in g.groupby("group"):
        desc[b][grp] = describe(gg.cycle_life)
    lb = LB[LB.batch == b]["lower_bound"].to_numpy()
    desc[b]["n_lower_bound_only"] = len(lb)
    desc[b]["lower_bounds"] = sorted(lb.tolist())
    desc[b]["km_median_incl_censored"] = km_median(g.cycle_life.to_numpy(), lb) if len(lb) else desc[b]["all_labeled"]["median"]
desc["pooled_labeled"] = describe(LAB.cycle_life)
results["a_descriptive"] = desc
# Batch 1 중도절단 상세: 하한값이 라벨 셀보다 확실히 긴지(최장수 확정) vs 라벨 셀보다 짧을 수도 있는지
cen = T[T.censored].sort_values("cycle_life")
b1_lab = LAB[LAB.batch == "batch1"].cycle_life
results["a_batch1_censoring"] = {
    "cells": [{"cell_key": r.cell_key, "lower_bound": r.cycle_life, "QD_last_Ah": r.QD_last,
               "n_labeled_longer_than_lb": int((b1_lab > r.cycle_life).sum())} for r in cen.itertuples()],
    "confirmed_longest": cen[cen.cycle_life > b1_lab.max()].cell_key.tolist(),     # 하한 > 라벨 최댓값 1,074
    "undetermined": cen[cen.cycle_life <= b1_lab.max()].cell_key.tolist(),         # 하한 879~906
    "labeled_max": b1_lab.max(),
    "note": "하한 ≥1,177인 00~04만 '라벨 셀 전부보다 김'이 확정. 08·10·12·13·22(하한 879~906)는 하한보다 긴 라벨 셀 7~8개"
            "(880~1,074)보다 짧을 수도 있음(12·13은 기록 종료 시 QD 0.913·0.923 Ah로 이미 EOL에 근접)",
}
results["a_observed_range"] = {
    "labeled_min": LAB.cycle_life.min(), "labeled_max": LAB.cycle_life.max(),
    "n_labeled_below_392": int((LAB.cycle_life < 392).sum()),
    "max_lower_bound": LB.lower_bound.max(),
    "note": "노션/원논문 범위 150~2,300 중 본 파일 라벨 셀은 392~1,935; 150~391 구간 셀 0개",
}

# ── 그림 (PDF 16 cm 폭 기준 간결판): 배치별 히스토그램 + 하한값 레인 + 노션 기준선 ──
#    상세 수치(박스플롯·Tukey 울타리·기술통계)는 본문과 q1_results.json 에 둔다.
bins = np.arange(HIST_RANGE[0], HIST_RANGE[1] + 1, BIN_W)
fig, axes_h = plt.subplots(3, 1, figsize=(7.4, 6.2), sharex=True, gridspec_kw=dict(hspace=0.66))
for i, (b, axh) in enumerate(zip(BATCH_NAMES, axes_h)):
    col = ps.BATCH_COLOR[b]
    g = LAB[LAB.batch == b]
    fc = np.histogram(g[g.group == "fastcharge"].cycle_life, bins)[0]
    ns = np.histogram(g[g.group == "newstructure"].cycle_life, bins)[0]
    axh.bar(bins[:-1], fc, width=BIN_W, align="edge", color=col, edgecolor="white", linewidth=0.8)
    axh.bar(bins[:-1], ns, width=BIN_W, align="edge", bottom=fc, color=tint(col, 0.5), edgecolor=col, linewidth=0.7)
    ymax = int((fc + ns).max())
    # 하한값만 아는 셀: 빈도 막대와 분리된 회색 레인에 실제 하한값 위치(|)만 표시 → 빈도로 오독되지 않음
    lb = np.sort(LB[LB.batch == b]["lower_bound"].to_numpy())
    lane_y = ymax * 1.28 + 0.6
    if len(lb):
        axh.axhspan(lane_y - ymax * 0.11 - 0.3, lane_y + ymax * 0.11 + 0.3, color="#F3F4F6", zorder=0, lw=0)
        axh.plot(lb, [lane_y] * len(lb), marker="|", ms=9, mew=1.6, color=col, ls="none", zorder=4)
        if b == "batch1":
            axh.text(lb.max() + 40, lane_y, f"← 중도절단 {len(lb)}셀 하한값 (실제 수명 ≥ 눈금, 빈도 아님)",
                     ha="left", va="center", fontsize=8.4, color="#374151")
        else:
            axh.text(lb.min() - 40, lane_y, f"EOL 미도달 {len(lb)}셀 하한값 (>{lb.min():,.0f}) →",
                     ha="right", va="center", fontsize=8.4, color="#374151")
    axh.set_ylim(0, lane_y + ymax * 0.2 + 0.5)
    axh.set_yticks(np.arange(0, ymax + 1, 4 if ymax > 8 else 2))
    for x0, c0, ls in ((SHORT_T, ps.SHORT, "--"), (LBL_T, "#6B7280", ":"), (LONG_T, ps.LONG, "--")):
        axh.axvline(x0, color=c0, ls=ls, lw=1.0, zorder=1)
    if i == 0:
        ytxt = ymax * 0.95
        axh.text(SHORT_T - 10, ytxt, "단수명 <500", color=ps.SHORT, ha="right", va="top", fontsize=8.4)
        axh.text(LBL_T + 10, ytxt, "550", color="#4B5563", ha="left", va="top", fontsize=8.4)
        axh.text(LONG_T + 10, ytxt, "장수명 >1,000", color=ps.LONG, ha="left", va="top", fontsize=8.4)
    d = desc[b]["all_labeled"]
    x = g.cycle_life
    n_long, n_short, n_550 = int((x > LONG_T).sum()), int((x < SHORT_T).sum()), int((x < LBL_T).sum())
    head = {"batch1": "Batch 1 (학습)", "batch2": "Batch 2 (테스트)", "batch3": "Batch 3 (추가 검증)"}[b]
    med = f"{d['median']:,.1f}" if d["median"] % 1 else f"{d['median']:,.0f}"
    axh.set_title(f"{head} — 라벨 {d['n']}셀 · 중앙값 {med} · 왜도 {d['skew']:+.2f}", loc="left",
                  fontsize=9.6, color=col, pad=4)
    axh.text(1.0, 1.03, f">1,000: {n_long}  ·  <500: {n_short}  ·  <550: {n_550}", transform=axh.transAxes,
             ha="right", va="bottom", fontsize=8.6, color="#374151")
    if b == "batch2":
        axh.text(565, ymax * 0.72, "◀ fastcharge 30셀\n    (392~514)", ha="left", va="center", fontsize=8.6,
                 color=col, fontweight="bold")
        axh.text(1215, ymax * 0.12, "◀ newstructure 9셀 (777~1,186)", ha="left", va="bottom", fontsize=8.6,
                 color=col, fontweight="bold")
    axh.set_ylabel("셀 수", fontsize=8.8)
    axh.tick_params(labelsize=8.6)
    axh.set_xlim(*HIST_RANGE)
    axh.set_xticks(np.arange(250, 2301, 250))
    axh.tick_params(axis="x", labelbottom=True)
axes_h[-1].set_xlabel("Cycle Life (방전용량이 EOL 0.88 Ah 미만이 되는 사이클) · 50사이클 bin, 세 배치 동일", fontsize=8.8)
c2 = ps.BATCH_COLOR["batch2"]
handles = [Patch(fc="#6B7280", ec="white", label="진한 막대 = fastcharge"),
           Patch(fc=tint("#6B7280", 0.5), ec="#6B7280", label="연한 막대 = newstructure"),
           Line2D([], [], marker="|", ms=9, mew=1.6, color="#4B5563", ls="none", label="하한값만 아는 셀 (회색 레인)")]
fig.legend(handles=handles, loc="upper left", ncol=3, bbox_to_anchor=(0.085, 0.955), fontsize=8.6,
           handlelength=1.4, columnspacing=1.4)
fig.text(0.085, 0.995, "Q1(a)·(b) Cycle Life 분포 — 배치마다 위치와 모양이 다름", ha="left", va="top",
         fontsize=11.5, fontweight="bold")
ps.save(fig, "q1_hist_150_2300")

# ════════════════════════════════════════════════════════════════
# (b) 장수명(>1,000) / 단수명(<500) 비율 + 550 라벨 분할
# ════════════════════════════════════════════════════════════════
CATS = ["<500 (단수명)", "500~549", "550~1,000", ">1,000 (장수명)", "≥550 확정·구간 미정"]
CAT_COLOR = [ps.SHORT, "#F7B4B4", "#D1D5DB", ps.LONG, "white"]


def life_category(row) -> str | None:
    """라벨 셀은 값으로, 하한값 셀은 하한값이 구간을 확정할 때만 배정 (확정 불가면 '구간 미정')."""
    if row["labeled"]:
        v = row["cycle_life"]
        return CATS[0] if v < SHORT_T else CATS[1] if v < LBL_T else CATS[2] if v <= LONG_T else CATS[3]
    if np.isfinite(row["lower_bound"]):
        v = row["lower_bound"]
        return CATS[3] if v > LONG_T else CATS[4]          # 하한 879~906 → ≥550 확정, >1000 여부 미정
    return None


T["life_cat"] = T.apply(life_category, axis=1)
rows_b = [("batch1", None, "Batch 1 (학습)"), ("batch2", None, "Batch 2 (테스트)"),
          ("batch2", "fastcharge", "– fastcharge"), ("batch2", "newstructure", "– newstructure"),
          ("batch3", None, "Batch 3 (추가 검증)")]
ratio = {}
for b, grp, name in rows_b:
    g = T[(T.batch == b) & T.life_test]
    if grp:
        g = g[g.group == grp]
    gl = g[g.labeled]
    key = b if grp is None else f"{b}_{grp}"
    cnt = g.life_cat.value_counts().reindex(CATS, fill_value=0)
    ratio[key] = {
        "n_life_test": len(g), "n_labeled": len(gl),
        # 라벨 셀만 (노션 기준 그대로)
        "labeled_long_gt1000": int((gl.cycle_life > LONG_T).sum()),
        "labeled_short_lt500": int((gl.cycle_life < SHORT_T).sum()),
        "labeled_mid_500_1000": int(((gl.cycle_life >= SHORT_T) & (gl.cycle_life <= LONG_T)).sum()),
        "labeled_long_ratio": (gl.cycle_life > LONG_T).mean(), "labeled_short_ratio": (gl.cycle_life < SHORT_T).mean(),
        "labeled_lt550": int((gl.cycle_life < LBL_T).sum()), "labeled_ge550": int((gl.cycle_life >= LBL_T).sum()),
        "labeled_lt550_ratio": (gl.cycle_life < LBL_T).mean(),
        # 하한값 반영 (중도절단 포함 전체 수명시험 셀)
        "all_counts_by_category": cnt.to_dict(),
        "all_long_ratio": cnt[CATS[3]] / len(g), "all_short_ratio": cnt[CATS[0]] / len(g),
        "all_lt550": int(cnt[CATS[0]] + cnt[CATS[1]]),
        "all_lt550_ratio": (cnt[CATS[0]] + cnt[CATS[1]]) / len(g),
    }
results["b_ratios"] = ratio

fig, ax = plt.subplots(figsize=(12.5, 4.9))
ypos = [4.6, 3.4, 2.65, 1.9, 0.7]
for (b, grp, name), y in zip(rows_b, ypos):
    key = b if grp is None else f"{b}_{grp}"
    r = ratio[key]
    n = r["n_life_test"]
    left = 0
    hgt = 0.62 if grp is None else 0.5
    for cat, colr in zip(CATS, CAT_COLOR):
        k = r["all_counts_by_category"][cat]
        w = k / n * 100
        if k == 0:
            continue
        ax.barh(y, w, left=left, height=hgt, color=colr, edgecolor="white" if colr != "white" else "#9CA3AF",
                linewidth=1.6 if colr != "white" else 0.9, hatch="////" if colr == "white" else None)
        if w >= 7:
            tc = "white" if colr in (ps.SHORT, ps.LONG) else "#1F2937"
            ax.text(left + w / 2, y, f"{k}\n({w:.0f}%)", ha="center", va="center", fontsize=8.8, color=tc,
                    linespacing=1.1, fontweight="bold" if colr in (ps.SHORT, ps.LONG) else None)
        else:
            ax.text(left + w / 2, y, f"{k}", ha="center", va="center", fontsize=8.6, color="#1F2937")
        left += w
    nl = r["n_labeled"]
    side_all = (f"[전체 {n}] 장 {r['all_counts_by_category'][CATS[3]]} ({r['all_long_ratio']*100:.1f}%) · "
                f"단 {r['all_counts_by_category'][CATS[0]]} ({r['all_short_ratio']*100:.1f}%) · "
                f"<550 {r['all_lt550']} ({r['all_lt550_ratio']*100:.1f}%)")
    side_lab = (f"[라벨 {nl}] 장 {r['labeled_long_gt1000']} ({r['labeled_long_ratio']*100:.1f}%) · "
                f"단 {r['labeled_short_lt500']} ({r['labeled_short_ratio']*100:.1f}%) · "
                f"<550 {r['labeled_lt550']} ({r['labeled_lt550_ratio']*100:.1f}%)")
    if n != nl:
        ax.text(102, y + 0.13, side_all, va="center", ha="left", fontsize=8.9, color="#1F2937")
        ax.text(102, y - 0.15, side_lab, va="center", ha="left", fontsize=8.9, color="#6B7280")
    else:                                              # 하한값 셀이 없으면 두 기준이 같음 → 한 줄
        ax.text(102, y, side_all.replace(f"[전체 {n}]", f"[전체 = 라벨 {n}]"), va="center", ha="left",
                fontsize=8.9, color="#1F2937")
    ax.text(-1.5, y, f"{name}\n(수명시험 {n})", ha="right", va="center", fontsize=10 if grp is None else 9.3,
            fontweight="bold" if grp is None else None, color=ps.BATCH_COLOR[b] if grp is None else "#4B5563")
ax.set_xlim(0, 100)
ax.set_ylim(0.1, 5.3)
ax.set_yticks([])
ax.set_xticks([0, 25, 50, 75, 100])
ax.set_xticklabels(["0%", "25%", "50%", "75%", "100%"])
ax.grid(axis="y", visible=False)
ax.spines[["left"]].set_visible(False)
ax.set_xlabel("막대 = 수명시험 셀 전체 대비 비율 (하한값 반영: 중도절단·EOL 미도달 셀은 하한값이 구간을 확정할 때만 배정)")
ax.text(102, 5.12, "오른쪽 수치: [전체] = 하한값 반영(막대와 동일 기준) · [라벨] = EOL 도달 라벨 셀만",
        ha="left", va="bottom", fontsize=8.6, color="#6B7280")
leg = [Patch(fc=c, ec="#9CA3AF" if c == "white" else "white", hatch="////" if c == "white" else None, label=l)
       for c, l in zip(CAT_COLOR, CATS)]
ax.legend(handles=leg, loc="lower left", bbox_to_anchor=(0, 1.0), ncol=5, fontsize=9.2, handlelength=1.6)
fig.suptitle("Q1(b) 장수명(>1,000)·단수명(<500) 비율 — 단수명은 Batch 2 fastcharge에만 존재",
             x=0.02, y=1.05, ha="left", fontsize=14, fontweight="bold")
r1, r2 = ratio["batch1"], ratio["batch2"]
fig.text(0.02, 0.975, f"550 분류 라벨 기준 '단수명(0)': Batch 1 라벨 셀 {r1['n_labeled']}개 중 {r1['labeled_lt550']}개"
         f"({r1['labeled_lt550_ratio']*100:.1f}%; 하한값 반영 전체 {r1['n_life_test']}개 중 {r1['all_lt550']}개) vs "
         f"Batch 2 {r2['n_labeled']}개 중 {r2['labeled_lt550']}개({r2['labeled_lt550_ratio']*100:.1f}%) "
         "→ 학습·테스트 간 클래스 비율이 정반대", ha="left", fontsize=10.3, color="#374151")
ps.save(fig, "q1_long_short_ratio")

# ════════════════════════════════════════════════════════════════
# (+) 노션 문장 검증: 'Batch 1/2는 유사하지만, Batch 3는 분포 차이 있음'
# ════════════════════════════════════════════════════════════════
life = {b: LAB[LAB.batch == b].cycle_life.to_numpy() for b in BATCH_NAMES}
life["batch2_fastcharge"] = LAB[(LAB.batch == "batch2") & (LAB.group == "fastcharge")].cycle_life.to_numpy()
life["batch2_newstructure"] = LAB[(LAB.batch == "batch2") & (LAB.group == "newstructure")].cycle_life.to_numpy()
pairs = [("batch1", "batch2"), ("batch1", "batch3"), ("batch2", "batch3"),
         ("batch1", "batch2_fastcharge"), ("batch1", "batch2_newstructure"), ("batch2_newstructure", "batch3")]
tests = {}
for a, b in pairs:
    ks = stats.ks_2samp(life[a], life[b])
    mw = stats.mannwhitneyu(life[a], life[b], alternative="two-sided")
    tests[f"{a}_vs_{b}"] = {"n": [len(life[a]), len(life[b])], "ks_D": ks.statistic, "ks_p": ks.pvalue,
                             "mwu_p": mw.pvalue, "median_a": np.median(life[a]), "median_b": np.median(life[b])}
# 동일 프로토콜 · 다른 배치/구조 비교
multi = (T[T.life_test].groupby("base_policy")
         .filter(lambda g: g[["batch", "group"]].drop_duplicates().shape[0] > 1))
same_proto = {}
for pol, g in multi.groupby("base_policy"):
    same_proto[pol] = {}
    for (b, grp), gg in g.groupby(["batch", "group"]):
        same_proto[pol][f"{b}_{grp}"] = {"labeled_lives": sorted(gg[gg.labeled].cycle_life.tolist()),
                                        "lower_bounds": sorted(gg.lower_bound.dropna().tolist()),
                                        "median_labeled": gg[gg.labeled].cycle_life.median()}
# 배치 내 프로토콜이 설명하는 수명 분산 (복제셀 간 산포 vs 전체 산포)
proto_var = {}
for b in BATCH_NAMES:
    g = LAB[LAB.batch == b]
    gm = g.groupby("policy").cycle_life
    ss_w = ((g.cycle_life - gm.transform("mean")) ** 2).sum()
    ss_t = ((g.cycle_life - g.cycle_life.mean()) ** 2).sum()
    k = g.policy.nunique()
    proto_var[b] = {"n": len(g), "n_policies": k, "eta2_policy": 1 - ss_w / ss_t,
                    "pooled_within_policy_sd": np.sqrt(ss_w / (len(g) - k)), "total_sd": g.cycle_life.std(ddof=1)}
# 학습(Batch 1) 라벨 범위 밖 테스트 셀 & 범위 제한 모델의 MAPE 하한
b1min, b1max = life["batch1"].min(), life["batch1"].max()
range_check = {"batch1_label_range": [b1min, b1max]}
for b in ("batch2", "batch3"):
    y = life[b]
    below, above = (y < b1min).sum(), (y > b1max).sum()
    floor = np.maximum.reduce([np.zeros_like(y), b1min - y, y - b1max]) / y * 100
    range_check[b] = {"n": len(y), "n_below": int(below), "n_above": int(above),
                      "frac_outside": (below + above) / len(y), "mape_floor_pct": floor.mean()}
# 같은 프로토콜의 (배치, 구조) 그룹 간 '모든 쌍' 중앙값 비 (큰 값/작은 값) — 체리피킹 없이 전 범위 보고
same_proto_ratio = {}
for pol, d in same_proto.items():
    ks_ = sorted(d, key=lambda k: d[k]["median_labeled"])
    for i_, ka in enumerate(ks_):
        for kb in ks_[i_ + 1:]:
            same_proto_ratio[f"{pol}: {kb}/{ka}"] = {
                "ratio": d[kb]["median_labeled"] / d[ka]["median_labeled"],
                "median_hi": d[kb]["median_labeled"], "median_lo": d[ka]["median_labeled"],
                "n_hi": len(d[kb]["labeled_lives"]), "n_lo": len(d[ka]["labeled_lives"]),
                "same_structure": ka.split("_", 1)[1] == kb.split("_", 1)[1]}
rv = [v["ratio"] for v in same_proto_ratio.values()]
rv_same = [v["ratio"] for v in same_proto_ratio.values() if v["same_structure"]]
grp_ns = [len(v["labeled_lives"]) for d in same_proto.values() for v in d.values()]
# Batch 1 내부 프로토콜 효과(프로토콜 평균의 범위)와 비교
b1_pm = LAB[LAB.batch == "batch1"].groupby("policy").cycle_life.mean()
b1_rep48 = same_proto["4.8C(80%)-4.8C"]["batch1_fastcharge"]["labeled_lives"]
shared_b1_b2fc = sorted(set(LAB[LAB.batch == "batch1"].base_policy) &
                        set(LAB[(LAB.batch == "batch2") & (LAB.group == "fastcharge")].base_policy))
b2fc_only = sorted(set(LAB[(LAB.batch == "batch2") & (LAB.group == "fastcharge")].base_policy) -
                   set(LAB[LAB.batch == "batch1"].base_policy))
results["notion_claim_test"] = {
    "same_protocol_median_ratio_all_pairs": same_proto_ratio,
    "same_protocol_ratio_range": [min(rv), max(rv)],
    "same_protocol_ratio_range_same_structure": [min(rv_same), max(rv_same)],
    "same_protocol_group_n_range": [min(grp_ns), max(grp_ns)],
    "batch1_4.8C_replicates": b1_rep48,
    "batch1_policy_mean_range": {"min": b1_pm.min(), "min_policy": b1_pm.idxmin(), "max": b1_pm.max(),
                                 "max_policy": b1_pm.idxmax(), "ratio": b1_pm.max() / b1_pm.min()},
    "shared_protocols_batch1_batch2fc": shared_b1_b2fc,
    "batch2fc_protocols_absent_in_batch1": b2fc_only,
    "tests": tests, "same_protocol": same_proto, "protocol_variance": proto_var,
    "train_range_check": range_check}

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6.6), gridspec_kw=dict(width_ratios=[1, 1.05], wspace=0.30))
# ── 좌: ECDF ──
ecdf_specs = [("batch1", "Batch 1 (n=36)", ps.BATCH_COLOR["batch1"], "-", 2.2),
              ("batch2", "Batch 2 전체 (n=39)", ps.BATCH_COLOR["batch2"], "-", 2.2),
              ("batch2_newstructure", "Batch 2 newstructure만 (n=9)", ps.BATCH_COLOR["batch2"], ":", 1.8),
              ("batch3", "Batch 3 (n=44)", ps.BATCH_COLOR["batch3"], "-", 2.2)]
for key, lab_, colr, ls, lw in ecdf_specs:
    x = np.sort(life[key])
    yv = np.arange(1, len(x) + 1) / len(x)
    ax1.step(np.r_[HIST_RANGE[0], x], np.r_[0, yv], where="post", color=colr, ls=ls, lw=lw, label=lab_)
ax1.axhline(0.5, color="#9CA3AF", lw=0.8, ls="--")
for key, dx, ha in (("batch1", -22, "right"), ("batch2", -22, "right"), ("batch3", 22, "left")):
    m = np.median(life[key])
    ax1.plot(m, 0.5, "o", ms=7, color=ps.BATCH_COLOR[key], mec="white", zorder=5)
    ax1.text(m + dx, 0.535, f"{m:,.0f}" if m % 1 == 0 else f"{m:,.1f}", ha=ha, va="bottom", fontsize=9,
             color=ps.BATCH_COLOR[key], fontweight="bold", zorder=6, bbox=dict(fc="white", ec="none", pad=1))
ax1.set_xlim(300, 2000)
ax1.set_ylim(0, 1.02)
ax1.set_xlabel("Cycle Life (라벨 셀)")
ax1.set_ylabel("누적 비율 (ECDF)")
ax1.set_title("누적분포 비교 — ● = 중앙값", loc="left")
ax1.legend(loc="lower right", fontsize=9, bbox_to_anchor=(1.0, 0.0))
t_ = tests
ks_txt = ("2표본 KS 검정 (D, p)\n"
          f"B1 vs B2   D={t_['batch1_vs_batch2']['ks_D']:.2f}, p={t_['batch1_vs_batch2']['ks_p']:.1e}\n"
          f"B1 vs B3   D={t_['batch1_vs_batch3']['ks_D']:.2f}, p={t_['batch1_vs_batch3']['ks_p']:.1e}\n"
          f"B2 vs B3   D={t_['batch2_vs_batch3']['ks_D']:.2f}, p={t_['batch2_vs_batch3']['ks_p']:.1e}\n"
          f"B1 vs B2-fast   D={t_['batch1_vs_batch2_fastcharge']['ks_D']:.2f} (완전 분리)\n"
          f"B2-new vs B3   D={t_['batch2_newstructure_vs_batch3']['ks_D']:.2f}, "
          f"p={t_['batch2_newstructure_vs_batch3']['ks_p']:.2f}\n"
          "   n=9로 검정력 낮음\n"
          "   → 차이 검출 못 함 ≠ 같음")
# 상자 위치: x ≥ 1,215(라벨 셀 ECDF가 0.75 이상인 영역) · y 0.74 이하 → 어떤 계단과도 겹치지 않음
ax1.text(1215, 0.74, ks_txt, ha="left", va="top", fontsize=8.7,
         linespacing=1.45, bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#D1D5DB"))
# ── 우: 동일 프로토콜, 다른 배치/구조 ──
order = [("4.8C(80%)-4.8C", ["batch1_fastcharge", "batch2_fastcharge", "batch2_newstructure", "batch3_newstructure"]),
         ("6C(60%)-3C", ["batch1_fastcharge", "batch2_fastcharge"]),
         ("5.2C(58%)-4C", ["batch2_fastcharge", "batch2_newstructure"]),
         ("5.6C(26%)-4.5C", ["batch2_fastcharge", "batch2_newstructure"])]
y, yt, ytl, heads = 0, [], [], []
rng = np.random.default_rng(RANDOM_STATE)
for pol, keys in order:
    y -= 0.55
    heads.append((pol, y))
    for k in keys:
        y -= 1
        b, grp = k.split("_", 1)
        d = same_proto[pol][k]
        colr = ps.BATCH_COLOR[b]
        mk = "o" if grp == "fastcharge" else "s"
        v = np.array(d["labeled_lives"])
        ax2.scatter(v, y + rng.uniform(-0.13, 0.13, len(v)), s=46, marker=mk, color=colr, ec="white", lw=0.7,
                    zorder=3)
        for lbv in d["lower_bounds"]:
            ax2.plot(lbv, y, marker=">", ms=8.5, mfc="white", mec=colr, mew=1.6, ls="none", zorder=3)
        med = d["median_labeled"]
        ax2.plot([med, med], [y - 0.33, y + 0.33], color="#111827", lw=2.0, zorder=4)
        extra = f" (+{len(d['lower_bounds'])}개 >{min(d['lower_bounds']):,.0f})" if d["lower_bounds"] else ""
        ax2.text(1.015, y, f"중앙값 {med:,.0f}{extra}", ha="left", va="center", fontsize=8.8, color="#374151",
                 transform=ax2.get_yaxis_transform(), clip_on=False)
        yt.append(y)
        ytl.append(f"{BL[b]} · {grp}")
    y -= 0.25
for pol, yy in heads:
    pr = [v["ratio"] for k, v in same_proto_ratio.items() if k.startswith(pol + ":")]
    rtxt = f"그룹 간 중앙값 비 {min(pr):.2f}~{max(pr):.2f}배" if len(pr) > 1 else f"중앙값 비 {pr[0]:.2f}배"
    th = ax2.text(310, yy, pol, ha="left", va="center", fontsize=10, fontweight="bold", color="#111827")
    ax2.annotate(f"({rtxt})", xy=(1, 0.5), xycoords=th, xytext=(6, 0), textcoords="offset points",
                 ha="left", va="center", fontsize=8.8, color="#4B5563")
ax2.text(1110, -2.2, f"같은 프로토콜이어도 쌍에 따라 차이가 다름:\n"
         f"· B1 fastcharge 753 ≈ B2 newstructure 809 (1.07배)\n"
         f"· B1 복제셀 2개 자체가 {b1_rep48[0]:,.0f} vs {b1_rep48[1]:,.0f} ({b1_rep48[1] / b1_rep48[0]:.2f}배)\n"
         f"· 각 그룹 n = {min(grp_ns)}~{max(grp_ns)} → 중앙값 비는 불안정",
         ha="left", va="center", fontsize=8.8, color="#374151", linespacing=1.45,
         bbox=dict(boxstyle="round,pad=0.45", fc="white", ec="#D1D5DB"))
ax2.set_yticks(yt)
ax2.set_yticklabels(ytl, fontsize=9)
ax2.set_xlim(300, 2300)
ax2.set_ylim(y - 0.25, 0.05)
ax2.grid(axis="y", visible=False)
ax2.set_xlabel("Cycle Life  (검은 세로선 = 라벨 셀 중앙값, ● fastcharge, ■ newstructure, ▷ EOL 미도달 하한값)")
ax2.set_title("같은 충전 프로토콜, 다른 배치·셀 구조", loc="left")
fig.suptitle("노션 문장 검증 — 본 파일 기준 'Batch 3 상이'는 확인, 'Batch 1/2 유사'는 확인되지 않음", x=0.04, y=1.03,
             ha="left", fontsize=14, fontweight="bold")
fig.text(0.04, 0.965, "Batch 1↔2 차이(D=0.77)가 Batch 1↔3(D=0.47)보다 큼 · Batch 2 newstructure vs Batch 3는 차이 검출 안 됨"
         f"(n=9, p=0.73) · 같은 프로토콜의 배치·구조 간 중앙값 비는 쌍에 따라 {min(rv):.2f}~{max(rv):.2f}배"
         f"(각 그룹 n={min(grp_ns)}~{max(grp_ns)})", ha="left", fontsize=10.3, color="#374151")
ps.save(fig, "q1_batch_compare")

# ════════════════════════════════════════════════════════════════
# (c) 이상치 셀 식별 - 왜 유독 짧은가?
# ════════════════════════════════════════════════════════════════
# 규칙 1: Tukey 1.5×IQR (배치 내, 라벨 셀). Batch 2 는 이봉이라 배치×그룹 기준도 함께.
# 규칙 2: robust z = (x − 중앙값) / (1.4826·MAD), |z| > 3.5
# 규칙 3(운영 정의): '유독 짧은 셀' = 배치×그룹 내 하위 10% (P10 이하) → 근거 비교 대상
out_rules = {}
for scope, keys in (("batch", ["batch"]), ("batch_group", ["batch", "group"])):
    for kk, g in LAB.groupby(keys):
        kk = kk if isinstance(kk, str) else "_".join(kk)
        x = g.cycle_life
        q1, q3 = x.quantile([0.25, 0.75])
        lo, hi = q1 - 1.5 * (q3 - q1), q3 + 1.5 * (q3 - q1)
        mad = stats.median_abs_deviation(x, scale="normal")
        rz = (x - x.median()) / mad
        out_rules[f"{scope}:{kk}"] = {
            "n": len(x), "q1": q1, "q3": q3, "lower_fence": lo, "upper_fence": hi,
            "low_outliers": g[x < lo].cell_key.tolist(), "high_outliers": g[x > hi].cell_key.tolist(),
            "robust_z_min": rz.min(), "robust_z_max": rz.max(),
            "robust_z_gt3.5": g[rz.abs() > 3.5].cell_key.tolist(), "p10": x.quantile(0.10)}
LAB["bg"] = LAB.batch + "_" + LAB.group
LAB["p10"] = LAB.groupby("bg").cycle_life.transform(lambda s: s.quantile(0.10))
SHORT_CELLS = LAB[LAB.cycle_life <= LAB.p10].copy()


def mates_median(row) -> float:
    m = LAB[(LAB.batch == row.batch) & (LAB.policy == row.policy) & (LAB.cell_key != row.cell_key)]
    return m.cycle_life.median() if len(m) else np.nan


SHORT_CELLS["mates_lives"] = SHORT_CELLS.apply(
    lambda r: sorted(LAB[(LAB.batch == r.batch) & (LAB.policy == r.policy) & (LAB.cell_key != r.cell_key)]
                     .cycle_life.tolist()), axis=1)
SHORT_CELLS["mates_median"] = SHORT_CELLS.apply(mates_median, axis=1)
SHORT_CELLS["life_over_mates"] = SHORT_CELLS.cycle_life / SHORT_CELLS.mates_median
SHORT_CELLS["group_q25"] = SHORT_CELLS.bg.map(LAB.groupby("bg").cycle_life.quantile(0.25))
# 동료셀(같은 배치·같은 프로토콜) 중앙값도 그룹 하위 25% 이내면 '프로토콜형', 아니면 '개별 셀형'
SHORT_CELLS["short_type"] = np.where(SHORT_CELLS.mates_median <= SHORT_CELLS.group_q25, "프로토콜형", "개별 셀형")
SHORT_CELLS["life_pct_in_group"] = SHORT_CELLS.apply(
    lambda r: (LAB[LAB.bg == r.bg].cycle_life <= r.cycle_life).mean() * 100, axis=1)

# 근거 지표 — '도메인상 수명에 불리한 방향' (+1: 클수록 불리, −1: 작을수록 불리)
EVID = [("C1", "1단계 C-rate\nC1", +1, "{:.1f}C"),
        ("C2", "2단계 C-rate\nC2", +1, "{:.1f}C"),
        ("avgC_80", "평균 C-rate\n(0→80%)", +1, "{:.2f}C"),
        ("chargetime_2_6", "충전시간\n(cyc 2~6)", -1, "{:.1f}분"),
        ("dQD_100_10", "용량 변화\nQD(100)−QD(10)", -1, "{:+.1f}mAh"),
        ("log10_var_dQ", "log10 Var\nΔQ(100−10)", +1, "{:.2f}"),
        ("QD_10", "초기 용량\nQD(10)", -1, "{:.3f}Ah"),
        ("IR_mean_2_100", "내부저항 평균\n(cyc 2~100)", +1, "{:.2f}mΩ"),
        ("Tmax_mean_2_100", "최고온도 평균\n(cyc 2~100)", +1, "{:.1f}°C")]
b1 = LAB[LAB.batch == "batch1"]
rho_b1 = {k: stats.spearmanr(b1[k], b1.cycle_life).statistic for k, *_ in EVID}
p_b1 = {k: stats.spearmanr(b1[k], b1.cycle_life).pvalue for k, *_ in EVID}


def risk_pct(row, col, sign) -> float:
    """배치×그룹 내 '불리한 방향' 백분위 (100 = 그룹에서 가장 불리). 그룹 내 변동이 사실상 없으면 NaN."""
    g = LAB[LAB.bg == row.bg][col].dropna()                 # Batch 2 #40~46 IR 결측 제외
    if not np.isfinite(row[col]) or (g.max() - g.min()) / abs(g.median()) < 0.01:
        return np.nan
    v = sign * g.to_numpy()
    x = sign * row[col]
    return ((v < x).sum() + 0.5 * (v == x).sum()) / len(v) * 100            # 동점은 중간 순위


for k, _, s, _ in EVID:
    SHORT_CELLS[f"rp_{k}"] = SHORT_CELLS.apply(lambda r: risk_pct(r, k, s), axis=1)
SHORT_CELLS = SHORT_CELLS.sort_values(["batch", "group", "cycle_life"]).reset_index(drop=True)

# 요약: 짧은 셀 중 각 근거가 '불리한 쪽 상위 30%'(백분위 ≥70)인 비율
evid_summary = {}
for k, *_ in EVID:
    v = SHORT_CELLS[f"rp_{k}"].dropna()
    evid_summary[k] = {"n_evaluable": len(v), "n_pct_ge70": int((v >= 70).sum()),
                       "median_risk_pct": v.median() if len(v) else np.nan, "spearman_rho_batch1": rho_b1[k]}
# Batch 1만의 근거(피처 판단의 주 근거) vs Batch 2·3 짧은 셀(일관성 점검) 분리
sc_b1 = SHORT_CELLS[SHORT_CELLS.batch == "batch1"]
sc_oth = SHORT_CELLS[SHORT_CELLS.batch != "batch1"]
evid_split = {k: {"batch1_n": len(sc_b1),
                  # 셀 번호 → 백분위 (순서 혼동 방지: 목록이 아니라 셀 키로 보관)
                  "batch1_pct_by_cell": {f"#{int(i):02d}": round(float(v), 1)
                                         for i, v in zip(sc_b1.idx, sc_b1[f"rp_{k}"])},
                  "batch1_pcts": sc_b1[f"rp_{k}"].round(1).tolist(),
                  "batch1_n_ge70": int((sc_b1[f"rp_{k}"] >= 70).sum()),
                  "batch1_n_ge90": int((sc_b1[f"rp_{k}"] >= 90).sum()),
                  "batch23_n_evaluable": int(sc_oth[f"rp_{k}"].notna().sum()),
                  "batch23_n_ge70": int((sc_oth[f"rp_{k}"] >= 70).sum())} for k, *_ in EVID}
# '본 셀/동료' 비 요약 (프로토콜형)
pt = SHORT_CELLS[SHORT_CELLS.short_type == "프로토콜형"]
in_band = pt[(pt.life_over_mates >= 0.95) & (pt.life_over_mates <= 1.05)]
peer_ratio_summary = {"n_protocol_type": len(pt), "n_within_0.95_1.05": len(in_band),
                      "within_range": [in_band.life_over_mates.min(), in_band.life_over_mates.max()],
                      "outside": pt[~pt.cell_key.isin(in_band.cell_key)][["cell_key", "life_over_mates"]]
                      .to_dict(orient="records"),
                      "all_range": [pt.life_over_mates.min(), pt.life_over_mates.max()]}
# 배치×그룹별 '짧은 프로토콜' 패턴 (프로토콜 중앙값 순위) — 배치마다 따로 서술하기 위한 표
proto_rank = {}
for bg, g in LAB.groupby("bg"):
    pr_ = (g.groupby("policy").agg(n=("cycle_life", "size"), median=("cycle_life", "median"),
                                   C1=("C1", "first"), Q1=("Q1", "first"), C2=("C2", "first"),
                                   avgC_80=("avgC_80", "first")).sort_values("median"))
    proto_rank[bg] = {"p10": g.cycle_life.quantile(0.10),
                      "protocols": pr_.reset_index().to_dict(orient="records")}
# Batch 1 짧은 셀(P10) vs 나머지 — 근거 지표 '중앙값'(평균 병기)
b1s = b1[b1.cycle_life <= b1.cycle_life.quantile(0.10)]
b1r = b1[b1.cycle_life > b1.cycle_life.quantile(0.10)]
b1_contrast = {k: {"n_short": len(b1s), "n_rest": len(b1r), "short_median": b1s[k].median(), "rest_median": b1r[k].median(),
                   "short_mean": b1s[k].mean(), "rest_mean": b1r[k].mean(),
                   "mwu_p": stats.mannwhitneyu(b1s[k], b1r[k]).pvalue} for k, *_ in EVID}
# Batch 1 단일 사이클 QD 잡음 vs QD(100)−QD(10) 크기: cycle 10~100 QD를 2차식으로 적합한 잔차 SD
qd_noise = []
for ck in b1.cell_key:
    s_ = clean_summary(CELL[ck]["summary"]).set_index("cycle").loc[10:100, "QD"].dropna()
    res_ = s_.to_numpy() - np.polyval(np.polyfit(s_.index.to_numpy(), s_.to_numpy(), 2), s_.index.to_numpy())
    qd_noise.append(res_.std(ddof=3) * 1000)
dqd_abs = (b1.dQD_100_10.abs() * 1000)
qd_noise_b1 = {"resid_sd_mAh_median": float(np.median(qd_noise)), "resid_sd_mAh_p90": float(np.quantile(qd_noise, 0.9)),
               "diff_noise_sd_mAh_median": float(np.median(qd_noise) * np.sqrt(2)),
               "abs_dQD_mAh_median": dqd_abs.median(), "abs_dQD_mAh_q25": dqd_abs.quantile(0.25),
               "n_abs_dQD_lt_2x_diffnoise": int((dqd_abs < 2 * np.median(qd_noise) * np.sqrt(2)).sum()), "n": len(b1),
               "method": "cycle 10~100 QD에 2차 다항식 적합 → 잔차 SD(단일 사이클 잡음). 두 사이클 차의 잡음 SD = √2 × 잔차 SD"}


# 배치×그룹별 Spearman(지표, 수명) — n·p·지표 범위 함께 (Batch 2·3는 일반화 일관성 점검용, 선택 근거 아님)
def sp_row(g, k):
    v = g[[k, "cycle_life"]].dropna()
    rel = (v[k].max() - v[k].min()) / abs(v[k].median()) if len(v) else np.nan
    out = {"n": len(v), "min": v[k].min(), "max": v[k].max(), "rel_range": rel}
    if len(v) > 2 and v[k].nunique() > 1:
        r_ = stats.spearmanr(v[k], v.cycle_life)
        out.update(rho=r_.statistic, p=r_.pvalue)
    out["interpretable"] = bool(rel >= 0.01)          # 상대 범위 1% 미만이면 해석 불가로 표시
    return out


spearman_bg = {bg: {k: sp_row(g, k) for k, *_ in EVID} for bg, g in LAB.groupby("bg")}


# ── 데이터 품질 점검 (수명 '값'의 통계적 이상치와 별개) ──
#   QD 점프: 연속 사이클 |ΔQD| > 10 mAh 횟수 (전체 기록), 초기 QD 점프: cycle 2~100에서 |ΔQD| > 3 mAh 횟수
#   IR 스파이크: |IR − 이동중앙값(11)| > 1 mΩ 횟수 (전체 기록)
def quality(c: dict) -> dict:
    s = clean_summary(c["summary"])
    s = s[s.cycle >= 2]
    q = s.QD.to_numpy() * 1000
    d = np.diff(q)
    d = d[np.isfinite(d)]
    qe = s[s.cycle <= 100].QD.to_numpy() * 1000
    de = np.diff(qe)
    de = de[np.isfinite(de)]
    ir = s.IR.to_numpy() * 1000
    irm = pd.Series(ir).rolling(11, center=True, min_periods=5).median().to_numpy()
    tm = s.Tmax.to_numpy()
    steps = [abs(np.nanmedian(tm[t:t + 50]) - np.nanmedian(tm[t - 50:t])) for t in range(50, len(tm) - 50)]
    return {"qd_jump10": int((np.abs(d) > 10).sum()), "qd_jump3_early": int((np.abs(de) > 3).sum()),
            "ir_spike1": int((np.abs(ir - irm) > 1).sum()) if np.isfinite(ir).any() else np.nan,
            "tmax_step_max": float(np.nanmax(steps)) if steps else np.nan,               # 참고용(플래그 미사용)
            "tmax_step_at": int(s.cycle.iloc[50 + int(np.nanargmax(steps))]) if steps else -1}


QC = pd.DataFrame([{"cell_key": c["cell_key"], **quality(c)} for c in cells]).merge(
    T[["cell_key", "batch", "idx", "group", "policy", "cycle_life", "labeled", "life_test", "QD_10"]], on="cell_key")
QC = QC[QC.life_test].copy()
QC["qd10_rank_low"] = QC.groupby(["batch", "group"]).QD_10.rank(method="min")     # 1 = 그룹 내 최저 초기용량
QC["flag"] = (QC.qd_jump10 >= 3) | (QC.ir_spike1 >= 5)                             # 운영 정의 (배치 최댓값 수준)
q_base = {b: {"qd_jump10_median": QC[QC.batch == b].qd_jump10.median(), "qd_jump10_max": QC[QC.batch == b].qd_jump10.max(),
              "qd_jump3_early_median": QC[QC.batch == b].qd_jump3_early.median(),
              "qd_jump3_early_p90": QC[QC.batch == b].qd_jump3_early.quantile(0.9),
              "ir_spike1_median": QC[QC.batch == b].ir_spike1.median(),
              "tmax_step_max_p90": QC[QC.batch == b].tmax_step_max.quantile(0.9),
              "flagged": QC[(QC.batch == b) & QC.flag].cell_key.tolist()} for b in BATCH_NAMES}
PAPER_B3_REMOVED = [2, 23, 32, 37, 38, 39]       # 원논문 공개 코드가 Batch 3에서 제거한 것으로 알려진 채널 번호(셀 번호 대응은 미확인)
b3q = QC[QC.batch == "batch3"].set_index("idx")
b3_check = []
for i_ in sorted(set(PAPER_B3_REMOVED) | {7, 45}):
    r_ = b3q.loc[i_]
    mates = LAB[(LAB.batch == "batch3") & (LAB.policy == r_.policy) & (LAB.idx != i_)].cycle_life
    b3_check.append({"idx": i_, "cell_key": r_.cell_key, "policy": r_.policy, "cycle_life": r_.cycle_life,
                     "labeled": bool(r_.labeled), "paper_removed_channel": i_ in PAPER_B3_REMOVED,
                     "tukey_high": r_.cell_key in out_rules["batch:batch3"]["high_outliers"],
                     "mates_lives": sorted(mates.tolist()), "life_over_mates_median": (r_.cycle_life / mates.median())
                     if r_.labeled else np.nan,
                     "qd_jump10": r_.qd_jump10, "qd_jump3_early": r_.qd_jump3_early, "ir_spike1": r_.ir_spike1,
                     "tmax_step_max": r_.tmax_step_max, "tmax_step_at": r_.tmax_step_at,
                     "QD_10": r_.QD_10, "qd10_rank_low_of_46": int(r_.qd10_rank_low), "flag": bool(r_.flag)})
short_q = QC[QC.cell_key.isin(SHORT_CELLS.cell_key)][["cell_key", "qd_jump10", "qd_jump3_early", "ir_spike1", "flag"]]
results["c_data_quality"] = {
    "definition": "QD 점프 = 연속 사이클 |ΔQD|>10 mAh 횟수(전체 기록); 초기 QD 점프 = cycle 2~100 |ΔQD|>3 mAh; "
                  "IR 스파이크 = |IR−이동중앙값(11)|>1 mΩ; flag = QD 점프 ≥3 또는 IR 스파이크 ≥5 (운영 정의)",
    "batch_baseline": q_base, "batch3_paper_removed_and_upper_outliers": b3_check,
    "notion_day2_batch3_note": "노션 DAY 2 ▶ Batch 3: '이상치 제거 고려 : 배치 수집 시기 사이에 수개월 공백이 있어, "
                               "일부 셀은 데이터 품질 문제로 원논문에서도 제거됨'",
    "paper_channel_to_cell_mapping": "원논문 제거 채널 번호(c2·c23·c32·c37·c38·c39)와 본 파일 셀 번호의 대응은 미확인 "
                                     "(EOL 미도달 셀이 23·32라는 점은 대응 가능성을 시사할 뿐 확인은 아님)",
    "short_cells": short_q.to_dict(orient="records"), "n_short_flagged": int(short_q.flag.sum()),
}
# 충전시간이 프로토콜과 중복되는지 (Batch 1): Spearman(chargetime_2_6, avgC_80)
_r = stats.spearmanr(b1.chargetime_2_6, b1.avgC_80)
redund_b1 = {"chargetime_vs_avgC80_rho": _r.statistic, "p": _r.pvalue}
# robust z 값 (Batch 3 상단)
_x3 = LAB[LAB.batch == "batch3"].set_index("idx").cycle_life
robust_z_b3 = ((_x3 - _x3.median()) / stats.median_abs_deviation(_x3, scale="normal")).loc[[7, 38, 45]].to_dict()
results["c_outliers"] = {
    "batch1_chargetime_avgC80_redundancy": redund_b1,
    "batch3_upper_robust_z": robust_z_b3,
    "spearman_by_batch_group": spearman_bg,
    "evidence_split_batch1_vs_batch23": evid_split,
    "peer_ratio_summary": peer_ratio_summary,
    "protocol_rank_by_batch_group": proto_rank,
    "batch1_qd_noise": qd_noise_b1,
    "rules": out_rules,
    "short_type_rule": "동일 배치·동일 프로토콜 동료셀 수명 중앙값 ≤ 배치×그룹 Q1(25%) → 프로토콜형, 아니면 개별 셀형",
    "short_type_counts": SHORT_CELLS.short_type.value_counts().to_dict(),
    "short_cells_p10": SHORT_CELLS[["cell_key", "policy", "group", "cycle_life", "life_pct_in_group", "mates_lives",
                                    "mates_median", "life_over_mates", "group_q25", "short_type",
                                    "C1", "Q1", "C2", "avgC_80"] +
                                   [k for k, *_ in EVID if k not in ("C1", "C2", "avgC_80")] +
                                   [f"rp_{k}" for k, *_ in EVID]].to_dict(orient="records"),
    "evidence_summary": evid_summary,
    "batch1_short_vs_rest": b1_contrast,
    "risk_direction": {k: ("클수록 불리" if s > 0 else "작을수록 불리") for k, _, s, _ in EVID},
}

# ── Batch 1 복제셀 산포 분해: 'ΔQ 분산은 프로토콜 간 신호, 복제셀 간 산포는 Qcc_init과 연관' (Q5-(d)와 같은 계산) ──
#   dQ_logvar = log10 Var ΔQ_{100−10}(V)   (위 log10_var_dQ 와 동일)
#   Qcc_init  = cycle 2~6 Qdlin 의 2.0 V 끝점(CC 방전용량) 중앙값 — 원시 Qdlin '절대 수준' 이다.
#   노션 경고: '충전 커브 시작 시점이 배치별로 상이 : Qdlin 변수를 단순 비교하면 왜곡 발생' → 배치 이동량을 라벨 없이 정량화.
def qcc_init(c: dict) -> float:
    v = np.array([c["Qdlin"][k - 1][-1] for k in range(2, 7)], float)       # data.qdlin 규칙: 행 k−1 = cycle k
    v[(v <= 0) | (v > 1.2)] = np.nan
    return float(np.nanmedian(v))


T["Qcc_init"] = T.cell_key.map({k: qcc_init(c) for k, c in CELL.items()})
RB = T[(T.batch == "batch1") & T.labeled].copy()
RB["ly"] = np.log10(RB.cycle_life)
RB["lv"] = RB.log10_var_dQ
pm_ = RB.groupby("policy")[["lv", "Qcc_init", "ly"]].mean()
multi = RB.groupby("policy").policy.transform("size") >= 2
rep = {"n_cells": len(RB), "n_policies": len(pm_), "n_policies_with_replicates": int((RB.groupby("policy").size() >= 2).sum()),
       "n_cells_in_replicated_policies": int(multi.sum())}
for f in ("lv", "Qcc_init"):
    dv = (RB[f] - RB.groupby("policy")[f].transform("mean"))[multi]
    dy = (RB.ly - RB.groupby("policy").ly.transform("mean"))[multi]
    pr = stats.pearsonr(pm_[f], pm_.ly)
    wr = stats.pearsonr(dv, dy)
    rep[f] = {"policy_mean_pearson_r": pr.statistic, "policy_mean_p": pr.pvalue,
              "within_policy_pearson_r": wr.statistic, "within_policy_p": wr.pvalue}
pairs_ = []
for pol, d_ in RB.groupby("policy"):
    if len(d_) == 2:
        a_, b_ = d_.iloc[0], d_.iloc[1]
        pairs_.append((a_.ly - b_.ly, a_.lv - b_.lv, a_.Qcc_init - b_.Qcc_init))
pairs_ = np.array(pairs_)
k_lv = int((np.sign(pairs_[:, 1]) == -np.sign(pairs_[:, 0])).sum())      # 기대 방향: 분산 ↑ → 수명 ↓
k_q = int((np.sign(pairs_[:, 2]) == np.sign(pairs_[:, 0])).sum())        # 기대 방향: Qcc ↑ → 수명 ↑
rep["pairs"] = {"n_pairs": len(pairs_), "lv_expected_direction": k_lv, "Qcc_expected_direction": k_q,
                "Qcc_sign_test_p": stats.binomtest(k_q, len(pairs_)).pvalue,
                "lv_sign_test_p": stats.binomtest(k_lv, len(pairs_)).pvalue}
# 기술 회귀(B1 36셀, OLS, 예측 아님): log10 life ~ dQ_logvar (M0) / + Qcc_init (M1) — 배치 이동량을 예측 배율로 환산하기 위한 계수
X1_ = np.column_stack([np.ones(len(RB)), RB.lv, RB.Qcc_init])
coef1 = np.linalg.lstsq(X1_, RB.ly, rcond=None)[0]
X0_ = np.column_stack([np.ones(len(RB)), RB.lv])
coef0 = np.linalg.lstsq(X0_, RB.ly, rcond=None)[0]
rep["descriptive_ols_B1"] = {"M0_coef_[b0,lv]": coef0.tolist(), "M1_coef_[b0,lv,Qcc_init(Ah)]": coef1.tolist(),
                             "note": "B1 36셀 OLS 기술 회귀(모델 선택 아님). M0/M1 채택은 DAY 2 B1 정책 단위 grouped-CV 1-SE 규칙으로만 결정"}
ref_q = T[T.batch == "batch1"].Qcc_init.median()                     # B1 46셀(라벨 미사용) 중앙값
sd_q = RB.Qcc_init.std()
offs = {}
for (b_, g_), d_ in T[(T.batch != "batch1") & T.life_test].groupby(["batch", "group"]):
    o_ = d_.Qcc_init.median() - ref_q
    offs[f"{b_}_{g_}"] = {"n": len(d_), "offset_mAh": o_ * 1000, "offset_in_B1_SD": o_ / sd_q,
                          "pred_factor_M1": 10 ** (coef1[2] * o_),
                          "n_outside_B1_labeled_range": int(((d_.Qcc_init < RB.Qcc_init.min()) |
                                                             (d_.Qcc_init > RB.Qcc_init.max())).sum())}
rep["Qcc_init_unlabeled_batch_offsets"] = {
    "reference": "Batch 1 46셀 Qcc_init 중앙값(라벨 미사용)", "B1_median_Ah": ref_q, "B1_labeled_sd_mAh": sd_q * 1000,
    "by_batch_group": offs,
    "notion_warning": "노션 DAY 2 ▶ Batch 3: '충전 커브 시작 시점이 배치별로 상이 : Qdlin 변수를 단순 비교하면 왜곡 발생'",
    "headroom_caveat": "고정 EOL 0.88 Ah 라벨에서는 초기 용량이 클수록 EOL까지 여유(헤드룸)가 커서 Qcc_init 효과 일부는 라벨 정의의 산물일 수 "
                       "있음(노션 SOH = 현재/초기 용량은 상대 정의)"}
# 같은 배치·같은 프로토콜 복제셀의 수명 최대/최소 비 (ESS 해석용)
wpr = {}
for b_ in BATCH_NAMES:
    L_ = LAB[LAB.batch == b_].groupby("policy").cycle_life.agg(["min", "max", "size"])
    L_ = L_[L_["size"] >= 2]
    rr = L_["max"] / L_["min"]
    wpr[b_] = {"n_policies": len(L_), "max_ratio": rr.max(), "max_policy": rr.idxmax(), "median_ratio": rr.median()}
rep["within_protocol_max_min_ratio"] = wpr
results["c_replicate_spread_batch1"] = jsonable(rep)

# ── 그림 (PDF 16 cm 폭 기준 간결판): 짧은 셀 근거 히트맵 — Batch 1에서 판단 가능한 열 + 대조용 IR 만 ──
#    (C1·C2·충전시간·QD(10)·Tmax 열과 셀별 상세 수치는 q1_results.json c_outliers.short_cells_p10 에 보관)
EVID_FIG = [("avgC_80", "평균\nC-rate\n(0→80%)", +1, "{:.2f}"),
            ("dQD_100_10", "QD(100)\n−QD(10)\n(mAh)", -1, "{:+.1f}"),
            ("log10_var_dQ", "log10 Var\nΔQ\n(100−10)", +1, "{:.2f}"),
            ("IR_mean_2_100", "내부저항\n(mΩ, cyc\n2~100)", +1, "{:.1f}")]
nr, ncol_ev = len(SHORT_CELLS), len(EVID_FIG)
cmap = plt.get_cmap("Reds")
fig, ax = plt.subplots(figsize=(7.3, 5.9))
fig.subplots_adjust(left=0.235, right=0.99, top=0.86, bottom=0.03)
xc = {"life": 0.45, "mates": 1.55, "type": 2.75}
x0 = 3.8
DX = 1.1                     # 근거 열 간격
for i, r in SHORT_CELLS.iterrows():
    ax.text(xc["life"], i, f"{r.cycle_life:,.0f}", ha="center", va="center", fontsize=9, fontweight="bold",
            color=ps.SHORT)
    if np.isfinite(r.mates_median):
        mt = (", ".join(f"{v:,.0f}" for v in r.mates_lives) if len(r.mates_lives) <= 2
              else f"{len(r.mates_lives)}셀 중앙 {r.mates_median:,.0f}")
        ax.text(xc["mates"], i, f"{mt}\n(비 {r.life_over_mates:.2f})", ha="center", va="center", fontsize=7.8,
                color="#1F2937", linespacing=1.1)
    tcol = "#1F2937" if r.short_type == "프로토콜형" else "#7C3AED"
    ax.text(xc["type"], i, r.short_type, ha="center", va="center", fontsize=8.2, color=tcol, fontweight="bold")
    for j, (k, _, s, fmt) in enumerate(EVID_FIG):
        p = r[f"rp_{k}"]
        val = r[k] * (1000 if k == "dQD_100_10" else 1)
        if np.isfinite(p):
            fcol = cmap(0.08 + 0.8 * p / 100)
            tc = "white" if p >= 62 else "#111827"
            txt = f"{fmt.format(val)}\n{p:.0f}%"
        else:
            fcol, tc, txt = "#F3F4F6", "#6B7280", f"{fmt.format(val)}\n그룹 내 동일"
        ax.add_patch(Rectangle((x0 + DX * j - 0.52, i - 0.45), 1.04, 0.9, fc=fcol, ec="white", lw=1.2))
        ax.text(x0 + DX * j, i, txt, ha="center", va="center", fontsize=7.9, color=tc, linespacing=1.1)
ylabels = [f"{BL[r.batch].replace('Batch ', 'B')} #{r.idx:02d}  {r.policy.replace('-newstructure', '')}"
           for _, r in SHORT_CELLS.iterrows()]
ax.set_yticks(range(nr))
ax.set_yticklabels(ylabels, fontsize=8.3)
for tl, (_, r) in zip(ax.get_yticklabels(), SHORT_CELLS.iterrows()):
    tl.set_color(ps.BATCH_COLOR[r.batch])
prev = None
for i, r in SHORT_CELLS.iterrows():
    if prev is not None and r.bg != prev:
        ax.axhline(i - 0.5, color="#6B7280", lw=0.8)
    prev = r.bg
for bg, g in SHORT_CELLS.groupby("bg", sort=False):
    b, grp = bg.split("_", 1)
    p10 = LAB[LAB.bg == bg].cycle_life.quantile(0.1)
    gab = {"fastcharge": "fc", "newstructure": "ns"}[grp]
    ax.text(x0 + DX * (ncol_ev - 1) + 0.62, g.index.to_numpy().mean(), f"{BL[b].replace('Batch ', 'B')} {gab}\nP10 {p10:,.0f}",
            ha="left", va="center", fontsize=7.5, color=ps.BATCH_COLOR[b], linespacing=1.15)
heads_txt = ["수명", "동료셀 수명\n(본 셀/동료 비)", "유형"] + [h for _, h, _, _ in EVID_FIG]
xs = [xc["life"], xc["mates"], xc["type"]] + [x0 + DX * j for j in range(ncol_ev)]
for xh, h in zip(xs, heads_txt):
    ax.text(xh, -0.65, h, ha="center", va="bottom", fontsize=8.0, fontweight="bold", color="#111827", linespacing=1.1)
for j, (k, _, s, _) in enumerate(EVID_FIG):
    sig = p_b1[k] < 0.05
    tag = "" if sig else "\n(비유의)"
    ax.text(x0 + DX * j, nr - 0.38, f"ρ_B1 {rho_b1[k]:+.2f}\np={p_b1[k]:.2g}{tag}", ha="center", va="top", fontsize=7.9,
            color="#374151" if sig else "#9CA3AF", fontweight="bold" if sig else None, linespacing=1.15)
ax.text(xc["mates"] + 0.9, nr - 0.38, "ρ_B1 = Batch 1 라벨 36셀\nSpearman(지표, 수명) →", ha="right", va="top",
        fontsize=7.6, color="#6B7280")
ax.set_xlim(-0.05, x0 + DX * (ncol_ev - 1) + 1.45)
ax.set_ylim(nr + 0.75, -1.75)
ax.set_xticks([])
ax.grid(False)
for sp in ax.spines.values():
    sp.set_visible(False)
ax.tick_params(axis="y", length=0)
n_proto = int((SHORT_CELLS.short_type == "프로토콜형").sum())
fig.text(0.015, 0.995, f"Q1(c) '유독 짧은 셀' {nr}개(배치×그룹 하위 10%) — {n_proto}개는 동료셀도 짧은 프로토콜형",
         ha="left", va="top", fontsize=11.2, fontweight="bold")
fig.text(0.015, 0.953, "색·% = 배치×그룹 내 '수명에 불리한 방향' 백분위(100% = 가장 불리) · 프로토콜형 = 동료셀 중앙값도\n"
         "그룹 하위 25% · 단위: C-rate C, 내부저항 mΩ · fc = fastcharge, ns = newstructure", ha="left", va="top",
         fontsize=8.2, color="#374151", linespacing=1.3)
ps.save(fig, "q1_short_cells_evidence")

# ════════════════════════════════════════════════════════════════
# 저장
# ════════════════════════════════════════════════════════════════
results["figures"] = ["q1_hist_150_2300", "q1_long_short_ratio", "q1_batch_compare", "q1_short_cells_evidence"]
with open(OUT / "q1_results.json", "w", encoding="utf-8") as fh:
    json.dump(jsonable(results), fh, ensure_ascii=False, indent=2)

# 콘솔 요약
pd.set_option("display.width", 250)
pd.set_option("display.max_columns", 40)
print(pd.DataFrame({b: desc[b]["all_labeled"] for b in BATCH_NAMES}).round(3))
print({b: round(desc[b]["km_median_incl_censored"], 1) for b in BATCH_NAMES})
print(pd.DataFrame(ratio).T[["n_life_test", "n_labeled", "labeled_long_gt1000", "labeled_short_lt500",
                             "labeled_lt550", "all_long_ratio", "all_short_ratio", "all_lt550_ratio"]])
print(pd.DataFrame(tests).T.round(4))
print(pd.DataFrame(proto_var).T.round(3))
print(range_check)
print(SHORT_CELLS[["cell_key", "policy", "cycle_life", "mates_lives", "life_over_mates"] +
                  [f"rp_{k}" for k, *_ in EVID]].round(2).to_string())
print(pd.DataFrame(evid_summary).T.round(3))
print(pd.DataFrame(b1_contrast).T)
print(SHORT_CELLS[["cell_key", "mates_median", "group_q25", "short_type"]])
print({k: round(v["ratio"], 3) for k, v in same_proto_ratio.items()})
print(pd.DataFrame(results["c_data_quality"]["batch3_paper_removed_and_upper_outliers"]).to_string())
print(q_base)
print(redund_b1, robust_z_b3)
print(short_q.to_string())
print(peer_ratio_summary)
print(qd_noise_b1)
print(pd.DataFrame(evid_split).T)
print(results["notion_claim_test"]["batch1_policy_mean_range"], shared_b1_b2fc, b2fc_only)
print(json.dumps(results["c_replicate_spread_batch1"], ensure_ascii=False, indent=1))
print(pd.DataFrame({bg: {k: (round(v.get("rho", np.nan), 2), v["n"], round(v.get("p", np.nan), 3), v["interpretable"]) for k, v in d.items()} for bg, d in spearman_bg.items()}).to_string())
