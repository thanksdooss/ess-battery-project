"""DAY 1 EDA — Question 3. ΔQ(V) 곡선 - 초기 사이클에서 차이가 보이는가?

노션 하위 항목
  (a) 사이클 100번 - 사이클 10번의 Q(V) 차이 계산
  (b) 장수명 셀 vs 단수명 셀의 ΔQ 형태 비교
  (c) 이를 구분할 수 있는 통계값으로 피쳐 추출
추가 확인
  - (d) 노션 Batch 3 메모 '충전 커브 시작 시점이 배치별로 상이 : Qdlin 변수를 단순 비교하면 왜곡 발생'
        + 예외 후보 Qcc_init(원시 Qdlin 2.0 V CC 끝점, 사이클 2–6 중앙값 = 절대 Qdlin 수준)의
          라벨 없는 배치 오프셋(B1 46셀 중앙값 기준, mAh)과 B1 기술 회귀 계수로 환산한 예측 배율
  - (e) 분류 옵션(초기 5 사이클)용 ΔQ_{5-4}(V)
그림
  - q3_a, q3_d 는 보고서(PDF, 약 16 cm 폭)용 2×2 레이아웃(큰 글씨·주석 최소화). q3_b, q3_c 는 상세(부록)용.

규칙
  - ΔQ_{a-b}(V) = data.delta_q(cell, a, b) = Qdlin[a-1] − Qdlin[b-1]  (1000 포인트 전압 축 Vdlin)
  - 대상 셀 = cell_table 의 labeled (수명 라벨 있음 & 중도절단 아님): B1 36 · B2 39 · B3 44
  - 장수명 > 1,000 / 단수명 < 500 (노션), 분류 라벨 = cycle_life >= 550 (노션)
  - 모델 학습 없음(DAY 1). 상관계수·분포 비교만 수행. 피처 선택 근거는 Batch 1 기준
    (B1 상관, B1 피처 간 상관, B1 에서 log10 var 통제 편상관, B1 구분력 AUC),
    Batch 2/3 는 일반화 위험 점검용으로만 인용(분포 범위 비교, B1 설명용 추세선 위치).
  - B1 추세선(log10 수명 ~ log10 var)은 설명용 일관성 점검이며 테스트 성능이 아님.
    B2/B3 잔차는 설계 결정의 근거로 쓰지 않는다.

실행: cd ess-battery-project && python src/eda/q3_delta_q.py
산출: reports/figures/q3_*.png, results/eda/q3_results.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import plot_style as ps  # noqa: E402  (matplotlib Agg + 한글 폰트 설정)
from data import (BATCH_NAMES, EOL_AH, LABEL_THRESHOLD, RANDOM_STATE,  # noqa: E402,F401
                  cell_table, clean_summary, delta_q, load_all, qdlin)

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
OUT_JSON = ROOT / "results" / "eda" / "q3_results.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

LONG_TH, SHORT_TH = 1000, 500          # 노션: 장수명(>1,000) / 단수명(<500)
BN = {"batch1": "Batch 1", "batch2": "Batch 2", "batch3": "Batch 3"}
FEATS = {  # 피처명 → 한글 표기
    "min": "min ΔQ",
    "mean": "mean ΔQ",
    "var": "var ΔQ",
    "log_var": "log₁₀ var ΔQ",
    "log_absmin": "log₁₀ |min ΔQ|",
    "skew": "skewness ΔQ",
    "kurt": "kurtosis ΔQ",
    "q_2V": "ΔQ(2.0 V)",
    "V_at_min": "min 위치 전압",
}
EARLY_FEATS = ["log_var", "min", "q_2V"]   # ΔQ_{5-4} 에서 히트맵에 함께 보일 피처


# ───────────────────────────── 데이터 준비 ─────────────────────────────
cells = load_all()
ct = cell_table(cells).set_index("cell_key")
V = np.asarray(cells[0]["Vdlin"], float)
assert all(np.allclose(c["Vdlin"], V) for c in cells), "배치마다 Vdlin 이 다름"
lab = [c for c in cells if ct.loc[c["cell_key"], "labeled"]]


def dq_features(dq: np.ndarray) -> dict:
    """ΔQ(V) 1000 포인트 곡선 → 요약 통계 (분산은 ddof=0, skew/kurt 는 scipy 기본: 편향 보정 없음, 첨도는 excess)."""
    return {
        "min": float(dq.min()), "mean": float(dq.mean()), "var": float(dq.var()),
        "log_var": float(np.log10(dq.var())), "log_absmin": float(np.log10(abs(dq.min()))),
        "skew": float(stats.skew(dq)), "kurt": float(stats.kurtosis(dq)),
        "q_2V": float(dq[-1]),                 # Vdlin 마지막 점 = 2.0 V
        "V_at_min": float(V[dq.argmin()]),
    }


DQ = np.array([delta_q(c, 100, 10) for c in lab])      # (119, 1000)
DQ54 = np.array([delta_q(c, 5, 4) for c in lab])
Q10 = np.array([qdlin(c, 10) for c in lab])
assert np.isfinite(DQ).all() and np.isfinite(DQ54).all() and np.isfinite(Q10).all()

rows = []
for c, d, e in zip(lab, DQ, DQ54):
    r = {"cell_key": c["cell_key"], "batch": c["batch"], "group": ct.loc[c["cell_key"], "group"],
         "cycle_life": float(c["cycle_life"])}
    r.update(dq_features(d))
    r.update({f"e_{k}": v for k, v in dq_features(e).items()})
    rows.append(r)
df = pd.DataFrame(rows)
df["log_life"] = np.log10(df["cycle_life"])
df["label_550"] = (df["cycle_life"] >= LABEL_THRESHOLD).astype(int)
bmask = {b: (df["batch"] == b).to_numpy() for b in BATCH_NAMES}

R: dict = {"meta": {
    "n_labeled": {b: int(m.sum()) for b, m in bmask.items()},
    "voltage_grid": {"n": int(V.size), "V_first": float(V[0]), "V_last": float(V[-1])},
    "definition": "ΔQ_{a-b}(V) = Qdlin[row a-1] − Qdlin[row b-1] (data.delta_q); var 는 np.var(ddof=0); "
                  "skew/kurt 는 scipy.stats.skew/kurtosis 기본값(Fisher excess, bias=True); q_2V = ΔQ 의 마지막 점(V=2.0 V)",
}}

# 공통: 수명 색상 (빨강=단수명 … 회색 … 파랑=장수명, 로그 스케일)
LIFE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "life", ["#991B1B", ps.SHORT, "#B8B8B8", ps.LONG, "#1E3A8A"])
LIFE_NORM = mcolors.LogNorm(vmin=350, vmax=2000)
LIFE_TICKS = [400, 500, 550, 700, 1000, 1500, 2000]


def life_ticks(ax_or_cbar, axis="y"):
    from matplotlib.ticker import FixedLocator, NullLocator
    if hasattr(ax_or_cbar, "ax") and not hasattr(ax_or_cbar, "plot"):   # colorbar
        cb = ax_or_cbar
        cb.set_ticks(LIFE_TICKS)
        cb.set_ticklabels([f"{t:,}" for t in LIFE_TICKS])
        cb.ax.yaxis.set_minor_locator(NullLocator())
        return
    ax = ax_or_cbar
    a = ax.yaxis if axis == "y" else ax.xaxis
    a.set_major_locator(FixedLocator([400, 550, 700, 1000, 1500, 2000]))
    a.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    a.set_minor_locator(NullLocator())


# ════════════════════ (a) ΔQ_{100-10}(V) 계산 및 배치별 곡선 ════════════════════
summary_a = {}
for b in BATCH_NAMES:
    g = df[bmask[b]]
    summary_a[b] = {
        "n": int(len(g)),
        "life_range": [float(g.cycle_life.min()), float(g.cycle_life.max())],
        "min_dq_median": float(g["min"].median()),
        "min_dq_IQR": [float(g["min"].quantile(.25)), float(g["min"].quantile(.75))],
        "V_at_min_median": float(g["V_at_min"].median()),
        "log_var_mean": float(g["log_var"].mean()), "log_var_std": float(g["log_var"].std()),
        "frac_cells_min_below_0": float((g["min"] < 0).mean()),
        "dq_at_3p3V_abs_max": float(np.abs(DQ[bmask[b]][:, np.argmin(abs(V - 3.3))]).max()),
    }
R["a_curves"] = summary_a

# PDF(약 16 cm 폭)용: 2×2, 큰 글씨, 주석 최소화(세부 수치는 본문이 담당)
PDF_RC = {"font.size": 11, "axes.titlesize": 12.5, "axes.labelsize": 11.5, "xtick.labelsize": 10.5,
          "ytick.labelsize": 10.5, "legend.fontsize": 10}
SHORT_BN = {"batch1": "Batch 1 (학습)", "batch2": "Batch 2 (테스트)", "batch3": "Batch 3 (추가 검증)"}
_fc_mask = (df.group == "fastcharge").to_numpy()
r_logvar = {b: float(stats.pearsonr(df.loc[bmask[b], "log_var"], df.loc[bmask[b], "log_life"])[0]) for b in BATCH_NAMES}
r_logvar["batch2_fastcharge"] = float(stats.pearsonr(df.loc[bmask["batch2"] & _fc_mask, "log_var"],
                                                     df.loc[bmask["batch2"] & _fc_mask, "log_life"])[0])
with plt.rc_context(PDF_RC):
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 8.0))
    center = np.sqrt(LIFE_NORM.vmin * LIFE_NORM.vmax)
    for k, (ax, b) in enumerate(zip(axs.flat[:3], BATCH_NAMES)):
        idx = np.where(bmask[b])[0]
        order = idx[np.argsort(np.abs(np.log(df.cycle_life.to_numpy()[idx]) - np.log(center)))]  # 극단 수명이 위에 그려지게
        for i in order:
            ax.plot(V, DQ[i], color=LIFE_CMAP(LIFE_NORM(df.cycle_life.iat[i])), lw=1.0, alpha=0.9)
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xlim(2.0, 3.5)
        ax.set_ylim(-0.112, 0.026)
        ax.set_xlabel("전압 (V)")
        ax.set_ylabel(r"$\Delta Q_{100-10}(V)$ (Ah)")
        ax.set_title(f"{'①②③'[k]} {SHORT_BN[b]}, n={summary_a[b]['n']}", color=ps.BATCH_COLOR[b])
        txt = f"r = {r_logvar[b]:+.2f}".replace("-", "−")
        if b == "batch2":   # 프로토콜 이봉(고속충전 단수명 vs newstructure 장수명)이 r 을 부풀림 → 고속충전만 값 병기
            txt += f"\n고속충전만 {r_logvar['batch2_fastcharge']:+.2f}".replace("-", "−")
        ax.text(0.97, 0.04, txt, transform=ax.transAxes, ha="right", va="bottom", fontsize=11,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#ccc", alpha=0.95))
    # 수명 색 막대: ① 패널 왼쪽 아래 빈 영역에 가로로
    cax = axs[0, 0].inset_axes([0.06, 0.09, 0.46, 0.05])
    sm = plt.cm.ScalarMappable(norm=LIFE_NORM, cmap=LIFE_CMAP)
    cb = fig.colorbar(sm, cax=cax, orientation="horizontal")
    cb.set_ticks([400, 550, 1000, 2000])
    cb.set_ticklabels(["400", "550", "1,000", "2,000"])
    cb.ax.xaxis.set_minor_locator(plt.NullLocator())
    cb.ax.tick_params(labelsize=9.5)
    cb.set_label("곡선 색 = 수명 (사이클)", fontsize=9.5, labelpad=2)
    cb.ax.xaxis.set_label_position("top")
    # ④ 주 피처 산점도 (B1 추세선은 설명용, B1 범위 안만 실선)
    ax = axs[1, 1]
    _lr = stats.linregress(df.loc[bmask["batch1"], "log_var"], df.loc[bmask["batch1"], "log_life"])
    for b in BATCH_NAMES:
        g = df[bmask[b]]
        lbl = f"{SHORT_BN[b]}  r = {r_logvar[b]:+.2f}".replace("-", "−")
        ax.scatter(g.log_var, g.cycle_life, s=30, color=ps.BATCH_COLOR[b], alpha=0.85, edgecolor="white", lw=0.6, label=lbl)
    _x1 = df.loc[bmask["batch1"], "log_var"]
    xs = np.linspace(_x1.min(), _x1.max(), 50)
    ax.axvspan(_x1.min(), _x1.max(), color=ps.BATCH_COLOR["batch1"], alpha=0.07, lw=0)
    ax.plot(xs, 10 ** (_lr.intercept + _lr.slope * xs), color=ps.BATCH_COLOR["batch1"], lw=2.4,
            label="B1 추세선(설명용), 음영 = B1 범위")
    ax.set_yscale("log")
    life_ticks(ax)
    ax.set_ylim(300, 2600)
    ax.set_xlabel(r"$\log_{10}\,\mathrm{var}(\Delta Q_{100-10}(V))$")
    ax.set_ylabel("수명 (사이클, 로그 축)")
    ax.set_title("④ 주 피처 log₁₀ var ΔQ₁₀₀₋₁₀ 와 수명")
    ax.legend(loc="lower left", fontsize=9.3, handletextpad=0.3, borderaxespad=0.2, labelspacing=0.3)
    fig.text(0.5, -0.01, "r = Pearson r(log₁₀ var ΔQ₁₀₀₋₁₀, log₁₀ 수명) · Batch 2 의 r 은 프로토콜 이봉(고속충전 단수명 / newstructure 장수명)이 부풀린 값",
             ha="center", fontsize=10.5, color="#444")
    fig.tight_layout(rect=(0, 0.015, 1, 0.965))
    fig.suptitle("(a) ΔQ₁₀₀₋₁₀(V): 수명이 짧을수록(빨강) 2.9~3.0 V 에서 곡선이 깊고, 그 깊이가 수명과 로그–로그 선형",
                 fontsize=13, fontweight="bold", y=0.995)
    ps.save(fig, "q3_a_dq_curves_by_batch")
R["a_curves"]["r_log_var_vs_log_life"] = r_logvar


# ════════════════════ (b) 장수명 vs 단수명 ΔQ 형태 비교 ════════════════════
def band(ax, X, color, label, lw=2.0):
    med = np.median(X, 0)
    lo, hi = np.percentile(X, [25, 75], axis=0)
    ax.fill_between(V, lo, hi, color=color, alpha=0.18, lw=0)
    ax.plot(V, med, color=color, lw=lw, label=label)
    return med


def shape_stats(X: np.ndarray) -> dict:
    mins = X.min(1)
    return {"n": int(len(X)), "min_median": float(np.median(mins)),
            "min_IQR": [float(np.percentile(mins, 25)), float(np.percentile(mins, 75))],
            "V_at_min_median": float(np.median(V[X.argmin(1)])),
            "log_var_median": float(np.median(np.log10(X.var(1)))),
            "q_2V_median": float(np.median(X[:, -1])),
            "band_IQR_width_at_curve_min": float(np.subtract(*np.percentile(X[:, np.median(X, 0).argmin()], [75, 25])))}


life = df.cycle_life.to_numpy()
is_long, is_short = life > LONG_TH, life < SHORT_TH
R["b_long_short"] = {"notion_threshold": {
    "long_gt_1000_by_batch": {b: int((is_long & bmask[b]).sum()) for b in BATCH_NAMES},
    "short_lt_500_by_batch": {b: int((is_short & bmask[b]).sum()) for b in BATCH_NAMES},
    "long": shape_stats(DQ[is_long]), "short": shape_stats(DQ[is_short]),
    "short_groups": df.loc[is_short, "group"].value_counts().to_dict(),
}}
# Batch 2 내부 (노션 기준 두 그룹이 모두 있는 유일한 배치)
m2l, m2s = is_long & bmask["batch2"], is_short & bmask["batch2"]
R["b_long_short"]["notion_threshold"]["within_batch2"] = {
    "long": shape_stats(DQ[m2l]), "short": shape_stats(DQ[m2s]),
    "long_groups": df.loc[m2l, "group"].value_counts().to_dict(),
    "short_groups": df.loc[m2s, "group"].value_counts().to_dict(),
    "note": "Batch 2 내부 노션 기준은 장수명 3개 = 전부 newstructure, 단수명 28개 = 전부 fastcharge → 프로토콜과 완전 교란"}
# Batch 1 내부: 노션 장수명(>1,000) 셀 5개 vs 나머지 31개 (Batch 1 에는 <500 셀이 0개)
m1l = is_long & bmask["batch1"]
m1r = bmask["batch1"] & ~is_long
R["b_long_short"]["notion_threshold"]["within_batch1_long_vs_rest"] = {
    "long_gt_1000": {**shape_stats(DQ[m1l]), "life_range": [float(life[m1l].min()), float(life[m1l].max())],
                     "cells": df.loc[m1l, "cell_key"].tolist()},
    "rest_le_1000": {**shape_stats(DQ[m1r]), "life_range": [float(life[m1r].min()), float(life[m1r].max())]},
}
_w1 = R["b_long_short"]["notion_threshold"]["within_batch1_long_vs_rest"]
_w1["ratio_absmin_rest_over_long"] = _w1["rest_le_1000"]["min_median"] / _w1["long_gt_1000"]["min_median"]

# 배치 내 수명 3분위 (Batch 1 에는 <500 셀이 없으므로)
tert = {}
for b in BATCH_NAMES:
    g = df[bmask[b]]
    q1, q2 = g.cycle_life.quantile([1 / 3, 2 / 3])
    lo_m = bmask[b] & (life <= q1)
    hi_m = bmask[b] & (life >= q2)
    lo_s, hi_s = shape_stats(DQ[lo_m]), shape_stats(DQ[hi_m])
    # 형태(정규화) 비교: ΔQ / |min ΔQ|
    nlo = np.median(DQ[lo_m] / np.abs(DQ[lo_m].min(1, keepdims=True)), 0)
    nhi = np.median(DQ[hi_m] / np.abs(DQ[hi_m].min(1, keepdims=True)), 0)
    tert[b] = {"cut_life": [float(q1), float(q2)],
               "bottom": {**lo_s, "life_range": [float(life[lo_m].min()), float(life[lo_m].max())]},
               "top": {**hi_s, "life_range": [float(life[hi_m].min()), float(life[hi_m].max())]},
               "ratio_absmin_bottom_over_top": lo_s["min_median"] / hi_s["min_median"],
               "normalized_shape_r": float(np.corrcoef(nlo, nhi)[0, 1]),
               "normalized_shape_max_abs_diff": float(np.abs(nlo - nhi).max()),
               "normalized_at_3p1V": {"bottom": float(nlo[np.argmin(abs(V - 3.1))]), "top": float(nhi[np.argmin(abs(V - 3.1))])},
               "normalized_at_2p0V": {"bottom": float(nlo[-1]), "top": float(nhi[-1])},
               "top_groups": df.loc[hi_m, "group"].value_counts().to_dict(),
               "bottom_groups": df.loc[lo_m, "group"].value_counts().to_dict()}
    # 퍼짐(셀 간 산포) 점검: 절대 퍼짐 vs 상대 퍼짐(깊이로 나눈 값) vs 로그 피처의 퍼짐
    spread = {}
    for key, mm in (("bottom", lo_m), ("top", hi_m)):
        mins = DQ[mm].min(1)
        spread[key] = {
            "abs_IQR_at_curve_min": tert[b][key]["band_IQR_width_at_curve_min"],
            "IQR_of_cell_min": float(np.subtract(*np.percentile(mins, [75, 25]))),
            "rel_spread_IQR_over_abs_median_min": float(np.subtract(*np.percentile(mins, [75, 25])) / abs(np.median(mins))),
            "log_var_sd": float(np.log10(DQ[mm].var(1)).std(ddof=1)),
        }
    spread["abs_IQR_ratio_bottom_over_top"] = spread["bottom"]["abs_IQR_at_curve_min"] / spread["top"]["abs_IQR_at_curve_min"]
    tert[b]["spread"] = spread
    tert[b]["_masks"] = (lo_m, hi_m)
R["b_long_short"]["within_batch_tertiles"] = {b: {k: v for k, v in t.items() if k != "_masks"} for b, t in tert.items()}
# 수명 4분위 (Batch 1) — 참고
qs = df.loc[bmask["batch1"], "cycle_life"].quantile([.25, .5, .75]).to_numpy()
q_lab = np.digitize(life, qs)
R["b_long_short"]["batch1_quartiles"] = {
    f"Q{k + 1}": {"life_range": [float(life[bmask['batch1'] & (q_lab == k)].min()), float(life[bmask['batch1'] & (q_lab == k)].max())],
                  **shape_stats(DQ[bmask["batch1"] & (q_lab == k)])} for k in range(4)}

fig, axs = plt.subplots(2, 2, figsize=(13.5, 9.6))
# (A) 노션 기준, 전 배치 통합
ax = axs[0, 0]
nl = R["b_long_short"]["notion_threshold"]["long_gt_1000_by_batch"]
band(ax, DQ[is_short], ps.SHORT, f"단수명 <500 (n={is_short.sum()}, 전부 Batch 2 고속충전)")
band(ax, DQ[is_long], ps.LONG, f"장수명 >1,000 (n={is_long.sum()}: B1 {nl['batch1']} · B2 {nl['batch2']}(전부 newstructure) · B3 {nl['batch3']})")
ax.set_title("① 노션 기준(>1,000 vs <500), 3개 배치 통합 — 배치·프로토콜과 교란", fontsize=11.5)
ax.legend(loc="lower left", fontsize=9)
# (B) Batch 1 내부 3분위
ax = axs[0, 1]
t1 = tert["batch1"]
lo_m, hi_m = t1["_masks"]
band(ax, DQ[lo_m], ps.SHORT, f"하위 1/3 수명 {t1['bottom']['life_range'][0]:.0f}~{t1['bottom']['life_range'][1]:.0f} (n={lo_m.sum()})")
band(ax, DQ[hi_m], ps.LONG, f"상위 1/3 수명 {t1['top']['life_range'][0]:.0f}~{t1['top']['life_range'][1]:,.0f} (n={hi_m.sum()})")
ax.set_title("② Batch 1(학습) 내부 수명 3분위 — <500 셀이 없어 대체")
ax.legend(loc="lower left", fontsize=9)
for ax in (axs[0, 0], axs[0, 1]):
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlim(2.0, 3.5)
    ax.set_ylim(-0.09, 0.006)
    ax.set_xlabel("전압 (V)")
    ax.set_ylabel(r"$\Delta Q_{100-10}(V)$ (Ah) — 중앙값 & IQR")
# (C) 형태 비교: |min| 으로 정규화
ax = axs[1, 0]
for b, ls in zip(BATCH_NAMES, ("-", "--", ":")):
    lo_m, hi_m = tert[b]["_masks"]
    for m, col in ((lo_m, ps.SHORT), (hi_m, ps.LONG)):
        ax.plot(V, np.median(DQ[m] / np.abs(DQ[m].min(1, keepdims=True)), 0), color=col, ls=ls, lw=1.8)
ax.axhline(0, color="#555", lw=0.8)
ax.set_xlim(2.0, 3.5)
ax.set_xlabel("전압 (V)")
ax.set_ylabel("ΔQ / |min ΔQ|  (셀별 정규화, 중앙값)")
ax.set_title("③ |min| 으로 정규화: 2.0~2.95 V 모양은 겹치고, 차이는 3.0~3.15 V 회복 구간", fontsize=11.5)
h = [Line2D([], [], color=ps.SHORT, lw=2, label="하위 1/3 수명"), Line2D([], [], color=ps.LONG, lw=2, label="상위 1/3 수명")]
h += [Line2D([], [], color="#444", ls=ls, lw=1.6, label=BN[b]) for b, ls in zip(BATCH_NAMES, ("-", "--", ":"))]
ax.legend(handles=h, loc="lower left", fontsize=9, ncol=2)
txt = "하위 vs 상위 1/3 정규화 곡선 상관\n" + "\n".join(
    f"{BN[b]}: r = {tert[b]['normalized_shape_r']:.3f}" for b in BATCH_NAMES)
# 2.45~2.95 V 의 y ≈ 0 ~ −0.2 영역은 곡선이 지나가지 않는 빈 공간 → 3.0~3.15 V 곡선을 가리지 않음
ax.text(0.31, 0.95, txt, transform=ax.transAxes, ha="left", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc"))
# (D) 크기·위치 요약
ax = axs[1, 1]
xpos = np.arange(3)
for k, (key, col, lbl) in enumerate((("bottom", ps.SHORT, "하위 1/3 수명"), ("top", ps.LONG, "상위 1/3 수명"))):
    med = [abs(tert[b][key]["min_median"]) for b in BATCH_NAMES]
    iqr = np.array([[abs(tert[b][key]["min_IQR"][1]), abs(tert[b][key]["min_IQR"][0])] for b in BATCH_NAMES])
    x = xpos + (k - 0.5) * 0.28
    ax.errorbar(x, med, yerr=[np.array(med) - iqr[:, 0], iqr[:, 1] - np.array(med)], fmt="o", ms=9, color=col,
                capsize=4, lw=1.6, label=lbl)
    for xi, b, mv in zip(x, BATCH_NAMES, med):
        ax.annotate(f"{mv:.3f}\n@{tert[b][key]['V_at_min_median']:.2f}V", (xi, mv), xytext=(8 if k else -8, 0),
                    textcoords="offset points", ha="left" if k else "right", va="center", fontsize=8.6, color="#333")
for i, b in enumerate(BATCH_NAMES):
    ax.text(i, 0.083, f"×{tert[b]['ratio_absmin_bottom_over_top']:.1f}", ha="center", fontsize=10.5, fontweight="bold", color="#333")
ax.set_xticks(xpos, [f"{BN[b]}\n(3분위 경계 {tert[b]['cut_life'][0]:,.0f} / {tert[b]['cut_life'][1]:,.0f})"
                      + ("\n상위 1/3 = newstructure 9 + 고속충전 4" if b == "batch2" else "") for b in BATCH_NAMES], fontsize=9.2)
ax.set_xlim(-0.6, 2.6)
ax.set_ylim(0, 0.092)
ax.set_ylabel("|min ΔQ| 중앙값 (Ah), 막대 = IQR")
ax.set_title("④ 배치별 하위/상위 1/3 의 |min ΔQ| (위 숫자 = 하위÷상위 배율)")
ax.legend(loc="upper right", fontsize=9, bbox_to_anchor=(1.0, 0.9))
fig.suptitle("(b) 장수명 vs 단수명 ΔQ₁₀₀₋₁₀(V): 차이는 주로 '깊이(크기)'이고, 모양·최솟값 위치(≈2.9~3.0 V)는 비슷하다", fontsize=13.5, fontweight="bold", y=1.0)
fig.tight_layout()
ps.save(fig, "q3_b_long_vs_short_shape")


# ════════════════════ (c) 통계값 피쳐 추출 & 상관 ════════════════════
def corr_block(g: pd.DataFrame, cols: list[str]) -> dict:
    out = {}
    for f in cols:
        pr, pp = stats.pearsonr(g[f], g["log_life"])
        sr, sp = stats.spearmanr(g[f], g["log_life"])
        out[f] = {"pearson": float(pr), "pearson_p": float(pp), "spearman": float(sr), "spearman_p": float(sp), "n": int(len(g))}
    return out


SETS = {"batch1": df[bmask["batch1"]], "batch2": df[bmask["batch2"]],
        "batch2_fastcharge": df[bmask["batch2"] & (df.group == "fastcharge").to_numpy()],
        "batch3": df[bmask["batch3"]], "pooled": df}
feat_cols = list(FEATS) + [f"e_{k}" for k in FEATS]
CORR = {k: corr_block(g, feat_cols) for k, g in SETS.items()}
R["c_features"] = {"corr_with_log10_cycle_life": CORR}
# Batch 1 피처 간 상관 (중복성)
b1 = SETS["batch1"]
R["c_features"]["batch1_intercorr_pearson"] = b1[list(FEATS)].corr().round(3).to_dict()
best = max(FEATS, key=lambda f: abs(CORR["batch1"][f]["pearson"]) + abs(CORR["batch1"][f]["spearman"]))
R["c_features"]["best_by_batch1"] = best
R["c_features"]["batch_feature_summary"] = {
    b: {f: {"median": float(SETS[b][f].median()), "q25": float(SETS[b][f].quantile(.25)), "q75": float(SETS[b][f].quantile(.75))}
        for f in feat_cols} for b in BATCH_NAMES}
# Batch 1 단순 추세선(설명용; 예측 모델 아님) — B2/B3 가 같은 관계 위에 있는지 일관성 점검
sl, ic, r1, _, _ = stats.linregress(b1["log_var"], b1["log_life"])
resid = df["log_life"] - (ic + sl * df["log_var"])
R["c_features"]["batch1_trend_loglife_vs_logvar"] = {
    "slope": float(sl), "intercept": float(ic), "r": float(r1),
    "median_offset_log10_by_set": {
        "batch1": float(resid[bmask["batch1"]].median()), "batch2": float(resid[bmask["batch2"]].median()),
        "batch2_fastcharge": float(resid[bmask["batch2"] & (df.group == "fastcharge").to_numpy()].median()),
        "batch2_newstructure": float(resid[bmask["batch2"] & (df.group == "newstructure").to_numpy()].median()),
        "batch3": float(resid[bmask["batch3"]].median())},
    "frac_below_line": {b: float((resid[bmask[b]] < 0).mean()) for b in BATCH_NAMES},
    "batch1_log_var_range": [float(b1.log_var.min()), float(b1.log_var.max())],
    "frac_outside_batch1_log_var_range": {b: float(((df.log_var[bmask[b]] > b1.log_var.max()) | (df.log_var[bmask[b]] < b1.log_var.min())).mean()) for b in BATCH_NAMES},
}
_tr = R["c_features"]["batch1_trend_loglife_vs_logvar"]
_lr1 = stats.linregress(b1["log_var"], b1["log_life"])
_tr["slope_stderr"] = float(_lr1.stderr)
_tr["r2_batch1"] = float(_lr1.rvalue ** 2)
_tr["note"] = "설명용 일관성 점검(B1 에만 적합). B2/B3 잔차는 테스트 성능이 아니며 설계 결정의 근거로 쓰지 않음."
_sets_tr = {"batch1": bmask["batch1"], "batch2": bmask["batch2"],
            "batch2_fastcharge": bmask["batch2"] & (df.group == "fastcharge").to_numpy(),
            "batch2_newstructure": bmask["batch2"] & (df.group == "newstructure").to_numpy(),
            "batch3": bmask["batch3"]}
_tr["resid_rmse_log10_by_set"] = {k: float(np.sqrt(np.mean(resid[m] ** 2))) for k, m in _sets_tr.items()}
_lr3 = stats.linregress(SETS["batch3"]["log_var"], SETS["batch3"]["log_life"])
_tr["batch3_own_fit"] = {"slope": float(_lr3.slope), "slope_stderr": float(_lr3.stderr), "intercept": float(_lr3.intercept)}
# B1 안에서 |잔차| 가 분산 크기에 따라 달라지는가 (Y 쪽 이분산 점검)
_sr = stats.spearmanr(np.abs(resid[bmask["batch1"]]), b1["log_var"])
_tr["batch1_absresid_vs_logvar_spearman"] = {"rho": float(_sr.statistic), "p": float(_sr.pvalue)}
# 배치별 log10 var 분포와 B1 범위 밖 셀 수 (분포 비교, 일반화 위험 점검)
_lo1, _hi1 = b1.log_var.min(), b1.log_var.max()
R["c_features"]["log_var_distribution"] = {
    k: {"n": int(m.sum()), "median": float(df.log_var[m].median()), "mean": float(df.log_var[m].mean()),
        "q25": float(df.log_var[m].quantile(.25)), "q75": float(df.log_var[m].quantile(.75)),
        "n_above_b1_max": int((df.log_var[m] > _hi1).sum()), "n_below_b1_min": int((df.log_var[m] < _lo1).sum()),
        "n_outside_b1_range": int(((df.log_var[m] > _hi1) | (df.log_var[m] < _lo1)).sum())}
    for k, m in _sets_tr.items()}
_g_ns = df[_sets_tr["batch2_newstructure"]]
R["c_features"]["batch2_newstructure_only_log_var"] = {
    "pearson": float(stats.pearsonr(_g_ns.log_var, _g_ns.log_life)[0]), "pearson_p": float(stats.pearsonr(_g_ns.log_var, _g_ns.log_life)[1]),
    "spearman": float(stats.spearmanr(_g_ns.log_var, _g_ns.log_life)[0]), "n": int(len(_g_ns))}


def partial_corr_b1(f: str, ctrl: str = "log_var") -> dict:
    """Batch 1 에서 ctrl 을 선형 통제한 f 와 log10 수명의 편상관 (잔차끼리 Pearson, 자유도 n−3)."""
    x, y, z = b1[f].to_numpy(), b1["log_life"].to_numpy(), b1[ctrl].to_numpy()
    rx = x - np.polyval(np.polyfit(z, x, 1), z)
    ry = y - np.polyval(np.polyfit(z, y, 1), z)
    r = float(np.corrcoef(rx, ry)[0, 1])
    n = len(x)
    t = r * np.sqrt((n - 3) / (1 - r ** 2))
    return {"partial_r": r, "p": float(2 * stats.t.sf(abs(t), n - 3)), "n": int(n)}


R["c_features"]["batch1_partial_corr_given_log_var"] = {
    f: partial_corr_b1(f) for f in ("skew", "kurt", "q_2V", "min", "mean", "log_absmin", "V_at_min", "e_log_var")}

# ── Batch 1: log10 var 의 신호가 '프로토콜 간' 차이인가 '같은 프로토콜 복제셀 간' 차이인가 (Q5-(d) 와 같은 정의) ──
_b1p = b1.assign(policy=ct.loc[b1.cell_key, "policy"].to_numpy())
_pm = _b1p.groupby("policy")[["log_var", "log_life"]].mean()
_rep = _b1p[_b1p.groupby("policy")["log_life"].transform("size") == 2].copy()
for _c in ("log_var", "log_life"):
    _rep[_c + "_w"] = _rep[_c] - _rep.groupby("policy")[_c].transform("mean")
_agree = sum(int(np.sign(g.log_var.iat[0] - g.log_var.iat[1]) == -np.sign(g.log_life.iat[0] - g.log_life.iat[1]))
             for _, g in _rep.groupby("policy"))
R["c_features"]["batch1_between_vs_within_policy_log_var"] = {
    "n_policies": int(len(_pm)), "r_policy_means": float(np.corrcoef(_pm.log_var, _pm.log_life)[0, 1]),
    "n_replicate_pairs": int(_rep.policy.nunique()),
    "r_within_policy_deviations": float(np.corrcoef(_rep.log_var_w, _rep.log_life_w)[0, 1]),
    "pairs_expected_direction": int(_agree),
    "note": "정책 평균끼리의 상관 vs 같은 정책 2셀(복제쌍) 안의 편차끼리 상관. 복제셀 간 편차를 설명하는 후보(Qcc_init)는 Q5-(d)."}


# ── 구분력: 주 피처 log10 var ΔQ_{100-10} 가 장/단수명을 얼마나 '구분'하는가 ──
def auc_gt(a, b_) -> float:
    """P(a 그룹 값 > b 그룹 값) = Mann-Whitney U / (na·nb). 0.5 = 구분 불가, 1 = 완전 분리."""
    a, b_ = np.asarray(a), np.asarray(b_)
    return float(stats.mannwhitneyu(a, b_).statistic / (len(a) * len(b_)))


_t1lo, _t1hi = tert["batch1"]["_masks"]
_b2fc = _sets_tr["batch2_fastcharge"]
disc = {
    "definition": "AUC = P(수명이 짧은 쪽 셀의 log10 var ΔQ100-10 > 긴 쪽 셀의 값)",
    "batch1_notion_long_gt1000_vs_rest": {"auc": auc_gt(df.log_var[m1r], df.log_var[m1l]), "n_short_side": int(m1r.sum()), "n_long_side": int(m1l.sum())},
    "batch1_bottom_vs_top_tertile": {"auc": auc_gt(df.log_var[_t1lo], df.log_var[_t1hi]), "n": [int(_t1lo.sum()), int(_t1hi.sum())]},
    "pooled_notion_short_lt500_vs_long_gt1000": {"auc": auc_gt(df.log_var[is_short], df.log_var[is_long]), "n": [int(is_short.sum()), int(is_long.sum())],
                                                 "note": "단수명 28 = 전부 B2 고속충전, 장수명 31 = B3 23·B1 5·B2 newstructure 3 → 배치·프로토콜과 교란"},
    "batch2_notion_short_vs_long": {"auc": auc_gt(df.log_var[m2s], df.log_var[m2l]), "n": [int(m2s.sum()), int(m2l.sum())],
                                    "note": "B2 내부 노션 기준 = 프로토콜 그룹과 동일(교란)"},
    "consistency_within_batch_tertile_auc": {b: auc_gt(df.log_var[tert[b]["_masks"][0]], df.log_var[tert[b]["_masks"][1]]) for b in ("batch2", "batch3")},
    "batch2_fastcharge_tertile_auc": None,
}
_fc_life = life[_b2fc]
_fq1, _fq2 = np.quantile(_fc_life, [1 / 3, 2 / 3])
disc["batch2_fastcharge_tertile_auc"] = auc_gt(df.log_var[_b2fc & (life <= _fq1)], df.log_var[_b2fc & (life >= _fq2)])
_r1 = b1.log_var.rank(ascending=False)
disc["batch1_single_lt550_cell"] = {"cell_key": b1.loc[b1.cycle_life < LABEL_THRESHOLD, "cell_key"].iloc[0],
                                    "cycle_life": float(b1.loc[b1.cycle_life < LABEL_THRESHOLD, "cycle_life"].iloc[0]),
                                    "rank_log_var_desc": int(_r1[b1.cycle_life < LABEL_THRESHOLD].iloc[0]),
                                    "rank1_cell": b1.loc[_r1 == 1, ["cell_key", "cycle_life"]].iloc[0].to_dict()}
R["c_features"]["discrimination_log_var"] = disc


# ── 초기 5 사이클 변형 ΔQ_{5-4}(V) ──
def auc_short_higher(g: pd.DataFrame, f: str) -> float | None:
    """P(단수명(<550) 셀의 피처 > 장수명 셀의 피처) = Mann-Whitney U / (n0·n1). 0.5 = 구분 불가."""
    s, l_ = g.loc[g.label_550 == 0, f], g.loc[g.label_550 == 1, f]
    if len(s) == 0 or len(l_) == 0:
        return None
    return float(stats.mannwhitneyu(s, l_).statistic / (len(s) * len(l_)))


early = {"magnitude": {}, "label_550": {}}
for b in BATCH_NAMES:
    m = bmask[b]
    early["magnitude"][b] = {
        "std_dq54_mean": float(DQ54[m].std(1).mean()), "std_dq100_10_mean": float(DQ[m].std(1).mean()),
        "ratio_std": float(DQ[m].std(1).mean() / DQ54[m].std(1).mean()),
        "min_dq54_median": float(np.median(DQ54[m].min(1))), "max_dq54_median": float(np.median(DQ54[m].max(1))),
        "e_log_var_median": float(np.median(df.loc[m, "e_log_var"])),
    }
for k, g in SETS.items():
    n0 = int((g.label_550 == 0).sum())
    early["label_550"][k] = {"n_short_lt550": n0, "n_long_ge550": int((g.label_550 == 1).sum()),
                             "auc_short_higher": {f: auc_short_higher(g, f) for f in ("e_log_var", "e_min", "e_q_2V", "log_var")}}
    if n0 == 1:   # 단수명 1개뿐 → AUC 대신 순위
        early["label_550"][k]["rank_of_single_short_cell(1=largest)"] = {
            f: int(g[f].rank(ascending=False)[g.label_550 == 0].iloc[0]) for f in ("e_log_var", "log_var")}
        early["label_550"][k]["single_short_cell"] = g.loc[g.label_550 == 0, ["cell_key", "cycle_life"]].iloc[0].to_dict()
early["magnitude"]["pooled"] = {"max_dq54_median": float(np.median(DQ54.max(1))), "min_dq54_median": float(np.median(DQ54.min(1)))}
early["batch2_label_equals_group"] =bool((SETS["batch2"].label_550 == (SETS["batch2"].group == "newstructure").astype(int)).all())
early["batch2_e_log_var_median_by_group"] = SETS["batch2"].groupby("group")["e_log_var"].median().to_dict()
R["early_dq_5_4"] = early

# ── 그림 (c) ──
fig = plt.figure(figsize=(15, 12.2))
gs = fig.add_gridspec(2, 2, height_ratios=[1.05, 1], hspace=0.40, wspace=0.22)
row_keys = list(FEATS) + [f"e_{k}" for k in EARLY_FEATS]
row_lbl = [f"{FEATS[k]}" for k in FEATS] + [f"[5−4] {FEATS[k]}" for k in EARLY_FEATS]
col_keys = ["batch1", "batch2", "batch2_fastcharge", "batch3", "pooled"]
col_lbl = ["Batch 1\n(학습, n=36)", "Batch 2\n(n=39)", "B2 고속충전만\n(n=30)", "Batch 3\n(n=44)", "전체 통합\n(n=119)"]
for j, (meth, ttl) in enumerate((("pearson", "Pearson r"), ("spearman", "Spearman ρ"))):
    ax = fig.add_subplot(gs[0, j])
    M = np.array([[CORR[c][r][meth] for c in col_keys] for r in row_keys])
    im = ax.imshow(M, cmap="RdBu_r", vmin=-1, vmax=1, aspect="auto")
    for (i, k), v in np.ndenumerate(M):
        p = CORR[col_keys[k]][row_keys[i]][f"{meth}_p"]
        ax.text(k, i, f"{v:+.2f}" + ("" if p < 0.05 else "ⁿˢ"), ha="center", va="center", fontsize=9.3,
                color="white" if abs(v) > 0.6 else "#222", fontweight="bold" if row_keys[i] == best else None)
    ax.set_xticks(range(len(col_keys)), col_lbl, fontsize=9)
    ax.set_yticks(range(len(row_keys)), row_lbl if j == 0 else [""] * len(row_keys), fontsize=9.6)
    ax.axhline(len(FEATS) - 0.5, color="#111", lw=1.6)
    ax.axvline(0.5, color="#111", lw=1.2)
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title(f"log₁₀ 수명과의 {ttl}", fontsize=12)
    ax.tick_params(length=0)
cax = fig.add_axes([0.915, 0.555, 0.012, 0.31])
fig.colorbar(im, cax=cax).set_label("상관계수")
fig.text(0.5, 0.468, "위 9행 = ΔQ₁₀₀₋₁₀, 아래 3행 = ΔQ₅₋₄ (초기 5 사이클)  ·  ⁿˢ = p ≥ 0.05  ·  굵은 행 = Batch 1 기준 최선 피처",
         ha="center", fontsize=9.5, color="#444")

for j, (fx, xl, ttl) in enumerate((("log_var", r"$\log_{10}\,\mathrm{var}(\Delta Q_{100-10}(V))$", "⑤ 주 피처: log₁₀ var ΔQ₁₀₀₋₁₀"),
                                    ("e_log_var", r"$\log_{10}\,\mathrm{var}(\Delta Q_{5-4}(V))$",
                                     "⑥ 초기 5 사이클 변형: log₁₀ var ΔQ₅₋₄ — B1 에서만 약한 음의 추세"))):
    ax = fig.add_subplot(gs[1, j])
    for b in BATCH_NAMES:
        g = df[bmask[b]]
        r_b, p_b = CORR[b][fx]["pearson"], CORR[b][fx]["pearson_p"]
        lbl = f"{ps.BATCH_LABEL[b]}  r={r_b:+.2f}" + ("" if p_b < 0.05 else " (ns)")
        if b == "batch2" and fx == "log_var":
            lbl += f"\n    ↳ 프로토콜 이봉 포함, 고속충전만 r={CORR['batch2_fastcharge'][fx]['pearson']:+.2f}"
        ax.scatter(g[fx], g.cycle_life, s=42, color=ps.BATCH_COLOR[b], alpha=0.85, edgecolor="white", lw=0.7, label=lbl)
    if fx == "log_var":
        xs = np.linspace(df[fx].min() - 0.05, df[fx].max() + 0.05, 200)
        xin = (xs >= b1[fx].min()) & (xs <= b1[fx].max())
        ax.plot(xs[xin], 10 ** (ic + sl * xs[xin]), color=ps.BATCH_COLOR["batch1"], lw=2.2,
                label=f"Batch 1 추세선 (설명용, B1 범위 {b1[fx].min():.2f} ~ {b1[fx].max():.2f})".replace("-", "−"))
        ax.plot(xs[xs < b1[fx].min()], 10 ** (ic + sl * xs[xs < b1[fx].min()]), color=ps.BATCH_COLOR["batch1"], lw=1.2, ls="--", alpha=0.6,
                label="  같은 선의 B1 범위 밖 외삽 (점선)")
        ax.plot(xs[xs > b1[fx].max()], 10 ** (ic + sl * xs[xs > b1[fx].max()]), color=ps.BATCH_COLOR["batch1"], lw=1.2, ls="--", alpha=0.6)
        ax.axvspan(b1[fx].min(), b1[fx].max(), color=ps.BATCH_COLOR["batch1"], alpha=0.05, lw=0)
        dsc = R["c_features"]["discrimination_log_var"]
        ax.text(0.015, 0.025,
                "구분력 AUC (짧은 쪽 피처 > 긴 쪽 확률)\n"
                f"· B1 노션 장수명 >1,000 (n={dsc['batch1_notion_long_gt1000_vs_rest']['n_long_side']}) vs 나머지: "
                f"{dsc['batch1_notion_long_gt1000_vs_rest']['auc']:.2f}\n"
                f"· B1 수명 하위 1/3 vs 상위 1/3: {dsc['batch1_bottom_vs_top_tertile']['auc']:.2f}\n"
                f"· B1 유일한 <550 셀 순위: {dsc['batch1_single_lt550_cell']['rank_log_var_desc']}/36\n"
                f"· 통합 노션 <500 vs >1,000: {dsc['pooled_notion_short_lt500_vs_long_gt1000']['auc']:.2f}"
                " (배치 교란)",
                transform=ax.transAxes, ha="left", va="bottom", fontsize=8.4,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="#ccc", alpha=0.95))
    ax.axhline(LABEL_THRESHOLD, color="#666", ls=":", lw=1.2)
    ax.text(ax.get_xlim()[0] if False else 0.99, LABEL_THRESHOLD, "550 (분류 기준)", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=8.6, color="#555")
    ax.set_yscale("log")
    life_ticks(ax)
    ax.set_ylim(330, 3300)     # 위쪽 여백 = 범례 자리 (데이터 최대 1,935)
    ax.set_xlabel(xl)
    ax.set_ylabel("수명 (사이클, 로그 축)")
    ax.set_title(ttl, fontsize=11.5)
    ax.legend(loc="upper right", fontsize=8.3)
fig.suptitle("(c) ΔQ(V) 통계 피처와 log₁₀ 수명의 상관: 분산·최솟값 계열이 가장 강하고, ΔQ₅₋₄ 는 B1 에서만 약한 추세",
             fontsize=13.5, fontweight="bold", y=0.955)
ps.save(fig, "q3_c_feature_correlation")


# ════════════════════ Batch 3 메모: Qdlin 단순 비교 왜곡 ════════════════════
labb = df["batch"].to_numpy()
mean_q10 = {b: Q10[bmask[b]].mean(0) for b in BATCH_NAMES}
diff = {b: mean_q10[b] - mean_q10["batch1"] for b in ("batch2", "batch3")}


def v_at_q(q, target=0.5):
    return float(V[np.argmax(q > target)])


raw_info = []
for c in lab:
    r = c["raw"].get(10)                    # raw 키 = 행 인덱스 → 10 = cycle 11 (cycle 10 원시 시계열은 저장 안 됨)
    s = c["summary"]
    raw_info.append({"cell_key": c["cell_key"], "batch": c["batch"], "group": ct.loc[c["cell_key"], "group"],
                     "V0": float(r.V.iloc[0]), "I0": float(r.I.iloc[0]),
                     "cv_hold_Ah_cycle10": float(s.QD.iloc[9] - qdlin(c, 10)[-1]),
                     "Tavg_cycle10": float(clean_summary(s).Tavg.iloc[9]),
                     "IR_cycle10": float(clean_summary(s).IR.iloc[9])})
raw_df = pd.DataFrame(raw_info)

loglife_all = df["log_life"].to_numpy()


def corr_profile(X, mask):
    y = loglife_all[mask]
    Xm = X[mask]
    Xc = Xm - Xm.mean(0)
    yc = y - y.mean()
    den = np.sqrt((Xc ** 2).sum(0) * (yc ** 2).sum())
    with np.errstate(invalid="ignore", divide="ignore"):
        return (Xc * yc[:, None]).sum(0) / den


valid_V = V <= 3.25     # 3.3 V 이상은 Q≈0 이라 상관이 의미 없음
prof = {}
for name, X in (("Q10", Q10), ("DQ", DQ)):
    for k, m in (("batch1", bmask["batch1"]), ("batch2", bmask["batch2"]), ("batch3", bmask["batch3"]), ("pooled", np.ones(len(df), bool))):
        prof[(name, k)] = corr_profile(X, m)
ref_signal = {b: float(np.median(np.abs(DQ[bmask[b]].min(1)))) for b in BATCH_NAMES}
R["batch3_note_qdlin"] = {
    "Q10_at_3p5V_mean": {b: float(mean_q10[b][0]) for b in BATCH_NAMES},
    "Q10_at_2p0V_mean": {b: float(mean_q10[b][-1]) for b in BATCH_NAMES},
    "V_at_Q10_eq_0p5Ah_mean": {b: float(np.mean([v_at_q(q) for q in Q10[bmask[b]]])) for b in BATCH_NAMES},
    "batchmean_diff_vs_B1_maxabs": {b: {"value_Ah": float(d[np.abs(d).argmax()]), "at_V": float(V[np.abs(d).argmax()]),
                                        "at_2p0V": float(d[-1])} for b, d in diff.items()},
    "within_batch_std_Q10_at_3p1V": {b: float(Q10[bmask[b], np.argmin(abs(V - 3.1))].std()) for b in BATCH_NAMES},
    "median_abs_min_DQ": ref_signal,
    "max_abs_corr_V_le_3p25": {f"{n}_{k}": float(np.nanmax(np.abs(p[valid_V]))) for (n, k), p in prof.items()},
    "max_abs_corr_V_le_3p25_at_V": {f"{n}_{k}": float(V[valid_V][np.nanargmax(np.abs(p[valid_V]))]) for (n, k), p in prof.items()},
    "max_abs_corr_V_le_3p25_signed": {f"{n}_{k}": float(p[valid_V][np.nanargmax(np.abs(p[valid_V]))]) for (n, k), p in prof.items()},
    "corr_Q10_at_2p0V_vs_loglife": {k: float(prof[("Q10", k)][-1]) for k in ("batch1", "batch2", "batch3", "pooled")},
    "raw_cycle11_charge_start": raw_df.groupby("batch")[["V0", "I0"]].agg(["mean", "min", "max"]).round(4).to_dict(),
    "cv_hold_Ah_cycle10_mean": raw_df.groupby("batch")["cv_hold_Ah_cycle10"].mean().to_dict(),
    "by_batch_group": raw_df.groupby(["batch", "group"])[["V0", "I0", "cv_hold_Ah_cycle10"]].mean().round(5)
                            .reset_index().to_dict(orient="records"),
    "Tavg_cycle10_mean": raw_df.groupby("batch")["Tavg_cycle10"].mean().to_dict(),
    "IR_cycle10_mean": raw_df.groupby("batch")["IR_cycle10"].mean().to_dict(),
}
# 오프셋의 기원: 3.0~3.2 V 에서는 '전압 방향 이동 × 그 지점의 곡선 기울기', 2.0 V 에서는 총 CC 방전 용량 차이
_slope_b1 = np.gradient(mean_q10["batch1"], V)     # dQ/dV (V 가 감소하는 축이라 음수)
_shift = {}
for qlev in (0.1, 0.2, 0.3, 0.5, 0.7, 0.8, 0.9, 0.95):
    vv = {b: float(np.interp(qlev, mean_q10[b], V)) for b in BATCH_NAMES}   # 평균 곡선이 Q=qlev 가 되는 전압
    _shift[f"Q={qlev}"] = {"B3_minus_B1_mV": (vv["batch3"] - vv["batch1"]) * 1e3, "B2_minus_B1_mV": (vv["batch2"] - vv["batch1"]) * 1e3}
_jd = int(np.abs(diff["batch3"]).argmax())
R["batch3_note_qdlin"]["offset_mechanism"] = {
    "horizontal_shift_by_Q_level": _shift,
    "abs_slope_B1_mean_Ah_per_V": {f"{v:.2f}V": float(abs(_slope_b1[np.argmin(abs(V - v))])) for v in (3.05, 3.10, 3.15)},
    "at_max_diff": {"V": float(V[_jd]), "B1_mean_Q": float(mean_q10["batch1"][_jd]), "abs_slope": float(abs(_slope_b1[_jd])),
                    "shift_mV_at_that_Q": float((np.interp(mean_q10["batch1"][_jd], mean_q10["batch3"], V) - V[_jd]) * 1e3)},
    "note": "3.15 V 부근 +0.027 Ah ≈ 약 4 mV 이동 × 기울기 6.6 Ah/V. 2.0 V 의 −0.015 Ah 는 전압 이동이 아니라 총 CC 방전 용량 차이(곡선 차이가 zero_cross_V 아래에서 음수로 바뀜).",
}
_sg = np.sign(diff["batch3"]); _zc = np.where((_sg[:-1] > 0) & (_sg[1:] < 0) & (V[:-1] < 3.3))[0]
R["batch3_note_qdlin"]["offset_mechanism"]["zero_cross_V_B3_minus_B1"] = float(V[_zc[0]]) if len(_zc) else None


# ── 예외 후보 Qcc_init: 원시 Qdlin 의 2.0 V CC 끝점(사이클 2–6 중앙값) = '절대 Qdlin 수준' 그 자체 ──
#    노션 경고(충전 커브 시작 시점이 배치별로 상이 → Qdlin 단순 비교 왜곡)의 대상이므로 배치 오프셋을 라벨 없이 정량화한다.
def qcc_init(c: dict, a: int = 2, b: int = 6) -> float:
    v = np.array([qdlin(c, k)[-1] for k in range(a, b + 1)], float)
    v[(v <= 0) | (v > 1.2)] = np.nan          # Qdlin 끝점 이상값(≤0 또는 >1.2 Ah) 제외
    return float(np.nanmedian(v))


def qd_summary_cycle2(c: dict) -> float:
    s = clean_summary(c["summary"])
    return float(s.loc[s["cycle"] == 2, "QD"].iloc[0])


qcc_rows = []
for c in cells:                              # 라벨 유무와 무관(라벨 미사용 분포 비교) — 수명 시험 프로토콜만
    g_ = ct.loc[c["cell_key"], "group"]
    if g_ not in ("fastcharge", "newstructure"):
        continue
    qcc_rows.append({"cell_key": c["cell_key"], "batch": c["batch"], "group": g_,
                     "labeled": bool(ct.loc[c["cell_key"], "labeled"]),
                     "Qcc_init": qcc_init(c), "QD2_summary": qd_summary_cycle2(c)})
qcc_df = pd.DataFrame(qcc_rows)
_b1all = qcc_df[qcc_df.batch == "batch1"]
_b1lab = qcc_df[(qcc_df.batch == "batch1") & qcc_df.labeled].set_index("cell_key").loc[b1.cell_key]
# B1 기술 회귀(설명용, CV 아님): log10 수명 ~ log10 var ΔQ + Qcc_init → Qcc 계수(log10/Ah)로 오프셋을 예측 배율로 환산
_X = np.column_stack([np.ones(len(b1)), b1.log_var.to_numpy(), _b1lab.Qcc_init.to_numpy()])
_coef = np.linalg.lstsq(_X, b1.log_life.to_numpy(), rcond=None)[0]
_ref, _sd = float(_b1all.Qcc_init.median()), float(_b1lab.Qcc_init.std())
_ref_qd2 = float(_b1all.QD2_summary.median())
QCC_GROUPS = [("batch2", "fastcharge"), ("batch2", "newstructure"), ("batch3", "newstructure")]
qcc_shift = {}
for b_, g_ in QCC_GROUPS:
    x_ = qcc_df[(qcc_df.batch == b_) & (qcc_df.group == g_)]
    sh, sh2 = float(x_.Qcc_init.median() - _ref), float(x_.QD2_summary.median() - _ref_qd2)
    qcc_shift[f"{b_}_{g_}"] = {
        "n": int(len(x_)), "shift_mAh": sh * 1e3, "shift_in_B1_SD": sh / _sd, "pred_factor": float(10 ** (_coef[2] * sh)),
        "n_outside_B1_labeled_range": int(((x_.Qcc_init < _b1lab.Qcc_init.min()) | (x_.Qcc_init > _b1lab.Qcc_init.max())).sum()),
        "summary_QD2_shift_mAh": sh2 * 1e3, "summary_QD2_pred_factor_same_coef": float(10 ** (_coef[2] * sh2))}
R["batch3_note_qdlin"]["qcc_init_exception"] = {
    "definition": "Qcc_init = median_{k=2..6} Qdlin_k(2.0 V) (원시 Qdlin 의 CC 방전 끝점; ≤0·>1.2 Ah 끝점 제외). 절대 Qdlin 수준.",
    "reference": "Batch 1 46셀(중도절단 포함, 라벨 미사용) 중앙값", "ref_Ah": _ref, "B1_labeled_sd_mAh": _sd * 1e3,
    "batch1_descriptive_fit": {"intercept": float(_coef[0]), "coef_log_var": float(_coef[1]), "coef_Qcc_per_Ah": float(_coef[2]),
                               "plus10mAh_factor": float(10 ** (_coef[2] * 0.01)),
                               "r_log_var_Qcc": float(np.corrcoef(b1.log_var, _b1lab.Qcc_init)[0, 1]),
                               "note": "B1 labeled 36셀 in-sample 최소제곱(설명용, CV 아님). 채택 여부는 DAY 2 B1 정책 단위 grouped-CV 1-SE 규칙."},
    "shift_by_group": qcc_shift,
    "link_to_raw_offset": "B3−B1 배치 평균 Qdlin₁₀(2.0 V) 차이(−0.015 Ah)와 같은 현상(총 CC 방전 용량 차이)",
    "notion_warning": "충전 커브 시작 시점이 배치별로 상이 : Qdlin 변수를 단순 비교하면 왜곡 발생",
}

# PDF(약 16 cm 폭)용 2×2: ① 노션 메모 직접 확인(기록 시작점) → ② 원시 Qdlin 배치 오프셋 ≈ 수명 신호
#                        → ③ 원시 수준은 B1 안에서도 수명 설명력 약함(피처 제외 근거) → ④ 예외 후보 Qcc_init 의 배치 오프셋
CATS = [("batch1", "fastcharge", "B1\n고속충전"), ("batch2", "fastcharge", "B2\n고속충전"),
        ("batch2", "newstructure", "B2\nnewstructure"), ("batch3", "newstructure", "B3\nnewstructure")]
with plt.rc_context(PDF_RC):
    fig, axs = plt.subplots(2, 2, figsize=(10.4, 8.4))
    rng = np.random.default_rng(RANDOM_STATE)
    # ① 원시 충전 시계열의 첫 기록 전압 (사이클 11; 사이클 10 원시 시계열은 저장 안 됨)
    ax = axs[0, 0]
    for i, (b, grp, _) in enumerate(CATS):
        g = raw_df[(raw_df.batch == b) & (raw_df.group == grp)]
        col = ps.BATCH_COLOR[b]
        ax.boxplot(g.V0, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.18, edgecolor=col), medianprops=dict(color=col, lw=2),
                   whiskerprops=dict(color=col), capprops=dict(color=col))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(g)), g.V0, s=14, color=col, alpha=0.75, zorder=3)
        ax.text(i, g.V0.max() + 0.012, f"평균 {g.V0.mean():.2f} V", ha="center", va="bottom", fontsize=10, color="#333")
    ax.set_xticks(range(len(CATS)), [c[2] for c in CATS], fontsize=10)
    ax.set_xlim(-0.6, len(CATS) - 0.4)
    ax.set_ylim(1.98, 2.43)
    ax.set_ylabel("원시 기록 첫 전압 (V), 사이클 11")
    ax.set_title("① 충전 기록 시작점이 배치·프로토콜마다 다름")
    # ② 배치 평균 원시 Qdlin₁₀ 차이 vs 수명 신호 크기
    ax = axs[0, 1]
    for b in ("batch2", "batch3"):
        ax.plot(V, diff[b], color=ps.BATCH_COLOR[b], lw=2, label=f"{BN[b]} − Batch 1")
    for b in ("batch1", "batch3"):
        ax.axhline(ref_signal[b], color=ps.BATCH_COLOR[b], ls="--", lw=1)
        ax.text(2.03, ref_signal[b] + 0.001, f"{BN[b]} 수명 신호 |min ΔQ| {ref_signal[b]:.3f}", va="bottom",
                fontsize=9.5, color=ps.BATCH_COLOR[b])
    ax.axhline(0, color="#555", lw=0.8)
    d3 = R["batch3_note_qdlin"]["batchmean_diff_vs_B1_maxabs"]["batch3"]
    om = R["batch3_note_qdlin"]["offset_mechanism"]["at_max_diff"]
    ax.annotate(f"{d3['value_Ah']:+.3f} Ah\n(≈{om['shift_mV_at_that_Q']:.0f} mV 이동)", (d3["at_V"], d3["value_Ah"]),
                xytext=(2.55, 0.006), fontsize=10, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color="#333", shrinkB=2))
    ax.annotate(f"2.0 V: {d3['at_2p0V']:+.3f} Ah\n= ④ 의 CC 끝점 차이".replace("-", "−"), (2.0, d3["at_2p0V"]), xytext=(2.25, -0.036),
                fontsize=10, va="center", arrowprops=dict(arrowstyle="->", color="#333"))
    ax.set_xlim(2.0, 3.5)
    ax.set_ylim(-0.047, 0.047)
    ax.set_xlabel("전압 (V)")
    ax.set_ylabel("배치 평균 Qdlin₁₀ 차이 (Ah)")
    ax.set_title("② 원시 Qdlin₁₀ 배치 오프셋 ≈ 수명 신호 크기")
    ax.legend(loc="lower right", fontsize=9.5)
    # ③ 전압별 수명 상관 (배치 안): 원시 Qdlin₁₀ 수준(점선) vs ΔQ(실선)
    ax = axs[1, 0]
    for b in BATCH_NAMES:
        ax.plot(V[valid_V], prof[("DQ", b)][valid_V], color=ps.BATCH_COLOR[b], lw=2)
        ax.plot(V[valid_V], prof[("Q10", b)][valid_V], color=ps.BATCH_COLOR[b], lw=1.4, ls=(0, (4, 2)), alpha=0.9)
    ax.axhline(0, color="#555", lw=0.8)
    ax.set_xlim(2.0, 3.25)
    ax.set_ylim(-0.3, 1.0)
    ax.set_xlabel("전압 (V)")
    ax.set_ylabel("log₁₀ 수명과의 Pearson r")
    ax.set_title("③ 배치 안 수명 상관: 원시 수준(점선) ≪ ΔQ(실선)")
    _mx = R["batch3_note_qdlin"]["max_abs_corr_V_le_3p25"]
    ax.text(2.03, 0.36, f"B1 최대 |r|: ΔQ {_mx['DQ_batch1']:.2f} vs 원시 {_mx['Q10_batch1']:.2f}", fontsize=10, color="#222")
    h = [Line2D([], [], color="#444", lw=2, label="ΔQ₁₀₀₋₁₀"), Line2D([], [], color="#444", lw=1.4, ls=(0, (4, 2)), label="원시 Qdlin₁₀")]
    h += [Line2D([], [], color=ps.BATCH_COLOR[b], lw=3, label=BN[b]) for b in BATCH_NAMES]
    ax.legend(handles=h, loc="lower left", fontsize=9.3, ncol=3, columnspacing=0.8, handlelength=1.6)
    # ④ 후보 Qcc_init (2.0 V CC 끝점) 의 배치 이동 — 라벨 미사용 (mAh·B1 SD 단위만, 적합 계수 배율은 그리지 않음)
    ax = axs[1, 1]
    QI = R["batch3_note_qdlin"]["qcc_init_exception"]
    ax.axhspan(-QI["B1_labeled_sd_mAh"], QI["B1_labeled_sd_mAh"], color="#999", alpha=0.12, lw=0)
    ax.axhline(0, color=ps.BATCH_COLOR["batch1"], lw=1, ls="--")
    for i, (b, grp, _) in enumerate(CATS):
        y = (qcc_df[(qcc_df.batch == b) & (qcc_df.group == grp)].Qcc_init - QI["ref_Ah"]) * 1e3
        col = ps.BATCH_COLOR[b]
        ax.boxplot(y, positions=[i], widths=0.5, showfliers=False, patch_artist=True,
                   boxprops=dict(facecolor=col, alpha=0.18, edgecolor=col), medianprops=dict(color=col, lw=2),
                   whiskerprops=dict(color=col), capprops=dict(color=col))
        ax.scatter(i + rng.uniform(-0.15, 0.15, len(y)), y, s=12, color=col, alpha=0.7, zorder=3)
        if b != "batch1":
            s_ = QI["shift_by_group"][f"{b}_{grp}"]
            ax.text(i, 21, f"{s_['shift_mAh']:+.1f} mAh\n({s_['shift_in_B1_SD']:+.1f} SD)".replace("-", "−"),
                    ha="center", va="bottom", fontsize=10.5, fontweight="bold", color="#222")
    ax.set_xticks(range(len(CATS)), [f"{c[2]}" for c in CATS], fontsize=10)
    ax.set_xlim(-0.6, len(CATS) - 0.4)
    ax.set_ylim(-45, 33)
    ax.set_ylabel("Qcc_init − B1 중앙값 (mAh)")
    ax.set_title("④ 후보 초기 용량 Qcc_init(2.0 V CC 끝점)의 배치 이동")
    fig.text(0.5, -0.012,
             f"④ 라벨 미사용 · 기준 = B1 46셀 중앙값 · 회색 = B1 ±1 SD({QI['B1_labeled_sd_mAh']:.1f} mAh) · "
             f"괄호 = B1 SD 단위 이동량",
             ha="center", fontsize=10.5, color="#444")
    fig.tight_layout(rect=(0, 0.015, 1, 0.965))
    fig.suptitle("(d) 노션 메모 'Qdlin 단순 비교 시 왜곡' 점검: 원시 Qdlin 수준의 배치 오프셋은 수명 신호와 같은 크기",
                 fontsize=13, fontweight="bold", y=0.995)
    ps.save(fig, "q3_d_qdlin_batch_offset")

# ─────────────────────────── 저장 ───────────────────────────
R["per_cell_features"] = df.round(6).to_dict(orient="records")


def _clean(o):
    if isinstance(o, dict):
        return {str(k) if not isinstance(k, tuple) else "|".join(map(str, k)): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(f"{float(o):.6g}")   # 유효숫자 6자리 (p-value 보존)
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


OUT_JSON.write_text(json.dumps(_clean(R), ensure_ascii=False, indent=1))
print("saved", OUT_JSON)
print(json.dumps(_clean({k: R[k] for k in ("meta", "a_curves")}), ensure_ascii=False, indent=1))
print("best feature (B1):", best)
for k in ("batch1", "batch2", "batch2_fastcharge", "batch3", "pooled"):
    print(k, {f: (round(CORR[k][f]["pearson"], 3), round(CORR[k][f]["spearman"], 3)) for f in ("log_var", "log_absmin", "min", "e_log_var")})
