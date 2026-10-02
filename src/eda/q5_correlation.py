"""DAY 1 EDA — Question 5. 상관관계 - 어떤 신호가 수명과 연관되어 있는가?

노션 하위 항목
  (a) 초기 사이클 피쳐들과 Cycle Life 상관 계수 확인
  (b) 가장 강한 관계 식별
  (c) 멀티클리니어리티 문제 확인

규칙
  - 초기 사이클(≤100)만 사용. Batch 1 의 cycle 1 은 더미이므로 모든 배치에서 cycle 2 부터 사용(배치 간 정의 통일).
  - clean_summary 로 측정 오류를 NaN 처리한 뒤 피처 계산.
  - 상관 분석 대상 = labeled 셀(라벨 있음 & 중도절단 아님): Batch 1 36, Batch 2 39, Batch 3 44.
  - 피처 선별 판단은 Batch 1(학습셋) 기준. Batch 2/3 는 '부호·강도 일관성' 확인(일반화 위험 점검)에만 사용.
  - 모델 학습·튜닝 없음(기술통계와 상관만). 배치별 log–log 직선은 수준 비교용 기술 회귀이며 예측에 쓰지 않음.
  - 다중공선성 정리: ① 중복 그룹 대표 1개 ② VIF>5 순차 제거(ΔQ 분산 보호)
    ③ 대표가 모두 빠진 그룹은 차순위 피처를 재편입(보호 피처 외 VIF ≤ 5 일 때만).
    ④ 측정 정의를 통일한 DAY 2 M2 Elastic-Net 풀(Qcc_init·CC break-in·실측 충전시간)의 VIF.
  - 공통 표기: 용량 기울기 = mAh/100 cycle, 음수 = 감소, 추정량 OLS(np.polyfit; Theil–Sen 값은 참고 열).
    첨도 = Fisher excess(정규분포 = 0).

산출물
  results/eda/early_features.csv   셀당 1행 초기 피처 테이블(전체 139셀, labeled 플래그 포함)
  results/eda/q5_results.json      수치 결과
  reports/figures/q5_*.png         그림 (q5_corr_ranking = Spearman 단일 패널, _3metrics = 3지표판)
실행: cd ess-battery-project && python src/eda/q5_correlation.py  (그다음 src/eda/q5d_conditional.py)
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402
from scipy.cluster import hierarchy  # noqa: E402
from scipy.spatial.distance import squareform  # noqa: E402

import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.legend_handler import HandlerTuple  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

from data import (BATCH_NAMES, RANDOM_STATE, ROOT, cell_table, clean_summary,  # noqa: E402
                  delta_q, load_all)
from eda.q5d_conditional import cc_breakin_mAh, qcc_init  # noqa: E402  (CC 끝점 피처 정의 공유)

OUT = ROOT / "results" / "eda"
OUT.mkdir(parents=True, exist_ok=True)

FIRST, LAST = 2, 100          # 초기 사이클 창 (Batch 1 cycle 1 더미 제외)
SPREAD_CONST = 0.05           # 배치 내 표준편차가 Batch 1 의 5% 미만 → '사실상 상수' (상관 해석 불가)
VIF_MAX = 5.0                 # VIF 가지치기 기준
N_BOOT = 5000

# ── 피처 정의 (컬럼명, 한글 라벨, 도메인 그룹) ─────────────────────────────
FEATURES = {
    "dQ_logvar":         ("log₁₀ Var(ΔQ₁₀₀₋₁₀)",        "ΔQ(V)"),
    "dQ_logabsmin":      ("log₁₀ |min ΔQ₁₀₀₋₁₀|",       "ΔQ(V)"),
    "dQ_skew":           ("ΔQ₁₀₀₋₁₀ 왜도",               "ΔQ(V)"),
    "dQ_kurt":           ("ΔQ₁₀₀₋₁₀ 첨도 (Fisher)",      "ΔQ(V)"),
    "QD_2":              ("방전용량 QD(2)",               "용량"),
    "QD_max_minus_2":    ("max QD − QD(2)",              "용량"),
    "fade_slope_2_100":  ("용량 기울기 (2–100, −=감소)",  "용량"),
    "fade_int_2_100":    ("용량 적합 절편 (2–100)",       "용량"),
    "fade_slope_91_100": ("용량 기울기 (91–100, −=감소)", "용량"),
    "fade_int_91_100":   ("용량 적합 절편 (91–100)",      "용량"),
    "chargetime_avg5":   ("평균 충전시간 (cycle 2–6)",    "충전"),
    "Tmax_max":          ("최고온도 max (2–100)",         "온도"),
    "Tavg_mean":         ("평균온도 평균 (2–100)",        "온도"),
    "Tmin_min":          ("최저온도 min (2–100)",         "온도"),
    "IR_2":              ("내부저항 IR(2)",               "저항"),
    "IR_min":            ("내부저항 min (2–100)",         "저항"),
    "IR_diff":           ("IR(100) − IR(2)",             "저항"),
    "avgC_80":           ("평균 C-rate (0→80%)",          "프로토콜"),
    "t80_min":           ("0→80% 충전시간 (프로토콜)",     "프로토콜"),
}
FEATS = list(FEATURES)
LBL = {k: v[0] for k, v in FEATURES.items()}
GRP = {k: v[1] for k, v in FEATURES.items()}
# DAY 2 M2 풀에만 쓰는 CC 끝점 피처 (19개 상관 표에는 넣지 않음)
LBL.update({"Qcc_init": "Qcc_init (CC 초기 용량)", "cc_breakin": "CC break-in 상승폭"})
GRP.update({"Qcc_init": "용량", "cc_breakin": "용량"})
M2_POOL = ["dQ_logvar", "Qcc_init", "cc_breakin", "dQ_skew", "dQ_kurt", "fade_slope_91_100",
           "chargetime_avg5", "Tavg_mean", "IR_min", "IR_diff"]
SLOPE_SCALE = 1e5            # Ah/cycle → mAh/100 cycle
# 프로토콜 라벨 색은 Batch 1 막대(파랑)와 헷갈리지 않도록 자홍색 사용
GROUP_COLOR = {"ΔQ(V)": "#7C3AED", "용량": "#0F766E", "충전": "#B45309", "온도": "#BE123C",
               "저항": "#475569", "프로토콜": "#A21CAF"}
CONST_GRAY = "#9CA3AF"
SHORT = {"batch1": "Batch 1", "batch2": "Batch 2", "batch3": "Batch 3"}


# ════════════════════════════════════════════════════════════════════════
# (a) 초기 사이클 피처 테이블
# ════════════════════════════════════════════════════════════════════════
def _linfit(cyc: np.ndarray, q: np.ndarray, a: int, b: int) -> tuple[float, float]:
    """OLS 선형적합. 기울기는 mAh/100 cycle(음수 = 감소), 절편은 Ah."""
    m = (cyc >= a) & (cyc <= b) & np.isfinite(q)
    slope, intercept = np.polyfit(cyc[m], q[m], 1)
    return float(slope * SLOPE_SCALE), float(intercept)


def _theilsen(cyc: np.ndarray, q: np.ndarray, a: int, b: int) -> float:
    """참고용 Theil–Sen 기울기 (mAh/100 cycle, 음수 = 감소)."""
    m = (cyc >= a) & (cyc <= b) & np.isfinite(q)
    return float(stats.theilslopes(q[m], cyc[m])[0] * SLOPE_SCALE)


def early_features(c: dict) -> dict:
    s = clean_summary(c["summary"])
    s = s[(s["cycle"] >= FIRST) & (s["cycle"] <= LAST)].set_index("cycle")
    cyc = s.index.to_numpy(float)
    qd = s["QD"].to_numpy(float)
    dq = delta_q(c, 100, 10)                       # 노션: 사이클 100 − 사이클 10
    dq = dq[np.isfinite(dq)]
    f = {
        "dQ_logvar": np.log10(np.var(dq)),
        "dQ_logabsmin": np.log10(np.abs(dq.min())),
        "dQ_skew": stats.skew(dq),
        "dQ_kurt": stats.kurtosis(dq, fisher=True),             # Fisher excess (정규 = 0)
        "QD_2": s.at[2.0, "QD"],
        "QD_max_minus_2": np.nanmax(qd) - s.at[2.0, "QD"],
    }
    f["fade_slope_2_100"], f["fade_int_2_100"] = _linfit(cyc, qd, 2, 100)
    f["fade_slope_91_100"], f["fade_int_91_100"] = _linfit(cyc, qd, 91, 100)
    f["fade_slope_2_100_ts"] = _theilsen(cyc, qd, 2, 100)        # 참고 열(상관 표 밖)
    f["Qcc_init"] = qcc_init(c)                                    # CC 끝점(2.0 V) cycle 2–6 중앙값, Ah
    f["cc_breakin"] = cc_breakin_mAh(c)                            # mAh
    f["chargetime_avg5"] = s.loc[2:6, "chargetime"].mean()      # 첫 5 사이클 = cycle 2–6
    f["Tmax_max"] = s["Tmax"].max()
    f["Tavg_mean"] = s["Tavg"].mean()
    tav = s["Tavg"].interpolate(limit_direction="both").to_numpy(float)
    f["Tavg_int"] = float(np.trapezoid(tav, cyc))                 # 원논문식 적분 (= 평균×98 과 사실상 동일)
    f["Tmin_min"] = s["Tmin"].min()
    f["IR_2"] = s.at[2.0, "IR"]
    f["IR_min"] = s["IR"].min()
    f["IR_diff"] = s.at[100.0, "IR"] - s.at[2.0, "IR"]
    return {k: float(v) for k, v in f.items()}


def build_table() -> pd.DataFrame:
    cells = load_all()
    ct = cell_table(cells)
    feat = pd.DataFrame([{"cell_key": c["cell_key"], **early_features(c)} for c in cells])
    df = ct[["batch", "cell_key", "idx", "policy", "group", "cycle_life", "censored", "labeled",
             "label_550", "avgC_80", "t80_min"]].merge(feat, on="cell_key")
    df["log10_cycle_life"] = np.log10(df["cycle_life"])
    cols = ["batch", "cell_key", "idx", "policy", "group", "cycle_life", "log10_cycle_life", "censored",
            "labeled", "label_550"] + FEATS + ["Tavg_int", "fade_slope_2_100_ts", "Qcc_init", "cc_breakin"]
    return df[cols]


# ════════════════════════════════════════════════════════════════════════
# 상관 계산
# ════════════════════════════════════════════════════════════════════════
def corr_row(x: pd.Series, y: pd.Series, sd_ref: float) -> dict:
    m = x.notna() & y.notna()
    x, y = x[m].to_numpy(float), y[m].to_numpy(float)
    spread = float(np.std(x, ddof=1) / sd_ref)
    pr, pp = stats.pearsonr(x, y)
    prl, ppl = stats.pearsonr(x, np.log10(y))
    sr, sp = stats.spearmanr(x, y)
    srl = stats.spearmanr(x, np.log10(y))[0]                    # 단조변환이라 spearman 과 동일(검증용)
    z = (x - x.mean()) / x.std(ddof=1)
    k = np.abs(z) < 3                                            # |z|≥3 극단 셀 제외 Pearson (영향점 점검)
    prl_trim = stats.pearsonr(x[k], np.log10(y[k]))[0] if k.sum() > 3 else np.nan
    return {"n": int(m.sum()), "sd_ratio_vs_batch1": spread, "constant": bool(spread < SPREAD_CONST),
            "pearson_log10life_trim3z": float(prl_trim), "n_trim3z": int((~k).sum()),
            "pearson_life": pr, "p_pearson_life": pp,
            "pearson_log10life": prl, "p_pearson_log10life": ppl,
            "spearman": sr, "p_spearman": sp, "spearman_log10life": float(srl),
            "x_min": float(x.min()), "x_max": float(x.max())}


def boot_ci(x: np.ndarray, y: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    """셀 단위 부트스트랩 95% CI (Spearman)."""
    n = len(x)
    vals = np.empty(N_BOOT)
    for i in range(N_BOOT):
        j = rng.integers(0, n, n)
        vals[i] = stats.spearmanr(x[j], y[j])[0]
    vals = vals[np.isfinite(vals)]
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def r_crit(n: int, alpha: float = 0.05) -> float:
    t = stats.t.ppf(1 - alpha / 2, n - 2)
    return float(t / np.sqrt(n - 2 + t ** 2))


def consistency(r1: float, r_other: dict[str, float], const: dict[str, bool], rc: float) -> str:
    """Batch 1 신호가 Batch 2/3 에서 유지되는지 판정 (Spearman 기준)."""
    tags = []
    for b in ("batch2", "batch3"):
        r = r_other[b]
        if const[b]:
            tags.append(f"{SHORT[b]}: 상수(해석 불가)")
        elif np.sign(r) != np.sign(r1) and abs(r) >= 0.2:
            tags.append(f"{SHORT[b]}: 부호 반전")
        elif abs(r) < rc or abs(r) < 0.5 * abs(r1):
            tags.append(f"{SHORT[b]}: 약화/소멸")
        else:
            tags.append(f"{SHORT[b]}: 유지")
    if all(t.endswith("유지") for t in tags):
        return "일관"
    return " / ".join(tags)


# ════════════════════════════════════════════════════════════════════════
# 다중공선성 (VIF)
# ════════════════════════════════════════════════════════════════════════
def vif_table(X: pd.DataFrame) -> pd.Series:
    Z = (X - X.mean()) / X.std(ddof=0)
    try:
        from statsmodels.stats.outliers_influence import variance_inflation_factor
        import statsmodels.api as sm
        A = sm.add_constant(Z.to_numpy(float))
        v = [variance_inflation_factor(A, i + 1) for i in range(Z.shape[1])]
        how = "statsmodels"
    except ImportError:                                            # 대체: VIF = 1/(1−R²)
        v = []
        for col in Z.columns:
            y = Z[col].to_numpy(float)
            A = np.column_stack([np.ones(len(Z)), Z.drop(columns=col).to_numpy(float)])
            beta, *_ = np.linalg.lstsq(A, y, rcond=None)
            r2 = 1 - np.sum((y - A @ beta) ** 2) / np.sum((y - y.mean()) ** 2)
            v.append(1 / max(1 - r2, 1e-12))
        how = "lstsq"
    s = pd.Series(v, index=X.columns, dtype=float)
    s.attrs["how"] = how
    return s


# ════════════════════════════════════════════════════════════════════════
# 그림
# ════════════════════════════════════════════════════════════════════════
def fig_ranking(C: pd.DataFrame, order: list[str], ci: dict, rc: float, single: bool = True) -> None:
    """single=True: Spearman 단일 패널(보고서용, q5_corr_ranking) / False: 3지표판(q5_corr_ranking_3metrics)."""
    metrics = [("spearman", "Spearman ρ\n(cycle_life · log₁₀ 동일)"),
               ("pearson_log10life", "Pearson r\n(log₁₀ cycle_life)"),
               ("pearson_life", "Pearson r\n(cycle_life)")]
    if single:
        metrics = metrics[:1]
    n = len(order)
    if single:
        fig, ax0 = plt.subplots(figsize=(8.6, 0.28 * n + 1.9))
        axes = [ax0]
    else:
        fig, axes = plt.subplots(1, 3, figsize=(15.5, 0.42 * n + 2.2), sharey=True, gridspec_kw={"wspace": 0.08})
    fsz = 11.0 if single else 10
    y = np.arange(n)[::-1]
    for ax, (m, title) in zip(axes, metrics):
        ax.axvspan(-rc, rc, color="#E5E7EB", alpha=0.7, lw=0, zorder=0)
        ax.axvline(0, color="#6B7280", lw=0.8, zorder=1)
        v1 = [C.loc[("batch1", f), m] for f in order]
        ax.barh(y, v1, height=0.62, color=ps.BATCH_COLOR["batch1"], alpha=0.85, zorder=2)
        if m == "spearman":
            lo = [v - ci[f][0] for v, f in zip(v1, order)]
            hi = [ci[f][1] - v for v, f in zip(v1, order)]
            ax.errorbar(v1, y, xerr=[lo, hi], fmt="none", ecolor="#1E3A8A", elinewidth=0.9,
                        capsize=2, zorder=3)
        for b, mk, dy in (("batch2", "o", 0.13), ("batch3", "D", -0.13)):
            for f, yy in zip(order, y):
                r = C.loc[(b, f)]
                if r["constant"]:      # 상수: 배치 모양(○/◇)을 회색 빈 테두리로 + 안에 ×
                    ax.plot(r[m], yy + dy, marker=mk, ms=8.5 if mk == "o" else 7.5, mfc="white", mec=CONST_GRAY,
                            mew=1.3, ls="none", zorder=4)
                    ax.plot(r[m], yy + dy, marker="x", ms=4.6, mew=1.3, color=CONST_GRAY, ls="none", zorder=5)
                else:
                    ax.plot(r[m], yy + dy, marker=mk, ms=6.5, mfc=ps.BATCH_COLOR[b], mec="white",
                            mew=0.7, ls="none", zorder=4)
        ax.set_xlim(-1, 1)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        if not single:
            ax.set_title(title, fontsize=11.5)
        ax.grid(axis="y", visible=False)
    axes[0].set_yticks(y)
    axes[0].set_yticklabels([LBL[f] for f in order], fontsize=fsz)
    for t, f in zip(axes[0].get_yticklabels(), order):
        t.set_color(GROUP_COLOR[GRP[f]])
    axes[len(axes) // 2].set_xlabel(("Spearman ρ with cycle_life" if single else "상관계수")
                                    + " (labeled: B1 n=36 · B2 n=39 · B3 n=44)", fontsize=fsz)
    handles = [Patch(color=ps.BATCH_COLOR["batch1"], alpha=0.85),
               Line2D([], [], marker="o", ls="none", mfc=ps.BATCH_COLOR["batch2"], mec="white", ms=7),
               Line2D([], [], marker="D", ls="none", mfc=ps.BATCH_COLOR["batch3"], mec="white", ms=7),
               (Line2D([], [], marker="o", ls="none", mfc="white", mec=CONST_GRAY, mew=1.3, ms=8.5),
                Line2D([], [], marker="x", ls="none", color=CONST_GRAY, mew=1.3, ms=4.6)),
               (Line2D([], [], marker="D", ls="none", mfc="white", mec=CONST_GRAY, mew=1.3, ms=7.5),
                Line2D([], [], marker="x", ls="none", color=CONST_GRAY, mew=1.3, ms=4.6)),
               Patch(color="#E5E7EB")]
    if single:
        labels = ["Batch 1 (학습; 오차막대 = 부트스트랩 95% CI)", "Batch 2", "Batch 3",
                  "Batch 2 상수 (해석 불가)", "Batch 3 상수 (해석 불가)", f"|ρ| < {rc:.2f}: B1에서 p>0.05"]
        fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.43, 0.095), fontsize=9.6,
                   handlelength=1.5, columnspacing=1.2, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})
        fig.subplots_adjust(top=0.94, bottom=0.165, left=0.33, right=0.98)
        ax0.set_title("초기 100 사이클 피처 ↔ Cycle Life (Batch 1 |ρ| 순)", fontsize=12.5, loc="left", x=-0.42)
        ps.save(fig, "q5_corr_ranking")
        return
    labels = ["Batch 1 (학습, 막대; 오차막대=부트스트랩 95% CI)", "Batch 2", "Batch 3",
              "Batch 2 사실상 상수(배치 내 표준편차 < Batch 1의 5%) → 해석 불가",
              "Batch 3 사실상 상수(같은 기준) → 해석 불가", f"|r| < {rc:.2f} : Batch 1(n=36)에서 p>0.05"]
    leg1 = fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.035), fontsize=9.5,
                      handlelength=1.5, columnspacing=1.6, handler_map={tuple: HandlerTuple(ndivide=1, pad=0)})
    gh = [Line2D([], [], ls="none", marker="s", ms=9, color=c, label=g) for g, c in GROUP_COLOR.items()]
    fig.legend(handles=gh, loc="upper center", ncol=6, bbox_to_anchor=(0.5, -0.025), fontsize=9.5,
               title="피처 이름 색 = 도메인 그룹", title_fontsize=9.5, columnspacing=1.4, handletextpad=0.3)
    fig.add_artist(leg1)
    fig.subplots_adjust(top=0.88, bottom=0.11)
    fig.suptitle("초기 100 사이클 피처 ↔ Cycle Life 상관 — Batch 1 |Spearman ρ| 순 정렬",
                 fontsize=14, fontweight="bold", y=0.965)
    ps.save(fig, "q5_corr_ranking_3metrics")


def _zscore(s: pd.Series) -> pd.Series:
    return (s - s.mean()) / s.std(ddof=1)


def fig_top_signals(L: pd.DataFrame, C: pd.DataFrame, panels: list[tuple], extra: dict) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14.0, 10.6))
    zcache = {}
    for ax, (f, mode, title) in zip(axes.ravel(), panels):
        for b, mk in (("batch1", "o"), ("batch2", "o"), ("batch3", "D")):
            d = L[L["batch"] == b]
            x = _zscore(d[f]) if mode == "z" else d[f]
            zcache[(b, f)] = x
            ax.scatter(x, d["cycle_life"], s=30, marker=mk, color=ps.BATCH_COLOR[b], alpha=0.8,
                       edgecolor="white", lw=0.5, label=SHORT[b], zorder=3)
            if mode == "z" and not C.loc[(b, f), "constant"]:        # Theil–Sen(이상치에 강건) 추세선
                m = x.notna()
                k, c0, *_ = stats.theilslopes(np.log10(d["cycle_life"][m]), x[m])
                xx = np.linspace(x[m].min(), x[m].max(), 50)
                ax.plot(xx, 10 ** (k * xx + c0), color=ps.BATCH_COLOR[b], lw=1.8, alpha=0.9, zorder=2)
        ax.set_yscale("log")
        ax.set_ylim(360, 2250)
        ax.set_yticks([400, 600, 800, 1000, 1500, 2000])
        ax.set_yticklabels(["400", "600", "800", "1000", "1500", "2000"])
        ax.minorticks_off()
        ax.set_ylabel("Cycle Life (로그 축)")
        ax.set_xlabel(f"{LBL[f]}" + ("  — 배치 내 표준화(z), 실선 = Theil–Sen 추세" if mode == "z" else ""))
        rr = []
        for b in BATCH_NAMES:
            r = C.loc[(b, f)]
            rr.append(f"{SHORT[b]} " + ("상수" if r["constant"] else f"{r['spearman']:+.2f}"))
        ax.set_title(title + "\nSpearman ρ:  " + "  ·  ".join(rr), fontsize=11.3, loc="left")
    a1, a2, a3, a4 = axes.ravel()
    kw = dict(fontsize=8.6)
    box = dict(boxstyle="round,pad=0.45", fc="white", ec="#D1D5DB", alpha=0.92)
    # ① Batch 1 값 범위 + Batch 1 기술 회귀선 + 수준 비교
    lo, hi = extra["b1_range_logvar"]
    k1, c1 = extra["b1_line"]
    a1.set_xlim(-5.95, -2.95)
    a1.axvspan(lo, hi, color=ps.BATCH_COLOR["batch1"], alpha=0.07, lw=0, zorder=0)
    a1.text((lo + hi) / 2, 2150, f"Batch 1 값 범위\n(Batch 3의 {extra['b3_below']:.0%}는\n이보다 왼쪽 = 외삽)",
            color=ps.BATCH_COLOR["batch1"], fontsize=8.8, ha="center", va="top", linespacing=1.35)
    a1.set_xlabel(f"{LBL['dQ_logvar']}  — 실선 = Batch 1 기술 회귀선(log–log), 점선 = B1 범위 밖 연장")
    xin = np.linspace(lo, hi, 20)
    a1.plot(xin, 10 ** (k1 * xin + c1), color=ps.BATCH_COLOR["batch1"], lw=1.6, zorder=2)
    for xs in (np.linspace(-5.3, lo, 20), np.linspace(hi, -3.0, 20)):
        a1.plot(xs, 10 ** (k1 * xs + c1), color=ps.BATCH_COLOR["batch1"], lw=1.4, ls=(0, (2, 2)), zorder=2)
    w = extra["b2_window"]
    a1.axvspan(w["lo"], w["hi"], ymin=0, ymax=0.035, color=ps.BATCH_COLOR["batch2"], alpha=0.55, lw=0, zorder=1)
    rt = extra["ratio_to_b1_line"]
    a1.text(0.02, 0.03,
            "B1 회귀선 대비 수명비 (중앙값)\n"
            f"  Batch 3 {rt['batch3']:.2f} · B2 newstructure {rt['b2_newstructure']:.2f}\n"
            f"  B2 fastcharge {rt['b2_fastcharge']:.2f} ({rt['b2_fastcharge'] - 1:+.0%})\n"
            "같은 ΔQ 분산 창(x축 주황 띠)의 수명 중앙값\n"
            f"  B1 {w['b1_n']}셀 중앙 {w['b1_median']:.0f}\n"
            f"  B2 fastcharge {w['b2_n']}셀 중앙 {w['b2_median']:.0f} ({w['rel_diff']:+.0%})",
            transform=a1.transAxes, fontsize=8.3, color="#374151", va="bottom", linespacing=1.4, bbox=box)
    # ② Batch 2 두 집단 / fastcharge 내부 추세 / Batch 3 4.8C(80%) 집단
    zf = zcache[("batch2", "chargetime_avg5")]
    lf = L.loc[L["batch"] == "batch2", "cycle_life"]
    mf = (L.loc[L["batch"] == "batch2", "group"] == "fastcharge").to_numpy()
    kf, cf, *_ = stats.theilslopes(np.log10(lf[mf]), zf[mf])
    xx = np.linspace(zf[mf].min(), zf[mf].max(), 20)
    a2.plot(xx, 10 ** (kf * xx + cf), color="#7C2D12", lw=2.0, ls=(0, (2.5, 1.5)), zorder=4)
    a2.annotate("Batch 2 newstructure 9셀\n(충전 10.04분, 수명 777–1186)", xy=(-1.78, 1120), xytext=(-2.3, 1750),
                color=ps.BATCH_COLOR["batch2"], arrowprops=dict(arrowstyle="->", color=ps.BATCH_COLOR["batch2"]), **kw)
    fc = extra["b2_fast_ct"]
    a2.annotate(f"Batch 2 fastcharge 30셀 (10.17–10.22분, 수명 392–514)\n"
                f"집단 안에서도 ρ {fc['rho']:+.2f} (p={fc['p']:.4f}, 진한 점선) — 충전시간 폭 {fc['range_s']:.0f}초\n"
                f"이 폭의 {fc['between_policy_var_share']:.0%}는 프로토콜 {fc['n_policies']}개 사이 차이 → "
                f"프로토콜 평균 제거 시 ρ {fc['within_policy_spearman_demeaned']:+.2f} (p={fc['within_policy_p']:.2f})",
                xy=(0.55, 440), xytext=(-2.35, 250), va="bottom", bbox=box,
                color=ps.BATCH_COLOR["batch2"],
                arrowprops=dict(arrowstyle="->", color=ps.BATCH_COLOR["batch2"], relpos=(0.86, 1.0)), **kw)
    a2.set_ylim(235, 2250)            # 아래 주석 공간 확보 (데이터 최솟값 392)
    a2.annotate(f"Batch 3 4.8C(80%)-4.8C 6셀 (11.04분)\n→ Pearson(log) {extra['b3_ct_plog']:+.2f} 를 견인",
                xy=(2.4, 1650), xytext=(0.15, 1800), color=ps.BATCH_COLOR["batch3"],
                arrowprops=dict(arrowstyle="->", color=ps.BATCH_COLOR["batch3"]), **kw)
    # ③ Batch 3 3.7C(31%)-5.9C 3셀 표시 (|z|≥3 은 그중 2셀)
    z3 = zcache[("batch3", "fade_slope_2_100")]
    d3 = L[L["batch"] == "batch3"]
    ring = d3["cell_key"].isin(extra["b3_slope_ring"]).to_numpy()
    a3.scatter(z3[ring], d3["cycle_life"][ring], s=150, facecolors="none", edgecolors="#111827", lw=1.2, zorder=5)
    fs = extra["b3_slope"]
    a3.text(0.02, 0.97,
            f"○ Batch 3 3.7C(31%)-5.9C 3셀 (급한 용량 감소)\n"
            f"Batch 3 Pearson(log) {fs['plog']:+.2f}\n"
            f"· |z|≥3 2셀({', '.join(k.replace('batch3-', '#') for k in fs['trim_keys'])}) 제외: {fs['plog_trim']:+.2f}\n"
            f"· 3셀 모두 제외: {fs['plog_excl3']:+.2f} (Spearman {fs['sp_excl3']:+.2f})",
            transform=a3.transAxes, va="top", color=ps.BATCH_COLOR["batch3"], linespacing=1.45, bbox=box, **kw)
    # ④ 등시간 충전
    a4.annotate("Batch 2·3: 전 셀 4.79–4.81C\n(0→80% 10분 등시간 충전)", xy=(4.79, 1200), xytext=(4.05, 1650),
                fontsize=9, arrowprops=dict(arrowstyle="->", color="#6B7280"), color="#374151")
    h, l = a1.get_legend_handles_labels()
    fig.legend(h, [ps.BATCH_LABEL[b] for b in BATCH_NAMES], loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.955),
               fontsize=10, markerscale=1.2)
    fig.suptitle("가장 강한 신호와 배치 간 일관성 — ΔQ 크기 신호(분산·|min|)만 세 배치에서 부호·강도 유지",
                 fontsize=14, fontweight="bold", y=0.995)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    ps.save(fig, "q5_top_signals")


def fig_heatmap(R: pd.DataFrame, order: list[str], groups: list[list[str]], thr: float) -> None:
    """엄격한 하삼각(i>j)만 표시: 행 = order[1:], 열 = order[:-1]."""
    R = R.loc[order, order]
    n = len(order)
    rows, cols = order[1:], order[:-1]
    A = R.loc[rows, cols].to_numpy()
    mask = np.array([[j > i for j in range(n - 1)] for i in range(n - 1)])     # 행 r=i+1, 열 j: j ≤ i 만 표시
    fig, ax = plt.subplots(figsize=(11.5, 9.6))
    im = ax.imshow(np.ma.array(A, mask=mask), cmap="RdBu_r", vmin=-1, vmax=1)
    for i in range(n - 1):
        for j in range(i + 1):
            v = A[i, j]
            ax.text(j, i, f"{v:.2f}".replace("0.", ".").replace("-.", "−."), ha="center", va="center",
                    fontsize=7.4, color="white" if abs(v) >= 0.6 else "#1F2937",
                    fontweight="bold" if abs(v) >= thr else "normal")
    ax.set_xticks(range(n - 1))
    ax.set_yticks(range(n - 1))
    ax.set_xticklabels([LBL[f] for f in cols], rotation=55, ha="right", fontsize=9.2)
    ax.set_yticklabels([LBL[f] for f in rows], fontsize=9.2)
    for t, f in zip(ax.get_xticklabels(), cols):
        t.set_color(GROUP_COLOR[GRP[f]])
    for t, f in zip(ax.get_yticklabels(), rows):
        t.set_color(GROUP_COLOR[GRP[f]])
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)
    # 중복 그룹 윤곽 (평균 |ρ| ≥ thr 군집) — 하삼각 계단 모양. 원래 인덱스 (r, c) → 화면 (x=c, y=r−1)
    pos = {f: i for i, f in enumerate(order)}
    for k, g in enumerate(groups):
        idx = sorted(pos[f] for f in g)
        a, b = idx[0], idx[-1]
        pts = [(a - .5, a - .5), (a - .5, b - .5), (b - .5, b - .5)]
        for q in range(b - 1, a - 1, -1):
            pts += [(q + .5, q - .5), (q - .5, q - .5)]
        ax.add_patch(plt.Polygon(pts, closed=True, fill=False, ec="black", lw=2.0, zorder=5))
        ax.text(a + 0.6, a - 0.6, f"G{k + 1}", fontsize=10.5, fontweight="bold", va="bottom", ha="left")
    cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02)
    cb.set_label("Spearman ρ (Batch 1, n=36)")
    ax.set_title(f"Batch 1 초기 피처 간 상관 (Spearman) — 굵은 숫자 = |ρ| ≥ {thr}, 윤곽 G1–G{len(groups)} = 평균 |ρ| ≥ {thr} 군집",
                 fontsize=12.5, pad=28)
    handles = [Patch(color=c, label=g) for g, c in GROUP_COLOR.items()]
    ax.legend(handles=handles, loc="upper right", bbox_to_anchor=(1.0, 1.0), fontsize=9.5, title="라벨 색 = 도메인 그룹",
              title_fontsize=9.5)
    ps.save(fig, "q5_feature_corr_heatmap")


def fig_vif(v_before: pd.Series, v_rep: pd.Series, v_after: pd.Series, step1: dict, dropped: list[dict],
            readmit: list[dict], R: pd.DataFrame, v_m2: pd.Series, protect: str = "dQ_logvar") -> None:
    n_max = len(v_before)
    fig, axes = plt.subplots(1, 4, figsize=(23, 6.8), gridspec_kw={"wspace": 0.66})
    panels = ((axes[0], v_before, f"① 주요 후보 {len(v_before)}개\n(원논문식 피처 + 프로토콜)"),
              (axes[1], v_rep, f"② 중복 그룹(|ρ|≥0.8)마다 대표 1개\n(Batch 1 |ρ(life)| 최대) → {len(v_rep)}개"),
              (axes[2], v_after, f"③ VIF>5 순차 제거 + 빈 그룹 대표 재편입\n→ {len(v_after)}개"),
              (axes[3], v_m2, f"④ DAY 2 M2 Elastic-Net 풀 {len(v_m2)}개\n(측정 정의 통일: CC 끝점·실측 충전시간)"))
    for ax, v, title in panels:
        v = v.sort_values()
        off = n_max - len(v)                                    # 막대 두께를 패널 간 통일
        yy = np.arange(len(v)) + off
        colors = ["#DC2626" if x >= 10 else ("#F59E0B" if x >= 5 else "#10B981") for x in v]
        ax.barh(yy, v.clip(upper=1e5), color=colors, height=0.65)
        ax.set_xscale("log")
        ax.set_xlim(0.8, 2e5)
        ax.set_ylim(-0.6, n_max - 0.4)
        ax.axvline(5, color="#F59E0B", ls="--", lw=1)
        ax.axvline(10, color="#DC2626", ls="--", lw=1)
        ax.set_yticks(yy)
        ax.set_yticklabels([LBL[f] for f in v.index], fontsize=9.5)
        for t, f in zip(ax.get_yticklabels(), v.index):
            t.set_color(GROUP_COLOR[GRP[f]])
        for i, x in zip(yy, v):     # 값 라벨은 점선(5·10) 오른쪽 열에 두어 겹치지 않게
            ax.text(max(x * 1.18, 13.5), i, f"{x:,.1f}" if x < 100 else f"{x:,.0f}", va="center", fontsize=8.8)
        ax.set_title(title, fontsize=11.2, loc="left")
        ax.set_xlabel("VIF (로그 축) — 5: 주의 · 10: 심각")
        ax.grid(axis="y", visible=False)
    # 빈 공간에 제거 내역 기록
    by_keep: dict[str, list[str]] = {}
    for f, k in step1.items():
        by_keep.setdefault(k, []).append(f)
    lines = ["1단계에서 뺀 피처 → 남긴 대표 (Spearman ρ)"]
    for k, fs in by_keep.items():
        for f in fs:
            lines.append(f"· {LBL[f]} → {LBL[k]} ({R.loc[f, k]:+.2f})")
    axes[1].text(0.03, 0.02, "\n".join(lines), transform=axes[1].transAxes, fontsize=8.6, va="bottom", ha="left",
                 linespacing=1.45, bbox=dict(boxstyle="round,pad=0.5", fc="#F9FAFB", ec="#D1D5DB"))
    lines = ["2단계에서 뺀 피처 (VIF)"]
    for d in dropped:
        f = d["feature"]
        partners = R.loc[f].drop(index=f).loc[[c for c in v_rep.index if c != f]].abs().sort_values(ascending=False)[:2]
        lines.append(f"· {LBL[f]} — VIF {d['vif']:.1f}")
        lines.append("   겹치는 피처(ρ):")
        lines += [f"   – {LBL[c]} ({R.loc[f, c]:+.2f})" for c in partners.index]
    for d in readmit:
        lines.append("3단계: 대표가 빠진 그룹에 차순위 재편입")
        lines.append(f"· + {LBL[d['feature']]} (재편입 후 최대 VIF {d['max_vif_unprotected']:.1f})")
    vu = v_after.drop(index=protect, errors="ignore")
    lines.append(f"→ {len(v_after)}개: 최대 VIF {vu.max():.1f}")
    lines.append(f"   (보호 피처 ΔQ 분산 {v_after.get(protect, np.nan):.1f})")
    axes[2].text(0.03, 0.02, "\n".join(lines), transform=axes[2].transAxes, fontsize=8.6, va="bottom", ha="left",
                 linespacing=1.45, bbox=dict(boxstyle="round,pad=0.5", fc="#F9FAFB", ec="#D1D5DB"))
    lines = ["③ 대비 대표 교체 (측정 정의 통일)",
             "· QD(2) → Qcc_init (CC 끝점, CV 방전분 제외)",
             "· max QD − QD(2) → CC break-in 상승폭",
             "· 평균 C-rate(정책값) → 실측 충전시간",
             "· 기울기 91–100·왜도·첨도·온도·IR 유지",
             f"→ 최대 VIF {v_m2.max():.1f} ({LBL[v_m2.idxmax()]})"]
    axes[3].text(0.03, 0.02, "\n".join(lines), transform=axes[3].transAxes, fontsize=8.6, va="bottom", ha="left",
                 linespacing=1.45, bbox=dict(boxstyle="round,pad=0.5", fc="#F9FAFB", ec="#D1D5DB"))
    handles = [Patch(color="#10B981", label="VIF < 5"), Patch(color="#F59E0B", label="5 ≤ VIF < 10"),
               Patch(color="#DC2626", label="VIF ≥ 10")]
    fig.legend(handles=handles, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.04), fontsize=10)
    fig.suptitle(f"Batch 1 다중공선성 (VIF, n=36) — 후보 {n_max}개 중 {int((v_before >= 10).sum())}개가 VIF≥10 → 그룹 대표 + 가지치기 후 "
                 f"최대 {vu.max():.1f} (ΔQ 분산 {v_after.get(protect, np.nan):.1f}) · DAY 2 M2 풀 최대 {v_m2.max():.1f}",
                 fontsize=14, fontweight="bold", y=1.03)
    ps.save(fig, "q5_vif")


# ════════════════════════════════════════════════════════════════════════
def main() -> dict:
    df = build_table()
    df.to_csv(OUT / "early_features.csv", index=False)
    L = df[df["labeled"]].copy()

    # ── (a) 배치별 상관 ─────────────────────────────────────────────
    rows = []
    sd1 = L.loc[L["batch"] == "batch1", FEATS].std(ddof=1)
    for b in BATCH_NAMES:
        d = L[L["batch"] == b]
        for f in FEATS:
            rows.append({"batch": b, "feature": f, **corr_row(d[f], d["cycle_life"], sd1[f])})
    C = pd.DataFrame(rows).set_index(["batch", "feature"])
    n1 = int((L["batch"] == "batch1").sum())
    rc = r_crit(n1)

    rng = np.random.default_rng(RANDOM_STATE)
    d1 = L[L["batch"] == "batch1"]
    ci = {f: boot_ci(d1[f].to_numpy(float), d1["cycle_life"].to_numpy(float), rng) for f in FEATS}

    order = sorted(FEATS, key=lambda f: -abs(C.loc[("batch1", f), "spearman"]))

    # ── (b) 가장 강한 관계 + 배치 일관성 ─────────────────────────────
    cons = {}
    for f in FEATS:
        r1 = C.loc[("batch1", f), "spearman"]
        cons[f] = consistency(r1, {b: C.loc[(b, f), "spearman"] for b in ("batch2", "batch3")},
                              {b: bool(C.loc[(b, f), "constant"]) for b in ("batch2", "batch3")}, rc)
    sig_b1 = [f for f in order if abs(C.loc[("batch1", f), "spearman"]) >= rc]
    # 다중비교 보정 (Batch 1, 19개 피처 동시 검정)
    pv = C.loc["batch1"].loc[FEATS, "p_spearman"]
    bonf = [f for f in order if pv[f] < 0.05 / len(FEATS)]
    rk = pv.rank(method="first")
    q = (pv * len(FEATS) / rk)[pv.sort_values(ascending=False).index].cummin().clip(upper=1)  # Benjamini–Hochberg
    bh = [f for f in order if q[f] < 0.05]

    # Batch 2 하위집단(혼합 효과 점검): fastcharge 30 / newstructure 9
    sub = {}
    for g in ("fastcharge", "newstructure"):
        d = L[(L["batch"] == "batch2") & (L["group"] == g)]
        sub[g] = {"n": int(len(d)), "life_min": float(d["cycle_life"].min()), "life_max": float(d["cycle_life"].max()),
                  "spearman": {f: float(stats.spearmanr(d[f], d["cycle_life"], nan_policy="omit")[0]) for f in FEATS}}

    # 전체 풀링(참고: 원논문 Fig.2 ρ=−0.93 과 비교용)
    pooled = {f: {"spearman": float(stats.spearmanr(L[f], L["cycle_life"], nan_policy="omit")[0]),
                  "pearson_log10life": float(stats.pearsonr(L[f], np.log10(L["cycle_life"]))[0])}
              for f in ["dQ_logvar", "dQ_logabsmin"]}

    # 공변량 이동: Batch 2/3 의 ΔQ 분산이 Batch 1 값 범위 밖에 있는 비율 (라벨 안 씀)
    lo, hi = d1["dQ_logvar"].min(), d1["dQ_logvar"].max()
    shift = {}
    for b in BATCH_NAMES:
        x = L.loc[L["batch"] == b, "dQ_logvar"]
        shift[b] = {"median": float(x.median()), "min": float(x.min()), "max": float(x.max()),
                    "frac_below_b1_min": float((x < lo).mean()), "frac_above_b1_max": float((x > hi).mean())}
    # 배치별 기술적 기울기: log10(cycle_life) ~ dQ_logvar (각 배치 내부 OLS, 기술통계)
    slopes = {}
    for b in BATCH_NAMES:
        d = L[L["batch"] == b]
        k, c0 = np.polyfit(d["dQ_logvar"], np.log10(d["cycle_life"]), 1)
        slopes[b] = {"slope": float(k), "intercept": float(c0),
                     "life_at_batch1_median_x": float(10 ** (k * d1["dQ_logvar"].median() + c0)),
                     "note": "배치 전체 OLS. Batch 2 는 두 집단 사이 빈 구간을 잇는 선이라 수준 비교에 쓰지 않음"}
    # Batch 2 는 두 집단(newstructure −4.45~−4.10 / fastcharge −3.71~−3.09)이 x축에서 분리 → 집단 내부 기울기
    for g in ("fastcharge", "newstructure"):
        d = L[(L["batch"] == "batch2") & (L["group"] == g)]
        k, c0 = np.polyfit(d["dQ_logvar"], np.log10(d["cycle_life"]), 1)
        slopes[f"batch2_{g}"] = {"slope": float(k), "intercept": float(c0), "n": int(len(d)),
                                 "x_min": float(d["dQ_logvar"].min()), "x_max": float(d["dQ_logvar"].max())}
    med1 = float(d1["dQ_logvar"].median())
    near_b1_median = {b: int(((L.loc[L["batch"] == b, "dQ_logvar"] - med1).abs() <= 0.25).sum()) for b in BATCH_NAMES}
    # 수준 비교 ① Batch 1 기술 회귀선(log–log) 대비 수명비 = 실제 수명 / B1 선이 주는 수명 (배치·집단별 중앙값)
    k1, c1 = slopes["batch1"]["slope"], slopes["batch1"]["intercept"]
    ratio_all = L["cycle_life"] / 10 ** (k1 * L["dQ_logvar"] + c1)
    in_b1 = (L["dQ_logvar"] >= lo) & (L["dQ_logvar"] <= hi)
    ratio = {}
    for key, m in (("batch1", L["batch"] == "batch1"), ("batch3", L["batch"] == "batch3"),
                   ("b2_fastcharge", (L["batch"] == "batch2") & (L["group"] == "fastcharge")),
                   ("b2_newstructure", (L["batch"] == "batch2") & (L["group"] == "newstructure"))):
        ratio[key] = {"median_all": float(ratio_all[m].median()), "n_all": int(m.sum()),
                      "median_in_b1_range": float(ratio_all[m & in_b1].median()) if (m & in_b1).any() else None,
                      "n_in_b1_range": int((m & in_b1).sum())}
    # 수준 비교 ② 같은 ΔQ 분산 창(B1 범위 ∩ B2 fastcharge 범위)에서 수명 중앙값 직접 비교
    fcm = (L["batch"] == "batch2") & (L["group"] == "fastcharge")
    wlo, whi = max(lo, L.loc[fcm, "dQ_logvar"].min()), min(hi, L.loc[fcm, "dQ_logvar"].max())
    w1 = d1[(d1["dQ_logvar"] >= wlo) & (d1["dQ_logvar"] <= whi)]
    w2 = L[fcm & (L["dQ_logvar"] >= wlo) & (L["dQ_logvar"] <= whi)]
    window = {"lo": float(wlo), "hi": float(whi),
              "b1_n": int(len(w1)), "b1_median": float(w1["cycle_life"].median()),
              "b1_life_range": [float(w1["cycle_life"].min()), float(w1["cycle_life"].max())],
              "b1_median_x": float(w1["dQ_logvar"].median()),
              "b2_n": int(len(w2)), "b2_median": float(w2["cycle_life"].median()),
              "b2_life_range": [float(w2["cycle_life"].min()), float(w2["cycle_life"].max())],
              "b2_median_x": float(w2["dQ_logvar"].median()),
              "rel_diff": float(w2["cycle_life"].median() / w1["cycle_life"].median() - 1),
              "mannwhitney_p": float(stats.mannwhitneyu(w1["cycle_life"], w2["cycle_life"]).pvalue)}

    # 평균 충전시간: Batch 2 fastcharge 내부에서도 음의 상관이 남는지
    dfc = L[fcm]
    r_, p_ = stats.spearmanr(dfc["chargetime_avg5"], dfc["cycle_life"])
    b2_fast_ct = {"rho": float(r_), "p": float(p_), "n": int(len(dfc)),
                  "ct_min": float(dfc["chargetime_avg5"].min()), "ct_max": float(dfc["chargetime_avg5"].max()),
                  "range_s": float((dfc["chargetime_avg5"].max() - dfc["chargetime_avg5"].min()) * 60)}
    # 그 3초 폭 중 프로토콜(9개) 사이 몫 vs 프로토콜 안 몫 — 설계 t80은 모두 10분이지만 단계 구성은 다름
    gfc = dfc.groupby("policy")
    ct_, ylog_ = dfc["chargetime_avg5"], np.log10(dfc["cycle_life"])
    ct_pm = gfc["chargetime_avg5"].transform("mean")
    y_pm = ylog_.groupby(dfc["policy"]).transform("mean")
    r_w, p_w = stats.spearmanr(ct_ - ct_pm, ylog_ - y_pm)                     # 프로토콜 평균 제거(양쪽)
    pm = pd.DataFrame({"ct": ct_, "y": ylog_, "policy": dfc["policy"]}).groupby("policy").mean()
    r_pm, p_pm = stats.spearmanr(pm["ct"], pm["y"])                           # 프로토콜 평균끼리
    b2_fast_ct.update({
        "n_policies": int(gfc.ngroups),
        "between_policy_var_share": float(((ct_pm - ct_.mean()) ** 2).sum() / ((ct_ - ct_.mean()) ** 2).sum()),
        "kruskal_p": float(stats.kruskal(*[x["chargetime_avg5"].to_numpy() for _, x in gfc]).pvalue),
        "life_between_policy_var_share_log": float(((y_pm - ylog_.mean()) ** 2).sum() / ((ylog_ - ylog_.mean()) ** 2).sum()),
        "within_policy_spearman_demeaned": float(r_w), "within_policy_p": float(p_w),
        "policy_mean_spearman": float(r_pm), "policy_mean_p": float(p_pm),
        "policy_mean_ct": gfc["chargetime_avg5"].mean().sort_values().round(3).to_dict()})

    # 용량 감소 기울기(2–100): Batch 3 영향점 점검 — |z|≥3 셀 vs 3.7C(31%)-5.9C 프로토콜 3셀
    d3 = L[L["batch"] == "batch3"]
    z3 = _zscore(d3["fade_slope_2_100"])
    trim_keys = d3.loc[z3.abs() >= 3, "cell_key"].tolist()
    ring_keys = d3.loc[d3["policy"].str.startswith("3.7C(31%)-5.9C"), "cell_key"].tolist()
    k3 = ~d3["cell_key"].isin(ring_keys)
    b3_slope = {"plog": float(C.loc[("batch3", "fade_slope_2_100"), "pearson_log10life"]),
                "plog_trim": float(C.loc[("batch3", "fade_slope_2_100"), "pearson_log10life_trim3z"]),
                "trim_keys": trim_keys, "trim_z": {k: float(z3[d3["cell_key"] == k].iloc[0]) for k in trim_keys},
                "ring_keys": ring_keys, "ring_z": {k: float(z3[d3["cell_key"] == k].iloc[0]) for k in ring_keys},
                "plog_excl3": float(stats.pearsonr(d3.loc[k3, "fade_slope_2_100"], np.log10(d3.loc[k3, "cycle_life"]))[0]),
                "sp_excl3": float(stats.spearmanr(d3.loc[k3, "fade_slope_2_100"], d3.loc[k3, "cycle_life"])[0])}

    # 타깃 척도 점검(Batch 1 만): 단일 피처 기술 회귀의 잔차 이분산 — 원척도 vs log₁₀ 척도
    def _bp(resid: np.ndarray, x: np.ndarray) -> float:
        """Breusch–Pagan LM 검정 p값(설명변수 1개): e² 를 x 에 회귀한 R² × n ~ χ²(1)."""
        e2 = resid ** 2
        kk, cc = np.polyfit(x, e2, 1)
        r2 = 1 - np.sum((e2 - (kk * x + cc)) ** 2) / np.sum((e2 - e2.mean()) ** 2)
        return float(stats.chi2.sf(len(x) * r2, 1))
    xb, yb = d1["dQ_logvar"].to_numpy(float), d1["cycle_life"].to_numpy(float)
    target_diag = {"skew_life": float(stats.skew(yb, bias=False)), "skew_log10life": float(stats.skew(np.log10(yb), bias=False))}
    for nm, t in (("raw", yb), ("log10", np.log10(yb))):
        kk, cc = np.polyfit(xb, t, 1)
        fit = kk * xb + cc
        e = t - fit
        rs = stats.spearmanr(np.abs(e), fit)
        target_diag[nm] = {"spearman_absresid_vs_fitted": float(rs[0]), "p": float(rs[1]),
                           "breusch_pagan_p": _bp(e, xb), "shapiro_p": float(stats.shapiro(e).pvalue)}

    # 절대 수준 피처의 배치 오프셋 점검(라벨 미사용): QD(2) 배치별 중앙값
    qd2_median = {b: float(L.loc[L["batch"] == b, "QD_2"].median()) for b in BATCH_NAMES}

    # ── (c) 다중공선성 ──────────────────────────────────────────────
    X1 = d1[FEATS]
    R = X1.corr(method="spearman")
    D = 1 - R.abs().to_numpy()
    np.fill_diagonal(D, 0)
    Zl = hierarchy.linkage(squareform(D, checks=False), method="average")
    hm_order = [FEATS[i] for i in hierarchy.leaves_list(Zl)]
    THR = 0.8
    cl = hierarchy.fcluster(Zl, t=1 - THR, criterion="distance")       # 평균 |ρ| ≥ 0.8 로 묶인 군집
    groups = []
    for k in np.unique(cl):
        g = [f for f, c in zip(FEATS, cl) if c == k]
        if len(g) > 1:
            groups.append(sorted(g, key=hm_order.index))
    groups = sorted(groups, key=lambda g: hm_order.index(g[0]))
    pairs = [(a, b, float(R.loc[a, b])) for i, a in enumerate(FEATS) for b in FEATS[i + 1:]
             if abs(R.loc[a, b]) >= THR]
    pairs = sorted(pairs, key=lambda t: -abs(t[2]))
    pearson_pairs = {f"{a}~{b}": float(X1[a].corr(X1[b])) for a, b, _ in pairs}

    # 주요 후보 = 원논문식 피처(Severson 2019 SI) + 프로토콜. (Tmin_min·IR_2 는 같은 그룹 대표와 중복이라 제외)
    main_cands = ["dQ_logvar", "dQ_logabsmin", "dQ_skew", "dQ_kurt", "QD_2", "QD_max_minus_2",
                  "fade_slope_2_100", "fade_int_2_100", "fade_slope_91_100", "fade_int_91_100",
                  "chargetime_avg5", "Tmax_max", "Tavg_mean", "IR_min", "IR_diff", "avgC_80", "t80_min"]
    rho1 = {f: abs(C.loc[("batch1", f), "spearman"]) for f in FEATS}
    # 1단계: |ρ|≥0.8 중복 그룹마다 Batch 1 |ρ(life)| 최대인 피처 1개만 남김
    drop_step1 = {}
    for g in groups:
        keep = max(g, key=lambda f: (round(rho1[f], 6), -FEATS.index(f)))   # 동률이면 FEATS 앞쪽(avgC_80)
        for f in g:
            if f != keep:
                drop_step1[f] = keep
    rep = [f for f in main_cands if f not in drop_step1]
    # 2단계: VIF > 5 가 남으면 VIF 최대 피처를 하나씩 제거 (최강 신호 dQ_logvar 는 보호)
    reduced, drop_step2 = list(rep), []
    while True:
        v = vif_table(X1[reduced])
        cand = v.drop(index="dQ_logvar", errors="ignore")
        if cand.max() <= VIF_MAX:
            break
        worst = cand.idxmax()
        drop_step2.append({"feature": worst, "vif": float(cand.max())})
        reduced.remove(worst)
    # 3단계: 2단계에서 대표가 빠져 아무 피처도 남지 않은 그룹에, 1단계에서 뺀 차순위 피처를 재편입.
    #        재편입 후 보호 피처 외 모든 VIF ≤ 5 일 때만 받아들임 (그룹 정보를 통째로 잃지 않기 위함)
    readmit = []
    for g in groups:
        if any(f in reduced for f in g):
            continue
        for f in sorted([f for f in g if f in main_cands and f not in [d["feature"] for d in drop_step2]],
                        key=lambda f: -rho1[f]):
            v = vif_table(X1[reduced + [f]])
            vu = v.drop(index="dQ_logvar", errors="ignore")
            if vu.max() <= VIF_MAX:
                reduced.append(f)
                readmit.append({"feature": f, "group": g, "max_vif_unprotected": float(vu.max()),
                                "vif_protected_dQ_logvar": float(v.get("dQ_logvar", np.nan)),
                                "batch1_spearman": float(C.loc[("batch1", f), "spearman"]),
                                "batch1_p": float(C.loc[("batch1", f), "p_spearman"])})
                break
    v_before = vif_table(X1[main_cands])
    v_rep = vif_table(X1[rep])
    v_after = vif_table(X1[reduced])
    v_step2 = vif_table(X1[[f for f in reduced if f not in [d["feature"] for d in readmit]]])
    # 그룹 내부 겹침 정도: Spearman 과 Pearson 평균 |r| (정의상 중복 vs 통계적 겹침 구분용)
    group_overlap = []
    for g in groups:
        pr = [(a, b) for i, a in enumerate(g) for b in g[i + 1:]]
        group_overlap.append({"group": g, "mean_abs_spearman": float(np.mean([abs(R.loc[a, b]) for a, b in pr])),
                              "mean_abs_pearson": float(np.mean([abs(X1[a].corr(X1[b])) for a, b in pr]))})
    v_all = vif_table(X1[FEATS])
    paper_sets = {   # 참고: 원논문 피처 조합 자체의 공선성
        "paper_discharge": ["dQ_logvar", "dQ_logabsmin", "dQ_skew", "dQ_kurt", "QD_2", "QD_max_minus_2"],
        "paper_full": ["dQ_logvar", "dQ_logabsmin", "fade_slope_2_100", "fade_int_2_100", "QD_2",
                       "chargetime_avg5", "Tavg_mean", "IR_min", "IR_diff"]}
    v_paper = {k: vif_table(X1[v]).round(2).to_dict() for k, v in paper_sets.items()}
    # ④ 측정 정의를 통일한 DAY 2 M2 Elastic-Net 풀 (전략 문서와 같은 10개)
    v_m2 = vif_table(d1[M2_POOL])
    v_m2_dis = vif_table(d1[M2_POOL[:5]])
    m2_pairs = {f"{a}~{b}": float(d1[a].corr(d1[b], method="spearman"))
                for i, a in enumerate(M2_POOL) for b in M2_POOL[i + 1:]}
    m2_top_pairs = dict(sorted(m2_pairs.items(), key=lambda t: -abs(t[1]))[:5])
    # 공통 표기 점검: 용량 기울기 Theil–Sen 판의 Spearman (참고)
    ts_ref = {b: float(stats.spearmanr(L.loc[L["batch"] == b, "fade_slope_2_100_ts"],
                                       L.loc[L["batch"] == b, "cycle_life"])[0]) for b in BATCH_NAMES}
    ts_ref["r_with_ols_batch1"] = float(d1["fade_slope_2_100_ts"].corr(d1["fade_slope_2_100"]))

    # ── 그림 ───────────────────────────────────────────────────────
    fig_ranking(C, order, ci, rc, single=True)
    fig_ranking(C, order, ci, rc, single=False)
    fig_top_signals(L, C, [
        ("dQ_logvar", "raw", "① 최강 신호: log₁₀ Var(ΔQ₁₀₀₋₁₀) — 세 배치 모두 음(−)의 상관"),
        ("chargetime_avg5", "z", "② 평균 충전시간(cycle 2–6) — Batch 2 에서 부호 반전"),
        ("fade_slope_2_100", "z", "③ 용량 기울기(2–100, OLS, −=감소) — Batch 2·3 에서 약화/반전"),
        ("avgC_80", "raw", "④ 평균 C-rate(0→80%) — Batch 2·3 에서는 변동 자체가 없음"),
    ], {"b1_range_logvar": (lo, hi), "b1_line": (k1, c1),
        "b3_below": shift["batch3"]["frac_below_b1_min"],
        "ratio_to_b1_line": {"batch3": ratio["batch3"]["median_all"],
                             "b2_fastcharge": ratio["b2_fastcharge"]["median_all"],
                             "b2_newstructure": ratio["b2_newstructure"]["median_all"]},
        "b2_window": window, "b2_fast_ct": b2_fast_ct,
        "b3_ct_plog": C.loc[("batch3", "chargetime_avg5"), "pearson_log10life"],
        "b3_slope": b3_slope, "b3_slope_ring": ring_keys})
    fig_heatmap(R, hm_order, groups, THR)
    fig_vif(v_before, v_rep, v_after, drop_step1, drop_step2, readmit, R, v_m2)

    # ── 결과 저장 ───────────────────────────────────────────────────
    corr_out = {b: {f: {k: (float(v) if isinstance(v, (float, np.floating)) else v)
                        for k, v in C.loc[(b, f)].to_dict().items()} for f in FEATS} for b in BATCH_NAMES}
    res = {
        "settings": {"cycles": [FIRST, LAST], "dq": "Qdlin[cycle100] − Qdlin[cycle10] (data.delta_q)",
                     "cells": "labeled only", "n": {b: int((L["batch"] == b).sum()) for b in BATCH_NAMES},
                     "constant_rule": f"sd(batch)/sd(batch1) < {SPREAD_CONST}", "vif_max": VIF_MAX, "n_boot": N_BOOT, "random_state": RANDOM_STATE,
                     "r_crit_batch1_alpha05": rc, "kurtosis": "Fisher excess(정규=0), scipy fisher=True",
                     "fade_slope": "mAh/100 cycle, 음수=감소, OLS(np.polyfit); fade_slope_2_100_ts = Theil–Sen 참고",
                     "vif_method": v_before.attrs.get("how")},
        "correlations": corr_out,
        "batch1_spearman_boot_ci95": {f: list(ci[f]) for f in FEATS},
        "batch1_rank_by_abs_spearman": order,
        "batch1_significant_features": sig_b1,
        "batch1_bonferroni_sig_features": bonf,
        "batch1_bh_q": {f: float(q[f]) for f in FEATS},
        "batch1_bh_sig_features_q05": bh,
        "consistency_spearman": cons,
        "batch2_subgroup_spearman": sub,
        "pooled_all_labeled": pooled,
        "dq_logvar_covariate_shift_vs_batch1": shift,
        "per_batch_slope_log10life_on_dQ_logvar": slopes,
        "batch1_median_dQ_logvar": med1,
        "n_cells_within_0.25_of_batch1_median": near_b1_median,
        "ratio_life_to_batch1_line": ratio,
        "b2_fastcharge_vs_batch1_same_window": window,
        "b2_fastcharge_chargetime_spearman": b2_fast_ct,
        "b3_fade_slope_influence": b3_slope,
        "batch1_target_scale_diagnostics": target_diag,
        "qd2_median_by_batch": qd2_median,
        "tavg_int_vs_mean_pearson_batch1": float(d1["Tavg_int"].corr(d1["Tavg_mean"])),
        "batch1_redundant_groups_avg_abs_rho_ge_0.8": groups,
        "batch1_group_overlap_spearman_vs_pearson": group_overlap,
        "batch1_pairs_abs_spearman_ge_0.8": [{"a": a, "b": b, "spearman": r, "pearson": pearson_pairs[f"{a}~{b}"]}
                                             for a, b, r in pairs],
        "batch1_feature_spearman_matrix": R.round(4).to_dict(),
        "vif_main_candidates": v_before.round(3).to_dict(),
        "step1_dropped_redundant_to_representative": drop_step1,
        "vif_after_step1_representatives": v_rep.round(3).to_dict(),
        "step2_dropped_by_vif": drop_step2,
        "vif_after_step2": v_step2.round(3).to_dict(),
        "step3_readmitted": readmit,
        "reduced_set": reduced,
        "vif_reduced_set": v_after.round(3).to_dict(),
        "vif_paper_feature_sets": v_paper,
        "vif_all_19": v_all.round(3).to_dict(),
        "vif_m2_pool": v_m2.round(3).to_dict(),
        "vif_m2_discharge_subpool": v_m2_dis.round(3).to_dict(),
        "m2_pool_top_abs_spearman_pairs": m2_top_pairs,
        "fade_slope_2_100_theilsen_spearman": ts_ref,
        "batch1_median_fade_slope_mAh_per_100": {k: float(L.loc[L["batch"] == b, k].median())
                                                 for b in ["batch1"] for k in ("fade_slope_2_100", "fade_slope_91_100")},
        "heatmap_order": hm_order,
    }
    with open(OUT / "q5_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
    return res


if __name__ == "__main__":
    r = main()
    pd.set_option("display.width", 200)
    print("n:", r["settings"]["n"], " r_crit(B1):", round(r["settings"]["r_crit_batch1_alpha05"], 3))
    print("rank:", r["batch1_rank_by_abs_spearman"][:6])
    for f in r["batch1_rank_by_abs_spearman"]:
        s = [round(r["correlations"][b][f]["spearman"], 2) for b in BATCH_NAMES]
        pl = [round(r["correlations"][b][f]["pearson_log10life"], 2) for b in BATCH_NAMES]
        print(f"{f:20s} sp={s} plog={pl} {r['consistency_spearman'][f]}")
    print("groups:", r["batch1_redundant_groups_avg_abs_rho_ge_0.8"])
    print("VIF before:", r["vif_main_candidates"])
    print("VIF after:", r["vif_reduced_set"])
    print("readmit:", r["step3_readmitted"])
    for k in ("per_batch_slope_log10life_on_dQ_logvar", "ratio_life_to_batch1_line", "b2_fastcharge_vs_batch1_same_window",
              "b2_fastcharge_chargetime_spearman", "b3_fade_slope_influence", "batch1_target_scale_diagnostics",
              "qd2_median_by_batch", "n_cells_within_0.25_of_batch1_median", "pooled_all_labeled",
              "batch1_group_overlap_spearman_vs_pearson", "vif_m2_pool", "vif_m2_discharge_subpool",
              "m2_pool_top_abs_spearman_pairs", "fade_slope_2_100_theilsen_spearman"):
        print(k, ":", r[k])
