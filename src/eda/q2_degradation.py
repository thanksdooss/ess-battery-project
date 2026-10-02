"""DAY 1 EDA — Question 2. 열화 곡선 - 방전 용량이 어떻게 감소하는가?

노션 하위 항목 3개를 순서대로 답하고, 전략 결정에 필요한 추가 점검 1개를 붙인다.
  (a) 사이클 별 Qd 추이 시각화
  (b) 열화 속도가 일정한가, 가속되는가?
  (c) Knee point - 급격한 열화 시작점 탐색
  (d) (추가) 초기 용량 수준(QD_init·Qcc_init)과 수명 — 노션 Batch 3 메모의 Qdlin 경고와 함께 점검

산출물
  reports/figures/q2_qd_curves.png    (a)(c) 배치별 QD-사이클 곡선(knee 표시) + 정규화 곡선(knee 위치 중앙값)  ← PDF
  reports/figures/q2_fade_rate.png    (b) 수명 구간별 용량 기울기, 첫 100 사이클 확대(B1 수명 3분위 + B2·B3 참고)  ← PDF
  reports/figures/q2_early_slope.png  (b) 보조: 초기(2~100) 용량 기울기 vs 수명(B2 는 fastcharge/newstructure 구분)
  reports/figures/q2_knee.png         (c) 보조: Knee·onset 검출 예시, knee·onset-수명 관계(구성상 상관 주의),
                                          knee/수명 비, knee 이후 잔여 사이클(수명 대비 비율 병기)
  results/eda/q2_results.json         수치 결과 전부 (셀 단위 표 포함)

방법 정의 (재현용, 모든 수치는 이 정의를 따른다)
  * 전처리: data.clean_summary → cycle ≥ 2 만 사용(Batch 1 cycle 1 은 더미, 배치 간 통일)
            → 라벨 셀은 cycle ≤ cycle_life 까지만(Batch 2 는 EOL 이후도 기록되어 있어 잘라 통일)
  * QD_s(평활 곡선): clean QD 결측 선형보간 → 중심 이동중앙값(창 9 사이클, min_periods=1).
                    단발성 측정 스파이크 제거용(B1 에서도 cycle 2~100 에 3 mAh 넘는 스파이크가 있음). 그림·knee 검출에 사용.
  * QD_init: clean QD 의 cycle 2~6 중앙값.   QD@c: clean QD 의 cycle c-2~c+2 중앙값.
  * ΔQD(100-10) = QD@100 − QD@10 (노션 ΔQ 정의 '사이클 100 − 사이클 10'과 같은 사이클 쌍).
  * 용량 기울기 [mAh/100 사이클, 음수 = 감소] = 1e5 × Theil–Sen 기울기(clean QD[Ah] ~ cycle) — 스파이크에 강건.
      (프로젝트 공통 규약: 단위 mAh/100 사이클, 음수 = 감소, 추정량 Theil–Sen)
      초기 = cycle 2~100(절대 사이클, 조기예측 창), 수명 구간 = cycle/cycle_life 의 10% 구간,
      중기 = 40~60%, 말기 = 80~100%.
  * 말기 20% 손실 비중 = (QD_s(0.8L) − QD_s(L)) / (QD_init − QD_s(L)). 선형 열화면 0.20.
  * 단일 사이클 잡음 σ — 두 추정량을 함께 보고(추정량에 따라 수치가 달라지므로):
      ① 고주파(사이클 간) σ = 1.4826·MAD(Δ²QD)/√6 (clean QD 2차 차분, cycle 2~100) — 백색잡음만 반영.
      ② 추세 잔차 σ = 1.4826·MAD(clean QD − 3차 다항 추세) (cycle 2~100) — 매끄러운 추세에서 벗어난 정도 전체.
  * 초기 용량 상승(break-in): 상승폭 = max(QD_s) − QD_init, 정점 사이클 = argmax,
      '초기값 위 지속' = 정점 이후 QD_s/QD_init < 1 이 처음 되는 사이클 / cycle_life.
  * 수명 배율 보정 말기 기울기 = 말기 기울기 × cycle_life / 100 (mAh per 수명 단위) — 수명이 짧아 생기는 절대 차이 제거.
  * Knee(주 방법): Bacon–Watts 모형의 γ→0 극한(Fermín-Cueto et al., 2020, Energy and AI 1:100006)
      QD_s = α0 + α1(x−x1) + α2|x−x1| (연속 2구간 선형), x1 을 [첫 사이클+10, 마지막−10] 정수 격자에서
      SSE 최소가 되도록 선택. knee 이전 기울기 = α1−α2, 이후 = α1+α2.
      주의: knee 를 각 셀의 [12, cycle_life−10] 안에서 찾으므로 knee–cycle_life 상관은 구성상 높다
      (B1 knee/L 비를 셀 간 무작위로 섞는 순열 점검으로 그 기준선을 함께 보고).
  * Knee-onset(보조, 노션 '급격한 열화 시작점'): 이중 Bacon–Watts(γ→0) = 연속 3구간 선형회귀
      QD_s = β0 + β1·x + β2·(x−a)+ + β3·(x−b)+, a<b 를 정수 격자 전탐색(각 구간 ≥10 사이클). onset = a.
  * Knee(교차 확인): Kneedle(Satopää et al., 2011) — cycle·QD_s 를 [0,1] 정규화 후
      시작–끝 현(chord)에서 가장 멀리 떨어진 점 argmax(y_n − (1 − x_n)).
  * Knee 뚜렷도 = knee 이후/이전 기울기 비. < 5 이면 'knee 불명확(준선형 열화)'로 표시.
  * 장수명/단수명: 노션 Q1 정의 그대로 장수명 > 1,000, 단수명 < 500 (그 사이 = 중간).
      단수명(<500) 라벨 셀은 전부 Batch 2 이고 Batch 1 에는 >1,000 이 5셀, <500 이 0셀이므로,
      학습셋 내부 비교는 Batch 1 수명 3분위(각 12셀)로 한다. 노션 그룹 비교는 배치별로 층화해 보고.
  * 초기 빠른 열화 표시: 초기 기울기 < −9 mAh/100 사이클(Batch 1 중앙값 −1.55 의 약 6배).
  * (d) 초기 용량 수준 점검 (fastcharge·newstructure 셀, 라벨 무관 분포 + Batch 1 라벨 36셀 상관):
      Qcc_init = cycle 2~6 의 data.qdlin(cell,k)[-1] (2.0 V 도달 시점의 CC 방전용량, 원시 Qdlin 의 절대 수준) 중앙값,
                 ≤0 또는 >1.2 Ah 값은 NaN.   QD_2 = clean QD 의 cycle 2 값(summary 기준).
      dQ_logvar = log10(var(ΔQ_{100-10}(V))), ddof=0 (Q3 주 피처와 같은 정의).
      편상관 = log10(cycle_life) 와 피처를 각각 통제변수에 OLS 회귀한 잔차의 Pearson r (t 검정, df = n−2−k).
      배치 이동 = 각 배치·프로토콜 그룹 중앙값 − Batch 1 전체 46셀 중앙값 (라벨 미사용).
      예측 배율 = 10^(β × 이동), β = Batch 1 라벨 36셀 OLS log10(L) ~ dQ_logvar + Qcc_init 의 Qcc_init 계수(기술 회귀).
      헤드룸 단순 환산 = +10 mAh / |EOL 직전 25 사이클 Theil–Sen 기울기| / cycle_life (곡선 모양이 같고 수직 이동만 있다고 가정).

실행: cd ess-battery-project && python src/eda/q2_degradation.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
from data import *  # noqa: E402,F401,F403
from data import BATCH_NAMES, EOL_AH, LABEL_THRESHOLD, RANDOM_STATE, cell_table, clean_summary, load_all  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from scipy import stats  # noqa: E402

import plot_style as ps  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
OUT_JSON = ROOT / "results" / "eda" / "q2_results.json"
OUT_JSON.parent.mkdir(parents=True, exist_ok=True)

LONG_TH, SHORT_TH = 1000, 500          # 노션 Q1: 장수명(>1,000) / 단수명(<500)
SMOOTH_WIN = 9                          # 이동중앙값 창(사이클)
KNEE_MARGIN = 10                        # knee 후보에서 양 끝 제외 사이클 수
ACC_FLAG = 5.0                          # knee 이후/이전 기울기 비 < 5 → knee 불명확
FAST_EARLY = -9.0                       # 초기(2~100) 기울기 < −9 mAh/100cyc → '초기부터 빠른 준선형 열화' 표시
SLOPE_UNIT = "mAh/100 사이클, 음수=감소"   # 프로젝트 공통 기울기 규약
DIP_INFO: dict = {}                     # 배치 공통 일시 하락 점검 결과(그림 주석용)
DECILES = np.round(np.arange(0, 1.0001, 0.1), 2)
BS = {"batch1": "B1", "batch2": "B2", "batch3": "B3"}
TERTILES = ["하위 1/3", "중위 1/3", "상위 1/3"]   # Batch 1 수명 3분위 (짧음 → 김)
TER_COL = {"하위 1/3": ps.SHORT, "중위 1/3": "#6b6b6b", "상위 1/3": ps.LONG}
MARK = {"batch1": "o", "batch2": "s", "batch3": "^"}

# cycle_life 색: 단일 색상(파랑) 순차 램프, 밝음=단수명 → 진함=장수명 (ps.LONG 계열과 의미 일치)
LIFE_CMAP = mcolors.LinearSegmentedColormap.from_list(
    "life_blue", ["#9ec5f4", "#5598e7", "#2a78d6", "#1c5cab", "#104281", "#0a2a55"])
LIFE_NORM = mcolors.LogNorm(vmin=380, vmax=2000)


# ───────────────────────── 곡선 처리 ─────────────────────────
def qd_series(c: dict, upto: float | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(cycle, clean QD, QD_s). cycle ≥ 2, upto 가 있으면 cycle ≤ upto."""
    s = clean_summary(c["summary"])
    s = s[s["cycle"] >= 2]
    if upto is not None and np.isfinite(upto):
        s = s[s["cycle"] <= upto]
    q = s["QD"].reset_index(drop=True)
    qs = q.interpolate(limit_direction="both").rolling(SMOOTH_WIN, center=True, min_periods=1).median()
    return s["cycle"].to_numpy(float), q.to_numpy(float), qs.to_numpy(float)


def qd_at(c: dict, cyc: int, half: int = 2) -> float:
    s = clean_summary(c["summary"])
    m = (s["cycle"] >= cyc - half) & (s["cycle"] <= cyc + half)
    return float(np.nanmedian(s.loc[m, "QD"]))


def fade_slope(x: np.ndarray, q: np.ndarray, mask: np.ndarray) -> float:
    """용량 기울기 [mAh/100 사이클, 음수 = 감소]. Theil–Sen 추정."""
    m = mask & np.isfinite(q)
    if m.sum() < 5:
        return np.nan
    return float(1e5 * stats.theilslopes(q[m], x[m])[0])


def bacon_watts_knee(x: np.ndarray, y: np.ndarray, margin: int = KNEE_MARGIN) -> dict:
    """Bacon–Watts(γ→0) = 연속 2구간 선형회귀, 전격자 탐색."""
    cand = x[(x >= x[0] + margin) & (x <= x[-1] - margin)]
    best = (np.inf, np.nan, None)
    for x1 in cand:
        A = np.column_stack([np.ones_like(x), x - x1, np.abs(x - x1)])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        sse = float(np.sum((A @ coef - y) ** 2))
        if sse < best[0]:
            best = (sse, float(x1), coef)
    sse, x1, coef = best
    sse_lin = float(np.sum((np.polyval(np.polyfit(x, y, 1), x) - y) ** 2))
    return {"knee": x1, "a0": float(coef[0]), "a1": float(coef[1]), "a2": float(coef[2]),
            # 기울기 [mAh/100 사이클, 음수 = 감소]
            "slope_pre": float((coef[1] - coef[2]) * 1e5), "slope_post": float((coef[1] + coef[2]) * 1e5),
            "sse_ratio_vs_linear": sse / sse_lin}


def three_segment_onset(x: np.ndarray, y: np.ndarray, margin: int = KNEE_MARGIN) -> dict:
    """이중 Bacon–Watts(γ→0) = 연속 3구간 선형. 정규방정식 + 전격자 탐색(각 구간 ≥ margin 사이클)."""
    t = (x - x[0]) / (x[-1] - x[0])
    yc = y - y.mean()
    ci = np.where((x >= x[0] + margin) & (x <= x[-1] - margin))[0]
    H = np.maximum(t[:, None] - t[ci][None, :], 0.0)                # n × m 힌지 기저
    n, sx, sxx = len(t), t.sum(), (t * t).sum()
    sy, sxy, syy = yc.sum(), (t * yc).sum(), (yc * yc).sum()
    h1, hx, hy, HH = H.sum(0), H.T @ t, H.T @ yc, H.T @ H
    best = (np.inf, None, None, None)
    xc = x[ci]
    for i in range(len(ci)):
        js = np.where(xc >= xc[i] + margin)[0]
        if not len(js):
            continue
        k = len(js)
        G = np.empty((k, 4, 4))
        G[:, 0, 0], G[:, 0, 1], G[:, 1, 1] = n, sx, sxx
        G[:, 0, 2], G[:, 1, 2], G[:, 2, 2] = h1[i], hx[i], HH[i, i]
        G[:, 0, 3], G[:, 1, 3], G[:, 2, 3], G[:, 3, 3] = h1[js], hx[js], HH[i, js], HH[js, js]
        G[:, 1, 0], G[:, 2, 0], G[:, 2, 1] = G[:, 0, 1], G[:, 0, 2], G[:, 1, 2]
        G[:, 3, 0], G[:, 3, 1], G[:, 3, 2] = G[:, 0, 3], G[:, 1, 3], G[:, 2, 3]
        r = np.empty((k, 4))
        r[:, 0], r[:, 1], r[:, 2], r[:, 3] = sy, sxy, hy[i], hy[js]
        beta = np.linalg.solve(G, r[..., None])[..., 0]
        sse = syy - (beta * r).sum(1)
        j = int(np.argmin(sse))
        if sse[j] < best[0]:
            best = (float(sse[j]), float(xc[i]), float(xc[js[j]]), beta[j])
    _, a, b, be = best
    sc = (x[-1] - x[0]) / 1e5                                       # t 척도 기울기 → mAh/100 사이클 (음수 = 감소)
    return {"onset": a, "second_break": b, "slope_seg1": float(be[1] / sc),
            "slope_seg2": float((be[1] + be[2]) / sc), "slope_seg3": float((be[1] + be[2] + be[3]) / sc)}


def initial_rise(x: np.ndarray, qs: np.ndarray, q_init: float, L: float) -> dict:
    """초기 용량 상승폭(mAh), 정점 사이클, 정점 이후 초기값 아래로 처음 내려오는 시점(수명 비)."""
    i = int(np.nanargmax(qs))
    peak = float(x[i])
    below = np.where((x > peak) & (qs / q_init < 1.0))[0]
    return {"rise_mAh": float((qs[i] - q_init) * 1000), "rise_peak_cycle": peak,
            "frac_life_above_init": float(x[below[0]] / L) if len(below) else np.nan}


def kneedle_knee(x: np.ndarray, y: np.ndarray) -> float:
    xn = (x - x.min()) / (x.max() - x.min())
    yn = (y - y.min()) / (y.max() - y.min())
    return float(x[np.argmax(yn - (1 - xn))])


def batch_dip_check(cells: list[dict]) -> dict:
    """cycle 45~56 최소 QD 가 cycle 36~42 중앙값보다 3 mAh 이상 낮은 셀 수 (110 사이클 이상 기록된 셀)."""
    out = {}
    for b in BATCH_NAMES:
        n = n_dip = 0
        for c in [c for c in cells if c["batch"] == b]:
            s = clean_summary(c["summary"])
            if len(s) < 110:
                continue
            q = s.set_index("cycle")["QD"]
            base = np.nanmedian(q.loc[36:42])
            n += 1
            n_dip += int((np.nanmin(q.loc[45:56]) - base) * 1000 < -3)
        out[b] = {"n": n, "n_dip": n_dip}
    return out


def life_group(life: float) -> str:
    if life > LONG_TH:
        return "장수명(>1,000)"
    if life < SHORT_TH:
        return "단수명(<500)"
    return "중간(500~1,000)"


# ───────────────────────── 셀 단위 지표 ─────────────────────────
def analyse() -> tuple[pd.DataFrame, pd.DataFrame, dict, list[dict]]:
    cells = load_all()
    meta = cell_table(cells)
    by_key = {c["cell_key"]: c for c in cells}
    rows, dec_rows = [], []
    for c in cells:
        m = meta.loc[meta.cell_key == c["cell_key"]].iloc[0]
        if not m.labeled:
            continue
        L = float(c["cycle_life"])
        x, q, qs = qd_series(c, upto=L)
        q_init = float(np.nanmedian(q[:5]))                       # cycle 2~6
        f = x / L
        r = {"cell_key": c["cell_key"], "batch": c["batch"], "policy": m.policy, "group": m.group,
             "C1": m.C1, "Q1": m.Q1, "C2": m.C2, "cycle_life": L, "life_group": life_group(L),
             "label_550": int(L >= LABEL_THRESHOLD), "QD_init": q_init,
             "SOH_at_EOL_vs_init": EOL_AH / q_init}
        # (b) 초기 100 사이클
        qd10, qd100 = qd_at(c, 10), qd_at(c, 100)
        r["QD_at10"], r["QD_at100"] = qd10, qd100
        r["dQD_100_10_mAh"] = (qd100 - qd10) * 1000
        early = x <= 100
        r["slope_2_100"] = fade_slope(x, q, early)
        r["QDmax_minus_QD2_mAh"] = float((np.nanmax(q[early]) - q[0]) * 1000)
        # 단발 스파이크(평활 곡선에서 3 mAh 넘게 벗어난 단일 사이클) 수 — 중앙값 창 사용 근거(배치 무관 일반 규칙)
        r["n_spike_gt3mAh_2_100"] = int((np.abs(q[early] - qs[early]) * 1000 > 3).sum())
        d2 = np.diff(q[early & np.isfinite(q)], 2)                 # 2차 차분 → 선형 추세 제거
        r["noise_sigma_mAh"] = float(1.4826 * np.median(np.abs(d2 - np.median(d2))) / np.sqrt(6) * 1000)
        r["snr_dQD_100_10"] = abs(r["dQD_100_10_mAh"]) / r["noise_sigma_mAh"]
        me = early & np.isfinite(q)                                 # 추세 잔차 기반 잡음(3차 다항 추세)
        res = q[me] - np.polyval(np.polyfit(x[me], q[me], 3), x[me])
        r["noise_sigma_trend_mAh"] = float(1.4826 * np.median(np.abs(res - np.median(res))) * 1000)
        r["snr_dQD_100_10_trend"] = abs(r["dQD_100_10_mAh"]) / r["noise_sigma_trend_mAh"]
        r.update(initial_rise(x, qs, q_init, L))
        # 수명 구간별
        for lo, hi in zip(DECILES[:-1], DECILES[1:]):
            dec_rows.append({"cell_key": c["cell_key"], "batch": c["batch"], "life_group": r["life_group"],
                             "frac_mid": (lo + hi) / 2,
                             "slope": fade_slope(x, q, (f >= lo) & (f < hi if hi < 1 else f <= 1.0))})
        r["slope_mid_40_60"] = fade_slope(x, q, (f >= 0.4) & (f < 0.6))
        r["slope_late_80_100"] = fade_slope(x, q, f >= 0.8)
        r["late_over_mid"] = r["slope_late_80_100"] / r["slope_mid_40_60"]
        r["slope_late_x_life"] = r["slope_late_80_100"] * L / 100     # mAh per 수명 단위(수명 배율 보정)
        # EOL 직전 기울기(헤드룸 단순 환산용): cycle L−25 ~ L
        r["slope_at_EOL"] = fade_slope(x, q, x >= L - 25)
        q_end = float(qs[-1])
        q80 = float(np.interp(0.8 * L, x, qs))
        r["loss_total_mAh"] = (q_init - q_end) * 1000
        r["share_loss_last20pct"] = (q80 - q_end) / (q_init - q_end)
        r["share_loss_first100"] = (qd10 - qd100) / (q_init - q_end)
        r["SOH_at100_vs_init"] = qd100 / q_init
        # (c) knee
        bw = bacon_watts_knee(x, qs)
        r.update({"knee_bw": bw["knee"], "slope_pre_knee": bw["slope_pre"], "slope_post_knee": bw["slope_post"],
                  "knee_sse_ratio_vs_linear": bw["sse_ratio_vs_linear"],
                  "bw_a0": bw["a0"], "bw_a1": bw["a1"], "bw_a2": bw["a2"]})
        r["knee_kneedle"] = kneedle_knee(x, qs)
        r["knee_ratio_bw"] = r["knee_bw"] / L
        r["knee_ratio_kneedle"] = r["knee_kneedle"] / L
        r["post_knee_cycles"] = L - r["knee_bw"]
        r["post_knee_frac"] = r["post_knee_cycles"] / L
        on = three_segment_onset(x, qs)
        r.update({"onset_3seg": on["onset"], "second_break_3seg": on["second_break"],
                  "onset_ratio": on["onset"] / L, "slope_seg1": on["slope_seg1"], "slope_seg2": on["slope_seg2"],
                  "slope_seg3": on["slope_seg3"]})
        r["knee_sharpness"] = r["slope_post_knee"] / r["slope_pre_knee"]
        r["knee_unclear"] = bool(r["knee_sharpness"] < ACC_FLAG)
        r["QD_at_knee"] = float(np.interp(r["knee_bw"], x, qs))
        r["SOH_at_knee_vs_init"] = r["QD_at_knee"] / q_init
        rows.append(r)
    df = pd.DataFrame(rows)
    m1 = df.batch == "batch1"
    df["b1_tertile"] = None
    df.loc[m1, "b1_tertile"] = pd.qcut(df.loc[m1, "cycle_life"], 3, labels=TERTILES).astype(str)
    dec = pd.DataFrame(dec_rows)
    return df, dec, by_key, cells


def spearman(a, b) -> dict:
    a, b = np.asarray(a, float), np.asarray(b, float)
    ok = np.isfinite(a) & np.isfinite(b)
    rho, p = stats.spearmanr(a[ok], b[ok])
    r_log, p_log = stats.pearsonr(a[ok], np.log10(b[ok]))
    return {"n": int(ok.sum()), "spearman_rho": float(rho), "spearman_p": float(p),
            "pearson_r_log10life": float(r_log), "pearson_p": float(p_log)}


def q(s: pd.Series) -> dict:
    s = s.dropna()
    return {"n": int(len(s)), "median": float(s.median()), "q1": float(s.quantile(0.25)),
            "q3": float(s.quantile(0.75)), "min": float(s.min()), "max": float(s.max()), "mean": float(s.mean())}


# ───────────────────────── 그림 (a) ─────────────────────────
PDF_RC = {"font.size": 10.5, "axes.titlesize": 11.5, "axes.labelsize": 10.5, "xtick.labelsize": 9.5,
          "ytick.labelsize": 9.5, "legend.fontsize": 9.5}   # PDF 폭(~16 cm)에서 읽히도록 큰 글자·적은 주석


def fig_curves(df: pd.DataFrame, cells: list[dict]) -> None:
    """(a)+(c) PDF 그림: 위 = 절대 사이클 QD(knee 점), 아래 = 정규화 곡선(knee 위치 중앙값). 세부 수치는 본문에."""
    meta = cell_table(cells)
    by_key = {c["cell_key"]: c for c in cells}
    with plt.rc_context(PDF_RC):
        fig = plt.figure(figsize=(11.5, 7.4))
        gs = GridSpec(2, 4, figure=fig, width_ratios=[1, 1, 1, 0.04], wspace=0.10, hspace=0.42, top=0.88)
        ax_top = [fig.add_subplot(gs[0, i]) for i in range(3)]
        ax_bot = [fig.add_subplot(gs[1, i]) for i in range(3)]
        cax = fig.add_subplot(gs[:, 3])
        grid = np.linspace(0, 1, 201)
        role = {"batch1": "학습", "batch2": "테스트", "batch3": "추가 검증"}
        for i, b in enumerate(BATCH_NAMES):
            ax, axn = ax_top[i], ax_bot[i]
            mb = meta[meta.batch == b]
            for c in [c for c in cells if c["batch"] == b]:
                m = mb.loc[mb.cell_key == c["cell_key"]].iloc[0]
                x, _, qs = qd_series(c)
                if m.labeled:
                    ax.plot(x, qs, color=LIFE_CMAP(LIFE_NORM(c["cycle_life"])), lw=0.8, alpha=0.9, zorder=3)
                elif m.censored:
                    ax.plot(x, qs, color="#8a8a8a", lw=0.8, ls="--", alpha=0.9, zorder=2)
                else:
                    ax.plot(x, qs, color="#b5b5b5", lw=0.8, ls=":", alpha=0.9, zorder=1)
            d = df[df.batch == b]
            ax.scatter(d.knee_bw, d.QD_at_knee, s=9, color="#111", zorder=5, lw=0)
            ax.axvspan(0, 100, color="#cfe3fa", alpha=0.9, zorder=0, lw=0)
            ax.axhline(EOL_AH, color=ps.SHORT, lw=1.0, ls="--", zorder=4)
            ax.set_xlim(0, 2300)
            ax.set_ylim(0.80, 1.135)
            ax.set_xticks([0, 500, 1000, 1500, 2000])
            n_lab, n_cens = int(mb.labeled.sum()), int(mb.censored.sum())
            ax.set_title(f"{BS[b]} · {role[b]} (라벨 {n_lab}셀)", color=ps.BATCH_COLOR[b])
            ax.set_xlabel("사이클")
            if i == 0:
                ax.set_ylabel("방전 용량 QD (Ah)")
                ax.text(1250, EOL_AH + 0.004, "EOL 0.88 Ah", color=ps.SHORT, fontsize=9.5, va="bottom")
            else:
                ax.set_yticklabels([])
            # 정규화 곡선
            curves = []
            for _, r in d.iterrows():
                x, _, qs = qd_series(by_key[r.cell_key], upto=r.cycle_life)
                xn, yn = x / r.cycle_life, qs / r.QD_init
                axn.plot(xn, yn, color=LIFE_CMAP(LIFE_NORM(r.cycle_life)), lw=0.7, alpha=0.55)
                curves.append(np.interp(grid, xn, yn, left=np.nan, right=np.nan))
            stack = np.vstack(curves)
            enough = np.isfinite(stack).sum(axis=0) >= 0.5 * len(curves)
            med = np.full(grid.size, np.nan)
            med[enough] = np.nanmedian(stack[:, enough], axis=0)
            axn.plot(grid, med, color="#111111", lw=2.0)
            axn.axhline(1.0, color="#444", lw=0.8, ls=":", zorder=1)
            kr = float(d.knee_ratio_bw.median())
            axn.axvline(kr, color="#111", lw=1.1, ls="--")
            eol_rel = float((EOL_AH / d.QD_init).median())
            axn.axhline(eol_rel, color=ps.SHORT, lw=1.0, ls="--")
            axn.text(0.02, eol_rel + 0.003, f"EOL = 초기의 {eol_rel*100:.1f}%", color=ps.SHORT, fontsize=9.5, va="bottom")
            sh = float(d.share_loss_last20pct.median())
            axn.axvspan(0.8, 1.0, color="#f3d6d6", alpha=0.5, zorder=0, lw=0)
            axn.text(0.9, 1.032, f"손실의\n{sh*100:.0f}%", ha="center", va="top", fontsize=9.5, color="#7a1f1f",
                     fontweight="bold")
            axn.set_xlim(0, 1.0)
            axn.set_ylim(0.78, 1.04)
            axn.set_xlabel("수명 진행률 (사이클 / cycle_life)")
            if i == 0:
                axn.set_ylabel("QD / 초기 QD")
            else:
                axn.set_yticklabels([])
            axn.set_title(f"{BS[b]} 정규화 · knee 위치 중앙값 {kr:.2f} (파선)", fontsize=10.5)
        sm = ScalarMappable(norm=LIFE_NORM, cmap=LIFE_CMAP)
        cb = fig.colorbar(sm, cax=cax)
        ticks = [400, 600, 1000, 1500, 2000]
        cb.set_ticks(ticks)
        cb.set_ticklabels([f"{t:,}" for t in ticks])
        cb.minorticks_off()
        cb.set_label("cycle_life (로그 눈금)")
        handles = [Line2D([], [], color=LIFE_CMAP(LIFE_NORM(800)), lw=1.4, label="라벨 셀 (색 = 수명)"),
                   Line2D([], [], color="#8a8a8a", lw=1.2, ls="--", label="중도절단 / 라벨 없음"),
                   Line2D([], [], color="#111", marker="o", ms=4, ls="none", label="knee (Bacon–Watts)"),
                   plt.Rectangle((0, 0), 1, 1, fc="#cfe3fa", label="첫 100 사이클 (모델 입력)"),
                   Line2D([], [], color="#111", lw=2.0, label="배치 중앙값 곡선"),
                   plt.Rectangle((0, 0), 1, 1, fc="#f3d6d6", label="마지막 20% 수명")]
        fig.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.968), ncol=6, fontsize=9.3,
                   handlelength=1.6, columnspacing=1.1)
        fig.suptitle("Q2-(a)(c) 방전 용량은 초기 소폭 상승 → 완만한 감소 → 수명 약 3/4 지점(knee) 이후 급락",
                     fontsize=13, fontweight="bold", y=1.01)
        ps.save(fig, "q2_qd_curves")


# ───────────────────────── 그림 (b) ─────────────────────────
def _first100_matrix(d: pd.DataFrame, by_key: dict, cyc: np.ndarray) -> np.ndarray:
    M = []
    for _, r in d.iterrows():
        x, _, qs = qd_series(by_key[r.cell_key], upto=100)
        M.append(np.interp(cyc, x, (qs - r.QD_at10) * 1000))
    return np.vstack(M)


def fig_fade(df: pd.DataFrame, dec: pd.DataFrame, cells: list[dict]) -> None:
    """(b) PDF 그림: ① 수명 구간별 용량 기울기 ② 첫 100 사이클 QD − QD@10. 세부 수치는 본문에."""
    with plt.rc_context(PDF_RC):
        fig, axs = plt.subplots(1, 2, figsize=(11.5, 4.7), gridspec_kw={"width_ratios": [1.0, 1.15], "wspace": 0.24})
        # ① 수명 구간별 기울기
        ax = axs[0]
        for b in BATCH_NAMES:
            g = dec[dec.batch == b].groupby("frac_mid")["slope"]
            med, lo, hi = g.median(), g.quantile(0.25), g.quantile(0.75)
            xs = med.index.to_numpy() * 100
            ratio = df.loc[df.batch == b, "late_over_mid"].median()
            ax.fill_between(xs, lo, hi, color=ps.BATCH_COLOR[b], alpha=0.15, lw=0)
            ax.plot(xs, med, marker=MARK[b], ms=5, lw=1.8, color=ps.BATCH_COLOR[b],
                    label=f"{BS[b]}  말기/중기 {ratio:.1f}배")
        ax.axhline(0, color="#555", lw=0.8)
        ax.set_xticks(np.arange(0, 101, 20))
        ax.set_xlim(0, 100)
        ax.set_xlabel("수명 진행률 (%, cycle/cycle_life) · 10% 구간 중앙에 표시")
        ax.set_ylabel("용량 기울기 (mAh/100 사이클, 음수=감소)")
        ax.set_title("① 수명 구간별 기울기 — 말기에 약 9배 가속")
        ax.legend(loc="lower left", title="중앙값 · IQR 음영", title_fontsize=9.3)

        # ② 첫 100 사이클
        ax = axs[1]
        by_key = {c["cell_key"]: c for c in cells}
        cyc = np.arange(2, 101)
        b1 = df[df.batch == "batch1"]
        for t in TERTILES:
            d = b1[b1.b1_tertile == t]
            M = _first100_matrix(d, by_key, cyc)
            if t != "중위 1/3":
                ax.fill_between(cyc, np.percentile(M, 25, 0), np.percentile(M, 75, 0), color=TER_COL[t], alpha=0.10,
                                lw=0)
            ax.plot(cyc, np.median(M, 0), color=TER_COL[t], lw=2.2,
                    label=f"B1 수명 {t} ({int(d.cycle_life.min())}~{int(d.cycle_life.max())})")
        for b in ("batch2", "batch3"):
            d = df[df.batch == b]
            med = np.median(_first100_matrix(d, by_key, cyc), 0)
            ax.plot(cyc, med, color=ps.BATCH_COLOR[b], lw=1.6, ls="--", label=f"{BS[b]} 전체 중앙값 (참고)")
        for b, yv in (("batch1", 6.9), ("batch2", 6.9)):
            pk = float(df.loc[df.batch == b, "rise_peak_cycle"].median())
            ax.plot([pk], [yv], marker="v", ms=8, color=ps.BATCH_COLOR[b], clip_on=False)
            ax.text(pk, yv + 0.5, f"{BS[b]} 상승 정점\ncycle {pk:.0f}", ha="center", va="bottom", fontsize=9,
                    color=ps.BATCH_COLOR[b])
        dip = DIP_INFO.get("batch2", {})
        if dip:
            ax.annotate(f"B2 공통 일시 하락\n({dip['n_dip']}/{dip['n']}셀, cycle 45~56)", xy=(51, -0.9), xytext=(51, 4.2),
                        fontsize=9, color="#9a3412", ha="center", va="center",
                        arrowprops=dict(arrowstyle="->", color="#9a3412", lw=0.8))
        ax.axhline(0, color="#555", lw=0.8)
        ax.axvline(10, color="#777", lw=0.8, ls=":")
        ax.set_xlim(2, 100)
        ax.set_ylim(-7.5, 9.5)
        ax.set_xlabel("사이클")
        ax.set_ylabel("QD − QD@10 (mAh)")
        ax.set_title("② 첫 100 사이클 — 변화는 수 mAh (총 손실의 1% 미만)")
        ax.legend(loc="lower left", fontsize=8.6, ncol=2, handlelength=1.6, columnspacing=0.8)
        fig.suptitle("Q2-(b) 열화 속도는 일정하지 않다 — 말기에 가속, 조기 예측 창(첫 100 사이클)의 용량 변화는 매우 작다",
                     fontsize=12.5, fontweight="bold", y=1.02)
        ps.save(fig, "q2_fade_rate")


def fig_early_slope(df: pd.DataFrame) -> None:
    """(b) 보조 그림: 초기(2~100) 용량 기울기 vs 수명 — B2 는 fastcharge/newstructure 구분."""
    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    groups = [("batch1", None, "B1"), ("batch2", "fastcharge", "B2 fastcharge"),
              ("batch2", "newstructure", "B2 newstructure"), ("batch3", None, "B3")]
    for b, g, lab in groups:
        d = df[df.batch == b] if g is None else df[(df.batch == b) & (df.group == g)]
        rho, p = stats.spearmanr(d.slope_2_100, d.cycle_life)
        hollow = g == "newstructure"
        ax.scatter(d.cycle_life, d.slope_2_100, s=34, marker=MARK[b],
                   facecolor="white" if hollow else ps.BATCH_COLOR[b],
                   edgecolor=ps.BATCH_COLOR[b] if hollow else "white", lw=1.2 if hollow else 0.6, zorder=3,
                   label=f"{lab}  ρ = {rho:+.2f} (p = {p:.2g}, n = {len(d)})")
    un = df[df.slope_2_100 < FAST_EARLY]
    ax.scatter(un.cycle_life, un.slope_2_100, s=150, facecolor="none", edgecolor="#333", lw=1.0, zorder=4,
               label=f"초기 기울기 < {FAST_EARLY:.0f} ({len(un)}셀, 준선형 열화)")
    ax.axhline(0, color="#555", lw=0.8)
    for v in (SHORT_TH, LONG_TH):
        ax.axvline(v, color="#999", lw=0.8, ls="--")
    ax.set_xscale("log")
    ax.set_xticks([400, 500, 700, 1000, 1500, 2000])
    ax.set_xticklabels(["400", "500", "700", "1,000", "1,500", "2,000"])
    ax.minorticks_off()
    ax.set_xlabel("cycle_life (로그 눈금)")
    ax.set_ylabel(f"초기 용량 기울기, cycle 2~100 ({SLOPE_UNIT})")
    ax.set_title("Q2-(b) 보조: 초기 용량 기울기 vs 수명 — 유의한 관계는 B1뿐(ρ +0.52)")
    ax.legend(loc="lower right", fontsize=8.6, title="Spearman (집단별)", title_fontsize=8.6)
    ps.save(fig, "q2_early_slope")

# ───────────────────────── 그림 (c) ─────────────────────────
def fig_knee(df: pd.DataFrame, cells: list[dict], perm: dict) -> list[str]:
    by_key = {c["cell_key"]: c for c in cells}
    # 예시 셀: 배치별 '뚜렷한 knee' 셀 중 수명이 배치 중앙값에 가장 가까운 셀 + Batch 1 의 knee 불명확 셀
    ex = []
    for b in BATCH_NAMES:
        d = df[(df.batch == b) & ~df.knee_unclear]
        if b == "batch2":
            d = d[d.group == "fastcharge"]
        med = df.loc[df.batch == b, "cycle_life"].median()
        ex.append(d.iloc[(d.cycle_life - med).abs().argsort().iloc[0]].cell_key)
    unclear_b1 = df[(df.batch == "batch1") & df.knee_unclear].sort_values("knee_sharpness")
    ex.append(unclear_b1.iloc[0].cell_key if len(unclear_b1) else df[df.knee_unclear].iloc[0].cell_key)

    fig = plt.figure(figsize=(17, 10.4))
    gs = GridSpec(2, 12, figure=fig, hspace=0.45, wspace=1.2)
    top = [fig.add_subplot(gs[0, 3 * i:3 * i + 3]) for i in range(4)]
    bot = [fig.add_subplot(gs[1, 4 * i:4 * i + 4]) for i in range(3)]
    for k, (ax, key) in enumerate(zip(top, ex)):
        r = df[df.cell_key == key].iloc[0]
        x, qraw, qs = qd_series(by_key[key], upto=r.cycle_life)
        ax.plot(x, qraw, ".", ms=2, color="#c4c4c4", zorder=1, label="clean QD")
        ax.plot(x, qs, color=ps.BATCH_COLOR[r.batch], lw=1.6, zorder=2, label="QD_s (이동중앙값)")
        fit = r.bw_a0 + r.bw_a1 * (x - r.knee_bw) + r.bw_a2 * np.abs(x - r.knee_bw)
        ax.plot(x, fit, color="#111", lw=1.3, ls="--", zorder=3, label="Bacon–Watts 2구간 적합 (실선 = knee)")
        ax.axvline(r.knee_bw, color="#111", lw=1.0, zorder=3)
        ax.axvline(r.onset_3seg, color="#8a2be2", lw=1.2, ls=":", zorder=3, label="knee-onset (3구간 적합)")
        ax.plot([r.knee_kneedle], [np.interp(r.knee_kneedle, x, qs)], marker="D", ms=7, color="#eda100",
                mec="#5a3d00", zorder=4, ls="none", label="Kneedle knee")
        ax.axvspan(0, 100, color="#ddebfb", alpha=0.8, zorder=0, lw=0)
        ax.axhline(EOL_AH, color=ps.SHORT, lw=0.9, ls="--")
        ax.set_xlim(0, r.cycle_life * 1.03)
        lo = min(np.nanmin(qs), EOL_AH) - 0.01
        ax.set_ylim(lo, np.nanmax(qs) + 0.012)
        tag = "knee 불명확(준선형)" if r.knee_unclear else "전형적 셀"
        ax.set_title(f"{r.cell_key} · {r.policy}\n{tag}, 수명 {int(r.cycle_life)}", fontsize=10.5)
        ax.text(0.04, 0.05, f"knee = {int(r.knee_bw)} ({r.knee_ratio_bw*100:.0f}% 수명)\n"
                f"onset = {int(r.onset_3seg)} ({r.onset_ratio*100:.0f}% 수명)\n"
                f"기울기 {r.slope_pre_knee:.0f} → {r.slope_post_knee:.0f} mAh/100cyc (×{r.knee_sharpness:.1f})",
                transform=ax.transAxes, fontsize=8.6,
                va="bottom", bbox=dict(boxstyle="round,pad=0.25", fc="white", ec="#dddddd", alpha=0.95))
        ax.set_xlabel("사이클")
        if k == 0:
            ax.set_ylabel("QD (Ah)")
    y0, y1 = top[0].get_ylim()
    top[0].text(50, y0 + 0.62 * (y1 - y0), "첫 100\n사이클", ha="center", va="center", fontsize=8.5, color="#1c5cab")
    h, lab = top[0].get_legend_handles_labels()
    fig.legend(h, lab, loc="upper center", bbox_to_anchor=(0.56, 0.958), ncol=5, fontsize=9.3)
    fig.text(0.07, 0.94, "① knee 검출 예시 (파란 음영 = 첫 100 사이클)", fontsize=11.5, fontweight="bold", va="center")

    # ② knee vs cycle_life (+ onset)
    ax = bot[0]
    lim = [0, 2000]
    ax.plot(lim, lim, color="#999", lw=0.9, ls="--")
    ax.text(1750, 1820, "knee = 수명", color="#777", fontsize=8.5, rotation=0, ha="right")
    ax.axhline(100, color="#1c5cab", lw=1.0, ls=":")
    ax.text(1990, 112, "cycle 100 (조기예측 입력 끝)", color="#1c5cab", fontsize=8.5, ha="right", va="bottom")
    for b in BATCH_NAMES:
        d = df[df.batch == b]
        ax.scatter(d.cycle_life, d.knee_bw, s=30, marker=MARK[b], color=ps.BATCH_COLOR[b], edgecolor="white",
                   lw=0.5, label=f"{BS[b]}  knee ≤ 100: {(d.knee_bw <= 100).sum()}/{len(d)}, "
                                 f"onset ≤ 100: {(d.onset_3seg <= 100).sum()}/{len(d)}")
    ax.scatter(df.cycle_life, df.onset_3seg, s=14, marker="x", color="#8a2be2", lw=0.8, alpha=0.75,
               label=f"knee-onset (3구간 적합, 최소 {int(df.onset_3seg.min())})")
    unc = df[df.knee_unclear]
    ax.scatter(unc.cycle_life, unc.knee_bw, s=110, facecolor="none", edgecolor="#333", lw=0.9, label="knee 불명확 셀")
    ax.set_xlim(*lim)
    ax.set_ylim(*lim)
    ax.set_xlabel("cycle_life")
    ax.set_ylabel("사이클 (knee: Bacon–Watts, onset: 3구간)")
    ax.set_title("② knee·onset 시점 vs 수명 — 전부 cycle 100 이후")
    ax.legend(loc="upper left", fontsize=8.2)
    ax.text(0.0, -0.165, f"※ 주의: knee를 셀마다 [12, cycle_life−10] 안에서 찾으므로\n"
            f"   knee–수명 상관(ρ {df[df.batch=='batch1'].pipe(lambda d: stats.spearmanr(d.knee_bw, d.cycle_life)[0]):.2f}"
            f"·{df[df.batch=='batch2'].pipe(lambda d: stats.spearmanr(d.knee_bw, d.cycle_life)[0]):.2f}"
            f"·{df[df.batch=='batch3'].pipe(lambda d: stats.spearmanr(d.knee_bw, d.cycle_life)[0]):.2f})은 구성상 높다.\n"
            f"   B1 knee/수명 비를 셀끼리 무작위로 섞어도 ρ ≈ {perm['median_rho']:.2f} → 이 ρ 자체는 증거가 아님.\n"
            f"   정보가 있는 것은 ③의 'knee/수명 비가 수명과 무관(B1 ρ = −0.07)'이라는 점.",
            transform=ax.transAxes, ha="left", va="top", fontsize=8.4, color="#333")

    # ③ knee / cycle_life
    ax = bot[1]
    rng = np.random.default_rng(RANDOM_STATE)
    data = [df.loc[df.batch == b, "knee_ratio_bw"].to_numpy() for b in BATCH_NAMES]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#111", lw=1.6))
    for patch, b in zip(bp["boxes"], BATCH_NAMES):
        patch.set_facecolor(mcolors.to_rgba(ps.BATCH_COLOR[b], 0.18))
        patch.set_edgecolor(ps.BATCH_COLOR[b])
    for i, b in enumerate(BATCH_NAMES):
        d = df[df.batch == b]
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(d)), d.knee_ratio_bw, s=16, color=ps.BATCH_COLOR[b],
                   marker=MARK[b], alpha=0.85, zorder=3)
        ax.scatter(i + 0.32, d.knee_ratio_kneedle.median(), marker="D", s=40, color="#eda100", edgecolor="#5a3d00",
                   zorder=4, label="Kneedle 중앙값" if i == 0 else None)
        ax.scatter(i + 0.32, d.onset_ratio.median(), marker="x", s=46, color="#8a2be2", lw=1.4,
                   zorder=4, label="knee-onset 중앙값" if i == 0 else None)
        rr = stats.spearmanr(d.knee_ratio_bw, d.cycle_life)[0]
        ax.text(i, 0.935, f"{d.knee_ratio_bw.median()*100:.0f}%", ha="center", fontsize=10, fontweight="bold",
                color=ps.BATCH_COLOR[b])
        ax.text(i, 0.905, f"비–수명 ρ {rr:+.2f}", ha="center", fontsize=8.3, color="#444")
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"{BS[b]}" for b in BATCH_NAMES])
    ax.set_ylim(0.45, 0.96)
    ax.set_ylabel("knee / cycle_life")
    ax.set_title("③ knee는 수명의 약 3/4 지점 — B1에서는 그 비율이 수명과 무관")
    ax.legend(loc="lower right", fontsize=8.5)

    # ④ knee 이후 남은 사이클 (수명 대비 비율 병기)
    ax = bot[2]
    data = [df.loc[df.batch == b, "post_knee_cycles"].to_numpy() for b in BATCH_NAMES]
    bp = ax.boxplot(data, positions=[0, 1, 2], widths=0.5, showfliers=False, patch_artist=True,
                    medianprops=dict(color="#111", lw=1.6))
    for patch, b in zip(bp["boxes"], BATCH_NAMES):
        patch.set_facecolor(mcolors.to_rgba(ps.BATCH_COLOR[b], 0.18))
        patch.set_edgecolor(ps.BATCH_COLOR[b])
    for i, b in enumerate(BATCH_NAMES):
        d = df[df.batch == b]
        ax.scatter(i + rng.uniform(-0.13, 0.13, len(d)), d.post_knee_cycles, s=16, color=ps.BATCH_COLOR[b],
                   marker=MARK[b], alpha=0.85, zorder=3)
        ax.text(i, 448, f"{d.post_knee_cycles.median():g} 사이클\n(수명의 {d.post_knee_frac.median()*100:.0f}%)",
                ha="center", va="bottom", fontsize=9.6, fontweight="bold", color=ps.BATCH_COLOR[b])
    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels([f"{BS[b]}" for b in BATCH_NAMES])
    ax.set_ylim(0, 520)
    ax.set_ylabel("cycle_life − knee (사이클)")
    ax.set_title("④ knee 이후 EOL까지 남은 사이클 (괄호: 수명 대비)")
    fig.suptitle("Q2-(c) Knee point — 모든 셀에서 cycle 100 이후(knee 최소 "
                 f"{int(df.knee_bw.min())}, onset 최소 {int(df.onset_3seg.min())}), 수명의 75~79% 지점(배치 중앙값)",
                 fontsize=14, fontweight="bold", y=1.0)
    ps.save(fig, "q2_knee")
    return ex


# ───────────────────────── 요약 ─────────────────────────
def knee_permutation(df: pd.DataFrame, n_perm: int = 2000) -> dict:
    """knee/L 비를 셀 간 무작위로 섞은 뒤(knee* = 섞인 비 × L) knee*–L Spearman 분포 → 구성상 기준선."""
    out = {}
    rng = np.random.default_rng(RANDOM_STATE)
    for b in BATCH_NAMES:
        d = df[df.batch == b]
        L, kr = d.cycle_life.to_numpy(), d.knee_ratio_bw.to_numpy()
        rh = [stats.spearmanr(rng.permutation(kr) * L, L)[0] for _ in range(n_perm)]
        out[b] = {"observed_rho": float(stats.spearmanr(d.knee_bw, L)[0]), "median_rho": float(np.median(rh)),
                  "rho_2p5": float(np.percentile(rh, 2.5)), "rho_97p5": float(np.percentile(rh, 97.5)),
                  "frac_perm_ge_observed": float(np.mean(np.array(rh) >= stats.spearmanr(d.knee_bw, L)[0]))}
    return out


def _resid(v: np.ndarray, ctrl: list[np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.ones(len(v))] + ctrl)
    return v - X @ np.linalg.lstsq(X, v, rcond=None)[0]


def partial_corr(f, y, ctrl: list[np.ndarray]) -> dict:
    """y 와 f 를 ctrl 에 각각 OLS 회귀한 잔차의 Pearson r, t 검정(df = n−2−k)."""
    f, y = np.asarray(f, float), np.asarray(y, float)
    r = float(stats.pearsonr(_resid(f, ctrl), _resid(y, ctrl))[0])
    n, k = len(y), len(ctrl)
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2))
    return {"r": r, "p": float(2 * stats.t.sf(abs(t), n - 2 - k)), "n": n, "k": k}


def initial_capacity_check(df: pd.DataFrame, cells: list[dict]) -> dict:
    """(d) 초기 용량 수준(summary QD_init·QD_2, CC 끝점 Qcc_init)과 수명 — 노션 Qdlin 경고와 함께 점검.

    Qcc_init 은 원시 Qdlin 의 2.0 V 절대 수준이므로, 노션 Batch 3 메모('충전 커브 시작 시점이 배치별로 상이 :
    Qdlin 변수를 단순 비교하면 왜곡 발생')가 경고한 배치 간 비교에 해당한다 → 라벨 없는 배치 이동량을 함께 보고.
    """
    by_key = {c["cell_key"]: c for c in cells}
    meta = cell_table(cells)
    meta = meta[meta.group.isin(["fastcharge", "newstructure"])].copy()

    def qcc(c: dict, a: int = 2, b: int = 6) -> float:
        v = np.array([qdlin(c, k)[-1] for k in range(a, b + 1)], float)
        v[(v <= 0) | (v > 1.2)] = np.nan
        return float(np.nanmedian(v))

    def s_qd(c: dict, a: int, b: int) -> float:
        s = clean_summary(c["summary"])
        return float(np.nanmedian(s.loc[(s.cycle >= a) & (s.cycle <= b), "QD"]))

    meta["Qcc_init"] = [qcc(by_key[k]) for k in meta.cell_key]
    meta["QD_init"] = [s_qd(by_key[k], 2, 6) for k in meta.cell_key]
    meta["QD_2"] = [s_qd(by_key[k], 2, 2) for k in meta.cell_key]
    meta["dQ_logvar"] = [float(np.log10(np.var(delta_q(by_key[k], 100, 10)))) for k in meta.cell_key]
    n_bad = {b: int(sum(int(np.sum((by_key[k]["Qdlin"][1:100, -1] <= 0) | (by_key[k]["Qdlin"][1:100, -1] > 1.2)))
                        for k in meta.loc[meta.batch == b, "cell_key"])) for b in BATCH_NAMES}
    B = meta[(meta.batch == "batch1") & meta.labeled].merge(
        df[["cell_key", "slope_2_100", "dQD_100_10_mAh", "QDmax_minus_QD2_mAh", "slope_at_EOL"]], on="cell_key")
    ly = np.log10(B.cycle_life.to_numpy(float))
    lv = B.dQ_logvar.to_numpy(float)
    out: dict = {"n_batch1_labeled": int(len(B)), "qdlin_endpoint_invalid_cycles2_100": n_bad}
    # Batch 1 상관: 단독 Spearman, dQ_logvar 조건부 편상관
    out["batch1_corr"] = {}
    for f in ["Qcc_init", "QD_init", "QD_2", "slope_2_100", "dQD_100_10_mAh", "QDmax_minus_QD2_mAh"]:
        rho, p = stats.spearmanr(B[f], B.cycle_life)
        out["batch1_corr"][f] = {"spearman_rho": float(rho), "spearman_p": float(p),
                                 "partial_given_dQ_logvar": partial_corr(B[f], ly, [lv]),
                                 "partial_given_dQ_logvar_Qcc": (partial_corr(B[f], ly, [lv, B.Qcc_init.to_numpy()])
                                                                 if f != "Qcc_init" else None)}
    out["batch1_r_QD_init_vs_Qcc_init"] = float(np.corrcoef(B.QD_init, B.Qcc_init)[0, 1])
    out["batch1_r_dQ_logvar_vs_Qcc_init"] = float(np.corrcoef(lv, B.Qcc_init)[0, 1])
    # 기술 회귀(in-sample, CV 아님): log10 L ~ dQ_logvar + Qcc_init → 이동량을 예측 배율로 환산
    X = np.column_stack([np.ones(len(B)), lv, B.Qcc_init])
    beta = np.linalg.lstsq(X, ly, rcond=None)[0]
    out["batch1_ols_logL_on_dQ_logvar_Qcc_init"] = {"intercept": float(beta[0]), "b_dQ_logvar": float(beta[1]),
                                                    "b_Qcc_init_per_Ah": float(beta[2]),
                                                    "factor_per_plus10mAh": float(10 ** (beta[2] * 0.01))}
    # 라벨 없는 배치 이동량 (Batch 1 전체 46셀 중앙값 대비)
    out["batch_shift_vs_batch1_all46_median"] = {}
    for f in ["Qcc_init", "QD_init", "QD_2"]:
        mu, sd = float(meta.loc[meta.batch == "batch1", f].median()), float(B[f].std())
        rows = {}
        for (b, g), d in meta[meta.batch != "batch1"].groupby(["batch", "group"]):
            off = float(d[f].median() - mu)
            rows[f"{b}-{g}"] = {"n": int(len(d)), "shift_mAh": off * 1000, "shift_in_B1_SD": off / sd,
                                "pred_factor_with_Qcc_coef": float(10 ** (beta[2] * off)),
                                "n_outside_B1_labeled_range": int(((d[f] < B[f].min()) | (d[f] > B[f].max())).sum())}
        out["batch_shift_vs_batch1_all46_median"][f] = {"batch1_all46_median_Ah": mu, **rows}
    # 헤드룸 단순 환산: 곡선 모양이 같고 수직 이동만 있다고 가정할 때 +10 mAh 가 늘리는 수명 비율
    ext = 10.0 / np.abs(B.slope_at_EOL.to_numpy(float) / 100.0) / B.cycle_life.to_numpy(float)
    out["headroom_plus10mAh_extra_life_frac_batch1"] = {"median": float(np.median(ext)),
                                                        "q1": float(np.percentile(ext, 25)),
                                                        "q3": float(np.percentile(ext, 75)),
                                                        "slope_at_EOL_median": float(B.slope_at_EOL.median())}
    # 노션 상대 SOH(현재/초기) 80% 재라벨 가능 여부: 기록 안에서 자기 초기값의 80% 에 도달한 라벨 셀 수
    out["cells_reaching_80pct_of_own_initial_in_record"] = {}
    for b in BATCH_NAMES:
        d = meta[(meta.batch == b) & meta.labeled]
        hit = [bool(np.nanmin(clean_summary(by_key[k]["summary"]).QD) <= 0.8 * qi) for k, qi in zip(d.cell_key, d.QD_init)]
        out["cells_reaching_80pct_of_own_initial_in_record"][b] = {"n": int(len(d)), "n_reach": int(sum(hit))}
    return out


def summarise(df: pd.DataFrame, dec: pd.DataFrame, examples: list[str], perm: dict, cap: dict) -> dict:
    res: dict = {"definitions": __doc__.split("방법 정의")[1].split("실행:")[0].strip(),
                 "n_labeled": {b: int((df.batch == b).sum()) for b in BATCH_NAMES}, "per_batch": {}}
    cols = ["QD_init", "SOH_at_EOL_vs_init", "dQD_100_10_mAh", "slope_2_100", "QDmax_minus_QD2_mAh",
            "noise_sigma_mAh", "noise_sigma_trend_mAh", "snr_dQD_100_10", "snr_dQD_100_10_trend",
            "n_spike_gt3mAh_2_100", "slope_at_EOL",
            "rise_mAh", "rise_peak_cycle", "frac_life_above_init",
            "slope_mid_40_60", "slope_late_80_100", "late_over_mid", "slope_late_x_life", "loss_total_mAh",
            "share_loss_last20pct", "share_loss_first100", "SOH_at100_vs_init", "knee_bw", "knee_kneedle",
            "knee_ratio_bw", "knee_ratio_kneedle", "post_knee_cycles", "post_knee_frac", "slope_pre_knee",
            "slope_post_knee", "knee_sharpness", "SOH_at_knee_vs_init", "knee_sse_ratio_vs_linear",
            "onset_3seg", "onset_ratio", "second_break_3seg", "slope_seg1", "slope_seg2", "slope_seg3"]
    corr_cols = ["slope_2_100", "dQD_100_10_mAh", "QDmax_minus_QD2_mAh", "QD_init", "rise_mAh",
                 "slope_mid_40_60", "slope_late_80_100", "knee_bw", "knee_kneedle", "post_knee_cycles",
                 "knee_ratio_bw", "onset_3seg", "onset_ratio", "slope_pre_knee", "slope_post_knee",
                 "knee_sharpness", "share_loss_last20pct"]
    for b in BATCH_NAMES:
        d = df[df.batch == b]
        pb = {c: q(d[c]) for c in cols}
        pb["corr_with_cycle_life"] = {c: spearman(d[c], d.cycle_life) for c in corr_cols}
        pb["n_dQD_100_10_positive"] = int((d.dQD_100_10_mAh > 0).sum())
        pb["n_rise_gt_1mAh"] = int((d.rise_mAh > 1).sum())
        pb["n_cells_with_spike_gt3mAh_2_100"] = int((d.n_spike_gt3mAh_2_100 > 0).sum())
        pb["n_spikes_gt3mAh_2_100_total"] = int(d.n_spike_gt3mAh_2_100.sum())
        pb["n_abs_dQD_100_10_below_2sigma"] = int((d.dQD_100_10_mAh.abs() < 2 * d.noise_sigma_mAh).sum())
        pb["n_abs_dQD_100_10_below_2sigma_trend"] = int((d.dQD_100_10_mAh.abs() < 2 * d.noise_sigma_trend_mAh).sum())
        pb["n_knee_le_100_bw"] = int((d.knee_bw <= 100).sum())
        pb["n_knee_le_100_kneedle"] = int((d.knee_kneedle <= 100).sum())
        pb["n_onset_le_100"] = int((d.onset_3seg <= 100).sum())
        pb["knee_method_agreement"] = {
            "spearman_bw_vs_kneedle": float(stats.spearmanr(d.knee_bw, d.knee_kneedle)[0]),
            "median_bw_minus_kneedle_cycles": float((d.knee_bw - d.knee_kneedle).median()),
            "median_abs_diff_ratio": float((d.knee_ratio_bw - d.knee_ratio_kneedle).abs().median()),
            "median_bw_minus_onset_cycles": float((d.knee_bw - d.onset_3seg).median())}
        pb["knee_unclear_cells"] = d.loc[d.knee_unclear, ["cell_key", "policy", "cycle_life", "knee_sharpness",
                                                           "slope_2_100"]].to_dict("records")
        pb["decile_slope_median"] = dec[dec.batch == b].groupby("frac_mid")["slope"].median().round(4).to_dict()
        res["per_batch"][b] = pb
    res["knee_life_corr_permutation_baseline"] = perm
    # Batch 1 내부: 수명 3분위 (학습셋 근거)
    b1 = df[df.batch == "batch1"].reset_index(drop=True)
    tcols = ["slope_2_100", "dQD_100_10_mAh", "rise_mAh", "slope_mid_40_60", "slope_late_80_100",
             "knee_ratio_bw", "post_knee_frac"]
    res["batch1_life_tertiles"] = {
        t: {"n": int(len(d)), "life_min": float(d.cycle_life.min()), "life_max": float(d.cycle_life.max()),
            **{c: q(d[c]) for c in tcols}} for t, d in b1.groupby("b1_tertile")}
    # Batch 2 그룹별 (fastcharge / newstructure 혼재 효과 분리)
    d2 = df[(df.batch == "batch2") & (df.group == "fastcharge")]
    d2n = df[(df.batch == "batch2") & (df.group == "newstructure")]
    res["batch2_by_group"] = {
        "fastcharge": {"n": len(d2), "life_range": [float(d2.cycle_life.min()), float(d2.cycle_life.max())],
                       "slope_2_100": q(d2.slope_2_100), "dQD_100_10_mAh": q(d2.dQD_100_10_mAh),
                       "rise_mAh": q(d2.rise_mAh), "rise_peak_cycle": q(d2.rise_peak_cycle), "n_dQD_positive": int((d2.dQD_100_10_mAh > 0).sum()),
                       "corr_early_slope_life": spearman(d2.slope_2_100, d2.cycle_life)},
        "newstructure": {"n": len(d2n), "life_range": [float(d2n.cycle_life.min()), float(d2n.cycle_life.max())],
                         "slope_2_100": q(d2n.slope_2_100), "dQD_100_10_mAh": q(d2n.dQD_100_10_mAh),
                         "rise_mAh": q(d2n.rise_mAh), "rise_peak_cycle": q(d2n.rise_peak_cycle), "n_dQD_positive": int((d2n.dQD_100_10_mAh > 0).sum()),
                         "corr_early_slope_life": spearman(d2n.slope_2_100, d2n.cycle_life)}}
    # Batch 1 견고성: 준선형 셀 제외 시 + bootstrap CI
    d1 = df[(df.batch == "batch1") & (df.slope_2_100 >= FAST_EARLY)]
    rng = np.random.default_rng(RANDOM_STATE)
    boot = {"slope_2_100": [], "dQD_100_10_mAh": [], "QDmax_minus_QD2_mAh": []}
    for _ in range(2000):
        ix = rng.integers(0, len(b1), len(b1))
        for k in boot:
            boot[k].append(stats.spearmanr(b1.loc[ix, k], b1.loc[ix, "cycle_life"])[0])
    res["batch1_bootstrap_spearman_ci95"] = {k: [float(np.nanpercentile(v, 2.5)), float(np.nanpercentile(v, 97.5))]
                                             for k, v in boot.items()}
    pol = df.loc[df.slope_2_100 < FAST_EARLY, "policy"].unique()
    res["fast_early_policies_membership"] = {
        pp: {"n_cells_with_policy": int((df.policy == pp).sum()),
             "n_fast": int(((df.policy == pp) & (df.slope_2_100 < FAST_EARLY)).sum())} for pp in pol}
    res["n_fast_early_outside_these_policies"] = int(((~df.policy.isin(pol)) & (df.slope_2_100 < FAST_EARLY)).sum())
    res["batch1_robustness_excl_fast_linear"] = {
        "excluded": df[(df.batch == "batch1") & (df.slope_2_100 < FAST_EARLY)].cell_key.tolist(),
        "corr_early_slope_life": spearman(d1.slope_2_100, d1.cycle_life),
        "corr_dQD_100_10_life": spearman(d1.dQD_100_10_mAh, d1.cycle_life),
        "corr_QDmax_minus_QD2_life": spearman(d1.QDmax_minus_QD2_mAh, d1.cycle_life)}
    # 장/단수명 (노션 기준): 배치 통합(교락 주의) + 배치별 층화
    gcols = ["slope_2_100", "dQD_100_10_mAh", "slope_mid_40_60", "slope_late_80_100", "knee_bw",
             "knee_ratio_bw", "post_knee_cycles", "share_loss_last20pct"]
    res["life_group_comparison_pooled_CONFOUNDED_with_batch"] = {
        g: {"n": int(len(d)), "batch_counts": d.batch.value_counts().to_dict(),
            **{c: q(d[c]) for c in gcols}} for g, d in df.groupby("life_group")}
    res["life_group_comparison_by_batch"] = {
        b: {g: {"n": int(len(d)), "group_counts": d.group.value_counts().to_dict(),
                **{c: float(d[c].median()) for c in gcols}} for g, d in df[df.batch == b].groupby("life_group")}
        for b in BATCH_NAMES}
    # 배치 간 검정 (기술통계용)
    b1, b2, b3 = (df[df.batch == b] for b in BATCH_NAMES)
    mw = lambda a, b_: float(stats.mannwhitneyu(a, b_).pvalue)  # noqa: E731
    res["between_batch_tests_mannwhitney_p"] = {
        "early_slope_b1_vs_b2": mw(b1.slope_2_100, b2.slope_2_100),
        "early_slope_b1_vs_b3": mw(b1.slope_2_100, b3.slope_2_100),
        "knee_ratio_b1_vs_b2": mw(b1.knee_ratio_bw, b2.knee_ratio_bw),
        "knee_ratio_b1_vs_b3": mw(b1.knee_ratio_bw, b3.knee_ratio_bw),
        "post_knee_frac_b1_vs_b2": mw(b1.post_knee_frac, b2.post_knee_frac),
        "post_knee_frac_b1_vs_b3": mw(b1.post_knee_frac, b3.post_knee_frac),
        "QD_init_b1_vs_b2": mw(b1.QD_init, b2.QD_init),
        "QD_init_b1_vs_b3": mw(b1.QD_init, b3.QD_init),
        "late_slope_b1_vs_b2": mw(b1.slope_late_80_100, b2.slope_late_80_100),
        "late_slope_x_life_b1_vs_b2": mw(b1.slope_late_x_life, b2.slope_late_x_life),
        "late_slope_x_life_b1_vs_b3": mw(b1.slope_late_x_life, b3.slope_late_x_life),
        "rise_mAh_b1_vs_b2": mw(b1.rise_mAh, b2.rise_mAh),
        "rise_mAh_b1_vs_b3": mw(b1.rise_mAh, b3.rise_mAh)}
    res["ratios_b2_over_b1"] = {
        "late_slope_abs": float(b2.slope_late_80_100.median() / b1.slope_late_80_100.median()),
        "late_slope_x_life": float(b2.slope_late_x_life.median() / b1.slope_late_x_life.median()),
        "noise_sigma_diff2": float(b2.noise_sigma_mAh.median() / b1.noise_sigma_mAh.median()),
        "noise_sigma_trend": float(b2.noise_sigma_trend_mAh.median() / b1.noise_sigma_trend_mAh.median()),
        "rise_mAh": float(b2.rise_mAh.median() / b1.rise_mAh.median())}
    res["pooled"] = {"n": int(len(df)), "n_knee_le_100": int((df.knee_bw <= 100).sum()),
                     "n_onset_le_100": int((df.onset_3seg <= 100).sum()),
                     "min_knee_bw": float(df.knee_bw.min()), "min_knee_cell": df.loc[df.knee_bw.idxmin(), "cell_key"],
                     "min_onset": float(df.onset_3seg.min()), "min_onset_cell": df.loc[df.onset_3seg.idxmin(), "cell_key"],
                     "n_knee_unclear": int(df.knee_unclear.sum()),
                     "n_two_phase_shape": int((~df.knee_unclear).sum()),
                     "late_over_mid_range_of_batch_medians": [float(df.groupby("batch").late_over_mid.median().min()),
                                                              float(df.groupby("batch").late_over_mid.median().max())],
                     "fast_linear_cells_slope_lt_minus9": df.loc[df.slope_2_100 < FAST_EARLY,
                                                                    ["cell_key", "policy", "C1", "Q1", "C2", "cycle_life",
                                                                     "slope_2_100", "knee_sharpness"]].to_dict("records")}
    res["example_cells_in_q2_knee"] = examples
    res["q2d_initial_capacity_check"] = cap
    res["slope_convention"] = "용량 기울기 = 1e5 × Theil–Sen(clean QD[Ah] ~ cycle) [mAh/100 사이클, 음수 = 감소]"
    res["batchwide_dip_cycles45_56_gt3mAh"] = DIP_INFO
    res["cells"] = df.drop(columns=["bw_a0", "bw_a1", "bw_a2"]).round(6).to_dict("records")
    return res


def _clean(o):
    if isinstance(o, dict):
        return {str(k): _clean(v) for k, v in o.items()}
    if isinstance(o, (list, tuple)):
        return [_clean(v) for v in o]
    if isinstance(o, (np.integer,)):
        return int(o)
    if isinstance(o, (np.floating, float)):
        return None if not np.isfinite(o) else float(o)
    if isinstance(o, np.bool_):
        return bool(o)
    return o


def main() -> None:
    df, dec, _, cells = analyse()
    DIP_INFO.update(batch_dip_check(cells))
    fig_curves(df, cells)
    fig_fade(df, dec, cells)
    fig_early_slope(df)
    perm = knee_permutation(df)
    examples = fig_knee(df, cells, perm["batch1"])
    cap = initial_capacity_check(df, cells)
    res = summarise(df, dec, examples, perm, cap)
    OUT_JSON.write_text(json.dumps(_clean(res), ensure_ascii=False, indent=2))
    # 콘솔 요약
    show = ["QD_init", "dQD_100_10_mAh", "slope_2_100", "noise_sigma_mAh", "noise_sigma_trend_mAh",
            "snr_dQD_100_10", "snr_dQD_100_10_trend", "rise_mAh", "rise_peak_cycle", "frac_life_above_init",
            "slope_mid_40_60", "slope_late_80_100", "late_over_mid", "slope_late_x_life", "share_loss_last20pct",
            "knee_bw", "knee_ratio_bw", "knee_ratio_kneedle", "post_knee_cycles", "post_knee_frac",
            "onset_3seg", "onset_ratio", "knee_sharpness"]
    pd.set_option("display.width", 220)
    print(df.groupby("batch")[show].median().round(3).T)
    for b in BATCH_NAMES:
        cc = res["per_batch"][b]["corr_with_cycle_life"]
        print(b, {k: round(v["spearman_rho"], 3) for k, v in cc.items()})
    print("knee unclear:", df.loc[df.knee_unclear, ["cell_key", "policy", "knee_sharpness"]].to_string())
    print("examples:", examples)
    print("perm:", json.dumps(_clean(perm)))
    print("B1 tertiles:", json.dumps(_clean({t: {k: (v["median"] if isinstance(v, dict) else v) for k, v in d.items()}
                                             for t, d in res["batch1_life_tertiles"].items()}), ensure_ascii=False))
    print("MW:", json.dumps(_clean(res["between_batch_tests_mannwhitney_p"])))
    print("ratios:", json.dumps(_clean(res["ratios_b2_over_b1"])))
    print("Q2-(d):", json.dumps(_clean(cap), ensure_ascii=False))
    print("saved", OUT_JSON)


if __name__ == "__main__":
    main()
