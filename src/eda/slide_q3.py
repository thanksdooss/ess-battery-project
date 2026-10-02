"""DAY 1 슬라이드용 Q3 그림 — '짧게 사는 셀일수록 ΔQ가 깊고, 깊이 하나가 수명과 직선 관계'.

  (왼쪽) Batch 1(학습) 36셀의 ΔQ₁₀₀₋₁₀(V) 곡선, 선 색 = 수명(빨강 = 짧음, 파랑 = 긺).
         y 단위는 mAh(읽기 쉬운 정수 눈금). 색 막대 대신 직접 라벨('수명 짧은 셀'/'수명 긴 셀')과
         가장 깊은 구간(2.9~3.0 V) 띠만 표시.
  (오른쪽) ΔQ 깊이 = log₁₀ var(ΔQ₁₀₀₋₁₀) vs 수명(로그 축).
         Batch 1 은 진하게 + 추세선(설명용, B1 범위 안) + r, Batch 2/3 은 옅은 점(일반화 점검용).
         Batch 2 고속 충전 점 무리(392~514, ΔQ 깊이 ≥ −3.71)가 선 아래 → 과대예측 위험 주석.

슬라이드(16:9, 그림 영역 약 178 × 112 mm)에 그대로 넣는 그림이라 큰 제목은 없고(헤드라인이 메시지 담당),
글자는 18 cm 폭으로 줄여도 읽히도록 크게 둔다.

수치 출처(그림에 그리는 숫자는 모두 아래 파일과 대조 후 assert):
  - results/eda/q3_results.json : meta.n_labeled(36/39/44), a_curves.batch1.V_at_min_median(2.93 V),
    c_features.batch1_trend_loglife_vs_logvar(r −0.84, 기울기·절편, B1 log var 범위)
  - results/eda/q3_final.json  : '2.9~3.0 V', 'r −0.84' 문구

실행: cd ess-battery-project && python src/eda/slide_q3.py
산출: reports/figures/slide_q3.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import numpy as np  # noqa: E402
from scipy import stats  # noqa: E402

import plot_style as ps  # noqa: E402  (matplotlib Agg + 한글 폰트)
from data import BATCH_NAMES, cell_table, delta_q, load_all  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
RES = json.loads((ROOT / "results" / "eda" / "q3_results.json").read_text())
FIN = json.loads((ROOT / "results" / "eda" / "q3_final.json").read_text())

# ───────────────────────────── 데이터 ─────────────────────────────
cells = load_all()
ct = cell_table(cells).set_index("cell_key")
lab = [c for c in cells if ct.loc[c["cell_key"], "labeled"]]
V = np.asarray(lab[0]["Vdlin"], float)

D = {b: [] for b in BATCH_NAMES}          # batch → list of (cycle_life, ΔQ curve)
for c in lab:
    D[c["batch"]].append((float(c["cycle_life"]), delta_q(c, 100, 10)))

life = {b: np.array([x[0] for x in D[b]]) for b in BATCH_NAMES}
curves = {b: np.array([x[1] for x in D[b]]) for b in BATCH_NAMES}
logvar = {b: np.log10(curves[b].var(axis=1)) for b in BATCH_NAMES}   # np.var ddof=0 (q3 정의와 동일)

# ── 출처 대조 ──
for b in BATCH_NAMES:
    assert len(life[b]) == RES["meta"]["n_labeled"][b], b
lr = stats.linregress(logvar["batch1"], np.log10(life["batch1"]))
TR = RES["c_features"]["batch1_trend_loglife_vs_logvar"]
assert abs(lr.rvalue - TR["r"]) < 1e-4 and abs(lr.slope - TR["slope"]) < 1e-4 and abs(lr.intercept - TR["intercept"]) < 1e-4
assert abs(lr.rvalue - RES["a_curves"]["r_log_var_vs_log_life"]["batch1"]) < 1e-4
assert np.allclose([logvar["batch1"].min(), logvar["batch1"].max()], TR["batch1_log_var_range"], atol=1e-4)
v_at_min = np.median(V[curves["batch1"].argmin(axis=1)])
assert abs(v_at_min - RES["a_curves"]["batch1"]["V_at_min_median"]) < 1e-3       # 2.93 V
assert "2.9~3.0 V" in FIN["before_after"] and "r −0.84" in FIN["before_after"]
R_TXT = f"r = {lr.rvalue:.2f}".replace("-", "−")                                   # 'r = −0.84'
N1 = len(life["batch1"])                                                            # 36

# ───────────────────────────── 스타일 ─────────────────────────────
INK, MUTED = "#0B1220", "#475569"
LIFE_CMAP = mcolors.LinearSegmentedColormap.from_list("life", ["#B91C1C", ps.SHORT, "#BDBDBD", ps.LONG, "#1E3A8A"])
LIFE_NORM = mcolors.LogNorm(vmin=life["batch1"].min(), vmax=life["batch1"].max())   # B1 534 ~ 1,074
RC = {"font.size": 13, "axes.titlesize": 14.5, "axes.labelsize": 13.5, "xtick.labelsize": 12,
      "ytick.labelsize": 12, "legend.fontsize": 12, "axes.titleweight": "bold", "axes.titlepad": 9,
      "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#94A3B8",
      "grid.alpha": 0.22}

with plt.rc_context(RC):
    fig, (axL, axR) = plt.subplots(1, 2, figsize=(8.6, 5.55), gridspec_kw={"wspace": 0.34})
    fig.subplots_adjust(left=0.105, right=0.985, bottom=0.135, top=0.915)

    # ── ① 왼쪽: Batch 1 ΔQ 곡선 (선 색 = 수명, 빨강 짧음 → 회색 → 파랑 긺) ──
    axL.axvspan(2.9, 3.0, color="#FDE68A", alpha=0.6, lw=0, zorder=0)
    mid = np.sqrt(LIFE_NORM.vmin * LIFE_NORM.vmax)
    for i in np.argsort(-np.abs(np.log(life["batch1"] / mid))):          # 수명 극단(진한 색)이 위에
        axL.plot(V, curves["batch1"][i] * 1000, color=LIFE_CMAP(LIFE_NORM(life["batch1"][i])), lw=1.25, alpha=0.95)
    axL.axhline(0, color="#64748B", lw=0.9)
    axL.set_xlim(2.0, 3.5)
    axL.set_ylim(-77, 5)
    axL.xaxis.set_major_locator(FixedLocator([2.0, 2.5, 3.0, 3.5]))
    axL.yaxis.set_major_locator(FixedLocator([0, -20, -40, -60]))
    axL.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.0f}".replace("-", "−")))
    axL.set_xlabel("방전 전압 (V)")
    axL.set_ylabel("ΔQ = Q(100) − Q(10)  (mAh)")
    axL.set_title(f"① Batch 1(학습) {N1}셀의 ΔQ 곡선", loc="left")
    # 핵심 표시 3개: 가장 깊은 구간 / 짧은 수명 = 깊음 / 긴 수명 = 얕음
    axL.text(2.95, -74.5, "가장 깊은 곳 2.9~3.0 V", ha="center", va="bottom", fontsize=12.5, color="#92400E",
             fontweight="bold")
    axL.annotate("수명 짧은 셀", xy=(2.70, -46.0), xytext=(2.04, -50.5), ha="left", va="center", fontsize=13,
                 color=ps.SHORT, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=ps.SHORT, lw=1.3, mutation_scale=12, shrinkA=2, shrinkB=1))
    axL.annotate("수명 긴 셀", xy=(2.915, -24.0), xytext=(2.915, -9.5), ha="center", va="center", fontsize=13,
                 color=ps.LONG, fontweight="bold",
                 arrowprops=dict(arrowstyle="-|>", color=ps.LONG, lw=1.3, mutation_scale=12, shrinkA=4, shrinkB=1))

    # ── ② 오른쪽: ΔQ 깊이 vs 수명 ──
    h1 = axR.scatter(logvar["batch1"], life["batch1"], s=54, color=ps.BATCH_COLOR["batch1"], edgecolor="white",
                     lw=0.9, label="Batch 1(학습)", zorder=4)
    h2 = axR.scatter(logvar["batch2"], life["batch2"], s=26, color=ps.BATCH_COLOR["batch2"], alpha=0.30, lw=0,
                     label="Batch 2(테스트)", zorder=2)
    h3 = axR.scatter(logvar["batch3"], life["batch3"], s=26, color=ps.BATCH_COLOR["batch3"], alpha=0.30, lw=0,
                     label="Batch 3(추가 검증)", zorder=2)
    xs = np.linspace(logvar["batch1"].min(), logvar["batch1"].max(), 50)   # 추세선은 B1 범위 안만(설명용)
    axR.plot(xs, 10 ** (lr.intercept + lr.slope * xs), color=INK, lw=2.4, zorder=5, solid_capstyle="round")
    x_pt = -3.47
    axR.annotate(R_TXT, xy=(x_pt, 10 ** (lr.intercept + lr.slope * x_pt)), xytext=(-3.30, 880),
                 ha="center", va="center", fontsize=16, fontweight="bold", color=INK,
                 arrowprops=dict(arrowstyle="-", color=INK, lw=1.0, shrinkA=9, shrinkB=2))
    # Batch 2 고속 충전 셀(점 무리)이 B1 추세선보다 아래 → 과대예측 위험(부록 Gap 가설 H1), 설명용 주석
    axR.annotate("Batch 2 고속 충전은\n선보다 아래\n→ 과대예측 위험", xy=(-3.74, 455), xytext=(-5.12, 470),
                 ha="left", va="center", fontsize=11.5, color="#C2410C", fontweight="bold", linespacing=1.2,
                 arrowprops=dict(arrowstyle="-|>", color="#C2410C", lw=1.1, mutation_scale=11, shrinkA=4, shrinkB=2))
    axR.set_yscale("log")
    axR.yaxis.set_major_locator(FixedLocator([400, 700, 1000, 2000]))
    axR.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:,.0f}"))
    axR.yaxis.set_minor_locator(NullLocator())
    axR.set_ylim(340, 2300)
    axR.set_xlim(-5.2, -2.9)
    axR.xaxis.set_major_locator(FixedLocator([-5.0, -4.5, -4.0, -3.5, -3.0]))
    axR.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:.1f}".replace("-", "−")))
    axR.set_xlabel("ΔQ 깊이 = log₁₀ var(ΔQ)   깊을수록 →")
    axR.set_ylabel("수명 (사이클, 로그 축)")
    axR.set_title("② ΔQ 깊이 vs 수명", loc="left")
    leg = axR.legend(handles=[h1, h2, h3], loc="upper right", handletextpad=0.2, borderaxespad=0.1,
                     labelspacing=0.35, markerscale=1.25)
    for t in leg.get_texts():
        t.set_color(INK)

    ps.save(fig, "slide_q3")

print("saved", ps.FIG / "slide_q3.png")
print({"n": {b: len(life[b]) for b in BATCH_NAMES}, "r_b1": round(lr.rvalue, 4), "V_at_min_med": round(float(v_at_min), 3),
       "b1_life": [life['batch1'].min(), life['batch1'].max()]})
