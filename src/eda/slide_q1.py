"""DAY 1 슬라이드용 Q1 그림 — Cycle Life 분포 (배치 3개를 한 패널에 위아래로 쌓은 히스토그램).

슬라이드 메시지: 배치마다 분포가 다르고, Batch 1에는 단수명(<500)이 없다(<550도 단 1개).

  · x축 150 ~ 2,300 (노션 「150 ~ 2,300 사이클 Histogram」), 50사이클 bin — 세 배치 동일
  · 세로선 500(단수명) · 550(분류 라벨 기준) · 1,000(장수명)
  · 레인마다 n(라벨 셀), 단수명(<500) 수, 장수명(>1,000) 수
  · 하한값만 아는 셀(B1 중도절단 10개, B3 EOL 미도달 2개)은 빈도 막대와 분리해 ▷ 마커로 표시

수치 원천: src/data.py(cell_table)로 다시 계산하고, results/eda/q1_results.json 의
b_ratios[*].labeled_* · a_descriptive[*].lower_bounds 와 일치하는지 assert 로 확인한다.

실행:  python src/eda/slide_q1.py   →  reports/figures/slide_q1.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.colors import to_rgb  # noqa: E402
from matplotlib.patches import FancyBboxPatch  # noqa: E402

import plot_style as ps  # noqa: E402
from data import BATCH_NAMES, LABEL_THRESHOLD, cell_table, load_all  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
RES = json.loads((ROOT / "results" / "eda" / "q1_results.json").read_text())

HIST_RANGE = (150, 2300)     # 노션 「150 ~ 2,300 사이클 Histogram」
BIN_W = 50                   # q1_cycle_life.py 와 같은 bin (43개)
SHORT_T, LONG_T = 500, 1000  # 노션 「장수명(>1,000)/단수명(<500)」
LBL_T = LABEL_THRESHOLD      # 550 (분류 라벨 기준)

INK, INK2, MUTED, LINE = "#0B1220", "#334155", "#64748B", "#CBD5E1"
SHORT_ZONE = "#FDF1F1"       # 단수명 구간(<500) 옅은 배경

plt.rcParams.update({"font.size": 13, "axes.labelsize": 13.5, "xtick.labelsize": 12, "ytick.labelsize": 12,
                     "axes.grid": False})


def tint(hex_color: str, w: float) -> tuple:
    r, g, b = to_rgb(hex_color)
    return (r + (1 - r) * w, g + (1 - g) * w, b + (1 - b) * w)


# ── 데이터 (q1_cycle_life.py 와 같은 정의) ──────────────────────────
T = cell_table(load_all())
T["life_test"] = T["group"].isin(["fastcharge", "newstructure"])
T["lower_bound"] = np.where(T["censored"], T["cycle_life"],
                            np.where(T["life_test"] & T["cycle_life"].isna(), T["n_cycles"], np.nan))
LAB = T[T["labeled"]]
LB = T[T["lower_bound"].notna()]

stats = {}
for b in BATCH_NAMES:
    x = LAB[LAB.batch == b].cycle_life.to_numpy()
    lb = np.sort(LB[LB.batch == b].lower_bound.to_numpy())
    s = {"n": len(x), "short": int((x < SHORT_T).sum()), "long": int((x > LONG_T).sum()),
         "lt550": int((x < LBL_T).sum()), "lb": lb, "x": x}
    # 원천 JSON 과 대조 (그림에 그리는 모든 숫자)
    r = RES["b_ratios"][b]
    assert s["n"] == r["n_labeled"] == RES["a_descriptive"][b]["all_labeled"]["n"], b
    assert s["short"] == r["labeled_short_lt500"], b
    assert s["long"] == r["labeled_long_gt1000"], b
    assert s["lt550"] == r["labeled_lt550"], b
    assert lb.tolist() == RES["a_descriptive"][b]["lower_bounds"], b
    stats[b] = s
assert stats["batch1"]["lt550"] == 1 and len(stats["batch1"]["lb"]) == 10

# ── 그림: 한 패널 안에 3개 레인 (같은 x축·같은 bin·같은 세로 눈금) ─────────
bins = np.arange(HIST_RANGE[0], HIST_RANGE[1] + 1, BIN_W)
LANE_H, GAP = 17.0, 2.5                       # 레인 높이(셀 수 단위)와 레인 사이 간격
base = {b: (len(BATCH_NAMES) - 1 - i) * (LANE_H + GAP) for i, b in enumerate(BATCH_NAMES)}
HEAD = {"batch1": "Batch 1(학습)", "batch2": "Batch 2(테스트)", "batch3": "Batch 3(추가 검증)"}

fig = plt.figure(figsize=(8.6, 5.55))
ax = fig.add_axes([0.085, 0.125, 0.9, 0.80])
top = base["batch1"] + LANE_H
ax.set_xlim(*HIST_RANGE)
ax.set_ylim(-0.6, top + 0.4)

# 단수명 구간 배경 + 기준선 3개
ax.axvspan(HIST_RANGE[0], SHORT_T, color=SHORT_ZONE, lw=0, zorder=0)
for x0, ls, lw, c in ((SHORT_T, (0, (4, 3)), 1.3, MUTED), (LBL_T, "-", 1.3, INK), (LONG_T, (0, (4, 3)), 1.3, MUTED)):
    ax.axvline(x0, color=c, ls=ls, lw=lw, zorder=3)
tr = ax.get_xaxis_transform()
ax.text(SHORT_T - 12, 1.012, "단수명 <500", transform=tr, ha="right", va="bottom", fontsize=12, color=INK2)
ax.text(LBL_T + 12, 1.012, "550 분류 기준", transform=tr, ha="left", va="bottom", fontsize=12, color=INK,
        fontweight="bold")
ax.text(LONG_T + 12, 1.012, "장수명 >1,000", transform=tr, ha="left", va="bottom", fontsize=12, color=INK2)

yt, ytl = [], []
for b in BATCH_NAMES:
    y0, s, col = base[b], stats[b], ps.BATCH_COLOR[b]
    cnt = np.histogram(s["x"], bins)[0]
    assert cnt.sum() == s["n"], b                      # 모든 라벨 셀이 150~2,300 안에 있음
    # 눈금선(0·10)과 기준선
    ax.plot(HIST_RANGE, [y0 + 10] * 2, color=LINE, lw=0.8, ls=(0, (2, 3)), zorder=1)
    ax.plot(HIST_RANGE, [y0, y0], color="#94A3B8", lw=1.0, zorder=4)
    yt += [y0, y0 + 10]
    ytl += ["0", "10"]
    ax.bar(bins[:-1], cnt, width=BIN_W, align="edge", bottom=y0, color=col, edgecolor="white", linewidth=1.0,
           zorder=2)
    # 레인 요약 (오른쪽 위)
    t1 = ax.text(HIST_RANGE[1] - 15, y0 + LANE_H - 0.3, f"{HEAD[b]}  n={s['n']}", ha="right", va="top",
                 fontsize=13.5, fontweight="bold", color=INK, zorder=6)
    ax.annotate("■ ", xy=(0, 0.5), xycoords=t1, ha="right", va="center", fontsize=13.5, color=col)
    ax.text(HIST_RANGE[1] - 15, y0 + LANE_H - 4.3,
            f"단수명(<500) {s['short']}개 · 장수명(>1,000) {s['long']}개", ha="right", va="top",
            fontsize=12.5, color=INK2, zorder=6)
    # 하한값만 아는 셀: ▷ 마커를 같은 50사이클 구간 중앙에 쌓는다 (빈도 막대와 분리된 옅은 띠)
    lb = s["lb"]
    if len(lb):
        k = np.floor((lb - HIST_RANGE[0]) / BIN_W).astype(int)
        xc = HIST_RANGE[0] + (k + 0.5) * BIN_W
        rows = np.array([int((k[:i] == k[i]).sum()) for i in range(len(k))])
        step = 1.8
        yb = y0 + (10.0 if b == "batch1" else 1.15)
        ys = yb + rows * step
        x_lo, x_hi = xc.min() - BIN_W * 0.75, xc.max() + BIN_W * 0.75
        ax.add_patch(FancyBboxPatch((x_lo, yb - 0.9), x_hi - x_lo, rows.max() * step + 1.8,
                                    boxstyle="round,pad=0,rounding_size=12", facecolor=tint(col, 0.85),
                                    edgecolor="none", zorder=1.5, mutation_aspect=0.08))
        ax.plot(xc, ys, ls="none", marker=">", ms=8, mfc=tint(col, 0.45), mec=col, mew=1.2, zorder=5)
        lab = f"▷ 중도절단 {len(lb)}개(하한값)" if b == "batch1" else f"▷ EOL 미도달 {len(lb)}개(하한값)"
        if b == "batch1":
            ax.text((x_lo + x_hi) / 2, yb + rows.max() * step + 1.2, lab, ha="center", va="bottom",
                    fontsize=12, color=INK2, zorder=6,
                    bbox=dict(boxstyle="square,pad=0.1", fc="white", ec="none"))
        else:
            ax.text(HIST_RANGE[1] - 15, yb + rows.max() * step + 1.3, lab, ha="right", va="bottom",
                    fontsize=12, color=INK2, zorder=6)

# 핵심 메시지 표시: Batch 1의 550 미만 단 1개
y1 = base["batch1"]
ax.annotate("550 미만\n단 1개", xy=(LBL_T - BIN_W / 2, y1 + 1.15), xytext=(LBL_T - 95, y1 + 5.4),
            ha="right", va="center", fontsize=12.5, fontweight="bold", color=INK, linespacing=1.15,
            arrowprops=dict(arrowstyle="-|>", color=INK, lw=1.2, shrinkA=2, shrinkB=1,
                            connectionstyle="arc3,rad=-0.25"), zorder=7)

ax.set_yticks(yt)
ax.set_yticklabels(ytl)
ax.tick_params(axis="y", length=0, pad=4, colors=MUTED)
ax.set_ylabel("셀 수", color=INK2)
ax.set_xticks([150, 500, 1000, 1500, 2000, 2300])
ax.set_xticklabels(["150", "500", "1,000", "1,500", "2,000", "2,300"])
ax.set_xticks(np.arange(250, 2300, 250), minor=True)
ax.tick_params(axis="x", which="major", length=4, color=MUTED)
ax.tick_params(axis="x", which="minor", length=2.5, color=LINE)
ax.set_xlabel("Cycle Life (사이클, 50 단위 구간 · 세 배치 동일)", color=INK2)
for sp in ("left", "top", "right"):
    ax.spines[sp].set_visible(False)
ax.spines["bottom"].set_visible(False)

out = ps.save(fig, "slide_q1")
print(out)
for b in BATCH_NAMES:
    s = stats[b]
    print(b, {k: (v if k not in ("x",) else None) for k, v in s.items() if k != "x"})
