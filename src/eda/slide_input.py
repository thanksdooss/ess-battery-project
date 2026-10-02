"""DAY 1 슬라이드용 INPUT 그림 — '학습(Batch 1)과 테스트(Batch 2)의 수명 범위·구성이 다르다'.

  (위) 배치별 셀 구성: 가로 누적 막대 1줄 = 배치 1개.
       진한 색 = fastcharge(라벨), 옅은 색 = newstructure(라벨), 빗금 = 중도절단(하한값만 있음),
       회색 = 라벨 없음(NaN: Batch 2 varcharge·slowcycle, Batch 3 EOL 미도달).
  (아래) 라벨 셀의 수명(cycle_life) 점 분포 + 그룹별 최소~최대 범위, 550(장/단수명 경계) 점선,
       Batch 1(학습) 라벨 범위를 세로 띠로 표시해 테스트 셀이 학습 범위 밖에 있음을 보여 준다.

슬라이드(16:9, 그림 영역 약 178 × 112 mm)에 그대로 넣는 그림이라 큰 제목은 없고(헤드라인이 메시지 담당),
글자는 18 cm 폭으로 줄여도 읽히도록 크게 둔다(기본 13 pt, 축 제목 13.5 pt, 눈금 12.5 pt, 주석 11.5~12 pt).
figsize (8.6, 5.25) in → bbox tight 저장 후 가로:세로 ≈ 1.53 (슬라이드 그림 칸 ≈ 1.55 에 맞춤).
막대 안 글자는 조각 폭을 넘으면 12 → 11.5 → 11 pt 로만 줄이고, 그래도 넘치면 assert 로 멈춘다.

수치 출처(그림에 그리는 숫자는 모두 아래 파일과 대조 후 assert):
  - results/eda/q1_results.json : a_descriptive.{batch}.{group}.n/min/max (36 · 30 · 9 · 44셀,
    534~1,074 / 392~514 / 777~1,186 / 541~1,935), a_descriptive.batch1.n_lower_bound_only(10),
    a_descriptive.batch3.lower_bounds(2셀), b_ratios.{batch}.labeled_lt550(1 / 30 / 1)
  - results/eda/q1_final.json   : '534~1,074', '392~514', '777~1,186', '541~1,935' 문구
  - reports/day1_design_report.md §1 INPUT 표 : 라벨/중도절단/NaN = 36/10/0 · 39/0/8 · 44/0/2,
    구성 fastcharge 46 / fastcharge 30·newstructure 9·varcharge 4·slowcycle 4 / newstructure 46

실행: cd ess-battery-project && python src/eda/slide_input.py
산출: reports/figures/slide_input.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import numpy as np  # noqa: E402

import plot_style as ps  # noqa: E402  (matplotlib Agg + 한글 폰트)
from data import BATCH_NAMES, LABEL_THRESHOLD, cell_table, load_all  # noqa: E402

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import colors as mcolors  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
RES = json.loads((ROOT / "results" / "eda" / "q1_results.json").read_text())
FIN = (ROOT / "results" / "eda" / "q1_final.json").read_text()
RPT = (ROOT / "reports" / "day1_design_report.md").read_text()

# ───────────────────────────── 데이터 ─────────────────────────────
ct = cell_table(load_all())
ct["state"] = np.where(ct["labeled"], "labeled", np.where(ct["censored"], "censored", "nan"))
# NaN 이 아닌데 라벨도 중도절단도 아닌 셀은 없어야 한다
assert not (ct["state"].eq("nan") & np.isfinite(ct["cycle_life"])).any()

cnt = ct.groupby(["batch", "group", "state"]).size().to_dict()
lab = ct[ct["labeled"]]

# ── 출처 대조 1: 구성 (보고서 §1 INPUT 표) ──
EXPECT_COUNTS = {
    ("batch1", "fastcharge", "labeled"): 36, ("batch1", "fastcharge", "censored"): 10,
    ("batch2", "fastcharge", "labeled"): 30, ("batch2", "newstructure", "labeled"): 9,
    ("batch2", "varcharge", "nan"): 4, ("batch2", "slowcycle", "nan"): 4,
    ("batch3", "newstructure", "labeled"): 44, ("batch3", "newstructure", "nan"): 2,
}
assert cnt == EXPECT_COUNTS, cnt
for row in ("| fastcharge 46 | 36 / 10 / 0 |",
            "| fastcharge 30 · newstructure 9 · varcharge 4 · slowcycle 4 | 39 / 0 / 8 |",
            "| newstructure 46 | 44 / 0 / 2 |"):
    assert row in RPT, row
assert RES["a_descriptive"]["batch1"]["n_lower_bound_only"] == 10
assert len(RES["a_descriptive"]["batch3"]["lower_bounds"]) == 2

# ── 출처 대조 2: 라벨 셀 수명 범위 (q1_results.json a_descriptive, q1_final.json 문구) ──
GROUPS = {"batch1": ["fastcharge"], "batch2": ["fastcharge", "newstructure"], "batch3": ["newstructure"]}
RANGE = {}
for b, gs in GROUPS.items():
    for g in gs:
        x = lab.query("batch == @b and group == @g")["cycle_life"].to_numpy()
        ref = RES["a_descriptive"][b][g]
        assert len(x) == ref["n"] and x.min() == ref["min"] and x.max() == ref["max"], (b, g)
        RANGE[(b, g)] = (int(x.min()), int(x.max()), len(x))
for s in ("534~1,074", "392~514", "777~1,186", "541~1,935"):
    assert s in FIN, s
LT550 = {b: int((lab.query("batch == @b")["cycle_life"] < LABEL_THRESHOLD).sum()) for b in BATCH_NAMES}
for b in BATCH_NAMES:
    assert LT550[b] == RES["b_ratios"][b]["labeled_lt550"], b
assert LT550 == {"batch1": 1, "batch2": 30, "batch3": 1}

# ───────────────────────────── 스타일 ─────────────────────────────
INK, MUTED, FAINT = "#0B1220", "#475569", "#94A3B8"
NAN_GRAY = "#CBD5E1"
FS = 13.0          # 기본 글자
FS_TICK = 12.5     # 눈금
FS_NOTE = 12.0     # 그림 안 주석
plt.rcParams.update({"font.size": FS, "axes.titlesize": 13.5, "axes.labelsize": 13.5,
                     "xtick.labelsize": FS_TICK, "ytick.labelsize": FS_TICK, "hatch.linewidth": 1.1})


def tint(hex_color: str, a: float) -> tuple:
    """흰색과 섞은 옅은 색 (a = 원색 비율)."""
    r, g, b = mcolors.to_rgb(hex_color)
    return (1 - a + a * r, 1 - a + a * g, 1 - a + a * b)


def kfmt(v: float) -> str:
    return f"{int(v):,}"


YLAB = {"batch1": "Batch 1\n(학습)", "batch2": "Batch 2\n(테스트)", "batch3": "Batch 3\n(추가 검증)"}
Y = {b: i for i, b in enumerate(BATCH_NAMES)}          # 위에서 아래로 Batch 1 → 3 (축 반전)

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.6, 5.25), gridspec_kw={"height_ratios": [1.0, 1.25], "hspace": 0.30})
FIT = []   # (글자 artist, 조각 [x0, x1]) — 막대 안 글자가 조각 폭 안에 들어가는지 저장 후 점검

# ───────────────────────── (위) 셀 구성 누적 막대 ─────────────────────────
# (그룹, 상태, 막대 안 글자, 채우기 종류)
SEGS = {
    "batch1": [("fastcharge", "labeled", "라벨 36", "solid"),
               ("fastcharge", "censored", "중도절단 10", "hatch")],
    "batch2": [("fastcharge", "labeled", "fastcharge 30", "solid"),
               ("newstructure", "labeled", "newstructure 9", "light"),
               ("varcharge", "nan", "4", "gray"),
               ("slowcycle", "nan", "4", "gray")],
    "batch3": [("newstructure", "labeled", "라벨 44", "light"),
               ("newstructure", "nan", "2", "gray")],
}
BAR_H = 0.56
GAP = 0.16   # 조각 사이 흰 틈 (셀 단위)
for b, segs in SEGS.items():
    col = ps.BATCH_COLOR[b]
    x0, y = 0.0, Y[b]
    for g, st, text, kind in segs:
        n = cnt[(b, g, st)]
        w = n - GAP
        if kind == "solid":
            fc, ec, hatch, tc = col, col, None, "white"
        elif kind == "light":
            fc, ec, hatch, tc = tint(col, 0.30), tint(col, 0.30), None, INK
        elif kind == "hatch":
            fc, ec, hatch, tc = "white", tint(col, 0.55), "////", INK
        else:
            fc, ec, hatch, tc = NAN_GRAY, NAN_GRAY, None, INK
        ax1.barh(y, w, left=x0, height=BAR_H, color=fc, edgecolor=ec, hatch=hatch,
                 linewidth=0 if hatch is None else 1.1)
        kw = dict(ha="center", va="center", fontsize=FS_NOTE, color=tc,
                  fontweight="bold" if kind != "gray" else "normal")
        if kind == "hatch":   # 빗금 위 글자가 읽히도록 흰 바탕
            kw["bbox"] = dict(boxstyle="round,pad=0.15", fc="white", ec="none")
        FIT.append((ax1.text(x0 + w / 2, y, text, **kw), (x0, x0 + w)))
        x0 += n
    total = int(ct.query("batch == @b").shape[0])
    ax1.text(x0 + 0.5, y, f"{total}셀", ha="left", va="center", fontsize=FS_NOTE, color=MUTED)

# 회색(라벨 없음) 조각 설명 — 막대 바로 아래, 오른쪽 끝 정렬
NOTE_Y = BAR_H / 2 + 0.05
ax1.text(47 - GAP, Y["batch2"] + NOTE_Y, "varcharge 4 · slowcycle 4 → 라벨 없음", ha="right", va="top",
         fontsize=11.5, color=MUTED)
ax1.text(46 - GAP, Y["batch3"] + NOTE_Y, "46셀 모두 newstructure · 2 = EOL 미도달 → 라벨 없음", ha="right", va="top",
         fontsize=11.5, color=MUTED)
ax1.text(46 - GAP, Y["batch1"] + NOTE_Y, "46셀 모두 fastcharge · 중도절단 = 하한값만 있음", ha="right", va="top",
         fontsize=11.5, color=MUTED)

ax1.set_xlim(0, 50.6)
ax1.set_ylim(2.68, -0.36)
ax1.yaxis.set_major_locator(FixedLocator([Y[b] for b in BATCH_NAMES]))
ax1.set_yticklabels([YLAB[b] for b in BATCH_NAMES], fontsize=FS_TICK)
ax1.tick_params(axis="y", length=0, pad=6)
ax1.set_xticks([])
for s in ("left", "bottom", "top", "right"):
    ax1.spines[s].set_visible(False)
ax1.grid(False)
ax1.set_title("① 셀 구성", loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=4)

# ───────────────────────── (아래) 라벨 셀 수명 점 분포 ─────────────────────────
XMIN, XMAX = 300, 2050
b1lo, b1hi, _ = RANGE[("batch1", "fastcharge")]
ax2.axvspan(b1lo, b1hi, color=tint(ps.BATCH_COLOR["batch1"], 0.13), zorder=0, lw=0)

RED = "#B91C1C"
ax2.axvline(LABEL_THRESHOLD, color=RED, lw=1.5, ls=(0, (4, 3)), zorder=1)
# 550 설명은 Batch 3 행 왼쪽 빈 곳(라벨 셀 최소 541 > 그 왼쪽은 비어 있음)
ax2.text(LABEL_THRESHOLD - 22, Y["batch3"] - 0.42, f"{LABEL_THRESHOLD}", ha="right", va="bottom",
         fontsize=FS_NOTE, color=RED, fontweight="bold")
ax2.text(LABEL_THRESHOLD - 22, Y["batch3"] - 0.40, "장/단수명\n경계", ha="right", va="top",
         fontsize=11.5, color=RED, linespacing=1.05)

rng = np.random.default_rng(42)
JIT = 0.15
YB = 0.28          # 범위 막대 위치(점 무리 위쪽)
for (b, g), (lo, hi, n) in RANGE.items():
    col = ps.BATCH_COLOR[b]
    x = lab.query("batch == @b and group == @g")["cycle_life"].to_numpy()
    yj = Y[b] + rng.uniform(-JIT, JIT, len(x))
    if g == "fastcharge":
        ax2.scatter(x, yj, s=32, color=col, edgecolor="white", linewidth=0.6, zorder=3)
    else:
        ax2.scatter(x, yj, s=32, color=tint(col, 0.40), edgecolor=col, linewidth=0.9, zorder=3)
    yb = Y[b] - YB
    ax2.plot([lo, hi], [yb, yb], color=col, lw=1.6, solid_capstyle="butt", zorder=2)
    for xx in (lo, hi):
        ax2.plot([xx, xx], [yb - 0.06, yb + 0.06], color=col, lw=1.6, zorder=2)

    rng_txt = f"{kfmt(lo)}~{kfmt(hi)}"
    if b == "batch1":
        ax2.text((lo + hi) / 2, yb - 0.08, f"학습 범위 {rng_txt}", ha="center", va="bottom",
                 fontsize=FS_NOTE, color=INK, fontweight="bold")
    elif b == "batch3":
        ax2.text((lo + hi) / 2, yb - 0.08, rng_txt, ha="center", va="bottom",
                 fontsize=FS_NOTE, color=INK, fontweight="bold")
    elif g == "fastcharge":     # Batch 2 fastcharge: 묶음 위, 왼쪽 끝 정렬
        ax2.text(lo, yb - 0.08, f"fastcharge {rng_txt}", ha="left", va="bottom",
                 fontsize=FS_NOTE, color=INK, fontweight="bold",
                 bbox=dict(boxstyle="square,pad=0.05", fc=(1, 1, 1, 0.75), ec="none"))
    else:                       # Batch 2 newstructure: 범위 막대 오른쪽
        ax2.text(hi + 25, yb, f"newstructure {rng_txt}", ha="left", va="center",
                 fontsize=FS_NOTE, color=INK, fontweight="bold")

ax2.set_xlim(XMIN, XMAX)
ax2.set_ylim(2.3, -0.62)
ax2.yaxis.set_major_locator(FixedLocator([Y[b] for b in BATCH_NAMES]))
ax2.set_yticklabels([YLAB[b] for b in BATCH_NAMES], fontsize=FS_TICK)
ax2.tick_params(axis="y", length=0, pad=6)
ax2.xaxis.set_major_locator(FixedLocator([500, 1000, 1500, 2000]))
ax2.xaxis.set_major_formatter(FuncFormatter(lambda v, _: kfmt(v)))
ax2.tick_params(axis="x", labelsize=FS_TICK, colors=MUTED, labelcolor=INK)
ax2.spines["left"].set_visible(False)
ax2.spines["bottom"].set_color(FAINT)
ax2.grid(axis="y", visible=False)
ax2.grid(axis="x", color="#E2E8F0", alpha=1.0, lw=0.8)
ax2.set_axisbelow(True)
ax2.set_xlabel("수명 (cycle_life, 사이클)", fontsize=13.5, color=INK, labelpad=5)
ax2.set_title("② 라벨 셀의 수명 범위", loc="left", fontsize=13.5, fontweight="bold", color=INK, pad=4)

# ── 막대 안 글자가 조각 폭 안에 들어가는지 점검 ──
fig.canvas.draw()
r = fig.canvas.get_renderer()
inv = ax1.transData.inverted()
PAD = 0.25   # 조각 양끝 여백(셀 단위)


def _span(t):
    bb = t.get_window_extent(r)
    return inv.transform([[bb.x0, 0], [bb.x1, 0]])[:, 0]


for t, (a, z) in FIT:
    for fs in (FS_NOTE, 11.5, 11.0):     # 넘칠 때만 글자를 한 단계씩 줄인다(최소 11 pt)
        t.set_fontsize(fs)
        x0d, x1d = _span(t)
        if x0d >= a + PAD and x1d <= z - PAD:
            break
    assert x0d >= a and x1d <= z, f"막대 글자 넘침: {t.get_text()!r} {x0d:.2f}~{x1d:.2f} > {a:.2f}~{z:.2f}"
    if fs != FS_NOTE:
        print(f"  막대 글자 축소: {t.get_text()!r} → {fs} pt")

out = ps.save(fig, "slide_input")
print(out)
