"""DAY 1 슬라이드용 그림 — Q5 상관관계 (메시지: 세 배치에서 유지되는 신호는 ΔQ 크기 하나다).

- 행 = Batch 1(학습) |Spearman ρ| 상위 8개 초기(cycle 2–100) 피처, 중복 묶음(B1 군집, 평균 |ρ| ≥ 0.8)끼리 모음.
  괄호는 피처 2개 이상인 묶음에만(ΔQ 왜도는 어느 묶음에도 없어 '단독'). 전체 6묶음은 slide_q5b.py(9쪽 히트맵).
- 막대 = Batch 1 ρ, ● = Batch 2(테스트), ◆ = Batch 3(추가 검증). 배치 안에서 사실상 상수인 값은 그리지 않는다.
- 오른쪽 = 묶음별로 Batch 2·3에서 부호·강도가 유지되는지(q5_results.json 의 consistency_spearman 판정).

수치 출처(그리기 전에 대조):
  results/eda/q5_results.json  correlations.{batch}.{feature}.spearman / constant,
                               batch1_rank_by_abs_spearman, consistency_spearman,
                               batch1_redundant_groups_avg_abs_rho_ge_0.8, batch1_group_overlap_spearman_vs_pearson
  results/eda/early_features.csv  labeled 셀로 Spearman 을 다시 계산해 JSON 과 일치하는지 확인
  results/eda/q5_final.json    takeaway 문장 속 'ρ −0.83' 확인
실행: cd ess-battery-project && python src/eda/slide_q5.py   →  reports/figures/slide_q5.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch, Rectangle  # noqa: E402
from matplotlib.transforms import blended_transform_factory  # noqa: E402

from data import BATCH_NAMES, ROOT  # noqa: E402

RES = ROOT / "results" / "eda"
TOP_N = 8
BASE = 13.0

# 슬라이드용 한글 피처 이름 (내부 변수명은 쓰지 않는다)
NAME = {
    "dQ_logvar": "ΔQ 깊이 (분산·로그) ★",
    "dQ_logabsmin": "ΔQ 최저점 (로그)",
    "fade_slope_2_100": "용량 기울기 (2–100 사이클)",
    "QD_max_minus_2": "용량 상승폭 (최대 − 초기)",
    "avgC_80": "충전 C-rate (0→80%)",
    "t80_min": "0→80% 충전시간 (설정값)",
    "chargetime_avg5": "충전시간 실측 (초기 5회)",
    "dQ_skew": "ΔQ 왜도 (모양)",
}
# 묶음 이름 — B1 군집 그룹의 대표 개념
GROUP_NAME = {"dQ_logvar": "ΔQ 깊이 계열", "fade_slope_2_100": "용량 변화", "avgC_80": "충전 조건",
              "dQ_skew": "ΔQ 모양"}
VERDICT_WORD = {"유지": "유지", "부호 반전": "반전", "약화/소멸": "약화", "상수(해석 불가)": "상수"}


def load() -> tuple[dict, list[list[str]], pd.DataFrame]:
    r = json.loads((RES / "q5_results.json").read_text())
    top = r["batch1_rank_by_abs_spearman"][:TOP_N]
    # 같은 정보 묶음: B1 군집(평균 |ρ| ≥ 0.8) 중 상위 8개에 든 것. 묶음 순서 = 묶음 내 최고 |ρ| 순
    groups = []
    for f in top:
        if any(f in g for g in groups):
            continue
        g = next((gg for gg in r["batch1_redundant_groups_avg_abs_rho_ge_0.8"] if f in gg), [f])
        groups.append([x for x in top if x in g])           # 묶음 안은 B1 |ρ| 순
    rows = []
    for g in groups:
        for f in g:
            c = {b: r["correlations"][b][f] for b in BATCH_NAMES}
            rows.append({"feature": f, "group": g[0],
                         **{f"rho_{b}": c[b]["spearman"] for b in BATCH_NAMES},
                         **{f"const_{b}": c[b]["constant"] for b in BATCH_NAMES},
                         "consistency": r["consistency_spearman"][f]})
    return r, groups, pd.DataFrame(rows)


def check(r: dict, groups: list[list[str]], T: pd.DataFrame) -> dict:
    """그림에 쓰는 모든 숫자를 원천 파일과 대조."""
    ef = pd.read_csv(RES / "early_features.csv")
    lab = ef[ef["labeled"]]
    n = lab.groupby("batch").size().to_dict()
    assert n == r["settings"]["n"] == {"batch1": 36, "batch2": 39, "batch3": 44}, n
    for _, row in T.iterrows():
        for b in BATCH_NAMES:
            d = lab[lab["batch"] == b]
            rho = stats.spearmanr(d[row["feature"]], d["cycle_life"], nan_policy="omit")[0]
            assert abs(rho - row[f"rho_{b}"]) < 1e-9, (row["feature"], b, rho, row[f"rho_{b}"])
    # ΔQ 크기 수치 (q5_final.json takeaway 'ρ −0.83', 보고서 '−0.83 / −0.71 / −0.80')
    dq = T.set_index("feature").loc["dQ_logvar"]
    shown = tuple(round(dq[f"rho_{b}"], 2) for b in BATCH_NAMES)
    assert shown == (-0.83, -0.71, -0.80), shown
    fin = json.loads((RES / "q5_final.json").read_text())
    assert "ρ −0.83" in fin["takeaway"]
    # '세 배치 모두 유지' 판정은 ΔQ 크기 두 피처뿐
    kept = [f for f in T["feature"] if r["consistency_spearman"][f] == "일관"]
    assert kept == ["dQ_logvar", "dQ_logabsmin"], kept
    # 묶음 내부 상관(B1): 그림에 보이는 묶음은 모두 쌍별 |ρ| ≥ 0.9 (묶음 정의 = 평균 |ρ| ≥ 0.8, 9쪽 히트맵과 같음)
    M = r["batch1_feature_spearman_matrix"]
    within = {g[0]: min(abs(M[a][b]) for i, a in enumerate(g) for b in g[i + 1:]) for g in groups if len(g) > 1}
    assert min(within.values()) >= 0.9, within
    return {"n": n, "dq_rho": shown, "within_group_min_pair_abs_rho": within}


def group_verdict(T: pd.DataFrame, g0: str) -> str:
    """묶음 안 피처들의 Batch 2·3 판정을 합친 짧은 문구."""
    sub = T[T["group"] == g0]
    if (sub["consistency"] == "일관").all():
        return "유지 ✓"
    words = []
    for s in sub["consistency"]:
        for part in s.split(" / "):
            w = VERDICT_WORD[part.split(": ", 1)[1]]
            if w not in words:
                words.append(w)
    order = ["상수", "반전", "약화"]
    return "·".join(w for w in order if w in words) + " ✗"


def draw(r: dict, groups: list[list[str]], T: pd.DataFrame) -> Path:
    plt.rcParams.update({"font.size": BASE, "axes.labelsize": BASE, "xtick.labelsize": 12,
                         "ytick.labelsize": 12.5})
    INK, MUTED, FAINT = "#0F172A", "#475569", "#94A3B8"
    HL = "#FEF3C7"                                     # ΔQ 크기 강조 배경

    # 행 위치: 묶음 사이에 여백
    gap, y, ypos, gi = 0.55, 0.0, [], {}
    for k, g in enumerate(groups):
        if k:
            y += gap
        for f in g:
            ypos.append(-y)
            y += 1.0
    T = T.assign(y=ypos)

    fig = plt.figure(figsize=(8.6, 5.55))
    L, R, B, H = 0.300, 0.745, 0.150, 0.735
    ax = fig.add_axes([L, B, R - L, H])
    ax.set_xlim(-1, 1)
    ax.set_ylim(T["y"].min() - 0.62, T["y"].max() + 0.62)
    ax.axvline(0, color="#64748B", lw=1.0, zorder=1)
    ax.grid(axis="y", visible=False)
    ax.grid(axis="x", alpha=0.35)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=8)

    # 강조 띠: ΔQ 크기 묶음 (라벨~판정 칸까지)
    tb = blended_transform_factory(fig.transFigure, ax.transData)
    dq_rows = T[T["group"] == groups[0][0]]
    y_top, y_bot = dq_rows["y"].max() + 0.5, dq_rows["y"].min() - 0.5
    fig.patches.append(Rectangle((0.005, y_bot), 0.99, y_top - y_bot, transform=tb, facecolor=HL,
                                 edgecolor="none", zorder=-10))
    ax.set_facecolor("none")

    # 막대(B1) + 점(B2, B3)
    ax.barh(T["y"], T["rho_batch1"], height=0.56, color=ps.BATCH_COLOR["batch1"], alpha=0.9, zorder=2)
    for b, mk, ms, dy in (("batch2", "o", 9.5, 0.0), ("batch3", "D", 8.5, 0.0)):
        m = ~T[f"const_{b}"]
        ax.plot(T.loc[m, f"rho_{b}"], T.loc[m, "y"] + dy, marker=mk, ms=ms, ls="none",
                mfc=ps.BATCH_COLOR[b], mec="white", mew=1.4, zorder=4)

    # y 라벨
    ax.set_yticks(T["y"])
    ax.set_yticklabels([NAME[f] for f in T["feature"]])
    for t, f in zip(ax.get_yticklabels(), T["feature"]):
        is_dq = f in groups[0]
        t.set_color(INK if is_dq else "#334155")
        t.set_fontweight("bold" if is_dq else "normal")

    # x 축: 가운데 축 이름, 양 끝에 방향 힌트(같은 줄)
    ax.set_xticks([-1, -0.5, 0, 0.5, 1])
    ax.set_xticklabels(["−1", "−0.5", "0", "+0.5", "+1"])
    ax.tick_params(axis="x", pad=4)
    yl = -0.118                                        # 축 이름과 방향 힌트를 같은 기준선에
    ax.text(0.5, yl, "수명과의 순위상관 ρ", transform=ax.transAxes, ha="center", va="top", fontsize=BASE,
            color=INK)
    ax.text(0.0, yl, "← 값↑ 수명↓", transform=ax.transAxes, ha="left", va="top", fontsize=11.5, color=MUTED)
    ax.text(1.0, yl, "값↑ 수명↑ →", transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=MUTED)

    # 상수 행: 점이 없는 이유를 행 안에 직접 적는다
    cst = T[T["const_batch2"] & T["const_batch3"]]
    if len(cst):
        yc = cst["y"].max()                            # C-rate 행 오른쪽 빈 칸
        ax.text(0.05, yc, "Batch 2·3은 전 셀 동일\n→ 비교 불가 (점 없음)", ha="left", va="center",
                fontsize=11.5, color=MUTED, linespacing=1.3, zorder=5)

    # 핵심 수치: ΔQ 분산 행 오른쪽 빈 칸
    d0 = T.iloc[0]
    vals = " · ".join(f"{d0[f'rho_{b}']:+.2f}".replace("-", "−") for b in BATCH_NAMES)
    ax.text(0.06, d0["y"], vals, ha="left", va="center", fontsize=12, color=INK, fontweight="bold", zorder=5)
    ax.text(0.06, d0["y"] - 0.55, "Batch 1 · 2 · 3", ha="left", va="center", fontsize=11.5, color=MUTED, zorder=5)

    # 오른쪽 판정 칸 (묶음 단위 괄호)
    xb, xt = R + 0.018, R + 0.034
    ax.text(xt, T["y"].max() + 0.95, "Batch 2·3 확인용", transform=tb, ha="left", va="center",
            fontsize=12, color=MUTED)
    for g in groups:
        sub = T[T["group"] == g[0]]
        top, bot = sub["y"].max() + 0.32, sub["y"].min() - 0.32
        if len(g) > 1:                                 # 괄호 = 중복 묶음. 피처 1개(묶음 아님)는 괄호 없이 '단독'
            ax.plot([xb + 0.008, xb, xb, xb + 0.008], [top, top, bot, bot], transform=tb, color=FAINT, lw=1.6,
                    clip_on=False, solid_joinstyle="miter")
        mid = (top + bot) / 2
        verdict = group_verdict(T, g[0])
        is_dq = g is groups[0]
        ax.text(xt, mid + 0.25, GROUP_NAME[g[0]] + (f" ({len(g)}개)" if len(g) > 1 else " (단독)"), transform=tb,
                ha="left", va="center", fontsize=12, color=INK if is_dq else MUTED)
        ax.text(xt, mid - 0.25, verdict, transform=tb, ha="left", va="center", fontsize=12.5,
                color=INK, fontweight="bold")
    ax.text(xt, T["y"].min() - 1.0, "괄호 = 중복 묶음\n(전체 6묶음은 9쪽)", transform=tb,
            ha="left", va="top", fontsize=11.5, color=MUTED, linespacing=1.3)

    # 범례 (맨 위 한 줄)
    handles = [Patch(color=ps.BATCH_COLOR["batch1"], alpha=0.9),
               Line2D([], [], marker="o", ls="none", ms=9.5, mfc=ps.BATCH_COLOR["batch2"], mec="white", mew=1.2),
               Line2D([], [], marker="D", ls="none", ms=8.5, mfc=ps.BATCH_COLOR["batch3"], mec="white", mew=1.2)]
    fig.legend(handles, ["Batch 1(학습)", "Batch 2(테스트)", "Batch 3(추가 검증)"], loc="upper left",
               bbox_to_anchor=(L - 0.012, 0.995), ncol=3, fontsize=12.5, handlelength=1.3,
               handletextpad=0.5, columnspacing=1.4, borderaxespad=0)

    return ps.save(fig, "slide_q5")


def main() -> None:
    r, groups, T = load()
    info = check(r, groups, T)
    p = draw(r, groups, T)
    print("groups:", groups)
    print(T[["feature", "rho_batch1", "rho_batch2", "rho_batch3", "const_batch2", "const_batch3",
             "consistency"]].round(3).to_string(index=False))
    print("verdicts:", {g[0]: group_verdict(T, g[0]) for g in groups})
    print("checked:", info)
    print("saved:", p)


if __name__ == "__main__":
    main()
