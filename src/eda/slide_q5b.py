"""DAY 1 슬라이드용 그림 — Q5 「멀티클리니어리티 문제 확인」 (메시지: 비슷하게 움직이는 피처가 6묶음으로 겹친다).

- 행·열 = Batch 1(학습) 중복 묶음(평균 |ρ| ≥ 0.8)에 든 피처 14개, 묶음끼리 모아 정렬. 열 번호 = 행 번호.
- 색 = Batch 1 Spearman ρ (labeled 36셀). 숫자는 묶음 안 상관만(굵게). 굵은 테두리 = 묶음.
- 오른쪽 = 묶음 안 평균 |ρ|를 Batch 1·2·3에서 비교(판단은 Batch 1, Batch 2·3은 확인용).
  배치 안에서 사실상 상수인 피처가 든 묶음은 '상수'(상관 해석 불가)로 적는다.

수치 출처(그리기 전에 대조):
  results/eda/q5_results.json  batch1_redundant_groups_avg_abs_rho_ge_0.8, batch1_group_overlap_spearman_vs_pearson,
                               batch1_feature_spearman_matrix, correlations.{batch}.{feature}.constant,
                               vif_main_candidates, vif_m2_pool (슬라이드 본문 숫자)
  results/eda/q5d_results.json  qcc_robustness.vif_2feat (주 모델 M1 = ΔQ 깊이 + 초기 용량의 VIF)
  results/eda/early_features.csv  labeled 셀로 Spearman 을 다시 계산해 JSON 과 일치하는지 확인
  results/eda/q5d_cell_features.csv  초기 용량(Qcc_init)이 용량 수준 묶음과 같은 정보인지(ρ) 확인
실행: cd ess-battery-project && python src/eda/slide_q5b.py   →  reports/figures/slide_q5b.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import Normalize  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from data import BATCH_NAMES, ROOT  # noqa: E402

RES = ROOT / "results" / "eda"

# 묶음 순서 = 슬라이드 읽는 순서(핵심 신호 → 설정값 → 용량 → 온도 → 저항). 피처 이름은 slide_q5.py 와 같게.
GROUPS = [
    ("ΔQ 깊이 계열", ["dQ_logvar", "dQ_logabsmin"]),
    ("충전 조건", ["avgC_80", "t80_min", "chargetime_avg5"]),
    ("용량 수준", ["QD_2", "fade_int_2_100", "fade_int_91_100"]),
    ("용량 변화", ["fade_slope_2_100", "QD_max_minus_2"]),
    ("온도", ["Tavg_mean", "Tmax_max"]),
    ("내부저항", ["IR_min", "IR_2"]),
]
NAME = {
    "dQ_logvar": "ΔQ 깊이 (분산·로그)",
    "dQ_logabsmin": "ΔQ 최저점 (로그)",
    "avgC_80": "충전 C-rate (0→80%)",
    "t80_min": "0→80% 충전시간 (설정값)",
    "chargetime_avg5": "충전시간 실측 (초기 5회)",
    "QD_2": "방전용량 (2 사이클)",
    "fade_int_2_100": "용량 추세선 절편 (2–100)",
    "fade_int_91_100": "용량 추세선 절편 (91–100)",
    "fade_slope_2_100": "용량 기울기 (2–100 사이클)",
    "QD_max_minus_2": "용량 상승폭 (최대 − 초기)",
    "Tavg_mean": "평균 온도",
    "Tmax_max": "최고 온도",
    "IR_min": "내부저항 최저값",
    "IR_2": "내부저항 (2 사이클)",
}
SHORT_B = {"batch1": "Batch 1", "batch2": "2", "batch3": "3"}


def load() -> tuple[dict, list[str], pd.DataFrame, pd.DataFrame]:
    r = json.loads((RES / "q5_results.json").read_text())
    order = [f for _, g in GROUPS for f in g]
    M = pd.DataFrame(r["batch1_feature_spearman_matrix"]).loc[order, order]   # 대조용(반올림 저장값)
    ef = pd.read_csv(RES / "early_features.csv")
    return r, order, M, ef[ef["labeled"]]


def b1_matrix(lab: pd.DataFrame, order: list[str]) -> pd.DataFrame:
    """그림에 쓰는 Batch 1 Spearman 행렬(원자료에서 다시 계산한 정밀값)."""
    return lab[lab["batch"] == "batch1"][order].corr(method="spearman")


def within(lab: pd.DataFrame, r: dict) -> pd.DataFrame:
    """묶음 안 모든 쌍의 평균 |Spearman ρ| — 배치별. 상수 피처가 있으면 NaN(=상수)."""
    rows = []
    for gname, g in GROUPS:
        row = {"group": gname}
        for b in BATCH_NAMES:
            if any(r["correlations"][b][f]["constant"] for f in g):
                row[b] = np.nan
                continue
            C = lab[lab["batch"] == b][g].corr(method="spearman")
            row[b] = float(np.mean([abs(C.iloc[i, j]) for i in range(len(g)) for j in range(i + 1, len(g))]))
        rows.append(row)
    return pd.DataFrame(rows).set_index("group")


def check(r: dict, order: list[str], M: pd.DataFrame, lab: pd.DataFrame, W: pd.DataFrame) -> dict:
    """그림·슬라이드 본문에 쓰는 모든 숫자를 원천 파일과 대조."""
    n = lab.groupby("batch").size().to_dict()
    assert n == r["settings"]["n"] == {"batch1": 36, "batch2": 39, "batch3": 44}, n
    # 묶음 = JSON 의 B1 군집과 정확히 같은 6개(순서만 다름)
    src = {frozenset(g) for g in r["batch1_redundant_groups_avg_abs_rho_ge_0.8"]}
    assert src == {frozenset(g) for _, g in GROUPS}, src
    # 행렬 = early_features.csv 로 다시 계산한 Batch 1 Spearman (JSON 은 소수 4자리 반올림 저장)
    C = lab[lab["batch"] == "batch1"][order].corr(method="spearman")
    assert np.allclose(C.to_numpy(), M.to_numpy(), atol=5.1e-5), (C - M).abs().max().max()
    # Batch 1 묶음 평균 |ρ| = JSON overlap 값
    ov = {frozenset(o["group"]): o["mean_abs_spearman"] for o in r["batch1_group_overlap_spearman_vs_pearson"]}
    for gname, g in GROUPS:
        assert abs(ov[frozenset(g)] - W.loc[gname, "batch1"]) < 1e-9, gname
    b1 = W["batch1"]
    assert round(b1.min(), 2) == 0.82 and round(b1.max(), 2) == 0.99, b1
    # 본문: '19개 중 14개', '6개 묶음'
    assert len(r["batch1_feature_spearman_matrix"]) == 19 and len(order) == 14 and len(GROUPS) == 6
    # 본문: ΔQ 묶음은 세 배치 모두 0.98 이상, 충전 묶음은 Batch 2·3 상수
    assert (W.loc["ΔQ 깊이 계열"] >= 0.98).all(), W.loc["ΔQ 깊이 계열"]
    assert W.loc["충전 조건", ["batch2", "batch3"]].isna().all()
    # 본문 ②: 충전 C-rate↔충전시간, 2사이클 방전용량↔추세선 절편(2–100)은 사실상 같은 값(|ρ|, |r| ≥ 0.99),
    #         ΔQ 깊이↔최저점은 같은 곡선의 두 요약(세 배치 모두 ρ ≥ 0.98 — 위 W 확인)
    b1 = lab[lab["batch"] == "batch1"]
    for a, b in (("avgC_80", "t80_min"), ("chargetime_avg5", "t80_min"), ("QD_2", "fade_int_2_100")):
        assert abs(M.loc[a, b]) >= 0.99 and abs(b1[[a, b]].corr().iloc[0, 1]) >= 0.99, (a, b)
    # 본문 ③: 주요 후보 17개를 다 넣으면 VIF 최대 14,628 → 묶음당 최대 1개인 실제 모델은 모두 5 미만
    #         (주 모델 M1 = ΔQ 깊이 + 초기 용량 1.1, 비교 풀 M2 4.9)
    v0, v2 = pd.Series(r["vif_main_candidates"]), pd.Series(r["vif_m2_pool"])
    assert (len(v0), int((v0 >= 10).sum()), round(v0.max())) == (17, 13, 14628), v0
    v1 = json.loads((RES / "q5d_results.json").read_text())["qcc_robustness"]["vif_2feat"]
    assert round(v1, 1) == 1.1 and round(v2.max(), 1) == 4.9 and v2.max() < 5, (v1, v2.max())
    # 그림 주석: 후보 초기 용량(Qcc_init)은 용량 수준 묶음과 같은 정보(Batch 1 ρ ≥ 0.99) → 그 묶음의 대표
    q = pd.read_csv(RES / "q5d_cell_features.csv")
    q = q[q["labeled"] & (q["batch"] == "batch1")]
    rho_q = q[["Qcc_init", "QD_2"]].corr(method="spearman").iloc[0, 1]
    assert len(q) == 36 and round(rho_q, 2) >= 0.99, rho_q
    return {"n": n, "within": W.round(3).to_dict(), "vif_all_max": round(v0.max()), "vif_m1": round(v1, 3),
            "vif_m2_max": round(v2.max(), 3), "rho_qcc_qd2": round(rho_q, 3)}


def fmt(v: float) -> str:
    s = f"{abs(v):.2f}"
    s = "1" if s == "1.00" else s.lstrip("0")
    return ("−" if v < 0 else "") + s


def draw(order: list[str], M: pd.DataFrame, W: pd.DataFrame) -> Path:
    INK, MUTED, FAINT = "#0F172A", "#475569", "#94A3B8"
    n = len(order)
    FW, FH = 9.3, 5.45                                  # inch
    X0, Y0, S = 2.62, 0.78, 4.25                        # 행렬 왼쪽·아래·한 변(inch)
    fig = plt.figure(figsize=(FW, FH))
    ax = fig.add_axes([X0 / FW, Y0 / FH, S / FW, S / FH])
    cmap, norm = plt.get_cmap("RdBu_r"), Normalize(-1, 1)

    A = M.to_numpy()
    img = np.array(cmap(norm(A)))
    for i in range(n):
        img[i, i] = (0.90, 0.91, 0.93, 1.0)             # 대각선(자기 자신)은 회색
    ax.imshow(img, interpolation="nearest")
    ax.set_xlim(-0.5, n - 0.5)
    ax.set_ylim(n - 0.5, -0.5)
    ax.grid(False)
    for sp in ax.spines.values():
        sp.set_visible(False)

    gid = {f: k for k, (_, g) in enumerate(GROUPS) for f in g}
    for i in range(n):
        for j in range(n):
            v = A[i, j]
            if i == j or gid[order[i]] != gid[order[j]]:   # 숫자는 묶음 안 상관만
                continue
            s = fmt(v)
            ax.text(j, i, s, ha="center", va="center", fontsize=9.0 if len(s) >= 4 else 9.6,
                    fontweight="bold", color="white" if abs(v) >= 0.62 else INK)

    # 묶음 테두리
    start = 0
    for _, g in GROUPS:
        k = len(g)
        ax.add_patch(Rectangle((start - 0.5, start - 0.5), k, k, fill=False, ec=INK, lw=2.4, zorder=5,
                               clip_on=False))
        start += k

    # 행 이름(번호 + 이름) · 열 번호(위)
    ax.set_yticks(range(n))
    ax.set_yticklabels([f"{NAME[f]}  {i + 1:>2d}" for i, f in enumerate(order)], fontsize=12.2)
    ax.tick_params(axis="y", length=0, pad=5)
    ax.xaxis.tick_top()
    ax.set_xticks(range(n))
    ax.set_xticklabels([str(i + 1) for i in range(n)], fontsize=10.5, color=MUTED)
    ax.tick_params(axis="x", length=0, pad=3)
    for t, f in zip(ax.get_yticklabels(), order):
        dq = gid[f] == 0
        t.set_color(INK if dq else "#334155")
        t.set_fontweight("bold" if dq else "normal")

    # 오른쪽 표: 묶음 이름 + 묶음 안 평균 |ρ| (Batch 1 · 2 · 3)
    to_fig_y = lambda row: (Y0 + S * (n - 0.5 - row) / n) / FH   # 행 중심(데이터 y) → figure y
    xg = (X0 + S + 0.16) / FW
    xs = [(X0 + S + 1.62 + 0.52 * k) / FW for k in range(3)]
    ytop = (Y0 + S + 0.16) / FH
    fig.text(xg, ytop, "묶음", ha="left", va="bottom", fontsize=11.5, color=MUTED)
    for x, b in zip(xs, BATCH_NAMES):
        fig.text(x, ytop, SHORT_B[b], ha="center", va="bottom", fontsize=11.5, color=ps.BATCH_COLOR[b],
                 fontweight="bold")
    fig.text((xs[0] + xs[-1]) / 2, ytop + 0.27 / FH, "묶음 안 평균 상관 |ρ|", ha="center", va="bottom",
             fontsize=11.5, color=MUTED)
    start = 0
    for k, (gname, g) in enumerate(GROUPS):
        yc = to_fig_y(start + (len(g) - 1) / 2)
        fig.text(xg, yc, gname, ha="left", va="center", fontsize=12.2, color=INK,
                 fontweight="bold" if k == 0 else "normal")
        for x, b in zip(xs, BATCH_NAMES):
            v = W.loc[gname, b]
            if np.isnan(v):
                fig.text(x, yc, "상수", ha="center", va="center", fontsize=11.5, color=MUTED)
            else:
                fig.text(x, yc, f"{v:.2f}".lstrip("0"), ha="center", va="center", fontsize=12,
                         color=INK if b == "batch1" else MUTED, fontweight="bold" if b == "batch1" else "normal")
        if k:                                            # 묶음 경계선
            yb = (Y0 + S * (n - start) / n) / FH
            fig.add_artist(plt.Line2D([xg, xs[-1] + 0.26 / FW], [yb, yb], color="#E2E8F0", lw=1.0))
        start += len(g)
    fig.text(xg, (Y0 - 0.12) / FH, "충전 '상수' = Batch 2·3은 전 셀이\n같은 10분 충전이라 비교 불가",
             ha="left", va="top", fontsize=11.5, color=MUTED, linespacing=1.3)

    # 색 막대(행렬 아래)
    cax = fig.add_axes([(X0 + 0.35) / FW, 0.30 / FH, (S - 0.7) / FW, 0.13 / FH])
    cb = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
    cb.set_ticks([-1, 0, 1])
    cb.set_ticklabels(["−1 반대로 움직임", "0", "+1 같이 움직임"])
    cb.ax.tick_params(labelsize=11.5, length=0, pad=3, colors=MUTED)
    tl = cb.ax.get_xticklabels()                        # 양 끝 눈금 이름을 막대 안쪽으로(옆 주석과 붙지 않게)
    tl[0].set_ha("left")
    tl[-1].set_ha("right")
    cb.outline.set_visible(False)
    fig.text((X0 - 0.12) / FW, 0.36 / FH, "순위상관 ρ (Batch 1)", ha="right", va="center", fontsize=11.5,
             color=MUTED)
    return ps.save(fig, "slide_q5b")


def main() -> None:
    r, order, M, lab = load()
    W = within(lab, r)
    info = check(r, order, M, lab, W)
    p = draw(order, b1_matrix(lab, order), W)
    print(W.round(3).to_string())
    print("checked:", json.dumps(info, ensure_ascii=False, default=str))
    print("saved:", p)


if __name__ == "__main__":
    main()
