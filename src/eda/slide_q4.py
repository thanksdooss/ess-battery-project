"""DAY 1 슬라이드용 Q4 그림 — 충전 조건과 수명.

메시지: Batch 1에서 고전류 구간이 길수록 수명이 짧아지지만(상관, 단수명 <500은 0개),
그 정보는 주 피처 ΔQ 깊이(dQ_logvar)가 이미 담고 있다(순위상관 0.90, ΔQ 통제 후 편상관 −0.24)
→ 충전 조건은 피처에서 뺀다. 판단 근거는 Batch 1만 쓴다(Batch 2·3은 쓰지 않음).

  (왼쪽) 세 배치 비교: 0→80% 평균 C-rate(정책 설계값, data.parse_policy) vs 수명(라벨 셀, 로그).
         Batch 1 은 정책마다 4.0~5.4C, Batch 2·3 은 모두 ≈4.8C(10분 충전)라 별도 열로 둔다.
         같은 이름의 정책 4.8C(80%)-4.8C 평균 수명(B1 753 / B2 고속 484 / B2 newstructure 872)을 막대로 표시.
         → 충전 속도가 같아도 배치마다 수명이 달라, 충전 조건은 테스트 배치에서 사실상 상수다.
  (오른쪽) Batch 1 라벨 36셀: ΔQ 깊이(dQ_logvar) vs 고전류 비중 지표 I_qw80(실측 충전량 가중 평균 전류, C),
         점 색 = 수명. 두 변수 순위상관 0.90, 수명과의 상관 −0.79 → ΔQ 통제 후 −0.24.

숫자 출처(스크립트가 assert로 대조): results/eda/q4_cell_features.csv,
results/eda/q4_results.json (a_policy_life.table, b_crate_vs_life.c1_8_policies,
b_crate_vs_life.corr_batch1_cell.I_qw80, d_strategy_link.iqw_vs_dQ_logvar_spearman / partial_iqw_given_logvar).

실행: cd ess-battery-project && python src/eda/slide_q4.py
산출: reports/figures/slide_q4.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/
from data import parse_policy  # noqa: E402
import plot_style as ps  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, Normalize  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]   # 프로젝트 루트
RES = ROOT / "results" / "eda"

# ── 데이터 ─────────────────────────────────────────────────────
cells = pd.read_csv(RES / "q4_cell_features.csv")
q4 = json.loads((RES / "q4_results.json").read_text())

# 평균 C-rate는 data.parse_policy(정책 문자열 → 설계값)로 다시 계산해 CSV 값과 대조
cells["avgC"] = [parse_policy(p)["avgC_80"] for p in cells["policy"]]
assert np.allclose(cells["avgC"], cells["avgC_80"], atol=1e-3)

b1 = cells[cells["batch"] == "batch1"]
b1_lab = b1[b1["labeled"]]
pol = (b1_lab.groupby("policy")
       .agg(life=("cycle_life", "mean"), n=("cycle_life", "size"), avgC=("avgC", "first"))
       .sort_values("life", ascending=False).reset_index())

# 출처 대조: 정책별 평균 수명(a_policy_life.table)
src_tab = {r["policy"]: r for r in q4["a_policy_life"]["table"] if r["batch"] == "batch1" and r["n_labeled"]}
assert len(pol) == len(src_tab) == 20
for r in pol.itertuples():
    assert abs(r.life - src_tab[r.policy]["mean"]) < 0.05 and r.n == src_tab[r.policy]["n_labeled"]
assert round(pol["life"].max(), 1) == 1074.0 and round(pol["life"].min(), 1) == 546.5
assert (b1_lab["cycle_life"] < 500).sum() == 0          # 노션 단수명(<500) 셀 0개
c8 = q4["b_crate_vs_life"]["c1_8_policies"]
for name in ("8C(15%)-3.6C", "8C(25%)-3.6C", "8C(35%)-3.6C"):
    assert abs(pol.set_index("policy").loc[name, "life"] - c8[name]["mean"]) < 0.05

# 출처 대조: 오른쪽 패널(B1 라벨 36셀) — I_qw80 ↔ dQ_logvar 순위상관, I_qw80 ↔ 수명, ΔQ 통제 편상관
from scipy import stats  # noqa: E402
assert len(b1_lab) == 36
dsl = q4["d_strategy_link"]
rho_iq_dq = stats.spearmanr(b1_lab["I_qw80"], b1_lab["dQ_logvar"])[0]
assert abs(rho_iq_dq - dsl["iqw_vs_dQ_logvar_spearman"]["rho"]) < 5e-4, rho_iq_dq
# CSV 의 I_qw80 은 소수 4자리로 반올림돼 순위 동률이 생기므로(−0.781) JSON 값(−0.785, 원값 계산)을 표시에 쓴다
rho_iq_life_json = q4["b_crate_vs_life"]["corr_batch1_cell"]["I_qw80"]["spearman"]["rho"]
assert abs(stats.spearmanr(b1_lab["I_qw80"], b1_lab["cycle_life"])[0] - rho_iq_life_json) < 5e-3
rho_iq_life = rho_iq_life_json
# 편상관: log10 수명과 I_qw80 을 각각 dQ_logvar 에 OLS 회귀한 잔차끼리 Pearson
X = np.c_[np.ones(36), b1_lab["dQ_logvar"]]
res = lambda y: y - X @ np.linalg.lstsq(X, y, rcond=None)[0]
pr = np.corrcoef(res(np.log10(b1_lab["cycle_life"].to_numpy())), res(b1_lab["I_qw80"].to_numpy()))[0, 1]
assert abs(pr - dsl["partial_iqw_given_logvar"]["r"]) < 5e-4, pr
RHO_IQ_DQ, RHO_IQ_LIFE, PR_IQ = rho_iq_dq, rho_iq_life, pr

# ── 스타일 ─────────────────────────────────────────────────────
plt.rcParams.update({
    "font.size": 13, "axes.labelsize": 13.5, "axes.titlesize": 14, "xtick.labelsize": 12,
    "ytick.labelsize": 12, "axes.titleweight": "bold", "axes.titlepad": 9,
})
INK, MUTED = "#0B1220", "#64748B"
# 평균 C-rate 순차 색(Batch 1 파랑 한 색상, 느림 = 연함 → 빠름 = 진함). 오른쪽 B1 점과 같은 척도.
CMAP = LinearSegmentedColormap.from_list("b1seq", ["#BFDBFE", "#60A5FA", "#2563EB", "#1E3A8A"])
NORM = Normalize(vmin=3.6, vmax=5.4)
HL = {"8C(15%)-3.6C": "8C(15%)", "8C(25%)-3.6C": "8C(25%)", "8C(35%)-3.6C": "8C(35%)"}

fig = plt.figure(figsize=(8.6, 5.2))
gs = fig.add_gridspec(1, 2, width_ratios=[1.7, 1.08], wspace=0.25)
ax = fig.add_subplot(gs[0])
axr = fig.add_subplot(gs[1])

# ── 왼쪽: 세 배치 비교 — 충전 속도(0→80% 평균 C-rate, 정책 설계값) vs 수명 ─────────
# Batch 1 은 실제 C-rate 위치(3.6~5.4)에, Batch 2·3 은 모두 ≈4.8C(10분 충전)라 오른쪽 열로 따로 놓는다.
lab = cells[cells["labeled"]].copy()
B1L, B2F = lab[lab.batch == "batch1"], lab[(lab.batch == "batch2") & (lab.group == "fastcharge")]
B2N, B3L = lab[(lab.batch == "batch2") & (lab.group == "newstructure")], lab[lab.batch == "batch3"]
assert (len(B1L), len(B2F), len(B2N), len(B3L)) == (36, 30, 9, 44)
for d in (B2F, B2N, B3L):                                   # Batch 2·3 은 모두 10분 충전(≈4.8C)
    assert d["avgC"].between(4.78, 4.82).all()
assert abs(B1L["avgC"].min() - 4.0) < 1e-6 and abs(B1L["avgC"].max() - 5.4) < 1e-6   # 라벨 셀 기준(3.6C·4C 정책은 중도절단)
P48 = "4.8C(80%)-4.8C"                                       # 같은 이름의 정책 — 배치별 평균 수명
m48 = {"B1": B1L[B1L.policy == P48]["cycle_life"].mean(),
       "B2F": B2F[B2F.policy == P48]["cycle_life"].mean(),
       "B2N": B2N[B2N.policy == P48 + "-newstructure"]["cycle_life"].mean()}
assert (round(m48["B1"]), round(m48["B2F"]), round(m48["B2N"])) == (753, 484, 872), m48
assert round(B2F["cycle_life"].min()) == 392 and round(B3L["cycle_life"].max()) == 1935

rng = np.random.default_rng(42)
COLX = {"B2F": 6.05, "B2N": 6.55, "B3": 7.05}
ax.axvspan(5.72, 7.38, color="#F1F5F9", zorder=0)
ax.scatter(B1L["avgC"] + rng.uniform(-0.035, 0.035, len(B1L)), B1L["cycle_life"], s=40,
           color=ps.BATCH_COLOR["batch1"], edgecolor="white", linewidth=0.6, zorder=3)
for key, d, face in (("B2F", B2F, ps.BATCH_COLOR["batch2"]), ("B2N", B2N, "#FDBA74"),
                     ("B3", B3L, ps.BATCH_COLOR["batch3"])):
    ax.scatter(COLX[key] + rng.uniform(-0.13, 0.13, len(d)), d["cycle_life"], s=34, color=face,
               edgecolor="white", linewidth=0.6, zorder=3)
# 같은 4.8C(80%)-4.8C 정책: 배치별 평균 수명(가로 막대)
for key, xc, w in (("B1", 4.8, 0.16), ("B2F", COLX["B2F"], 0.2), ("B2N", COLX["B2N"], 0.2)):
    ax.plot([xc - w, xc + w], [m48[key]] * 2, color=INK, lw=2.6, zorder=4, solid_capstyle="butt")
for key, xc, w in (("B1", 4.8, 0.16), ("B2F", COLX["B2F"], 0.2), ("B2N", COLX["B2N"], 0.2)):
    left = key == "B2N"                                   # 872 는 B3 점과 겹치지 않게 막대 왼쪽에
    ax.text(xc - w - 0.03 if left else xc + w + 0.03, m48[key], f"{m48[key]:,.0f}", ha="right" if left else "left",
            va="center", fontsize=12, fontweight="bold", color=INK, zorder=5)
ax.text(0.02, 0.025, f"검은 막대 = 같은 정책 {P48}의 평균 수명", transform=ax.transAxes, ha="left", va="bottom",
        fontsize=11, color=INK)
ax.axhline(500, color="#DC2626", lw=1.3, ls=(0, (4, 3)), zorder=1)
ax.text(3.9, 470, "단수명 <500", ha="left", va="top", fontsize=11.5, color="#DC2626")

ax.set_yscale("log")
ax.set_ylim(320, 3100)
ax.set_yticks([400, 500, 700, 1000, 1500, 2000])
ax.set_yticklabels(["400", "500", "700", "1,000", "1,500", "2,000"])
ax.minorticks_off()
ax.set_xlim(3.82, 7.38)
ax.set_xticks([4.0, 4.4, 4.8, 5.2, COLX["B2F"], COLX["B2N"], COLX["B3"]])
ax.set_xticklabels(["4.0", "4.4", "4.8", "5.2", "B2\n고속", "B2\nnew", "B3"], fontsize=11.5)
for lbl, c in zip(ax.get_xticklabels()[4:], (ps.BATCH_COLOR["batch2"], "#EA8A3F", ps.BATCH_COLOR["batch3"])):
    lbl.set_color(c)
    lbl.set_fontweight("bold")
ax.text(4.65, 2900, "Batch 1(학습)\n정책마다 다름", ha="center", va="top", fontsize=11.5, color=ps.BATCH_COLOR["batch1"],
        fontweight="bold")
ax.text(6.55, 2900, "Batch 2·3\n모두 10분 충전(≈4.8C)", ha="center", va="top", fontsize=11.5, color=INK,
        fontweight="bold")
ax.set_xlabel("0→80% 평균 C-rate (정책 설계값)")
ax.set_ylabel("수명 (사이클, 로그)")
ax.grid(axis="x", visible=False)
ax.set_title("충전 속도가 같아도 배치마다 수명이 다르다", loc="left", color=INK)

# ── 오른쪽: Batch 1 — ΔQ 깊이 vs 고전류 비중(I_qw80), 점 색 = 수명 ─────────
from matplotlib import colors as mcolors  # noqa: E402
LIFE_CMAP = mcolors.LinearSegmentedColormap.from_list("life", ["#B91C1C", ps.SHORT, "#BDBDBD", ps.LONG, "#1E3A8A"])
LIFE_NORM = mcolors.LogNorm(vmin=b1_lab["cycle_life"].min(), vmax=b1_lab["cycle_life"].max())
axr.scatter(b1_lab["dQ_logvar"], b1_lab["I_qw80"], s=46, c=[LIFE_CMAP(LIFE_NORM(v)) for v in b1_lab["cycle_life"]],
            edgecolor="white", linewidth=0.7, zorder=3)
axr.set_xlim(-4.3, -3.25)
axr.set_ylim(4.08, 5.75)
axr.set_xticks([-4.0, -3.5])
axr.set_xticklabels(["−4.0", "−3.5"])
axr.set_yticks([4.5, 5.0, 5.5])
axr.set_xlabel("ΔQ 깊이   깊을수록 →")
axr.set_ylabel("고전류 비중 지표 (가중 평균 전류, C)")
axr.grid(axis="x", visible=False)
axr.set_title("고전류 비중 ≈ ΔQ 깊이", loc="left", color=INK)
axr.text(0.04, 0.965, f"순위상관 {RHO_IQ_DQ:.2f}", transform=axr.transAxes, ha="left", va="top",
         fontsize=13.5, fontweight="bold", color=INK)
axr.text(0.04, 0.875, "점 색: 빨강 = 수명 짧음\n          파랑 = 수명 긺", transform=axr.transAxes, ha="left",
         va="top", fontsize=11, color=MUTED, linespacing=1.2)
axr.text(0.97, 0.035, "수명과의 상관\n"
         f"{RHO_IQ_LIFE:.2f} → ΔQ 통제 후 {PR_IQ:.2f}".replace("-", "−"), transform=axr.transAxes,
         ha="right", va="bottom", fontsize=11.5, color=INK, linespacing=1.25)

out = ps.save(fig, "slide_q4")
print("saved", out)
print(pol.to_string())
