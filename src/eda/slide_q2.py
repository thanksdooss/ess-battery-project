"""DAY 1 슬라이드용 Q2 그림 — '세 배치 모두 첫 100 사이클엔 용량이 거의 그대로이고, 급락(knee)은 그 뒤에 온다.'

노션 DAY 1 「각 Batch Set에 대한 EDA를 수행하고 Batch간 특징을 비교함」 + Q2 「사이클 별 Qd 추이 시각화」.

패널 ①②③ : Batch 1(학습) / Batch 2(테스트) / Batch 3(추가 검증) 라벨 셀의 방전 용량 곡선
            (색 = cycle_life, 세 배치 같은 로그 눈금), 첫 100 사이클 띠, EOL 0.88 Ah 선, knee(Bacon–Watts) 점.
패널 ④     : 배치별 용량 기울기 중앙값 — 첫 100 사이클 / 수명 40~60% / 수명 80~100%
            (mAh/100 사이클, 음수 = 감소), 말기÷중기 셀별 비율 중앙값의 배치 범위.

수치 출처(그림에 찍히는 모든 숫자):
  results/eda/q2_results.json
    per_batch.{batch}.slope_2_100 / slope_mid_40_60 / slope_late_80_100 .median  (막대 값)
    per_batch.{batch}.late_over_mid.median, pooled.late_over_mid_range_of_batch_medians  (가속 배율)
    per_batch.{batch}.knee_ratio_bw.median, per_batch.{batch}.knee_bw.min, pooled.n_knee_le_100  (knee 주석)
    per_batch.{batch}.SOH_at100_vs_init.median                                    (첫 100 사이클 SOH ≈ 100%)
    cells[*].knee_bw, cells[*].QD_at_knee                                         (knee 점 위치)
  곡선은 q2_degradation.qd_series(평활 규칙 동일)로 다시 그리고, knee 점이 곡선 위에 있는지 assert 로 확인한다.

실행: python src/eda/slide_q2.py  →  reports/figures/slide_q2.png
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "src" / "eda"))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.cm import ScalarMappable  # noqa: E402

import plot_style as ps  # noqa: E402
from data import BATCH_NAMES, EOL_AH, load_batch  # noqa: E402
from q2_degradation import LIFE_CMAP, LIFE_NORM, qd_series  # noqa: E402  (같은 평활 규칙·같은 색 램프)

RES = json.loads((ROOT / "results" / "eda" / "q2_results.json").read_text())
PB = RES["per_batch"]
KNEE = {c["cell_key"]: c for c in RES["cells"]}

INK, MUTED = "#0B1220", "#475569"
BAND = "#DCEBFB"                 # 첫 100 사이클(모델 입력 창)
EOL_C = ps.SHORT
TITLE = {"batch1": "Batch 1(학습)", "batch2": "Batch 2(테스트)", "batch3": "Batch 3(추가 검증)"}
XMAX = {"batch1": 1150, "batch2": 1250, "batch3": 2000}

RC = {"font.size": 12.5, "axes.titlesize": 13.5, "axes.labelsize": 12.5, "xtick.labelsize": 11.5,
      "ytick.labelsize": 11.5, "axes.titleweight": "bold", "axes.titlepad": 6,
      "axes.labelcolor": INK, "xtick.color": MUTED, "ytick.color": MUTED, "axes.edgecolor": "#94A3B8"}


def check() -> dict:
    """그림에 찍는 숫자를 원천 JSON 과 대조."""
    n = RES["n_labeled"]
    assert n == {"batch1": 36, "batch2": 39, "batch3": 44}, n
    assert RES["pooled"]["n_knee_le_100"] == 0                       # 세 배치 119셀 모두 knee > 100 사이클
    for b in BATCH_NAMES:
        assert abs(PB[b]["SOH_at100_vs_init"]["median"] - 1.0) < 0.005, b   # cycle 100 SOH ≈ 100%
        assert PB[b]["knee_bw"]["min"] > 100, b
    lo, hi = RES["pooled"]["late_over_mid_range_of_batch_medians"]
    assert (round(lo, 1), round(hi, 1)) == (8.5, 9.6), (lo, hi)
    return {"knee_ratio": {b: round(PB[b]["knee_ratio_bw"]["median"], 2) for b in BATCH_NAMES},
            "knee_min": {b: PB[b]["knee_bw"]["min"] for b in BATCH_NAMES}, "late_over_mid": (lo, hi)}


def curves(ax, b: str) -> None:
    cells = [c for c in load_batch(b) if c["cell_key"] in KNEE]
    assert len(cells) == RES["n_labeled"][b]
    lives = np.array([KNEE[c["cell_key"]]["cycle_life"] for c in cells])
    ax.axvspan(0, 100, color=BAND, lw=0, zorder=0)
    kx, ky = [], []
    for i in np.argsort(-lives):                                    # 긴 수명(진한색)을 먼저, 짧은 수명을 위에
        c, k = cells[i], KNEE[cells[i]["cell_key"]]
        x, _, qs = qd_series(c, upto=k["cycle_life"])
        ax.plot(x, qs, color=LIFE_CMAP(LIFE_NORM(k["cycle_life"])), lw=1.05, alpha=0.95, zorder=2)
        if np.isfinite(k.get("knee_bw") or np.nan):
            assert abs(np.interp(k["knee_bw"], x, qs) - k["QD_at_knee"]) < 2e-6, c["cell_key"]
            kx.append(k["knee_bw"])
            ky.append(k["QD_at_knee"])
    ax.scatter(kx, ky, s=16, color=INK, edgecolor="white", linewidth=0.6, zorder=5)
    ax.axhline(EOL_AH, color=EOL_C, lw=1.2, ls=(0, (5, 3)), zorder=3)
    ax.set_xlim(0, XMAX[b])
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.set_ylim(0.862, 1.142)
    ax.set_yticks([0.90, 1.00, 1.10])
    ax.grid(True, axis="y", alpha=0.25)
    ax.grid(False, axis="x")
    ax.set_title(f"{TITLE[b]} {len(cells)}셀", loc="left", color=ps.BATCH_COLOR[b])
    kr = PB[b]["knee_ratio_bw"]["median"]
    ax.text(0.98, 0.95, f"knee = 수명의 {kr:.2f}\n(최소 {PB[b]['knee_bw']['min']:.0f} 사이클)",
            transform=ax.transAxes, ha="right", va="top", fontsize=11.5, color=INK, linespacing=1.25)


def main() -> Path:
    info = check()
    with plt.rc_context(RC):
        fig, axs = plt.subplots(2, 2, figsize=(9.2, 5.4), gridspec_kw={"hspace": 0.55, "wspace": 0.3})
        for ax, b in zip([axs[0, 0], axs[0, 1], axs[1, 0]], BATCH_NAMES):
            curves(ax, b)
        axs[0, 0].set_ylabel("방전 용량 (Ah)")
        axs[1, 0].set_ylabel("방전 용량 (Ah)")
        axs[1, 0].set_xlabel("사이클")
        axs[0, 1].set_xlabel("사이클")
        axs[0, 0].text(108, EOL_AH + 0.006, f"EOL {EOL_AH:.2f} Ah", color=EOL_C, fontsize=11, fontweight="bold",
                       va="bottom", ha="left")

        # 수명 색 막대(① 패널 왼쪽 아래 — 곡선이 없는 빈 곳)
        cax = axs[0, 0].inset_axes([0.035, 0.3, 0.25, 0.045])
        cb = fig.colorbar(ScalarMappable(norm=LIFE_NORM, cmap=LIFE_CMAP), cax=cax, orientation="horizontal")
        cb.set_ticks([400, 2000])
        cb.set_ticklabels(["400", "2,000"])
        cb.minorticks_off()
        cb.outline.set_visible(False)
        cax.tick_params(labelsize=10.5, length=0, pad=2, colors=MUTED)
        cax.set_title("색 = 수명", fontsize=10.5, color=MUTED, fontweight="normal", pad=3, loc="left")

        # ④ 배치별 구간 기울기
        axb = axs[1, 1]
        segs = [("slope_2_100", "첫 100\n사이클"), ("slope_mid_40_60", "수명\n40~60%"), ("slope_late_80_100", "수명\n80~100%")]
        w = 0.26
        for k, b in enumerate(BATCH_NAMES):
            vals = [PB[b][s]["median"] for s, _ in segs]
            xs = np.arange(3) + (k - 1) * w
            axb.bar(xs, vals, width=w * 0.92, color=ps.BATCH_COLOR[b], alpha=0.9, zorder=3,
                    label=TITLE[b].replace("(", " (").replace(")", ")"))
            axb.text(xs[2], vals[2] - 4, f"{vals[2]:.0f}".replace("-", "−"), ha="center", va="top", fontsize=10.5,
                     color=INK)
        axb.axhline(0, color=MUTED, lw=1.0, zorder=4)
        axb.set_xticks(np.arange(3))
        axb.set_xticklabels([t for _, t in segs], color=INK)
        axb.tick_params(axis="x", length=0, pad=4)
        axb.set_ylim(-185, 12)
        axb.set_yticks([0, -50, -100, -150])
        axb.set_yticklabels(["0", "−50", "−100", "−150"])
        axb.set_ylabel("mAh / 100 사이클")
        axb.set_title("용량 기울기 (중앙값, 음수 = 감소)", loc="left", color=INK)
        axb.grid(True, axis="y", alpha=0.25)
        axb.grid(False, axis="x")
        lo, hi = info["late_over_mid"]
        axb.text(0.03, 0.30, f"말기 ÷ 중기\n{lo:.1f}~{hi:.1f}배 가속\n(셀별 중앙값)", transform=axb.transAxes, ha="left",
                 va="center", fontsize=12, fontweight="bold", color=INK, linespacing=1.2)
        axb.legend(loc="center", bbox_to_anchor=(0.5, 0.6), fontsize=10, frameon=False, handlelength=1.0,
                   borderaxespad=0.2)
        fig.subplots_adjust(left=0.075, right=0.985, top=0.95, bottom=0.1)
        out = ps.save(fig, "slide_q2")
    print("saved", out)
    print("checked", info)
    for b in BATCH_NAMES:
        print(b, {s: round(PB[b][s]["median"], 2) for s in ("slope_2_100", "slope_mid_40_60", "slope_late_80_100",
                                                             "late_over_mid")})
    return out


if __name__ == "__main__":
    main()
