"""DAY 1 EDA — Question 5 (추가 점검 d). 조건부 기여: dQ_logvar 통제 편상관 & 복제쌍 분석 (Batch 1)

노션 Q5 하위 항목 (a)(b)(c) 다음에 붙이는 추가 점검이다(노션 bullet 아님).
  ① 후보 16개의 편상관: dQ_logvar 통제 / dQ_logvar + Qcc_init 통제 (Bonferroni 기준선)
  ② 정책 평균 단위 vs 정책 내부(복제쌍) 관계: dQ_logvar, Qcc_init
  ③ Qcc_init vs summary QD_2 배치별 수준 이동 (Batch 2/3 는 라벨 미사용) + CV 방전분
  ④ Batch 1 Hold-out 셀 고정 (사전 등록 절차 재현) → results/holdout_cells.csv

규칙
  - 피처 선택 근거는 Batch 1 labeled 36셀만 쓴다. Batch 2/3 은 라벨 없이 피처 분포(수준 이동)만 본다.
  - 모델 학습·튜닝 없음. 계수는 Batch 1 기술 회귀(in-sample)이며 예측 이동량 환산에만 쓴다.
  - Qcc_init = cycle 2–6 Qdlin 2.0 V 끝점(= CC 방전 종료 용량)의 중앙값. ≤0 또는 >1.2 Ah 값은 NaN.
    노션 경고 '충전 커브 시작 시점이 배치별로 상이 : Qdlin 변수를 단순 비교하면 왜곡 발생'에 해당하는
    Qdlin 절대 수준 피처이므로 배치 이동량을 함께 기록한다.

산출물
  results/eda/q5d_results.json, results/eda/q5d_cell_features.csv, results/holdout_cells.csv
  reports/figures/q5d_partial_corr.png, q5d_replicate_pairs.png, q5d_qcc_batch_offset.png
실행 순서: python src/eda/q5_correlation.py → python src/eda/q5d_conditional.py
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))   # src/

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from data import RANDOM_STATE, ROOT, clean_summary, load_all, qdlin  # noqa: E402

OUT = ROOT / "results" / "eda"
QCC_WIN = (2, 6)          # Qcc_init: cycle 2–6 중앙값 (원논문 QD2 의 CC 정의판, 단일 사이클 잡음 완화)
QCC_SENS = (8, 12)        # 민감도 정의
BREAKIN_SMOOTH = 9        # CC break-in: 9-사이클 이동중앙값
N_BOOT = 3000


# ════════════════════════════════════════════════════════════════════════
# CC 끝점 피처 (q5_correlation.py 도 이 함수를 가져다 쓴다)
# ════════════════════════════════════════════════════════════════════════
def qcc_series(c: dict, a: int = 2, b: int = 100) -> pd.Series:
    """cycle a..b 의 Qdlin 2.0 V 끝점(CC 방전 종료 용량, Ah). 이상값(≤0, >1.2 Ah)은 NaN."""
    v = np.array([qdlin(c, k)[-1] for k in range(a, b + 1)], float)
    v[(v <= 0) | (v > 1.2)] = np.nan
    return pd.Series(v, index=range(a, b + 1))


def qcc_init(c: dict, a: int = QCC_WIN[0], b: int = QCC_WIN[1]) -> float:
    return float(np.nanmedian(qcc_series(c, a, b)))


def cc_breakin_mAh(c: dict) -> float:
    """CC break-in 상승폭 = cycle 2–100 CC 끝점의 9-사이클 이동중앙값 최댓값 − Qcc_init (mAh)."""
    s = qcc_series(c).interpolate(limit_direction="both")
    sm = s.rolling(BREAKIN_SMOOTH, center=True, min_periods=1).median()
    return float((sm.max() - qcc_init(c)) * 1000)


def cv_discharge_mAh(c: dict, a: int = QCC_WIN[0], b: int = QCC_WIN[1]) -> float:
    """summary QD − CC 끝점 (= 2.0 V CV 유지 구간 방전분), cycle a..b 중앙값 (mAh)."""
    s = clean_summary(c["summary"]).set_index("cycle")
    qd = np.array([s.at[float(k), "QD"] if float(k) in s.index else np.nan for k in range(a, b + 1)], float)
    return float(np.nanmedian(qd - qcc_series(c, a, b).to_numpy()) * 1000)


# ════════════════════════════════════════════════════════════════════════
# 통계 도우미
# ════════════════════════════════════════════════════════════════════════
def _resid(v: np.ndarray, ctrl: list[np.ndarray]) -> np.ndarray:
    X = np.column_stack([np.ones(len(v))] + ctrl)
    beta = np.linalg.lstsq(X, v, rcond=None)[0]
    return v - X @ beta


def partial_r(v, y, ctrl: list[np.ndarray]) -> tuple[float, float, int]:
    """Pearson 편상관(통제변수 선형 잔차끼리) + t 검정 p (df = n−2−k)."""
    v = np.asarray(v, float)
    m = np.isfinite(v) & np.isfinite(y)
    for c in ctrl:
        m &= np.isfinite(c)
    cc = [c[m] for c in ctrl]
    r = float(np.corrcoef(_resid(v[m], cc), _resid(y[m], cc))[0, 1])
    n, k = int(m.sum()), len(ctrl)
    t = r * np.sqrt((n - 2 - k) / (1 - r ** 2))
    return r, float(2 * stats.t.sf(abs(t), n - 2 - k)), n


def r_crit(df: int, alpha: float) -> float:
    t = stats.t.ppf(1 - alpha / 2, df)
    return float(t / np.sqrt(df + t ** 2))


def between_share(v: pd.Series, g: pd.Series) -> float:
    """일원 분산분해: 정책 사이 제곱합 / 전체 제곱합."""
    gm = v.groupby(g).transform("mean")
    return float(((gm - v.mean()) ** 2).sum() / ((v - v.mean()) ** 2).sum())


# ════════════════════════════════════════════════════════════════════════
# Hold-out (사전 등록 절차, DAY 1 전략 문서와 동일)
# ════════════════════════════════════════════════════════════════════════
def holdout_split(L: pd.DataFrame) -> tuple[list[str], dict]:
    """정책을 평균 수명 순으로 정렬 → 4개씩 5개 층. 최저·최고 정책은 후보에서 제외.
    층마다 np.random.default_rng(42).integers 로 정책 1개를 뽑는다."""
    pm = L.groupby("policy")["cycle_life"].mean().sort_values(kind="mergesort")
    pol = list(pm.index)
    lo, hi = pol[0], pol[-1]
    rng = np.random.default_rng(RANDOM_STATE)
    ho, strata = [], []
    for s in range(5):
        cand = [p for p in pol[4 * s:4 * s + 4] if p not in (lo, hi)]
        pick = cand[int(rng.integers(len(cand)))]
        ho.append(pick)
        strata.append({"stratum": s + 1, "policies": pol[4 * s:4 * s + 4], "candidates": cand, "picked": pick})
    return ho, {"policy_mean_life_sorted": pm.round(1).to_dict(), "excluded_extremes": [lo, hi], "strata": strata}


# ════════════════════════════════════════════════════════════════════════
LBL = {
    "Qcc_init": "Qcc_init (CC 초기 용량)",
    "QD_2": "summary QD(2) (CV 포함)",
    "cc_breakin": "CC break-in 상승폭",
    "QD_max_minus_2": "summary max QD − QD(2)",
    "dQ_logabsmin": "log₁₀ |min ΔQ|",
    "dQ_skew": "ΔQ 왜도",
    "dQ_kurt": "ΔQ 첨도 (Fisher)",
    "fade_slope_2_100": "용량 기울기 2–100",
    "fade_slope_91_100": "용량 기울기 91–100",
    "chargetime_avg5": "실측 충전시간 (2–6)",
    "avgC_80": "평균 C-rate (정책값)",
    "I_qw80": "충전전류 I_qw80 (실측)",
    "dT_chg": "충전 중 온도 상승",
    "Tavg_mean": "평균온도",
    "IR_min": "IR 최솟값",
    "IR_diff": "IR(100) − IR(2)",
}
CANDS = list(LBL)


def build() -> pd.DataFrame:
    ef = pd.read_csv(OUT / "early_features.csv")      # q5_correlation.py 산출물 (Qcc_init, cc_breakin 포함)
    q4 = pd.read_csv(OUT / "q4_cell_features.csv")[["cell_key", "I_qw80", "dT_chg"]]
    cells = {c["cell_key"]: c for c in load_all()}
    ef = ef.merge(q4, on="cell_key", how="left")
    ef["Qcc_init_8_12"] = [qcc_init(cells[k], *QCC_SENS) for k in ef["cell_key"]]
    ef["cv_dis_mAh"] = [cv_discharge_mAh(cells[k]) for k in ef["cell_key"]]
    ef["QD_init_2_6"] = [float(np.nanmedian(clean_summary(cells[k]["summary"]).set_index("cycle")
                                            .loc[QCC_WIN[0]:QCC_WIN[1], "QD"])) for k in ef["cell_key"]]
    ef["n_qcc_invalid_2_100"] = [int(qcc_series(cells[k]).isna().sum()) for k in ef["cell_key"]]
    return ef


def main() -> dict:
    import plot_style as ps  # noqa: F401  (그림 함수에서 사용; 한글 폰트 설정)
    ef = build()
    ef = ef[ef["group"].isin(["fastcharge", "newstructure"])].reset_index(drop=True)
    B = ef[(ef["batch"] == "batch1") & ef["labeled"]].reset_index(drop=True)
    ly = np.log10(B["cycle_life"].to_numpy(float))
    lv = B["dQ_logvar"].to_numpy(float)
    qc = B["Qcc_init"].to_numpy(float)

    # ── ① 편상관 ────────────────────────────────────────────────────
    part = {}
    for f in CANDS:
        v = B[f].to_numpy(float)
        rho, p_rho = stats.spearmanr(v, ly, nan_policy="omit")
        r1, p1, n1 = partial_r(v, ly, [lv])
        row = {"spearman_alone": float(rho), "p_alone": float(p_rho), "n": n1,
               "partial_logvar": r1, "p_partial_logvar": p1}
        if f != "Qcc_init":
            r2, p2, _ = partial_r(v, ly, [lv, qc])
            row.update({"partial_logvar_qcc": r2, "p_partial_logvar_qcc": p2})
        row["spearman_with_logvar"] = float(stats.spearmanr(v, lv, nan_policy="omit")[0])
        part[f] = row
    m1, m2 = len(CANDS), len(CANDS) - 1
    bonf = {"m_logvar": m1, "m_logvar_qcc": m2,
            "rcrit_logvar": r_crit(len(B) - 3, 0.05 / m1), "rcrit_logvar_qcc": r_crit(len(B) - 4, 0.05 / m2),
            "rcrit_nominal_logvar": r_crit(len(B) - 3, 0.05)}
    bonf["pass_logvar"] = [f for f in CANDS if part[f]["p_partial_logvar"] < 0.05 / m1]
    bonf["pass_logvar_qcc"] = [f for f in CANDS if f != "Qcc_init" and part[f]["p_partial_logvar_qcc"] < 0.05 / m2]
    bonf["nominal_logvar_qcc"] = [f for f in CANDS if f != "Qcc_init" and part[f]["p_partial_logvar_qcc"] < 0.05]

    # Qcc_init 편상관 강건성: 한 셀씩 제외 / 부트스트랩 / 민감도 정의(cycle 8–12)
    loo = []
    for i in range(len(B)):
        m = np.arange(len(B)) != i
        loo.append(partial_r(qc[m], ly[m], [lv[m]])[0])
    rng = np.random.default_rng(RANDOM_STATE)
    bs = []
    for _ in range(N_BOOT):
        j = rng.integers(0, len(B), len(B))
        bs.append(partial_r(qc[j], ly[j], [lv[j]])[0])
    q812 = B["Qcc_init_8_12"].to_numpy(float)
    qcc_robust = {"loo_min": float(min(loo)), "loo_max": float(max(loo)),
                  "boot_ci95": [float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))],
                  "sens_8_12_r_with_2_6": float(np.corrcoef(q812, qc)[0, 1]),
                  "sens_8_12_partial_logvar": partial_r(q812, ly, [lv])[0],
                  "r_qcc_logvar": float(np.corrcoef(qc, lv)[0, 1]),
                  "vif_2feat": float(1 / (1 - np.corrcoef(qc, lv)[0, 1] ** 2)),
                  "r_qcc_qd2": float(np.corrcoef(qc, B["QD_2"])[0, 1]),
                  "r_qcc_qdinit_2_6": float(np.corrcoef(qc, B["QD_init_2_6"])[0, 1])}

    # B1 기술 회귀 (in-sample, CV 아님): M0 vs M1, summary QD_2 판
    def ols(cols):
        X = np.column_stack([np.ones(len(B))] + [B[c].to_numpy(float) for c in cols])
        b = np.linalg.lstsq(X, ly, rcond=None)[0]
        e = ly - X @ b
        mape = float(np.mean(np.abs(10 ** (X @ b) - 10 ** ly) / 10 ** ly) * 100)
        return b, float(np.sqrt(np.mean(e ** 2))), mape
    b0, rmse0, mape0 = ols(["dQ_logvar"])
    b1, rmse1, mape1 = ols(["dQ_logvar", "Qcc_init"])
    bq, rmseq, mapeq = ols(["dQ_logvar", "QD_2"])
    desc = {"M0": {"coef": b0.tolist(), "rmse_log10": rmse0, "mape_in_sample": mape0},
            "M1": {"coef": b1.tolist(), "rmse_log10": rmse1, "mape_in_sample": mape1,
                   "qcc_coef_per_Ah": float(b1[2]), "plus10mAh_life_factor": float(10 ** (b1[2] * 0.01))},
            "M1_summaryQD2": {"coef": bq.tolist(), "rmse_log10": rmseq, "qd2_coef_per_Ah": float(bq[2])},
            "note": "Batch 1 labeled 36셀 in-sample 기술 회귀. 모델 선택 근거 아님(DAY 2 그룹 CV 1-SE 규칙으로 결정)."}
    # 중도절단 하한 점검 (B1 00~04: 피처가 라벨 범위 밖인 자연 외삽 실험)
    C = ef[(ef["batch"] == "batch1") & ef["censored"]].copy()
    C["pred_M0"] = 10 ** (b0[0] + b0[1] * C["dQ_logvar"])
    C["pred_M1"] = 10 ** (b1[0] + b1[1] * C["dQ_logvar"] + b1[2] * C["Qcc_init"])
    c04 = C[C["idx"] <= 4]
    # break-in 을 더한 변형: 중도절단 10셀 하한 위반 (계수 의미가 피처 구성에 따라 바뀌는지 = 외삽 안정성)
    viol = {}
    for name, cols in (("M0", ["dQ_logvar"]), ("M1", ["dQ_logvar", "Qcc_init"]),
                       ("M1+summary_breakin", ["dQ_logvar", "Qcc_init", "QD_max_minus_2"]),
                       ("M1+cc_breakin", ["dQ_logvar", "Qcc_init", "cc_breakin"])):
        bb = ols(cols)[0]
        pr = 10 ** (bb[0] + sum(bb[i + 1] * C[c] for i, c in enumerate(cols)))
        bad = C[pr < C["cycle_life"]]
        viol[name] = {"n_violations_of_10": int(len(bad)),
                      "cells": [{"cell": k, "pred": float(round(v, 0)), "lower_bound": float(lb),
                                 "rel": float(v / lb - 1)} for k, v, lb in zip(bad["cell_key"], pr[bad.index], bad["cycle_life"])]}
    desc["censored_lower_bound_check_all10"] = viol
    desc["censored_00_04"] = {"lower_bound": c04["cycle_life"].tolist(), "pred_M0": c04["pred_M0"].round(0).tolist(),
                              "pred_M1": c04["pred_M1"].round(0).tolist(),
                              "M0_meets_lb": int((c04["pred_M0"] >= c04["cycle_life"]).sum()),
                              "M1_meets_lb": int((c04["pred_M1"] >= c04["cycle_life"]).sum())}

    # ── ② 정책 평균 단위 vs 정책 내부(복제쌍) ─────────────────────────
    B["ly"] = ly
    pm = B.groupby("policy")[["dQ_logvar", "Qcc_init", "ly"]].mean()
    size = B.groupby("policy")["ly"].transform("size")
    mm = (size == 2).to_numpy()
    pairs = []
    for p, d in B[mm].groupby("policy"):
        d = d.sort_values("cycle_life")
        s_, l_ = d.iloc[0], d.iloc[1]                         # 짧은 셀, 긴 셀
        pairs.append({"policy": p, "long": l_["cell_key"], "short": s_["cell_key"],
                      "d_ly": float(l_["ly"] - s_["ly"]), "d_life": float(l_["cycle_life"] - s_["cycle_life"]),
                      "d_logvar": float(l_["dQ_logvar"] - s_["dQ_logvar"]),
                      "d_qcc_mAh": float((l_["Qcc_init"] - s_["Qcc_init"]) * 1000)})
    P = pd.DataFrame(pairs)
    rep = {}
    for f, col, sign in (("dQ_logvar", "d_logvar", -1), ("Qcc_init", "d_qcc_mAh", +1)):
        dv = (B[f] - B.groupby("policy")[f].transform("mean"))[mm]
        dy = (B["ly"] - B.groupby("policy")["ly"].transform("mean"))[mm]
        k = int((np.sign(P[col]) == sign).sum())
        rep[f] = {"policy_mean_r": float(np.corrcoef(pm[f], pm["ly"])[0, 1]), "n_policies": int(len(pm)),
                  "policy_mean_p": float(stats.pearsonr(pm[f], pm["ly"])[1]),
                  "within_policy_r": float(np.corrcoef(dv, dy)[0, 1]),
                  "within_policy_p": float(stats.pearsonr(dv, dy)[1]), "n_cells_in_pairs": int(mm.sum()),
                  "pairs_expected_direction": k, "n_pairs": int(len(P)),
                  "sign_test_p": float(stats.binomtest(k, len(P)).pvalue),
                  "between_policy_var_share": between_share(B[f], B["policy"])}
    rep["log10_life_between_policy_var_share"] = between_share(B["ly"], B["policy"])
    rep["median_abs_pair_d_life"] = float(P["d_life"].abs().median())
    rep["median_abs_pair_d_qcc_mAh"] = float(P["d_qcc_mAh"].abs().median())

    # ── ③ 배치 수준 이동 (라벨 미사용) ────────────────────────────────
    ref_all = ef[ef["batch"] == "batch1"]                        # B1 46셀 전체 (라벨 무관)
    offsets = {}
    for w in ("Qcc_init", "QD_2"):
        mu, sd = float(ref_all[w].median()), float(B[w].std(ddof=1))
        lo_, hi_ = float(B[w].min()), float(B[w].max())
        # 두 측정 정의 모두 같은 B1 계수(M1 의 Qcc_init 계수, log10/Ah)로 환산 → 이동량 차이만 비교
        coef = desc["M1"]["qcc_coef_per_Ah"]
        own = desc["M1"]["qcc_coef_per_Ah"] if w == "Qcc_init" else desc["M1_summaryQD2"]["qd2_coef_per_Ah"]
        o = {"b1_median_all46_Ah": mu, "b1_labeled_sd_mAh": sd * 1000, "coef_per_Ah": coef,
             "note": "pred_factor = 10**(M1 Qcc 계수 × 이동량); pred_factor_own_coef = 해당 피처로 적합한 2-피처 계수 사용"}
        for (b, g), d in ef[ef["batch"] != "batch1"].groupby(["batch", "group"]):
            sh = float(d[w].median() - mu)
            o[f"{b}_{g}"] = {"n": int(len(d)), "shift_mAh": sh * 1000, "shift_sd": sh / sd,
                             "out_of_b1_range": int(((d[w] < lo_) | (d[w] > hi_)).sum()),
                             "pred_factor": float(10 ** (coef * sh)), "pred_factor_own_coef": float(10 ** (own * sh))}
        offsets[w] = o
    cvd = {f"{b}_{g}": {"median_mAh": float(d["cv_dis_mAh"].median()), "n": int(len(d))}
           for (b, g), d in ef.groupby(["batch", "group"])}
    invalid = ef.groupby("batch")["n_qcc_invalid_2_100"].sum().astype(int).to_dict()

    # ── ④ Hold-out ─────────────────────────────────────────────────
    ho, ho_info = holdout_split(B)
    hc = B[["cell_key", "idx", "policy", "cycle_life", "dQ_logvar", "Qcc_init"]].copy()
    hc["split"] = np.where(hc["policy"].isin(ho), "holdout", "train")
    ROOT.joinpath("results").mkdir(exist_ok=True)
    hc.sort_values(["split", "idx"])[["cell_key", "policy", "cycle_life", "split"]].to_csv(
        ROOT / "results" / "holdout_cells.csv", index=False)
    H, T = hc[hc["split"] == "holdout"], hc[hc["split"] == "train"]
    out_rng = []
    for _, r in H.iterrows():
        for f in ("dQ_logvar", "Qcc_init"):
            if not T[f].min() <= r[f] <= T[f].max():
                out_rng.append({"cell": r["cell_key"], "feature": f, "value": float(r[f]),
                                "train_range": [float(T[f].min()), float(T[f].max())]})
    expected = [16, 17, 23, 30, 31, 32, 33, 44, 45]
    ho_info.update({"holdout_policies": ho, "holdout_idx": sorted(H["idx"].astype(int).tolist()),
                    "matches_preregistered": sorted(H["idx"].astype(int).tolist()) == expected,
                    "holdout_n": int(len(H)), "holdout_life": [float(H.cycle_life.min()), float(H.cycle_life.max()),
                                                               float(H.cycle_life.median())],
                    "train_n": int(len(T)), "train_policies": int(T.policy.nunique()),
                    "train_life": [float(T.cycle_life.min()), float(T.cycle_life.max()), float(T.cycle_life.median())],
                    "holdout_out_of_train_feature_range": out_rng})

    # ── 그림 ───────────────────────────────────────────────────────
    fig_partial(part, bonf)
    fig_pairs(B, pm, P, rep)
    fig_offsets(ef, B, offsets, cvd)

    res = {"settings": {"n_batch1": int(len(B)), "qcc_window": QCC_WIN, "qcc_sensitivity": QCC_SENS,
                        "breakin_smooth": BREAKIN_SMOOTH, "n_boot": N_BOOT, "random_state": RANDOM_STATE,
                        "partial": "Pearson, 통제변수 OLS 잔차끼리, log10(cycle_life), df=n−2−k"},
           "partial_correlations": part, "bonferroni": bonf, "qcc_robustness": qcc_robust,
           "descriptive_fits": desc, "replicate_pairs": rep, "pairs_table": P.round(4).to_dict(orient="records"),
           "batch_offsets_label_free": offsets, "cv_discharge_cycle2_6_mAh": cvd,
           "qdlin_endpoint_invalid_values_cycle2_100": invalid, "holdout": ho_info}
    with open(OUT / "q5d_results.json", "w", encoding="utf-8") as fh:
        json.dump(res, fh, ensure_ascii=False, indent=2, default=float)
    ef[["batch", "cell_key", "idx", "policy", "group", "cycle_life", "labeled", "censored", "dQ_logvar", "Qcc_init",
        "Qcc_init_8_12", "cc_breakin", "QD_2", "QD_init_2_6", "cv_dis_mAh", "I_qw80", "dT_chg"]].to_csv(
        OUT / "q5d_cell_features.csv", index=False)
    return res


# ════════════════════════════════════════════════════════════════════════
# 그림
# ════════════════════════════════════════════════════════════════════════
def fig_partial(part: dict, bonf: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    import plot_style as ps
    order = sorted(CANDS, key=lambda f: -abs(part[f]["partial_logvar"]))
    n = len(order)
    fig, ax = plt.subplots(figsize=(9.2, 7.4))
    y = np.arange(n)[::-1]
    c1, c2 = "#93C5FD", "#1E3A8A"
    rn = bonf["rcrit_nominal_logvar"]
    ax.axvspan(-rn, rn, color="#F3F4F6", lw=0, zorder=0)
    v1 = [part[f]["partial_logvar"] for f in order]
    v2 = [part[f].get("partial_logvar_qcc", np.nan) for f in order]
    ax.barh(y + 0.19, v1, height=0.36, color=c1, zorder=2)
    ax.barh(y - 0.19, v2, height=0.36, color=c2, zorder=2)
    for yy, a, b in zip(y, v1, v2):
        ax.text(a + (0.02 if a >= 0 else -0.02), yy + 0.19, f"{a:+.2f}", va="center",
                ha="left" if a >= 0 else "right", fontsize=9, color="#374151", zorder=6,
                bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
        if np.isfinite(b):
            ax.text(b + (0.02 if b >= 0 else -0.02), yy - 0.19, f"{b:+.2f}", va="center",
                    ha="left" if b >= 0 else "right", fontsize=9, color=c2, fontweight="bold", zorder=6,
                    bbox=dict(boxstyle="square,pad=0.05", fc="white", ec="none", alpha=0.85))
        else:
            ax.text(0.02, yy - 0.19, "(통제변수)", va="center", fontsize=8.5, color="#6B7280")
    rb = bonf["rcrit_logvar_qcc"]
    for s in (-1, 1):
        ax.axvline(s * rb, color="#DC2626", ls="--", lw=1.2, zorder=3)
    ax.axvline(0, color="#6B7280", lw=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([LBL[f] for f in order], fontsize=10.5)
    ax.set_xlim(-0.95, 0.95)
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("편상관 r (log₁₀ cycle_life, Batch 1 labeled n=36)", fontsize=10.5)
    h = [Patch(color=c1, label="log₁₀Var(ΔQ) 통제"), Patch(color=c2, label="log₁₀Var(ΔQ) + Qcc_init 통제"),
         Line2D([], [], color="#DC2626", ls="--", label=f"Bonferroni 기준 |r|≈{rb:.2f} (후보 {bonf['m_logvar_qcc']}–{bonf['m_logvar']}개)"),
         Patch(color="#F3F4F6", label=f"|r|<{rn:.2f}: 보정 전 p>0.05")]
    ax.legend(handles=h, loc="lower center", bbox_to_anchor=(0.42, -0.25), ncol=2, fontsize=9.5)
    ax.set_title("ΔQ 분산을 통제한 뒤 남는 정보 — Batch 1 편상관", fontsize=12.5, loc="left")
    ps.save(fig, "q5d_partial_corr")


def fig_pairs(B: pd.DataFrame, pm: pd.DataFrame, P: pd.DataFrame, rep: dict) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    import plot_style as ps
    fig, axes = plt.subplots(2, 2, figsize=(8.8, 6.6), gridspec_kw={"hspace": 0.85, "wspace": 0.34})
    blue, dark = ps.BATCH_COLOR["batch1"], "#111827"
    specs = (("dQ_logvar", "d_logvar", 1.0, "log₁₀Var(ΔQ₁₀₀₋₁₀)", "Δ log₁₀Var(ΔQ)   (긴 셀 − 짧은 셀)", -1),
             ("Qcc_init", "d_qcc_mAh", 1000.0, "Qcc_init (mAh)", "Δ Qcc_init, mAh   (긴 셀 − 짧은 셀)", +1))
    for row, (f, col, sc, xl, dxl, sign) in enumerate(specs):
        r = rep[f]
        a = axes[row, 0]
        a.scatter(pm[f] * sc, 10 ** pm["ly"], s=40, color=blue, edgecolor="white", lw=0.6, zorder=3)
        k, c0 = np.polyfit(pm[f] * sc, pm["ly"], 1)
        xx = np.linspace((pm[f] * sc).min(), (pm[f] * sc).max(), 20)
        a.plot(xx, 10 ** (k * xx + c0), color=blue, lw=1.4, alpha=0.8)
        a.set_yscale("log")
        a.set_yticks([600, 800, 1000])
        a.set_yticklabels(["600", "800", "1000"])
        a.minorticks_off()
        a.set_xlabel(xl, fontsize=10.5)
        a.set_ylabel("정책 평균 수명", fontsize=10.5)
        a.set_title(f"정책 평균 {r['n_policies']}개: r = {r['policy_mean_r']:+.2f}", fontsize=11.5, loc="left")
        b = axes[row, 1]
        exp = np.sign(P[col]) == sign
        b.axvline(0, color="#6B7280", lw=0.9)
        b.scatter(P.loc[exp, col], P.loc[exp, "d_life"], s=44, color=dark, zorder=3)
        b.scatter(P.loc[~exp, col], P.loc[~exp, "d_life"], s=44, facecolor="white", edgecolor=dark, lw=1.2, zorder=3)
        b.set_xlabel(dxl, fontsize=10.5)
        b.set_ylabel("Δ 수명 (사이클)", fontsize=10.5)
        b.set_ylim(0, P["d_life"].max() * 1.45)
        lim = np.abs(P[col]).max() * 1.15
        b.set_xlim(-lim, lim)
        b.set_title(f"복제쌍 {r['n_pairs']}개: 기대 방향 {r['pairs_expected_direction']}/{r['n_pairs']}"
                    f" (부호검정 p={r['sign_test_p']:.3f})", fontsize=11.5, loc="left")
        h = [Line2D([], [], marker="o", ls="none", color=dark, ms=6.5),
             Line2D([], [], marker="o", ls="none", mfc="white", mec=dark, ms=6.5)]
        exp_txt = "기대 방향(Δx<0)" if sign < 0 else "기대 방향(Δx>0)"
        b.legend(h, [exp_txt, "반대 방향"], loc="upper left", fontsize=9.3, ncol=2, handletextpad=0.2,
                 columnspacing=0.8, title=f"정책 내부 r = {r['within_policy_r']:+.2f}", title_fontsize=9.6,
                 alignment="left")
    fig.text(0.01, 0.985, "① log₁₀Var(ΔQ): 정책(프로토콜) 사이 수명 차이와 함께 움직임", fontsize=12.2, fontweight="bold")
    fig.text(0.01, 0.485, "② Qcc_init: 같은 정책 복제셀 사이 수명 차이와 함께 움직임", fontsize=12.2, fontweight="bold")
    ps.save(fig, "q5d_replicate_pairs")


def fig_offsets(ef: pd.DataFrame, B: pd.DataFrame, offsets: dict, cvd: dict) -> None:
    import matplotlib.pyplot as plt
    import plot_style as ps
    groups = [("batch1", "fastcharge", "Batch 1", ps.BATCH_COLOR["batch1"], "o"),
              ("batch2", "fastcharge", "B2 fastcharge", ps.BATCH_COLOR["batch2"], "o"),
              ("batch2", "newstructure", "B2 newstructure", "#9A3412", "s"),
              ("batch3", "newstructure", "Batch 3", ps.BATCH_COLOR["batch3"], "D")]
    mu = offsets["Qcc_init"]["b1_median_all46_Ah"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 5.2), gridspec_kw={"width_ratios": [1.15, 1], "wspace": 0.36})
    for b, g, name, col, mk in groups:
        d = ef[(ef["batch"] == b) & (ef["group"] == g)]
        a1.scatter((d["Qcc_init"] - mu) * 1000, d["cv_dis_mAh"], s=30, marker=mk, color=col, alpha=0.8,
                   edgecolor="white", lw=0.5, label=f"{name.replace('Batch ', 'B')} ({len(d)})")
    a1.axvline(0, color="#6B7280", lw=0.9, ls=":")
    a1.set_xlabel("Qcc_init − B1 중앙값 (mAh)  ← CC 수준 이동", fontsize=10.5)
    a1.set_ylabel("CV 방전분 = summary QD − CC 끝점 (mAh, cycle 2–6)", fontsize=10.5)
    a1.set_xlim(-44, 36)
    a1.legend(fontsize=9.5, loc="upper right", handletextpad=0.2)
    a1.set_title("summary QD = CC 끝점 + CV 방전분 (라벨 미사용)", fontsize=11.5, loc="left")
    # 오른쪽: 중앙값 이동 (summary QD(2) vs Qcc_init) + 예측 환산
    rows = [("batch2_fastcharge", "B2 fastcharge"), ("batch2_newstructure", "B2 newstructure"),
            ("batch3_newstructure", "Batch 3")]
    y = np.arange(len(rows))[::-1]
    sq = [offsets["QD_2"][k]["shift_mAh"] for k, _ in rows]
    cq = [offsets["Qcc_init"][k]["shift_mAh"] for k, _ in rows]
    a2.barh(y + 0.18, sq, height=0.34, color="#D1D5DB", label="summary QD(2) (CV 포함)")
    a2.barh(y - 0.18, cq, height=0.34, color="#1E3A8A", label="Qcc_init (CC 끝점)")
    for yy, k, s_, c_ in zip(y, [k for k, _ in rows], sq, cq):
        fs, fc = offsets["QD_2"][k]["pred_factor"], offsets["Qcc_init"][k]["pred_factor"]
        a2.text(s_ + (0.6 if s_ >= 0 else -0.6), yy + 0.18, f"{s_:+.1f} (×{fs:.2f})", va="center",
                ha="left" if s_ >= 0 else "right", fontsize=9.5, color="#4B5563")
        a2.text(c_ + (0.6 if c_ >= 0 else -0.6), yy - 0.18, f"{c_:+.1f} (×{fc:.2f})", va="center",
                ha="left" if c_ >= 0 else "right", fontsize=9.5, color="#1E3A8A", fontweight="bold")
    a2.axvline(0, color="#6B7280", lw=0.9)
    a2.set_yticks(y)
    a2.set_yticklabels([n for _, n in rows], fontsize=10.5)
    a2.set_xlim(-31, 31)
    a2.grid(axis="y", visible=False)
    a2.set_xlabel("B1 중앙값 대비 배치 중앙값 이동 (mAh)\n괄호 = 같은 B1 계수(+10 mAh → ×1.08)로 환산한 예측 배율", fontsize=10.5)
    a2.legend(fontsize=9.5, loc="lower right")
    a2.set_title("절대 수준 피처의 배치 이동 (라벨 미사용)", fontsize=11.5, loc="left")
    ps.save(fig, "q5d_qcc_batch_offset")


if __name__ == "__main__":
    r = main()
    pc = r["partial_correlations"]
    for f in CANDS:
        print(f"{f:18s} rho={pc[f]['spearman_alone']:+.2f} | logvar {pc[f]['partial_logvar']:+.3f} (p={pc[f]['p_partial_logvar']:.2g})"
              f" | +Qcc {pc[f].get('partial_logvar_qcc', np.nan):+.3f} (p={pc[f].get('p_partial_logvar_qcc', np.nan):.2g})")
    for k in ("bonferroni", "qcc_robustness", "descriptive_fits", "replicate_pairs", "batch_offsets_label_free",
              "cv_discharge_cycle2_6_mAh", "qdlin_endpoint_invalid_values_cycle2_100", "holdout"):
        print(k, ":", json.dumps(r[k], ensure_ascii=False, default=float)[:1500])
