"""DAY 2 오류 분석 — README 「오류 분석」·「ESS 도메인 해석」과 Gap 해석의 근거를 만든다.

원칙
- 저장된 예측 파일만 읽는다. 모델 재적합·재평가·선택 변경은 하지 않는다.
- Test(Batch 2)·Batch 3 단계는 이미 1회 실행됐다(results/.test_done, .batch3_done). 이 스크립트는 그 단계를
  다시 부르지 않고, 저장된 예측으로 다시 계산한 MAPE가 보고 수치와 같은지만 확인한다.
- Batch 2·3 라벨은 사후 해석에만 쓴다. 개선 방향은 '향후 과제'로만 쓰고 B2 라벨 재학습을 전제로 하지 않는다.
- 상관은 인과가 아니다. 원인 가설은 '근거가 가리키는 후보'로만 쓴다.

입력
  results/test_predictions.csv · batch3_predictions.csv · valid_predictions.csv · cv_oof.csv  (저장된 예측)
  results/features.csv · diagnostic_features.csv                                            (피처·정점 사이클)
  results/test.json · batch3.json · model_performance*.csv · model_candidates.csv            (보고 수치)
  results/batch3_excl_corrected.json   Batch 3 원저자 코드 제거 4셀(#02·#37·#42·#43) 제외 지표 (train.py report 단계)
  results/rest_gap_summary.json        Batch 2 기록 공백 점검 (src/check_rest_gap.py, 없으면 생략)
  results/eda/q5_results.json                                                                (EDA Q5 B1 선·수명비)
출력
  reports/figures/day2_pred_vs_true.png   예측 vs 실제 (B1 CV-OOF, Valid, B2, B3), y=x 와 ±20% 띠
  reports/figures/day2_b2_error.png       B2 오차 vs ΔQ 깊이 (B1 관계선, 수명비 0.76)
  reports/figures/day2_m1_vs_m0.png       M1 vs M0 부호 오차와 초기 용량 효과
  results/error_analysis.json             최악 10셀·공통점·H1~H6 판정·Gap 해석·원인 가설·개선 방향·도메인 해석
실행: cd ess-battery-project && python src/error_analysis.py
"""
from __future__ import annotations

import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import plot_style as ps  # noqa: E402  (한글 폰트·save)
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullLocator  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
RES = ROOT / "results"
OUT_JSON = RES / "error_analysis.json"

SEL, BASE = "M1", "M0"
TARGET = 9.1
B1_LABEL_MIN, B1_LABEL_MAX = 534.0, 1074.0     # B1 라벨 36셀 수명 범위 (DAY 1 Q1)
SHORT = 550.0                                   # 노션 분류 경계 (사후 스크리닝 해석용)
BAND = 0.20                                     # 그림의 ±20% 띠
# 원논문 Table 1 (Severson et al. 2019): 평균 % 오차와 RMSE. primary_excl = 이상 셀 1개를 뺀 1차 테스트(42셀)
PAPER_T1 = {"Variance": {"primary_excl": 13.2, "secondary": 11.4, "secondary_rmse": 196},
            "Discharge": {"primary_excl": 10.1, "secondary": 8.6, "secondary_rmse": 173},
            "Full": {"primary_excl": 7.5, "secondary": 10.7, "secondary_rmse": 214}}
PAPER_N = {"primary_excl": 42, "secondary": 40}

# 그림 글꼴 — 본문 12pt 이상
plt.rcParams.update({
    "font.size": 13, "axes.titlesize": 14.5, "axes.labelsize": 13.5, "xtick.labelsize": 12.5,
    "ytick.labelsize": 12.5, "legend.fontsize": 12, "figure.titlesize": 16,
})
C_B1, C_FC, C_NS, C_B3 = "#2563EB", "#EA580C", "#7C3AED", "#059669"   # validate_palette.js 4색 PASS
C_M0, C_M1, INK, MUTED = "#9CA3AF", "#111827", "#111827", "#6B7280"
KO = {"dQ_logvar": "ΔQ 깊이", "Qcc_init": "초기 용량"}


# ───────────────────────────── 로드 ─────────────────────────────
def sha12(p: Path) -> str:
    return hashlib.sha256(p.read_bytes()).hexdigest()[:12]


def load() -> dict:
    for m in (RES / ".test_done", RES / ".batch3_done"):
        assert m.exists() and "completed" in m.read_text(), f"{m.name} 없음 — 이 분석은 저장된 1회 결과만 읽는다"
    F = pd.read_csv(RES / "features.csv")
    D = pd.read_csv(RES / "diagnostic_features.csv")
    F = F.merge(D[["cell_key", "cc_breakin_peak_cycle", "qd_breakin_peak_cycle"]], on="cell_key", how="left")
    test, b3 = (json.loads((RES / f).read_text()) for f in ("test.json", "batch3.json"))
    b3x = json.loads((RES / "batch3_excl_corrected.json").read_text())   # 정정된 제외 목록·지표 (저장 예측에서 계산)
    gap_p = RES / "rest_gap_summary.json"
    rest_gap = json.loads(gap_p.read_text()) if gap_p.exists() else None
    q5 = json.loads((RES / "eda" / "q5_results.json").read_text())
    assert test["selected_id"] == b3["selected_id"] == SEL

    def preds(fname: str) -> pd.DataFrame:
        P = pd.read_csv(RES / fname)
        a = P[P.model == SEL].set_index("cell_key")
        b = P[P.model == BASE].set_index("cell_key")[["pred", "signed_pct", "ape"]].add_suffix("_M0")
        out = a.join(b, how="inner")
        assert len(out) == len(a) == len(b)
        return out.join(F.set_index("cell_key")[["Qcc_init", "cc_breakin", "cc_breakin_peak_cycle",
                                                 "qd_breakin_peak_cycle"]])

    # CV-OOF: 셀별로 seed 20회의 out-of-fold 예측을 기하평균 (그림·참고용; 보고 Train MAPE는 seed×fold 평균)
    O = pd.read_csv(RES / "cv_oof.csv")
    assert (O.groupby(["model", "cell_key"]).size() == 20).all()

    def oof(model: str) -> pd.DataFrame:
        o = O[O.model == model]
        return o.groupby("cell_key").agg(true=("true", "first"), policy=("policy", "first"),
                                         pred=("pred", lambda x: float(10 ** np.log10(x).mean())))
    oof1, oof0 = oof(SEL), oof(BASE)
    oof1 = oof1.join(oof0[["pred"]].rename(columns={"pred": "pred_M0"}))
    for d in (oof1,):
        d["signed_pct"] = (d.pred / d.true - 1) * 100
        d["signed_pct_M0"] = (d.pred_M0 / d.true - 1) * 100
    oof1 = oof1.join(F.set_index("cell_key")[["dQ_logvar", "Qcc_init"]])

    perf = {k: pd.read_csv(RES / f) for k, f in (("M1", "model_performance_batch3.csv"),
                                                    ("M0", "model_performance_batch3_M0.csv"))}
    return dict(F=F, test=test, b3=b3, b3x=b3x, rest_gap=rest_gap, q5=q5, O=O, oof=oof1,
                B2=preds("test_predictions.csv"), B3=preds("batch3_predictions.csv"),
                V=preds("valid_predictions.csv"), perf=perf,
                cand=pd.read_csv(RES / "model_candidates.csv"))


def reported(perf: pd.DataFrame) -> dict:
    """model_performance_batch3*.csv 에서 행 이름 → 값 (Gap 행은 둘째 열에 이름이 있다)."""
    out, test_seen = {}, 0
    for _, r in perf.iterrows():
        name = r.iloc[0] if isinstance(r.iloc[0], str) and r.iloc[0] else r.iloc[1]
        if name == "Gap (Target-Test)":
            name = "Gap (Target-Test)" if test_seen < 2 else "Gap (Target-Test, B3)"
        if str(name).startswith("Test"):
            test_seen += 1
        out[name] = float(r["MAPE (%)"])
    return out


def consistency(S: dict) -> dict:
    """저장된 예측으로 다시 계산한 MAPE == 보고 수치 (모델·수치 불변 확인)."""
    O = S["O"]
    chk = {}
    for model in (SEL, BASE):
        rep = reported(S["perf"][model])
        cv = O[O.model == model].groupby(["seed", "fold"]).ape.mean().mean()
        d = {"Train (Batch 1 CV)": cv,
             "Valid (Batch 1 Hold-out)": S["V"]["ape" if model == SEL else "ape_M0"].mean(),
             "Test (Batch 2)": S["B2"]["ape" if model == SEL else "ape_M0"].mean(),
             "Test (Batch 3)": S["B3"]["ape" if model == SEL else "ape_M0"].mean()}
        for k, v in d.items():
            assert abs(round(v, 2) - rep[k]) < 0.006, (model, k, v, rep[k])
        chk[model] = {"recomputed": {k: round(v, 3) for k, v in d.items()}, "reported": rep}
    return chk


# ───────────────────────────── 계산 ─────────────────────────────
def paper_split_overlap() -> dict:
    """원저자 1차 테스트 분할(LoadData.m: test_ind = [1:2:84, 84])의 Batch 1 부분에서 같은 충전 정책 셀이
    학습 쪽에도 있는지 센다. 셀 순서·정책 이름만 쓴다(라벨 미사용)."""
    F = pd.read_csv(RES / "features.csv")
    b1 = F[F.batch == "batch1"].sort_values("idx")
    b1 = b1[~b1.idx.isin([8, 10, 12, 13, 22])].reset_index(drop=True)   # 원저자 코드가 Batch 1에서 뺀 5셀
    test, train = b1.iloc[0::2], b1.iloc[1::2]                         # MATLAB 1:2:… 의 Batch 1 부분 = 0, 2, 4, …
    tr_pol = set(train.policy)
    rep = b1.policy.value_counts()
    rep = set(rep[rep >= 2].index)
    split = {p for p in rep if p in tr_pol and p in set(test.policy)}
    return {"정의": "원저자 1차 분할의 Batch 1 부분(41셀, #08·10·12·13·22 제외)을 셀 순서로 번갈아 test/train으로 나눔",
            "n_test_b1": int(len(test)), "n_train_b1": int(len(train)),
            "n_test_with_same_policy_in_train": int(test.policy.isin(tr_pol).sum()),
            "n_replicated_policies": len(rep), "n_replicated_policies_split": len(split),
            "paper_9p1_as_weighted_mean_full": round((PAPER_N["primary_excl"] * PAPER_T1["Full"]["primary_excl"]
                                                      + PAPER_N["secondary"] * PAPER_T1["Full"]["secondary"])
                                                     / (PAPER_N["primary_excl"] + PAPER_N["secondary"]), 2)}


def b1_line(S: dict) -> tuple[float, float]:
    s = S["q5"]["per_batch_slope_log10life_on_dQ_logvar"]["batch1"]
    c = S["test"]["M0"]["coefs_raw_fit36"]
    assert abs(s["slope"] - c["dQ_logvar"]) < 1e-9 and abs(s["intercept"] - c["intercept"]) < 1e-9
    return s["slope"], s["intercept"]   # EDA Q5의 B1 선 = M0(36셀 재적합)과 같은 선


def rnd(x, k=1):
    return None if x is None or (isinstance(x, float) and np.isnan(x)) else round(float(x), k)


def worst_cells(d: pd.DataFrame, rng: list[float], q_med: float, removed: set) -> list[dict]:
    rows = []
    for ck, r in d.sort_values("ape", ascending=False).head(10).iterrows():
        rows.append({
            "셀": ck, "그룹": r.group, "충전 정책": r.policy,
            "실제 수명": int(r.true), "예측 수명 (M1)": int(round(r.pred)), "APE (%)": rnd(r.ape),
            "부호 오차 (%)": rnd(r.signed_pct), "방향": "과대예측" if r.signed_pct > 0 else "과소예측",
            "M0 예측": int(round(r.pred_M0)), "M0 부호 오차 (%)": rnd(r.signed_pct_M0),
            "ΔQ 깊이 (dQ_logvar)": rnd(r.dQ_logvar, 3),
            "ΔQ 학습 범위": "밖" if r.out_of_train_range else "안",
            "ΔQ 범위 밖 방향": (None if not r.out_of_train_range else
                             ("더 얕음(장수명 쪽)" if r.dQ_logvar < rng[0] else "더 깊음(단수명 쪽)")),
            "초기 용량 (Qcc_init, Ah)": rnd(r.Qcc_init, 4),
            "초기 용량 − B1 중앙값 (mAh)": rnd((r.Qcc_init - q_med) * 1000),
            "break-in 상승폭 (mAh)": rnd(r.cc_breakin, 2),
            "break-in 정점 사이클 (CC 끝점, 주)": int(r.cc_breakin_peak_cycle),
            "break-in 정점 사이클 (요약 QD, 보조)": int(r.qd_breakin_peak_cycle),
            "실제 수명 vs B1 라벨 범위": ("위(>1,074)" if r.true > B1_LABEL_MAX else
                                    "아래(<534)" if r.true < B1_LABEL_MIN else "안"),
            "원저자 코드 제거 셀": ck in removed,
        })
    return rows


def top_vs_rest(d: pd.DataFrame, q_med: float) -> dict:
    top = d.sort_values("ape", ascending=False).head(10)
    rest = d.drop(top.index)

    def s(x: pd.DataFrame) -> dict:
        return {"n": int(len(x)), "MAPE": rnd(x.ape.mean()), "과대예측 셀": int((x.signed_pct > 0).sum()),
                "fastcharge 셀": int((x.group == "fastcharge").sum()),
                "ΔQ 학습 범위 안 셀": int((~x.out_of_train_range).sum()),
                "실제 수명 중앙값": rnd(x.true.median(), 0),
                "실제 수명 > 1,074 셀": int((x.true > B1_LABEL_MAX).sum()),
                "실제 수명 < 550 셀": int((x.true < SHORT).sum()),
                "초기 용량 차이 중앙값 (mAh)": rnd((x.Qcc_init.median() - q_med) * 1000),
                "break-in 상승폭 중앙값 (mAh)": rnd(x.cc_breakin.median(), 2),
                "CC 정점 사이클 중앙값": rnd(x.cc_breakin_peak_cycle.median(), 1),
                "M1−M0 부호 오차 차이 평균 (%p)": rnd((x.signed_pct - x.signed_pct_M0).mean())}
    return {"상위 10셀": s(top), "나머지": s(rest)}


def group_stats(d: pd.DataFrame) -> dict:
    out = {}
    for g, x in [("전체", d), *list(d.groupby("group"))]:
        lr = np.log10(x.pred / x.true)
        out[g] = {"n": int(len(x)), "MAPE": rnd(x.ape.mean(), 2), "평균 부호 오차 (%)": rnd(x.signed_pct.mean()),
                  "부호 오차 SD (%p)": rnd(x.signed_pct.std()), "예측÷실제 중앙값": rnd((x.pred / x.true).median(), 3),
                  "과대예측": f"{int((x.signed_pct > 0).sum())}/{len(x)}",
                  "M0 MAPE": rnd(x.ape_M0.mean(), 2), "M0 평균 부호 오차 (%)": rnd(x.signed_pct_M0.mean()),
                  "M0 예측÷실제 중앙값": rnd((x.pred_M0 / x.true).median(), 3),
                  "수명 순위 Spearman (예측 vs 실제)": rnd(stats.spearmanr(x.pred, x.true).statistic, 2),
                  "Spearman(로그 잔차, 실제 수명)": rnd(stats.spearmanr(lr, x.true).statistic, 2),
                  "범위 안 MAPE": rnd(x.ape[~x.out_of_train_range].mean()) if (~x.out_of_train_range).any() else None,
                  "범위 밖 MAPE": rnd(x.ape[x.out_of_train_range].mean()) if x.out_of_train_range.any() else None,
                  "범위 안/밖 n": [int((~x.out_of_train_range).sum()), int(x.out_of_train_range.sum())]}
    return out


def _mape(pred: pd.Series, true: pd.Series) -> float:
    return float(np.mean(np.abs(pred / true - 1)) * 100)


def _partial_r(d: pd.DataFrame) -> tuple[float, float, int]:
    """초기 용량 vs log10 수명의 편상관 (ΔQ 깊이를 양쪽에서 선형으로 뺀 뒤 Pearson). 기술 통계, 모델과 무관."""
    A = np.column_stack([np.ones(len(d)), d.dQ_logvar.to_numpy(float)])
    y, x = np.log10(d.cycle_life.to_numpy(float)), d.Qcc_init.to_numpy(float)
    ry = y - A @ np.linalg.lstsq(A, y, rcond=None)[0]
    rx = x - A @ np.linalg.lstsq(A, x, rcond=None)[0]
    r = stats.pearsonr(rx, ry)
    return float(r.statistic), float(r.pvalue), int(len(d))


def posthoc(S: dict, pre: dict) -> dict:
    """라벨을 쓴 사후 계산 (보고 성능 아님 · 모델 재적합 없음 · 선택 불변).

    ① 공통 배율 제거: 각 배치의 M1 예측을 '예측÷실제 중앙값' 하나로 나눈 MAPE → 오차 중 셀 공통 치우침의 몫.
    ② 초기 용량 배치 이동 되돌리기: M1 예측 ÷ 테스트 전에 라벨 없이 구한 그룹별 예상 배율(pred_factor).
       M0 → (M1, 이동 되돌림) → M1 순서로 M1의 이득을 '배치 수준 이동 효과'와 '배치 안 셀 차이 효과'로 나눈다.
    ③ 초기 용량 편상관(ΔQ 깊이 통제)을 배치·그룹별로 다시 계산한다(B1 값은 DAY 1 Q5와 같다).
    """
    F, B2, B3 = S["F"], S["B2"], S["B3"]
    fac = {"B2": B2.group.map({"fastcharge": pre["B2 fastcharge"]["pred_factor"],
                                "newstructure": pre["B2 newstructure"]["pred_factor"]}),
           "B3": B3.group.map({"newstructure": pre["B3"]["pred_factor"]})}
    out = {"설명": "라벨을 쓴 사후 계산이다. 보고 성능이 아니며, 모델·선택·보고 수치는 바뀌지 않는다."}
    for k, d in (("B2", B2), ("B3", B3)):
        assert fac[k].notna().all()
        c = float((d.pred / d.true).median())
        adj = d.pred / fac[k]
        out[k] = {"M1 MAPE (보고)": rnd(_mape(d.pred, d.true), 2),
                  "공통 배율 (예측÷실제 중앙값)": rnd(c, 3),
                  "공통 배율 제거 후 M1 MAPE": rnd(_mape(d.pred / c, d.true), 2),
                  "초기 용량 이동 되돌린 M1 MAPE": rnd(_mape(adj, d.true), 2),
                  "초기 용량 이동 되돌린 M1 평균 부호 오차 (%)": rnd(((adj / d.true - 1) * 100).mean()),
                  "M0 MAPE (보고)": rnd(_mape(d.pred_M0, d.true), 2),
                  "M1÷M0 예측 비 중앙값": rnd(float((d.pred / d.pred_M0).median()), 3),
                  "수명 순위 Spearman M1 / M0": [rnd(stats.spearmanr(d.pred, d.true).statistic, 2),
                                             rnd(stats.spearmanr(d.pred_M0, d.true).statistic, 2)]}
    b2 = out["B2"]
    out["B2 M1 이득 분해 (%p)"] = {
        "M0 → M1 전체": rnd(b2["M0 MAPE (보고)"] - b2["M1 MAPE (보고)"]),
        "배치 안 셀 차이 (M0 → 이동 되돌린 M1)": rnd(b2["M0 MAPE (보고)"] - b2["초기 용량 이동 되돌린 M1 MAPE"]),
        "배치 수준 초기 용량 이동 (이동 되돌린 M1 → M1)": rnd(b2["초기 용량 이동 되돌린 M1 MAPE"] - b2["M1 MAPE (보고)"])}
    out["Gap(Valid-Test) 중 공통 배율의 몫"] = rnd(
        (b2["M1 MAPE (보고)"] - b2["공통 배율 제거 후 M1 MAPE"])
        / (b2["M1 MAPE (보고)"] - float(S["V"].ape.mean())), 2)
    L = F[F.labeled]
    pr = {}
    for name, d in (("B1", L[L.batch == "batch1"]), ("B2 fastcharge", L[(L.batch == "batch2") & (L.group == "fastcharge")]),
                    ("B2 newstructure", L[(L.batch == "batch2") & (L.group == "newstructure")]), ("B3", L[L.batch == "batch3"])):
        r, p, n = _partial_r(d)
        pr[name] = {"r": rnd(r, 2), "p": rnd(p, 4), "n": n}
    out["초기 용량 편상관 (ΔQ 깊이 통제, log10 수명)"] = pr
    return out


def analysis(S: dict) -> dict:
    F, B2, B3, V, oof = S["F"], S["B2"], S["B3"], S["V"], S["oof"]
    test, b3 = S["test"], S["b3"]
    rng = test["selected"]["train_dQ_logvar_range"]
    q_med = test["selected"]["qcc_shift_decomposition"]["b1_all46_median_Ah"]
    removed = set(S["b3x"]["removed_labeled_cells"])   # 원저자 공개 코드가 뺀 라벨 셀 #02·#37·#42·#43
    k, c = b1_line(S)
    L = F[F.labeled]

    # EDA Q5 수명비 (B1 선 대비) — 라벨로 계산한 DAY 1 기술 통계, 모델과 무관
    ratio = S["q5"]["ratio_life_to_batch1_line"]
    window = S["q5"]["b2_fastcharge_vs_batch1_same_window"]
    slopes = S["q5"]["per_batch_slope_log10life_on_dQ_logvar"]

    # 같은 충전 정책, 다른 배치 (사후 기술 통계)
    same_policy = {}
    for p in sorted(set(L[L.batch == "batch1"].policy) & set(L[L.batch == "batch2"].policy)):
        g = L[L.policy == p].groupby("batch")
        same_policy[p] = {b: {"n": int(len(x)), "수명 중앙값": rnd(x.cycle_life.median(), 0),
                              "ΔQ 깊이 중앙값": rnd(x.dQ_logvar.median(), 2)} for b, x in g}

    gs2, gs3 = group_stats(B2), group_stats(B3)
    vs = group_stats(V)

    # 사후 스크리닝: 예측 < 550 을 '단수명 경보'로 쓴다면 (B2)
    scr = {"실제 단수명(<550)": int((B2.true < SHORT).sum()), "M1 경보(예측<550)": int((B2.pred < SHORT).sum()),
           "경보 중 실제 단수명": int(((B2.pred < SHORT) & (B2.true < SHORT)).sum()),
           "오경보(경보인데 실제 ≥550)": int(((B2.pred < SHORT) & (B2.true >= SHORT)).sum()),
           "M0 경보 중 실제 단수명": int(((B2.pred_M0 < SHORT) & (B2.true < SHORT)).sum())}
    scr["놓친 단수명"] = scr["실제 단수명(<550)"] - scr["경보 중 실제 단수명"]

    # 오차 기여: B2 짧은 셀(<550)이 APE 합에서 차지하는 비율 (H3)
    short = B2.true < SHORT
    h3 = {"n_short": int(short.sum()), "APE 합 비율": rnd(B2.ape[short].sum() / B2.ape.sum() * 100),
          "짧은 셀 MAPE": rnd(B2.ape[short].mean()), "나머지 MAPE": rnd(B2.ape[~short].mean()),
          "B1 #20·#21 OOF (M1, 보고값)": "+1.5% · +5.4%"}

    # B3 라벨 범위 밖(>1,074) 장수명 셀
    lg = B3.true > B1_LABEL_MAX
    b3_long = {"n": int(lg.sum()), "MAPE": rnd(B3.ape[lg].mean()), "나머지 MAPE": rnd(B3.ape[~lg].mean()),
               "평균 부호 오차 M1 (%)": rnd(B3.signed_pct[lg].mean()), "평균 부호 오차 M0 (%)": rnd(B3.signed_pct_M0[lg].mean()),
               "나머지 평균 부호 오차 M1 (%)": rnd(B3.signed_pct[~lg].mean()),
               "나머지 평균 부호 오차 M0 (%)": rnd(B3.signed_pct_M0[~lg].mean())}

    # M1 ÷ M0 예측 비와 초기 용량 이동 (Qcc_init 효과)
    def m1m0(d: pd.DataFrame) -> dict:
        r = (d.pred / d.pred_M0 - 1) * 100
        q = (d.Qcc_init - q_med) * 1000
        return {"M1÷M0−1 중앙값 (%)": rnd(r.median()), "Pearson r(초기 용량 차이, M1÷M0)": rnd(stats.pearsonr(q, r).statistic, 2)}
    qcc = {"B1 CV-OOF": m1m0(oof.assign(pred=oof.pred)),
           "B2 fastcharge": m1m0(B2[B2.group == "fastcharge"]), "B2 newstructure": m1m0(B2[B2.group == "newstructure"]),
           "B3": m1m0(B3)}
    pre = {"B2 fastcharge": test["selected"]["qcc_shift_decomposition"]["groups"]["fastcharge"],
           "B2 newstructure": test["selected"]["qcc_shift_decomposition"]["groups"]["newstructure"],
           "B3": b3["selected"]["qcc_shift_decomposition"]["groups"]["newstructure"]}
    for g, v in pre.items():
        qcc[g]["사전 이동량 (mAh, 라벨 없음)"] = rnd(v["shift_mAh"])
        qcc[g]["사전 예상 배율"] = rnd(v["pred_factor"], 3)

    # 예측구간 포함률 (보고된 값)
    cov = {"B2": test["selected"]["pi_cv_resid_10_90"], "B3": b3["selected"]["pi_cv_resid_10_90"]}

    post = posthoc(S, pre)

    bc2 = test["selected"]["breakin_check"]["overall"]
    det = pd.read_csv(RES / "model_performance_test_detail.csv")
    fc_row = det[(det.batch == "batch2") & (det.model == SEL) & (det.group == "fastcharge")].iloc[0]

    cand = S["cand"].set_index("model")
    b2_range = (float(cand.loc[["M0", "M1", "M2", "M3", "M4_RF", "M4_GBR"], "Test MAPE (부록)"].min()),
                float(cand.loc[["M0", "M1", "M2", "M3", "M4_RF", "M4_GBR"], "Test MAPE (부록)"].max()))

    return dict(rng=rng, q_med=q_med, removed=sorted(removed), k=k, c=c, ratio=ratio, window=window, slopes=slopes,
                same_policy=same_policy, gs2=gs2, gs3=gs3, vs=vs, scr=scr, h3=h3, b3_long=b3_long, qcc=qcc, cov=cov,
                bc2=bc2, fc_breakin=dict(rho=float(fc_row.h1_rho_cc_breakin), p=float(fc_row.h1_p_cc_breakin),
                                        rho_peak=float(fc_row.h1_rho_cc_breakin_peak_cycle),
                                        p_peak=float(fc_row.h1_p_cc_breakin_peak_cycle)),
                b2_cand_range=b2_range, post=post,
                top2=worst_cells(B2, rng, q_med, removed), top3=worst_cells(B3, rng, q_med, removed),
                tvr2=top_vs_rest(B2, q_med), tvr3=top_vs_rest(B3, q_med),
                excl=S["b3x"]["selected"], excl_m0=S["b3x"]["M0"], cell43=test["cell_43_check"][0])


# ───────────────────────────── 그림 ─────────────────────────────
LIFE_TICKS = [400, 500, 600, 800, 1000, 1500, 2000]


def log_life_axis(ax, axis="both", lim=(350, 2150)):
    fmt = FuncFormatter(lambda v, _: f"{int(v):,}")
    for a in ("x", "y"):
        if axis in (a, "both"):
            getattr(ax, f"set_{a}scale")("log")
            getattr(ax, f"set_{a}lim")(*lim)
            ax_ = getattr(ax, f"{a}axis")
            ax_.set_major_locator(FixedLocator(LIFE_TICKS))
            ax_.set_major_formatter(fmt)
            ax_.set_minor_locator(NullLocator())


def fig_pred_vs_true(S: dict, A: dict) -> Path:
    oof, V, B2, B3 = S["oof"], S["V"], S["B2"], S["B3"]
    rep = reported(S["perf"][SEL])
    fig, axs = plt.subplots(2, 2, figsize=(11.6, 12.0))
    lim = (350, 2150)
    xs = np.geomspace(*lim, 50)
    panels = [
        (axs[0, 0], "① Train — Batch 1 CV (out-of-fold)", f"MAPE {rep['Train (Batch 1 CV)']:.2f}% (20회 반복 평균)",
         [(oof, C_B1, "o", "학습 27셀 (셀별 20회 예측 평균)", True)]),
        (axs[0, 1], "② Valid — Batch 1 Hold-out", f"MAPE {rep['Valid (Batch 1 Hold-out)']:.2f}% (9셀·5정책)",
         [(V, C_B1, "s", "Hold-out 9셀", True)]),
        (axs[1, 0], "③ Test — Batch 2", f"MAPE {rep['Test (Batch 2)']:.2f}% · 과대예측 {(B2.signed_pct > 0).sum()}/{len(B2)}",
         [(B2[B2.group == "fastcharge"], C_FC, "o", f"fastcharge {int((B2.group == 'fastcharge').sum())}셀", True),
          (B2[B2.group == "newstructure"], C_NS, "s", f"newstructure {int((B2.group == 'newstructure').sum())}셀", True)]),
        (axs[1, 1], "④ 추가 검증 — Batch 3", f"MAPE {rep['Test (Batch 3)']:.2f}% · 과대예측 {(B3.signed_pct > 0).sum()}/{len(B3)}",
         [(B3[~B3.index.isin(A["removed"])], C_B3, "o", f"newstructure {len(B3) - len(A['removed'])}셀", True),
          (B3[B3.index.isin(A["removed"])], C_B3, "o", f"원저자 코드 제거 {len(A['removed'])}셀 (속 빈 점)", False)]),
    ]
    for ax, title, sub, series in panels:
        ax.fill_between(xs, xs * (1 - BAND), xs * (1 + BAND), color="#E5E7EB", alpha=0.75, lw=0, zorder=0,
                        label="±20% 띠")
        ax.plot(xs, xs, color=MUTED, lw=1.4, ls="--", zorder=1, label="y = x (완벽한 예측)")
        ax.axvspan(B1_LABEL_MAX, lim[1], color="#FEF3C7", alpha=0.45, lw=0, zorder=0)
        for d, col, mk, lab, filled in series:
            ax.scatter(d.true, d.pred, s=70, marker=mk, facecolor=col if filled else "white", edgecolor=col if not filled else "white",
                       linewidth=1.6 if not filled else 0.9, alpha=0.95, zorder=3, label=lab)
        log_life_axis(ax, lim=lim)
        ax.set_title(f"{title}\n{sub}", loc="left", fontsize=14)
        ax.set_xlabel("실제 수명 (사이클)")
        ax.set_ylabel("예측 수명 (사이클, M1)")
        ax.set_aspect("equal")
        ax.legend(loc="upper left", fontsize=11.5, handletextpad=0.3, borderaxespad=0.3)
        ax.text(lim[1] * 0.97, lim[0] * 1.06, "학습 라벨 최댓값\n1,074 초과 구간", ha="right", va="bottom",
                fontsize=11.5, color="#92400E")
    # ③ 주석: 대부분 y=x 위쪽
    axs[1, 0].annotate("fastcharge 셀이\ny=x 위로 몰림\n= 수명을 길게 예측", xy=(450, 610), xytext=(365, 760),
                       fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    axs[1, 1].annotate("1,600 이상 장수명 셀은\ny=x 아래\n= 수명을 짧게 예측", xy=(1800, 1220), xytext=(1150, 520),
                       fontsize=12, color=INK, ha="center", arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    fig.suptitle("예측 vs 실제 수명 (선택 모델 M1, 로그 축) — Batch 1 안에서는 맞고, Batch 2에서는 한쪽으로 치우친다",
                 x=0.02, ha="left", fontsize=15.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0, 1, 0.965), h_pad=2.2, w_pad=1.0)
    return ps.save(fig, "day2_pred_vs_true")


def fig_b2_error(S: dict, A: dict) -> Path:
    F, B2, oof = S["F"], S["B2"], S["oof"]
    L1 = F[(F.batch == "batch1") & F.labeled]
    k, c = A["k"], A["c"]
    rng = A["rng"]
    r_fc = A["ratio"]["b2_fastcharge"]["median_all"]
    r_ns = A["ratio"]["b2_newstructure"]["median_all"]
    fc, ns = B2[B2.group == "fastcharge"], B2[B2.group == "newstructure"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.2, 7.4))
    xs = np.linspace(-4.6, -3.0, 100)
    # (a) 수명 vs ΔQ 깊이 + B1 관계선
    for ax in (a1, a2):
        ax.axvspan(rng[0], rng[1], color="#DBEAFE", alpha=0.45, lw=0, zorder=0)
        ax.set_xlim(-4.6, -3.0)
    a1.plot(xs, 10 ** (k * xs + c), color=C_B1, lw=2.2, zorder=2, label="Batch 1 관계선 (EDA Q5 = M0)")
    a1.plot(xs, r_fc * 10 ** (k * xs + c), color=C_FC, lw=2, ls="--", zorder=2,
            label=f"Batch 1 선 × {r_fc:.2f} (fastcharge 수명비)")
    a1.scatter(L1.dQ_logvar, L1.cycle_life, s=55, color=C_B1, edgecolor="white", lw=0.8, zorder=3, label="Batch 1 학습 36셀")
    a1.scatter(fc.dQ_logvar, fc.true, s=70, color=C_FC, edgecolor="white", lw=0.8, zorder=4, label="Batch 2 fastcharge 30셀")
    a1.scatter(ns.dQ_logvar, ns.true, s=70, marker="s", color=C_NS, edgecolor="white", lw=0.8, zorder=4,
               label="Batch 2 newstructure 9셀")
    a1.set_yscale("log")
    a1.set_ylim(350, 1500)
    a1.yaxis.set_major_locator(FixedLocator([400, 500, 600, 800, 1000, 1200, 1500]))
    a1.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{int(v):,}"))
    a1.yaxis.set_minor_locator(NullLocator())
    a1.set_xlabel("ΔQ 깊이 (dQ_logvar) →  클수록 ΔQ 곡선이 깊음")
    a1.set_ylabel("실제 수명 (사이클, 로그 축)")
    a1.set_title("(a) 같은 ΔQ 깊이에서 Batch 2 fastcharge 수명이 더 짧다", loc="left")
    a1.text(rng[0] + 0.02, 1430, "학습 ΔQ 범위", fontsize=12, color="#1E40AF", va="top")
    a1.annotate(f"수명비 {r_fc:.2f}\n(같은 ΔQ인데 약 {round((1 - r_fc) * 100)}% 짧음)", xy=(-3.55, r_fc * 10 ** (k * -3.55 + c)),
                xytext=(-4.55, 430), fontsize=12, color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    a1.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=2, fontsize=11.5, handletextpad=0.3, columnspacing=1.2)

    # (b) M1 예측 ÷ 실제 vs ΔQ 깊이
    a2.axhspan(1 - BAND, 1 + BAND, color="#E5E7EB", alpha=0.75, lw=0, zorder=0)
    a2.axhline(1, color=MUTED, lw=1.4, ls="--", zorder=1)
    a2.scatter(oof.dQ_logvar, oof.pred / oof.true, s=40, color="#93C5FD", edgecolor="white", lw=0.6, zorder=2,
               label="Batch 1 CV-OOF 27셀 (참고)")
    a2.scatter(fc.dQ_logvar, fc.pred / fc.true, s=70, color=C_FC, edgecolor="white", lw=0.8, zorder=4,
               label="Batch 2 fastcharge")
    a2.scatter(ns.dQ_logvar, ns.pred / ns.true, s=70, marker="s", color=C_NS, edgecolor="white", lw=0.8, zorder=4,
               label="Batch 2 newstructure")
    fin, fout = fc[~fc.out_of_train_range], fc[fc.out_of_train_range]
    for d, txt, xyt, ha in ((fin, "fastcharge 범위 안", (-4.02, 1.53), "center"), (fout, "fastcharge 범위 밖", (-3.02, 1.53), "right")):
        m = float((d.pred / d.true).median())
        x0, x1 = float(d.dQ_logvar.min()), float(d.dQ_logvar.max())
        a2.plot([x0, x1], [m, m], color=C_FC, lw=3.5, alpha=0.6, zorder=3, solid_capstyle="round")
        a2.annotate(f"{txt}\n중앙값 ×{m:.2f}", xy=((x0 + x1) / 2, m), xytext=xyt, ha=ha, va="center", fontsize=11.5,
                    color=INK, arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1))
    a2.set_yscale("log")
    a2.set_ylim(0.62, 1.62)
    a2.yaxis.set_major_locator(FixedLocator([0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6]))
    a2.yaxis.set_major_formatter(FuncFormatter(lambda v, _: f"×{v:.1f}"))
    a2.yaxis.set_minor_locator(NullLocator())
    a2.set_xlabel("ΔQ 깊이 (dQ_logvar) →  클수록 ΔQ 곡선이 깊음")
    a2.set_ylabel("M1 예측 ÷ 실제 (로그 축, ×1.0 = 정확)")
    a2.set_title("(b) fastcharge 기준: 학습 범위 안 셀이 더 크게 틀림 → 외삽보다 수준 이동", loc="left")
    a2.text(-4.58, 1.21, "±20% 띠", fontsize=11.5, color=MUTED, va="bottom")
    a2.legend(loc="upper center", bbox_to_anchor=(0.5, -0.14), ncol=3, fontsize=11.5, handletextpad=0.3, columnspacing=1.0)
    a2.text(-3.02, 0.645, f"B2 fastcharge 안의 ΔQ–수명 기울기 {A['slopes']['batch2_fastcharge']['slope']:.2f}\n"
                          f"(Batch 1 {A['slopes']['batch1']['slope']:.2f}) → 수명이 거의 평평",
            ha="right", va="bottom", fontsize=11.5, color=INK)
    fig.suptitle("Batch 2 오차 분해 — 같은 ΔQ 깊이에서도 Batch 2 fastcharge 수명이 Batch 1보다 짧다",
                 x=0.01, ha="left", fontsize=15.5, fontweight="bold")
    fig.subplots_adjust(left=0.07, right=0.985, top=0.88, bottom=0.25, wspace=0.24)
    return ps.save(fig, "day2_b2_error")


def fig_m1_vs_m0(S: dict, A: dict) -> Path:
    oof, V, B2, B3 = S["oof"], S["V"], S["B2"], S["B3"]
    q_med = A["q_med"]
    groups = [("Batch 1 CV\n\n(27셀)", oof, C_B1), ("Valid\n\n(9셀)", V, C_B1),
              ("Batch 2\nfastcharge\n(30셀)", B2[B2.group == "fastcharge"], C_FC),
              ("Batch 2\nnewstructure\n(9셀)", B2[B2.group == "newstructure"], C_NS),
              ("Batch 3\n\n(44셀)", B3, C_B3)]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(15.6, 7.0), gridspec_kw={"width_ratios": [1.2, 1], "wspace": 0.2})
    rs = np.random.default_rng(42)
    a1.axhline(0, color=MUTED, lw=1.3, zorder=1)
    for i, (lab, d, col) in enumerate(groups):
        for off, colname, mcol, filled in ((-0.19, "signed_pct_M0", C_M0, False), (0.19, "signed_pct", C_M1, True)):
            y = d[colname].values
            a1.scatter(i + off + rs.uniform(-0.07, 0.07, len(y)), y, s=16, color=col, alpha=0.35, lw=0, zorder=2)
            m = float(np.mean(y))
            a1.scatter([i + off], [m], s=150, marker="D", facecolor=mcol if filled else "white", edgecolor=mcol if not filled else "white",
                       lw=2.2 if not filled else 1.0, zorder=4)
            a1.text(i + off + (0.11 if off > 0 else -0.11), m, f"{m:+.1f}", ha="left" if off > 0 else "right", va="center",
                    fontsize=12, color=INK, fontweight="bold" if filled else "normal", zorder=5,
                    bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))
    a1.set_xticks(range(len(groups)))
    a1.set_xticklabels([g[0] for g in groups], fontsize=12)
    a1.set_xlim(-0.6, len(groups) - 0.4)
    a1.set_ylim(-50, 70)
    a1.set_ylabel("부호 있는 오차 (%)   + = 수명을 길게 예측")
    a1.set_title("(a) 평균 부호 오차: M0(ΔQ만) → M1(+초기 용량)", loc="left")
    a1.legend(handles=[Line2D([], [], marker="D", ls="", markersize=10, markerfacecolor="white", markeredgecolor=C_M0, mew=2.2, label="M0 평균 (ΔQ 깊이만)"),
                       Line2D([], [], marker="D", ls="", markersize=10, color=C_M1, label="M1 평균 (ΔQ 깊이 + 초기 용량)"),
                       Line2D([], [], marker="o", ls="", markersize=6, color=MUTED, alpha=0.5, label="개별 셀 (왼쪽 M0, 오른쪽 M1)")],
              loc="upper left", fontsize=11.2)
    a1.annotate("Batch 2: 초기 용량이\n과대예측을 일부 상쇄", xy=(2.19, 21.4), xytext=(3.1, 58), fontsize=11.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))
    a1.annotate("Batch 3: 편향 없던 예측을\n과소예측으로 밀어냄", xy=(4.19, -7.1), xytext=(3.55, -40), fontsize=11.5, ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.2))

    # (b) M1 ÷ M0 예측 비 vs 초기 용량 차이
    a2.axhline(0, color=MUTED, lw=1.3)
    a2.axvline(0, color=MUTED, lw=1.0, ls=":")
    pts = [("Batch 1 (CV-OOF·Valid)", pd.concat([oof[["Qcc_init", "pred", "pred_M0"]], V[["Qcc_init", "pred", "pred_M0"]]]), C_B1, "o"),
           ("Batch 2 fastcharge", B2[B2.group == "fastcharge"], C_FC, "o"),
           ("Batch 2 newstructure", B2[B2.group == "newstructure"], C_NS, "s"),
           ("Batch 3", B3, C_B3, "o")]
    for lab, d, col, mk in pts:
        a2.scatter((d.Qcc_init - q_med) * 1000, (d.pred / d.pred_M0 - 1) * 100, s=55, marker=mk, color=col,
                   edgecolor="white", lw=0.7, alpha=0.9, zorder=3, label=lab)
    for g, col, dy in (("B2 fastcharge", C_FC, 1), ("B2 newstructure", C_NS, 1), ("B3", C_B3, 1)):
        v = A["qcc"][g]
        x, y = v["사전 이동량 (mAh, 라벨 없음)"], (v["사전 예상 배율"] - 1) * 100
        a2.scatter([x], [y], s=230, marker="X", color=col, edgecolor=INK, lw=1.3, zorder=5)
    a2.text(21.0, -13.0, "X = 테스트 전 라벨 없이 계산한\n그룹 평균 이동량과 예상 배율\n주황 ×0.96 · 보라 ×0.94 · 초록 ×0.90", fontsize=11.5,
            color=INK, va="top", ha="right")
    a2.set_xlabel("초기 용량 − Batch 1 중앙값 (mAh)")
    a2.set_ylabel("M1 예측이 M0보다 높은 정도 (%)")
    a2.set_title("(b) 모델 계산 확인: 초기 용량이 낮을수록 M1이 M0보다 낮게 예측", loc="left")
    a2.set_xlim(-42, 22)
    a2.set_ylim(-26, 14)
    a2.legend(loc="upper left", fontsize=11.2, handletextpad=0.3)
    fig.suptitle("M1 vs M0 — 초기 용량을 더하면 테스트 배치 예측이 함께 내려간다 (노션 Qdlin 경고와 같은 방향, 원인 미확인)",
                 x=0.01, ha="left", fontsize=15.5, fontweight="bold")
    fig.subplots_adjust(left=0.065, right=0.985, top=0.87, bottom=0.14)
    return ps.save(fig, "day2_m1_vs_m0")


# ───────────────────────────── 서술 (수치는 모두 계산값) ─────────────────────────────
def f1(x) -> str:
    return f"{x:.1f}"


def narrative(S: dict, A: dict, chk: dict) -> dict:
    r1, r0 = chk[SEL]["reported"], chk[BASE]["reported"]
    g2, g3, vs = A["gs2"], A["gs3"], A["vs"]
    fc, ns = g2["fastcharge"], g2["newstructure"]
    sp = A["same_policy"]
    p48 = sp.get("4.8C(80%)-4.8C")
    r_fc = A["ratio"]["b2_fastcharge"]["median_all"]
    r_b3 = A["ratio"]["batch3"]["median_all"]
    tv2, tv3 = A["tvr2"]["상위 10셀"], A["tvr3"]["상위 10셀"]
    top2, top3 = A["top2"], A["top3"]
    qcc = A["qcc"]
    excl_mape = A["excl"]["mape"]
    PO = paper_split_overlap()
    cov2 = A["cov"]["B2"]
    scr = A["scr"]

    # 사전 가정 확인 — 서술이 데이터와 어긋나면 멈춘다
    assert tv2["과대예측 셀"] == 10 and tv2["fastcharge 셀"] >= 7
    assert tv3["과대예측 셀"] <= 3
    assert r1["Gap (Train-Valid)"] < 0 and r1["Gap (Valid-Test)"] > 0 and r1["Gap (Batch2-Batch3)"] < 0
    assert g3["전체"]["평균 부호 오차 (%)"] < 0 < g3["전체"]["M0 평균 부호 오차 (%)"]
    assert scr["오경보(경보인데 실제 ≥550)"] == 0

    top2_fc = [r for r in top2 if r["그룹"] == "fastcharge"]
    top2_ns = [r for r in top2 if r["그룹"] == "newstructure"]
    top2_late = [r for r in top2 if r["break-in 정점 사이클 (CC 끝점, 주)"] >= 40]
    top3_under = [r for r in top3 if r["방향"] == "과소예측"]
    top3_long = [r for r in top3_under if r["실제 수명 vs B1 라벨 범위"].startswith("위")]
    top3_long1600 = [r for r in top3_long if r["실제 수명"] >= 1600]
    top3_long_dq_out = [r for r in top3_long if r["ΔQ 학습 범위"] == "밖"]
    top3_dq_in = [r for r in top3_under if r["ΔQ 학습 범위"] == "안"]
    top3_dq_in_qcc_max = max(r["초기 용량 − B1 중앙값 (mAh)"] for r in top3_dq_in)
    top3_dq_in_gap = [r["부호 오차 (%)"] - r["M0 부호 오차 (%)"] for r in top3_dq_in]
    top3_removed = [r for r in top3 if r["원저자 코드 제거 셀"]]
    top3_removed_rank = [i + 1 for i, r in enumerate(top3) if r["원저자 코드 제거 셀"]]
    fc_med_peak = float(S["B2"][S["B2"].group == "fastcharge"].cc_breakin_peak_cycle.median())
    nsd = S["B2"][S["B2"].group == "newstructure"]
    ns_big = nsd[nsd.signed_pct > 20]
    assert ns_big.true.max() < 850 and (nsd.true < 850).sum() == len(ns_big) + 1   # 크게 틀린 셀 = 그룹의 짧은 쪽
    Vd = S["V"].sort_values("ape", ascending=False)
    t_med = S["F"][S["F"].labeled].groupby("batch").Tavg_mean.median()
    assert top3_dq_in_qcc_max <= -10 and all(g < 0 for g in top3_dq_in_gap)

    worst = {
        "batch2": {
            "공통점": [
                f"상위 10셀 모두 과대예측이다. 모델이 수명을 실제보다 길게 봤다.",
                f"10셀 중 {len(top2_fc)}셀이 fastcharge 단수명 셀이다(실제 {min(r['실제 수명'] for r in top2_fc)}~{max(r['실제 수명'] for r in top2_fc)} 사이클).",
                f"나머지 {len(top2_ns)}셀은 newstructure 중 수명이 짧은 쪽이고, ΔQ 깊이가 학습 범위보다 얕은(장수명 쪽) 외삽 구간이다.",
                f"10셀 중 {tv2['ΔQ 학습 범위 안 셀']}셀은 ΔQ 깊이가 학습 범위 안이다. 그래서 외삽만으로는 설명되지 않고, 같은 ΔQ 깊이에서 수명이 짧은 '수준 이동'(같은 ΔQ 깊이에서 수명이 전체적으로 일정 비율 짧아지는 것)이 함께 있다.",
                f"{'위와 '.join(str(i + 1) for i, r in enumerate(top2) if r in top2_late)}위 셀은 초기 용량 상승(break-in)이 크고 늦게 끝났다(정점 사이클 {', '.join(str(r['break-in 정점 사이클 (CC 끝점, 주)']) for r in top2_late)}; fastcharge 중앙값 {fc_med_peak:g}). 다만 10셀 전체의 공통점은 아니다.",
                f"상위 10셀은 나머지보다 초기 용량이 Batch 1에 가깝다(차이 중앙값 {tv2['초기 용량 차이 중앙값 (mAh)']:+.1f} vs {A['tvr2']['나머지']['초기 용량 차이 중앙값 (mAh)']:+.1f} mAh). 그래서 M1의 하향 효과를 덜 받아 과대예측이 그대로 남았다.",
            ],
        },
        "batch3": {
            "공통점": [
                f"상위 10셀 중 {len(top3_under)}셀이 과소예측이다. Batch 2와 방향이 반대다.",
                f"과소예측 {len(top3_under)}셀 중 {len(top3_long)}셀은 실제 수명이 학습 라벨 최댓값(1,074)보다 길다. 그중 {len(top3_long1600)}셀은 1,600 사이클 이상이다.",
                f"이 장수명 {len(top3_long)}셀 중 {len(top3_long_dq_out)}셀은 ΔQ 깊이도 학습 범위보다 얕다. 피처와 라벨 모두 처음 보는 구간이라 외삽 오차가 크다.",
                f"Batch 3 전체에서 1,074 사이클을 넘는 {A['b3_long']['n']}셀의 MAPE는 {A['b3_long']['MAPE']}%로, 나머지 셀({A['b3_long']['나머지 MAPE']}%)의 약 {A['b3_long']['MAPE'] / A['b3_long']['나머지 MAPE']:.1f}배다.",
                f"ΔQ 깊이가 학습 범위 안인 {len(top3_dq_in)}셀도 과소예측됐다. 모두 초기 용량이 Batch 1보다 {np.floor(abs(top3_dq_in_qcc_max)):.0f} mAh 이상 낮아, M1이 M0보다 더 짧게 예측했다.",
                f"10셀 중 {len(top3_removed)}셀({'·'.join(f'{k}위' for k in top3_removed_rank)})은 원저자 공개 코드가 데이터 품질 문제로 뺀 셀이다. 1위 {top3[0]['셀']}은 원논문 2차 테스트셋에도 들어 있다.",
            ],
        },
    }

    def pct(x):
        return f"{x:+.1f}%"

    # Batch 2 기록 공백 (src/check_rest_gap.py, 라벨 없이 원본 .mat 시간 기록으로 확인)
    rg = (S["rest_gap"] or {}).get("라벨 셀 요약", {})
    rg2 = rg.get("batch2")
    if rg2:
        rgfc = rg2["by_group"]["fastcharge"]
        gap_txt = (f"Batch 2 라벨 {rg2['n_labeled']}셀 중 {rg2['n_single_long_gap_in_window']}셀에 ΔQ 창(cycle 10→100) 안의 "
                   f"약 {rg2['gap_hours_range'][0]:.1f}~{rg2['gap_hours_range'][1]:.1f}시간 기록 공백이 있다(평소 한 사이클 약 {rg2['median_cycle_hours']:.1f}시간, 테스트 뒤 원본 시간 기록에서 찾음). "
                   f"공백 뒤 용량은 중앙값 {rg2['qd_change_mAh_median']:+.1f} mAh 바뀌었다({rg2['n_qd_change_le_-3mAh']}/{rg2['n_labeled']}셀이 −3 mAh 이하). "
                   f"Batch 1·3 라벨 셀에는 이런 공백이 없다(src/check_rest_gap.py).")
        gap_peak_txt = (f"fastcharge {rgfc['n']}셀 중 {rgfc['n_qd_peak_after_gap']}셀은 요약 QD 정점이 공백 뒤에 있다(CC 끝점 정점은 {rgfc['n_cc_peak_after_gap']}셀). "
                        "DAY 1 Q2의 'Batch 2 break-in이 늦다'는 관찰의 일부는 공백 뒤 회복일 수 있다.")
        gap_rho = S["rest_gap"]["사후_오차와의_관계"]["batch2 전체"]
    else:
        gap_txt = gap_peak_txt = None
        gap_rho = None

    post = A["post"]
    hyp_note = ("H1·H2의 기준값(수명비 0.76 → 약 1.3배 등)은 DAY 1 EDA에서 Batch 2 라벨로 계산했다(노션 DAY 1 요구). "
                "그래서 '지지'는 눈을 가린 사전 예측의 적중이 아니라, 테스트 결과가 DAY 1 기록과 일관된다는 확인이다.")
    hyp = [
        {"id": "H1", "가설": "Gap(Valid−Test)는 크게 (+)이고, Batch 2 fastcharge를 약 1.3배 과대예측할 것이다. 원인 후보(상관): break-in 정점(cycle 81 vs 37)·셀 로트·충전 구조.",
         "verdict": "supported", "판정": "지지 (일관성 확인). 원인 후보 break-in은 미확인",
         "근거": [f"Gap(Valid−Test)는 +{r1['Gap (Valid-Test)']:.2f}%p다(M0 +{r0['Gap (Valid-Test)']:.2f}).",
                f"M0 기준 fastcharge 예측÷실제 중앙값은 ×{fc['M0 예측÷실제 중앙값']:.2f}로, DAY 1 기록(약 1.3배)과 맞다.",
                f"fastcharge {fc['n']}셀 중 {fc['과대예측'].split('/')[0]}셀을 과대예측했다(M1).",
                f"break-in: 사전에 정한 39셀 전체 점검에서 잔차와의 순위상관은 정점 사이클 CC 끝점(주) {A['bc2']['rho_cc_breakin_peak_cycle']:+.2f}(p {A['bc2']['p_cc_breakin_peak_cycle']:.2f}), 요약 용량(보조, DAY 1의 81 vs 37 정의를 cycle 2~100으로 자름) {A['bc2']['rho_qd_breakin_peak_cycle']:+.2f}(p {A['bc2']['p_qd_breakin_peak_cycle']:.3f}), 상승폭 {A['bc2']['rho_cc_breakin']:+.2f}(p {A['bc2']['p_cc_breakin']:.2f})다. fastcharge 안에서 상승폭이 클수록 더 과대예측한 것(ρ {A['fc_breakin']['rho']:+.2f})은 사전 지정 밖의 탐색 결과다.",
                f"사전 점검 셀(Batch 2 #43)의 오차는 {A['cell43']['ape']:.1f}%로 특이 셀이 아니다.",
                "Batch 2 기록 공백(원인 후보)은 테스트 뒤 원본 시간 기록에서 찾았으므로 이 판정의 근거에 넣지 않았다.",
                hyp_note],
         "통계 상세": {"spearman_전체_CC정점": [rnd(A['bc2']['rho_cc_breakin_peak_cycle'], 2), rnd(A['bc2']['p_cc_breakin_peak_cycle'], 3)],
                   "spearman_전체_상승폭": [rnd(A['bc2']['rho_cc_breakin'], 2), rnd(A['bc2']['p_cc_breakin'], 3)],
                   "spearman_전체_QD정점(보조)": [rnd(A['bc2']['rho_qd_breakin_peak_cycle'], 2), rnd(A['bc2']['p_qd_breakin_peak_cycle'], 3)],
                   "spearman_fastcharge_상승폭": [rnd(A['fc_breakin']['rho'], 2), rnd(A['fc_breakin']['p'], 3)],
                   "spearman_fastcharge_CC정점": [rnd(A['fc_breakin']['rho_peak'], 2), rnd(A['fc_breakin']['p_peak'], 3)],
                   "형식": "[ρ, p]"}},
        {"id": "H2", "가설": "B2 newstructure는 편향이 작고 산포가 클 것이다(수명비 0.94, 6/9가 외삽 구간).",
         "verdict": "partial", "판정": "부분 지지",
         "근거": [f"중앙값 기준 편향은 작다(예측÷실제 ×{ns['예측÷실제 중앙값']:.2f}).",
                f"평균 부호 오차는 {pct(ns['평균 부호 오차 (%)'])}라 작지 않다. 20% 넘게 과대예측된 {len(ns_big)}셀이 평균을 끌어올렸고, 모두 이 그룹에서 수명이 짧은 쪽(850 사이클 미만)이다.",
                f"산포는 예상대로 크다(부호 오차 SD {ns['부호 오차 SD (%p)']}%p, fastcharge {fc['부호 오차 SD (%p)']}%p).",
                f"학습 ΔQ 범위 밖 셀이 더 크게 틀렸다(밖 {ns['범위 밖 MAPE']}%, 안 {ns['범위 안 MAPE']}%)."],
         "통계 상세": {"n_범위안_밖": ns["범위 안/밖 n"], "spearman_로그잔차_vs_실제수명": ns["Spearman(로그 잔차, 실제 수명)"]}},
        {"id": "H3", "가설": "B2 MAPE는 짧은 셀(392~514)이 지배할 것이다. 기준은 B1 #20·#21 out-of-fold 오차다.",
         "verdict": "partial", "판정": "부분 지지 (셀 수 비중과 비슷)",
         "근거": [f"550 사이클 미만 {A['h3']['n_short']}셀(Batch 2의 {A['h3']['n_short'] / len(S['B2']) * 100:.0f}%)이 전체 APE 합의 {A['h3']['APE 합 비율']}%를 차지한다. 셀 수 비중과 비슷해 '지배'라고 보기는 약하다.",
                f"짧은 셀 MAPE {A['h3']['짧은 셀 MAPE']}%, 나머지 {A['h3']['나머지 MAPE']}%다. 셀당 오차는 조금 더 크다.",
                "Batch 1의 짧은 두 셀(#20·#21)은 CV에서 잘 맞았다(+1.5%·+5.4%). 짧은 수명만으로는 설명되지 않을 수 있지만, 2셀뿐이라 근거는 약하다.",
                "Batch 2에서는 550 미만 셀이 모두 fastcharge라 '짧음'과 'Batch 2 fastcharge'를 가를 수 없다."]},
        {"id": "H4", "가설": "Gap(Train−Valid)는 n=9 CI와 비교하고, CI를 넘는 (+)면 과적합으로 본다.",
         "verdict": "supported", "판정": "지지 (과적합 신호 없음)",
         "근거": [f"Gap(Train−Valid)는 {r1['Gap (Train-Valid)']:+.2f}%p로 (−)다.",
                f"Train CV {r1['Train (Batch 1 CV)']:.2f}%가 Valid 95% CI [3.47, 9.91] 안에 있다.",
                "Valid는 9셀이라 검정력이 낮다. '큰 과적합은 없다' 수준의 근거다."]},
        {"id": "H5", "가설": "Gap(Target−Test)는 (+)일 것이다. B2로 튜닝해 줄이지 않는다.",
         "verdict": "supported", "판정": "지지",
         "근거": [f"Gap(Target−Test)는 +{r1['Gap (Target-Test)']:.2f}%p다(M0 +{r0['Gap (Target-Test)']:.2f}).",
                f"후보 6개의 Batch 2 MAPE(사후 부록 값)가 모두 {A['b2_cand_range'][0]:.1f}~{A['b2_cand_range'][1]:.1f}%라 9.1%를 넘는다. 모델 선택만의 문제가 아니다.",
                "Batch 2 라벨로 재보정하지 않았다(누수 방지)."]},
        {"id": "H6", "가설": "Gap(Batch2−Batch3)는 (−)일 것이다. M1이면 초기 용량 이동(−14.5 mAh)이 B3 예측을 ×0.90 쪽으로 낮출 수 있다.",
         "verdict": "supported", "판정": "지지",
         "근거": [f"Gap(Batch2−Batch3)는 {r1['Gap (Batch2-Batch3)']:+.2f}%p다(M0 {r0['Gap (Batch2-Batch3)']:+.2f}).",
                f"B3 평균 부호 오차는 M1 {pct(g3['전체']['평균 부호 오차 (%)'])}, M0 {pct(g3['전체']['M0 평균 부호 오차 (%)'])}다.",
                f"M1 예측은 M0보다 중앙값 {abs(qcc['B3']['M1÷M0−1 중앙값 (%)'])}% 낮다(×{post['B3']['M1÷M0 예측 비 중앙값']:.2f}). 테스트 전에 라벨 없이 계산한 예상 배율(×{qcc['B3']['사전 예상 배율']:.2f})과 방향·크기가 비슷하다.",
                f"초기 용량은 Batch 3를 짧게 예측하도록 밀었지만, MAPE({r1['Test (Batch 3)']:.2f} vs {r0['Test (Batch 3)']:.2f})와 순위상관({post['B3']['수명 순위 Spearman M1 / M0'][0]:.2f} vs {post['B3']['수명 순위 Spearman M1 / M0'][1]:.2f})은 M0보다 나았다.",
                f"사후 확인: 라벨 없이 잰 이동을 되돌리면 M1의 Batch 3 평균 부호 오차는 {pct(post['B3']['초기 용량 이동 되돌린 M1 평균 부호 오차 (%)'])}다(라벨 사용, 보고 성능 아님)."]},
    ]

    dec = post["B2 M1 이득 분해 (%p)"]
    r_ns = A["ratio"]["b2_newstructure"]["median_all"]
    gaps = [
        {"Gap": "Gap (Train-Valid)", "M1": r1["Gap (Train-Valid)"], "M0": r0["Gap (Train-Valid)"],
         "한 줄 해석": "Batch 1 안에서는 처음 보는 충전 정책 셀도 CV만큼 맞혔다. 과적합 신호는 없다.",
         "해석": [f"Hold-out 오차({r1['Valid (Batch 1 Hold-out)']:.2f}%)가 CV 평균({r1['Train (Batch 1 CV)']:.2f}%)보다 조금 작다.",
                f"Valid가 오히려 좋은 이유: Hold-out은 설계 단계에서 수명순 양 끝 정책을 넣지 않았다(DAY 1 §11). Valid 수명은 {Vd.true.min():,.0f}~{Vd.true.max():,.0f}이고, 학습 라벨의 양 끝({B1_LABEL_MIN:,.0f}·{B1_LABEL_MAX:,.0f})은 Train에 있다.",
                "Hold-out은 9셀뿐이라 95% CI가 3.5~9.9%로 넓다. '큰 과적합은 없다' 정도만 말할 수 있다.",
                f"9셀 중 가장 큰 오차는 한 셀의 {Vd.signed_pct.iloc[0]:+.1f}%이고, 나머지 8셀은 모두 {np.ceil(Vd.ape.iloc[1]):.0f}% 이내다.",
                f"M0와 M1의 Valid는 반올림하면 같은 6.31이지만 우연이다(M0 {Vd.ape_M0.mean():.3f}, M1 {Vd.ape.mean():.3f}). 9셀로는 두 모델을 가를 수 없다.",
                "Train ≈ Valid이므로, 뒤의 큰 Gap은 과적합이 아니라 다른 원인에서 온다."]},
        {"Gap": "Gap (Valid-Test)", "M1": r1["Gap (Valid-Test)"], "M0": r0["Gap (Valid-Test)"],
         "한 줄 해석": "가장 큰 Gap이다. 모델 종류를 바꿔서 생긴 Gap은 아니고, 대부분이 Batch 2 셀 공통의 한 방향 치우침(길게 예측)이다.",
         "해석": [f"Batch 2 {g2['전체']['n']}셀 중 {g2['전체']['과대예측'].split('/')[0]}셀을 과대예측했다. 오차가 한 방향이라 잡음이 아니라 편향이다.",
                f"사후 분석: 예측을 셀 공통 배율 하나(×{post['B2']['공통 배율 (예측÷실제 중앙값)']:.2f})로 나누면 MAPE가 {post['B2']['M1 MAPE (보고)']:.2f} → {post['B2']['공통 배율 제거 후 M1 MAPE']:.2f}%로 준다. Gap(Valid−Test)의 약 {post['Gap(Valid-Test) 중 공통 배율의 몫'] * 100:.0f}%가 이 공통 치우침이다(라벨 사용, 보고 성능 아님).",
                f"같은 ΔQ 깊이에서 Batch 2 fastcharge 수명은 Batch 1 선의 {r_fc:.2f}배다(EDA Q5).",
                (f"충전 정책이 같아도 수명이 다르다. 4.8C(80%)-4.8C 셀의 수명 중앙값은 Batch 1 {p48['batch1']['수명 중앙값']:g}, Batch 2 {p48['batch2']['수명 중앙값']:g} 사이클이다."
                 if p48 else "충전 정책이 같아도 배치에 따라 수명이 다르다."),
                f"후보 6개의 Batch 2 MAPE(사후 부록 값)가 모두 {A['b2_cand_range'][0]:.1f}~{A['b2_cand_range'][1]:.1f}%다. 모델 종류를 바꿔서 생긴 Gap은 아니다.",
                *([gap_txt, "모든 Batch 2 셀에 공백이 있어 비교군이 없다. 이 공백이 치우침의 원인인지는 가릴 수 없다(셀 로트·시험 시기도 후보)."] if gap_txt else
                  ["치우침이 왜 생겼는지(셀 로트·시험 시기·초기 용량 상승)는 이 데이터로 확인하지 못했다."]),
                f"초기 용량을 더한 M1이 M0보다 Gap을 {dec['M0 → M1 전체']:.1f}%p 줄였다. 사후 분해로는 약 {dec['배치 수준 초기 용량 이동 (이동 되돌린 M1 → M1)']:.1f}%p가 Batch 2 초기 용량이 전체적으로 낮아 예측이 함께 내려간 효과이고, 약 {dec['배치 안 셀 차이 (M0 → 이동 되돌린 M1)']:.1f}%p는 배치 안 셀 차이를 더 잘 설명한 몫이다. 이 이동이 측정 차이인지 실제 용량 차이인지는 확인하지 못했다."]},
        {"Gap": "Gap (Target-Test)", "M1": r1["Gap (Target-Test)"], "M0": r0["Gap (Target-Test)"],
         "한 줄 해석": "원논문 9.1%에 미달한다. 대부분 위의 배치 이동에서 온 것으로 본다.",
         "해석": [f"(아래 원논문 사실은 테스트 뒤 원저자 공개 코드와 논문 Table 1에서 확인했다.) 원논문은 9.1%의 계산식을 적지 않았다. Table 1 Full 모델의 1차 테스트(같은 시기 배치 {PAPER_N['primary_excl']}셀) {PAPER_T1['Full']['primary_excl']}%와 2차 테스트(나중 배치 2018-04-12 = 우리 Batch 3, {PAPER_N['secondary']}셀) {PAPER_T1['Full']['secondary']}%를 셀 수로 가중평균하면 {PO['paper_9p1_as_weighted_mean_full']:.2f}%로, 두 값을 합친 것으로 보인다.",
                f"원저자 공개 코드(LoadData.m)의 1차 테스트는 2017-05-12·2017-06-30 두 배치의 셀을 학습과 번갈아 나눈 것이다. Batch 1 부분만 봐도 1차 테스트 {PO['n_test_b1']}셀 중 {PO['n_test_with_same_policy_in_train']}셀은 같은 충전 정책 셀이 학습에 있다(복제 정책 {PO['n_replicated_policies']}개 중 {PO['n_replicated_policies_split']}개가 양쪽으로 갈림). 정책째 뗀 우리 Hold-out이 더 엄격한 조건이다.",
                f"원논문은 우리 Batch 2(2018-02-20)를 테스트하지 않았다. 조건이 맞는 비교는 같은 시기 {PAPER_T1['Full']['primary_excl']}% vs Hold-out {r1['Valid (Batch 1 Hold-out)']:.2f}%, 나중 배치 {PAPER_T1['Full']['secondary']}% vs Batch 3 같은 40셀 {excl_mape:.2f}%다.",
                f"노션은 'Batch 1/2 유사'라고 했지만, 받은 Batch 2의 fastcharge {fc['n']}셀은 모두 학습 최저 수명({B1_LABEL_MIN:,.0f})보다 짧다. 학습 라벨 범위 밖을 맞혀야 하는 테스트다.",
                f"Batch 2에는 충전 단계 구성이 다른 newstructure {ns['n']}셀도 섞여 있다.",
                f"550 미만 단수명 {A['h3']['n_short']}셀(Batch 2의 {A['h3']['n_short'] / len(S['B2']) * 100:.0f}%)이 오차 합의 {A['h3']['APE 합 비율']:.0f}%를 차지한다. 셀 수 비중과 비슷하고, 셀당 오차는 조금 더 크다(MAPE {A['h3']['짧은 셀 MAPE']} vs {A['h3']['나머지 MAPE']}).",
                f"같은 모델이 Batch 3에서는 {r1['Test (Batch 3)']:.2f}%로 목표에 가깝다.",
                "Batch 2 라벨로 튜닝해 Gap을 줄이는 것은 누수라서 하지 않았다."]},
        {"Gap": "Gap (Batch2-Batch3)", "M1": r1["Gap (Batch2-Batch3)"], "M0": r0["Gap (Batch2-Batch3)"],
         "한 줄 해석": "Batch 3에서 오히려 좋아졌다. ΔQ 깊이에서는 배치 과적합 신호가 보이지 않고, 초기 용량은 배치에 민감하다.",
         "해석": ["분포가 다른 것과 관계가 다른 것은 다르다.",
                f"Batch 3는 수명이 더 길지만, 같은 ΔQ 깊이에서 Batch 1과 거의 같은 수명을 보인다(수명비 {r_b3:.2f}).",
                f"Batch 2 fastcharge는 관계 자체가 아래로 이동했다(수명비 {r_fc:.2f}). Batch 2 newstructure는 {r_ns:.2f}로 덜하다.",
                f"Batch 3 오차는 학습 라벨보다 긴 셀(1,074 초과)의 과소예측에 몰려 있다(그 {A['b3_long']['n']}셀 MAPE {A['b3_long']['MAPE']}%).",
                f"Batch 3에서 기준선 이동의 영향을 받은 쪽은 ΔQ 깊이가 아니라 초기 용량이다. M1은 평균 {pct(g3['전체']['평균 부호 오차 (%)'])}, M0은 {pct(g3['전체']['M0 평균 부호 오차 (%)'])}다.",
                "배치가 3개뿐이라 '과적합 신호가 보이지 않는다'는 판단의 근거는 제한적이다."]},
        {"Gap": "Gap (Target-Test, Batch 3)", "M1": r1["Gap (Target-Test, B3)"], "M0": r0["Gap (Target-Test, B3)"],
         "한 줄 해석": "원논문의 2차 테스트셋인 Batch 3에서는 목표와의 차이가 작다.",
         "해석": [f"Batch 3 MAPE {r1['Test (Batch 3)']:.2f}%는 목표보다 {r1['Gap (Target-Test, B3)']:.2f}%p 높다.",
                f"원저자 공개 코드가 뺀 4셀(#02·#37·#42·#43, 테스트 뒤 확인)을 빼면 {excl_mape:.2f}%로, 차이가 {excl_mape - TARGET:+.2f}%p로 준다. 이 40셀은 원논문 2차 테스트셋과 같은 셀이다.",
                f"같은 40셀에서 원논문 Table 1은 Variance {PAPER_T1['Variance']['secondary']}%, Discharge {PAPER_T1['Discharge']['secondary']}%, Full {PAPER_T1['Full']['secondary']}%다. 우리 M0 {A['excl_m0']['mape']:.2f}%, M1 {excl_mape:.2f}%는 Variance·Full과 비슷하고, 방전 피처 모델보다 {excl_mape - PAPER_T1['Discharge']['secondary']:.1f}%p 나쁘다.",
                "다만 M1은 Batch 3를 평균적으로 짧게 예측한다. 교체를 일찍 잡는 쪽(비용)으로 치우친다."]},
    ]

    causes = [
        {"원인 가설": "배치(로트·시험 시기) 수준 이동", "설명": "같은 ΔQ 깊이·같은 충전 정책에서도 Batch 2 fastcharge의 수명이 짧다. 초기 100 사이클 피처에 담기지 않은 배치 요인이 있다.",
         "근거": [f"수명비 {r_fc:.2f}(EDA Q5, 같은 ΔQ 창 중앙값 {A['window']['rel_diff'] * 100:+.0f}%).",
                (f"4.8C(80%)-4.8C 수명 중앙값: Batch 1 {p48['batch1']['수명 중앙값']:g} vs Batch 2 {p48['batch2']['수명 중앙값']:g} 사이클." if p48 else ""),
                f"사후: 셀 공통 배율 하나를 빼면 MAPE {post['B2']['M1 MAPE (보고)']:.2f} → {post['B2']['공통 배율 제거 후 M1 MAPE']:.2f}%(라벨 사용)."],
         "주의": "후보는 셀 로트·시험 시기·충전 단계 구성(휴지·CV 길이) 차이다. 이 데이터로는 어느 것인지 가를 수 없다(상관)."},
        {"원인 가설": "Batch 2 fastcharge 안에서 ΔQ 깊이–수명 관계가 평평함", "설명": "ΔQ 깊이가 달라도 수명은 392~514로 거의 같다. 그래서 ΔQ가 얕은(학습 범위 안) 셀일수록 B1 선이 수명을 더 길게 준다.",
         "근거": [f"기울기 {A['slopes']['batch2_fastcharge']['slope']:.2f} (Batch 1 {A['slopes']['batch1']['slope']:.2f}).",
                f"B2 MAPE 범위 안 {g2['전체']['범위 안 MAPE']}% > 범위 밖 {g2['전체']['범위 밖 MAPE']}%."],
         "주의": f"DAY 1 Q2에서 B2 fastcharge 다수가 cycle 100 근처까지 용량이 오르고 있었다(요약 QD 정점). ΔQ₁₀₀₋₁₀ 창이 열화가 아니라 break-in을 함께 잴 가능성이 있다. fastcharge 안의 상승폭 상관(ρ {A['fc_breakin']['rho']:+.2f})은 이를 일부 지지하지만, 사전 지정 밖의 탐색 결과이고, 39셀 전체의 상승폭 상관은 {A['bc2']['rho_cc_breakin']:+.2f}(p {A['bc2']['p_cc_breakin']:.2f})로 약하다."},
        *([{"원인 가설": "Batch 2 ΔQ 창 안의 약 66시간 기록 공백 (Batch 2만의 시험 이력)",
            "설명": "ΔQ = Q100 − Q10 창 안에서 시험이 멈췄다가 다시 시작됐다. 같은 ΔQ 깊이라도 Batch 2는 다른 이력을 잰 값일 수 있다.",
            "근거": [gap_txt, gap_peak_txt,
                   f"사후: 공백 뒤 용량 변화(뒤 − 앞, 떨어지면 음수)와 M1 부호 오차의 순위상관 {gap_rho['spearman_rho']:+.2f}(p {gap_rho['p']:.3f}). 배치 안에서는 많이 떨어진 셀일수록 오히려 덜 길게 예측됐다. 모든 Batch 2 셀에 공백이 있어 배치 수준 영향은 가릴 수 없다."],
            "주의": "모든 Batch 2 셀에 있어 비교군이 없다. 원인으로 확정하지 않는다(상관). 개선은 공백을 라벨 없이 감지해 공백 없는 창으로 ΔQ를 다시 정의하고 Batch 1 CV로만 검증하는 것이다."}]
          if gap_txt else []),
        {"원인 가설": "초기 용량(Qcc_init) 절대 수준의 배치 이동 (노션 Qdlin 경고와 같은 방향)", "설명": "초기 용량은 Qdlin 절대 수준이라 배치마다 기준선이 다를 수 있다. 모든 테스트 배치가 Batch 1보다 낮아 M1 예측을 일괄적으로 낮췄다.",
         "근거": [f"라벨 없이 잰 Batch 3 이동 {qcc['B3']['사전 이동량 (mAh, 라벨 없음)']} mAh → 예상 배율 ×{qcc['B3']['사전 예상 배율']:.2f}(테스트 전 기록).",
                f"B3 평균 부호 오차 M0 {pct(g3['전체']['M0 평균 부호 오차 (%)'])} → M1 {pct(g3['전체']['평균 부호 오차 (%)'])}.",
                f"사후: 이동을 되돌리면 M1의 B3 평균 부호 오차는 {pct(post['B3']['초기 용량 이동 되돌린 M1 평균 부호 오차 (%)'])}다(라벨 사용)."],
         "주의": (f"Batch 2에서 M1이 M0보다 줄인 {dec['M0 → M1 전체']:.1f}%p 중 약 {dec['배치 수준 초기 용량 이동 (이동 되돌린 M1 → M1)']:.1f}%p는 이 배치 수준 이동 효과다(사후 분해). "
                f"Batch 2 fastcharge 안에서도 초기 용량은 ΔQ 깊이를 통제한 뒤 수명과 함께 움직였다(편상관 {post['초기 용량 편상관 (ΔQ 깊이 통제, log10 수명)']['B2 fastcharge']['r']:+.2f}). "
                "이 이동이 측정 차이(노션 경고)인지 실제 용량 차이인지는 확인하지 못했다. 실제 용량 차이라면 고정 EOL(0.88 Ah)까지 여유가 줄어 수명이 짧아지는 것이 맞다.")},
        {"원인 가설": "학습 라벨 범위 밖 외삽 (장수명 쪽)", "설명": "Batch 1 학습 라벨은 534~1,074 사이클뿐이다. 그보다 긴 셀은 ΔQ 깊이도 학습 범위 밖인 경우가 많아 두 겹의 외삽이 된다.",
         "근거": [f"Batch 3 1,074 초과 {A['b3_long']['n']}셀 MAPE {A['b3_long']['MAPE']}%.",
                f"같은 셀의 M0 평균 부호 오차도 {pct(A['b3_long']['평균 부호 오차 M0 (%)'])}로 과소예측이다."],
         "주의": ("Batch 1의 장수명 셀(#00~04)은 중도절단이라 학습에서 빠졌다. 원저자 공개 코드(테스트 뒤 확인)는 이 5셀에 다음 실험(2017-06-30) 파일의 이어진 기록을 붙여 "
                "수명 라벨(약 1,430~2,240)을 확정해 썼다. 과제 데이터에는 그 파일이 없어 우리 학습 라벨 상한은 1,074다. "
                "이 정보 손실은 장수명 쪽 과소예측의 원인 후보다(미검증, 생존분석 모델은 돌리지 않았다).")},
        {"원인 가설": "데이터 품질 문제 셀", "설명": "Batch 3 최악 셀 중 일부는 원저자 공개 코드가 품질 문제로 뺀 셀이다(테스트 뒤 확인. Load Data.ipynb: b3c2·23·32·37·42·43, 번호 = .mat 순서 = 이 저장소 셀 번호).",
         "근거": [f"Batch 3 상위 10셀 중 {len(top3_removed)}셀({'·'.join(f'{k}위' for k in top3_removed_rank)}).", f"4셀 제외 시 MAPE {excl_mape:.2f}%."],
         "주의": "제거 여부는 사전에 정한 '포함/제외 병기' 규칙대로 보고만 했다. 6셀 중 #23·#32는 EOL 미도달이라 라벨이 없어 원래 평가 대상이 아니다. 처음에는 목록을 #38·#39로 잘못 적었고, 저장된 예측으로 다시 계산해 바로잡았다."},
    ]

    improve = [
        {"방향": "새 로트 투입 전 '분포 이동 경보' (라벨 불필요)", "내용": "ΔQ 깊이·초기 용량·break-in 정점이 학습 분포를 벗어나면 예측을 '참고용'으로 낮춘다.",
         "근거": f"Batch 2에서 80% 예측구간이 실제 수명을 담은 비율은 {cov2['n_covered']}/{len(S['B2'])}셀뿐이다. 구간만으로는 배치 이동을 경고하지 못한다."},
        {"방향": "로트별 참조 셀로 수준(절편) 보정", "내용": "새 로트마다 소수 셀을 EOL까지(또는 가속 조건으로) 돌려 수명 수준(절편)만 다시 맞춘다. 이번 과제에서는 B2 라벨을 쓰지 않았고, 향후 데이터 수집 설계로 제안한다.",
         "근거": "오차의 대부분이 한 방향(과대예측) 공통 치우침이라(사후: 공통 배율 제거 시 MAPE가 절반 이하), 기울기보다 절편 보정의 효과가 클 것으로 본다(가설)."},
        {"방향": "초기 용량을 배치에 덜 민감한 정의로 바꾸고, 새 로트에서는 이동 경보로 쓰기",
         "내용": ("절대 수준 대신 셀 자기 기준(상대값)으로 정의하거나, 측정 장비·프로토콜 차이를 보정한다. 선택은 사전 규칙대로 M1을 유지한다. "
                "새 로트에서는 라벨 없이 잴 수 있는 초기 용량 이동량과 M1·M0 예측 차이를 함께 계산해, 차이가 크면 '학습 분포 밖' 경보를 띄운다. "
                "모델 교체(M0/M1) 여부는 새 검증 데이터로 정하며, 이번 테스트 결과로 바꾸지 않는다(사후 권고, 재선택 아님)."),
         "근거": (f"M1은 두 테스트 배치 모두 M0보다 MAPE가 낮았다(B2 {r1['Test (Batch 2)']:.2f} vs {r0['Test (Batch 2)']:.2f}, B3 {r1['Test (Batch 3)']:.2f} vs {r0['Test (Batch 3)']:.2f}). "
                f"그러나 B3를 평균 {pct(g3['전체']['평균 부호 오차 (%)'])} 짧게 예측했고, 이 방향은 라벨 없이 잰 이동량(×{qcc['B3']['사전 예상 배율']:.2f})으로 테스트 전에 알 수 있었다.")},
        {"방향": "break-in 이후 창으로 ΔQ 재정의 실험", "내용": "용량 상승이 끝난 뒤의 고정 길이 창에서 ΔQ를 다시 잰다. 조기 예측(100 사이클)과의 맞교환을 Batch 1 CV로 먼저 검증한다. 이 피처를 Batch 2로 다시 채점하면 더 이상 공정한 테스트가 아니다.",
         "근거": "B2 fastcharge 안에서 ΔQ–수명 기울기가 평평했고, break-in이 큰 셀일수록 더 과대예측했다(사전 지정 밖의 탐색 결과)."},
        {"방향": "중도절단 셀을 학습에 넣기 (생존분석·Tobit형: 중도절단 셀의 '최소 수명' 정보를 반영하는 회귀)", "내용": "Batch 1 장수명 #00~04의 '최소 1,177 사이클 이상' 정보를 학습에 반영한다.",
         "근거": f"Batch 3 장수명 셀(1,074 초과)이 과소예측됐다(MAPE {A['b3_long']['MAPE']}%)."},
        {"방향": "여러 배치로 학습하고 '배치 단위'로 검증", "내용": "새 데이터가 생기면 배치 하나를 통째로 빼는 교차검증으로 배치 일반화를 직접 잰다. 시험 조건 메타데이터(휴지·CV 길이·챔버 온도·로트 번호)도 함께 기록한다. 향후 제안이며 이번에는 하지 않았다. Batch 2를 학습에 쓰면 Batch 2는 더 이상 테스트로 쓸 수 없다.",
         "근거": "이번 Gap(Valid−Test)는 배치 내부 검증(CV·Hold-out)으로는 미리 보이지 않았다."},
    ]

    rank3 = g3["전체"]["수명 순위 Spearman (예측 vs 실제)"]
    short_share = scr["실제 단수명(<550)"] / len(S["B2"]) * 100
    late = float(g2["전체"]["평균 부호 오차 (%)"]) / 100 * 450
    domain = {
        "활용 가능한 의사결정": [
            {"의사결정": "같은 로트 안 상대 순위 → 점검·교체 우선순위, 팩 구성(수율 선별)",
             "설명": f"절대 수명은 틀려도 순서는 대체로 맞힌다. 한 그룹 44셀인 Batch 3에서 예측–실제 순위상관이 {rank3:.2f}다. 다만 Batch 2 fastcharge 안에서는 {fc['수명 순위 Spearman (예측 vs 실제)']:.2f}로 약해지므로, 순위는 같은 로트·조건 안에서만 쓴다."},
            {"의사결정": "단수명 조기 경보 (예측 < 550)",
             "설명": (f"경보가 울린 {scr['M1 경보(예측<550)']}셀은 모두 실제 단수명이었다(오경보 0). "
                    f"다만 Batch 2는 {short_share:.0f}%가 단수명이고 모델이 거의 모든 셀을 길게 예측했으므로, 오경보 0은 이 편향의 부산물에 가깝다. "
                    f"550 임계는 DAY 1 §9에서 정한 스크리닝 기준이고, 경보 적중 확인은 테스트 라벨을 본 사후 분석이다. 실제 단수명 {scr['실제 단수명(<550)']}셀 중 {scr['놓친 단수명']}셀은 놓쳤다. '경보 없음'을 '안전'으로 읽으면 안 된다.")},
            {"의사결정": "학습과 같은 로트·조건에서의 교체 시점·예산 계획 (교체 비용 = CAPEX 30~40%, 노션 배경)",
             "설명": f"Batch 1 안에서는 오차가 약 {r1['Valid (Batch 1 Hold-out)']:.0f}~{r1['Train (Batch 1 CV)']:.0f}%다. 800 사이클 셀이면 약 ±50 사이클 수준으로 교체 시점을 미리 잡을 수 있다."},
        ],
        "한계": [
            {"한계": "로트가 바뀌면 수명을 길게 본다",
             "설명": (f"Batch 2에서 평균 {pct(g2['전체']['평균 부호 오차 (%)'])} 과대예측했다. ESS에서는 교체가 늦어져 용량 부족·피크 대응 실패·위약금으로 이어지는 더 위험한 방향이다. "
                    f"운영으로 옮기면 수명 450 사이클 셀의 교체를 약 {round(late, -1):.0f} 사이클 늦게 잡는 셈이고, 하루 1사이클 운전을 가정하면 약 3개월이다(가정).")},
            {"한계": "예측구간이 배치 이동 앞에서 무너진다", "설명": f"80% 구간이 Batch 2에서 {cov2['n_covered']}/{len(S['B2'])}셀만 담았다. 같은 로트에서 만든 불확실성은 새 로트에 그대로 쓸 수 없다."},
            {"한계": "실험실 조건과 현장 조건이 다르다", "설명": f"데이터는 항온 챔버의 완전 충방전·고속 충전 시험이다(셀 평균 온도 중앙값 {t_med.min():.0f}~{t_med.max():.0f}°C). 실제 BESS는 부분 충방전, 낮은 C-rate, 온도 변동, 대기 중 열화(캘린더 열화)가 섞인다."},
            {"한계": "입력을 현장에서 바로 얻기 어렵다", "설명": "ΔQ(V)는 사이클 10과 100의 완전 정전류 방전 곡선이 필요하다. 운전 중 BMS 데이터만으로는 계산할 수 없다."},
            {"한계": "초기 용량 효과는 '건강도'가 아니다", "설명": "고정 EOL(0.88 Ah)에서는 초기 용량이 곧 EOL까지의 여유다. 효과 일부가 라벨 정의에서 나오므로 셀 건강 지표로 해석하지 않는다."},
            {"한계": "학습 데이터가 작고 단일하다", "설명": "학습 라벨은 한 로트의 36셀, 한 셀 모델이며 수명 534~1,074 사이클뿐이다. 그 밖의 수명은 외삽이다."},
        ],
        "실 배포에 필요한 것": [
            "여러 로트·시험 시기의 학습 데이터와, 로트마다 수명 수준을 맞출 참조 셀.",
            "새 로트·새 랙의 피처가 학습 분포 안에 있는지 보는 분포 이동 감시와 신뢰도 표시.",
            "현장에서 ΔQ(V)를 잴 정기 기준 성능 시험(RPT) 절차와 BMS 데이터 파이프라인.",
            "운영 중 사이클이 쌓이면 예측을 갱신하는 RUL(잔여 수명) 방식으로의 확장.",
            "과대예측(교체 지연)의 비용이 더 크다는 비대칭 비용을 반영한 보수적 운용 기준.",
            "부분 충방전·저 C-rate·온도 변동 같은 현장 조건에서의 재검증.",
        ],
    }
    return dict(worst_commonalities=worst, hypotheses=hyp, gaps=gaps, causes=causes, improve=improve, domain=domain)


# ───────────────────────────── main ─────────────────────────────
def main() -> None:
    S = load()
    chk = consistency(S)
    A = analysis(S)
    figs = {"pred_vs_true": fig_pred_vs_true(S, A), "b2_error": fig_b2_error(S, A), "m1_vs_m0": fig_m1_vs_m0(S, A)}
    N = narrative(S, A, chk)
    inputs = ["test_predictions.csv", "batch3_predictions.csv", "valid_predictions.csv", "cv_oof.csv", "features.csv",
              "diagnostic_features.csv", "test.json", "batch3.json", "batch3_excl_corrected.json", "eda/q5_results.json"]
    inputs += ["rest_gap_summary.json"] if S["rest_gap"] else []
    out = {
        "meta": {"created": datetime.now().isoformat(timespec="seconds"), "script": "src/error_analysis.py",
                 "selected": SEL, "baseline": BASE,
                 "원칙": "저장된 예측만 사용. 모델 재적합·재평가·선택 변경 없음. Test/Batch 3 단계 재실행 없음. B2/B3 라벨은 사후 해석에만 사용.",
                 "부호": "부호 오차 = (예측 − 실제)/실제 × 100. + = 수명을 길게 예측(과대예측).",
                 "용어": {"ΔQ 깊이": "dQ_logvar = log10 var(ΔQ₁₀₀₋₁₀(V)), 클수록 ΔQ 곡선이 깊음(단수명 쪽)",
                        "초기 용량": "Qcc_init = cycle 2~6 CC 방전 2.0 V 끝점 중앙값 (Ah)",
                        "break-in 정점 사이클": "초기 용량이 오르다 최고점에 닿는 사이클 (CC 끝점 = 주, 요약 QD = 보조)",
                        "학습 ΔQ 범위": f"[{A['rng'][0]:.3f}, {A['rng'][1]:.3f}] (Batch 1 학습 셀)",
                        "CV-OOF 그림": "셀별 20회 out-of-fold 예측의 기하평균. 보고 Train MAPE는 seed×fold 평균"},
                 "input_sha256_12": {f: sha12(RES / f) for f in inputs}},
        "consistency_check": chk,
        "gap_interpretation": N["gaps"],
        "worst_cells": {
            "batch2": {"top10": A["top2"], "공통점": N["worst_commonalities"]["batch2"]["공통점"], "상위10_vs_나머지": A["tvr2"]},
            "batch3": {"top10": A["top3"], "공통점": N["worst_commonalities"]["batch3"]["공통점"], "상위10_vs_나머지": A["tvr3"]},
        },
        "hypothesis_verdicts": N["hypotheses"],
        "cause_hypotheses": N["causes"],
        "improvement_directions": N["improve"],
        "domain_interpretation": N["domain"],
        "supporting_numbers": {
            "group_stats_batch2": A["gs2"], "group_stats_batch3": A["gs3"], "group_stats_valid": A["vs"],
            "eda_q5_life_ratio_to_b1_line": A["ratio"], "eda_q5_same_window": A["window"],
            "eda_q5_slopes": {k: {"slope": rnd(v["slope"], 3)} for k, v in A["slopes"].items()},
            "same_policy_cross_batch": A["same_policy"], "screening_lt550_batch2": A["scr"], "h3_short_cells": A["h3"],
            "batch3_beyond_label_max": A["b3_long"],
            "qcc_effect_m1_vs_m0": {"주의": "M1÷M0 예측 비와 초기 용량 차이의 Pearson r은 모델 구조상 당연히 높다(log10(M1/M0)가 두 피처의 선형식). 모델 계산 확인이며 오차 원인의 증거가 아니다. 원인 근거는 posthoc_label_based 와 사전 예상 배율을 본다.",
                                    **A["qcc"]},
            "posthoc_label_based": A["post"],
            "pi_coverage_10_90": {k: {"nominal": v["nominal"], "covered": v["n_covered"], "coverage": rnd(v["coverage"], 3)} for k, v in A["cov"].items()},
            "b2_candidate_test_mape_range": A["b2_cand_range"], "batch3_excluding_paper_removed_mape": rnd(A["excl"]["mape"], 2),
            "batch3_excluding_paper_removed_mape_M0": rnd(A["excl_m0"]["mape"], 2),
            "paper_removed_cells": A["removed"],
            "paper_table1": PAPER_T1, "paper_primary_split_overlap": paper_split_overlap(),
            "batch2_rest_gap": S["rest_gap"],
        },
        "figures": {
            "reports/figures/day2_pred_vs_true.png": "예측 vs 실제(로그 축, y=x·±20% 띠). B1 CV·Valid는 띠 안, B2는 y=x 위로 치우침, B3 장수명은 아래.",
            "reports/figures/day2_b2_error.png": "(a) 같은 ΔQ 깊이에서 B2 fastcharge 수명이 B1 선의 0.76배. (b) M1 예측÷실제 vs ΔQ 깊이 — fastcharge에서는 범위 안 셀이 더 크게 과대예측(newstructure는 범위 밖 셀이 더 크게 틀림).",
            "reports/figures/day2_m1_vs_m0.png": "(a) 그룹별 평균 부호 오차 M0→M1. (b) 모델 계산 확인: M1÷M0 예측 비 vs 초기 용량 차이(구조상 당연한 관계). X = 테스트 전 라벨 없이 계산한 그룹별 예상 배율.",
        },
    }
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2, default=lambda o: o.item() if hasattr(o, "item") else str(o)))
    print("wrote", OUT_JSON.relative_to(ROOT))
    for p in figs.values():
        print("wrote", Path(p).relative_to(ROOT))


if __name__ == "__main__":
    main()
