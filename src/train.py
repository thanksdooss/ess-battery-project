"""DAY 2 모델 개발·평가 — DAY 1 전략(reports/day1_design_report.md §8~11, results/eda/strategy_final.json)을 그대로 구현한다.

과제: Regression. 모델 내부 타깃 y = log10(cycle_life), 평가는 10^ŷ 의 MAPE(%) (smearing 보정·클리핑 없음).
입력: results/features.csv (src/features.py, cycle 2~100 만 사용)

단계 (--stage)
  select : Train 27셀(Hold-out 9셀 제외)·15정책에서 정책 단위 GroupKFold(5, shuffle, seed 42~61) × 20회 nested CV
           (내부 GroupKFold(4)·원척도 MAPE 로 튜닝) → 외삽 게이트(#00~04) → 사전 고정 1-SE 규칙
           → results/selection.json, results/cv_results.csv, results/cv_oof.csv
           + 기록 전용 민감도: B1 잡음 셀(#09·38~45) dQ_logvar 를 21점 이동중앙값 판으로 바꿔 선택 모델·M0 를
             같은 nested CV 로 다시 잰다 (selection.json 'sensitivity_dq_med21', results/sensitivity_dq_med21_cv.csv).
             선택은 원정의로만 정한다.
  valid  : 선택 설정을 Train 27셀로 적합해 Hold-out 9셀을 1회 예측(M0 행·다른 후보는 기록용) → results/valid.json
           + M3 구간 비교(GPR predict(return_std=True) 10~90% vs CV 잔차 10~90%: 포함률·폭), 21점 민감도 Valid
  test   : (1회, guard) B1 라벨 36셀로 재적합 → Batch 2 라벨 39셀 → results/test.json, results/.test_done
  batch3 : (선택, 1회, guard) 같은 고정 모델 → Batch 3 라벨 44셀(2026-10-01 1회 실행은 DAY 1 목록 #02·37·38·39 포함/제외 → 10-02 사후 정정 #02·37·42·43, 아래 사후 정정 블록)
  report : results/model_performance.csv (노션 Regression 포맷), _M0.csv, model_candidates.csv,
           (test 후) model_performance_test_detail.csv (Test 비고 동반 표), (batch3 후) _batch3*.csv

Test 비고 (DAY 1 §11): 그룹별 MAPE, 부호 있는 오차, 과대예측 비율, dQ_logvar 학습 범위 안/밖 MAPE,
  계수 × 라벨 없는 Qcc_init 이동량(그룹 전체 셀 중앙값 − B1 46셀 중앙값, DAY 1 q5d 와 같은 정의),
  H1 사전 점검: 로그 잔차 vs break-in 상승폭 · 정점 사이클 Spearman (breakin_check).
  정점 사이클은 results/diagnostic_features.csv(라벨 없음, cycle ≤100)에서만 읽는다 — features.csv 에 넣지 않아
  features_csv_sha256 과 선택 결과가 그대로다. 이 파일은 test/batch3 단계만 읽는다.

라벨 규칙 (테스트 1회 원칙)
  - select/valid 는 load_batch1() 만 쓴다: features.csv 에서 Batch 2/3 행은 파싱 단계에서 건너뛰어
    Batch 2/3 cycle_life 가 이 경로에 들어오지 않는다.
  - Batch 2/3 라벨은 stage_test / stage_batch3 의 load_labeled_batch() 에서만 읽는다
    (Qcc 이동량용 load_batch_features() 는 라벨 열을 읽지 않는다).
  - test/batch3 는 selection.json 이 없거나 완료 표식(.test_done/.batch3_done)이 있으면 거부(--force 는 run_log 에 기록).
    표식은 덧붙이기 모드라 --force 재실행에도 첫 실행 기록이 남는다.
  - select 는 .test_done 또는 .batch3_done 이 있으면 거부(--force 는 run_log 에 기록).

Gap 부호(노션 (+) 의미 유지): Gap(Train-Valid) = Valid − Train, Gap(Valid-Test) = Test − Valid,
Gap(Target-Test) = Test − 9.1, Gap(Batch2-Batch3) = Batch 3 − Batch 2.

실행: cd ess-battery-project && python src/train.py --stage select   (→ valid → test(1회) → batch3(선택, 1회) → report)
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import sys
import time
import warnings
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from joblib import Parallel, delayed  # noqa: E402
from scipy import stats  # noqa: E402
from sklearn.dummy import DummyRegressor  # noqa: E402
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor  # noqa: E402
from sklearn.gaussian_process import GaussianProcessRegressor  # noqa: E402
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, DotProduct, WhiteKernel  # noqa: E402
from sklearn.impute import SimpleImputer  # noqa: E402
from sklearn.linear_model import ElasticNet, HuberRegressor, LinearRegression, Ridge  # noqa: E402
from sklearn.metrics import make_scorer  # noqa: E402
from sklearn.model_selection import GridSearchCV, GroupKFold  # noqa: E402
from sklearn.pipeline import Pipeline  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

from data import RANDOM_STATE, ROOT  # noqa: E402
from features import (DIAG_CSV, DIAG_FEATURES, FEATURES_CSV, LAST, M1_FEATURES, M2_DISCHARGE,  # noqa: E402
                      M2_POOL, SENS_FEATURES)

RES = ROOT / "results"
HOLDOUT_CSV = RES / "holdout_cells.csv"
SELECTION_JSON, VALID_JSON = RES / "selection.json", RES / "valid.json"
TEST_JSON, BATCH3_JSON = RES / "test.json", RES / "batch3.json"
TEST_DONE, BATCH3_DONE = RES / ".test_done", RES / ".batch3_done"
RUN_LOG = RES / "run_log.txt"

SEEDS = list(range(42, 62))          # 외부 반복 seed 42~61 (20회)
N_OUTER, N_INNER, N_FINAL = 5, 4, 5  # 외부 GroupKFold(5), 내부 GroupKFold(4), 최종 하이퍼파라미터 GroupKFold(5, seed 42)
N_BOOT = 2000                        # Hold-out 셀 부트스트랩
TARGET_MAPE = 9.1                    # 원논문 Regression MAPE (노션 Target)
HOLDOUT_IDX = [16, 17, 23, 30, 31, 32, 33, 44, 45]
GATE_CELLS = [f"batch1-{i:02d}" for i in (0, 1, 2, 3, 4)]                 # 외삽 게이트 (하한 1,177~1,227)
GATE_DIAG = [f"batch1-{i:02d}" for i in (8, 10, 12, 13, 22)]            # 진단만
GATE_MAX_VIOL = 3                    # #00~04 중 하한 위반 ≥ 3 → 부적격
OOF_FOCUS = ["batch1-20", "batch1-21"]                                   # B1 하단 셀(<560) out-of-fold 오차
B3_PAPER_REMOVED = ["batch3-02", "batch3-37", "batch3-42", "batch3-43"]  # 10-02 사후 정정: 원저자 Load Data.ipynb 가 지운 b3c2·37·42·43 (b3cN = .mat 순서 = idx). DAY 1 strategy_final 목록과 다름
B3_PAPER_REMOVED_ALL = B3_PAPER_REMOVED + ["batch3-23", "batch3-32"]      # 6셀 전체 (#23·#32 는 EOL 미도달, 라벨 없음)
NOISY_B1 = [f"batch1-{i:02d}" for i in (9, 38, 39, 40, 41, 42, 43, 44, 45)]  # ΔQ 고주파 잡음 셀 (DAY 1 Q3-(a))
SENS_DQ = SENS_FEATURES[0]           # 'dQ_logvar_med21': 잡음 셀에서만 dQ_logvar 대신 넣는 21점 이동중앙값 판
PI_Q = (10, 90)                      # 예측구간: CV 로그 잔차 10~90% 분위
PI_Z = float(stats.norm.ppf(PI_Q[1] / 100))   # GPR 구간: log10 척도 μ ± z·σ (z = 1.2816)
PI_MIN_N = 10                        # 외삽 구간 잔차가 이보다 적으면 전체 잔차 분위 사용
LIFE_RANGE_NOTION = (150, 2300)      # 노션 수명 범위 (밖 예측 = 외삽 플래그)
PEAK_EDGE = 96                       # 정점 ≥ 96 = 9-사이클 중심 창이 cycle 100 에서 잘리는 끝단 (창 절단 가능 셀 수 보고)
BREAKIN_COLS = ["cc_breakin", *DIAG_FEATURES]   # H1 점검 변수: 상승폭(features.csv) + 정점 사이클 주·보조(진단 파일)
LABEL_COLS = ("cycle_life", "labeled", "censored")


# ════════════════════════════════════════════════════════════════════════
# 지표
# ════════════════════════════════════════════════════════════════════════
def mape_pct(true_life, pred_life) -> float:
    t, p = np.asarray(true_life, float), np.asarray(pred_life, float)
    return float(np.mean(np.abs(p - t) / t) * 100)


def mape_from_log(y_log, p_log) -> float:
    """내부 튜닝 scorer: log10 타깃을 10^ 로 되돌린 원척도 MAPE(%)."""
    return mape_pct(10 ** np.asarray(y_log, float), 10 ** np.asarray(p_log, float))


MAPE_SCORER = make_scorer(mape_from_log, greater_is_better=False)


def bootstrap_ci(ape: np.ndarray, n_boot: int = N_BOOT) -> list[float]:
    """셀 단위 부트스트랩 95% CI (percentile, default_rng(42))."""
    rng = np.random.default_rng(RANDOM_STATE)
    idx = rng.integers(0, len(ape), size=(n_boot, len(ape)))
    m = ape[idx].mean(axis=1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]


# ════════════════════════════════════════════════════════════════════════
# 후보 모델 (DAY 1 §10 표의 그리드 그대로)
# ════════════════════════════════════════════════════════════════════════
@dataclass(frozen=True)
class Spec:
    id: str
    name: str
    features: tuple
    make: Callable[[], object]
    grid: dict            # GridSearchCV param_grid, {} = 튜닝 없음
    candidate: bool       # True = 선택 후보(M0~M4), False = 참고 행(선택 대상 아님)


def gpr_kernel():
    """선형(DotProduct, 외삽 유지) + 비선형(RBF) + 잡음(White). 파라미터는 fold 안 주변우도로 추정."""
    return (ConstantKernel(1.0) * DotProduct(sigma_0=1.0)
            + ConstantKernel(0.1) * RBF(length_scale=1.0, length_scale_bounds=(0.1, 100.0))
            + WhiteKernel(noise_level=1e-3))


def _m0():
    return LinearRegression()


def _m1():
    return Pipeline([("scale", StandardScaler()), ("model", Ridge())])


def _m2():
    return Pipeline([("impute", SimpleImputer(strategy="median")), ("scale", StandardScaler()),
                     ("model", ElasticNet(max_iter=100000, random_state=RANDOM_STATE))])


def _m3():
    gpr = GaussianProcessRegressor(kernel=gpr_kernel(), normalize_y=True, n_restarts_optimizer=5,
                                   random_state=RANDOM_STATE)
    return Pipeline([("scale", StandardScaler()), ("model", gpr)])


def _rf():
    return RandomForestRegressor(n_estimators=500, max_features=1.0, random_state=RANDOM_STATE, n_jobs=1)


def _gbr():
    return GradientBoostingRegressor(learning_rate=0.05, max_depth=2, min_samples_leaf=3, subsample=0.8,
                                     random_state=RANDOM_STATE)


def _const():
    return DummyRegressor(strategy="median")


def _huber():
    return Pipeline([("scale", StandardScaler()), ("model", HuberRegressor(epsilon=1.35, max_iter=1000))])


def _logspace(a: float, b: float, n: int) -> list[float]:
    return [float(x) for x in np.logspace(a, b, n)]


EN_GRID = {"model__alpha": _logspace(-4, 0, 30), "model__l1_ratio": [0.1, 0.3, 0.5, 0.7, 0.9, 1.0]}
SPECS = {s.id: s for s in [
    Spec("M0", "OLS log10(life) ~ dQ_logvar (원논문 variance model)", ("dQ_logvar",), _m0, {}, True),
    Spec("M1", "StandardScaler → Ridge (dQ_logvar + Qcc_init)", tuple(M1_FEATURES), _m1,
         {"model__alpha": _logspace(-4, 2, 25)}, True),
    Spec("M2", "SimpleImputer(median) → StandardScaler → ElasticNet (10개 풀)", tuple(M2_POOL), _m2, EN_GRID, True),
    Spec("M3", "StandardScaler → GPR(C·DotProduct + C·RBF + White) (M1 입력)", tuple(M1_FEATURES), _m3, {}, True),
    Spec("M4_RF", "RandomForest 500 trees (M1 입력)", tuple(M1_FEATURES), _rf,
         {"max_depth": [2, 3], "min_samples_leaf": [3, 5]}, True),
    Spec("M4_GBR", "GradientBoosting lr 0.05·depth 2 (M1 입력)", tuple(M1_FEATURES), _gbr,
         {"n_estimators": [100, 300]}, True),
    # ── 참고 행 (DAY 1 계획에 기록하기로 한 비교값, 선택 대상 아님) ──
    Spec("C0", "상수 예측 = 학습 fold 중앙값(log10 척도)", ("dQ_logvar",), _const, {}, False),
    Spec("M1_huber", "StandardScaler → HuberRegressor(epsilon=1.35) (M1 입력)", tuple(M1_FEATURES), _huber, {}, False),
    Spec("M2_discharge", "M2 discharge 하위 풀 5개 ElasticNet", tuple(M2_DISCHARGE), _m2, EN_GRID, False),
]}
CANDIDATES = ["M0", "M1", "M2", "M3", "M4_RF", "M4_GBR"]   # 단순 → 복잡 순서 (M0 < M1 < M2 < M3 < M4)
REFERENCE = ["C0", "M1_huber", "M2_discharge"]
ALL_IDS = CANDIDATES + REFERENCE


# ════════════════════════════════════════════════════════════════════════
# 데이터 (select/valid 는 Batch 1 만)
# ════════════════════════════════════════════════════════════════════════
def load_batch1() -> pd.DataFrame:
    """select/valid 전용 로더. batch 열만 먼저 읽고 Batch 2/3 행은 파싱 단계에서 건너뛴다
    → Batch 2/3 cycle_life 값은 이 경로에서 한 번도 읽히지 않는다."""
    batch = pd.read_csv(FEATURES_CSV, usecols=["batch"])["batch"].to_numpy()
    skip = [i + 1 for i in np.flatnonzero(batch != "batch1")]           # +1 = 헤더 행
    b1 = pd.read_csv(FEATURES_CSV, skiprows=skip)
    assert (b1["batch"] == "batch1").all() and len(b1) == 46
    return b1.sort_values("idx").reset_index(drop=True)


def load_labeled_batch(batch: str) -> pd.DataFrame:
    """test/batch3 단계 전용: 해당 배치의 라벨 셀(라벨 있음 & 중도절단 아님)."""
    df = pd.read_csv(FEATURES_CSV)
    d = df[(df["batch"] == batch) & df["labeled"]].sort_values("idx").reset_index(drop=True)
    return d


def load_batch_features(batch: str) -> pd.DataFrame:
    """외부 배치 전체 셀의 피처만 (라벨·라벨 플래그 열을 읽지 않는다). Qcc 이동량 분해(라벨 없음)용."""
    cols = [c for c in pd.read_csv(FEATURES_CSV, nrows=0).columns if c not in LABEL_COLS]
    df = pd.read_csv(FEATURES_CSV, usecols=cols)
    return df[df["batch"] == batch].sort_values("idx").reset_index(drop=True)


def load_diagnostics(batch: str) -> pd.DataFrame:
    """test/batch3 단계 전용: 진단 파일(라벨 열 없음, cycle ≤100)에서 한 배치의 정점 사이클."""
    dg = pd.read_csv(DIAG_CSV)
    assert not set(LABEL_COLS) & set(dg.columns), "진단 파일에 라벨 열이 있다"
    assert int(dg["max_cycle_used"].max()) <= LAST, "진단 파일이 cycle 100 이후 정보를 썼다"
    return dg[dg["batch"] == batch].reset_index(drop=True)


def split_batch1(b1: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """results/holdout_cells.csv 의 고정 분할 → (Train 27셀, Hold-out 9셀). 라벨 셀만."""
    ho = pd.read_csv(HOLDOUT_CSV)
    lab = b1[b1["labeled"]]
    H = lab[lab["cell_key"].isin(ho.loc[ho["split"] == "holdout", "cell_key"])].reset_index(drop=True)
    T = lab[lab["cell_key"].isin(ho.loc[ho["split"] == "train", "cell_key"])].reset_index(drop=True)
    assert sorted(H["idx"].tolist()) == HOLDOUT_IDX and H["policy"].nunique() == 5
    assert len(T) == 27 and T["policy"].nunique() == 15 and not set(T["policy"]) & set(H["policy"])
    chk = ho.set_index("cell_key")["cycle_life"]
    assert np.allclose(chk.loc[lab["cell_key"]].to_numpy(), lab["cycle_life"].to_numpy())
    return T, H


def xy(spec: Spec, d: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    return d[list(spec.features)], np.log10(d["cycle_life"].to_numpy(float)), d["policy"].to_numpy()


def predict_life(est, spec: Spec, d: pd.DataFrame) -> np.ndarray:
    return 10 ** est.predict(d[list(spec.features)])


# ════════════════════════════════════════════════════════════════════════
# 적합 (튜닝 포함) / 고정 설정 재적합
# ════════════════════════════════════════════════════════════════════════
def _py(v):
    """JSON 저장용 파이썬 기본형 변환."""
    if isinstance(v, dict):
        return {str(k): _py(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [_py(x) for x in v]
    if isinstance(v, (np.integer,)):
        return int(v)
    if isinstance(v, (np.floating, float)):
        return None if not np.isfinite(v) else float(v)
    if isinstance(v, np.bool_):
        return bool(v)
    return v


def fit_config(spec: Spec, d: pd.DataFrame, cv) -> tuple[object, dict, float]:
    """그리드가 있으면 GridSearchCV(정책 그룹 cv, 원척도 MAPE) 후 전체 재적합, 없으면 바로 적합."""
    X, y, g = xy(spec, d)
    est = spec.make()
    if not spec.grid:
        return est.fit(X, y), {}, float("nan")
    gs = GridSearchCV(est, spec.grid, scoring=MAPE_SCORER, cv=cv, refit=True, n_jobs=1, error_score="raise")
    gs.fit(X, y, groups=g)
    return gs.best_estimator_, _py(gs.best_params_), float(-gs.best_score_)


def refit(spec: Spec, params: dict, d: pd.DataFrame):
    """selection.json 의 고정 하이퍼파라미터로 재적합 (GPR 커널은 fit 안에서 주변우도로 추정)."""
    est = spec.make()
    if params:
        est.set_params(**params)
    X, y, _ = xy(spec, d)
    return est.fit(X, y)


def refit_logged(spec: Spec, params: dict, d: pd.DataFrame) -> tuple[object, dict]:
    """refit + 경고 기록 (GPR 커널 경계 경고 등을 stderr 대신 결과 JSON 에 남긴다)."""
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est = refit(spec, params, d)
    return est, dict(Counter(_warn_key(w) for w in caught))


def _warn_key(w: warnings.WarningMessage) -> str:
    msg = re.sub(r"(?<![\w.])[-+]?\d[\d.e+-]*", "#", str(w.message).split("\n")[0])[:160]
    return f"{w.category.__name__}: {msg}"


# ════════════════════════════════════════════════════════════════════════
# Nested CV (외부 정책 GroupKFold(5)×20, 내부 GroupKFold(4))
# ════════════════════════════════════════════════════════════════════════
def outer_splits(T: pd.DataFrame, seed: int) -> list[tuple[np.ndarray, np.ndarray]]:
    cv = GroupKFold(n_splits=N_OUTER, shuffle=True, random_state=seed)
    return list(cv.split(T, groups=T["policy"].to_numpy()))


def outer_fold(sid: str, seed: int, fold: int, T: pd.DataFrame, tr: np.ndarray, te: np.ndarray) -> dict:
    """외부 fold 1개: 학습부에서 내부 튜닝 → 시험부 예측. 경고는 기록만 한다."""
    spec, Dtr, Dte = SPECS[sid], T.iloc[tr], T.iloc[te]
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est, params, inner = fit_config(spec, Dtr, GroupKFold(n_splits=N_INNER))
        pred = predict_life(est, spec, Dte)
    true = Dte["cycle_life"].to_numpy(float)
    lo, hi = Dtr["dQ_logvar"].min(), Dtr["dQ_logvar"].max()
    dist = np.maximum(0, np.maximum(lo - Dte["dQ_logvar"], Dte["dQ_logvar"] - hi)).to_numpy()
    row = {"model": sid, "role": "candidate" if spec.candidate else "reference", "seed": seed, "fold": fold,
           "n_test": len(te), "test_policies": ";".join(sorted(Dte["policy"].unique())),
           "mape": mape_pct(true, pred), "inner_cv_mape": inner, "params": json.dumps(params, sort_keys=True),
           "n_warnings": len(caught)}
    oof = [{"model": sid, "seed": seed, "fold": fold, "cell_key": k, "policy": p, "true": t, "pred": float(q),
            "ape": abs(q - t) / t * 100, "signed_pct": (q - t) / t * 100,
            "log_resid": float(np.log10(q) - np.log10(t)), "extrap_dist": float(e)}
           for k, p, t, q, e in zip(Dte["cell_key"], Dte["policy"], true, pred, dist)]
    return {"row": row, "oof": oof, "warn": Counter(_warn_key(w) for w in caught)}


def run_nested_cv(T: pd.DataFrame, ids: list[str], n_jobs: int) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    splits = {seed: outer_splits(T, seed) for seed in SEEDS}
    heavy_first = sorted(ids, key=lambda s: s != "M4_RF")                # 부하 분산: RF 를 먼저 배분
    tasks = [(sid, seed, k, tr, te) for sid in heavy_first for seed in SEEDS
             for k, (tr, te) in enumerate(splits[seed])]
    out = Parallel(n_jobs=n_jobs)(delayed(outer_fold)(sid, seed, k, T, tr, te) for sid, seed, k, tr, te in tasks)
    cv = pd.DataFrame([o["row"] for o in out]).sort_values(["model", "seed", "fold"]).reset_index(drop=True)
    oof = pd.DataFrame([r for o in out for r in o["oof"]]).sort_values(["model", "seed", "cell_key"])
    warn: dict[str, Counter] = {}
    for o in out:
        warn.setdefault(o["row"]["model"], Counter()).update(o["warn"])
    return cv, oof.reset_index(drop=True), {k: dict(v.most_common()) for k, v in warn.items()}


def summarize_cv(cv: pd.DataFrame) -> pd.DataFrame:
    """모델별 외부 fold MAPE 평균·SD(100개 fold), SE = 반복별 (5-fold SD/√5) 의 20회 평균."""
    rows = []
    for m, d in cv.groupby("model", sort=False):
        per_seed = d.groupby("seed")["mape"]
        rows.append({"model": m, "role": d["role"].iloc[0], "mean": d["mape"].mean(), "sd": d["mape"].std(ddof=1),
                     "se": float((per_seed.std(ddof=1) / np.sqrt(N_OUTER)).mean()),
                     "sd_of_repeat_means": per_seed.mean().std(ddof=1), "median": d["mape"].median(),
                     "min": d["mape"].min(), "max": d["mape"].max(), "n_folds": len(d)})
    s = pd.DataFrame(rows).set_index("model")
    return s.loc[[m for m in ALL_IDS if m in s.index]]


def oof_focus(oof: pd.DataFrame) -> dict:
    """#20·#21 (B1 하단) out-of-fold 오차: 20회 반복 평균 APE·부호 있는 오차."""
    f = oof[oof["cell_key"].isin(OOF_FOCUS)]
    g = f.groupby(["model", "cell_key"])[["ape", "signed_pct", "pred", "true"]].mean()
    return {m: {k: _py(r.to_dict()) for k, r in d.droplevel(0).iterrows()} for m, d in g.groupby(level=0)}


def pi_quantiles(oof_m: pd.DataFrame) -> dict:
    """예측구간용 로그 잔차(log10 pred − log10 true) 10·90% 분위: 전체 / 외삽 거리 0 / >0."""
    def q(d):
        r = d["log_resid"].to_numpy()
        return {"n": int(len(r)), "q10": float(np.percentile(r, PI_Q[0])) if len(r) else None,
                "q90": float(np.percentile(r, PI_Q[1])) if len(r) else None}
    return {"all": q(oof_m), "in_range": q(oof_m[oof_m["extrap_dist"] == 0]),
            "out_of_range": q(oof_m[oof_m["extrap_dist"] > 0]),
            "how": "구간 = [pred·10^(−q90), pred·10^(−q10)], 잔차 = 정책 단위 nested CV out-of-fold (20회 반복 합산)"}


# ════════════════════════════════════════════════════════════════════════
# 최종 하이퍼파라미터 (GroupKFold(5, seed 42)) · 외삽 게이트
# ════════════════════════════════════════════════════════════════════════
def gate_check(est, spec: Spec, b1: pd.DataFrame) -> dict:
    """B1 중도절단 셀 예측 vs 하한. #00~04 위반 ≥3 → 부적격, #08·10·12·13·22 는 진단."""
    C = b1[b1["censored"]].sort_values("idx")
    pred = predict_life(est, spec, C)
    cells = [{"cell": k, "lower_bound": float(lb), "pred": float(p), "violates": bool(p < lb),
              "rel_to_lb": float(p / lb - 1), "dQ_logvar": float(x)}
             for k, lb, p, x in zip(C["cell_key"], C["cycle_life"], pred, C["dQ_logvar"])]
    gate = [c for c in cells if c["cell"] in GATE_CELLS]
    diag = [c for c in cells if c["cell"] in GATE_DIAG]
    n_v = sum(c["violates"] for c in gate)
    return {"n_violations_00_04": n_v, "eligible": n_v < GATE_MAX_VIOL, "cells_00_04": gate,
            "n_violations_diag": sum(c["violates"] for c in diag), "cells_diag": diag}


def linear_coefs(est, features: tuple) -> dict | None:
    """선형 모델의 원단위 계수(log10 life per feature unit)와 절편. 비선형이면 None."""
    if isinstance(est, LinearRegression):
        return {"intercept": float(est.intercept_), **{f: float(c) for f, c in zip(features, est.coef_)}}
    if isinstance(est, Pipeline) and isinstance(est[-1], (Ridge, ElasticNet, HuberRegressor)):
        sc, m = est.named_steps["scale"], est[-1]
        raw = m.coef_ / sc.scale_
        return {"intercept": float(m.intercept_ - np.sum(m.coef_ * sc.mean_ / sc.scale_)),
                **{f: float(c) for f, c in zip(features, raw)},
                "n_nonzero": int(np.sum(np.abs(m.coef_) > 0))}
    return None


def final_config(spec: Spec, T: pd.DataFrame, b1: pd.DataFrame) -> dict:
    cv = GroupKFold(n_splits=N_FINAL, shuffle=True, random_state=RANDOM_STATE)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        est, params, score = fit_config(spec, T, cv)
        ins = mape_pct(T["cycle_life"], predict_life(est, spec, T))
        gate = gate_check(est, spec, b1)
    out = {"params": params, "final_cv_mape_groupkfold5_seed42": score, "in_sample_mape": ins, "gate": gate,
           "coefs_raw": linear_coefs(est, spec.features), "warnings": dict(Counter(_warn_key(w) for w in caught))}
    if spec.id == "M3":
        out["kernel_learned"] = str(est[-1].kernel_)
        out["log_marginal_likelihood"] = float(est[-1].log_marginal_likelihood_value_)
    return out


# ════════════════════════════════════════════════════════════════════════
# 사전 고정 선택 규칙: ① 최저 평균 ② +1 SE 이내 최단순 ③ 외삽 게이트
# ════════════════════════════════════════════════════════════════════════
def one_se_rule(S: pd.DataFrame, eligible: dict, ids: list[str]) -> dict:
    sub = S.loc[ids]
    best = str(sub["mean"].idxmin())
    thr = float(sub.at[best, "mean"] + sub.at[best, "se"])
    within = [m for m in ids if sub.at[m, "mean"] <= thr]
    ok = [m for m in within if eligible[m]]
    fallback = not ok
    if fallback:                                       # 1-SE 안에 게이트 통과 모델이 없을 때만
        ok = sorted([m for m in ids if eligible[m]], key=lambda m: sub.at[m, "mean"])
    return {"best": best, "best_mean": float(sub.at[best, "mean"]), "best_se": float(sub.at[best, "se"]),
            "threshold": thr, "within_1se": within, "within_1se_eligible": [m for m in within if eligible[m]],
            "selected": ok[0] if ok else None, "fallback_used": fallback}


def apply_rule(S: pd.DataFrame, final: dict) -> dict:
    eligible = {m: final[m]["gate"]["eligible"] for m in CANDIDATES}
    primary = one_se_rule(S, eligible, CANDIDATES)                        # 규칙 문장 순서 그대로
    alt = one_se_rule(S, eligible, [m for m in CANDIDATES if eligible[m]])  # 해석 점검: 게이트 먼저
    sel = primary["selected"]
    why = (f"① 외부 MAPE 평균 최저 = {primary['best']} {primary['best_mean']:.3f}% "
           f"② 임계 = 최저 + 1SE({primary['best_se']:.3f}) = {primary['threshold']:.3f}%, 임계 이내 = "
           f"{primary['within_1se']} ③ 게이트 통과 = {primary['within_1se_eligible']} → 단순성 순서 "
           f"{' < '.join(CANDIDATES)} 에서 첫 모델 = {sel}")
    return {"eligible": eligible, "primary": primary, "alt_gate_first": alt,
            "alt_agrees": alt["selected"] == sel, "selected": sel, "why": why}


# ════════════════════════════════════════════════════════════════════════
# 평가 (Valid / Test / Batch 3 공통)
# ════════════════════════════════════════════════════════════════════════
def pi_bounds(pred: np.ndarray, out_rng: np.ndarray, piq: dict | None) -> tuple[np.ndarray, np.ndarray] | None:
    if not piq:
        return None
    use_out = piq["out_of_range"] if piq["out_of_range"]["n"] >= PI_MIN_N else piq["all"]
    q10 = np.where(out_rng, use_out["q10"], piq["in_range"]["q10"])
    q90 = np.where(out_rng, use_out["q90"], piq["in_range"]["q90"])
    return pred * 10 ** (-q90), pred * 10 ** (-q10)


def interval_stats(true: np.ndarray, lo: np.ndarray, hi: np.ndarray, pred: np.ndarray,
                   out_rng: np.ndarray) -> dict:
    """10~90% 구간의 포함률·폭(사이클, 예측 대비 상대폭) — 전체 / 피처 범위 안 / 밖."""
    cover, rel = (true >= lo) & (true <= hi), (hi - lo) / pred

    def part(m: np.ndarray) -> dict:
        if not m.any():
            return {"n": 0, "coverage": None, "mean_rel_width": None}
        return {"n": int(m.sum()), "coverage": float(cover[m].mean()), "mean_rel_width": float(rel[m].mean())}
    return {"nominal": (PI_Q[1] - PI_Q[0]) / 100, "coverage": float(cover.mean()), "n_covered": int(cover.sum()),
            "mean_width_cycles": float(np.mean(hi - lo)), "mean_rel_width": float(rel.mean()),
            "in_range": part(~out_rng), "out_of_range": part(out_rng)}


def gpr_interval(est, spec: Spec, d: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """M3 전용: GPR predict(return_std=True) 의 log10 척도 μ ± z·σ (z = Φ⁻¹(0.9)) → 10^(·) 로 되돌린 10~90% 구간.
    σ 는 WhiteKernel 잡음을 포함한 예측 표준편차(normalize_y=True 라 log10 단위). GPR 이 아니면 None."""
    if not (isinstance(est, Pipeline) and isinstance(est[-1], GaussianProcessRegressor)):
        return None
    mu, sd = est.predict(d[list(spec.features)], return_std=True)
    return 10 ** (mu - PI_Z * sd), 10 ** (mu + PI_Z * sd), sd


def evaluate(est, spec: Spec, fit_set: pd.DataFrame, d: pd.DataFrame, piq: dict | None = None) -> dict:
    """MAPE·부트스트랩 CI·셀별 APE·부호 있는 오차·과대예측 비율·피처 범위 안/밖·노션 수명 범위 밖 예측
    + 예측구간: CV 잔차 분위 구간(전 모델), GPR 표준편차 구간(M3 만, CV 잔차 구간과 나란히 비교)."""
    pred, true = predict_life(est, spec, d), d["cycle_life"].to_numpy(float)
    ape, signed = np.abs(pred - true) / true * 100, (pred - true) / true * 100
    lo, hi = fit_set["dQ_logvar"].min(), fit_set["dQ_logvar"].max()
    out_rng = ((d["dQ_logvar"] < lo) | (d["dQ_logvar"] > hi)).to_numpy()
    cells = pd.DataFrame({"cell_key": d["cell_key"], "group": d["group"], "policy": d["policy"], "true": true,
                          "pred": pred, "ape": ape, "signed_pct": signed, "dQ_logvar": d["dQ_logvar"],
                          "out_of_train_range": out_rng})
    res = {"n": int(len(d)), "mape": float(ape.mean()), "ci95": bootstrap_ci(ape),
           "rmse_cycles": float(np.sqrt(np.mean((pred - true) ** 2))), "signed_mean_pct": float(signed.mean()),
           "over_ratio": float(np.mean(pred > true)), "n_over": int(np.sum(pred > true)),
           "n_out_of_range": int(out_rng.sum()),
           "mape_in_range": float(ape[~out_rng].mean()) if (~out_rng).any() else None,
           "mape_out_of_range": float(ape[out_rng].mean()) if out_rng.any() else None,
           "n_pred_outside_150_2300": int(np.sum((pred < LIFE_RANGE_NOTION[0]) | (pred > LIFE_RANGE_NOTION[1]))),
           "train_dQ_logvar_range": [float(lo), float(hi)]}
    b = pi_bounds(pred, out_rng, piq)
    if b is not None:
        cells["pi_lo"], cells["pi_hi"] = b
        res["pi_coverage_10_90"] = float(np.mean((true >= b[0]) & (true <= b[1])))
        res["pi_cv_resid_10_90"] = interval_stats(true, b[0], b[1], pred, out_rng)
    g = gpr_interval(est, spec, d)
    if g is not None:
        cells["gpr_lo"], cells["gpr_hi"], cells["gpr_sd_log10"] = g
        res["pi_gpr_10_90"] = {**interval_stats(true, g[0], g[1], pred, out_rng),
                               "mean_sd_log10_in_range": float(g[2][~out_rng].mean()) if (~out_rng).any() else None,
                               "mean_sd_log10_out_of_range": float(g[2][out_rng].mean()) if out_rng.any() else None,
                               "how": f"GPR predict(return_std=True): 10^(μ ± {PI_Z:.4f}·σ), σ 는 log10 단위 "
                                      "(WhiteKernel 잡음 포함). 같은 셀의 CV 잔차 구간 = pi_cv_resid_10_90"}
    res["cells"] = _py(cells.to_dict(orient="records"))
    return res


def by_group(ev: dict) -> dict:
    """그룹별 MAPE·부호 있는 오차·과대예측 비율·피처 범위 안/밖 MAPE."""
    c = pd.DataFrame(ev["cells"])
    out = {}
    for g, d in c.groupby("group"):
        o = d["out_of_train_range"].astype(bool)
        out[g] = {"n": int(len(d)), "mape": float(d["ape"].mean()), "signed_mean_pct": float(d["signed_pct"].mean()),
                  "over_ratio": float(np.mean(d["pred"] > d["true"])), "n_over": int(np.sum(d["pred"] > d["true"])),
                  "n_out_of_range": int(o.sum()),
                  "mape_in_range": float(d.loc[~o, "ape"].mean()) if (~o).any() else None,
                  "mape_out_of_range": float(d.loc[o, "ape"].mean()) if o.any() else None}
    return out


# ════════════════════════════════════════════════════════════════════════
# 기록·guard
# ════════════════════════════════════════════════════════════════════════
def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def run_log(stage: str, msg: str) -> None:
    RES.mkdir(exist_ok=True)
    with open(RUN_LOG, "a", encoding="utf-8") as fh:
        fh.write(f"{datetime.now().isoformat(timespec='seconds')}\t{stage}\t{msg}\n")


def write_json(path: Path, obj: dict) -> None:
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(_py(obj), fh, ensure_ascii=False, indent=2)


def read_json(path: Path) -> dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def guard_once(stage: str, marker: Path, force: bool) -> None:
    """1회 평가 단계 보호: selection.json 필수, 완료 표식이 있으면 --force 없이는 거부(--force 는 기록)."""
    if not SELECTION_JSON.exists():
        raise SystemExit(f"[{stage}] results/selection.json 이 없다 — select 단계를 먼저 실행해야 한다.")
    if marker.exists():
        if not force:
            raise SystemExit(f"[{stage}] {marker.name} 존재 — 이미 1회 실행됨. 재실행은 --force (run_log 에 기록).")
        run_log(stage, f"FORCE: {marker.name} 존재하지만 --force 로 재실행")
        print(f"[{stage}] 경고: --force 로 재실행 (results/run_log.txt 에 기록)")


def mark_opened(marker: Path, force: bool) -> None:
    """라벨을 읽자마자 표식을 남긴다 → 평가 도중 실패해도 재실행은 --force(기록) 로만 가능.
    덧붙이기('a') 모드: --force 재실행에서도 첫 실행의 opened/completed 줄이 그대로 남는다."""
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write(f"opened\t{datetime.now().isoformat(timespec='seconds')}\t"
                 f"selection_sha256={sha256(SELECTION_JSON)}\tforced={force}\n")


def mark_done(marker: Path, out: dict) -> None:
    with open(marker, "a", encoding="utf-8") as fh:
        fh.write(f"completed\t{out['created']}\tselection_sha256={out['selection_sha256']}\tforced={out['forced']}\n")


def guard_select(force: bool) -> None:
    """테스트 후 재선택은 테스트 누수 → .test_done 또는 .batch3_done 이 있으면 --force 없이는 거부."""
    found = [m.name for m in (TEST_DONE, BATCH3_DONE) if m.exists()]
    if found:
        if not force:
            raise SystemExit(f"[select] {', '.join(found)} 존재 — 테스트 후 모델 재선택 금지. 재실행은 --force (기록됨).")
        run_log("select", f"FORCE: {', '.join(found)} 존재하지만 --force 로 재선택")
        print(f"[select] 경고: --force 로 재선택 ({', '.join(found)} 존재, results/run_log.txt 에 기록)")


# ════════════════════════════════════════════════════════════════════════
# 잡음 셀 ΔQ 21점 이동중앙값 민감도 (DAY 1 §10, 기록 전용 — 선택·Valid 본 값은 바뀌지 않는다)
# ════════════════════════════════════════════════════════════════════════
def with_smoothed_dq(d: pd.DataFrame) -> pd.DataFrame:
    """B1 잡음 셀(#09·38~45)만 dQ_logvar 를 21점 이동중앙값 판(dQ_logvar_med21)으로 바꾼 사본."""
    out = d.copy()
    m = out["cell_key"].isin(NOISY_B1)
    out.loc[m, "dQ_logvar"] = out.loc[m, SENS_DQ]
    return out


def rank_check_med21(b1: pd.DataFrame, H: pd.DataFrame) -> dict:
    """DAY 1 계획의 점검: 잡음 셀을 평활화해도 B1 라벨 36셀의 dQ_logvar 순위가 유지되는가."""
    lab = b1[b1["labeled"]].reset_index(drop=True)
    sm = with_smoothed_dq(lab)
    r0, r1 = lab["dQ_logvar"].rank(), sm["dQ_logvar"].rank()
    m = lab["cell_key"].isin(NOISY_B1)
    cells = [{"cell": k, "split": "holdout" if k in set(H["cell_key"]) else "train", "dQ_logvar": float(a),
              "dQ_logvar_med21": float(b), "diff": float(b - a), "rank36_raw": int(x), "rank36_med21": int(y)}
             for k, a, b, x, y in zip(lab.loc[m, "cell_key"], lab.loc[m, "dQ_logvar"], sm.loc[m, "dQ_logvar"],
                                      r0[m], r1[m])]
    return {"spearman_raw_vs_med21_labeled36": float(stats.spearmanr(lab["dQ_logvar"], sm["dQ_logvar"])[0]),
            "n_rank_changed_labeled36": int((r0 != r1).sum()), "max_abs_diff": float(max(abs(c["diff"]) for c in cells)),
            "cells": cells}


def sensitivity_med21_select(T: pd.DataFrame, H: pd.DataFrame, b1: pd.DataFrame, S: pd.DataFrame, final: dict,
                             sid: str, n_jobs: int) -> tuple[dict, pd.DataFrame]:
    """선택 모델·M0 를 같은 nested CV(같은 seed·분할)와 최종 GroupKFold(5, seed 42)로 평활 피처에서 다시 잰다."""
    ids = list(dict.fromkeys([sid, "M0"]))
    Ts, b1s = with_smoothed_dq(T), with_smoothed_dq(b1)
    cv, _, _ = run_nested_cv(Ts, ids, n_jobs)
    Ss = summarize_cv(cv)
    models = {}
    for m in ids:
        f = final_config(SPECS[m], Ts, b1s)
        models[m] = {"cv_mean": float(Ss.at[m, "mean"]), "cv_sd": float(Ss.at[m, "sd"]), "cv_se": float(Ss.at[m, "se"]),
                     "cv_mean_primary": float(S.at[m, "mean"]), "cv_mean_diff": float(Ss.at[m, "mean"] - S.at[m, "mean"]),
                     "final_params": f["params"], "final_params_primary": final[m]["params"],
                     "final_params_same": f["params"] == final[m]["params"],
                     "gate_violations_00_04": f["gate"]["n_violations_00_04"], "coefs_raw": f["coefs_raw"]}
    out = {"record_only": True, "noisy_cells": NOISY_B1, "feature": SENS_DQ,
           "definition": "ΔQ₁₀₀₋₁₀(V) 21점 이동중앙값(center, 끝단 min_periods=1) 후 log10 var(ddof=0). 잡음 셀만 대체",
           "rank_check": rank_check_med21(b1, H), "models": models,
           "note": "선택 규칙·selected·Valid 본 값은 원정의(평활화 없음)로만 정한다. 이 블록은 DAY 1 §10 민감도 기록이다."}
    cv.insert(0, "sensitivity", "dq_med21_noisy_b1")
    return out, cv


def sensitivity_med21_valid(sel: dict, T: pd.DataFrame, H: pd.DataFrame, primary: dict) -> dict:
    """select 단계 민감도 설정(평활 피처로 정한 하이퍼파라미터)으로 Train 27셀 재적합 → Hold-out 9셀 1회 (기록 전용)."""
    sens = sel["sensitivity_dq_med21"]
    Ts, Hs = with_smoothed_dq(T), with_smoothed_dq(H)
    true = Hs["cycle_life"].to_numpy(float)
    noisy_ho = Hs["cell_key"].isin(NOISY_B1).to_numpy()
    out = {}
    for m, info in sens["models"].items():
        spec = SPECS[m]
        pred = predict_life(refit(spec, info["final_params"], Ts), spec, Hs)
        ape = np.abs(pred - true) / true * 100
        prim = {c["cell_key"]: c["ape"] for c in primary[m]["cells"]}
        out[m] = {"valid_mape": float(ape.mean()), "ci95": bootstrap_ci(ape), "valid_mape_primary": primary[m]["mape"],
                  "valid_mape_diff": float(ape.mean() - primary[m]["mape"]),
                  "train_cv_mean": info["cv_mean"], "gap_train_valid": float(ape.mean() - info["cv_mean"]),
                  "noisy_holdout_cells": [{"cell": k, "ape_med21": float(a), "ape_primary": float(prim[k])}
                                          for k, a in zip(Hs.loc[noisy_ho, "cell_key"], ape[noisy_ho])]}
    return {"record_only": True, "models": out,
            "note": "잡음 셀 dQ_logvar 만 21점 이동중앙값 판으로 바꾼 기록용 민감도. 선택·Valid 본 값은 바뀌지 않는다."}


# ════════════════════════════════════════════════════════════════════════
# STAGE: select
# ════════════════════════════════════════════════════════════════════════
def print_cv_table(S: pd.DataFrame, final: dict, rule: dict) -> None:
    print(f"\n{'model':<13}{'mean':>8}{'±SD':>8}{'SE':>7}{'in-smp':>8}{'gate':>6}  final params")
    for m, r in S.iterrows():
        f = final[m]
        mark = "  ← 선택" if m == rule["selected"] else ""
        print(f"{m:<13}{r['mean']:8.3f}{r['sd']:8.3f}{r['se']:7.3f}{f['in_sample_mape']:8.2f}"
              f"{f['gate']['n_violations_00_04']:>4}/5  {f['params']}{mark}")
    print("\n" + rule["why"])


def stage_select(n_jobs: int, force: bool) -> dict:
    guard_select(force)
    t0 = time.time()
    b1 = load_batch1()
    T, H = split_batch1(b1)
    cv, oof, warn = run_nested_cv(T, ALL_IDS, n_jobs)
    t_cv = time.time() - t0
    S = summarize_cv(cv)
    final = {m: final_config(SPECS[m], T, b1) for m in ALL_IDS}
    rule = apply_rule(S, final)
    sel = rule["selected"]
    sens, sens_cv = sensitivity_med21_select(T, H, b1, S, final, sel, n_jobs)   # 기록 전용 (선택 확정 후)
    inner_freq = {m: dict(Counter(d["params"]).most_common(5)) for m, d in cv.groupby("model")}
    out = {
        "stage": "select", "created": datetime.now().isoformat(timespec="seconds"),
        "features_csv_sha256": sha256(FEATURES_CSV), "holdout_csv_sha256": sha256(HOLDOUT_CSV),
        "settings": {"target": "log10(cycle_life), MAPE on 10^ŷ (%)", "seeds": SEEDS, "outer": "GroupKFold(5, shuffle=True, random_state=seed) by policy",
                     "inner": "GroupKFold(4) by policy (no shuffle), scorer = 원척도 MAPE",
                     "final_hyperparameters": "GroupKFold(5, shuffle=True, random_state=42) by policy on Train 27",
                     "se": "mean over 20 repeats of (5-fold SD[ddof=1] / sqrt(5))",
                     "sd": "SD[ddof=1] over all 100 outer-fold MAPEs", "random_state": RANDOM_STATE,
                     "gate": f"#00~04 하한 위반 ≥{GATE_MAX_VIOL}/5 → 부적격 (#08·10·12·13·22 진단)",
                     "order": " < ".join(CANDIDATES), "grids": {m: SPECS[m].grid for m in ALL_IDS},
                     "features": {m: list(SPECS[m].features) for m in ALL_IDS},
                     "names": {m: SPECS[m].name for m in ALL_IDS}},
        "data": {"train_cells": T["cell_key"].tolist(), "holdout_cells": H["cell_key"].tolist(),
                 "n_train": len(T), "n_train_policies": int(T["policy"].nunique()),
                 "train_life_range": [float(T["cycle_life"].min()), float(T["cycle_life"].max())],
                 "train_dQ_logvar_range": [float(T["dQ_logvar"].min()), float(T["dQ_logvar"].max())]},
        "cv_summary": S.reset_index().to_dict(orient="records"),
        "inner_params_top5": inner_freq, "oof_focus_20_21": oof_focus(oof),
        "final": final, "rule": rule, "selected": {"id": sel, "name": SPECS[sel].name,
                                                    "features": list(SPECS[sel].features),
                                                    "params": final[sel]["params"]},
        "pi_quantiles": {m: pi_quantiles(oof[oof["model"] == m]) for m in ALL_IDS},
        "sensitivity_dq_med21": sens,
        "cv_warnings": warn, "runtime_sec": {"nested_cv": t_cv, "total": time.time() - t0}, "n_jobs": n_jobs,
    }
    cv.to_csv(RES / "cv_results.csv", index=False)
    sens_cv.to_csv(RES / "sensitivity_dq_med21_cv.csv", index=False)
    oof.to_csv(RES / "cv_oof.csv", index=False)
    write_json(SELECTION_JSON, out)
    run_log("select", f"selected={sel} params={final[sel]['params']} sha={sha256(SELECTION_JSON)[:12]} "
                      f"runtime={time.time() - t0:.0f}s")
    print_cv_table(S, final, rule)
    print(f"\nruntime: nested CV {t_cv:.0f}s, total {time.time() - t0:.0f}s → results/selection.json")
    return out


# ════════════════════════════════════════════════════════════════════════
# STAGE: valid
# ════════════════════════════════════════════════════════════════════════
def stage_valid() -> dict:
    if not SELECTION_JSON.exists():
        raise SystemExit("[valid] results/selection.json 이 없다 — select 를 먼저 실행해야 한다.")
    sel = read_json(SELECTION_JSON)
    assert sel["features_csv_sha256"] == sha256(FEATURES_CSV), "features.csv 가 select 이후 바뀌었다"
    b1 = load_batch1()
    T, H = split_batch1(b1)
    cvm = {r["model"]: r for r in sel["cv_summary"]}
    res = {}
    for m in ALL_IDS:
        spec, params = SPECS[m], sel["final"][m]["params"]
        est, fit_warn = refit_logged(spec, params, T)
        _check_same_as_select(est, spec, b1, sel["final"][m]["gate"])
        ev = evaluate(est, spec, T, H, sel["pi_quantiles"][m])
        ev["fit_warnings"] = fit_warn
        ev["train_cv_mean"] = cvm[m]["mean"]
        ev["gap_train_valid"] = ev["mape"] - cvm[m]["mean"]                    # Valid − Train
        ev["train_cv_below_valid_ci"] = bool(cvm[m]["mean"] < ev["ci95"][0])   # H4: CI 를 넘는 (+) Gap
        res[m] = ev
    sid = sel["selected"]["id"]
    out = {"stage": "valid", "created": datetime.now().isoformat(timespec="seconds"),
           "selection_sha256": sha256(SELECTION_JSON), "selected_id": sid, "selected": res[sid],
           "M0": res["M0"], "all_models": {m: {k: v for k, v in res[m].items() if k != "cells"} for m in ALL_IDS},
           "interval_comparison_M3": interval_comparison(res["M3"]),
           "sensitivity_dq_med21": sensitivity_med21_valid(sel, T, H, res),
           "note": "선택 설정을 Train 27셀로 적합해 Hold-out 9셀을 1회 예측. 다른 후보 값은 기록용이며 선택을 바꾸지 않는다."}
    write_json(VALID_JSON, out)
    _save_predictions(res, RES / "valid_predictions.csv")                    # 전 모델 셀별 예측·구간 (M3 GPR 구간 포함)
    run_log("valid", f"selected={sid} valid_mape={res[sid]['mape']:.3f} M0={res['M0']['mape']:.3f} "
                     f"selection_sha={out['selection_sha256'][:12]}")
    _print_valid(res, sid, out)
    return out


def interval_comparison(ev: dict) -> dict:
    """M3: GPR 표준편차 구간 vs CV 잔차 분위 구간 (같은 셀, 같은 10~90% 명목) — DAY 1 §10 M3 '불확실성 보조'."""
    keys = ("nominal", "coverage", "n_covered", "mean_width_cycles", "mean_rel_width", "in_range", "out_of_range")
    cols = ["cell_key", "true", "pred", "pi_lo", "pi_hi", "gpr_lo", "gpr_hi", "gpr_sd_log10", "out_of_train_range"]
    return {"n": ev["n"], "gpr_std": {k: ev["pi_gpr_10_90"][k] for k in keys},
            "cv_resid": {k: ev["pi_cv_resid_10_90"][k] for k in keys} if "pi_cv_resid_10_90" in ev else None,
            "gpr_mean_sd_log10_in_out": [ev["pi_gpr_10_90"]["mean_sd_log10_in_range"],
                                         ev["pi_gpr_10_90"]["mean_sd_log10_out_of_range"]],
            "cells": [{k: c.get(k) for k in cols} for c in ev["cells"]]}


def _check_same_as_select(est, spec: Spec, b1: pd.DataFrame, gate: dict) -> None:
    """재적합 모델이 select 단계 최종 모델과 같은지(중도절단 셀 예측 일치) 확인."""
    C = b1[b1["censored"]].sort_values("idx")
    pred = predict_life(est, spec, C)
    ref = {c["cell"]: c["pred"] for c in gate["cells_00_04"] + gate["cells_diag"]}
    assert np.allclose(pred, [ref[k] for k in C["cell_key"]], rtol=1e-8), f"{spec.id}: select 최종 모델과 불일치"


def _print_valid(res: dict, sid: str, out: dict) -> None:
    print(f"\n{'model':<13}{'Valid':>8}{'95% CI':>18}{'Train CV':>10}{'Gap(T-V)':>10}")
    for m, r in res.items():
        mark = "  ← 선택" if m == sid else ""
        print(f"{m:<13}{r['mape']:8.2f}   [{r['ci95'][0]:5.2f}, {r['ci95'][1]:5.2f}]{r['train_cv_mean']:10.2f}"
              f"{r['gap_train_valid']:+10.2f}{mark}")
    print("\n셀별 APE (선택 모델):")
    print(pd.DataFrame(res[sid]["cells"])[["cell_key", "policy", "true", "pred", "ape", "signed_pct"]]
          .round(1).to_string(index=False))
    ic = out["interval_comparison_M3"]
    for k in ("gpr_std", "cv_resid"):
        if ic[k]:
            print(f"M3 10~90% 구간 [{k}]: 포함 {ic[k]['n_covered']}/{ic['n']}, 평균 폭 {ic[k]['mean_width_cycles']:.0f} 사이클 "
                  f"(예측 대비 {ic[k]['mean_rel_width'] * 100:.1f}%)")
    for m, r in out["sensitivity_dq_med21"]["models"].items():
        print(f"민감도 21점 중앙값 [{m}]: Train CV {r['train_cv_mean']:.3f}, Valid {r['valid_mape']:.3f} "
              f"(원정의 {r['valid_mape_primary']:.3f}, 차 {r['valid_mape_diff']:+.3f})")


# ════════════════════════════════════════════════════════════════════════
# STAGE: test (Batch 2, 1회) / batch3 (선택, 1회)
# ════════════════════════════════════════════════════════════════════════
def qcc_shift_decomposition(coefs: dict | None, b1: pd.DataFrame, d_all: pd.DataFrame, ev: dict) -> dict | None:
    """노션 Qdlin 경고 대응 (c): Test 오차를 '계수 × 라벨 없는 Qcc_init 이동량'으로 분해 (DAY 1 q5d 와 같은 정의).
    이동량 = 외부 배치 그룹 '전체 셀'(라벨 무관) Qcc_init 중앙값 − B1 46셀 중앙값, SD 단위 = B1 라벨 36셀 SD(ddof=1).
    Qcc 몫(log10) = 계수 × 이동량 → 예측 배율 10^(·). 관측 평균 로그 잔차(라벨 셀) − Qcc 몫 = 나머지."""
    if not coefs or "Qcc_init" not in coefs:
        return None
    k, ref = coefs["Qcc_init"], float(b1["Qcc_init"].median())
    lab = b1[b1["labeled"]]
    sd, lo, hi = float(lab["Qcc_init"].std(ddof=1)), float(lab["Qcc_init"].min()), float(lab["Qcc_init"].max())
    c = pd.DataFrame(ev["cells"])
    c["log_resid"] = np.log10(c["pred"]) - np.log10(c["true"])
    out = {}
    for g, dl in c.groupby("group"):
        a = d_all[d_all["group"] == g]
        sh, obs = float(a["Qcc_init"].median() - ref), float(dl["log_resid"].mean())
        out[g] = {"n_all_cells": int(len(a)), "n_labeled_eval": int(len(dl)), "shift_mAh": sh * 1000,
                  "shift_in_b1_sd": sh / sd, "n_out_of_b1_labeled_range": int(((a["Qcc_init"] < lo) | (a["Qcc_init"] > hi)).sum()),
                  "qcc_part_log10": k * sh, "pred_factor": float(10 ** (k * sh)),
                  "observed_mean_log_resid": obs, "observed_factor": float(10 ** obs), "remainder_log10": obs - k * sh}
    return {"b1_all46_median_Ah": ref, "b1_labeled36_sd_mAh": sd * 1000, "coef_log10_per_Ah": k, "groups": out,
            "how": "이동량은 그룹 전체 셀(라벨 미사용) 기준. 관측 잔차만 라벨 셀(평가 셀)에서 계산"}


def _spearman(r: pd.Series, x: pd.Series) -> tuple[float | None, float | None]:
    """Spearman (ρ, p). 셀 < 3 이거나 한쪽이 상수면 (None, None)."""
    if len(r) < 3 or r.nunique() < 2 or x.nunique() < 2:
        return None, None
    rho, p = stats.spearmanr(r, x)
    return float(rho), float(p)


def _breakin_block(c: pd.DataFrame) -> dict:
    """셀 묶음 하나: 로그 잔차 vs 상승폭·정점 사이클(주·보조) Spearman + 정점 중앙값·창 끝 정점 셀 수."""
    out = {"n": int(len(c))}
    for col in BREAKIN_COLS:
        out[f"rho_{col}"], out[f"p_{col}"] = _spearman(c["log_resid"], c[col])
    for col in DIAG_FEATURES:
        out[f"median_{col}"] = float(c[col].median())
        out[f"n_ge{PEAK_EDGE}_{col}"] = int((c[col] >= PEAK_EDGE).sum())
    return out


def breakin_check(ev: dict, d: pd.DataFrame, diag: pd.DataFrame) -> dict:
    """H1 사전 지정 점검: 로그 잔차(log10 pred − log10 true, + = 과대예측) vs CC break-in 상승폭 · 정점 사이클.
    정점 사이클(주) = cc_breakin 과 같은 끝점 9-사이클 이동중앙값의 최댓값 사이클, (보조) = 요약 QD 정점 (cycle 2~100)."""
    c = (pd.DataFrame(ev["cells"]).merge(d[["cell_key", "cc_breakin"]], on="cell_key", validate="one_to_one")
         .merge(diag[["cell_key", *DIAG_FEATURES]], on="cell_key", how="left", validate="one_to_one"))
    assert len(c) == len(ev["cells"]) and c[DIAG_FEATURES].notna().all().all(), "진단 파일에 평가 셀이 빠졌다"
    c["log_resid"] = np.log10(c["pred"]) - np.log10(c["true"])
    a = _breakin_block(c)
    return {"spearman_logresid_vs_cc_breakin": a["rho_cc_breakin"], "p": a["p_cc_breakin"], "n": a["n"],
            "spearman_logresid_vs_peak_cycle": a["rho_cc_breakin_peak_cycle"], "p_peak_cycle": a["p_cc_breakin_peak_cycle"],
            "overall": a, "by_group": {g: _breakin_block(x) for g, x in c.groupby("group")},
            "how": "Spearman(로그 잔차, x). x = cc_breakin(상승폭, mAh) · cc_breakin_peak_cycle(주: CC 끝점 9-사이클 "
                   "이동중앙값 최댓값 사이클) · qd_breakin_peak_cycle(보조: DAY 1 q2 요약 QD 정점 정의를 cycle 2~100 으로 자름). "
                   f"정점은 results/diagnostic_features.csv (라벨 없음). n_ge{PEAK_EDGE}_* = 창 끝 절단 가능 셀 수."}


def evaluate_external(sel: dict, b1: pd.DataFrame, d: pd.DataFrame, d_all: pd.DataFrame, diag: pd.DataFrame) -> dict:
    """B1 라벨 36셀로 고정 설정 재적합 → 외부 배치 라벨 셀 d 평가 (모든 후보·참고 행).
    d_all = 같은 배치 전체 셀의 라벨 없는 피처 (Qcc 이동량 분해용), diag = 같은 배치 진단 파일 행 (H1 정점 사이클)."""
    fit36 = b1[b1["labeled"]].reset_index(drop=True)
    assert len(fit36) == 36
    res = {}
    for m in ALL_IDS:
        spec = SPECS[m]
        est, fit_warn = refit_logged(spec, sel["final"][m]["params"], fit36)
        ev = evaluate(est, spec, fit36, d, sel["pi_quantiles"][m])
        ev["fit_warnings"] = fit_warn
        ev["by_group"] = by_group(ev)
        ev["coefs_raw_fit36"] = linear_coefs(est, spec.features)
        ev["qcc_shift_decomposition"] = qcc_shift_decomposition(ev["coefs_raw_fit36"], b1, d_all, ev)
        ev["breakin_check"] = breakin_check(ev, d, diag)
        res[m] = ev
    return res


def _external_out(stage: str, sel: dict, res: dict, forced: bool, extra: dict) -> dict:
    sid = sel["selected"]["id"]
    return {"stage": stage, "created": datetime.now().isoformat(timespec="seconds"), "forced": forced,
            "selection_sha256": sha256(SELECTION_JSON), "features_csv_sha256": sha256(FEATURES_CSV),
            "diagnostic_features_sha256": sha256(DIAG_CSV), "selected_id": sid, "selected": res[sid], "M0": res["M0"],
            "appendix_other_models": {m: {k: v for k, v in res[m].items() if k != "cells"} for m in ALL_IDS},
            "interval_comparison_M3": interval_comparison(res["M3"]),
            "note": "피처·하이퍼파라미터 고정, B1 라벨 36셀 재적합 후 1회 평가. 다른 후보 값은 사후 분석 부록이며 선택을 바꾸지 않는다.",
            **extra}


def _save_predictions(res: dict, path: Path) -> None:
    rows = [dict(r, model=m) for m in ALL_IDS for r in res[m]["cells"]]
    pd.DataFrame(rows).to_csv(path, index=False)


def stage_test(force: bool) -> dict:
    guard_once("test", TEST_DONE, force)
    sel = read_json(SELECTION_JSON)
    assert sel["features_csv_sha256"] == sha256(FEATURES_CSV), "features.csv 가 select 이후 바뀌었다"
    b1 = load_batch1()
    B2_all = load_batch_features("batch2")                # 라벨 없는 전체 47셀 (Qcc 이동량 분해)
    B2_diag = load_diagnostics("batch2")                  # 라벨 없는 정점 사이클 (H1 점검) — 라벨을 열기 전에 확인
    B2 = load_labeled_batch("batch2")                     # 여기서 처음으로 Batch 2 라벨을 읽는다
    mark_opened(TEST_DONE, force)                          # 라벨을 연 순간 1회 소진 (중간 실패도 재실행엔 --force)
    assert len(B2) == 39, f"Batch 2 라벨 셀 수 {len(B2)} ≠ 39"
    res = evaluate_external(sel, b1, B2, B2_all, B2_diag)
    focus = [c for c in res[sel["selected"]["id"]]["cells"] if c["cell_key"] == "batch2-43"]
    out = _external_out("test", sel, res, force, {"batch": "batch2", "cell_43_check": focus})
    write_json(TEST_JSON, out)
    _save_predictions(res, RES / "test_predictions.csv")
    mark_done(TEST_DONE, out)
    run_log("test", f"selected={out['selected_id']} test_mape={res[out['selected_id']]['mape']:.3f} "
                    f"M0={res['M0']['mape']:.3f} forced={force}")
    print(f"Test (Batch 2) MAPE: {out['selected_id']} {res[out['selected_id']]['mape']:.2f}% | M0 {res['M0']['mape']:.2f}%")
    print(_h1_note(res[out["selected_id"]]["breakin_check"]))
    return out


def stage_batch3(force: bool) -> dict:
    guard_once("batch3", BATCH3_DONE, force)
    sel = read_json(SELECTION_JSON)
    assert sel["features_csv_sha256"] == sha256(FEATURES_CSV), "features.csv 가 select 이후 바뀌었다"
    b1 = load_batch1()
    B3_all = load_batch_features("batch3")                # 라벨 없는 전체 46셀 (Qcc 이동량 분해)
    B3_diag = load_diagnostics("batch3")                  # 라벨 없는 정점 사이클 (H1 점검)
    B3 = load_labeled_batch("batch3")                     # Batch 3 라벨은 이 단계에서만 읽는다
    mark_opened(BATCH3_DONE, force)
    assert len(B3) == 44, f"Batch 3 라벨 셀 수 {len(B3)} ≠ 44"
    res_all = evaluate_external(sel, b1, B3, B3_all, B3_diag)
    B3x = B3[~B3["cell_key"].isin(B3_PAPER_REMOVED)].reset_index(drop=True)
    res_x = evaluate_external(sel, b1, B3x, B3_all[~B3_all["cell_key"].isin(B3_PAPER_REMOVED_ALL)], B3_diag)
    extra = {"batch": "batch3", "paper_removed_cells": B3_PAPER_REMOVED,
             "excluding_paper_removed": {"selected": {k: v for k, v in res_x[sel["selected"]["id"]].items() if k != "cells"},
                                         "M0": {k: v for k, v in res_x["M0"].items() if k != "cells"},
                                         "n": int(len(B3x))}}
    out = _external_out("batch3", sel, res_all, force, extra)
    write_json(BATCH3_JSON, out)
    _save_predictions(res_all, RES / "batch3_predictions.csv")
    mark_done(BATCH3_DONE, out)
    sid = out["selected_id"]
    run_log("batch3", f"selected={sid} b3_mape={res_all[sid]['mape']:.3f} excl4={res_x[sid]['mape']:.3f} forced={force}")
    print(f"Test (Batch 3) MAPE: {sid} {res_all[sid]['mape']:.2f}% (원저자 코드 제거 4셀 제외 {res_x[sid]['mape']:.2f}%)")
    return out


# ════════════════════════════════════════════════════════════════════════
# 사후 정정 (2026-10-02): Batch 3 '원저자 제거 셀' 제외 결과를 저장된 예측으로 다시 계산
# ════════════════════════════════════════════════════════════════════════
# 2026-10-01 batch3 실행은 제외 목록을 #02·37·38·39 로 잘못 썼다.
# 원저자 공개 코드(Load Data.ipynb: del b3c2·b3c23·b3c32·b3c37·b3c42·b3c43, 키 b3cN 의 N = .mat 순서 = 우리 idx;
# LoadData.m 도 같은 6셀)를 확인해 B3_PAPER_REMOVED 를 #02·37·42·43 으로 바로잡았다.
# batch3 단계는 다시 실행하지 않는다(1회 원칙). 같은 고정 모델의 셀별 예측이 results/batch3_predictions.csv 에
# 있으므로, 부분집합 지표는 그 파일에서 다시 계산하면 된다(재적합·라벨 재로딩 없음).
# 방법 검증: 잘못된 목록으로 같은 계산을 하면 batch3.json['excluding_paper_removed'] 가 그대로 재현되는지 assert 한다.
B3_EXCL_JSON = RES / "batch3_excl_corrected.json"
B3_PAPER_REMOVED_OLD = ["batch3-02", "batch3-37", "batch3-38", "batch3-39"]   # 2026-10-01 실행에 쓴 잘못된 목록
B3_EXCL_CHECK_KEYS = ("n", "mape", "ci95", "rmse_cycles", "signed_mean_pct", "over_ratio", "n_over", "n_out_of_range",
                      "mape_in_range", "mape_out_of_range", "pi_coverage_10_90", "n_pred_outside_150_2300")
_B3X_CACHE: dict = {}


def _ev_from_saved(P: pd.DataFrame, m: str, keep: list[str], b3: dict, b1: pd.DataFrame,
                   d_all: pd.DataFrame, d_feat: pd.DataFrame, diag: pd.DataFrame) -> dict:
    """저장된 셀별 예측(파일 순서 = 평가 순서)에서 evaluate() 와 같은 요약 지표를 다시 만든다."""
    d = P[(P["model"] == m) & P["cell_key"].isin(keep)].reset_index(drop=True)
    true, pred = d["true"].to_numpy(float), d["pred"].to_numpy(float)
    ape, signed = d["ape"].to_numpy(float), d["signed_pct"].to_numpy(float)
    out_rng = d["out_of_train_range"].astype(bool).to_numpy()
    lo, hi = b3["appendix_other_models"][m]["train_dQ_logvar_range"]
    res = {"n": int(len(d)), "mape": float(ape.mean()), "ci95": bootstrap_ci(ape),
           "rmse_cycles": float(np.sqrt(np.mean((pred - true) ** 2))), "signed_mean_pct": float(signed.mean()),
           "over_ratio": float(np.mean(pred > true)), "n_over": int(np.sum(pred > true)), "n_out_of_range": int(out_rng.sum()),
           "mape_in_range": float(ape[~out_rng].mean()) if (~out_rng).any() else None,
           "mape_out_of_range": float(ape[out_rng].mean()) if out_rng.any() else None,
           "n_pred_outside_150_2300": int(np.sum((pred < LIFE_RANGE_NOTION[0]) | (pred > LIFE_RANGE_NOTION[1]))),
           "train_dQ_logvar_range": [float(lo), float(hi)]}
    if d["pi_lo"].notna().all():
        plo, phi = d["pi_lo"].to_numpy(float), d["pi_hi"].to_numpy(float)
        res["pi_coverage_10_90"] = float(np.mean((true >= plo) & (true <= phi)))
        res["pi_cv_resid_10_90"] = interval_stats(true, plo, phi, pred, out_rng)
    cols = ["cell_key", "group", "policy", "true", "pred", "ape", "signed_pct", "dQ_logvar", "out_of_train_range"]
    res["cells"] = _py(d[cols].to_dict(orient="records"))
    res["by_group"] = by_group(res)
    res["coefs_raw_fit36"] = b3["appendix_other_models"][m]["coefs_raw_fit36"]
    res["qcc_shift_decomposition"] = qcc_shift_decomposition(res["coefs_raw_fit36"], b1, d_all, res)
    res["breakin_check"] = breakin_check(res, d_feat, diag)
    return res


def _excl_block(removed: list[str], b3: dict) -> dict:
    """removed(라벨 셀)를 뺀 Batch 3 부분집합의 선택 모델·M0 지표. Qcc 이동량은 #23·#32 까지 뺀 라벨 없는 피처로 잰다."""
    P = pd.read_csv(RES / "batch3_predictions.csv")
    sid = b3["selected_id"]
    keep = [k for k in P.loc[P["model"] == sid, "cell_key"] if k not in removed]
    b1, d_feat, diag = load_batch1(), load_batch_features("batch3"), load_diagnostics("batch3")
    d_all = d_feat[~d_feat["cell_key"].isin(removed + ["batch3-23", "batch3-32"])]
    d_keep = d_feat[d_feat["cell_key"].isin(keep)]
    out = {m: _ev_from_saved(P, m, keep, b3, b1, d_all, d_keep, diag) for m in dict.fromkeys([sid, "M0"])}
    return {"selected": out[sid], "M0": out["M0"], "n": int(len(keep))}


def _assert_same(new: dict, old: dict, tag: str) -> None:
    for k in B3_EXCL_CHECK_KEYS:
        assert np.allclose(np.asarray(new[k], float), np.asarray(old[k], float), rtol=1e-9, atol=1e-9), (tag, k)
    for g, v in (old.get("qcc_shift_decomposition") or {}).get("groups", {}).items():
        for k, x in v.items():
            assert np.isclose(new["qcc_shift_decomposition"]["groups"][g][k], x, rtol=1e-9), (tag, g, k)
    for k, x in old["breakin_check"]["overall"].items():
        if x is not None:
            assert np.isclose(new["breakin_check"]["overall"][k], x, rtol=1e-9), (tag, "breakin", k)


def batch3_excl_corrected() -> dict:
    """정정된 목록(B3_PAPER_REMOVED)으로 '원저자 제거 셀 제외' Batch 3 지표를 저장된 예측에서 계산한다.
    report 단계가 부른다. 1회 실행 기록(batch3.json·.batch3_done·run_log 의 excl4 줄)은 고치지 않는다."""
    if "v" in _B3X_CACHE:
        return _B3X_CACHE["v"]
    assert BATCH3_DONE.exists() and "completed" in BATCH3_DONE.read_text(), "batch3 1회 실행 기록이 없다"
    b3 = read_json(BATCH3_JSON)
    assert b3["selection_sha256"] == sha256(SELECTION_JSON), "batch3.json 이 현재 선택과 다르다"
    if b3["paper_removed_cells"] == B3_PAPER_REMOVED:                 # 처음부터 다시 실행한 경우: batch3 단계가 이미 정정 목록을 씀
        ex = b3["excluding_paper_removed"]
        out = {"created": datetime.now().isoformat(timespec="seconds"),
               "what": "Batch 3 '원저자 공개 코드가 뺀 셀' 제외 결과. 보고 성능 표의 'Test (Batch 3)' 비고에 쓴다.",
               "removed_labeled_cells": B3_PAPER_REMOVED, "removed_all_cells": B3_PAPER_REMOVED_ALL,
               "method": "batch3 단계가 이 목록으로 직접 계산한 값(batch3.json['excluding_paper_removed'])",
               "n": ex["selected"]["n"], "selected_id": b3["selected_id"],
               "selected": {k: v for k, v in ex["selected"].items() if k != "cells"},
               "M0": {k: v for k, v in ex["M0"].items() if k != "cells"}}
        _B3X_CACHE["v"] = out
        return out
    assert b3["paper_removed_cells"] == B3_PAPER_REMOVED_OLD, "batch3.json 의 제외 목록이 예상과 다르다"
    old = _excl_block(B3_PAPER_REMOVED_OLD, b3)                      # ① 방법 검증: 잘못된 목록 → 저장 기록 재현
    _assert_same(old["selected"], b3["excluding_paper_removed"]["selected"], "selected")
    _assert_same(old["M0"], b3["excluding_paper_removed"]["M0"], "M0")
    new = _excl_block(B3_PAPER_REMOVED, b3)                          # ② 정정 목록
    strip = lambda e: {k: v for k, v in e.items() if k != "cells"}  # noqa: E731
    out = {"created": datetime.now().isoformat(timespec="seconds"),
           "what": "Batch 3 '원저자 공개 코드가 뺀 셀' 제외 결과 (정정본). 보고 성능 표의 'Test (Batch 3)' 비고에 쓴다.",
           "removed_labeled_cells": B3_PAPER_REMOVED, "removed_all_cells": B3_PAPER_REMOVED_ALL,
           "evidence": "원저자 공개 코드 Load Data.ipynb: del batch3['b3c37','b3c2','b3c23','b3c32','b3c42','b3c43']; "
                       "BuildPkl_Batch3 의 키 'b3c'+str(i) 의 i = 2018-04-12 .mat 순서 = 이 저장소 idx. "
                       "LoadData.m(batch3(38)=[] → endcap>0.885 → nind=[3,40:41])도 같은 6셀. #23·#32 = EOL 미도달(라벨 없음).",
           "wrong_list_used_2026_10_01": B3_PAPER_REMOVED_OLD,
           "previous_wrong_values": {"selected_mape": b3["excluding_paper_removed"]["selected"]["mape"],
                                     "M0_mape": b3["excluding_paper_removed"]["M0"]["mape"]},
           "method": "results/batch3_predictions.csv(1회 실행의 셀별 예측)에서 부분집합 지표만 다시 계산. 재적합·batch3 단계 재실행 없음. "
                     "잘못된 목록으로 같은 계산을 하면 batch3.json['excluding_paper_removed']가 그대로 재현됨을 assert 로 확인.",
           "check_old_list_reproduces_batch3_json": True,
           "inputs_sha256": {f: sha256(RES / f) for f in ("batch3_predictions.csv", "batch3.json", "features.csv",
                                                          "diagnostic_features.csv", "selection.json")},
           "n": new["n"], "selected_id": b3["selected_id"], "selected": strip(new["selected"]), "M0": strip(new["M0"])}
    _B3X_CACHE["v"] = out
    return out


def write_batch3_excl_corrected() -> str:
    first = not B3_EXCL_JSON.exists()
    out = batch3_excl_corrected()
    write_json(B3_EXCL_JSON, out)
    if first and "previous_wrong_values" in out:
        run_log("correction", f"B3 원저자 제거 셀 #38·#39→#42·#43 정정, 저장 예측으로 제외 지표 재계산(재적합·batch3 재실행 없음): "
                              f"{out['selected_id']} {out['previous_wrong_values']['selected_mape']:.3f}→{out['selected']['mape']:.3f}, "
                              f"M0 {out['previous_wrong_values']['M0_mape']:.3f}→{out['M0']['mape']:.3f}")
    return B3_EXCL_JSON.name


# ════════════════════════════════════════════════════════════════════════
# STAGE: report (노션 Regression 포맷)
# ════════════════════════════════════════════════════════════════════════
NOTE_GAP_TV, NOTE_GAP_VT, NOTE_GAP_TT = "(+) : 과적합 의심", "(+) : 배치간 일반화 저하 의심", "Target : 원논문 9.1%"


def _f(v) -> str:
    return "" if v is None else f"{v:.2f}"


def _g(a, b) -> str:
    """Gap = a − b (부호 표시). 둘 중 하나라도 없으면 빈 칸."""
    return "" if a is None or b is None else f"{a - b:+.2f}"


def collect(m: str) -> dict:
    """모델 m 의 Train/Valid/Test/Batch3 값과 비고 재료를 모은다 (없는 값은 None)."""
    sel = read_json(SELECTION_JSON)
    cvm = {r["model"]: r for r in sel["cv_summary"]}[m]
    fin = sel["final"][m]
    v = read_json(VALID_JSON) if VALID_JSON.exists() else None
    t = read_json(TEST_JSON) if TEST_JSON.exists() else None
    b3 = read_json(BATCH3_JSON) if BATCH3_JSON.exists() else None
    for name, j in (("valid", v), ("test", t), ("batch3", b3)):
        if j is not None and j["selection_sha256"] != sha256(SELECTION_JSON):
            raise SystemExit(f"[report] {name}.json 이 현재 selection.json 과 다른 선택에서 나왔다.")
    vm = v["all_models"][m] if v else None
    tm = t["appendix_other_models"][m] if t else None
    bm = b3["appendix_other_models"][m] if b3 else None
    b3x = batch3_excl_corrected() if b3 else None      # 정정 목록(#02·37·42·43), 저장 예측에서 계산
    bx = (b3x["selected"] if m == b3["selected_id"] else b3x["M0"] if m == "M0" else None) if b3 else None
    c0 = {r["model"]: r for r in sel["cv_summary"]}["C0"]["mean"]
    focus = sel["oof_focus_20_21"].get(m, {})
    return {"id": m, "name": SPECS[m].name, "params": fin["params"], "cv": cvm, "in_sample": fin["in_sample_mape"],
            "c0": c0, "focus": focus, "valid": vm, "test": tm, "b3": bm, "b3_excl": bx}


SHORT = {"M0": "OLS", "M1": "Ridge", "M2": "ElasticNet", "M3": "GPR", "M4_RF": "RF", "M4_GBR": "GBR",
         "C0": "상수", "M1_huber": "Huber", "M2_discharge": "ElasticNet"}


def _params_str(m: str, params: dict) -> str:
    """'M1 Ridge(alpha=0.0001)' 형태의 짧은 모델 표기."""
    p = ", ".join(f"{k.split('__')[-1]}={v:.4g}" if isinstance(v, float) else f"{k.split('__')[-1]}={v}"
                  for k, v in params.items())
    return f"{m} {SHORT.get(m, '')}" + (f"({p})" if p else "")


DETAIL_CSV_NOTE = "범위 안/밖·초기 용량 이동·break-in 점검은 model_performance_test_detail.csv"


def _notes(c: dict) -> dict:
    """노션 표 비고 (읽기용 요약). 통계 상세(ρ·p, 피처 범위 안/밖, 초기 용량 이동 분해)는
    model_performance_test_detail.csv · model_candidates.csv · selection.json 에 그대로 있다. 수치는 바뀌지 않는다."""
    cv, f20 = c["cv"], c["focus"]
    oof = "·".join(f"#{k[-2:]} {v['signed_pct']:+.1f}%" for k, v in f20.items())
    tr = (f"{_params_str(c['id'], c['params'])}; 정책 단위 GroupKFold(5)×20회 nested 평균, fold SD {cv['sd']:.2f} "
          f"(SE {cv['se']:.2f}); 학습 적합 {c['in_sample']:.2f}; 상수 예측 {c['c0']:.2f}; 최단 2셀 CV 오차 {oof}")
    va = (f"Hold-out 9셀·5정책 고정 분할, 95% CI [{c['valid']['ci95'][0]:.2f}, {c['valid']['ci95'][1]:.2f}]"
          if c["valid"] else "")
    te = _ext_note(c["test"], "39셀 1회(Batch 1 36셀 재학습)") if c["test"] else ""
    b3 = ""
    if c["b3"]:
        b3 = _ext_note(c["b3"], "44셀 1회(Batch 1 36셀 재학습)" + (
            f"; 원저자 공개 코드 제거 4셀(#02·#37·#42·#43) 제외 시 {c['b3_excl']['mape']:.2f}" if c["b3_excl"] else ""))
    return {"train": tr, "valid": va, "test": te, "b3": b3}


def _ext_note(e: dict, head: str) -> str:
    """Test/Batch 3 행 비고 요약: 그룹별 MAPE·부호 있는 오차·과대예측 수 (DAY 1 §11 Test 비고).
    피처 범위 안/밖 MAPE·계수 × 라벨 없는 초기 용량 이동량·H1 break-in 점검은 동반 표
    model_performance_test_detail.csv 에 있다(_detail_rows)."""
    n_over = e.get("n_over", int(round(e["over_ratio"] * e["n"])))
    parts = [head, "그룹 MAPE " + " / ".join(f"{k} {v['mape']:.2f}" for k, v in e["by_group"].items()),
             f"평균 부호오차 {e['signed_mean_pct']:+.1f}%", f"과대예측 {n_over}/{e['n']}", DETAIL_CSV_NOTE]
    return "; ".join(parts)


def _rho(v) -> str:
    return "NA" if v is None else f"{v:+.2f}"


def _h1_note(b: dict) -> str:
    """H1 사전 점검 한 줄: 전체 로그 잔차 vs 상승폭 · 정점 사이클(주·보조) Spearman ρ (p)."""
    o = b["overall"]
    lab = {"cc_breakin": "상승폭", "cc_breakin_peak_cycle": "CC 정점 사이클", "qd_breakin_peak_cycle": "보조 QD 정점"}
    items = [f"{lab[c]} {_rho(o[f'rho_{c}'])}" + (f"(p={o[f'p_{c}']:.2g})" if o[f"p_{c}"] is not None else "")
             for c in BREAKIN_COLS]
    return f"H1 로그잔차 Spearman ρ(n={o['n']}): " + ", ".join(items)


def _detail_rows(model: str, batch: str, e: dict) -> list[dict]:
    """외부 배치 1개·모델 1개의 전체/그룹별 세부 행."""
    q = (e.get("qcc_shift_decomposition") or {}).get("groups", {})
    h = e.get("breakin_check") or {}
    rows = [{"batch": batch, "model": model, "group": "전체", **{k: e[k] for k in DETAIL_KEYS},
             **{f"h1_{k}": h.get("overall", {}).get(k) for k in DETAIL_H1}}]
    for g, v in e["by_group"].items():
        rows.append({"batch": batch, "model": model, "group": g, **{k: v[k] for k in DETAIL_KEYS},
                     **{f"qcc_{k}": q.get(g, {}).get(k) for k in DETAIL_QCC},
                     **{f"h1_{k}": h.get("by_group", {}).get(g, {}).get(k) for k in DETAIL_H1}})
    return rows


DETAIL_KEYS = ["n", "mape", "signed_mean_pct", "over_ratio", "n_out_of_range", "mape_in_range", "mape_out_of_range"]
DETAIL_QCC = ["n_all_cells", "shift_mAh", "shift_in_b1_sd", "pred_factor", "observed_factor", "remainder_log10"]
DETAIL_H1 = ([f"{a}_{c}" for c in BREAKIN_COLS for a in ("rho", "p")]
             + [f"{a}_{c}" for c in DIAG_FEATURES for a in ("median", f"n_ge{PEAK_EDGE}")])


def ext_detail_table(sid: str) -> pd.DataFrame | None:
    """Test 비고 동반 표: 선택 모델·M0 의 그룹별 MAPE·부호 오차·과대예측 비율·범위 안/밖·계수×Qcc 이동량
    ·H1 로그 잔차 vs break-in 상승폭·정점 사이클 Spearman (h1_* 열)."""
    t = read_json(TEST_JSON) if TEST_JSON.exists() else None
    b3 = read_json(BATCH3_JSON) if BATCH3_JSON.exists() else None
    if t is None and b3 is None:
        return None
    rows = []
    for m in dict.fromkeys([sid, "M0"]):
        if t:
            rows += _detail_rows(m, "batch2", t["appendix_other_models"][m])
        if b3:
            rows += _detail_rows(m, "batch3", b3["appendix_other_models"][m])
            x = batch3_excl_corrected()["selected" if m == sid else "M0"]
            rows += _detail_rows(m, "batch3 (원저자 코드 제거 4셀 제외)", x)
    df = pd.DataFrame(rows)
    return df.round({c: 4 for c in df.columns if df[c].dtype.kind == "f"})


def perf_table(c: dict) -> pd.DataFrame:
    tr = c["cv"]["mean"]
    va = c["valid"]["mape"] if c["valid"] else None
    te = c["test"]["mape"] if c["test"] else None
    n = _notes(c)
    rows = [("Train (Batch 1 CV)", _f(tr), n["train"]), ("Valid (Batch 1 Hold-out)", _f(va), n["valid"]),
            ("Test (Batch 2)", _f(te), n["test"]), ("Gap (Train-Valid)", _g(va, tr), NOTE_GAP_TV),
            ("Gap (Valid-Test)", _g(te, va), NOTE_GAP_VT), ("Gap (Target-Test)", _g(te, TARGET_MAPE), NOTE_GAP_TT)]
    return pd.DataFrame(rows, columns=["구분", "MAPE (%)", "비고"])


def perf_table_b3(c: dict) -> pd.DataFrame:
    base = perf_table(c)
    b2 = c["test"]["mape"] if c["test"] else None
    b3 = c["b3"]["mape"] if c["b3"] else None
    rows = [(r["구분"], "", r["MAPE (%)"], r["비고"]) if i < 3 else ("", r["구분"], r["MAPE (%)"], r["비고"])
            for i, r in base.iterrows()]
    rows += [("Test (Batch 3)", "", _f(b3), _notes(c)["b3"]),
             ("", "Gap (Batch2-Batch3)", _g(b3, b2), "Test 성능 간 비교"),
             ("", "Gap (Target-Test)", _g(b3, TARGET_MAPE), "Batch 3 기준, 원논문 성능 비교")]
    return pd.DataFrame(rows, columns=["구분", "", "MAPE (%)", "비고"])


def candidates_table() -> pd.DataFrame:
    """후보·참고 모델 비교표 (선택 근거 기록용). Test/Batch 3 값은 사후 부록이며 선택을 바꾸지 않는다."""
    sel = read_json(SELECTION_JSON)
    rule, cvs = sel["rule"], {r["model"]: r for r in sel["cv_summary"]}
    ext = {k: read_json(p)["appendix_other_models"] if p.exists() else None
           for k, p in (("test", TEST_JSON), ("b3", BATCH3_JSON))}
    vj = read_json(VALID_JSON)["all_models"] if VALID_JSON.exists() else None
    rows = []
    for m in ALL_IDS:
        f, g = sel["final"][m], sel["final"][m]["gate"]
        rows.append({"model": m, "역할": "후보" if SPECS[m].candidate else "참고", "설명": SPECS[m].name,
                     "features": "+".join(SPECS[m].features) if len(SPECS[m].features) <= 2 else f"{len(SPECS[m].features)}개 풀",
                     "final_params": _params_str(m, f["params"]), "Train CV MAPE": round(cvs[m]["mean"], 3),
                     "SD": round(cvs[m]["sd"], 3), "SE": round(cvs[m]["se"], 3),
                     "1SE 이내": (m in rule["primary"]["within_1se"]) if SPECS[m].candidate else None,
                     "게이트 위반(#00~04)": f"{g['n_violations_00_04']}/5", "게이트 통과": bool(g["eligible"]),
                     "in-sample MAPE": round(f["in_sample_mape"], 2),
                     "Valid MAPE": round(vj[m]["mape"], 3) if vj else None,
                     "Test MAPE (부록)": round(ext["test"][m]["mape"], 3) if ext["test"] else None,
                     "Batch3 MAPE (부록)": round(ext["b3"][m]["mape"], 3) if ext["b3"] else None,
                     "선택": m == rule["selected"]})
    return pd.DataFrame(rows)


def stage_report() -> None:
    if not SELECTION_JSON.exists():
        raise SystemExit("[report] results/selection.json 이 없다.")
    sid = read_json(SELECTION_JSON)["selected"]["id"]
    outs = {"model_performance.csv": perf_table(collect(sid)), "model_performance_M0.csv": perf_table(collect("M0")),
            "model_candidates.csv": candidates_table()}
    if BATCH3_JSON.exists():
        outs["model_performance_batch3.csv"] = perf_table_b3(collect(sid))
        outs["model_performance_batch3_M0.csv"] = perf_table_b3(collect("M0"))
    detail = ext_detail_table(sid)
    if detail is not None:
        outs["model_performance_test_detail.csv"] = detail
    for name, df in outs.items():
        df.to_csv(RES / name, index=False, encoding="utf-8")
        print(f"\n### {name}\n{to_markdown(df)}")
    written = list(outs) + ([write_batch3_excl_corrected()] if BATCH3_JSON.exists() else [])
    run_log("report", f"wrote {written}")


def to_markdown(df: pd.DataFrame) -> str:
    """tabulate 없이 쓰는 간단한 마크다운 표 (README 붙여넣기용)."""
    head = "| " + " | ".join(map(str, df.columns)) + " |"
    rule = "|" + "---|" * len(df.columns)
    body = ["| " + " | ".join(map(str, r)) + " |" for r in df.itertuples(index=False)]
    return "\n".join([head, rule, *body])


# ════════════════════════════════════════════════════════════════════════
def main() -> None:
    ap = argparse.ArgumentParser(description="DAY 2 모델 선택·검증·테스트·리포트")
    ap.add_argument("--stage", required=True, choices=["select", "valid", "test", "batch3", "report"])
    ap.add_argument("--force", action="store_true", help="1회 단계 재실행 (run_log 에 기록)")
    ap.add_argument("--n-jobs", type=int, default=max(1, min(8, (os.cpu_count() or 2) - 2)))
    a = ap.parse_args()
    run_log(a.stage, f"start force={a.force}")
    if a.stage == "select":
        stage_select(a.n_jobs, a.force)
    elif a.stage == "valid":
        stage_valid()
    elif a.stage == "test":
        stage_test(a.force)
    elif a.stage == "batch3":
        stage_batch3(a.force)
    else:
        stage_report()


if __name__ == "__main__":
    main()
