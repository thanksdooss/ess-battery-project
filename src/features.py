"""DAY 2 피처 테이블 — DAY 1 전략(reports/day1_design_report.md §8)에서 고정한 피처만 계산한다.

피처 (정의는 src/eda/q5_correlation.py · src/eda/q5d_conditional.py 와 같다 → DAY 1 보고서 수치 재현)
  dQ_logvar          log10(var(Qdlin[99] − Qdlin[9])) : cycle 100 − cycle 10, 1,000점, ddof=0, 평활화 없음 (주 피처)
  Qcc_init           cycle 2~6 Qdlin 2.0 V 끝점(CC 방전 종료 용량) 중앙값 [Ah], 끝점 ≤0·>1.2 Ah → NaN (후보)
  cc_breakin         cycle 2~100 끝점 9-사이클 이동중앙값 최댓값 − Qcc_init [mAh]          (M2 풀 전용)
  dQ_skew / dQ_kurt  ΔQ₁₀₀₋₁₀(V) 왜도 / 첨도(Fisher excess)                              (M2 풀 전용)
  fade_slope_91_100  clean_summary QD cycle 91~100 OLS 기울기 [mAh/100 cycle, 음수 = 감소] (M2 풀 전용)
  chargetime_avg5    clean_summary chargetime cycle 2~6 평균 [분]                          (M2 풀 전용)
  Tavg_mean          cycle 2~100 평균온도 평균                                              (M2 풀 전용)
  IR_min / IR_diff   cycle 2~100 IR 최솟값 / IR(100) − IR(2)                               (M2 풀 전용)
  dQ_logvar_med21    ΔQ₁₀₀₋₁₀(V) 에 21점 이동중앙값(center, 끝단 min_periods=1)을 씌운 뒤 log10 var
                     → 모델 입력 아님. DAY 1 §10 '잡음 셀 #09·38~45만 21점 중앙값 민감도' 기록 전용
                       (src/train.py 가 B1 잡음 셀에서만 dQ_logvar 대신 넣어 Train CV / Valid 를 다시 잰다)

진단 전용 (features.csv 에 넣지 않는다 → features_csv_sha256·선택 결과 불변)
  cc_breakin_peak_cycle  (주) cc_breakin 과 같은 시계열(cycle 2~100 끝점 9-사이클 이동중앙값)이 최댓값에 처음
                         닿는 사이클. 라벨 없음. DAY 1 H1 사전 점검 'B2 잔차 vs break-in 정점 사이클' 용
  qd_breakin_peak_cycle  (보조) DAY 1 q2_degradation.initial_rise 의 정점 정의(clean_summary QD → 보간 →
                         9-사이클 이동중앙값 → argmax)를 cycle 2~100 으로 자른 판. H1 문구 'cycle 81 vs 37' 이
                         이 요약 QD 정점(수명 전체)에서 나왔으므로 교차 확인용으로 함께 둔다
  → results/diagnostic_features.csv (src/train.py 의 test/batch3 단계만 읽는다)

규칙
  - 입력은 cycle 2~100 만 쓴다(B1 cycle 1 은 빈 더미). EarlyCycles 보기(view)가 접근한 사이클을 기록하고
    셀마다 '사용 사이클 최댓값 ≤ 100' 을 assert 한다(knee 등 미래 정보 누수 차단).
  - 배치별 예외 전처리·재중심화 없음. 결측은 NaN 으로 두고 모델 Pipeline 안에서만 대체한다(M2).
  - 라벨(cycle_life)은 세 배치 모두 파일에 남긴다. 단, 선택·검증 단계(src/train.py select/valid)는
    Batch 1 행만 읽는다. Batch 2/3 라벨은 test/batch3 단계에서만 읽는다.

산출물: results/features.csv (셀당 1행, 전체 139셀)
        results/diagnostic_features.csv (셀당 1행, 라벨 열 없음, 진단 전용)
실행:   cd ess-battery-project && python src/features.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

from data import ROOT, cell_table, clean_summary, load_all, qdlin  # noqa: E402

FIRST, LAST = 2, 100          # 초기 사이클 창 (B1 cycle 1 더미 제외, 모든 배치 동일)
QCC_WIN = (2, 6)              # Qcc_init 창
BREAKIN_SMOOTH = 9            # CC break-in 이동중앙값 창
QCC_VALID = (0.0, 1.2)        # Qdlin 2.0 V 끝점 유효 범위 [Ah] (≤0, >1.2 → NaN)
SLOPE_SCALE = 1e5             # Ah/cycle → mAh/100 cycle
DQ_MED_WIN = 21               # 잡음 셀 민감도용 ΔQ 이동중앙값 창 (DAY 1 §10)

MAIN_FEATURE = "dQ_logvar"
M1_FEATURES = ["dQ_logvar", "Qcc_init"]
M2_POOL = ["dQ_logvar", "Qcc_init", "cc_breakin", "dQ_skew", "dQ_kurt", "fade_slope_91_100",
           "chargetime_avg5", "Tavg_mean", "IR_min", "IR_diff"]          # DAY 1 §8 순서 그대로
M2_DISCHARGE = M2_POOL[:5]                                               # 원논문 discharge 대응 하위 풀
SENS_FEATURES = ["dQ_logvar_med21"]                                      # 민감도 기록 전용 (모델 입력 아님)
META = ["batch", "cell_key", "idx", "policy", "group", "labeled", "censored", "cycle_life"]
FEATURES_CSV = ROOT / "results" / "features.csv"
DIAG_FEATURES = ["cc_breakin_peak_cycle", "qd_breakin_peak_cycle"]       # 진단 전용 (모델 입력·features.csv 아님)
DIAG_META = ["batch", "cell_key", "idx", "policy", "group"]              # 라벨 열(cycle_life·labeled·censored) 없음
DIAG_CSV = ROOT / "results" / "diagnostic_features.csv"


# ════════════════════════════════════════════════════════════════════════
# 사이클 창 보기: cycle 2~100 밖 접근을 막고, 사용한 사이클을 기록한다
# ════════════════════════════════════════════════════════════════════════
class EarlyCycles:
    """한 셀의 cycle FIRST~LAST 만 보이는 보기. used 에 접근한 사이클 번호가 쌓인다."""

    def __init__(self, c: dict, first: int = FIRST, last: int = LAST):
        self._c, self.first, self.last = c, first, last
        s = clean_summary(c["summary"])
        s = s[(s["cycle"] >= first) & (s["cycle"] <= last)].set_index("cycle")
        self.summary = s
        self.used: set[int] = {int(k) for k in s.index}

    def qdlin(self, k: int) -> np.ndarray:
        assert self.first <= k <= self.last, f"{self._c['cell_key']}: cycle {k} 은 허용 창({self.first}~{self.last}) 밖"
        self.used.add(int(k))
        return qdlin(self._c, k)

    def max_cycle_used(self) -> int:
        return max(self.used)


# ════════════════════════════════════════════════════════════════════════
# ΔQ(V) 피처 (q5_correlation.early_features 와 같은 정의)
# ════════════════════════════════════════════════════════════════════════
def delta_q_100_10(v: EarlyCycles) -> np.ndarray:
    """ΔQ₁₀₀₋₁₀(V) = Qdlin[99] − Qdlin[9] (1,000점, 실제 3.5→2.0 V). 비유한 값만 뺀다."""
    dq = v.qdlin(100) - v.qdlin(10)
    return dq[np.isfinite(dq)]


def dq_features(v: EarlyCycles) -> dict:
    dq = delta_q_100_10(v)
    return {"dQ_logvar": np.log10(np.var(dq)),                 # ddof=0, 평활화 없음 (원논문 정의)
            "dQ_skew": stats.skew(dq),
            "dQ_kurt": stats.kurtosis(dq, fisher=True),          # Fisher excess (정규 = 0)
            "dQ_logvar_med21": dq_logvar_smoothed(dq)}           # 민감도 전용


def dq_logvar_smoothed(dq: np.ndarray, win: int = DQ_MED_WIN) -> float:
    """21점 이동중앙값(center=True, 끝단은 창이 줄어든 중앙값) 후 log10 var(ddof=0). 고주파 잡음 민감도용."""
    sm = pd.Series(dq).rolling(win, center=True, min_periods=1).median().to_numpy()
    return float(np.log10(np.var(sm)))


# ════════════════════════════════════════════════════════════════════════
# CC 끝점 피처 (q5d_conditional.qcc_series / qcc_init / cc_breakin_mAh 와 같은 정의)
# ════════════════════════════════════════════════════════════════════════
def qcc_series(v: EarlyCycles, a: int = FIRST, b: int = LAST) -> pd.Series:
    """cycle a..b 의 Qdlin 2.0 V 끝점(CC 방전 종료 용량, Ah). 이상값(≤0, >1.2 Ah)은 NaN."""
    e = np.array([v.qdlin(k)[-1] for k in range(a, b + 1)], float)
    e[(e <= QCC_VALID[0]) | (e > QCC_VALID[1])] = np.nan
    return pd.Series(e, index=range(a, b + 1))


def qcc_init(v: EarlyCycles) -> float:
    return float(np.nanmedian(qcc_series(v, *QCC_WIN)))


def qcc_smoothed(v: EarlyCycles) -> pd.Series:
    """cycle 2~100 끝점 → 결측 선형보간 → 9-사이클 중심 이동중앙값(끝단 min_periods=1). break-in 두 값의 공통 시계열."""
    s = qcc_series(v).interpolate(limit_direction="both")
    return s.rolling(BREAKIN_SMOOTH, center=True, min_periods=1).median()


def cc_breakin_mAh(v: EarlyCycles) -> float:
    """CC break-in 상승폭 = cycle 2~100 끝점 9-사이클 이동중앙값 최댓값 − Qcc_init (mAh)."""
    return float((qcc_smoothed(v).max() - qcc_init(v)) * 1000)


def cc_breakin_peak_cycle(v: EarlyCycles) -> int:
    """CC break-in 정점 사이클 = 같은 이동중앙값 시계열이 최댓값에 처음 닿는 사이클 (동점이면 가장 이른 사이클)."""
    return int(qcc_smoothed(v).idxmax())


def qd_breakin_peak_cycle(v: EarlyCycles) -> int:
    """요약 QD 정점 사이클 (q2_degradation.qd_series·initial_rise 와 같은 처리, cycle 2~100 만):
    clean QD 행 순서대로 결측 보간 → 9-행 중심 이동중앙값 → 처음 최댓값이 되는 사이클."""
    s = v.summary
    qs = s["QD"].reset_index(drop=True).interpolate(limit_direction="both")
    qs = qs.rolling(BREAKIN_SMOOTH, center=True, min_periods=1).median()
    return int(s.index[int(np.nanargmax(qs.to_numpy(float)))])


# ════════════════════════════════════════════════════════════════════════
# summary 피처 (clean_summary, cycle 2~100)
# ════════════════════════════════════════════════════════════════════════
def fade_slope_91_100(v: EarlyCycles) -> float:
    """QD cycle 91~100 OLS 기울기 (mAh/100 cycle, 음수 = 감소). 10점이라 OLS(np.polyfit)."""
    s = v.summary
    cyc, q = s.index.to_numpy(float), s["QD"].to_numpy(float)
    m = (cyc >= 91) & (cyc <= 100) & np.isfinite(q)
    return float(np.polyfit(cyc[m], q[m], 1)[0] * SLOPE_SCALE)


def summary_features(v: EarlyCycles) -> dict:
    s = v.summary
    return {"chargetime_avg5": s.loc[2:6, "chargetime"].mean(),   # 첫 5 사이클 = cycle 2~6
            "Tavg_mean": s["Tavg"].mean(),
            "IR_min": s["IR"].min(),
            "IR_diff": s.at[100.0, "IR"] - s.at[2.0, "IR"]}


# ════════════════════════════════════════════════════════════════════════
# 셀 → 1행, 셀 목록 → 테이블
# ════════════════════════════════════════════════════════════════════════
def cell_features(c: dict) -> dict:
    v = EarlyCycles(c)
    f = {**dq_features(v), "Qcc_init": qcc_init(v), "cc_breakin": cc_breakin_mAh(v),
         "fade_slope_91_100": fade_slope_91_100(v), **summary_features(v)}
    used = v.max_cycle_used()
    assert used <= LAST, f"{c['cell_key']}: 사용 사이클 최댓값 {used} > {LAST} (누수)"
    return {"cell_key": c["cell_key"], **{k: float(f[k]) for k in M2_POOL + SENS_FEATURES}, "max_cycle_used": used}


def cell_diagnostics(c: dict) -> dict:
    """진단 전용 값 (별도 보기로 계산해 사용 사이클 최댓값도 따로 assert)."""
    v = EarlyCycles(c)
    d = {"cc_breakin_peak_cycle": cc_breakin_peak_cycle(v), "qd_breakin_peak_cycle": qd_breakin_peak_cycle(v)}
    used = v.max_cycle_used()
    assert used <= LAST and all(FIRST <= x <= LAST for x in d.values()), f"{c['cell_key']}: 사용 사이클 {used}, 정점 {d}"
    return {"cell_key": c["cell_key"], **d, "max_cycle_used": used}


def build_diagnostic_table(cells: list[dict]) -> pd.DataFrame:
    """셀당 1행: 라벨 없는 메타 + 진단 값. features.csv 와 분리해 선택 입력(sha)에 영향이 없다."""
    meta = cell_table(cells)[DIAG_META]
    diag = pd.DataFrame([cell_diagnostics(c) for c in cells])
    df = meta.merge(diag, on="cell_key", how="left", validate="one_to_one")
    assert df["max_cycle_used"].max() <= LAST and not {"cycle_life", "labeled", "censored"} & set(df.columns)
    return df[DIAG_META + DIAG_FEATURES + ["max_cycle_used"]]


def build_feature_table(cells: list[dict]) -> pd.DataFrame:
    """셀당 1행: 메타(batch, cell_key, idx, policy, group, labeled, censored, cycle_life) + 피처 10개 + 민감도 1개."""
    meta = cell_table(cells)[META]
    feat = pd.DataFrame([cell_features(c) for c in cells])
    df = meta.merge(feat, on="cell_key", how="left", validate="one_to_one")
    assert df["max_cycle_used"].max() <= LAST
    return df[META + M2_POOL + SENS_FEATURES + ["max_cycle_used"]]


def check_against_day1(df: pd.DataFrame) -> pd.Series:
    """DAY 1 산출물(results/eda/early_features.csv)과 피처 값이 같은지 확인 → 피처별 최대 절대 차이."""
    ref_path = ROOT / "results" / "eda" / "early_features.csv"
    if not ref_path.exists():
        return pd.Series(dtype=float)
    ref = pd.read_csv(ref_path).set_index("cell_key")[M2_POOL]
    cur = df.set_index("cell_key")[M2_POOL].loc[ref.index]
    diff = (cur - ref).abs().max()
    nan_mismatch = (cur.isna() != ref.isna()).sum()
    assert int(nan_mismatch.sum()) == 0, f"NaN 위치가 DAY 1 과 다름: {nan_mismatch[nan_mismatch > 0].to_dict()}"
    return diff


def main() -> pd.DataFrame:
    cells = load_all()
    df = build_feature_table(cells)
    diff = check_against_day1(df)
    if len(diff):
        print("DAY 1 early_features.csv 대비 피처별 최대 |차이|:")
        print(diff.to_string())
        assert (diff.fillna(0) < 1e-9).all(), "DAY 1 피처 정의와 수치가 다름"
    FEATURES_CSV.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(FEATURES_CSV, index=False)
    n = df.groupby("batch").agg(cells=("cell_key", "size"), labeled=("labeled", "sum"), censored=("censored", "sum"))
    print(n.to_string())
    print(f"max cycle used = {int(df['max_cycle_used'].max())} (≤ {LAST}) → {FEATURES_CSV.relative_to(ROOT)}")
    write_diagnostics(cells)
    return df


def write_diagnostics(cells: list[dict]) -> pd.DataFrame:
    """진단 전용 파일 (라벨 없음). 배치·그룹별 정점 사이클 중앙값을 함께 출력한다."""
    dg = build_diagnostic_table(cells)
    dg.to_csv(DIAG_CSV, index=False)
    pk = dg.groupby(["batch", "group"])[DIAG_FEATURES].agg(["median", "min", "max"])
    print(pk.to_string())
    print(f"max cycle used = {int(dg['max_cycle_used'].max())} (≤ {LAST}) → {DIAG_CSV.relative_to(ROOT)}")
    return dg


if __name__ == "__main__":
    main()
