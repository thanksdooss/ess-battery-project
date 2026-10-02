"""Batch 2 기록 공백(약 66시간) 점검 — Gap(Valid-Test) 원인 후보를 원본에서 확인한다 (사후 점검, 모델·보고 수치 불변).

무엇을 보나
- 원본 .mat 의 사이클별 시간 t(분)로 사이클 길이(t 최댓값 − 최솟값)를 잰다. cycle 1~100 만 읽는다.
- ΔQ 창(cycle 10→100) 안에서 가장 긴 사이클을 '공백 사이클'로 본다. 24시간을 넘으면 긴 공백으로 센다.
  창 안에 24시간 넘는 사이클이 여러 개면 한 번의 정지가 아니라 시각 기록 이상으로 분류한다(Batch 1 #18).
- 공백 전후 방전 용량 변화(mAh) = 공백 뒤 6사이클 QD 중앙값 − 공백 앞 6사이클 QD 중앙값 (공백 사이클 자체는 뺀다).
  QD 는 data.clean_summary 를 거친 값이다. −3 mAh 이하를 '떨어짐'으로 센다.
- 회복 = cycle 98~100 QD 중앙값이 공백 뒤 6사이클 중앙값보다 높으면 '다시 오름'으로 센다.
- break-in 정점 사이클(results/diagnostic_features.csv, 라벨 없음)이 공백 뒤에 있는지 센다.

라벨
- 공백 검출·용량 변화·정점 비교는 라벨 값을 쓰지 않는다(셀 선택에 '라벨 있음' 표시만 쓴다).
- 마지막 블록(용량 변화 vs Test 오차 순위상관)만 저장된 Test 예측(results/test_predictions.csv)을 읽는다.
  사후 분석이며 보고 성능이 아니다. 모델 재적합·Test 재실행은 없다.

입력: data/raw/*.mat (원본, 저장소에 없음 → data/README.md), data/processed/*.pkl,
      results/features.csv · diagnostic_features.csv · test_predictions.csv
출력: results/rest_gap_check.csv (셀별), results/rest_gap_summary.json (배치별 요약)
실행: cd ess-battery-project && python src/check_rest_gap.py
"""
from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import h5py  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from scipy import stats  # noqa: E402

import data as D  # noqa: E402
from preprocess import BATCHES, RAW  # noqa: E402

RES = D.ROOT / "results"
WIN = (10, 100)          # ΔQ 창 = cycle 10 → 100 (노션 Q3)
LONG_H = 24.0            # 이보다 긴 사이클 = 긴 공백
K = 6                    # 공백 앞·뒤 비교 사이클 수
DROP_MAH = -3.0          # 이 값 이하 = 공백 뒤 용량이 떨어짐
OUT_CSV, OUT_JSON = RES / "rest_gap_check.csv", RES / "rest_gap_summary.json"
# 라벨 없는 Batch 2 #37·#38(varcharge·slowcycle)은 공백 직후 QD 가 비어 NaN 이 된다. 요약은 라벨 셀만 쓰므로 경고만 끈다.
warnings.filterwarnings("ignore", message="All-NaN slice", category=RuntimeWarning)


def cycle_hours(f, b, i: int, last: int = WIN[1]) -> np.ndarray:
    """셀 i 의 cycle 1~last 길이(시간). 행 k = cycle k+1 (docs/data_notes.md)."""
    c = f[b["cycles"][i, 0]]
    n = min(c["t"].shape[0], last)
    out = np.full(last, np.nan)
    for j in range(n):
        t = np.asarray(f[c["t"][j, 0]][()], float).ravel()
        if t.size > 1:
            out[j] = (np.nanmax(t) - np.nanmin(t)) / 60.0
    return out


def scan_batch(batch: str, meta: pd.DataFrame, diag: pd.DataFrame) -> list[dict]:
    cells = {c["cell_key"]: c for c in D.load_batch(batch)}
    rows = []
    with h5py.File(RAW / BATCHES[batch], "r") as f:
        b = f["batch"]
        for i in range(b["summary"].shape[0]):
            key = f"{batch}-{i:02d}"
            h = cycle_hours(f, b, i)
            w = h[WIN[0] - 1:WIN[1]]                      # cycle 10~100
            j = int(np.nanargmax(w)) + WIN[0] - 1          # 창 안 가장 긴 사이클의 행
            n_long = int(np.sum(w > LONG_H))
            qd = D.clean_summary(cells[key]["summary"])["QD"].to_numpy(float)
            pre, post = np.nanmedian(qd[j - K:j]), np.nanmedian(qd[j + 1:j + 1 + K])
            end = np.nanmedian(qd[WIN[1] - 3:WIN[1]])          # cycle 98~100
            m, g = meta.loc[key], diag.loc[key] if key in diag.index else None
            rows.append({
                "cell_key": key, "batch": batch, "group": m["group"], "labeled": bool(m["labeled"]),
                "gap_cycle": j + 1, "gap_hours": float(h[j]), "median_cycle_hours": float(np.nanmedian(w)),
                "n_cycles_over_24h_in_window": n_long,
                "kind": ("단일 긴 공백" if n_long == 1 else "시각 기록 이상(여러 사이클)" if n_long > 1 else "없음"),
                "qd_before_mAh": pre * 1000, "qd_after_mAh": post * 1000, "qd_change_mAh": (post - pre) * 1000,
                "qd_cycle98_100_mAh": end * 1000, "recovered_by_cycle100": bool(end > post),
                "cc_peak_cycle": None if g is None else int(g["cc_breakin_peak_cycle"]),
                "qd_peak_cycle": None if g is None else int(g["qd_breakin_peak_cycle"]),
            })
    return rows


def summarize(R: pd.DataFrame) -> dict:
    out = {}
    for batch, x in R[R["labeled"]].groupby("batch"):
        single = x[x["kind"] == "단일 긴 공백"]
        blk = {"n_labeled": int(len(x)), "n_single_long_gap_in_window": int(len(single)),
               "n_time_record_anomaly": int((x["kind"] == "시각 기록 이상(여러 사이클)").sum()),
               "anomaly_cells": x.loc[x["kind"] == "시각 기록 이상(여러 사이클)", "cell_key"].tolist()}
        if len(single):
            blk.update({"gap_hours_range": [round(single["gap_hours"].min(), 1), round(single["gap_hours"].max(), 1)],
                        "median_cycle_hours": round(float(single["median_cycle_hours"].median()), 2),
                        "qd_change_mAh_median": round(float(single["qd_change_mAh"].median()), 1),
                        f"n_qd_change_le_{DROP_MAH:g}mAh": int((single["qd_change_mAh"] <= DROP_MAH).sum()),
                        "n_recovered_by_cycle100": int(single["recovered_by_cycle100"].sum()),
                        "by_group": {}})
            for grp, y in single.groupby("group"):
                blk["by_group"][grp] = {
                    "n": int(len(y)), "gap_cycle_range": [int(y["gap_cycle"].min()), int(y["gap_cycle"].max())],
                    "qd_change_mAh_median": round(float(y["qd_change_mAh"].median()), 1),
                    "n_qd_peak_after_gap": int((y["qd_peak_cycle"] > y["gap_cycle"]).sum()),
                    "n_cc_peak_after_gap": int((y["cc_peak_cycle"] > y["gap_cycle"]).sum())}
        out[batch] = blk
    return out


def posthoc_error_link(R: pd.DataFrame) -> dict:
    """사후(라벨 사용): Batch 2 공백 전후 용량 변화 vs 저장된 M1 Test 부호 오차의 Spearman. 보고 성능 아님."""
    P = pd.read_csv(RES / "test_predictions.csv")
    x = R[(R["batch"] == "batch2") & R["labeled"]].merge(
        P.loc[P["model"] == "M1", ["cell_key", "signed_pct"]], on="cell_key", validate="one_to_one")
    out = {"주의": "저장된 Test 예측만 읽은 사후 분석이다. 상관은 인과가 아니며, 모든 Batch 2 셀에 공백이 있어 비교군이 없다."}
    for name, y in (("batch2 전체", x), *((f"batch2 {g}", z) for g, z in x.groupby("group"))):
        r = stats.spearmanr(y["qd_change_mAh"], y["signed_pct"])
        out[name] = {"n": int(len(y)), "spearman_rho": round(float(r.statistic), 2), "p": round(float(r.pvalue), 3)}
    return out


def main() -> None:
    F = pd.read_csv(RES / "features.csv")
    meta = F.set_index("cell_key")[["group", "labeled"]]
    diag = pd.read_csv(RES / "diagnostic_features.csv").set_index("cell_key")
    rows = [r for batch in ("batch1", "batch2", "batch3") for r in scan_batch(batch, meta, diag)]
    R = pd.DataFrame(rows)
    R.round(3).to_csv(OUT_CSV, index=False, encoding="utf-8")
    out = {"정의": {"작성": "테스트(2026-10-01) 뒤 원본 시간 기록 점검용으로 작성. 공백 검출·용량 변화·정점 비교는 라벨 값을 쓰지 않음(사후_오차와의_관계 블록만 저장된 Test 오차 사용)",
                  "창": f"cycle {WIN[0]}→{WIN[1]} (ΔQ 창)", "긴 공백": f"사이클 길이 > {LONG_H:g}시간",
                  "용량 변화": f"공백 뒤 {K}사이클 QD 중앙값 − 공백 앞 {K}사이클 QD 중앙값 (떨어지면 음수, 공백 사이클 제외, clean_summary)",
                  "떨어짐": f"용량 변화 ≤ {DROP_MAH:g} mAh",
                  "다시 오름": f"cycle 98~100 QD 중앙값 > 공백 뒤 {K}사이클 QD 중앙값",
                  "정점": "diagnostic_features.csv 의 cc_breakin_peak_cycle(CC 끝점, 주)·qd_breakin_peak_cycle(요약 QD, 보조)"},
           "라벨 셀 요약": summarize(R), "사후_오차와의_관계": posthoc_error_link(R)}
    OUT_JSON.write_text(json.dumps(out, ensure_ascii=False, indent=2))
    print(json.dumps(out, ensure_ascii=False, indent=2))
    print("wrote", OUT_CSV.relative_to(D.ROOT), OUT_JSON.relative_to(D.ROOT))


if __name__ == "__main__":
    main()
