"""MIT-Stanford 배터리 .mat(MATLAB v7.3 / HDF5) → 분석용 경량 pickle 추출.

원본 파일은 수 GB라 mat73로 통째로 읽으면 메모리를 많이 쓴다.
h5py로 필요한 필드만 셀 단위로 읽어 data/processed/{batch}.pkl 로 저장한다.

저장 내용 (셀 하나 = dict 하나)
- cell_key, batch, idx, policy, policy_readable, barcode, channel_id
- cycle_life            : 원본 값 (NaN = EOL 미도달)
- summary               : DataFrame (cycle, QD, QC, IR, Tmax, Tavg, Tmin, chargetime)
- Qdlin / Tdlin / dQdV  : (MAX_CYC+1, 1000) 배열, 행 번호 = cycles 리스트 인덱스
- raw                   : {cycle_index: DataFrame(t, V, I, T, Qc, Qd)} — 충전 패턴 확인용 일부 사이클
- Vdlin                 : 보간 전압 축 (1000,)

사용법: python src/preprocess.py            # data/raw 의 3개 배치 모두
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW, OUT = ROOT / "data" / "raw", ROOT / "data" / "processed"

BATCHES = {  # 노션 표의 Batch 정의 그대로
    "batch1": "2017-05-12_batchdata_updated_struct_errorcorrect.mat",  # 원논문 학습 데이터셋
    "batch2": "2018-02-20_batchdata_updated_struct_errorcorrect.mat",  # 원논문 1차 테스트셋
    "batch3": "2018-04-12_batchdata_updated_struct_errorcorrect.mat",  # 원논문 2차 테스트셋
}
MAX_CYC = 200          # Qdlin 등 1000포인트 곡선을 저장할 최대 사이클 인덱스
RAW_CYCLES = (1, 2, 5, 10, 100)   # 원시 시계열(t, V, I, T, Qc, Qd)을 저장할 사이클 인덱스
SUMMARY_MAP = {"cycle": "cycle", "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
               "Tmax": "Tmax", "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime"}


def _str(f, ref) -> str:
    a = f[ref][()]
    if a.dtype.kind in "ui" and a.size and a.ndim == 2 and a.shape[1] == 1 or a.dtype == np.uint16:
        return "".join(chr(int(x)) for x in a.flatten() if int(x) != 0)
    return str(a)


def _scalar(f, ref) -> float:
    a = np.asarray(f[ref][()], dtype=float).flatten()
    return float(a[0]) if a.size else float("nan")


def _vec(f, ref) -> np.ndarray:
    return np.asarray(f[ref][()], dtype=float).flatten()


def extract_cell(f, b, i: int, batch: str) -> dict:
    s = f[b["summary"][i, 0]]
    summary = pd.DataFrame({new: np.asarray(s[old][()], float).flatten() for old, new in SUMMARY_MAP.items()})
    c = f[b["cycles"][i, 0]]
    n_cyc = c["Qdlin"].shape[0]
    upto = min(n_cyc, MAX_CYC + 1)
    curves = {k: np.full((MAX_CYC + 1, 1000), np.nan, dtype=np.float32) for k in ("Qdlin", "Tdlin", "dQdV")}
    for j in range(upto):
        for key, src in (("Qdlin", "Qdlin"), ("Tdlin", "Tdlin"), ("dQdV", "discharge_dQdV")):
            v = _vec(f, c[src][j, 0])
            if v.size == 1000:
                curves[key][j] = v
    raw = {}
    for j in RAW_CYCLES:
        if j < n_cyc:
            cols = {k: _vec(f, c[k][j, 0]) for k in ("t", "V", "I", "T", "Qc", "Qd")}
            n = min(len(v) for v in cols.values())
            if n > 1:
                raw[j] = pd.DataFrame({k: v[:n] for k, v in cols.items()})
    return {
        "cell_key": f"{batch}-{i:02d}", "batch": batch, "idx": i,
        "policy": _str(f, b["policy"][i, 0]), "policy_readable": _str(f, b["policy_readable"][i, 0]),
        "barcode": _safe_str(f, b["barcode"][i, 0]), "channel_id": _safe_str(f, b["channel_id"][i, 0]),
        "cycle_life": _scalar(f, b["cycle_life"][i, 0]),
        "n_cycles_raw": int(n_cyc), "summary": summary, "raw": raw,
        "Vdlin": _vec(f, b["Vdlin"][i, 0]), **curves,
    }


def _safe_str(f, ref) -> str:
    try:
        return _str(f, ref)
    except Exception:  # noqa: BLE001 — barcode/channel_id 은 MATLAB string 타입이라 읽히지 않을 수 있음
        return ""


def extract_batch(batch: str, fname: str) -> list[dict]:
    with h5py.File(RAW / fname, "r") as f:
        b = f["batch"]
        cells = [extract_cell(f, b, i, batch) for i in range(b["summary"].shape[0])]
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / f"{batch}.pkl", "wb") as fh:
        pickle.dump(cells, fh, protocol=pickle.HIGHEST_PROTOCOL)
    return cells


if __name__ == "__main__":
    targets = sys.argv[1:] or list(BATCHES)
    for name in targets:
        cells = extract_batch(name, BATCHES.get(name, name))
        cl = np.array([c["cycle_life"] for c in cells])
        print(f"{name}: {len(cells)} cells, cycle_life NaN={np.isnan(cl).sum()}, "
              f"range={np.nanmin(cl):.0f}~{np.nanmax(cl):.0f}", flush=True)
