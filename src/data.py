"""공통 데이터 로더 — 모든 노트북/스크립트가 같은 규칙으로 데이터를 읽도록 한 곳에 모은다.

- load_batch / load_all : data/processed/{batch}.pkl (src/preprocess.py 산출물)
- parse_policy          : "C1(Q1%)-C2" 충전 프로토콜 파싱 → 0→80% 충전시간, 평균 C-rate
- cell_table            : 셀 단위 메타 테이블 (배치, 프로토콜, cycle_life, 품질 플래그)

규칙의 근거는 docs/data_notes.md 에 기록한다.
"""
from __future__ import annotations

import pickle
import re
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PROC = ROOT / "data" / "processed"
BATCH_NAMES = ("batch1", "batch2", "batch3")
NOMINAL_AH = 1.1                 # A123 APR18650M1A 정격 용량
EOL_AH = 0.8 * NOMINAL_AH        # 노션: EOL = 80% SOH 도달 → 0.88 Ah
LABEL_THRESHOLD = 550            # 노션: cycle_life >= 550 → 1(장수명)
RANDOM_STATE = 42                # 모든 분할·모델에서 고정

_POLICY = re.compile(r"^\s*([\d.]+)C\((\d+)%\)-([\d.]+)C")


def parse_policy(p: str) -> dict:
    """'5.4C(40%)-3.6C' → C1=5.4, Q1=40, C2=3.6.

    노션 정의: 'Q1%까지 C1 속도로 충전, 이후(80%까지) C2 속도로 충전'.
    t80  = 0→80% 충전에 걸리는 시간(분) = 60·(Q1/100)/C1 + 60·((80−Q1)/100)/C2
    avgC = 0→80% 평균 C-rate = 0.8 / (t80/60)
    """
    m = _POLICY.match(p or "")
    if not m:
        return {"C1": np.nan, "Q1": np.nan, "C2": np.nan, "t80_min": np.nan, "avgC_80": np.nan}
    c1, q1, c2 = float(m.group(1)), float(m.group(2)), float(m.group(3))
    t80 = 60 * (q1 / 100) / c1 + 60 * (max(80 - q1, 0) / 100) / c2
    return {"C1": c1, "Q1": q1, "C2": c2, "t80_min": t80, "avgC_80": 0.8 / (t80 / 60)}


@lru_cache(maxsize=None)
def load_batch(name: str) -> list[dict]:
    with open(PROC / f"{name}.pkl", "rb") as fh:
        return pickle.load(fh)


def load_all(names=BATCH_NAMES) -> list[dict]:
    return [c for n in names for c in load_batch(n)]


def protocol_group(policy: str) -> str:
    p = (policy or "").upper()
    if "VARCHARGE" in p:
        return "varcharge"
    if "SLOWCYCLE" in p:
        return "slowcycle"
    if "NEWSTRUCTURE" in p:
        return "newstructure"
    return "fastcharge" if _POLICY.match(policy or "") else "other"


def is_censored(c: dict) -> bool:
    """cycle_life 값은 있지만 기록 종료 시점에 아직 EOL(0.88 Ah)에 도달하지 않은 셀.

    마지막 20 사이클의 QD 최솟값이 0.885 Ah 초과면 '중도절단(censored)' — cycle_life 는 실제 수명의 하한값.
    """
    q = c["summary"]["QD"].to_numpy()
    return bool(np.isfinite(c["cycle_life"]) and np.nanmin(q[-20:]) > EOL_AH + 0.005)


def cell_table(cells: list[dict]) -> pd.DataFrame:
    rows = []
    for c in cells:
        s = c["summary"]
        cens = is_censored(c)
        life = c["cycle_life"]
        rows.append({
            "cell_key": c["cell_key"], "batch": c["batch"], "idx": c["idx"],
            "policy": c["policy_readable"], "group": protocol_group(c["policy_readable"]),
            "cycle_life": life, "n_cycles": len(s), "QD_last": float(s["QD"].iloc[-1]),
            "censored": cens,
            # 지도학습에 쓸 수 있는 셀: 수명 라벨이 있고 실제로 EOL 에 도달한 셀
            "labeled": bool(np.isfinite(life) and not cens),
            "label_550": (int(life >= LABEL_THRESHOLD) if np.isfinite(life) else np.nan),
            **parse_policy(c["policy_readable"]),
        })
    return pd.DataFrame(rows)


# ── 사이클 번호 규칙 ─────────────────────────────────────────────
# summary 의 k번째 행 = cycle k+1 (summary['cycle'] 이 1부터 시작)
# Qdlin/Tdlin/dQdV 의 k번째 행 = summary 의 k번째 행과 같은 사이클
# Batch 1 의 cycle 1 은 비어 있는 더미(QD=0, Qdlin 없음)

def row_of(cycle: int) -> int:
    return cycle - 1


def qdlin(c: dict, cycle: int) -> np.ndarray:
    return c["Qdlin"][row_of(cycle)].astype(float)


def delta_q(c: dict, a: int = 100, b: int = 10) -> np.ndarray:
    """노션: '사이클 100번 - 사이클 10번의 Q(V) 차이' → ΔQ_{a-b}(V) (1000 포인트)."""
    return qdlin(c, a) - qdlin(c, b)


def clean_summary(s: pd.DataFrame) -> pd.DataFrame:
    """측정 오류로 보이는 값을 NaN 으로 바꾼 summary 사본 (행은 지우지 않아 사이클 번호 유지).

    - QD/QC == 0 (Batch 1 cycle 1 더미 등), QD/QC > 1.2 Ah (정격 1.1 Ah 를 넘는 스파이크)
    - IR == 0 (측정 누락)
    - chargetime > 100 분 (정상 10~60분대, 480/960/3933 같은 기록 오류)
    - 온도 == 0 (측정 누락)
    """
    s = s.copy()
    for col in ("QD", "QC"):
        s.loc[(s[col] <= 0) | (s[col] > 1.2), col] = np.nan
    s.loc[s["IR"] <= 0, "IR"] = np.nan
    s.loc[(s["chargetime"] <= 0) | (s["chargetime"] > 100), "chargetime"] = np.nan
    for col in ("Tmax", "Tavg", "Tmin"):
        s.loc[s[col] <= 0, col] = np.nan
    return s
