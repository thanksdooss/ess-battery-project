"""그래프 공통 스타일 — 한글 폰트, 배치별 색상, 저장 함수."""
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib import font_manager  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
FIG = ROOT / "reports" / "figures"
FIG.mkdir(parents=True, exist_ok=True)

for _p in (Path.home() / "Library" / "Fonts").glob("Pretendard-*.otf"):
    font_manager.fontManager.addfont(str(_p))
_fonts = {f.name for f in font_manager.fontManager.ttflist}
_KR = next((n for n in ("Pretendard", "AppleGothic", "Apple SD Gothic Neo", "NanumGothic") if n in _fonts), "DejaVu Sans")

plt.rcParams.update({
    "font.family": _KR, "axes.unicode_minus": False, "figure.dpi": 110, "savefig.dpi": 180,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.25,
    "axes.titleweight": "bold", "axes.titlesize": 12, "axes.labelsize": 10.5, "legend.frameon": False,
})

BATCH_COLOR = {"batch1": "#2563EB", "batch2": "#EA580C", "batch3": "#059669"}
BATCH_LABEL = {"batch1": "Batch 1 (2017-05-12, 학습)", "batch2": "Batch 2 (2018-02-20, 테스트)",
               "batch3": "Batch 3 (2018-04-12, 추가 검증)"}
LONG, SHORT = "#2563EB", "#DC2626"   # 장수명 / 단수명


def save(fig, name: str) -> Path:
    p = FIG / f"{name}.png"
    fig.savefig(p, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return p
