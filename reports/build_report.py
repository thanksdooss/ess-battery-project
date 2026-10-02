"""reports/*.md (보고서 원고) → 디자인된 A4 PDF.

사용법: python reports/build_report.py reports/day1_design_report.md "DS-MINI-Design-울산_2반-김진녕.pdf"
필요: markdown-it-py, Google Chrome (headless print-to-pdf), Pretendard 폰트(없으면 시스템 한글 폰트)
"""
from __future__ import annotations

import html as H
import re
import subprocess
import sys
from pathlib import Path

from markdown_it import MarkdownIt

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

META = {
    "kicker": "SKALA DATA SCIENCE · MINI PROJECT",
    "course": "MIT–Stanford Battery Dataset (Severson et al., Nature Energy 2019) · Batch 1 / 2 / 3",
    "who": "울산 2반 · 김진녕 (1인)",
    "date": "2026. 10. 01",
}

CSS = r"""
@page { size: A4; margin: 19mm 16mm 17mm 16mm;
  @top-left { content: "ESS 배터리 수명 예측 · 모델 설계 전략 (DAY 1)"; font-family: Pretendard; font-size: 7.4pt; color: #64748B; vertical-align: bottom; padding-bottom: 4mm; }
  @top-right { content: "울산 2반 · 김진녕"; font-family: Pretendard; font-size: 7.4pt; color: #94A3B8; vertical-align: bottom; padding-bottom: 4mm; }
  @bottom-right { content: counter(page) " / " counter(pages); font-family: Pretendard; font-size: 7.8pt; font-weight: 700; color: #334155; vertical-align: top; padding-top: 4mm; }
  @bottom-left { content: "SKALA DS Mini Project"; font-family: Pretendard; font-size: 7.2pt; color: #94A3B8; vertical-align: top; padding-top: 4mm; }
}
@page cover { margin: 0; @top-left{content:none} @top-right{content:none} @bottom-left{content:none} @bottom-right{content:none} }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
:root { --ink:#0B1220; --text:#1E293B; --muted:#64748B; --line:#E2E8F0; --accent:#0F766E; --accent2:#0EA5A4; --tint:#ECFDF9; --warn:#B45309; }
body { margin:0; font-family: Pretendard, "Apple SD Gothic Neo", sans-serif; font-size: 9.2pt; line-height: 1.62; color: var(--text); word-break: keep-all; overflow-wrap: break-word; }
p { margin: 0 0 6pt; orphans: 3; widows: 3; }
strong { color: var(--ink); }
code { font-family: Menlo, Pretendard, monospace; font-size: .86em; background:#F1F5F9; border:.5pt solid #E2E8F0; border-radius:3pt; padding:0 3pt; }
ul, ol { margin: 3pt 0 7pt; padding-left: 15pt; } li { margin: 1.5pt 0; } li::marker { color: var(--accent); font-weight: 700; }
a { color: var(--accent); text-decoration: none; }

/* cover */
.cover { page: cover; width:210mm; height:297mm; position:relative; overflow:hidden; color:#fff; padding: 26mm 20mm 18mm;
  background: radial-gradient(120mm 110mm at 88% 10%, rgba(14,165,164,.35), transparent 70%), radial-gradient(130mm 120mm at 0% 100%, rgba(37,99,235,.30), transparent 70%), linear-gradient(160deg,#06101F 0%,#0B1F33 55%,#0F2E40 100%);
  display:flex; flex-direction:column; }
.cover .k { font-size: 8.5pt; letter-spacing: 3pt; font-weight: 700; color: #5EEAD4; }
.cover .t { margin-top: 70mm; font-size: 34pt; font-weight: 800; letter-spacing: -1pt; line-height: 1.15; }
.cover .t span { color: #5EEAD4; }
.cover .s { margin-top: 8pt; font-size: 14pt; color: #CBD5E1; font-weight: 600; }
.cover .curve { position:absolute; right: 12mm; top: 30mm; width: 120mm; opacity: .9; }
.cover .box { margin-top: 18mm; display:grid; grid-template-columns: repeat(3,1fr); gap: 7pt; }
.cover .box div { border: .6pt solid rgba(148,163,184,.3); background: rgba(255,255,255,.04); border-radius: 7pt; padding: 8pt 10pt; font-size: 8pt; color:#CBD5E1; }
.cover .box b { display:block; font-size: 12pt; color:#fff; margin-bottom: 2pt; }
.cover .foot { margin-top:auto; border-top: .6pt solid rgba(148,163,184,.35); padding-top: 8pt; display:flex; justify-content:space-between; font-size: 8.5pt; color:#94A3B8; }
.cover .foot b { color:#E2E8F0; }

/* sections */
section.sec { break-before: page; }
section.sec.cont { break-before: auto; margin-top: 14pt; }
.sec-h { display:flex; align-items: baseline; gap: 9pt; margin: 0 0 10pt; padding-bottom: 7pt; border-bottom: 2pt solid var(--accent); }
.sec-h .no { font-size: 20pt; font-weight: 900; color: var(--accent2); letter-spacing: -1pt; min-width: 24pt; }
.sec-h h2 { margin:0; font-size: 15.5pt; font-weight: 800; color: var(--ink); letter-spacing: -.4pt; line-height: 1.3; }
.sec-h .tag { margin-left:auto; font-size: 7.5pt; font-weight: 800; letter-spacing: 1.5pt; color: var(--accent); white-space: nowrap; }
h3 { font-size: 11.2pt; margin: 13pt 0 6pt; color: var(--ink); padding-left: 7pt; border-left: 3pt solid var(--accent2); break-after: avoid; }
h4 { font-size: 10pt; margin: 10pt 0 5pt; color: var(--ink); break-after: avoid; }

/* tables */
table { width:100%; border-collapse: collapse; font-size: 8.1pt; line-height: 1.45; margin: 5pt 0 10pt; }
thead th { background: var(--tint); color: var(--ink); font-weight: 700; text-align: left; padding: 4.5pt 6pt; border-bottom: 1.2pt solid var(--accent); }
tbody td { padding: 4pt 6pt; border-bottom: .5pt solid var(--line); vertical-align: top; }
tbody tr:nth-child(even) td { background: #FAFCFC; }
tbody td:first-child { font-weight: 600; color: var(--ink); }
tr { break-inside: avoid; }
table.mini td:first-child { width: 17%; color: var(--accent); font-weight: 800; }

/* figures */
figure { margin: 6pt 0 9pt; break-inside: avoid; text-align: center; }
figure img { max-width: 100%; max-height: 118mm; object-fit: contain; border: .5pt solid var(--line); border-radius: 5pt; }
figcaption { font-size: 7.8pt; color: var(--muted); margin-top: 3pt; }
figcaption b { color: var(--accent); }

/* callouts */
blockquote { margin: 6pt 0 9pt; padding: 7pt 11pt; background: #F8FAFC; border-left: 3pt solid #94A3B8; border-radius: 0 6pt 6pt 0; font-size: 8.7pt; break-inside: avoid; }
blockquote p:last-child { margin-bottom: 0; }
p.ess { background: var(--tint); border: .6pt solid #99F6E4; border-radius: 6pt; padding: 6pt 10pt; font-size: 8.7pt; break-inside: avoid; }
p.warn { background: #FFFBEB; border: .6pt solid #FCD34D; border-radius: 6pt; padding: 6pt 10pt; font-size: 8.7pt; }
hr { border:none; border-top: 1pt dashed var(--line); margin: 10pt 0; }
"""

COVER_SVG = """<svg class="curve" viewBox="0 0 400 260" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#5EEAD4"/><stop offset="1" stop-color="#60A5FA"/></linearGradient></defs>
<g stroke="rgba(148,163,184,.18)" stroke-width="1">""" + "".join(
    f'<line x1="0" y1="{y}" x2="400" y2="{y}"/>' for y in range(20, 260, 40)) + """</g>
<path d="M10,60 C120,62 190,66 250,78 C300,90 330,120 360,210" fill="none" stroke="url(#g)" stroke-width="4"/>
<path d="M10,60 C100,63 150,72 190,92 C220,110 238,150 252,225" fill="none" stroke="#F87171" stroke-width="3" opacity=".85"/>
<line x1="0" y1="190" x2="400" y2="190" stroke="#FBBF24" stroke-width="1.5" stroke-dasharray="6 5"/>
<text x="392" y="184" text-anchor="end" font-family="Pretendard" font-size="11" fill="#FBBF24">EOL 0.88 Ah</text>
<text x="14" y="50" font-family="Pretendard" font-size="11" fill="#CBD5E1">방전 용량 Qd</text>
</svg>"""

CODE_SPAN = re.compile(r"`[^`]*`")
BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")


def fix_bold(line: str) -> str:
    keep = []

    def stash(m):
        keep.append(m.group(0))
        return f"\x00{len(keep) - 1}\x00"
    t = CODE_SPAN.sub(stash, line)
    t = BOLD.sub(lambda m: f"<strong>{m.group(1)}</strong>", t)
    return re.sub(r"\x00(\d+)\x00", lambda m: keep[int(m.group(1))], t)


def md_to_html(md_text: str, base: Path) -> tuple[str, str]:
    lines, out, in_code = md_text.splitlines(), [], False
    title = ""
    for ln in lines:
        if ln.startswith("```"):
            in_code = not in_code
        if not in_code and not title and ln.startswith("# "):
            title = ln[2:].strip()
            continue
        out.append(ln if in_code else fix_bold(ln))
    md = MarkdownIt("commonmark", {"html": True}).enable("table")
    body = md.render("\n".join(out))

    def img(m):
        alt, src = H.unescape(m.group(2)), m.group(1)
        p = (base / src).resolve() if not src.startswith(("/", "file:")) else Path(src.replace("file://", ""))
        return f'<figure><img src="{p.as_uri()}" alt=""><figcaption>{alt}</figcaption></figure>'
    body = re.sub(r'<p><img src="([^"]+)" alt="([^"]*)" ?/?></p>', img, body)
    body = re.sub(r"<p>(<strong>ESS 관점:?</strong>)", r'<p class="ess">\1', body)
    body = re.sub(r"<p>(<strong>(?:주의|노션 경고|한계)[^<]*</strong>)", r'<p class="warn">\1', body)
    # 4행 미니 표(목적/해결 방법/비교/시사점) 표시
    body = re.sub(r"<table>(\s*<thead>\s*<tr>\s*<th>(?:구분|단계|항목)?</th>)", r'<table class="mini">\1', body)

    parts = re.split(r"(<h2>.*?</h2>)", body)
    secs = [parts[0]] if parts[0].strip() else []
    for i in range(1, len(parts), 2):
        h = re.sub(r"<[^>]+>", "", parts[i]).strip()
        m = re.match(r"(\d+)\.\s*(.*)", h)
        no, name = (m.group(1), m.group(2)) if m else ("", h)
        tag = "EDA" if no in {"2", "3", "4", "5", "6"} else ("STRATEGY" if no in {"7", "8", "9", "10", "11"} else ("DATA" if no == "1" else "SUMMARY"))
        secs.append(f'<section class="sec"><div class="sec-h"><span class="no">{no}</span><h2>{name}</h2>'
                    f'<span class="tag">{tag}</span></div>{parts[i + 1]}</section>')
    return title, "".join(secs)


def build(md_path: Path, pdf_name: str) -> Path:
    title, body = md_to_html(md_path.read_text(), md_path.parent)
    t_main, _, t_sub = title.partition("—")
    cover = f"""<section class="cover">{COVER_SVG}
<div class="k">{META['kicker']}</div>
<div class="t">{H.escape(t_main.strip())}<br><span>{H.escape(t_sub.strip())}</span></div>
<div class="s">{META['course']}</div>
<div class="box"><div><b>Regression</b>초기 100 사이클 → cycle_life</div>
<div><b>Batch 1 → Batch 2</b>학습 → 테스트 (Batch 3 선택)</div>
<div><b>ΔQ<sub>100−10</sub>(V)</b>핵심 신호 · log–log 선형</div></div>
<div class="foot"><div><b>{META['who']}</b> · 제출 파일: {H.escape(pdf_name)}</div><div><b>{META['date']}</b></div></div></section>"""
    html_text = (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{H.escape(title)}</title>'
                 f"<style>{CSS}</style></head><body>{cover}{body}</body></html>")
    html_path = md_path.with_suffix(".html")
    html_path.write_text(html_text)
    out = md_path.parent / pdf_name
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--generate-pdf-document-outline",
                    "--run-all-compositor-stages-before-draw", "--virtual-time-budget=8000",
                    f"--print-to-pdf={out}", html_path.as_uri()], check=True, capture_output=True)
    return out


if __name__ == "__main__":
    print(build(Path(sys.argv[1]).resolve(), sys.argv[2]))
