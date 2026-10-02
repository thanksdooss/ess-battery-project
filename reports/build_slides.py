"""슬라이드 원고(JSON) → 16:9 PDF.

사용법:
  python reports/build_slides.py reports/day1_slides.json "DS-MINI-Design-울산_2반-김진녕.pdf"
  python reports/build_slides.py reports/day1_slides.json out.pdf --check   # 넘침(overflow) 점검만

원고 형식 (reports/day1_slides.json)
{
  "meta": {"title": "...", "subtitle": "...", "who": "...", "date": "...", "file": "..."},
  "slides": [
    {"type": "cover"},
    {"type": "summary", "headline": "...", "conclusions": [{"title": "...", "text": "..."}], "kpis": [{"value": "...", "label": "...", "note": "..."}]},
    {"type": "eda", "tag": "Q1", "question": "노션 질문 원문", "headline": "...", "issue": "...",
     "figure": "figures/slide_q1.png", "figure_note": "...",
     "points": [{"label": "노션 하위 항목 원문", "text": "..."}], "decision": "..."},
    {"type": "table", "tag": "...", "headline": "...", "issue": "...", "columns": [...], "rows": [[...]],
     "col_widths": [..], "callout": {"title": "...", "text": "..."}, "note": "..."},
    {"type": "split", "tag": "...", "headline": "...", "issue": "...",
     "left": {"kind": "bullets|table|figure|cards", ...}, "right": {...}, "callout": {...}, "note": "..."},
    {"type": "section", "title": "부록", "subtitle": "..."},
    {"type": "figure", "tag": "...", "headline": "...", "issue": "...", "figure": "figures/x.png", "note": "...", "callout": {...}}
  ]
}
텍스트 안의 **굵게**, `코드`, 「노션 인용」은 그대로 쓰면 된다.
선택 키 "decision_label": eda 슬라이드 하단 바 라벨(기본 "시사점 → 결정"). "callout"은 목록이면 여러 개를 차례로 그린다.
선택 키 "fz": 본문(표·불릿·카드·포인트·요약) 글자 배율(기본 1, 1 미만 금지). 여백이 많은 슬라이드를 키워 채울 때 쓴다.
선택 키 "fig_flex": eda 슬라이드 그림 칸 너비 비율(기본 1.45, 오른쪽 포인트 칸 = 1). 큰 그림(히트맵)에 쓴다.
eda 포인트의 "label"은 노션 하위 항목 원문이라 「」로 감싼다. 노션 문구가 아닌 라벨(예: "해석")은 "quote": false.
"""
from __future__ import annotations

import html as H
import json
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CHROME = "/Applications/Google Chrome.app/Contents/MacOS/Google Chrome"

CSS = r"""
@page { size: 338.67mm 190.5mm; margin: 0; }
* { box-sizing: border-box; -webkit-print-color-adjust: exact; print-color-adjust: exact; }
:root { --ink:#0B1220; --text:#1E293B; --muted:#64748B; --faint:#94A3B8; --line:#E2E8F0;
        --accent:#0F766E; --accent2:#14B8A6; --tint:#ECFDF9; --warn:#B45309; --warnbg:#FFFBEB; }
html, body { margin:0; padding:0; }
body { font-family: Pretendard, "Apple SD Gothic Neo", sans-serif; color: var(--text); word-break: keep-all; overflow-wrap: break-word; }
.slide { --fz: 1; width: 338.67mm; height: 190.5mm; position: relative; overflow: hidden; break-after: page; background: #fff;
         padding: 11mm 14mm 13mm; display: flex; flex-direction: column; }
.slide:last-child { break-after: auto; }
.top { display:flex; align-items:center; gap: 8pt; font-size: 10.5pt; color: var(--muted); }
.tag { background: var(--ink); color:#fff; font-weight:800; font-size: 9.5pt; letter-spacing: .8pt; padding: 2.5pt 8pt; border-radius: 4pt; }
.tag.app { background: #475569; }
.q { color: var(--muted); font-weight: 600; }
h1.hl { font-size: 23pt; line-height: 1.25; font-weight: 800; color: var(--ink); letter-spacing: -.6pt; margin: 6pt 0 0; }
.issue { margin-top: 7pt; align-self: flex-start; font-size: 11pt; font-weight: 700; color: var(--accent);
         border: 1.2pt solid var(--accent2); background: var(--tint); border-radius: 20pt; padding: 3pt 12pt; }
.issue b { color: var(--ink); margin-right: 4pt; }
.body { flex: 1; min-height: 0; margin-top: 9pt; display: flex; gap: 14pt; }
.fig { flex: 1.45; min-width: 0; min-height: 0; display:flex; flex-direction:column; }
.fig img { width: 100%; flex: 1; min-height: 0; object-fit: contain; object-position: left top; }
.fig .fn { font-size: 8.5pt; color: var(--faint); margin-top: 3pt; }
.fig.full img { object-position: center top; }
.pts { flex: 1; min-width: 0; display:flex; flex-direction:column; justify-content: center; gap: calc(12pt * var(--fz)); padding-bottom: 6pt; }
.pt { border-left: 3pt solid var(--accent2); padding: 2pt 0 2pt 9pt; }
.pt .lb { font-size: calc(8.8pt * var(--fz)); color: var(--muted); font-weight: 700; margin-bottom: 1.5pt; }
.pt .lb span { color: var(--accent); font-weight: 900; margin-right: 4pt; }
.pt .tx { font-size: calc(12.2pt * var(--fz)); line-height: 1.45; color: var(--text); }
.decision { margin-top: 9pt; background: var(--ink); color: #fff; border-radius: 7pt; padding: 7pt 14pt;
            font-size: 12.8pt; font-weight: 700; display:flex; gap: 9pt; align-items: baseline; }
.decision .k { color: #5EEAD4; font-size: 9.5pt; font-weight: 900; letter-spacing: 1.2pt; white-space: nowrap; }
.foot { position:absolute; left: 14mm; right: 14mm; bottom: 5mm; display:flex; justify-content: space-between; font-size: 8pt; color: var(--faint); }
.foot b { color: var(--muted); }
strong, b { color: var(--ink); }
code { font-family: Menlo, Pretendard, monospace; font-size: .86em; background:#F1F5F9; border-radius: 3pt; padding: 0 3pt; }
/* tables */
table { width: 100%; border-collapse: collapse; font-size: calc(10.8pt * var(--fz)); line-height: 1.38; }
thead th { text-align:left; background: var(--tint); color: var(--ink); font-weight: 800; padding: calc(5pt * var(--fz)) calc(7pt * var(--fz)); border-bottom: 1.4pt solid var(--accent); }
tbody td { padding: calc(5pt * var(--fz)) calc(7pt * var(--fz)); border-bottom: .6pt solid var(--line); vertical-align: top; }
tbody td:first-child { font-weight: 700; color: var(--ink); }
tbody tr:nth-child(even) td { background: #FAFCFC; }
.tblwrap { flex: 1; min-height: 0; }
/* callout */
.callout { margin-top: 9pt; border: 1pt solid #FCD34D; background: var(--warnbg); border-radius: 7pt; padding: calc(7pt * var(--fz)) 12pt; font-size: calc(11pt * var(--fz)); line-height: 1.45; }
.callout .ct { font-weight: 800; color: var(--warn); margin-right: 6pt; }
.callout.info { border-color: #99F6E4; background: var(--tint); }
.callout.info .ct { color: var(--accent); }
.note { font-size: calc(8.8pt * var(--fz)); color: var(--faint); margin-top: 5pt; }
/* split */
.col { flex: 1; min-width: 0; display:flex; flex-direction: column; gap: calc(6pt * var(--fz)); }
.col h3 { margin: 0 0 2pt; font-size: calc(12.5pt * var(--fz)); color: var(--accent); }
ul.bl { margin: 0; padding-left: 15pt; font-size: calc(12pt * var(--fz)); line-height: 1.5; }
ul.bl li { margin: calc(3pt * var(--fz)) 0; }
ul.bl li::marker { color: var(--accent2); }
.cards { display:grid; grid-template-columns: 1fr; gap: calc(7pt * var(--fz)); }
.card { border: 1pt solid var(--line); border-radius: 8pt; padding: calc(8pt * var(--fz)) 11pt; }
.card .ttl { font-size: calc(12.5pt * var(--fz)); font-weight: 800; color: var(--ink); }
.card .txt { font-size: calc(10.8pt * var(--fz)); color: var(--text); margin-top: 2pt; line-height: 1.45; }
.card.on { border-color: var(--accent2); background: var(--tint); }
.card.mid { border-color: #FCD34D; background: var(--warnbg); }
.card.off { background: #F8FAFC; }
/* summary */
.concl { display:grid; grid-template-columns: repeat(3, 1fr); gap: 10pt; margin-top: calc(10pt * var(--fz)); }
.concl .c { border-top: 3pt solid var(--accent2); background: #F8FAFC; border-radius: 0 0 8pt 8pt; padding: calc(9pt * var(--fz)) 12pt; }
.concl .n { font-size: calc(9.5pt * var(--fz)); font-weight: 900; color: var(--accent); letter-spacing: 1pt; }
.concl .t { font-size: calc(14pt * var(--fz)); font-weight: 800; color: var(--ink); margin: 2pt 0 4pt; }
.concl .x { font-size: calc(11pt * var(--fz)); line-height: 1.5; }
.kpis { display:grid; grid-template-columns: repeat(4, 1fr); gap: 10pt; margin-top: calc(11pt * var(--fz)); }
.kpi { border: 1pt solid var(--line); border-radius: 8pt; padding: calc(8pt * var(--fz)) 12pt; }
.kpi .v { font-size: calc(21pt * var(--fz)); font-weight: 900; color: var(--accent); letter-spacing: -.5pt; }
.kpi .l { font-size: calc(10.5pt * var(--fz)); font-weight: 700; color: var(--ink); }
.kpi .nt { font-size: calc(9pt * var(--fz)); color: var(--muted); margin-top: 2pt; }
/* cover & section */
.cover { color:#fff; padding: 22mm 22mm 14mm;
  background: radial-gradient(140mm 120mm at 85% 15%, rgba(20,184,166,.35), transparent 70%), radial-gradient(160mm 120mm at 0% 100%, rgba(37,99,235,.28), transparent 70%), linear-gradient(155deg,#06101F 0%,#0B1F33 55%,#0F2E40 100%); }
.cover .k { font-size: 10pt; letter-spacing: 3pt; font-weight: 800; color: #5EEAD4; }
.cover .t { margin-top: 30mm; font-size: 40pt; font-weight: 900; letter-spacing: -1.2pt; line-height: 1.12; }
.cover .t span { color: #5EEAD4; }
.cover .s { margin-top: 10pt; font-size: 15pt; color: #CBD5E1; font-weight: 600; }
.cover .row { margin-top: 14mm; display:flex; gap: 10pt; }
.cover .row div { border: .8pt solid rgba(148,163,184,.35); background: rgba(255,255,255,.05); border-radius: 8pt; padding: 8pt 12pt; font-size: 10pt; color:#CBD5E1; }
.cover .row b { display:block; color:#fff; font-size: 13pt; }
.cover .ft { margin-top: auto; display:flex; justify-content: space-between; font-size: 10pt; color:#94A3B8; border-top: .8pt solid rgba(148,163,184,.35); padding-top: 8pt; }
.cover .ft b { color: #E2E8F0; }
.cover svg { position:absolute; right: 18mm; top: 22mm; width: 135mm; opacity: .95; }
.section { justify-content: center; background: #0B1220; color:#fff; }
.section .t { font-size: 34pt; font-weight: 900; }
.section .s { font-size: 13pt; color: #94A3B8; margin-top: 8pt; }
"""

COVER_SVG = """<svg viewBox="0 0 400 230" xmlns="http://www.w3.org/2000/svg">
<defs><linearGradient id="g" x1="0" x2="1"><stop offset="0" stop-color="#5EEAD4"/><stop offset="1" stop-color="#60A5FA"/></linearGradient></defs>
<g stroke="rgba(148,163,184,.16)" stroke-width="1">""" + "".join(f'<line x1="0" y1="{y}" x2="400" y2="{y}"/>' for y in range(15, 230, 35)) + """</g>
<rect x="10" y="20" width="62" height="190" fill="rgba(94,234,212,.08)"/>
<text x="14" y="205" font-family="Pretendard" font-size="10" fill="#5EEAD4">초기 100 사이클</text>
<path d="M10,55 C120,57 200,62 260,76 C305,88 335,118 365,195" fill="none" stroke="url(#g)" stroke-width="4"/>
<path d="M10,55 C95,58 150,68 190,88 C220,106 238,148 252,200" fill="none" stroke="#F87171" stroke-width="3" opacity=".9"/>
<line x1="0" y1="170" x2="400" y2="170" stroke="#FBBF24" stroke-width="1.5" stroke-dasharray="6 5"/>
<text x="80" y="163" text-anchor="start" font-family="Pretendard" font-size="11" fill="#FBBF24">EOL 0.88 Ah</text>
<text x="80" y="45" font-family="Pretendard" font-size="11" fill="#CBD5E1">방전 용량 Qd</text>
</svg>"""

CODE = re.compile(r"`([^`]+)`")
BOLD = re.compile(r"\*\*(?=\S)(.+?)(?<=\S)\*\*")
NOBRK = re.compile(r"([)%」])(?=[가-힣])")
NBSP = re.compile(r"(Batch|부록|\d) (?=A?\d|m?Ah\b|V\b)")
DQ_NBSP = re.compile(r"ΔQ (?=[가-힣])")
CMP_NBSP = re.compile(r" ([≤≥]) (?=[−+]?\d)")


def fmt(s) -> str:
    s = H.escape(str(s or ""))
    s = CODE.sub(lambda m: f"<code>{m.group(1)}</code>", s)
    s = BOLD.sub(lambda m: f"<b>{m.group(1)}</b>", s)
    s = NBSP.sub("\\1\u00a0", s)  # "Batch 2", "부록 A1" 안에서 줄바꿈 금지
    s = DQ_NBSP.sub("ΔQ\u00a0", s)  # "ΔQ 깊이"처럼 ΔQ가 줄 끝에 홀로 남지 않게
    s = CMP_NBSP.sub("\u00a0\\1\u00a0", s)  # "|r| ≤ 0.17"을 한 덩어리로
    s = NOBRK.sub("\\1\u2060", s)  # ")는", "%를" 같은 조사 앞 줄바꿈 방지
    return s.replace("\n", "<br>")


def img_src(path: str, base: Path) -> str:
    p = Path(path)
    if not p.is_absolute():
        p = (base / p).resolve()
    return p.as_uri()


def top(s, cls=""):
    q = f'<span class="q">{fmt(s.get("question"))}</span>' if s.get("question") else ""
    return f'<div class="top"><span class="tag {cls}">{fmt(s.get("tag", ""))}</span>{q}</div>'


def head(s, app=False):
    out = top(s, "app" if app else "") + f'<h1 class="hl">{fmt(s.get("headline"))}</h1>'
    if s.get("issue"):
        out += f'<div class="issue"><b>ISSUE)</b>{fmt(s["issue"])}</div>'
    return out


def table_html(cols, rows, widths=None):
    cg = "".join(f'<col style="width:{w}%">' for w in widths) if widths else ""
    th = "".join(f"<th>{fmt(c)}</th>" for c in cols)
    tr = "".join("<tr>" + "".join(f"<td>{fmt(c)}</td>" for c in r) + "</tr>" for r in rows)
    return f"<table><colgroup>{cg}</colgroup><thead><tr>{th}</tr></thead><tbody>{tr}</tbody></table>"


def callout_html(c):
    if not c:
        return ""
    if isinstance(c, list):  # 콜아웃 여러 개를 차례로
        return "".join(callout_html(x) for x in c)
    kind = c.get("kind", "warn")
    return f'<div class="callout {"info" if kind == "info" else ""}"><span class="ct">{fmt(c.get("title"))}</span>{fmt(c.get("text"))}</div>'


def block_html(b, base):
    k = b.get("kind")
    h3 = f"<h3>{fmt(b['title'])}</h3>" if b.get("title") else ""
    if k == "bullets":
        return h3 + '<ul class="bl">' + "".join(f"<li>{fmt(x)}</li>" for x in b["items"]) + "</ul>"
    if k == "table":
        return h3 + table_html(b["columns"], b["rows"], b.get("col_widths"))
    if k == "figure":
        fn = f'<div class="fn">{fmt(b.get("note"))}</div>' if b.get("note") else ""
        return h3 + f'<div class="fig"><img src="{img_src(b["path"], base)}">{fn}</div>'
    if k == "cards":
        return h3 + '<div class="cards">' + "".join(
            f'<div class="card {x.get("style", "")}"><div class="ttl">{fmt(x["title"])}</div><div class="txt">{fmt(x["text"])}</div></div>'
            for x in b["items"]) + "</div>"
    return ""


def foot(i, n, meta):
    return f'<div class="foot"><span><b>{fmt(meta["title"])}</b> · {fmt(meta["who"])}</span><span>{i} / {n}</span></div>'


def render(spec: dict, base: Path) -> str:
    meta, slides = spec["meta"], spec["slides"]
    n = len(slides)
    out = []
    for i, s in enumerate(slides, 1):
        t = s["type"]
        app = s.get("appendix", False)
        fz = max(1.0, float(s.get("fz", 1)))
        sec = f'<section class="slide" style="--fz:{fz:g}">'
        if t == "cover":
            t1, _, t2 = meta["title"].partition("—")
            boxes = "".join(f"<div><b>{fmt(b['t'])}</b>{fmt(b['s'])}</div>" for b in meta.get("cover_boxes", []))
            out.append(f'<section class="slide cover">{COVER_SVG}<div class="k">{fmt(meta.get("kicker"))}</div>'
                       f'<div class="t">{fmt(t1.strip())}<br><span>{fmt(t2.strip())}</span></div><div class="s">{fmt(meta.get("subtitle"))}</div>'
                       f'<div class="row">{boxes}</div><div class="ft"><span><b>{fmt(meta["who"])}</b> · {fmt(meta.get("file"))}</span><b>{fmt(meta.get("date"))}</b></div></section>')
        elif t == "section":
            out.append(f'<section class="slide section"><div class="t">{fmt(s["title"])}</div><div class="s">{fmt(s.get("subtitle"))}</div>{foot(i, n, meta)}</section>')
        elif t == "summary":
            cc = "".join(f'<div class="c"><div class="n">결론 {k}</div><div class="t">{fmt(c["title"])}</div><div class="x">{fmt(c["text"])}</div></div>'
                         for k, c in enumerate(s["conclusions"], 1))
            kp = "".join(f'<div class="kpi"><div class="v">{fmt(k["value"])}</div><div class="l">{fmt(k["label"])}</div><div class="nt">{fmt(k.get("note"))}</div></div>' for k in s.get("kpis", []))
            out.append(sec + f'{head(s)}<div class="concl">{cc}</div><div class="kpis">{kp}</div>{callout_html(s.get("callout"))}{foot(i, n, meta)}</section>')
        elif t == "eda":
            pts = "".join(f'<div class="pt"><div class="lb"><span>{"①②③④⑤"[k]}</span>'
                          + (f'「{fmt(p["label"])}」' if p.get("quote", True) else fmt(p["label"]))
                          + f'</div><div class="tx">{fmt(p["text"])}</div></div>'
                          for k, p in enumerate(s["points"]))
            fn = f'<div class="fn">{fmt(s.get("figure_note"))}</div>' if s.get("figure_note") else ""
            out.append(sec + f'{head(s, app)}<div class="body"><div class="fig" style="flex:{float(s.get("fig_flex", 1.45)):g}"><img src="{img_src(s["figure"], base)}">{fn}</div>'
                       f'<div class="pts">{pts}</div></div><div class="decision"><span class="k">{fmt(s.get("decision_label", "시사점 → 결정"))}</span><span>{fmt(s["decision"])}</span></div>{foot(i, n, meta)}</section>')
        elif t == "table":
            note = f'<div class="note">{fmt(s["note"])}</div>' if s.get("note") else ""
            out.append(sec + f'{head(s, app)}<div class="body"><div class="tblwrap">{table_html(s["columns"], s["rows"], s.get("col_widths"))}{note}</div></div>'
                       f'{callout_html(s.get("callout"))}{foot(i, n, meta)}</section>')
        elif t == "figure":  # 전체 폭 그림 한 장(부록 근거 그림용)
            note = f'<div class="note">{fmt(s["note"])}</div>' if s.get("note") else ""
            out.append(sec + f'{head(s, app)}<div class="body"><div class="fig full"><img src="{img_src(s["figure"], base)}">{note}</div></div>'
                       f'{callout_html(s.get("callout"))}{foot(i, n, meta)}</section>')
        elif t == "split":
            note = f'<div class="note">{fmt(s["note"])}</div>' if s.get("note") else ""
            ratio = s.get("ratio", [1, 1])
            out.append(sec + f'{head(s, app)}<div class="body"><div class="col" style="flex:{ratio[0]}">{block_html(s["left"], base)}</div>'
                       f'<div class="col" style="flex:{ratio[1]}">{block_html(s["right"], base)}{note}</div></div>{callout_html(s.get("callout"))}{foot(i, n, meta)}</section>')
    check_js = """<script>
window.addEventListener('load',()=>{const bad=[];document.querySelectorAll('.slide').forEach((el,i)=>{
 const r=el.getBoundingClientRect(); let worst=0;
 el.querySelectorAll('*').forEach(c=>{const b=c.getBoundingClientRect(); if(b.height>0){worst=Math.max(worst,b.bottom-r.bottom, b.right-r.right);}});
 el.querySelectorAll('.body,.tblwrap,.pts,.col').forEach(c=>{ if(c.scrollHeight>c.clientHeight+2) worst=Math.max(worst,c.scrollHeight-c.clientHeight);});
 if(worst>1) bad.push((i+1)+':'+Math.round(worst));});
 const p=document.createElement('pre'); p.id='overflow-report'; p.textContent=JSON.stringify(bad); p.style.display='none'; document.body.appendChild(p);});
</script>"""
    return (f'<!doctype html><html lang="ko"><head><meta charset="utf-8"><title>{fmt(meta["title"])}</title><style>{CSS}</style></head>'
            f'<body>{"".join(out)}{check_js}</body></html>')


def build(spec_path: Path, pdf_name: str, check_only: bool = False):
    spec = json.loads(spec_path.read_text())
    html_path = spec_path.with_suffix(".html")
    html_path.write_text(render(spec, spec_path.parent))
    dom = subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--virtual-time-budget=6000", "--window-size=1280,720",
                          "--dump-dom", html_path.as_uri()], capture_output=True, text=True).stdout
    m = re.search(r'id="overflow-report"[^>]*>(.*?)</pre>', dom, re.S)
    overflow = json.loads(H.unescape(m.group(1))) if m else ["check-failed"]
    if check_only:
        return None, overflow
    out = spec_path.parent / pdf_name
    subprocess.run([CHROME, "--headless=new", "--disable-gpu", "--no-pdf-header-footer", "--virtual-time-budget=8000",
                    f"--print-to-pdf={out}", html_path.as_uri()], check=True, capture_output=True)
    return out, overflow


if __name__ == "__main__":
    out, ov = build(Path(sys.argv[1]).resolve(), sys.argv[2], "--check" in sys.argv)
    print("PDF:", out)
    print("OVERFLOW (slide:px):", ov)
