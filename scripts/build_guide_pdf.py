"""
Build the comprehensive PDF user guide, with real dashboard screenshots.

What it does:
  1. captures screenshots of the live dashboard tabs (via headless Edge, using
     the ?tab= deep-link) into docs/img/,
  2. converts the markdown guides (GUIDE, ACTIVE_LEARNING, HYPERPARAMETERS,
     PORTABILITY) to a single styled HTML document, inserting the screenshots
     at the start of the matching sections,
  3. prints that HTML to docs/SmartSDG_Guide.pdf (headless Edge).

Requires: the dashboard server running on 127.0.0.1:8765, and Microsoft Edge.
Run:  python scripts\\build_guide_pdf.py
"""

import subprocess
import sys
import time
import urllib.request
from pathlib import Path

import markdown  # pip install markdown

ROOT = Path(__file__).resolve().parent.parent
DOCS = ROOT / "docs"
IMG = DOCS / "img"
PDF_OUT = DOCS / "SmartSDG_Guide.pdf"
SERVER = "http://127.0.0.1:8765"

EDGE_CANDIDATES = [
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
]

# tab -> (filename, caption) for screenshots
TABS = [
    ("sdg",       "Synthetic Data Generation — pick components, scene and quality filters"),
    ("train",     "Object Detection Training — per-camera, record mode"),
    ("inference", "Inference — Images mode (BBoxes) or Live Camera (webcam/IP cam/video)"),
    ("al",        "Active Learning — Harvest → Review candidates → AL experiment + gain chart"),
    ("datasets",  "Datasets — browse, preview boxes, edit labels, delete"),
]

# markdown files in order, each becomes a chapter
CHAPTERS = ["GUIDE.md", "LIGHTING_REALISM.md", "ACTIVE_LEARNING.md", "HYPERPARAMETERS.md", "PORTABILITY.md"]


def find_edge():
    for c in EDGE_CANDIDATES:
        if Path(c).exists():
            return c
    raise SystemExit("Microsoft Edge not found — cannot render screenshots/PDF.")


def server_up():
    try:
        urllib.request.urlopen(SERVER, timeout=3)
        return True
    except Exception:
        return False


def capture_screenshots(edge):
    IMG.mkdir(parents=True, exist_ok=True)
    shots = {}
    for tab, caption in TABS:
        out = IMG / f"tab_{tab}.png"
        if out.exists():
            out.unlink()
        url = f"{SERVER}/?tab={tab}"
        try:
            # unique user-data-dir per shot so instances don't share/【hang】on a lock
            subprocess.run([edge, "--headless=new", "--disable-gpu", "--hide-scrollbars",
                            "--no-first-run", "--no-default-browser-check",
                            f"--user-data-dir={IMG / ('_ud_' + tab)}",
                            f"--screenshot={out}", "--window-size=1280,900",
                            "--virtual-time-budget=3000", url],
                           capture_output=True, timeout=40)
        except subprocess.TimeoutExpired:
            print(f"  WARNING: '{tab}' screenshot timed out")
        time.sleep(0.3)
        if out.exists() and out.stat().st_size > 0:
            shots[tab] = (out, caption)
            print(f"  captured {tab} ({out.stat().st_size//1024} KB)")
        else:
            print(f"  WARNING: screenshot for '{tab}' failed")
    # clean the throwaway user-data dirs
    for d in IMG.glob("_ud_*"):
        try:
            __import__("shutil").rmtree(d, ignore_errors=True)
        except Exception:
            pass
    return shots


def build_html(shots):
    css = """
    body{font-family:'Segoe UI',Arial,sans-serif;color:#1a1a1a;max-width:900px;margin:0 auto;padding:24px;line-height:1.55}
    h1{color:#2b6a00;border-bottom:3px solid #76b900;padding-bottom:6px;margin-top:36px}
    h2{color:#2b6a00;border-bottom:1px solid #ccc;padding-bottom:3px;margin-top:28px}
    h3{color:#444}
    code{background:#f2f2f2;padding:1px 5px;border-radius:3px;font-family:Consolas,monospace;font-size:.9em}
    pre{background:#0d1017;color:#e6e9ef;padding:12px;border-radius:6px;overflow:auto;font-size:.82em;line-height:1.4}
    pre code{background:none;color:inherit;padding:0}
    table{border-collapse:collapse;width:100%;margin:10px 0;font-size:.9em}
    th,td{border:1px solid #ccc;padding:6px 9px;text-align:left}
    th{background:#eef6e0}
    img.shot{width:100%;border:1px solid #bbb;border-radius:6px;margin:8px 0 2px}
    .cap{font-size:.82em;color:#666;margin-bottom:14px;font-style:italic}
    .cover{text-align:center;margin:80px 0 60px}
    .cover h1{font-size:2.4em;border:none}
    .cover p{color:#555}
    @page{margin:18mm}
    """
    md = markdown.Markdown(extensions=["tables", "fenced_code", "toc"])
    parts = [f"<html><head><meta charset='utf-8'><style>{css}</style></head><body>"]
    # cover
    ver = (ROOT / "VERSION").read_text().strip() if (ROOT / "VERSION").exists() else ""
    parts.append(f"""<div class="cover">
      <h1>SmartSDG Pipeline</h1>
      <h2 style="border:none;color:#444">Comprehensive User Guide</h2>
      <p>Synthetic Data Generation → Training → Inference → Active Learning</p>
      <p>Version {ver} &nbsp;·&nbsp; generated {time.strftime('%Y-%m-%d')}</p>
    </div>""")

    # screenshot gallery up front
    if shots:
        parts.append("<h1>The dashboard at a glance</h1>")
        for tab, caption in TABS:
            if tab in shots:
                out, cap = shots[tab]
                rel = out.as_uri()
                parts.append(f'<img class="shot" src="{rel}"><div class="cap">{cap}</div>')

    # chapters from markdown
    for fn in CHAPTERS:
        f = DOCS / fn
        if f.exists():
            md.reset()
            parts.append(md.convert(f.read_text(encoding="utf-8")))

    parts.append("</body></html>")
    html_path = DOCS / "_guide_build.html"
    html_path.write_text("\n".join(parts), encoding="utf-8")
    return html_path


def main():
    edge = find_edge()
    if not server_up():
        print("Dashboard server not running on 127.0.0.1:8765 — start it first "
              "(scripts\\launch_dashboard.bat). Building PDF without live screenshots.")
        shots = {}
    else:
        print("Capturing dashboard screenshots...")
        shots = capture_screenshots(edge)

    print("Assembling HTML...")
    html = build_html(shots)
    print("Printing to PDF (Edge)...")
    if PDF_OUT.exists():
        PDF_OUT.unlink()
    try:
        subprocess.run([edge, "--headless=new", "--disable-gpu",
                        "--no-first-run", "--no-default-browser-check",
                        f"--user-data-dir={DOCS / '_ud_pdf'}",
                        f"--print-to-pdf={PDF_OUT}", "--no-pdf-header-footer",
                        html.as_uri()], capture_output=True, timeout=90)
    except subprocess.TimeoutExpired:
        print("  WARNING: PDF print timed out")
    __import__("shutil").rmtree(DOCS / "_ud_pdf", ignore_errors=True)
    time.sleep(1)
    if PDF_OUT.exists():
        print(f"DONE: {PDF_OUT}  ({PDF_OUT.stat().st_size//1024} KB)")
    else:
        print("ERROR: PDF was not produced.")
        sys.exit(1)


if __name__ == "__main__":
    main()
