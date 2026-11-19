"""Text-overlap check for paper 2: shared 8-word sequences with the base paper, the draft and reference abstracts.

Usage: python scripts/check_text_overlap.py [out.json]   (paths to the base paper and draft PDFs are set below)
Sources: the base paper (full text), the team's earlier draft, and the abstracts of all references
(fetched from OpenAlex by DOI). Reports every shared 8-gram and each source's overlap percentage.
"""
import json
import re
import sys
import urllib.request
from pathlib import Path

from pypdf import PdfReader

N = 8
ROOT = Path(r"C:\Github\Business economics")


def words(text):
    text = text.replace("\u2013", " ").replace("\u2014", " ").replace("-\n", "")
    return re.findall(r"[a-z0-9]+(?:'[a-z]+)?", text.lower())


def grams(ws):
    return {" ".join(ws[i:i + N]) for i in range(len(ws) - N + 1)}


def pdf_text(path, first=0, last=None):
    r = PdfReader(str(path))
    pages = r.pages[first:last]
    return "\n".join(p.extract_text() or "" for p in pages)


# our paper: everything before the reference list
ours_full = pdf_text(ROOT / "docs/paper2/latex/paper.pdf")
ours = ours_full[: ours_full.rfind("References")]
ours_w = words(ours)
ours_g = grams(ours_w)
print(f"our paper: {len(ours_w)} words, {len(ours_g)} distinct {N}-word sequences\n")

sources = {
    "base paper (Dastkhan et al., 2026, full text)": pdf_text(r"C:\Users\shiva\Downloads\s10479-026-07053-9.pdf"),
    "team's earlier draft (dockenf.pdf)": pdf_text(r"C:\Users\shiva\Downloads\dockenf.pdf"),
}

# abstracts of every reference, from OpenAlex
bib = (ROOT / "docs/paper2/latex/references.bib").read_text(encoding="utf-8")
dois = dict(re.findall(r"@\w+\{([^,]+),[^@]*?doi\s*=\s*\{([^}]+)\}", bib, flags=re.S))
abstracts, missing = {}, []
for key, doi in dois.items():
    try:
        req = urllib.request.Request("https://api.openalex.org/works/doi:" + doi,
                                     headers={"User-Agent": "overlap-check/1.0"})
        inv = json.load(urllib.request.urlopen(req, timeout=30)).get("abstract_inverted_index")
        if not inv:
            missing.append(key)
            continue
        pos = sorted((i, w) for w, idx in inv.items() for i in idx)
        abstracts[key] = " ".join(w for _, w in pos)
    except Exception:
        missing.append(key)
for key, text in abstracts.items():
    sources[f"abstract: {key}"] = text
print(f"abstracts retrieved: {len(abstracts)} of {len(dois)} references; not available: {', '.join(missing) or 'none'}\n")

report = []
for name, text in sources.items():
    shared = sorted(ours_g & grams(words(text)))
    pct = 100 * len(shared) / max(1, len(ours_g))
    report.append((pct, name, shared))
for pct, name, shared in sorted(report, reverse=True):
    print(f"{pct:5.2f}%  {len(shared):3d} shared  {name}")
print()
for pct, name, shared in sorted(report, reverse=True):
    if shared:
        print(f"--- {name}")
        for g in shared:
            print("   ", g)
json.dump({n: s for _, n, s in report}, open(sys.argv[1] if len(sys.argv) > 1 else "plag.json", "w"), indent=1)
