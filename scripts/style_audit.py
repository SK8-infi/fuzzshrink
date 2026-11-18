"""Style audit for the paper: flags patterns typical of machine-generated or formulaic prose.

Usage: python scripts/style_audit.py [path to body.tex] [path to paper.tex]
Reports counts per pattern and every flagged sentence, plus sentence-length variation ("burstiness").
"""

import re
import statistics
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
BODY = Path(ARGS[0]) if len(ARGS) > 0 else ROOT / "docs/paper2/latex/body.tex"
FRONT = Path(ARGS[1]) if len(ARGS) > 1 else ROOT / "docs/paper2/latex/paper.tex"


def plain(tex: str) -> str:
    tex = re.sub(r"%.*", "", tex)
    tex = re.sub(r"\\begin\{(table|figure|list)\}.*?\\end\{\1\}", " ", tex, flags=re.S)
    tex = re.sub(r"\\cite[pt]?\{[^}]*\}", "(CITE)", tex)
    tex = re.sub(r"\\(ref|label)\{[^}]*\}", "N", tex)
    tex = re.sub(r"\\(section|subsection|bmhead)\*?\{([^}]*)\}", r"\n\n\2.\n\n", tex)
    tex = re.sub(r"\\(textit|textbf|emph)\{([^}]*)\}", r"\2", tex)
    tex = tex.replace("~", " ").replace("--", "-").replace("\\%", "%").replace("\\$", "$")
    tex = re.sub(r"\\[a-zA-Z]+\*?(\[[^]]*\])?", "", tex)
    return re.sub(r"[{}]", "", tex)


front = FRONT.read_text(encoding="utf-8")
abstract = re.search(r"\\articleabstract\{(.*?)\}\n", front, flags=re.S)
text = plain((abstract.group(1) if abstract else "") + "\n\n" + BODY.read_text(encoding="utf-8"))
sentences = [s.strip() for s in re.split(r"(?<=[.?!])\s+(?=[A-Z(])", " ".join(text.split())) if len(s.split()) > 3]

PATTERNS = {
    # vocabulary that language models overuse
    "stock AI vocabulary": r"\b(delve|crucial|pivotal|robust(?!ness)|landscape|realm|tapestry|underscore[sd]?|showcase[sd]?|"
                           r"leverag\w+|seamless\w*|intricate|multifaceted|nuanced|holistic|paramount|navigat\w+|"
                           r"foster\w*|elevate[sd]?|unlock\w*|harness\w*|noteworthy|commendable|meticulous\w*)\b",
    "connective padding": r"\b(Furthermore|Moreover|Additionally|Notably|Importantly|Crucially|Ultimately|Overall|In essence|"
                          r"In summary|In conclusion|It is worth noting|It is important to note)\b",
    "intensifier": r"\b(precisely|exactly|clearly|truly|indeed|simply|very|highly|deeply|fundamentally|genuinely)\b",
    "'not X but Y' / 'rather than' frame": r"\b(not [^,.;]{1,40}, but|not merely|not only|rather than)\b",
    "rhetorical question": r"\?",
    "colon inside sentence": r"\w: \w",
    "semicolon": r";",
    "dash aside": r" - |\u2014",
    "enumeration First/Second/Third": r"\b(First|Second|Third|Fourth|Finally),",
    "value-laden 'honest'": r"\bhonest(ly)?\b",
    "aphoristic closer": r"\b(The (broader|real|key|central) (lesson|point|question)|By that standard|In other words|Put simply|Put differently)\b",
    "vague 'This' opener": r"^(This|That|These) (is|was|shows|means|makes|combination)\b",
    "hedge pile-up": r"\b(may|might|could|potentially|arguably|somewhat|largely|perhaps)\b",
}

flags = {name: [] for name in PATTERNS}
for s in sentences:
    for name, pat in PATTERNS.items():
        if re.search(pat, s, flags=0 if "opener" in name else re.I):
            flags[name].append(s)

# triplets "A, B and C" / "A, B, and C" (the tricolon habit)
tricolon = [s for s in sentences if len(re.findall(r"\b\w+(?: \w+){0,3}, \w+(?: \w+){0,3},? and \w+", s)) >= 1]
flags["tricolon list (A, B and C)"] = tricolon

lengths = [len(s.split()) for s in sentences]
print(f"sentences: {len(sentences)}   words: {sum(lengths)}")
print(f"sentence length: mean {statistics.mean(lengths):.1f}, sd {statistics.pstdev(lengths):.1f}, "
      f"coefficient of variation {statistics.pstdev(lengths) / statistics.mean(lengths):.2f} "
      f"(human academic prose is usually 0.45-0.65; flat prose is below 0.40)")
print(f"short sentences (<10 words): {sum(l < 10 for l in lengths)}   long (>35 words): {sum(l > 35 for l in lengths)}\n")
for name, hits in flags.items():
    print(f"{len(hits):4d}  {name}  ({100 * len(hits) / len(sentences):.0f}% of sentences)")
if "--show" in sys.argv:
    for name, hits in flags.items():
        if hits:
            print(f"\n=== {name}")
            for s in hits:
                print("  -", s[:220])
