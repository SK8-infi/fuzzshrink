"""Rewrite the five tables in body.tex in the base paper's full-width table style.

Run from docs/paper2/latex:  python make_tables.py
"""

from pathlib import Path

BS = "\\"
NL = BS + BS  # LaTeX line break / end of row


def head(*lines):
    """Stacked, left-aligned column header."""
    return BS + "multicolumn{1}{l}{" + BS + "tabhead{" + NL.join(lines) + "}}"


def table(label, caption, cols, header, rows):
    out = [BS + "begin{table}[ht]",
           BS + "caption{" + caption + "}" + BS + "label{" + label + "}",
           BS + "begin{tabular*}{" + BS + "textwidth}{@{" + BS + "extracolsep{" + BS + "fill}}" + cols + "@{}}",
           BS + "tabletoprule",
           " & ".join(header) + " " + NL,
           BS + "midrule"]
    out += [" & ".join(r) + " " + NL for r in rows]
    out += [BS + "botrule", BS + "end{tabular*}", BS + "end{table}"]
    return "\n".join(out)


PCT, DOL, PM, DIV = BS + "%", BS + "$", "$" + BS + "pm$", "$" + BS + "div$"

TABLES = [
    table("tab1", "Annual returns printed in the original paper and the returns implied by its own final-wealth figures (initial wealth " + DOL + "1,000; T7 and T10 are its Tables~7 and 10)",
          "lrrrr",
          [BS + "tabhead{Row (table in the" + NL + "original paper)}", head("Reported", "return (" + PCT + ")"),
           head("Implied by", "wealth (" + PCT + ")"), head("Reported", "Sharpe ratio"),
           head("Reported return", DIV + " reported risk")],
          [["Full model, highest-return pick (T7)", "565.89", "434.4", "11.93", "11.93"],
           ["Full model, highest-Sharpe pick (T7)", "388.45", "295.8", "8.95", "8.95"],
           ["Full model, lowest-risk pick (T7)", "178.26", "134.5", "6.65", "6.65"],
           ["Alternative Model 1 (T10)", "189.00", "136.9", "6.20", "5.91"],
           ["Alternative Model 2 (T10)", "23.61", "$-0.05$", "0.59", "0.59"],
           ["Alternative Model 3 (T10)", "88.14", "55.9", "2.74", "2.74"],
           ["Nasdaq Composite (T7)", "1.37", "30.4", "0.35", "0.35"],
           ["Nasdaq-100 (T7)", "1.23", "26.6", "0.31", "0.31"]]),
    table("tab2", "Nasdaq Composite and Nasdaq-100 in 2024, recomputed by us from daily index data", "lrrrrrr",
          ["Index", head("Monthly", "mean (" + PCT + ")"), head("Monthly", "s.d. (" + PCT + ")"),
           head("2024", "return (" + PCT + ")"), head("Wealth after", "costs (" + DOL + ")"),
           head("Sharpe,", "Eq.~15"), head("Standard", "Sharpe")],
          [["Nasdaq Composite", "2.18", "3.35", "28.6", "1,241", "2.54", "2.16"],
           ["Nasdaq-100", "1.92", "3.22", "24.9", "1,205", "2.30", "1.98"]]),
    table("tab3", "Wealth at the end of 2024 from an initial " + DOL + "1,000 under the two selection rules (average of five seeds)", "lrr",
          ["", head("Selected on 2024", "(the paper's rule)"),
           head("Selected on 2023,", "then run on 2024")],
          [["Our reconstruction", "2,543", "1,201"],
           ["QQQ (same transaction costs)", "--", "1,211"],
           ["Equal-weighted, all 85 stocks", "--", "1,121"],
           [BS + "citet{dastkhan2026}, as reported", "5,344", "--"]]),
    table("tab4", "Results for 2019--2024 when the configuration is re-chosen every January from the previous year's wealth (five seeds; "
          + PM + " gives the standard deviation across seeds)", "lrrr",
          ["Pipeline", head("Final wealth (" + DOL + ")"), head("Sharpe ratio"), head("Maximum drawdown (" + PCT + ")")],
          [["Original pipeline", "9,569 " + PM + " 4,193", "0.97 " + PM + " 0.15", "$-44.2$"],
           ["Alternative 1: random selection", "2,166 " + PM + " 1,097", "0.48 " + PM + " 0.26", "$-45.3$"],
           ["Alternative 2: historical means", "5,033 " + PM + " 1,326", "1.00 " + PM + " 0.16", "$-34.6$"],
           ["Alternative 3: equal weights", "5,491 " + PM + " 420", "0.92 " + PM + " 0.03", "$-49.3$"],
           ["QQQ", "2,776", "0.93", "$-35.0$"],
           ["Equal-weighted, all 85 stocks", "3,082", "1.05", "$-27.4$"]]),
    table("tab5", "Tests of the original paper's decomposition claims over 2019--2024, using HAC $t$-tests on monthly return "
          "differences. ``Supported (raw)'' passes the pre-registered rule before Holm adjustment but not after it", "lrrrl",
          ["Claim", head("Return gap", "(pp per year)"), head("Median $p$"), head("Holm $p$"), "Verdict"],
          [["Clustering outperforms random picks", "30.3", "0.058", "0.29", "Not supported"],
           ["LSTM outperforms historical means", "15.0", "0.094", "0.38", "Not supported"],
           ["GA weighting outperforms 1/$N$", "11.0", "0.105", "0.38", "Not supported"],
           ["Full pipeline outperforms QQQ", "27.4", "0.038", "0.23", "Supported (raw)"]]),
]

p = Path("body.tex")
t = p.read_text(encoding="utf-8")
begin, end = BS + "begin{table}", BS + "end{table}"
starts = [i for i in range(len(t)) if t.startswith(begin, i)]
assert len(starts) == 5, len(starts)
parts, pos = [], 0
for s, new in zip(starts, TABLES):
    e = t.index(end, s) + len(end)
    parts += [t[pos:s], new]
    pos = e
parts.append(t[pos:])
p.write_text("".join(parts), encoding="utf-8")
print("rewrote", len(TABLES), "tables")
