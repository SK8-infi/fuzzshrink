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
    table("tab1", "Reported annual return against the return implied by the reported final wealth (" + DOL + "1,000 start)",
          "lrrrr",
          [BS + "tabhead{Row (table in the" + NL + "original paper)}", head("Reported", "return (" + PCT + ")"),
           head("Implied by", "wealth (" + PCT + ")"), head("Reported", "Sharpe ratio"),
           head("Reported return", DIV + " reported risk")],
          [["Model, maximum return (T7)", "565.89", "434.4", "11.93", "11.93"],
           ["Model, maximum Sharpe ratio (T7)", "388.45", "295.8", "8.95", "8.95"],
           ["Model, minimum risk (T7)", "178.26", "134.5", "6.65", "6.65"],
           ["Alternative Model 1 (T10)", "189.00", "136.9", "6.20", "5.91"],
           ["Alternative Model 2 (T10)", "23.61", "$-0.05$", "0.59", "0.59"],
           ["Alternative Model 3 (T10)", "88.14", "55.9", "2.74", "2.74"],
           ["Nasdaq Composite (T7)", "1.37", "30.4", "0.35", "0.35"],
           ["Nasdaq-100 (T7)", "1.23", "26.6", "0.31", "0.31"]]),
    table("tab2", "The 2024 benchmarks recomputed from index data", "lrrrrrr",
          ["Index", head("Monthly", "mean (" + PCT + ")"), head("Monthly", "s.d. (" + PCT + ")"),
           head("2024", "return (" + PCT + ")"), head("Wealth after", "costs (" + DOL + ")"),
           head("Sharpe,", "Eq.~15"), head("Standard", "Sharpe")],
          [["Nasdaq Composite", "2.18", "3.35", "28.6", "1,241", "2.54", "2.16"],
           ["Nasdaq-100", "1.92", "3.22", "24.9", "1,205", "2.30", "1.98"]]),
    table("tab3", "Final wealth in 2024 from " + DOL + "1,000 (mean over five seeds)", "lrr",
          ["", head("Configuration chosen on", "2024 (as published)"),
           head("Configuration chosen on", "2023, applied to 2024")],
          [["Reconstruction of the pipeline", "2,543", "1,201"],
           ["QQQ, same costs", "--", "1,211"],
           ["Equal weight, 85 stocks", "--", "1,121"],
           ["Published result", "5,344", "--"]]),
    table("tab4", "2019--2024 with the configuration re-chosen each year on the previous year (five seeds; "
          + PM + " is the standard deviation across seeds)", "lrrr",
          ["Pipeline", head("Final wealth (" + DOL + ")"), head("Sharpe ratio"), head("Maximum drawdown (" + PCT + ")")],
          [["Original pipeline", "9,569 " + PM + " 4,193", "0.97 " + PM + " 0.15", "$-44.2$"],
           ["Alternative 1: random selection", "2,166 " + PM + " 1,097", "0.48 " + PM + " 0.26", "$-45.3$"],
           ["Alternative 2: historical means", "5,033 " + PM + " 1,326", "1.00 " + PM + " 0.16", "$-34.6$"],
           ["Alternative 3: equal weights", "5,491 " + PM + " 420", "0.92 " + PM + " 0.03", "$-49.3$"],
           ["QQQ", "2,776", "0.93", "$-35.0$"],
           ["Equal weight, 85 stocks", "3,082", "1.05", "$-27.4$"]]),
    table("tab5", "The original paper's decomposition claims, 2019--2024 (HAC $t$-tests on monthly return "
          "differences). ``Supported (raw)'' meets the pre-registered rule but not after Holm adjustment", "lrrrl",
          ["Claim", head("Return gap", "(pp per year)"), head("Median $p$"), head("Holm $p$"), "Verdict"],
          [["Clustering beats random selection", "30.3", "0.058", "0.29", "Not supported"],
           ["LSTM beats historical means", "15.0", "0.094", "0.38", "Not supported"],
           ["Genetic algorithm beats equal weights", "11.0", "0.105", "0.38", "Not supported"],
           ["Pipeline beats QQQ", "27.4", "0.038", "0.23", "Supported (raw)"]]),
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
