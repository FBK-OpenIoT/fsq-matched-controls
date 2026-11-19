"""The values printed in the letter's Table I and Table II, as the smoke test's reference.

    python -m fsq_matched_controls.analysis.expected_letter_values --tex <letter .tex> [--out <csv>]

expected_letter_values.csv (next to this file) was generated once with this command from the LaTeX source of the submitted
letter; the letter itself is not part of the repository. One row per printed cell:
    table      I or II
    region     France, Queensland, Estonia
    row        Table I: the setting (transductive, strict); Table II: the model row
    metric     Table I: acc, f1; Table II: accuracy, macro-F1
    item       Table I: control_pca, control_kmeans or the column of the margin; Table II: the column
    value      as printed (three decimals; "-0.000" and "+0.000" read as "0.000")
    bold       1 if printed bold (significant), else 0
    p          Table I: the raw p printed beside the margin ("<0.001" kept as printed), empty where none is printed
"""
from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd

CSV = Path(__file__).with_name("expected_letter_values.csv")
NUM = re.compile(r"[+-]?\d\.\d+")


def _norm(s):
    s = s.replace("$", "").replace("\\,", "").strip()
    return "0.000" if s in ("-0.000", "+0.000") else s


def parse_table1(tex):
    body = tex.split("\\label{tab:matched}")[1].split("\\bottomrule")[0]   # the header cells hold nested tabulars
    region = setting = None; out = []
    for line in body.split("\n"):
        m = re.search(r"\\multirow\{\d\}\{\*\}\{(France|Queensland|Estonia)\}", line)
        if m:
            region = m.group(1)
        m = re.search(r"\\multirow\{2\}\{\*\}\{(transductive|strict)\}", line)
        if m:
            setting = m.group(1)
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if len(cells) == 10 and cells[2] in ("Acc.", "F1"):
            met = "acc" if cells[2] == "Acc." else "f1"
            vals = []
            for c in cells[5:]:
                v = NUM.search(c.replace("$", "")).group(0)
                p = re.search(r"\(\$?<?\$?([\d.]+)\)", c)
                plt = "($<$" in c
                vals.append(dict(value=_norm(v), bold="\\sig{" in c, p=(("<" if plt else "") + p.group(1)) if p else None))
            out.append(dict(region=region, setting=setting, metric=met, pca=cells[3], kmeans=cells[4], cols=vals))
    return out


def parse_table2(tex):
    body = tex.split("\\label{tab:decomposition}")[1].split("\\end{tabular}")[0]
    region = None; out = []
    for line in body.split("\n"):
        m = re.search(r"\\textit\{(France|Queensland|Estonia) \(raw", line)
        if m:
            region = m.group(1); continue
        cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
        if region and len(cells) == 4 and cells[0].replace("\\quad", "").strip() in ("Reconstruction", "MP, slow teacher", "MP, band values", "Rounded PCA"):
            row = dict(region=region, model=cells[0].replace("\\quad", "").strip(), cols=[])
            for c in cells[1:]:
                if not c.strip():
                    row["cols"].append(None); continue
                a, f = c.split("/")
                row["cols"].append(((_norm(NUM.search(a.replace("$", "")).group(0)), "\\sig{" in a), (_norm(NUM.search(f.replace("$", "")).group(0)), "\\sig{" in f)))
            out.append(row)
    return out


TABLE2_COLS = ("no_quantizer_minus_raw", "tokens_minus_no_quantizer", "rounding")


def from_tex(tex):
    from fsq_matched_controls.analysis import table1
    rows = []
    for pr in parse_table1(tex):
        base = dict(table="I", region=pr["region"], row=pr["setting"], metric=pr["metric"])
        rows.append(dict(base, item="control_pca", value=pr["pca"], bold=0, p=""))
        rows.append(dict(base, item="control_kmeans", value=pr["kmeans"], bold=0, p=""))
        for col, cell in zip(table1.LETTER_COLS, pr["cols"]):
            rows.append(dict(base, item=col, value=cell["value"], bold=int(cell["bold"]), p=cell["p"] or ""))
    for pr in parse_table2(tex):
        for key, cell in zip(TABLE2_COLS, pr["cols"]):
            if cell is None:
                continue
            for mname, (v, b) in zip(("accuracy", "macro-F1"), cell):
                rows.append(dict(table="II", region=pr["region"], row=pr["model"], metric=mname, item=key, value=v, bold=int(b), p=""))
    return pd.DataFrame(rows, columns=["table", "region", "row", "metric", "item", "value", "bold", "p"])


def load():
    return pd.read_csv(CSV, dtype=str, keep_default_na=False)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tex", required=True, help="LaTeX source of the letter")
    ap.add_argument("--out", default=str(CSV))
    a = ap.parse_args(argv)
    df = from_tex(Path(a.tex).read_text(encoding="utf-8"))
    df.to_csv(a.out, index=False, lineterminator="\n")
    print(f"{len(df)} printed cells -> {a.out}")


if __name__ == "__main__":
    main()
