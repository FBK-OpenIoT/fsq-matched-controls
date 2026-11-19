"""Smoke test: regenerate Table I, Table II and Figure 1 from results/ and compare them with the letter.

    python -m fsq_matched_controls.analysis.smoke_test

1. Table I and Table II against the values printed in the letter (analysis/expected_letter_values.csv, see
   expected_letter_values.py): every number (as printed, three decimals), every raw p (Table I) and every bold flag.
2. Byte-identity of the regenerated CSVs against the original analysis scripts' outputs shipped in results/:
   twin_step_tests.csv, teacher_twin_continued_step_tests.csv, bands_twin_continued_step_tests.csv, and Table I's S7.1 records
   (results/letter_numbers/B.csv, item S71).
3. Figure 1: outputs/figure1_column.pdf against manuscript/figure1_column.pdf (bytes), and the figure's per-run constants against
   results/runs/n_probe_new_*.csv.
Exit status 0 only if every check passes.
"""
from __future__ import annotations

import filecmp
import glob
import sys

import pandas as pd

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import expected_letter_values, figure1, figure1_data, figure1_data_pre_ten_runs, step_tests, table1, table2, twins_continued
from fsq_matched_controls.analysis import letter_runs as L

def _fmt3(x, signed=True):
    s = f"{x:+.3f}" if signed else f"{x:.3f}"
    return s.replace("-0.000", "0.000").replace("+0.000", "0.000") if s in ("-0.000", "+0.000") else s


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    fails, notes = [], []
    expected = expected_letter_values.load()

    # ---- 1a. Table I
    T1, rows_p, vp, vp_pool = table1.table1()
    T1.to_csv(paths.OUTPUT / "table1.csv", index=False)
    s71 = table1.s71_records(rows_p, vp, vp_pool); s71.to_csv(paths.OUTPUT / "table1_s71.csv", index=False)
    e1 = expected[expected.table == "I"]; n1 = 0
    for _, e in e1.iterrows():
        g = T1[(T1.region == e.region) & (T1.setting == e.row) & (T1.metric == e.metric)]
        where = f"Table I {e.region} {e.row} {e.metric}"
        if e["item"] in ("control_pca", "control_kmeans"):
            n1 += 1; v = float(g[e["item"]].iloc[0]); got = f"{v:.3f}"
            if got != e.value:
                diff = abs(v - float(e.value))
                (notes if diff <= 0.0005 + 1e-9 else fails).append(f"{where} {e['item'].replace('_', ' ')}: regenerated {v:.4f} prints as {got}, letter {e.value} (|diff| {diff:.4f})")
            continue
        x = g[g.column == e["item"]].iloc[0]; n1 += 1
        if _fmt3(x.margin) != e.value:
            fails.append(f"{where} {e['item']}: margin {_fmt3(x.margin)} vs letter {e.value}")
        if int(bool(x.bold)) != int(e.bold):
            fails.append(f"{where} {e['item']}: bold {x.bold} vs letter {e.bold}")
        if e.p:
            n1 += 1
            if L.pfmt(x.p_raw) != e.p:
                fails.append(f"{where} {e['item']}: raw p {L.pfmt(x.p_raw)} vs letter {e.p}")
    printed = e1.groupby(["region", "row", "metric"]).ngroups
    print(f"Table I: {printed} printed rows, {n1} printed values and p-values compared (bold flags beside).")

    # ---- 1b. Table II
    T2 = table2.table2(); T2.to_csv(paths.OUTPUT / "table2.csv", index=False)
    n2 = 0
    for _, e in expected[expected.table == "II"].iterrows():
        x = T2[(T2.region == e.region) & (T2.model == e.row) & (T2.metric == e.metric)].iloc[0]; n2 += 1
        got = _fmt3(x[e["item"]])
        if got != e.value:
            fails.append(f"Table II {e.region} {e.row} {e['item']} {e.metric}: {got} vs letter {e.value}")
        if e["item"] == "tokens_minus_no_quantizer" and int(bool(x.bold)) != int(e.bold):
            fails.append(f"Table II {e.region} {e.row} {e.metric}: bold {x.bold} vs letter {e.bold}")
    print(f"Table II: {n2} printed values compared (bold flags beside).")

    # ---- 2. byte identity with the original scripts' outputs
    step_tests.main(); twins_continued.main()
    for out, orig in (("twin_step_tests.csv", paths.TABLE2 / "twin_step_tests.csv"), ("teacher_twin_continued_step_tests.csv", paths.TABLE2 / "teacher_twin_continued_step_tests.csv"),
                      ("bands_twin_continued_step_tests.csv", paths.TABLE2 / "bands_twin_continued_step_tests.csv")):
        same = filecmp.cmp(paths.OUTPUT / out, orig, shallow=False)
        print(f"byte-identical with the original output {orig.relative_to(paths.RESULTS)}: {same}")
        if not same:
            fails.append(f"{out} differs from {orig}")
    b = pd.read_csv(paths.LETTER_NUMBERS / "B.csv"); b = b[b["item"] == "S71"][s71.columns].reset_index(drop=True)
    a = pd.read_csv(paths.OUTPUT / "table1_s71.csv")
    same = bool((a.astype(str).values == b.astype(str).values).all()) and a.shape == b.shape
    print(f"Table I S7.1 records identical to the original output letter_numbers/B.csv (item S71, {a.size} cells): {same}")
    if not same:
        fails.append("table1_s71 differs from B.csv S71 records")

    # ---- 3. Figure 1
    figure1.main(["--palette", "okabe-ito", "--layout", "column", "--au-label", "Queensland", "--group-titles"])
    same = filecmp.cmp(paths.OUTPUT / "figure1_column.pdf", paths.MANUSCRIPT / "figure1_column.pdf", shallow=False)
    print(f"Figure 1 PDF byte-identical with manuscript/figure1_column.pdf: {same}")
    if not same:
        fails.append("figure1_column.pdf differs")
    new = pd.concat([pd.read_csv(p) for p in sorted(glob.glob(str(paths.RUN_RESULTS / "n_probe_new_*of*.csv")))]); new = new[new.repr_kind == "tokens"]; new["tag"] = new["tag"].astype(str)
    nfig = 0
    for row, (setting, reg) in L.SETTING.items():
        for obj, key in L.KEY.items():
            old = figure1_data_pre_ten_runs.PANELS[row][key]
            for tag, v in figure1_data.PANELS[row][key].items():
                nfig += 1
                if tag in old:
                    want = old[tag]
                else:
                    q = new[(new.objective == obj) & (new.setting == setting) & (new.region == reg) & (new.tag == tag)].iloc[0]
                    want = (round(float(q.accuracy), 4), round(float(q.macro_f1), 4))
                if tuple(v) != tuple(want):
                    fails.append(f"Figure 1 data {row} {key} {tag}: {v} vs per-run table {want}")
    print(f"Figure 1 per-run constants checked against the per-run tables: {nfig}")

    print("\nNotes (display rounding only):" if notes else "")
    for x in notes:
        print("  " + x)
    print(f"\n{'PASS' if not fails else 'FAIL'}: {len(fails)} mismatches")
    for x in fails:
        print("  " + x)
    return 0 if not fails else 1


if __name__ == "__main__":
    sys.exit(main())
