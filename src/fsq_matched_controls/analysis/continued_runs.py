"""Supplement S4.1 / S8: Table I's masked-prediction tests with the capped runs continued to early stopping.

    python -m fsq_matched_controls.analysis.continued_runs [--compare]     -> outputs/continued_runs_matched_tests_estonia.csv

Table I's statistics (letter_runs.analyse, unchanged) with the masked-prediction token scores of the continued runs substituted:
    LETTER                the capped runs, as in the letter;
    E1 only               the 19 continued CropGlobe runs replace their capped originals (best checkpoint);
    P (E1 + E8c, best)    and the two continued Estonia runs (seeds 4242 and 97, best checkpoint);
    S2 (E1 + E8c, final)  the same with the final-epoch checkpoints.
Reconstruction runs are never substituted. Inputs: results/supplement_S4_sensitivity/continued_runs_probe.csv (CropGlobe; columns
new_* = best checkpoint, final_* = final epoch) and continued_runs_estonia_probe_best.csv / _final.csv (Estonia).
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from fsq_matched_controls import paths
from fsq_matched_controls.analysis import letter_runs as L

ROW_OF = {("transductive", "FRA"): "France, transductive", ("transductive", "AUS"): "Australia, transductive",
          ("strict_FRA", "FRA"): "France, strict", ("strict_AUS", "AUS"): "Australia, strict"}
STAGES = {"LETTER": (False, False, False), "E1 only": (True, False, False), "P (E1 + E8c, best)": (True, True, False),
          "S2 (E1 + E8c, final)": (True, True, True)}


def _inputs():
    d = paths.SUPPLEMENT_S4
    cg_cont = pd.read_csv(d / "continued_runs_probe.csv", dtype={"seed": str}); cg_cont = cg_cont[cg_cont.repr_kind == "tokens"]
    ec_best = pd.read_csv(d / "continued_runs_estonia_probe_best.csv", dtype={"seed": str}); ec_best = ec_best[ec_best.repr_kind == "tokens"]
    ec_final = pd.read_csv(d / "continued_runs_estonia_probe_final.csv", dtype={"seed": str}); ec_final = ec_final[ec_final.repr_kind == "tokens"]
    return cg_cont, ec_best, ec_final


def build(use_cg, use_ec, final, cg_cont, ec_best, ec_final):
    data = L.collect("all")
    tags = {r: list(L.runs_df(r, "mp", "all").tag) for r in L.ROWS}
    if use_cg:
        for _, x in cg_cont.iterrows():
            row = ROW_OF[(x.setting, x.region)]; seed = str(x.seed)
            tag = "1904-new" if (x.setting == "transductive" and seed == "1904") else seed
            j = tags[row].index(tag)
            data[row]["mp"][j, 0] = float(x.final_acc if final else x.new_acc); data[row]["mp"][j, 1] = float(x.final_f1 if final else x.new_f1)
    if use_ec:
        for _, x in (ec_final if final else ec_best).iterrows():
            j = tags["Estonia"].index(str(x.seed))
            data["Estonia"]["mp"][j, 0] = float(x.accuracy); data["Estonia"]["mp"][j, 1] = float(x.macro_f1)
    return data


def table():
    cg_cont, ec_best, ec_final = _inputs()
    res = {k: L.analyse(build(*v, cg_cont, ec_best, ec_final), L.LOOKS) for k, v in STAGES.items()}
    return pd.DataFrame([dict(stage=st, row=r, col=c, metric=m, margin=L.find(res[st][0], r, c, m)["margin"], p_raw=L.find(res[st][0], r, c, m)["p_raw"],
                              p_holm=L.find(res[st][0], r, c, m)["p_holm"], holm_sig=L.find(res[st][0], r, c, m)["h_col"], verdict=res[st][1][(r, c)])
                         for st in STAGES for r in L.ROWS for c in ("MP-PCA", "MP-kmeans", "MP-recon") for m in ("acc", "f1")])


SETNAME = {"France, transductive": ("transductive CropGlobe", "FRA"), "Australia, transductive": ("transductive CropGlobe", "AUS"),
           "France, strict": ("strict FRA", "FRA"), "Australia, strict": ("strict AUS", "AUS"), "Estonia": ("EuroCropsML (Estonia)", "Estonia")}


def _holm(p):
    p = np.asarray(p, float); o = np.argsort(p); n = len(p); adj = np.empty(n); run = 0.0
    for rank, i in enumerate(o):
        run = max(run, (n - rank) * p[i]); adj[i] = min(1.0, run)
    return adj


def decomposition():
    """The decomposition of Table II for masked prediction against the band-value twin (a'), with the continued runs:
    (b) continuous latent before rounding, (c) tokens; stages LETTER, E1-BEST / E1-FINAL (CropGlobe), E8c-BEST / E8c-FINAL (Estonia);
    twin OLD (capped) and NEW (continued). Two-sided Welch of (c) and of (b) against (a'), Holm over the five rows per stage, twin and metric."""
    T = paths.TABLE2
    cg_cont = pd.read_csv(paths.SUPPLEMENT_S4 / "continued_runs_probe.csv", dtype={"seed": str})
    tr = pd.read_csv(T / "twin_probe_rows_transductive.csv"); ex = pd.read_csv(T / "twin_probe_rows_strict_estonia.csv")
    b4 = pd.read_csv(T / "bands_twin_continued_probe_best.csv")
    ec_cont = {"BEST": pd.read_csv(paths.SUPPLEMENT_S4 / "continued_runs_estonia_probe_best.csv", dtype={"seed": str}),
           "FINAL": pd.read_csv(paths.SUPPLEMENT_S4 / "continued_runs_estonia_probe_final.csv", dtype={"seed": str})}
    src = {"France, transductive": (tr, None, "FRA"), "Australia, transductive": (tr, None, "AUS"), "France, strict": (ex, "FRA", "FRA"),
           "Australia, strict": (ex, "AUS", "AUS"), "Estonia": (ex, "EC", "Estonia")}

    def twin(row, new):
        d, setting, reg = src[row]; d = d[(d.kind == "twin_bands") & (d.region == reg) & (d["repr"] == "continuous")]
        if setting is not None:
            d = d[d.setting == setting]
        o = d.set_index("seed")[["accuracy", "macro_f1"]].sort_index().copy()
        if new:
            st, rg = SETNAME[row]; n = b4[(b4.setting == st) & (b4.region == rg) & (b4["repr"] == "continuous")].set_index("seed")
            for s in n.index:
                o.loc[s, ["accuracy", "macro_f1"]] = n.loc[s, ["accuracy", "macro_f1"]].values
        return o

    data = {}
    for row in L.ROWS:
        f = L.runs_df(row, "mp", "all"); tags = list(f.tag)
        base_b = f[["cont_acc", "cont_f1"]].values.copy(); base_c = f[["tok_acc", "tok_f1"]].values.copy()
        stages = {"LETTER": (base_b, base_c, 0)}
        name = "E1" if row != "Estonia" else "E8c"
        for stage, (ca, cf) in (("BEST", ("new_acc", "new_f1")), ("FINAL", ("final_acc", "final_f1"))):
            b = base_b.copy(); c = base_c.copy(); n = 0
            if name == "E8c":
                for _, x in ec_cont[stage].iterrows():
                    j = tags.index(str(x.seed))
                    (b if x.repr_kind == "continuous" else c)[j] = (x.accuracy, x.macro_f1); n += int(x.repr_kind == "tokens")
            else:
                for _, x in cg_cont.iterrows():
                    if ROW_OF.get((x.setting, x.region)) != row:
                        continue
                    seed = str(x.seed); tag = "1904-new" if (x.setting == "transductive" and seed == "1904") else seed; j = tags.index(tag)
                    (b if x.repr_kind == "continuous" else c)[j] = (x[ca], x[cf]); n += int(x.repr_kind == "tokens")
            stages[f"{name}-{stage}"] = (b, c, n)
        for st, (b, c, n) in stages.items():
            data[(row, st)] = dict(b=b, c=c, nrep=n)
    rec = []
    for (row, st), d in data.items():
        for tw in ("OLD", "NEW"):
            A = twin(row, tw == "NEW")
            for mi, m in enumerate(("accuracy", "macro_f1")):
                a = A[m].values; b = d["b"][:, mi]; c = d["c"][:, mi]
                rec.append(dict(row=row, stage=st, twin=tw, metric=m, n_replaced=d["nrep"], mean_b=b.mean(), mean_c=c.mean(), rounding_c_minus_b=c.mean() - b.mean(),
                                mean_a=a.mean(), step_b_minus_a=b.mean() - a.mean(), disc_c_minus_a=c.mean() - a.mean(),
                                welch_p_ca=float(stats.ttest_ind(c, a, equal_var=False).pvalue), welch_p_ba=float(stats.ttest_ind(b, a, equal_var=False).pvalue)))
    R = pd.DataFrame(rec)
    for tag in ("ca", "ba"):
        R[f"holm_p_{tag}"] = np.nan
    for st in sorted({s for _, s in data}):
        for tw in ("OLD", "NEW"):
            for m in ("accuracy", "macro_f1"):
                pick = [R[(R.row == row) & (R.stage == (st if (row, st) in data else "LETTER")) & (R.twin == tw) & (R.metric == m)].index[0] for row in L.ROWS]
                for tag in ("ca", "ba"):
                    for ix, ad in zip(pick, _holm(R.loc[pick, f"welch_p_{tag}"].values)):
                        R.loc[ix, f"holm_p_{tag}@{st}"] = ad
    return R


LOW = {("transductive", s) for s in ("71", "11", "97", "1904", "23", "37")} | {("strict_FRA", s) for s in ("101", "4242", "71")}


def _probe_rows():
    # round_trip: the stored values are the original doubles only when parsed exactly
    P = pd.read_csv(paths.SUPPLEMENT_S4 / "continued_runs_probe.csv", dtype={"seed": str}, float_precision="round_trip")
    P["low_run"] = P.low_run.astype(bool)
    return P


def catchup():
    """France token scores of the continued CropGlobe runs against the k-means control: the nine low runs and the strong runs, and
    group means (Australia and the continuous latent descriptive)."""
    P = _probe_rows()
    fra = P[(P.repr_kind == "tokens") & (P.region == "FRA")]
    cat = []
    for grp, d in (("low run", fra[fra.low_run]), ("extended strong run", fra[~fra.low_run])):
        for _, x in d.iterrows():
            ka, kf = L.CTL[ROW_OF[(x.setting, "FRA")]]["kmeans"]
            cat.append(dict(group=grp, setting=x.setting, seed=x.seed, kmeans_control_acc=ka, kmeans_control_f1=kf, old_acc=x.old_acc, new_best_acc=x.new_acc,
                            final_acc=x.final_acc, exceeds_kmeans_best=bool(x.new_acc > ka), exceeds_kmeans_final=bool(x.final_acc > ka), old_f1=x.old_f1,
                            new_best_f1=x.new_f1, final_f1=x.final_f1, exceeds_kmeans_f1_best=bool(x.new_f1 > kf), d_acc=x.d_acc, d_f1=x.d_f1,
                            d_final_acc=x.d_final_acc, d_final_f1=x.d_final_f1, old_best_epoch=x.old_best_epoch, new_best_epoch=x.new_best_epoch,
                            new_best_is_old_best=x.new_best_is_old_best))
    C = pd.DataFrame(cat)
    summ = []
    for grp, d in C.groupby("group"):
        s = dict(group=f"MEAN over {len(d)} {grp}s, France tokens", setting="", seed="", n=len(d))
        for c in ("old_acc", "new_best_acc", "final_acc", "old_f1", "new_best_f1", "final_f1", "d_acc", "d_f1", "d_final_acc", "d_final_f1"):
            s[c] = d[c].mean()
        s["exceeds_kmeans_best"] = int(d.exceeds_kmeans_best.sum()); s["exceeds_kmeans_final"] = int(d.exceeds_kmeans_final.sum())
        summ.append(s)
    for kind, region in (("tokens", "AUS"), ("continuous", "FRA"), ("continuous", "AUS")):
        d0 = P[(P.repr_kind == kind) & (P.region == region)]
        for grp, mask in (("low runs", d0.low_run), ("extended strong runs", ~d0.low_run & ((d0.setting == "transductive") | (d0.setting == "strict_" + region)))):
            d = d0[mask]
            if grp == "low runs":
                d = d[(d.setting == "transductive") | (d.setting == "strict_" + region)]
            d = d[[(s, sd) in LOW if grp == "low runs" else (s, sd) not in LOW for s, sd in zip(d.setting, d.seed)]]
            if len(d):
                summ.append(dict(group=f"MEAN over {len(d)} {grp}, {region} {kind} (descriptive)", n=len(d), old_acc=d.old_acc.mean(), new_best_acc=d.new_acc.mean(),
                                 final_acc=d.final_acc.mean(), old_f1=d.old_f1.mean(), new_best_f1=d.new_f1.mean(), final_f1=d.final_f1.mean(), d_acc=d.d_acc.mean(),
                                 d_f1=d.d_f1.mean(), d_final_acc=d.d_final_acc.mean(), d_final_f1=d.d_final_f1.mean()))
    CC = pd.concat([C.astype(object), pd.DataFrame(summ).astype(object)], ignore_index=True)
    for col in ("exceeds_kmeans_best", "exceeds_kmeans_final", "exceeds_kmeans_f1_best", "new_best_is_old_best"):
        if col in CC:
            CC[col] = [(str(bool(v)) if not str(g).startswith("MEAN") else v) if not (isinstance(v, float) and np.isnan(v)) else "" for v, g in zip(CC[col], CC["group"])]
    CC["seed"] = ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in CC["seed"]]
    CC["setting"] = ["" if (isinstance(v, float) and np.isnan(v)) else str(v) for v in CC["setting"]]
    return CC


def _analyse_signflip(data, looks):
    from fsq_matched_controls.analysis.signflip_tests import signflip_p
    from fsq_matched_controls.stats.tests import holm_adjust, welch_greater
    out = []
    for r in L.ROWS:
        Pm, Rc = data[r]["mp"], data[r]["recon"]
        for col, arr_, c in (("MP-PCA", Pm, "PCA"), ("MP-kmeans", Pm, "kmeans"), ("R-PCA", Rc, "PCA"), ("R-kmeans", Rc, "kmeans")):
            for i, met in enumerate(("acc", "f1")):
                p = signflip_p(arr_[:, i] - L.CTL[r][c][i])
                out.append(dict(panel=r, col=col, metric=met, p_raw=p, p=min(1.0, p * looks[r])))
        for i, met in enumerate(("acc", "f1")):
            p = welch_greater(Pm[:, i], Rc[:, i])
            out.append(dict(panel=r, col="MP-recon", metric=met, p_raw=p, p=min(1.0, p * looks[r])))
    for col in L.COLS:
        for met in ("acc", "f1"):
            fam = [x for x in out if x["col"] == col and x["metric"] == met]
            for x, ad in zip(fam, holm_adjust([x["p"] for x in fam])):
                x["p_holm"] = float(ad); x["h_col"] = bool(ad <= L.ALPHA)
    v = {}
    for x in out:
        v.setdefault((x["panel"], x["col"]), []).append(x["h_col"])
    return out, {k: ("E" if all(b) else ("S" if any(b) else "N")) for k, b in v.items()}


def scenarios():
    """Table I with the continued CropGlobe runs, t and sign-flip: L letter; P all 19 continued runs (best checkpoint);
    S1 only the nine low runs (best); S2 all 19 (final-epoch checkpoint). Estonia and reconstruction stay the letter's."""
    P = _probe_rows(); tok = P[P.repr_kind == "tokens"]
    tags = {r: list(L.runs_df(r, "mp", "all").tag) for r in L.ROWS}
    base = L.collect("all")

    def scenario(which, flt):
        data = {r: dict(mp=base[r]["mp"].copy(), recon=base[r]["recon"]) for r in L.ROWS}
        n = 0
        for _, x in tok.iterrows():
            if not flt(x.setting, x.seed):
                continue
            row = ROW_OF[(x.setting, x.region)]
            j = tags[row].index("1904-new" if (x.setting == "transductive" and x.seed == "1904") else x.seed)
            data[row]["mp"][j, 0] = x.new_acc if which == "best" else x.final_acc
            data[row]["mp"][j, 1] = x.new_f1 if which == "best" else x.final_f1
            n += 1
        return data, n

    SC = {"L": ("letter as is", None, None), "P": ("all 19 extended runs, best checkpoint", "best", lambda s, sd: True),
          "S1": ("only the nine low runs, best checkpoint", "best", lambda s, sd: (s, sd) in LOW),
          "S2": ("all 19 extended runs, final-epoch checkpoint", "final", lambda s, sd: True)}
    long = []
    for k, (desc, which, flt) in SC.items():
        data, nrep = (base, 0) if which is None else scenario(which, flt)
        data = {r: dict(mp=data[r]["mp"], recon=data[r]["recon"]) for r in L.ROWS}
        rows_t, v_t, _ = L.analyse(data, L.LOOKS)
        rows_p, v_p = _analyse_signflip(data, L.LOOKS)
        for r in L.ROWS:
            reg_, setting = (r.split(", ") + [""])[:2] if "," in r else (r, "strict")
            for c in L.COLS:
                for met in ("acc", "f1"):
                    xt = L.find(rows_t, r, c, met); xp = next(z for z in rows_p if z["panel"] == r and z["col"] == c and z["metric"] == met)
                    long.append(dict(scenario=k, scenario_desc=desc, runs_replaced=nrep, region=reg_, setting=setting, comparison=c, metric=met,
                                     mean_margin=xt["margin"], t_p=xt["p_raw"], signflip_p=xp["p_raw"], t_p_adj=xt["p_holm"], signflip_p_adj=xp["p_holm"],
                                     t_survives=bool(xt["h_col"]), signflip_survives=bool(xp["h_col"]), verdict_t=v_t[(r, c)], verdict_signflip=v_p[(r, c)],
                                     runs_above_control=xt["above"],
                                     note=("Welch, same under both tests" if c == "MP-recon" else ("reconstruction unchanged" if c.startswith("R-") else ""))))
    return pd.DataFrame(long)


def main():
    paths.OUTPUT.mkdir(parents=True, exist_ok=True)
    catchup().to_csv(paths.OUTPUT / "continued_runs_catchup.csv", index=False)
    scenarios().to_csv(paths.OUTPUT / "continued_runs_matched_tests.csv", index=False)
    df = table()
    out = paths.OUTPUT / "continued_runs_matched_tests_estonia.csv"
    df.to_csv(out, index=False)
    v = df.drop_duplicates(["stage", "row", "col"]).pivot(index=["row", "col"], columns="stage", values="verdict")[list(STAGES)]
    print(v.to_string())
    out2 = paths.OUTPUT / "continued_runs_decomposition.csv"
    decomposition().to_csv(out2, index=False)
    print(f"-> {out}, {out2}")


def compare(outputs):
    """Regenerated tables against the stored ones (analysis.compare_tables)."""
    from fsq_matched_controls.analysis.compare_tables import report
    ok = report([(paths.OUTPUT / f"{name}.csv", folder / f"{name}.csv") for name, folder in outputs])
    print("PASS" if ok else "FAIL")
    return ok


if __name__ == "__main__":
    import sys
    main()
    if "--compare" in sys.argv:
        sys.exit(0 if compare([(n, paths.SUPPLEMENT_S4) for n in ("continued_runs_matched_tests_estonia", "continued_runs_decomposition", "continued_runs_catchup", "continued_runs_matched_tests")]) else 1)
