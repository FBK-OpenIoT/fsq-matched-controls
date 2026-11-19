"""Figure 1 data of the first runs of each panel (3 or 5 per objective, before the seed extension to ten). Constants only.

analysis.letter_runs reads these token scores for the earlier runs (as the letter's analysis did) and takes the later runs'
scores from results/runs/n_probe_new_*.csv; the control, raw-band and majority values are the same as in figure1_data.py.
"""

# ---- per-seed dots (tokens, logistic probe, harmonized) ----
PANELS = {
    "France, transductive": dict(
        predictive={"1904-new": (0.6665, 0.5622), "2024": (0.7557, 0.7608), "4242": (0.7322, 0.7290)},
        reconstruction={"42": (0.6284, 0.5732), "101": (0.6088, 0.5433), "1904": (0.6258, 0.5516)},
        pca=(0.6771, 0.6043), kmeans=(0.6960, 0.6274), raw=(0.7817, 0.7907), majority=(0.3012, 0.0926),
    ),
    "Australia, transductive": dict(
        predictive={"1904-new": (0.6624, 0.6331), "2024": (0.6960, 0.6669), "4242": (0.7327, 0.7148)},
        reconstruction={"42": (0.5638, 0.4177), "101": (0.5794, 0.4351), "1904": (0.5298, 0.3770)},
        pca=(0.5587, 0.4102), kmeans=(0.5662, 0.4297), raw=(0.7074, 0.6363), majority=(0.4739, 0.2143),
    ),
    "France, strict": dict(
        predictive={"1904": (0.7504, 0.7515), "2024": (0.7536, 0.7618), "4242": (0.6730, 0.6552),
                     "42": (0.7465, 0.7536), "101": (0.6638, 0.6481)},
        reconstruction={"42": (0.6128, 0.5423), "101": (0.6216, 0.5504), "1904": (0.6195, 0.5377)},
        pca=(0.6733, 0.5994), kmeans=(0.6900, 0.6196), raw=(0.7817, 0.7907), majority=(0.3012, 0.0926),
    ),
    "Australia, strict": dict(
        predictive={"1904": (0.6078, 0.5360), "2024": (0.6924, 0.6599), "4242": (0.6058, 0.5333),
                     "42": (0.7053, 0.6807), "101": (0.6426, 0.6221)},
        reconstruction={"42": (0.5590, 0.4239), "101": (0.5381, 0.3846), "1904": (0.5739, 0.4302)},
        pca=(0.5549, 0.4049), kmeans=(0.5662, 0.4299), raw=(0.7074, 0.6363), majority=(0.4739, 0.2143),
    ),
    "Estonia": dict(
        predictive={"1904-gc": (0.2020, 0.0812), "2024": (0.2053, 0.0772), "4242": (0.2086, 0.0785)},
        reconstruction={"42": (0.1453, 0.0343), "101": (0.1450, 0.0339), "1904": (0.1473, 0.0362)},
        # k-means: five-fit mean (results/runs/diagnostic72_k_readout.csv),
        # replacing an earlier single fit (0.1805 / 0.0585).
        pca=(0.1611, 0.0475), kmeans=(0.1687, 0.0513), raw=(0.2461, 0.0975), majority=(0.0764, 0.0017),
    ),
}

# Old-recipe CropGlobe reconstruction (cosine, batch 512 windows), transductive
# only (results/runs/diagnostic46_*.csv, tokens). Superseded by the matched recipe; supplement S2 only.
OLD_RECIPE_RECONSTRUCTION = {
    "France, transductive": {"42": (0.6279, 0.5540), "101": (0.6094, 0.5422), "1904-new": (0.6129, 0.5467)},
    "Australia, transductive": {"42": (0.5813, 0.4475), "101": (0.5835, 0.4375), "1904-new": (0.5687, 0.4229)},
}

# France runs the manuscript calls out as "low" (Section IV-D): three of eight
# masked-prediction runs across both settings score 0.664-0.673.
LOW_FRANCE_SEEDS = {("France, transductive", "1904-new"), ("France, strict", "4242"), ("France, strict", "101")}
