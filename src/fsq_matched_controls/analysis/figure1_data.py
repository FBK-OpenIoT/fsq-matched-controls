"""Figure 1 data: the per-run token scores (logistic probe, harmonized labels; ten runs per objective and panel, rounded to four
decimals) and the control, raw-band and majority-class scores of every panel. Constants only, verified against the per-run
result tables (results/runs/n_probe_*.csv for the runs; the controls as described in the supplement); no computation.

- k-means: the letter's k-means vocabulary (fitted on the four standardized principal-component scores, read as its centroid,
  mean of five fits).
- Reconstruction: the matched-recipe runs (plateau schedule, batch 448 windows). OLD_RECIPE_RECONSTRUCTION holds the three
  own-recipe runs (cosine schedule, batch 512 windows; supplement S2).
- The first runs of each panel (3 or 5) are those of figure1_data_pre_ten_runs.py; the others follow in seed order.
"""

# ---- per-seed dots (tokens, logistic probe, harmonized; ten runs per objective and panel) ----
PANELS = {
    "France, transductive": dict(
        predictive={"1904-new": (0.6665, 0.5622), "2024": (0.7557, 0.7608), "4242": (0.7322, 0.7290),
                     "11": (0.6581, 0.5594), "23": (0.6713, 0.6030), "37": (0.6766, 0.6720),
                     "53": (0.7580, 0.7618), "71": (0.6563, 0.6352), "89": (0.7495, 0.7560),
                     "97": (0.6590, 0.6272)},
        reconstruction={"42": (0.6284, 0.5732), "101": (0.6088, 0.5433), "1904": (0.6258, 0.5516),
                     "11": (0.6189, 0.5491), "23": (0.6222, 0.5644), "37": (0.6144, 0.5461),
                     "53": (0.6111, 0.5478), "71": (0.6442, 0.5865), "89": (0.6192, 0.5401),
                     "97": (0.6153, 0.5495)},
        pca=(0.6771, 0.6043), kmeans=(0.6960, 0.6274), raw=(0.7817, 0.7907), majority=(0.3012, 0.0926),
    ),
    "Australia, transductive": dict(
        predictive={"1904-new": (0.6624, 0.6331), "2024": (0.6960, 0.6669), "4242": (0.7327, 0.7148),
                     "11": (0.6557, 0.5633), "23": (0.6325, 0.5288), "37": (0.6691, 0.6435),
                     "53": (0.7095, 0.6850), "71": (0.6324, 0.5761), "89": (0.6844, 0.6530),
                     "97": (0.6949, 0.6494)},
        reconstruction={"42": (0.5638, 0.4177), "101": (0.5794, 0.4351), "1904": (0.5298, 0.3770),
                     "11": (0.5777, 0.4407), "23": (0.5635, 0.4182), "37": (0.5713, 0.4299),
                     "53": (0.5890, 0.4485), "71": (0.5942, 0.4513), "89": (0.5498, 0.3860),
                     "97": (0.5897, 0.4454)},
        pca=(0.5587, 0.4102), kmeans=(0.5662, 0.4297), raw=(0.7074, 0.6363), majority=(0.4739, 0.2143),
    ),
    "France, strict": dict(
        predictive={"1904": (0.7504, 0.7515), "2024": (0.7536, 0.7618), "4242": (0.6730, 0.6552),
                     "42": (0.7465, 0.7536), "101": (0.6638, 0.6481), "11": (0.7266, 0.7230),
                     "23": (0.7432, 0.7486), "37": (0.7571, 0.7598), "53": (0.7474, 0.7504),
                     "71": (0.6842, 0.6477)},
        reconstruction={"42": (0.6128, 0.5423), "101": (0.6216, 0.5504), "1904": (0.6195, 0.5377),
                     "11": (0.5987, 0.5239), "23": (0.6142, 0.5447), "37": (0.6073, 0.5483),
                     "53": (0.5994, 0.5449), "71": (0.6240, 0.5593), "89": (0.6078, 0.5430),
                     "97": (0.6265, 0.5586)},
        pca=(0.6733, 0.5994), kmeans=(0.6900, 0.6196), raw=(0.7817, 0.7907), majority=(0.3012, 0.0926),
    ),
    "Australia, strict": dict(
        predictive={"1904": (0.6078, 0.5360), "2024": (0.6924, 0.6599), "4242": (0.6058, 0.5333),
                     "42": (0.7053, 0.6807), "101": (0.6426, 0.6221), "11": (0.6636, 0.6382),
                     "23": (0.6205, 0.5601), "37": (0.6288, 0.5948), "53": (0.7128, 0.6822),
                     "71": (0.6177, 0.5468)},
        reconstruction={"42": (0.5590, 0.4239), "101": (0.5381, 0.3846), "1904": (0.5739, 0.4302),
                     "11": (0.5778, 0.4377), "23": (0.5872, 0.4466), "37": (0.5831, 0.4379),
                     "53": (0.5747, 0.4298), "71": (0.5659, 0.4197), "89": (0.5698, 0.4157),
                     "97": (0.5775, 0.4366)},
        pca=(0.5549, 0.4049), kmeans=(0.5662, 0.4299), raw=(0.7074, 0.6363), majority=(0.4739, 0.2143),
    ),
    "Estonia": dict(
        predictive={"1904-gc": (0.2020, 0.0812), "2024": (0.2053, 0.0772), "4242": (0.2086, 0.0785),
                     "11": (0.2063, 0.0875), "23": (0.1908, 0.0644), "37": (0.2059, 0.0832),
                     "53": (0.2014, 0.0779), "71": (0.2008, 0.0807), "89": (0.2110, 0.0819),
                     "97": (0.2031, 0.0783)},
        reconstruction={"42": (0.1453, 0.0343), "101": (0.1450, 0.0339), "1904": (0.1473, 0.0362),
                     "11": (0.1481, 0.0362), "23": (0.1432, 0.0333), "37": (0.1470, 0.0348),
                     "53": (0.1469, 0.0354), "71": (0.1480, 0.0369), "89": (0.1469, 0.0359),
                     "97": (0.1446, 0.0349)},
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

# France runs the manuscript calls out as "low" (Section IV-C): a France masked-prediction run is low if its token accuracy is not above the k-means control of its setting
# (0.696 transductive, 0.690 strict); 9 of 20 runs across both settings, scoring 0.656-0.684 (the other 11 score 0.727-0.758).
LOW_FRANCE_SEEDS = {("France, strict", "71"), ("France, strict", "101"), ("France, strict", "4242"), ("France, transductive", "11"), ("France, transductive", "23"), ("France, transductive", "37"), ("France, transductive", "71"), ("France, transductive", "97"), ("France, transductive", "1904-new")}
