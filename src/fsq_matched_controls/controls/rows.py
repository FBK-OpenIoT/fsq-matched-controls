"""The letter's five rows (region, setting): the fit pool of both controls and the standardization of rounded PCA.

Fit pool: exactly the data the compared encoder saw during pretraining (in this concatenation order).
    transductive (CropGlobe): the probe pool, France and Australia;
    strict France: the probe pool and Australia; strict Australia: the probe pool and France;
    Estonia (EuroCropsML): the Latvia + Portugal pretraining parcels, valid months only.
The probe trains on the probe pool and is evaluated on the row's region.

STANDARDIZE: the samples rounded PCA's standardization is fitted on (valid months).
"""
from __future__ import annotations

ROWS = {
    "France, transductive":    dict(dataset="cropglobe", pool=["train", "FRA", "AUS"], region="FRA"),
    "Australia, transductive": dict(dataset="cropglobe", pool=["train", "FRA", "AUS"], region="AUS"),
    "France, strict":          dict(dataset="cropglobe", pool=["train", "AUS"], region="FRA"),
    "Australia, strict":       dict(dataset="cropglobe", pool=["train", "FRA"], region="AUS"),
    "Estonia":                 dict(dataset="eurocropsml", pool=["train"], region="estonia"),
}

# Rounded PCA standardization, as the letter's numbers were produced:
#   transductive rows: the probe pool plus both evaluation regions together (a transductive run is evaluated on both);
#   strict rows: the probe pool plus the evaluated region;
#   Estonia: the probe pool only (valid months), i.e. the encoder's fit pool.
# Estonia's construction is fixed by a value that involves no probe: the stored entropy of the rounded codes on the probe pool
# (7.456372841944349 bits per month over valid months) is reproduced by this standardization (7.456373392963799; the difference is a
# few rounding ties), not by a standardization on the probe pool plus Estonia (7.4669).
# The probe's lbfgs fit stops at its iteration limit before converging on these features, so its scores vary across environments
# by up to about 5e-4; check_controls applies the platform tolerance.
STANDARDIZE = {
    "France, transductive":    ["train", "FRA", "AUS"],
    "Australia, transductive": ["train", "FRA", "AUS"],
    "France, strict":          ["train", "FRA"],
    "Australia, strict":       ["train", "AUS"],
    "Estonia":                 ["train"],
}
