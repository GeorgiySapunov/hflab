#!/usr/bin/env python3
import skrf as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

WaferID = "MuHA2023Oct-940nm"

GOODpd = ["01EBav1goodpd", "25DB", "25DBgood", "25DBgood2"]

BADpd = [
    "01EBav1badpd",
    "01EBav1badpdv2",
    "25DBbad",
    "25DBbad2",
    "25DBbad3",
    "25DBbad4",
]

BADpdlist = []
GOODpdlist = []

for BADpd_dir in BADpd:
    BADpd = Path("data") / WaferID / BADpd_dir / "PNA"
    BADpdlist_onelaser = sorted(list(BADpd.glob("*.s2p")))
    BADpdlist.extend(BADpdlist_onelaser)


for GOODpd_dir in GOODpd:
    GOODpd = Path("data") / WaferID / GOODpd_dir / "PNA"
    GOODpdlist_onelaser = sorted(list(GOODpd.glob("*.s2p")))
    GOODpdlist.extend(GOODpdlist_onelaser)

S21_list = []
for s2pBADpd, s2pGOODpd in zip(BADpdlist, GOODpdlist):

    s2pBADpd = rf.Network(s2pBADpd)
    s2pGOODpd = rf.Network(s2pGOODpd)

    s2pBADpd = s2pBADpd.to_dataframe("s")
    S21_Real = s2pBADpd[f"s 21"].values.real
    S21_Imag = s2pBADpd[f"s 21"].values.imag
    S21_Magnitude_bad = 10 * np.log10(S21_Real**2 + S21_Imag**2)

    s2pGOODpd = s2pGOODpd.to_dataframe("s")
    S21_Real = s2pGOODpd[f"s 21"].values.real
    S21_Imag = s2pGOODpd[f"s 21"].values.imag
    S21_Magnitude_good = 10 * np.log10(S21_Real**2 + S21_Imag**2)

    f = s2pBADpd.index.values
    assert s2pBADpd.index.values.all() == s2pGOODpd.index.values.all()

    S21_Magnitude = S21_Magnitude_bad - S21_Magnitude_good
    S21_list.append(S21_Magnitude.reshape(-1, 1))

S21_Magnitude_mean = np.hstack(S21_list).mean(axis=1)
photodiode_s2p = "resources/T3K7V9_DXM30BF_U00162.s2p"
photodiode_s2p = rf.Network(photodiode_s2p)

dict = {
    "s21logmag_mean_difference": S21_Magnitude_mean,
    "s21logmag_bad": S21_Magnitude_bad,
    "s21logm_good": S21_Magnitude_good,
}

vcsel_df = pd.DataFrame(dict, index=f)

pd_df = photodiode_s2p.to_dataframe("s")
PD_S21_Real = pd_df["s 21"].values.real
PD_S21_Imag = pd_df["s 21"].values.imag
pd_df["GOOD_pd_s21logmag"] = 10 * np.log10(PD_S21_Real**2 + PD_S21_Imag**2)

vcsel_df = vcsel_df.join(pd_df[["GOOD_pd_s21logmag"]], how="outer")
# vcsel_df["pd_s21_logm"] = vcsel_df["pd_s21logm"].interpolate()
vcsel_df = vcsel_df.interpolate()
pd_Magnitude = vcsel_df["GOOD_pd_s21logmag"].values
S21_Magnitude_mean = vcsel_df["s21logmag_mean_difference"].values
vcsel_df["BAD_pd_S21logmag"] = pd_Magnitude + S21_Magnitude_mean
print(vcsel_df)
# vcsel_df = vcsel_df[
#     ["GOOD_pd_s21logmag", "BAD_pd_S21logmag", "s21logmag_mean_difference"]
# ]

vcsel_df.plot()
plt.grid()
plt.show()
