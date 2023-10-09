#!/usr/bin/env python3

import sys
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

max_current = 22

fig = plt.figure(figsize=(20, 10))

ax1_rm = fig.add_subplot(331)
ax1_rm.set_ylabel("R_m, Om")
ax1_rm.set_xlabel("Current, mA")
ax1_rm.set_ylim([0, 200])
# ax1_rm.set_xlim(left=0)
ax1_rm.set_xlim([0, max_current])
ax1_rm.grid(which="both")
ax1_rm.minorticks_on()

ax2_rj = fig.add_subplot(332)
ax2_rj.set_ylabel("R_j, Om")
ax2_rj.set_xlabel("Current, mA")
# ax2_rj.set_xlim(left=0)
ax2_rj.set_xlim([0, max_current])
ax2_rj.set_ylim([0, 1000])
ax2_rj.grid(which="both")
ax2_rj.minorticks_on()

ax3_cp = fig.add_subplot(333)
ax3_cp.set_ylabel("C_p, fF")
ax3_cp.set_xlabel("Current, mA")
# ax3_cp.set_xlim(left=0)
ax3_cp.set_xlim([0, max_current])
ax3_cp.set_ylim([0, 100])
ax3_cp.grid(which="both")
ax3_cp.minorticks_on()

ax4_cm = fig.add_subplot(334)
ax4_cm.set_ylabel("C_m, fF")
ax4_cm.set_xlabel("Current, mA")
# ax4_cm.set_xlim(left=0)
ax4_cm.set_xlim([0, max_current])
ax4_cm.set_ylim([0, 200])
ax4_cm.grid(which="both")
ax4_cm.minorticks_on()

ax5_fr = fig.add_subplot(335)
ax5_fr.set_ylabel("f_r, GHz")
ax5_fr.set_xlabel("Current, mA")
# ax5_fr.set_xlim(left=0)
ax5_fr.set_xlim([0, max_current])
ax5_fr.set_ylim([0, 40])
ax5_fr.grid(which="both")
ax5_fr.minorticks_on()

ax6_fp = fig.add_subplot(336)
ax6_fp.set_ylabel("f_p, GHz")
ax6_fp.set_xlabel("Current, mA")
# ax6_fp.set_xlim(left=0)
ax6_fp.set_xlim([0, max_current])
ax6_fp.set_ylim([0, 40])
ax6_fp.grid(which="both")
ax6_fp.minorticks_on()

ax7_gamma = fig.add_subplot(337)
ax7_gamma.set_ylabel("gamma")
ax7_gamma.set_xlabel("Current, mA")
# ax7_gamma.set_xlim(left=0)
ax7_gamma.set_xlim([0, max_current])
ax7_gamma.set_ylim([0, 200])
ax7_gamma.grid(which="both")
ax7_gamma.minorticks_on()

ax8_c = fig.add_subplot(338)
ax8_c.set_ylabel("c")
ax8_c.set_xlabel("Current, mA")
# ax8_c.set_xlim(left=0)
ax8_c.set_xlim([0, max_current])
ax8_c.set_ylim([-100, 0])
ax8_c.grid(which="both")
ax8_c.minorticks_on()

ax9_f3db = fig.add_subplot(339)
ax9_f3db.set_ylabel("f_3dB, GHz")
ax9_f3db.set_xlabel("Current, mA")
# ax9_f3db.set_xlim(left=0)
ax9_f3db.set_xlim([0, max_current])
ax9_f3db.set_ylim([0, 40])
ax9_f3db.grid(which="both")
ax9_f3db.minorticks_on()

names = []
vcsel = []

for i, directory in enumerate(sys.argv[1:]):
    if directory[-1] != "/":  # TODO check it
        directory = directory + "/"
    walk = list(os.walk(directory))
    string_for_re = ".*\\.s2p"
    r = re.compile(string_for_re)
    files = walk[0][2]
    matched_files = list(filter(r.match, files))
    if not matched_files:
        print("No matching files, checking /PNA directory")
        if directory[-1] != "/":  # TODO check it
            directory = directory + "/PNA/"
        else:
            directory = directory + "PNA/"

    le = len(sys.argv[1:])
    print(f"[{i+1}/{le}] {directory}")

    name_from_dir = (
        directory.replace("/", "-")
        .removesuffix("-")
        .removesuffix("-PNA")
        .removeprefix("data-")
    )

    names.append(name_from_dir)
    vcsel.append(name_from_dir.split("-")[-1])

    df = pd.read_csv(
        directory + "reports/" + name_from_dir + "-report.csv", index_col=0
    )

    ax1_rm.plot(
        df["Current, mA"],
        df["R_m, Om"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax2_rj.plot(
        df["Current, mA"],
        df["R_j, Om"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax3_cp.plot(
        df["Current, mA"],
        df["C_p, fF"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax4_cm.plot(
        df["Current, mA"],
        df["C_m, fF"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax5_fr.plot(
        df["Current, mA"],
        df["f_r, GHz"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax6_fp.plot(
        df["Current, mA"],
        df["f_p, GHz"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax7_gamma.plot(
        df["Current, mA"],
        df["gamma"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax8_c.plot(
        df["Current, mA"],
        df["c"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )
    ax9_f3db.plot(
        df["Current, mA"],
        df["f_3dB, GHz"],
        marker="o",
        label=name_from_dir.split("-")[-1],
        alpha=0.5,
    )

fig.suptitle("-".join(name_from_dir.split("-")[:-1]) + " " + "-".join(vcsel))

legend_font_size = 5
ax1_rm.legend(loc=0, prop={"size": legend_font_size})
ax2_rj.legend(loc=0, prop={"size": legend_font_size})
ax3_cp.legend(loc=0, prop={"size": legend_font_size})
ax4_cm.legend(loc=0, prop={"size": legend_font_size})
ax5_fr.legend(loc=0, prop={"size": legend_font_size})
ax6_fp.legend(loc=0, prop={"size": legend_font_size})
ax7_gamma.legend(loc=0, prop={"size": legend_font_size})
ax8_c.legend(loc=0, prop={"size": legend_font_size})
ax9_f3db.legend(loc=0, prop={"size": legend_font_size})

if not os.path.exists("reports/"):  # make directories
    os.makedirs("reports/")
plt.savefig("reports/" + "-".join(vcsel) + ".png", dpi=600)  # save figure
