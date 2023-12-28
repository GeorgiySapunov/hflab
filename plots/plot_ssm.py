#!/usr/bin/env python3

import sys
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 1st row
fig_ec_ind_max = 100
fig_ec_res_max = 200
fig_ec_cap_max = 300
#
fig_max_current = 20  # mA
fig_max_freq = 20  # GHz
# 2nd row
fig_max_gamma = 100
fig_max_f_r2_for_gamma = 180
# 4th row
fig_K_max = 1.5
fig_gamma0_max = 20
fig_D_MCEF_max = 10

fig = plt.figure(figsize=(3 * 11.69, 3 * 8.27))
legend_font_size = 7

names = []
vcsel = []

ax1_l = fig.add_subplot(461)
ax2_r = fig.add_subplot(462)
ax22_r = fig.add_subplot(463)
ax23_r = fig.add_subplot(464)
ax3_c = fig.add_subplot(465)
ax32_c = fig.add_subplot(466)
ax7_gamma = fig.add_subplot(445)
ax8_fp = fig.add_subplot(446)
ax9_fr = fig.add_subplot(447)
ax10_f3db = fig.add_subplot(4, 4, 8)
ax11_sqrt_gamma = fig.add_subplot(4, 4, 9)
ax13_fr_for_D = fig.add_subplot(4, 4, 11)
ax14_f3dB_for_MCEF = fig.add_subplot(4, 4, 12)
ax15_K = fig.add_subplot(4, 4, 13)
ax16_gamma0 = fig.add_subplot(4, 4, 14)
ax17_D = fig.add_subplot(4, 4, 15)
ax18_MCEF = fig.add_subplot(4, 4, 16)

ax_list = [
    ax1_l,
    ax2_r,
    ax22_r,
    ax23_r,
    ax3_c,
    ax32_c,
    ax7_gamma,
    ax8_fp,
    ax9_fr,
    ax10_f3db,
    ax11_sqrt_gamma,
    ax13_fr_for_D,
    ax14_f3dB_for_MCEF,
    ax15_K,
    ax16_gamma0,
    ax17_D,
    ax18_MCEF,
]

for i, directory in enumerate(sys.argv[1:]):
    if directory[-1] != "/":
        directory = directory + "/"
    have_report_dir = os.path.exists(directory + "PNA_reports(s2p)")
    if have_report_dir:
        name_from_dir = (
            directory.replace("/", "-")
            .removesuffix("-")
            .removesuffix("-PNA")
            .removeprefix("data-")
        )
        print(directory)
        directory = directory + "PNA_reports(s2p)/"
        le = len(sys.argv[1:])
        print(f"[{i+1}/{le}] {directory}")

        names.append(name_from_dir)
        vcsel.append(name_from_dir.split("-")[-1])

        df = pd.read_csv(
            directory + name_from_dir + "-report(s2p).csv",
        )
        K_D_MCEF_df = pd.read_csv(
            directory + name_from_dir + "-K_D_MCEF.csv",
        )

        fig.suptitle(name_from_dir)

        # 1-st row
        ax1_l.set_title("Inductance of the equivalent circuit")
        ax1_l.plot(df["Current, mA"], df["L, pH"], label=name_from_dir, marker="o")
        ax1_l.set_ylabel("L, pH")
        ax1_l.set_xlabel("Current, mA")
        ax1_l.set_ylim([0, fig_ec_ind_max])
        ax1_l.set_xlim([0, fig_max_current])

        ax2_r.set_title("Resistance R_p")
        ax2_r.plot(df["Current, mA"], df["R_p, Om"], label=name_from_dir, marker="o")
        ax2_r.set_ylabel("Resistance, Om")
        ax2_r.set_xlabel("Current, mA")
        ax2_r.set_ylim([0, fig_ec_res_max])
        ax2_r.set_xlim([0, fig_max_current])

        ax22_r.set_title("Resistance R_m")
        ax22_r.plot(df["Current, mA"], df["R_m, Om"], label=name_from_dir, marker="o")
        ax22_r.set_ylabel("Resistance, Om")
        ax22_r.set_xlabel("Current, mA")
        ax22_r.set_ylim([0, fig_ec_res_max])
        ax22_r.set_xlim([0, fig_max_current])

        ax23_r.set_title("Resistance R_a")
        ax23_r.plot(df["Current, mA"], df["R_a, Om"], label=name_from_dir, marker="o")
        ax23_r.set_ylabel("Resistance, Om")
        ax23_r.set_xlabel("Current, mA")
        ax23_r.set_ylim([0, fig_ec_res_max])
        ax23_r.set_xlim([0, fig_max_current])

        ax3_c.set_title("Capacitance C_p")
        ax3_c.plot(df["Current, mA"], df["C_p, fF"], label=name_from_dir, marker="o")
        ax3_c.set_ylabel("Capacitance, fF")
        ax3_c.set_xlabel("Current, mA")
        ax3_c.set_xlim([0, fig_max_current])
        ax3_c.set_ylim([0, fig_ec_cap_max])

        ax32_c.set_title("Capacitance C_a")
        ax32_c.plot(df["Current, mA"], df["C_a, fF"], label=name_from_dir, marker="o")
        ax32_c.set_ylabel("Capacitance, fF")
        ax32_c.set_xlabel("Current, mA")
        ax32_c.set_xlim([0, fig_max_current])
        ax32_c.set_ylim([0, fig_ec_cap_max])

        # 2-nd row
        ax7_gamma.set_title("ɣ from S21 approximation")
        ax7_gamma.plot(
            df["Current, mA"],
            df["gamma"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax7_gamma.set_ylabel("ɣ")
        ax7_gamma.set_xlabel("Current, mA")
        ax7_gamma.set_xlim([0, fig_max_current])
        ax7_gamma.set_ylim([0, fig_max_gamma])

        ax8_fp.set_title("Parasitic cut-off frequiencies from S21 approximation")
        ax8_fp.plot(
            df["Current, mA"],
            df["f_p, GHz"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax8_fp.set_ylabel("f_p, GHz")
        ax8_fp.set_xlabel("Current, mA")
        ax8_fp.set_xlim([0, fig_max_current])
        ax8_fp.set_ylim([0, fig_max_freq])

        ax9_fr.set_title("Resonance frequencies from S21 approximation")
        ax9_fr.plot(
            df["Current, mA"],
            df["f_r, GHz"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax9_fr.set_ylabel("f_r, GHz")
        ax9_fr.set_xlabel("Current, mA")
        ax9_fr.set_xlim([0, fig_max_current])
        ax9_fr.set_ylim([0, fig_max_freq])

        ax10_f3db.set_title("f3dB frequencies from S21 approximation")
        ax10_f3db.plot(
            df["Current, mA"],
            df["f_3dB, GHz"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax10_f3db.set_ylabel("f_3dB, GHz")
        ax10_f3db.set_xlabel("Current, mA")
        ax10_f3db.set_xlim([0, fig_max_current])
        ax10_f3db.set_ylim([0, fig_max_freq])

        # 3-rd row
        ax11_sqrt_gamma.set_title("ɣ vs f_r^2")
        ax11_sqrt_gamma.plot(
            df["f_r, GHz"] ** 2,
            df["gamma"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax11_sqrt_gamma.set_ylabel("ɣ")
        ax11_sqrt_gamma.set_xlabel("f_r^2, GHz^2")
        ax11_sqrt_gamma.set_xlim(left=0, right=fig_max_f_r2_for_gamma)
        ax11_sqrt_gamma.set_ylim(bottom=0, top=fig_max_gamma)

        ax13_fr_for_D.set_title("f_r vs sqrt(I-Ith) for D factor derivation")
        ax13_fr_for_D.plot(
            df["sqrt(I-I_th), sqrt(mA)"],
            df["f_r, GHz"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax13_fr_for_D.set_ylabel("f_r, GHz")
        ax13_fr_for_D.set_xlabel("sqrt(I-I_th), sqrt(mA)")
        ax13_fr_for_D.set_xlim(left=0, right=np.sqrt(fig_max_current))
        ax13_fr_for_D.set_ylim(bottom=0, top=fig_max_freq)

        ax14_f3dB_for_MCEF.set_title("f_3dB vs sqrt(I-Ith) for MCEF derivation")
        ax14_f3dB_for_MCEF.plot(
            df["sqrt(I-I_th), sqrt(mA)"],
            df["f_3dB, GHz"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax14_f3dB_for_MCEF.set_ylabel("f_3dB, GHz")
        ax14_f3dB_for_MCEF.set_xlabel("sqrt(I-I_th), sqrt(mA)")
        ax14_f3dB_for_MCEF.set_xlim(left=0, right=np.sqrt(fig_max_current))
        ax14_f3dB_for_MCEF.set_ylim(bottom=0, top=fig_max_freq)

        # 4-th row
        ax15_K.set_title("K factor for different appoximation limits")
        ax15_K.plot(
            K_D_MCEF_df["f_r_2_max"],
            K_D_MCEF_df["K factor, ns"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax15_K.set_ylabel("K factor, ns")
        ax15_K.set_xlabel("max f_r^2, GHz^2")
        ax15_K.set_xlim(left=0, right=fig_max_f_r2_for_gamma)
        ax15_K.set_ylim(bottom=0, top=fig_K_max)

        ax16_gamma0.set_title("ɣ0 for different appoximation limits")
        ax16_gamma0.plot(
            K_D_MCEF_df["f_r_2_max"],
            K_D_MCEF_df["gamma0"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax16_gamma0.set_ylabel("ɣ_0")
        ax16_gamma0.set_xlabel("max f_r^2, GHz^2")
        ax16_gamma0.set_xlim(left=0, right=fig_max_f_r2_for_gamma)
        ax16_gamma0.set_ylim(bottom=0, top=fig_gamma0_max)

        ax17_D.set_title("D factor for different appoximation limits")
        ax17_D.plot(
            K_D_MCEF_df["max sqrt(I-I_th), sqrt(mA)"],
            K_D_MCEF_df["D factor"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax17_D.set_ylabel("D factor, GHz/sqrt(mA)")
        ax17_D.set_xlabel("max sqrt(I-I_th), sqrt(mA)")
        ax17_D.set_xlim(left=0, right=np.sqrt(fig_max_current))
        ax17_D.set_ylim(bottom=0, top=fig_D_MCEF_max)

        ax18_MCEF.set_title("MCEF for different appoximation limits")
        ax18_MCEF.plot(
            K_D_MCEF_df["max sqrt(I-I_th), sqrt(mA)"],
            K_D_MCEF_df["MCEF"],
            label=name_from_dir,
            marker="o",
            alpha=0.5,
        )
        ax18_MCEF.set_ylabel("MCEF, GHz/sqrt(mA)")
        ax18_MCEF.set_xlabel("max sqrt(I-I_th), sqrt(mA)")
        ax18_MCEF.set_xlim(left=0, right=np.sqrt(fig_max_current))
        ax18_MCEF.set_ylim(bottom=0, top=fig_D_MCEF_max)

fig.suptitle("-".join(name_from_dir.split("-")[:-1]) + " " + "-".join(vcsel))

for i in ax_list:
    i.legend(loc=0, prop={"size": legend_font_size})
    i.grid(which="both")
    i.minorticks_on()

# ax1_l.legend(loc=0, prop={"size": legend_font_size})
# ax2_r.legend(loc=0, prop={"size": legend_font_size})
# ax22_r.legend(loc=0, prop={"size": legend_font_size})
# ax23_r.legend(loc=0, prop={"size": legend_font_size})
# ax3_c.legend(loc=0, prop={"size": legend_font_size})
# ax32_c.legend(loc=0, prop={"size": legend_font_size})
# ax7_gamma.legend(loc=0, prop={"size": legend_font_size})
# ax8_fp.legend(loc=0, prop={"size": legend_font_size})
# ax9_fr.legend(loc=0, prop={"size": legend_font_size})
# ax10_f3db.legend(loc=0, prop={"size": legend_font_size})
# ax11_sqrt_gamma.legend(loc=0, prop={"size": legend_font_size})
# ax13_fr_for_D.legend(loc=0, prop={"size": legend_font_size})
# ax14_f3dB_for_MCEF.legend(loc=0, prop={"size": legend_font_size})
# ax15_K.legend(loc=0, prop={"size": legend_font_size})
# ax16_gamma0.legend(loc=0, prop={"size": legend_font_size})
# ax17_D.legend(loc=0, prop={"size": legend_font_size})
# ax18_MCEF.legend(loc=0, prop={"size": legend_font_size})

if not os.path.exists("reports/"):  # make directories
    os.makedirs("reports/")
plt.savefig("reports/" + "-".join(vcsel) + ".png", dpi=300)  # save figure
