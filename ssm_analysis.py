#!/usr/bin/env python3

import sys
import re
import os
import skrf as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from one_file_ssm_analysis import one_file_approximation


# directory, file_name, probe_port, limit
# one_file_approximation("data", "745.s2p", 2, 50)
def analyse(directory, probe_port=2, freqlimit=50):
    if directory[-1] != "/":  # TODO check it
        directory = directory + "/"
    # get filenames and currents
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
        # get filenames and currents
        walk = list(os.walk(directory))
        string_for_re = ".*\\.s2p"
        r = re.compile(string_for_re)
        files = walk[0][2]
        matched_files = list(filter(r.match, files))
        matched_files.sort()
    matched_files.sort()
    print(matched_files)

    df = pd.DataFrame(
        columns=[
            "Current, mA",
            "R_m, Om",
            "R_j, Om",
            "C_p, fF",
            "C_m, fF",
            "f_r, GHz",
            "f_p, GHz",
            "gamma",
            "c",
            "f_3dB, GHz",
        ]
    )

    for file in matched_files:
        R_m, R_j, C_p, C_m, f_r, f_p, gamma, c, f3dB = one_file_approximation(
            directory, file, probe_port, freqlimit
        )

        file_name_parser = file.split("-")
        r2 = re.compile(".*mA")
        current = list(filter(r2.match, file_name_parser))[0]
        if current.endswith(".s2p"):
            current = current.removesuffix(".s2p")
        current = float(current.removesuffix("mA"))

        print(f"current={current}")

        df.loc[len(df)] = [current, R_m, R_j, C_p, C_m, f_r, f_p, gamma, c, f3dB]

    df = df.sort_values("Current, mA")
    df.reset_index(drop=True, inplace=True)
    df.to_csv(directory + "/reports/" + directory.replace("/", "-") + "-report.csv")

    print(df)
    return df


#  _____ _
# |  ___(_) __ _ _   _ _ __ ___  ___
# | |_  | |/ _` | | | | '__/ _ \/ __|
# |  _| | | (_| | |_| | | |  __/\__ \
# |_|   |_|\__, |\__,_|_|  \___||___/
#          |___/
def makefigs(df, directory):
    fig = plt.figure(figsize=(20, 10))

    ax1_rm = fig.add_subplot(331)
    ax1_rm.plot(df["Current, mA"], df["R_m, Om"], marker="o")
    ax1_rm.set_ylabel("R_m, Om")
    ax1_rm.set_xlabel("Current, mA")
    ax1_rm.set_ylim([0, 100])
    ax1_rm.set_xlim(left=0)
    ax1_rm.grid(which="both")
    ax1_rm.minorticks_on()

    ax2_rj = fig.add_subplot(332)
    ax2_rj.plot(df["Current, mA"], df["R_j, Om"], marker="o")
    ax2_rj.set_ylabel("R_j, Om")
    ax2_rj.set_xlabel("Current, mA")
    ax2_rj.set_xlim(left=0)
    ax2_rj.set_ylim([0, 400])
    ax2_rj.grid(which="both")
    ax2_rj.minorticks_on()

    ax3_cp = fig.add_subplot(333)
    ax3_cp.plot(df["Current, mA"], df["C_p, fF"], marker="o")
    ax3_cp.set_ylabel("C_p, fF")
    ax3_cp.set_xlabel("Current, mA")
    ax3_cp.set_xlim(left=0)
    ax3_cp.set_ylim([60, 90])
    ax3_cp.grid(which="both")
    ax3_cp.minorticks_on()

    ax4_cm = fig.add_subplot(334)
    ax4_cm.plot(df["Current, mA"], df["C_m, fF"], marker="o")
    ax4_cm.set_ylabel("C_m, fF")
    ax4_cm.set_xlabel("Current, mA")
    ax4_cm.set_xlim(left=0)
    ax4_cm.set_ylim([0, 300])
    ax4_cm.grid(which="both")
    ax4_cm.minorticks_on()

    ax5_fr = fig.add_subplot(335)
    ax5_fr.plot(df["Current, mA"], df["f_r, GHz"], marker="o")
    ax5_fr.set_ylabel("f_r, GHz")
    ax5_fr.set_xlabel("Current, mA")
    ax5_fr.set_xlim(left=0)
    ax5_fr.set_ylim([0, 40])
    ax5_fr.grid(which="both")
    ax5_fr.minorticks_on()

    ax6_fp = fig.add_subplot(336)
    ax6_fp.plot(df["Current, mA"], df["f_p, GHz"], marker="o")
    ax6_fp.set_ylabel("f_p, GHz")
    ax6_fp.set_xlabel("Current, mA")
    ax6_fp.set_xlim(left=0)
    ax6_fp.set_ylim([0, 40])
    ax6_fp.grid(which="both")
    ax6_fp.minorticks_on()

    ax7_gamma = fig.add_subplot(337)
    ax7_gamma.plot(df["Current, mA"], df["gamma"], marker="o")
    ax7_gamma.set_ylabel("gamma")
    ax7_gamma.set_xlabel("Current, mA")
    ax7_gamma.set_xlim(left=0)
    ax7_gamma.set_ylim([0, 200])
    ax7_gamma.grid(which="both")
    ax7_gamma.minorticks_on()

    ax8_c = fig.add_subplot(338)
    ax8_c.plot(df["Current, mA"], df["c"], marker="o")
    ax8_c.set_ylabel("c")
    ax8_c.set_xlabel("Current, mA")
    ax8_c.set_xlim(left=0)
    ax8_c.set_ylim([-100, 0])
    ax8_c.grid(which="both")
    ax8_c.minorticks_on()

    ax9_f3db = fig.add_subplot(339)
    ax9_f3db.plot(df["Current, mA"], df["f_3dB, GHz"], marker="o")
    ax9_f3db.set_ylabel("f_3dB, GHz")
    ax9_f3db.set_xlabel("Current, mA")
    ax9_f3db.set_xlim(left=0)
    ax9_f3db.set_ylim([0, 40])
    ax9_f3db.grid(which="both")
    ax9_f3db.minorticks_on()

    if not os.path.exists(directory + "/reports/"):  # make directories
        os.makedirs(directory + "/reports/")
    plt.savefig(
        directory + "/reports/" + directory.replace("/", "-") + ".png"
    )  # save figure
    plt.savefig(
        directory + "/reports/" + directory.replace("/", "-") + ".png"
    )  # save figure

    if not os.path.exists("reports/"):  # make directories
        os.makedirs("reports/")
    plt.savefig("reports/" + directory.replace("/", "-") + ".png")  # save figure
    plt.savefig("reports/" + directory.replace("/", "-") + ".png")  # save figure
    plt.close()


for i, directory in enumerate(sys.argv[1:]):
    le = len(sys.argv[1:])
    print(f"[{i+1}/{le}] {directory}")
    df = analyse(directory)
    makefigs(df, directory)