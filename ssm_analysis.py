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


S21_MSE_threshold = 5
probe_port = 2
freqlimit = 40
fp_fixed = True


# directory, file_name, probe_port, limit
# one_file_approximation("data", "745.s2p", 2, 50)
def analyse(
    directory,
    s2p=True,
    probe_port=probe_port,
    freqlimit=freqlimit,
    S21_MSE_threshold=S21_MSE_threshold,
    fp_fixed=True,
):
    if directory[-1] != "/":  # TODO check it
        directory = directory + "/"
    start_directory = directory
    report_dir = start_directory + "PNA_reports/"
    print(report_dir)
    # get filenames and currents
    walk = list(os.walk(directory))
    if not s2p:
        # first check if you have a csv file from automatic system
        string_for_re = ".*\\.csv$"
        r = re.compile(string_for_re)
        files = walk[0][2]
        matched_csv_files = list(filter(r.match, files))
        if matched_csv_files:
            matched_csv_files.sort()
            matched_csv_file = [matched_csv_files[0]][0]
            print(f"Matched .csv file: {matched_csv_file}")
        else:
            return None, directory, report_dir
    elif s2p:
        # check for .s2p files
        string_for_re = ".*\\.s2p"
        r = re.compile(string_for_re)
        files = walk[0][2]
        matched_files = list(filter(r.match, files))
        if not matched_files:
            print("No matching files, checking /PNA directory")
            directory = directory + "PNA/"
            # get filenames and currents
            walk = list(os.walk(directory))
            if not walk:
                print(f"Can't find PNA data in {directory}")
                return None, directory, report_dir
            string_for_re = ".*\\.s2p"
            r = re.compile(string_for_re)
            files = walk[0][2]
            matched_files = list(filter(r.match, files))
            matched_files.sort()
        matched_files.sort()
        print(f"Matched .s2p files: {matched_files}")

    name_from_dir = (
        directory.replace("/", "-")
        .removesuffix("-")
        .removesuffix("-PNA")
        .removeprefix("data-")
    )

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
            "f_p(fixed), GHz",
            "f_r(f_p fixed), GHz",
            "gamma(f_p fixed)",
            "c(f_p fixed)",
            "f3dB(f_p fixed), GHz",
        ]
    )

    if s2p:
        report_dir = start_directory + "PNA_reports(s2p)/"
        for file in matched_files:
            (
                R_m,
                R_j,
                C_p,
                C_m,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_par_GHz,
                f_r2,
                gamma2,
                c2,
                f3dB2,
            ) = one_file_approximation(
                directory=directory,
                report_directory=report_dir,
                freqlimit=freqlimit,
                file_name=file,
                probe_port=probe_port,
                S21_MSE_threshold=S21_MSE_threshold,
                fp_fixed=fp_fixed,
            )

            file_name_parser = file.split("-")
            r2 = re.compile(".*mA")
            current = list(filter(r2.match, file_name_parser))[0]
            current = float(current.removesuffix(".s2p").removesuffix("mA"))
            print(f"current={current}")

            df.loc[len(df)] = [
                current,
                R_m,
                R_j,
                C_p,
                C_m,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_par_GHz,
                f_r2,
                gamma2,
                c2,
                f3dB2,
            ]

        df = df.sort_values("Current, mA")
        df.reset_index(drop=True, inplace=True)
        if not os.path.exists(report_dir):  # make directories
            os.makedirs(report_dir)
        df.to_csv(report_dir + name_from_dir + "-report(s2p).csv")
    elif not s2p:  # automatic system csv file parsing and processing
        report_dir = start_directory + "PNA_reports(auto)/"
        auto_file = pd.read_csv(
            directory + matched_csv_file, header=[0, 1, 2], sep="\t"
        )
        # print(auto_file.head())
        currents = (
            auto_file["VNA Current"][auto_file["VNA Current"] > 0]
            .dropna()
            .values.reshape(-1)
        )
        frequency = auto_file["Frequency1"].values.reshape(-1)
        abs_s21 = auto_file["Abs(S21)"].values.reshape(-1)
        phase_s21 = auto_file["Phase(S21)"].values.reshape(-1)
        abs_s11 = auto_file["Abs(S11)"].values.reshape(-1)
        phase_s11 = auto_file["Phase(S11)"].values.reshape(-1)
        points = np.where(abs_s21 == -999999999)[0][0]
        waferid_wl, coordinates, _ = matched_csv_file.split("_")
        waferid, wavelength = waferid_wl.split("-")
        coordinates = coordinates[:2] + coordinates[3:]
        # temperature = 25
        for i, current in enumerate(currents):
            print(f"current={current}")
            start = i * points + i
            stop = (i + 1) * points + i
            (
                R_m,
                R_j,
                C_p,
                C_m,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_par_GHz,
                f_r2,
                gamma2,
                c2,
                f3dB2,
            ) = one_file_approximation(
                directory=directory,
                report_directory=report_dir,
                freqlimit=freqlimit,
                file_name=None,
                probe_port=None,
                waferid=waferid,
                wavelength=wavelength,
                coordinates=coordinates,
                current=current,
                #     temperature=None,
                frequency=frequency[0:points],
                s11mag=abs_s11[start:stop],
                s11deg_rad=phase_s11[start:stop],
                s21mag=abs_s21[start:stop],
                s21deg=phase_s11[start:stop],
                S21_MSE_threshold=S21_MSE_threshold,
                fp_fixed=fp_fixed,
            )

            df.loc[len(df)] = [
                current,
                R_m,
                R_j,
                C_p,
                C_m,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_par_GHz,
                f_r2,
                gamma2,
                c2,
                f3dB2,
            ]

        df = df.sort_values("Current, mA")
        df.reset_index(drop=True, inplace=True)
        if not os.path.exists(report_dir):  # make directories
            os.makedirs(report_dir)
        df.to_csv(report_dir + name_from_dir + "-report(auto).csv")

    print(df)
    return df, directory, report_dir


#  _____ _
# |  ___(_) __ _ _   _ _ __ ___  ___
# | |_  | |/ _` | | | | '__/ _ \/ __|
# |  _| | | (_| | |_| | | |  __/\__ \
# |_|   |_|\__, |\__,_|_|  \___||___/
#          |___/
def makefigs(df, directory, s2p=True):
    if df is None:
        return
    name_from_dir = (
        directory.replace("/", "-")
        .removesuffix("-")
        .removesuffix("-PNA")
        .removeprefix("data-")
    )

    fig = plt.figure(figsize=(1.8 * 11.69, 1.8 * 8.27))
    fig.suptitle(name_from_dir)

    max_current = 20

    ax1_rm = fig.add_subplot(241)
    ax1_rm.plot(df["Current, mA"], df["R_m, Om"])
    ax1_rm.set_ylabel("R_m, Om")
    ax1_rm.set_xlabel("Current, mA")
    ax1_rm.set_ylim([0, 200])
    # ax1_rm.set_xlim(left=0)
    ax1_rm.set_xlim([0, max_current])
    ax1_rm.grid(which="both")
    ax1_rm.minorticks_on()

    ax2_rj = fig.add_subplot(242)
    ax2_rj.plot(df["Current, mA"], df["R_j, Om"])
    ax2_rj.set_ylabel("R_j, Om")
    ax2_rj.set_xlabel("Current, mA")
    # ax2_rj.set_xlim(left=0)
    ax2_rj.set_xlim([0, max_current])
    ax2_rj.set_ylim([0, 1000])
    ax2_rj.grid(which="both")
    ax2_rj.minorticks_on()

    ax3_cp = fig.add_subplot(243)
    ax3_cp.plot(df["Current, mA"], df["C_p, fF"])
    ax3_cp.set_ylabel("C_p, fF")
    ax3_cp.set_xlabel("Current, mA")
    # ax3_cp.set_xlim(left=0)
    ax3_cp.set_xlim([0, max_current])
    ax3_cp.set_ylim([0, 100])
    ax3_cp.grid(which="both")
    ax3_cp.minorticks_on()

    ax4_cm = fig.add_subplot(244)
    ax4_cm.plot(df["Current, mA"], df["C_m, fF"])
    ax4_cm.set_ylabel("C_m, fF")
    ax4_cm.set_xlabel("Current, mA")
    # ax4_cm.set_xlim(left=0)
    ax4_cm.set_xlim([0, max_current])
    ax4_cm.set_ylim([0, 1000])
    ax4_cm.grid(which="both")
    ax4_cm.minorticks_on()

    ax5_gamma = fig.add_subplot(245)
    ax5_gamma.plot(df["Current, mA"], df["gamma"], label="ɣ")
    if fp_fixed:
        ax5_gamma.plot(
            df["Current, mA"], df["gamma(f_p fixed)"], label="ɣ(f_p fixed)", alpha=0.5
        )
    ax5_gamma.set_ylabel("ɣ")
    ax5_gamma.set_xlabel("Current, mA")
    # ax5_gamma.set_xlim(left=0)
    ax5_gamma.set_xlim([0, max_current])
    ax5_gamma.set_ylim([0, 300])
    ax5_gamma.grid(which="both")
    ax5_gamma.minorticks_on()
    if fp_fixed:
        ax5_gamma.legend()

    ax6_fp = fig.add_subplot(246)
    ax6_fp.plot(df["Current, mA"], df["f_p, GHz"], label="from S21")
    if fp_fixed:
        ax6_fp.plot(
            df["Current, mA"],
            df["f_p(fixed), GHz"],
            label="from equivalent circuit",
            alpha=0.5,
        )
    ax6_fp.set_ylabel("f_p, GHz")
    ax6_fp.set_xlabel("Current, mA")
    # ax6_fp.set_xlim(left=0)
    ax6_fp.set_xlim([0, max_current])
    ax6_fp.set_ylim([0, 65])
    ax6_fp.grid(which="both")
    ax6_fp.minorticks_on()
    if fp_fixed:
        ax6_fp.legend()

    ax7_fr = fig.add_subplot(247)
    ax7_fr.plot(df["Current, mA"], df["f_r, GHz"], label="f_r")
    if fp_fixed:
        ax7_fr.plot(
            df["Current, mA"],
            df["f_r(f_p fixed), GHz"],
            label="f_r(f_p fixed)",
            alpha=0.5,
        )
    ax7_fr.set_ylabel("f_r, GHz")
    ax7_fr.set_xlabel("Current, mA")
    # ax7_fr.set_xlim(left=0)
    ax7_fr.set_xlim([0, max_current])
    ax7_fr.set_ylim([0, 65])
    ax7_fr.grid(which="both")
    ax7_fr.minorticks_on()
    if fp_fixed:
        ax7_fr.legend()

    # ax8_c = fig.add_subplot(338)
    # ax8_c.plot(df["Current, mA"], df["c"], marker="o")
    # ax8_c.set_ylabel("c")
    # ax8_c.set_xlabel("Current, mA")
    # # ax8_c.set_xlim(left=0)
    # ax8_c.set_xlim([0, max_current])
    # ax8_c.set_ylim([-100, 0])
    # ax8_c.grid(which="both")
    # ax8_c.minorticks_on()

    ax9_f3db = fig.add_subplot(248)
    ax9_f3db.plot(df["Current, mA"], df["f_3dB, GHz"], label="f3dB")
    if fp_fixed:
        ax9_f3db.plot(
            df["Current, mA"], df["f3dB(f_p fixed), GHz"], label="f3dB(f_p fixed)"
        )
    ax9_f3db.set_ylabel("f_3dB, GHz")
    ax9_f3db.set_xlabel("Current, mA")
    # ax9_f3db.set_xlim(left=0)
    ax9_f3db.set_xlim([0, max_current])
    ax9_f3db.set_ylim([0, 65])
    ax9_f3db.grid(which="both")
    ax9_f3db.minorticks_on()

    def annotate_max_f3db(x, y, ax=None):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = f"{xmax:.2f} mA, {ymax:.2f} GHz"
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90")
        kw = dict(
            xycoords="data",
            textcoords="axes fraction",
            arrowprops=arrowprops,
            bbox=bbox_props,
            ha="right",
            va="top",
        )
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.99, 0.99), **kw)
        return xmax

    annotate_max_f3db(df["Current, mA"], df["f_3dB, GHz"], ax=ax9_f3db)
    if fp_fixed:
        ax9_f3db.legend(loc=2)

    if not os.path.exists(directory):  # make directories
        os.makedirs(directory)
    plt.savefig(directory + name_from_dir + ".png")  # save figure
    # plt.savefig(
    #     directory + "/reports/" + directory.replace("/", "-") + ".png"
    # )  # save figure

    if not os.path.exists("reports/"):  # make directories
        os.makedirs("reports/")
    plt.savefig("reports/" + name_from_dir + ".png")  # save figure
    # plt.savefig("reports/" + directory.replace("/", "-") + ".png")  # save figure
    plt.close()


for i, directory in enumerate(sys.argv[1:]):
    le = len(sys.argv[1:])
    print(f"[{i+1}/{le}] {directory}")
    df, dir, report_dir = analyse(
        directory,
        s2p=True,
        probe_port=probe_port,
        freqlimit=freqlimit,
        S21_MSE_threshold=S21_MSE_threshold,
        fp_fixed=fp_fixed,
    )
    makefigs(df, report_dir, s2p=True)
    df, dir, report_dir = analyse(
        directory,
        s2p=False,
        probe_port=probe_port,
        freqlimit=freqlimit,
        S21_MSE_threshold=S21_MSE_threshold,
        fp_fixed=fp_fixed,
    )
    makefigs(df, report_dir, s2p=False)
