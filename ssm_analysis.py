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
from sklearn import linear_model

from one_file_ssm_analysis import one_file_approximation
from settings import settings

# Settings
S21_MSE_threshold = 5
probe_port = 1
fit_freqlimit = 20  # GHz
#
fp_fixed = True
fig_max_current = 20  # mA
fig_max_freq = None  # GHz
fig_max_gamma = None
fig_max_f_r2_for_gamma = None
i_th_fixed = None  # mA
D_MCEF_max = None
K_max = None
gamma0_max = None


# directory, file_name, probe_port, limit
# one_file_approximation("data", "745.s2p", 2, 50)
def analyse(
    directory,
    s2p=True,
    probe_port=probe_port,
    freqlimit=fit_freqlimit,
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
            "L, pH",
            "R_p, Om",
            "R_m, Om",
            "R_a, Om",
            "C_p, fF",
            "C_a, fF",
            "f_r, GHz",
            "f_p, GHz",
            "gamma",
            "c",
            "f_3dB, GHz",
            "f_p(fixed), GHz",
            "f_r(f_p fixed), GHz",
            "gamma(f_p fixed)",
            "c(f_p fixed)",
            "f_3dB(f_p fixed), GHz",
            "Temperature, °C",
            "Threshold current, mA",
        ]
    )

    if s2p:
        report_dir = start_directory + "PNA_reports(s2p)/"
        for file in matched_files:
            (
                L,
                R_p,
                R_m,
                R_a,
                C_p,
                C_a,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_p2,
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
            r2 = re.compile(".*°C")
            filt = list(filter(r2.match, file_name_parser))
            if filt:
                temperature = filt[0]
                temperature = float(temperature.removesuffix(".s2p").removesuffix("°C"))
            else:
                temperature = 25.0

            df.loc[len(df)] = [
                current,
                L,
                R_p,
                R_m,
                R_a,
                C_p,
                C_a,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_p2,
                f_r2,
                gamma2,
                c2,
                f3dB2,
                temperature,
                None,
            ]

        df = df.sort_values("Current, mA")
        df.reset_index(drop=True, inplace=True)

        if i_th_fixed:
            df[
                "Threshold current, mA"
            ].loc[  # saturable absorber manual threshold current
                df["Temperature, °C"] == temperature
            ] = float(
                t_th_fixed
            )
        else:
            temperature_list = df["Temperature, °C"].unique()
            liv_dir = directory.removesuffix("/PNA/") + "/LIV/"
            print(f"liv dir {liv_dir}")
            have_liv_dir = os.path.exists(liv_dir)
            if have_liv_dir:
                walk = list(os.walk(liv_dir))
                string_for_re = ".*\\.csv$"
                r = re.compile(string_for_re)
                files = walk[0][2]
                matched_files = list(filter(r.match, files))
                matched_files.sort()
                print(f"Matched LIV files: {matched_files}")
                for temperature in temperature_list:
                    string_for_re = f".*-{temperature}°C"
                    r = re.compile(string_for_re)
                    files = walk[0][2]
                    matched_files = list(filter(r.match, matched_files))
                    matched_files.sort()
                    if matched_files:
                        file = matched_files[0]
                        liv = pd.read_csv(liv_dir + file)
                        i = liv["Current, mA"]
                        l = liv["Output power, mW"]
                        first_der = np.gradient(l, i)
                        second_der = np.gradient(first_der, i)
                        if second_der.max() >= 5:
                            i_threshold = i[np.argmax(second_der >= 5)]  # mA
                            # l_threshold = l[np.argmax(second_der >= 5)]
                            df["Threshold current, mA"].loc[
                                df["Temperature, °C"] == temperature
                            ] = float(i_threshold)
                        else:
                            i_threshold = None
                    else:
                        i_threshold = None
                    print(f"I_threshold={i_threshold}")

        df["sqrt(I-I_th), sqrt(mA)"] = np.sqrt(
            df["Current, mA"] - df["Threshold current, mA"]
        )

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
        temperature = 25.0
        auto_I_th = float(auto_file["Threshold current"].iloc[0])
        for i, current in enumerate(currents):
            print(f"current={current}")
            start = i * points + i
            stop = (i + 1) * points + i
            (
                L,
                R_p,
                R_m,
                R_a,
                C_p,
                C_a,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_p2,
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
                L,
                R_p,
                R_m,
                R_a,
                C_p,
                C_a,
                f_r,
                f_p,
                gamma,
                c,
                f3dB,
                f_p2,
                f_r2,
                gamma2,
                c2,
                f3dB2,
                temperature,
                None,
            ]

        df = df.sort_values("Current, mA")
        df.reset_index(drop=True, inplace=True)

        temperature_list = df["Temperature, °C"].unique()

        df["Threshold current, mA"].loc[df["Temperature, °C"] == temperature] = float(
            auto_I_th
        )

        # # I_th from /LIV
        # liv_dir = directory.removesuffix("/PNA/") + "/LIV/"
        # print(f"liv dir {liv_dir}")
        # have_liv_dir = os.path.exists(liv_dir)
        # if have_liv_dir:
        #     walk = list(os.walk(liv_dir))
        #     string_for_re = ".*\\.csv$"
        #     r = re.compile(string_for_re)
        #     files = walk[0][2]
        #     matched_files = list(filter(r.match, files))
        #     matched_files.sort()
        #     print(f"Matched LIV files: {matched_files}")
        #     for temperature in temperature_list:
        #         string_for_re = f".*-{temperature}°C"
        #         r = re.compile(string_for_re)
        #         files = walk[0][2]
        #         matched_files = list(filter(r.match, matched_files))
        #         matched_files.sort()
        #         if matched_files:
        #             file = matched_files[0]
        #             liv = pd.read_csv(liv_dir + file)
        #             i = liv["Current, mA"]
        #             l = liv["Output power, mW"]
        #             first_der = np.gradient(l, i)
        #             second_der = np.gradient(first_der, i)
        #             if second_der.max() >= 5:
        #                 i_threshold = i[np.argmax(second_der >= 5)]  # mA
        #                 # l_threshold = l[np.argmax(second_der >= 5)]
        #                 df["Threshold current, mA"].loc[
        #                     df["Temperature, °C"] == temperature
        #                 ] = float(i_threshold)
        #             else:
        #                 i_threshold = None
        #         else:
        #             i_threshold = None
        #         print(f"I_threshold={i_threshold}")

        df["sqrt(I-I_th), sqrt(mA)"] = np.sqrt(
            df["Current, mA"] - df["Threshold current, mA"]
        )

        if not os.path.exists(report_dir):  # make directories
            os.makedirs(report_dir)
        df.to_csv(report_dir + name_from_dir + "-report(auto).csv")

    print(df)
    return df, directory, report_dir


def calc_K_D_MCEF(
    df,
    col_f_r="f_r, GHz",
    col_f_3dB="f_3dB, GHz",
    col_gamma="gamma",
    calc_i_limit=np.inf,
):
    newdf = (
        df[["Current, mA", "sqrt(I-I_th), sqrt(mA)", col_f_r, col_f_3dB, col_gamma]]
        .loc[df["Current, mA"] <= calc_i_limit]
        .dropna(subset=["sqrt(I-I_th), sqrt(mA)"])
    )
    calcdf = newdf[
        [
            "sqrt(I-I_th), sqrt(mA)",
            col_f_r,
            col_f_3dB,
            col_gamma,
        ]
    ].dropna()
    if len(calcdf) == 0:
        return None, None, None, None, None, None
    f_r_max = calcdf[col_f_r].iloc[-1]
    max_sqrtI = calcdf["sqrt(I-I_th), sqrt(mA)"].iloc[-1]
    if calcdf.shape[0] < 2:
        return max_sqrtI, f_r_max**2, None, None, None, None
    else:
        model = linear_model.LinearRegression(fit_intercept=False)
        X = calcdf["sqrt(I-I_th), sqrt(mA)"].values.reshape(-1, 1)
        y = calcdf[col_f_3dB]
        model.fit(X, y)
        MCEF = model.coef_[0]
        model = linear_model.LinearRegression(fit_intercept=False)
        X = calcdf["sqrt(I-I_th), sqrt(mA)"].values.reshape(-1, 1)
        y = calcdf[col_f_r]
        model.fit(X, y)
        D_par = model.coef_[0]
        model = linear_model.LinearRegression()
        X = calcdf[col_f_r] ** 2
        X = X.values.reshape(-1, 1)
        y = calcdf[col_gamma]
        model.fit(X, y)
        K_par = model.coef_[0]
        gamma0 = model.intercept_
        return max_sqrtI, f_r_max**2, MCEF, D_par, K_par, gamma0


def collect_K_D_MCEF(df, col_f_r="f_r, GHz", col_f_3dB="f_3dB, GHz", col_gamma="gamma"):
    K_D_MCEF_df = pd.DataFrame(
        columns=[
            "max sqrt(I-I_th), sqrt(mA)",
            "f_r_2_max",
            "MCEF",
            "D factor",
            "K factor, ns",
            "gamma0",
        ]
    ).dropna(subset=["max sqrt(I-I_th), sqrt(mA)"])
    if df is not None:
        for i in df["Current, mA"]:
            i = float(i)
            K_D_MCEF_df.loc[len(K_D_MCEF_df)] = calc_K_D_MCEF(
                df,
                col_f_r=col_f_r,
                col_f_3dB=col_f_3dB,
                col_gamma=col_gamma,
                calc_i_limit=i,
            )
        return K_D_MCEF_df.dropna()
    else:
        return None


#  _____ _
# |  ___(_) __ _ _   _ _ __ ___  ___
# | |_  | |/ _` | | | | '__/ _ \/ __|
# |  _| | | (_| | |_| | | |  __/\__ \
# |_|   |_|\__, |\__,_|_|  \___||___/
#          |___/
def makefigs(
    df,
    directory,
    s2p,
    fig_max_current,
    fig_max_freq,
    fig_max_gamma,
    fig_max_f_r2_for_gamma,
    K_D_MCEF_df,
    K_D_MCEF_df2,
):
    if df is None:
        return

    name_from_dir = (
        directory.replace("/", "-")
        .removesuffix("-")
        .removesuffix("-PNA")
        .removeprefix("data-")
    )

    fig = plt.figure(figsize=(3 * 11.69, 3 * 8.27))
    fig.suptitle(name_from_dir)

    # 1-st row
    ax1_l = fig.add_subplot(461)
    ax1_l.scatter(df["Current, mA"], df["L, pH"])
    ax1_l.set_ylabel("L, pH")
    ax1_l.set_xlabel("Current, mA")
    ax1_l.set_ylim([0, 300])
    # ax1_l.set_xlim(left=0)
    ax1_l.set_xlim([0, fig_max_current])
    ax1_l.grid(which="both")
    ax1_l.minorticks_on()

    ax2_r = fig.add_subplot(462)
    ax2_r.scatter(df["Current, mA"], df["R_p, Om"], label="R_p")
    ax2_r.scatter(df["Current, mA"], df["R_m, Om"], label="R_m")
    ax2_r.scatter(df["Current, mA"], df["R_a, Om"], label="R_a")
    ax2_r.set_ylabel("Resistance, Om")
    ax2_r.set_xlabel("Current, mA")
    ax2_r.set_ylim([0, 200])
    # ax2_r.set_xlim(left=0)
    ax2_r.set_xlim([0, fig_max_current])
    ax2_r.grid(which="both")
    ax2_r.minorticks_on()
    ax2_r.legend()

    ax3_c = fig.add_subplot(463)
    ax3_c.scatter(df["Current, mA"], df["C_p, fF"], label="C_p")
    ax3_c.scatter(df["Current, mA"], df["C_a, fF"], label="C_a")
    ax3_c.set_ylabel("Capacitance, fF")
    ax3_c.set_xlabel("Current, mA")
    # ax3_c.set_xlim(left=0)
    ax3_c.set_xlim([0, fig_max_current])
    ax3_c.set_ylim([0, 500])
    ax3_c.grid(which="both")
    ax3_c.minorticks_on()

    # 2-nd row
    ax7_gamma = fig.add_subplot(445)
    ax7_gamma.scatter(
        df["Current, mA"],
        df["gamma"],
        label="ɣ",
        alpha=0.5,
    )
    if fp_fixed:
        ax7_gamma.scatter(
            df["Current, mA"], df["gamma(f_p fixed)"], label="ɣ(f_p fixed)", alpha=0.5
        )
    ax7_gamma.set_ylabel("ɣ")
    ax7_gamma.set_xlabel("Current, mA")
    # ax7_gamma.set_xlim(left=0)
    ax7_gamma.set_xlim([0, fig_max_current])
    ax7_gamma.set_ylim([0, fig_max_gamma])
    ax7_gamma.grid(which="both")
    ax7_gamma.minorticks_on()
    if fp_fixed:
        ax7_gamma.legend()

    ax8_fp = fig.add_subplot(446)
    ax8_fp.scatter(
        df["Current, mA"],
        df["f_p, GHz"],
        label="from S21",
        alpha=0.5,
    )
    if fp_fixed:
        ax8_fp.scatter(
            df["Current, mA"],
            df["f_p(fixed), GHz"],
            label="from equivalent circuit",
            alpha=0.5,
        )
    ax8_fp.set_ylabel("f_p, GHz")
    ax8_fp.set_xlabel("Current, mA")
    # ax8_fp.set_xlim(left=0)
    ax8_fp.set_xlim([0, fig_max_current])
    ax8_fp.set_ylim([0, fig_max_freq])
    ax8_fp.grid(which="both")
    ax8_fp.minorticks_on()
    if fp_fixed:
        ax8_fp.legend()

    ax9_fr = fig.add_subplot(447)
    ax9_fr.scatter(
        df["Current, mA"],
        df["f_r, GHz"],
        label="f_r",
        alpha=0.5,
    )
    if fp_fixed:
        ax9_fr.scatter(
            df["Current, mA"],
            df["f_r(f_p fixed), GHz"],
            label="f_r(f_p fixed)",
            alpha=0.5,
        )
    ax9_fr.set_ylabel("f_r, GHz")
    ax9_fr.set_xlabel("Current, mA")
    # ax9_fr.set_xlim(left=0)
    ax9_fr.set_xlim([0, fig_max_current])
    ax9_fr.set_ylim([0, fig_max_freq])
    ax9_fr.grid(which="both")
    ax9_fr.minorticks_on()
    if fp_fixed:
        ax9_fr.legend()

    # ax8_c = fig.add_subplot(338)
    # ax8_c.plot(df["Current, mA"], df["c"], marker="o")
    # ax8_c.set_ylabel("c")
    # ax8_c.set_xlabel("Current, mA")
    # # ax8_c.set_xlim(left=0)
    # ax8_c.set_xlim([0, max_current])
    # ax8_c.set_ylim([-100, 0])
    # ax8_c.grid(which="both")
    # ax8_c.minorticks_on()

    ax10_f3db = fig.add_subplot(4, 4, 8)
    ax10_f3db.scatter(
        df["Current, mA"],
        df["f_3dB, GHz"],
        label="f_3dB",
        alpha=0.5,
    )
    if fp_fixed:
        ax10_f3db.scatter(
            df["Current, mA"],
            df["f_3dB(f_p fixed), GHz"],
            label="f_3dB(f_p fixed)",
            alpha=0.5,
        )
    ax10_f3db.set_ylabel("f_3dB, GHz")
    ax10_f3db.set_xlabel("Current, mA")
    # ax10_f3db.set_xlim(left=0)
    ax10_f3db.set_xlim([0, fig_max_current])
    ax10_f3db.set_ylim([0, fig_max_freq])
    ax10_f3db.grid(which="both")
    ax10_f3db.minorticks_on()

    # 3-rd row
    ax11_sqrt_gamma = fig.add_subplot(4, 4, 9)
    ax11_sqrt_gamma.scatter(
        df["f_r, GHz"] ** 2,
        df["gamma"],
        label="ɣ",
        alpha=0.5,
    )
    if fp_fixed:
        ax11_sqrt_gamma.scatter(
            df["f_r(f_p fixed), GHz"] ** 2,
            df["gamma(f_p fixed)"],
            label="ɣ(f_p fixed)",
            alpha=0.5,
        )
    ax11_sqrt_gamma.set_ylabel("ɣ")
    ax11_sqrt_gamma.set_xlabel("f_r^2, GHz^2")
    ax11_sqrt_gamma.set_xlim(left=0, right=fig_max_f_r2_for_gamma)
    ax11_sqrt_gamma.set_ylim(bottom=0, top=fig_max_gamma)
    ax11_sqrt_gamma.grid(which="both")
    ax11_sqrt_gamma.minorticks_on()
    if fp_fixed:
        ax11_sqrt_gamma.legend()

    # ax12_sqrt_fp = fig.add_subplot(4, 4, 10)
    # ax12_sqrt_fp.plot(
    #     df["sqrt(I-I_th), sqrt(mA)"],
    #     np.gradient(df["f_p, GHz"]),
    #     label="from S21",
    #     alpha=0.5,
    # )
    # if fp_fixed:
    #     ax12_sqrt_fp.plot(
    #         df["sqrt(I-I_th), sqrt(mA)"],
    #         np.gradient(df["f_p(fixed), GHz"]),
    #         label="from equivalent circuit",
    #         alpha=0.5,
    #     )
    # ax12_sqrt_fp.set_ylabel("dfp/d(sqrt(I-I_th), sqrt(mA))")
    # ax12_sqrt_fp.set_xlabel("sqrt(I-I_th), sqrt(mA)")
    # ax12_sqrt_fp.set_xlim(left=0)
    # ax12_sqrt_fp.set_ylim(bottom=-1, top=3)
    # ax12_sqrt_fp.grid(which="both")
    # ax12_sqrt_fp.minorticks_on()
    # if fp_fixed:
    #     ax12_sqrt_fp.legend()

    ax13_D = fig.add_subplot(4, 4, 11)
    # grad = np.gradient(df["f_r, GHz"], df["sqrt(I-I_th), sqrt(mA)"])
    # mean = grad[~np.isnan(grad)].mean()
    ax13_D.scatter(
        df["sqrt(I-I_th), sqrt(mA)"],
        # np.gradient(df["f_r, GHz"], df["sqrt(I-I_th), sqrt(mA)"]),
        # df["f_r, GHz"] / df["sqrt(I-I_th), sqrt(mA)"],
        df["f_r, GHz"],
        label=f"f_r",
        alpha=0.5,
    )
    # ax13_D.axhline(y=mean, color="b", linestyle="-.")
    if fp_fixed:
        # gradf = np.gradient(df["f_r(f_p fixed), GHz"], df["sqrt(I-I_th), sqrt(mA)"])
        # meanf = gradf[~np.isnan(gradf)].mean()
        ax13_D.scatter(
            df["sqrt(I-I_th), sqrt(mA)"],
            # np.gradient(df["f_r(f_p fixed), GHz"], df["sqrt(I-I_th), sqrt(mA)"]),
            # df["f_r(f_p fixed), GHz"] / df["sqrt(I-I_th), sqrt(mA)"],
            df["f_r(f_p fixed), GHz"],
            label="f_r(f_p fixed)",
            alpha=0.5,
        )
        # ax13_D.axhline(y=meanf, color="y", linestyle="-.")
    # ax13_D.set_ylabel("df_r/d(sqrt(I-I_th), sqrt(mA)), GHz/sqrt(mA)")
    ax13_D.set_ylabel("f_r, GHz")
    ax13_D.set_xlabel("sqrt(I-I_th), sqrt(mA)")
    ax13_D.set_xlim(left=0, right=np.sqrt(fig_max_current))
    ax13_D.set_ylim(bottom=0, top=fig_max_freq)
    ax13_D.grid(which="both")
    ax13_D.minorticks_on()
    if fp_fixed:
        ax13_D.legend()

    ax14_MCEF = fig.add_subplot(4, 4, 12)
    # grad = np.gradient(df["f_3dB, GHz"], df["sqrt(I-I_th), sqrt(mA)"])
    # mean = grad[~np.isnan(grad)].mean()
    ax14_MCEF.scatter(
        df["sqrt(I-I_th), sqrt(mA)"],
        # np.gradient(df["f_3dB, GHz"], df["sqrt(I-I_th), sqrt(mA)"]),
        # df["f_3dB, GHz"] / df["sqrt(I-I_th), sqrt(mA)"],
        df["f_3dB, GHz"],
        label=f"f_3dB",
        alpha=0.5,
    )
    # ax14_MCEF.axhline(y=mean, color="b", linestyle="-.")
    if fp_fixed:
        # gradf = np.gradient(df["f_3dB(f_p fixed), GHz"], df["sqrt(I-I_th), sqrt(mA)"])
        # meanf = gradf[~np.isnan(gradf)].mean()
        ax14_MCEF.scatter(
            df["sqrt(I-I_th), sqrt(mA)"],
            # np.gradient(df["f_3dB(f_p fixed), GHz"], df["sqrt(I-I_th), sqrt(mA)"]),
            # df["f_3dB(f_p fixed), GHz"] / df["sqrt(I-I_th), sqrt(mA)"],
            df["f_3dB(f_p fixed), GHz"],
            label=f"f_3dB(f_p fixed)",
            alpha=0.5,
        )
        # ax14_MCEF.axhline(y=meanf, color="y", linestyle="-.")
    # ax14_MCEF.set_ylabel("df_3dB/d(sqrt(I-I_th), sqrt(mA)), GHz/sqrt(mA)")
    ax14_MCEF.set_ylabel("f_3dB, GHz")
    ax14_MCEF.set_xlabel("sqrt(I-I_th), sqrt(mA)")
    ax14_MCEF.set_xlim(left=0, right=np.sqrt(fig_max_current))
    ax14_MCEF.set_ylim(bottom=0, top=fig_max_freq)
    ax14_MCEF.grid(which="both")
    ax14_MCEF.minorticks_on()
    if fp_fixed:
        ax14_MCEF.legend()

    # 4-th row
    ax15_K = fig.add_subplot(4, 4, 13)
    lns1 = ax15_K.plot(
        K_D_MCEF_df["f_r_2_max"],
        K_D_MCEF_df["K factor, ns"],
        label=f"K factor",
        alpha=0.5,
    )
    ax15_gamma0 = ax15_K.twinx()
    lns2 = ax15_gamma0.plot(
        K_D_MCEF_df["f_r_2_max"],
        K_D_MCEF_df["gamma0"],
        ":",
        label="ɣ_0",
        alpha=0.5,
    )
    if fp_fixed:
        lns3 = ax15_K.plot(
            K_D_MCEF_df2["f_r_2_max"],
            K_D_MCEF_df2["K factor, ns"],
            label=f"K factor(f_p fixed)",
            alpha=0.5,
        )
        lns4 = ax15_gamma0.plot(
            K_D_MCEF_df2["f_r_2_max"],
            K_D_MCEF_df2["gamma0"],
            ":",
            label="ɣ_0(f_p fixed)",
            alpha=0.5,
        )

    ax15_K.set_ylabel("K factor, ns")
    ax15_gamma0.set_ylabel("ɣ_0")
    ax15_K.set_xlabel("max f_r^2, GHz^2")
    ax15_K.set_xlim(left=0, right=fig_max_f_r2_for_gamma)
    ax15_K.set_ylim(bottom=0)
    ax15_gamma0.set_ylim(bottom=0)
    ax15_K.grid(which="both")
    ax15_K.minorticks_on()
    if fp_fixed:
        lns = lns1 + lns2 + lns3 + lns4
    else:
        lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax15_K.legend(lns, labs)

    ax16_D = fig.add_subplot(4, 4, 16)
    lns1 = ax16_D.plot(
        K_D_MCEF_df["max sqrt(I-I_th), sqrt(mA)"],
        K_D_MCEF_df["D factor"],
        label=f"D factor",
        alpha=0.5,
    )
    ax16_MCEF = ax16_D.twinx()
    lns2 = ax16_MCEF.plot(
        K_D_MCEF_df["max sqrt(I-I_th), sqrt(mA)"],
        K_D_MCEF_df["MCEF"],
        ":",
        label="MCEF",
        alpha=0.5,
    )
    if fp_fixed:
        lns3 = ax16_D.plot(
            K_D_MCEF_df2["max sqrt(I-I_th), sqrt(mA)"],
            K_D_MCEF_df2["D factor"],
            label=f"D factor(f_p fixed)",
            alpha=0.5,
        )
        lns4 = ax16_MCEF.plot(
            K_D_MCEF_df2["max sqrt(I-I_th), sqrt(mA)"],
            K_D_MCEF_df2["MCEF"],
            ":",
            label="MCEF(f_p fixed)",
            alpha=0.5,
        )

    ax16_D.set_ylabel("D factor")
    ax16_MCEF.set_ylabel("MCEF")
    ax16_D.set_xlabel("max sqrt(I-I_th), sqrt(mA)")
    ax16_D.set_xlim(left=0, right=np.sqrt(fig_max_current))
    ax16_D.set_ylim(bottom=0)
    ax16_MCEF.set_ylim(bottom=0)
    ax16_D.grid(which="both")
    ax16_D.minorticks_on()
    if fp_fixed:
        lns = lns1 + lns2 + lns3 + lns4
    else:
        lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax16_D.legend(lns, labs)

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

    annotate_max_f3db(df["Current, mA"], df["f_3dB, GHz"], ax=ax10_f3db)
    if fp_fixed:
        ax10_f3db.legend(loc=2)

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
        freqlimit=fit_freqlimit,
        S21_MSE_threshold=S21_MSE_threshold,
        fp_fixed=fp_fixed,
    )
    K_D_MCEF_df = collect_K_D_MCEF(
        df, col_f_r="f_r, GHz", col_f_3dB="f_3dB, GHz", col_gamma="gamma"
    )
    if fp_fixed:
        K_D_MCEF_df2 = collect_K_D_MCEF(
            df,
            col_f_r="f_r(f_p fixed), GHz",
            col_f_3dB="f_3dB(f_p fixed), GHz",
            col_gamma="gamma(f_p fixed)",
        )
    else:
        K_D_MCEF_df2 = None
    makefigs(
        df,
        report_dir,
        s2p=True,
        fig_max_current=fig_max_current,
        fig_max_freq=fig_max_freq,
        fig_max_gamma=fig_max_gamma,
        fig_max_f_r2_for_gamma=fig_max_f_r2_for_gamma,
        K_D_MCEF_df=K_D_MCEF_df,
        K_D_MCEF_df2=K_D_MCEF_df2,
    )

    print("K_D_MCEF_df")
    print(K_D_MCEF_df)
    print("\nK_D_MCEF_df2")
    print(K_D_MCEF_df2)

    df, dir, report_dir = analyse(
        directory,
        s2p=False,
        probe_port=probe_port,
        freqlimit=fit_freqlimit,
        S21_MSE_threshold=S21_MSE_threshold,
        fp_fixed=fp_fixed,
    )
    K_D_MCEF_df = collect_K_D_MCEF(
        df, col_f_r="f_r, GHz", col_f_3dB="f_3dB, GHz", col_gamma="gamma"
    )
    if fp_fixed:
        K_D_MCEF_df2 = collect_K_D_MCEF(
            df,
            col_f_r="f_r(f_p fixed), GHz",
            col_f_3dB="f_3dB(f_p fixed), GHz",
            col_gamma="gamma(f_p fixed)",
        )
    else:
        K_D_MCEF_df2 = None
    makefigs(
        df,
        report_dir,
        s2p=False,
        fig_max_current=fig_max_current,
        fig_max_freq=fig_max_freq,
        fig_max_gamma=fig_max_gamma,
        fig_max_f_r2_for_gamma=fig_max_f_r2_for_gamma,
        K_D_MCEF_df=K_D_MCEF_df,
        K_D_MCEF_df2=K_D_MCEF_df2,
    )
