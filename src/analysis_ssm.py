#!/usr/bin/env python3
#
# name convention for .s2p files:
# WaferID-wavelength-coordinates-currentmA
# e.g. HuiLi-850-0032-0.5mA

import sys
import re
import os
import yaml
import skrf as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from sklearn import linear_model
from pathlib import Path

# from builtins import str

from src.analysis_ssm_one_file import one_file_approximation


def name_from_directory(directory):
    name_from_dir = list(directory.parts)
    if name_from_dir[0] == "data":
        del name_from_dir[0]
    if name_from_dir[-1] == "PNA":
        del name_from_dir[-1]
    name_from_dir = "-".join(name_from_dir)
    return name_from_dir


# directory, file_name, probe_port, limit
# one_file_approximation("data", "745.s2p", 2, 50)
def analyze_ssm(
    directory,
    title=None,
    s2p=True,
    probe_port=1,
    threshold_decision_level=2,
    freqlimit=40,
    S21_MSE_threshold=5,
    i_th_fixed=False,
    fp_fixed=True,
    photodiode_s2p=None,
    report_dir="PNA_reports",  # TODO
    S11_bounds=None,
    S21_bounds=None,
):
    if isinstance(directory, str):
        directory = Path(directory)
    start_directory = directory
    report_dir = directory / report_dir
    # get filenames and currents
    if not s2p:
        matched_csv_files = sorted(
            directory.glob("*.csv")
        )  # TODO don't forget to iterate through files!
        matched_csv_files_stems = (
            f"{n}: " + i.stem for n, i in enumerate(matched_csv_files, start=1)
        )
        if matched_csv_files:
            auto_file_path = matched_csv_files[0]
            print(
                f"Matched .csv files: {len(matched_csv_files)}",
                *matched_csv_files_stems,
                "\n",
                sep="\n",
            )
            print(f"Processing .csv file: {auto_file_path.stem}")
        else:
            return None, directory, report_dir
    elif s2p:
        # check for .s2p files
        matched_s2p_files = sorted(directory.glob("*.s2p"))
        if not matched_s2p_files:
            print("Checking /PNA directory")
            directory = directory / "PNA"
            if not directory.is_dir():
                print(f"Can't find PNA data in {directory}")
                return None, directory, report_dir
            matched_s2p_files = sorted(directory.glob("*.s2p"))
        matched_s2p_files_stems = (
            f"{n}: " + i.stem for n, i in enumerate(matched_s2p_files, start=1)
        )
        print(
            f"Matched .s2p files: {len(matched_s2p_files)}",
            *matched_s2p_files_stems,
            sep="\n",
        )

    name_from_dir = name_from_directory(directory)

    df = pd.DataFrame(
        columns=[
            "Current, mA",
            "L, pH",
            "R_p_high, Om",
            "R_m, Om",
            "R_a, Om",
            "C_p_low, fF",
            "C_a, fF",
            "f1, GHz",
            "f2, GHz",
            "f3, GHz",
            "f_r, GHz",
            "f_p, GHz",
            "gamma, 1/ns",
            "c",
            "f_3dB, GHz",
            "f_p(fixed), GHz",
            "f_r(f_p fixed), GHz",
            "gamma(f_p fixed), 1/ns",
            "c(f_p fixed)",
            "f_3dB(f_p fixed), GHz",
            "Temperature, °C",
            "Threshold current, mA",
        ]
    )

    dict = {}

    if s2p:
        report_dir = start_directory / "PNA_reports(s2p)"
        report_dir.mkdir(exist_ok=True)
        for file_i, file in enumerate(matched_s2p_files):
            (
                f_GHz,
                S11_Real,
                S11_Imag,
                S21_Magnitude_to_fit,
                S21_Magnitude_fit,
                L,
                R_p_high,
                R_m,
                R_a,
                C_p_low,
                C_a,
                f1,
                f2,
                f3,
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
                title=title,
                freqlimit=freqlimit,
                file_path=file,
                probe_port=probe_port,
                S21_MSE_threshold=S21_MSE_threshold,
                fp_fixed=fp_fixed,
                photodiode_s2p=photodiode_s2p,
                S11_bounds=S11_bounds,
                S21_bounds=S21_bounds,
            )

            # parce file name for current and temperature
            file_name_parser = str(file.stem).split("-")
            r2 = re.compile(".*mA")
            current = list(filter(r2.match, file_name_parser))[0]
            current = float(current.removesuffix("mA"))
            print(f"[{file_i}/{len(matched_s2p_files)}] {current} mA ")
            r2 = re.compile(".*°C")
            filt = list(filter(r2.match, file_name_parser))
            if filt:
                temperature = filt[0]
                temperature = float(temperature.removesuffix("°C"))
            else:
                temperature = 25.0

            if len(S21_Magnitude_fit) < len(f_GHz):
                S21_Magnitude_fit = np.concatenate(
                    (
                        S21_Magnitude_fit - c,
                        np.array([None] * (len(f_GHz) - len(S21_Magnitude_fit))),
                    )
                )
            S21_Magnitude_to_fit = S21_Magnitude_to_fit - c

            S11r_name = f"{current} mA, {temperature} °C S11 Real "
            S11im_name = f"{current} mA, {temperature} °C S11 Imaginary"
            S21mag_name = f"{current} mA, {temperature} °C S21 LogMagnitude (dB)"
            S21mag_fit_name = (
                f"{current} mA, {temperature} °C S21 LogMagnitude Fit (dB)"
            )
            dict[S11r_name] = S11_Real
            dict[S11im_name] = S11_Imag
            dict[S21mag_name] = S21_Magnitude_to_fit
            dict[S21mag_fit_name] = S21_Magnitude_fit

            df.loc[len(df)] = [
                current,
                L,
                R_p_high,
                R_m,
                R_a,
                C_p_low,
                C_a,
                f1,
                f2,
                f3,
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
            df["Threshold current, mA"].loc[df["Temperature, °C"] == temperature] = (
                float(i_th_fixed)
            )
        else:
            temperature_list = df["Temperature, °C"].unique()
            liv_dir = start_directory / "LIV"
            if liv_dir.is_dir():
                for temperature in temperature_list:
                    matched_liv_files = sorted(liv_dir.glob(f"*-{temperature}°C*.csv"))
                    matched_liv_files_stems = (
                        f"{n}: " + i.stem
                        for n, i in enumerate(matched_liv_files, start=1)
                    )
                    print(
                        f"Matched LIV files: {len(matched_liv_files)}",
                        *matched_liv_files_stems,
                        "\n",
                        sep="\n",
                    )
                    if matched_liv_files:
                        file = matched_liv_files[0]
                        liv = pd.read_csv(file)
                        i = liv["Current, mA"]
                        l = liv["Output power, mW"]
                        first_der = np.gradient(l, i)
                        second_der = np.gradient(first_der, i)
                        if second_der.max() >= threshold_decision_level:
                            i_threshold = i[
                                np.argmax(second_der >= threshold_decision_level)
                            ]  # mA
                            # l_threshold = l[np.argmax(second_der >= 5)]
                            df.loc[
                                df["Temperature, °C"] == temperature,
                                "Threshold current, mA",
                            ] = float(i_threshold)
                    else:
                        i_threshold = None
                    print(f"I_threshold={i_threshold}")

        df["sqrt(I-I_th), sqrt(mA)"] = np.sqrt(
            df["Current, mA"] - df["Threshold current, mA"]
        )

        df.to_csv(report_dir / (name_from_dir + "-report(s2p).csv"), index=False)

        dict = pd.DataFrame(dict, index=f_GHz)
        dict.index.name = "Frequency, GHz"
        dict.to_csv(report_dir / (name_from_dir + ".csv"))

    elif not s2p:  # automatic system csv file parsing and processing
        report_dir = start_directory / f"PNA_reports({auto_file_path.stem})"
        report_dir.mkdir(exist_ok=True)
        auto_file = pd.read_csv(auto_file_path, header=[0, 1, 2], sep="\t")
        # print(auto_file.head())
        currents = (
            auto_file["VNA Current"][auto_file["VNA Current"] > 0]
            .dropna()
            .values.reshape(-1)
        )
        frequency = auto_file["Frequency1"].values.reshape(-1)
        abs_s21 = auto_file["Abs(S11)"].values.reshape(-1)
        re_s11 = auto_file["Abs(S21)"].values.reshape(-1)
        im_s11 = auto_file["Phase(S21)"].values.reshape(-1)
        points = np.where(abs_s21 == -999999999)[0][0]
        waferid_wl, coordinates, _ = auto_file_path.stem.split("_")
        waferid, wavelength = waferid_wl.split("-")
        coordinates = coordinates[:2] + coordinates[3:]
        temperature = 25.0
        auto_I_th = float(auto_file["Threshold current"].iloc[0].iloc[0])
        for i, current in enumerate(currents):
            print(f"[{i}/{len(currents)}] {current} mA ")
            start = i * points + i
            stop = (i + 1) * points + i
            (
                f_GHz,
                S11_Real,
                S11_Imag,
                S21_Magnitude_to_fit,
                S21_Magnitude_fit,
                L,
                R_p_high,
                R_m,
                R_a,
                C_p_low,
                C_a,
                f1,
                f2,
                f3,
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
                title=title,
                freqlimit=freqlimit,
                file_path=None,
                probe_port=None,
                waferid=waferid,
                wavelength=wavelength,
                coordinates=coordinates,
                current=current,
                #     temperature=None,
                frequency=frequency[0:points],
                s11re=re_s11[start:stop],
                s11im=im_s11[start:stop],
                s21mag=abs_s21[start:stop],
                S21_MSE_threshold=S21_MSE_threshold,
                fp_fixed=fp_fixed,
                photodiode_s2p=photodiode_s2p,
                S11_bounds=S11_bounds,
                S21_bounds=S21_bounds,
            )

            if len(S21_Magnitude_fit) < len(f_GHz):
                S21_Magnitude_fit = np.concatenate(
                    (
                        S21_Magnitude_fit - c,
                        np.array([None] * (len(f_GHz) - len(S21_Magnitude_fit))),
                    )
                )
            S21_Magnitude_to_fit = S21_Magnitude_to_fit - c

            S11r_name = f"{current} mA, {temperature} °C S11 Real"
            S11im_name = f"{current} mA, {temperature} °C S11 Imaginary"
            S21mag_name = f"{current} mA, {temperature} °C S21 LogMagnitude (dB)"
            S21mag_fit_name = (
                f"{current} mA, {temperature} °C S21 LogMagnitude Fit (dB)"
            )
            dict[S11r_name] = S11_Real
            dict[S11im_name] = S11_Imag
            dict[S21mag_name] = S21_Magnitude_to_fit
            dict[S21mag_fit_name] = S21_Magnitude_fit

            df.loc[len(df)] = [
                current,
                L,
                R_p_high,
                R_m,
                R_a,
                C_p_low,
                C_a,
                f1,
                f2,
                f3,
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

        df.loc[df["Temperature, °C"] == temperature, "Threshold current, mA"] = float(
            auto_I_th
        )

        df["sqrt(I-I_th), sqrt(mA)"] = np.sqrt(
            df["Current, mA"] - df["Threshold current, mA"]
        )

        df.to_csv(report_dir / (name_from_dir + "-report(auto).csv"), index=False)

        dict = pd.DataFrame(dict, index=f_GHz)
        dict.index.name = "Frequency, GHz"
        dict.to_csv(report_dir / (name_from_dir + ".csv"))

    return df, directory, report_dir


def calc_K_D_MCEF(
    df,
    col_f_r="f_r, GHz",
    col_f_3dB="f_3dB, GHz",
    col_gamma="gamma, 1/ns",
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


def collect_K_D_MCEF(
    df, col_f_r="f_r, GHz", col_f_3dB="f_3dB, GHz", col_gamma="gamma, 1/ns"
):
    K_D_MCEF_df = pd.DataFrame(
        columns=[
            "max sqrt(I-I_th), sqrt(mA)",
            "f_r_2_max",
            "MCEF",
            "D factor",
            "K factor, ns",
            "gamma0, 1/ns",
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
    K_D_MCEF_df,
    K_D_MCEF_df2,
    title=None,
    additional_report_directory=None,
    figure_ec_ind_max=None,
    figure_ec_res_max=None,
    figure_ec_cap_max=None,
    figure_ec_fitpar_max=None,
    figure_max_current=None,
    figure_max_freq=None,
    figure_max_gamma=None,
    figure_max_f_r2_for_gamma=None,
    figure_K_max=None,
    figure_gamma0_max=None,
    figure_D_MCEF_max=None,
    fp_fixed=None,
):
    if df is None:
        return

    if isinstance(directory, str):
        directory = Path(directory)

    name_from_dir = name_from_directory(directory)

    fig = plt.figure(figsize=(3 * 11.69, 3 * 8.27))
    if title:
        fig.suptitle(title, fontsize=40)
    else:
        fig.suptitle(name_from_dir, fontsize=40)

    # 1-st row
    ax1_l = fig.add_subplot(461)
    ax1_l.set_title("Inductance of the equivalent circuit")
    ax1_l.plot(df["Current, mA"], df["L, pH"], marker="o")
    ax1_l.set_xlabel("Current, mA")
    ax1_l.set_ylabel("L, pH")
    ax1_l.set_xlim(left=0, right=figure_max_current)
    ax1_l.set_ylim(bottom=0, top=figure_ec_ind_max)
    ax1_l.grid(which="both")
    ax1_l.minorticks_on()

    ax2_r = fig.add_subplot(462)
    ax2_r.set_title("Resistance of the equivalent circuit")
    ax2_r.plot(df["Current, mA"], df["R_p_high, Om"], label="R_p_high", marker="o")
    ax2_r.plot(df["Current, mA"], df["R_m, Om"], label="R_m", marker="D")
    ax2_r.plot(df["Current, mA"], df["R_a, Om"], label="R_a", marker="*")
    ax2_r.set_xlabel("Current, mA")
    ax2_r.set_ylabel("Resistance, Om")
    ax2_r.set_xlim(left=0, right=figure_max_current)
    ax2_r.set_ylim(bottom=0, top=figure_ec_res_max)
    ax2_r.grid(which="both")
    ax2_r.minorticks_on()
    ax2_r.legend()

    ax3_c = fig.add_subplot(463)
    ax3_c.set_title("Capacitance of the equivalent circuit")
    ax3_c.plot(df["Current, mA"], df["C_p_low, fF"], label="C_p_low", marker="o")
    ax3_c.plot(df["Current, mA"], df["C_a, fF"], label="C_a", marker="D")
    ax3_c.set_ylabel("Capacitance, fF")
    ax3_c.set_xlabel("Current, mA")
    ax3_c.set_xlim(left=0, right=figure_max_current)
    ax3_c.set_ylim(bottom=0, top=figure_ec_cap_max)
    ax3_c.grid(which="both")
    ax3_c.minorticks_on()
    ax3_c.legend()

    ax4_f = fig.add_subplot(464)
    ax4_f.set_title("Fitting parameters of the equivalent circuit")
    ax4_f.plot(df["Current, mA"], df["f1, GHz"], label="f1", marker="o")
    ax4_f.plot(df["Current, mA"], df["f2, GHz"], label="f2", marker="D")
    ax4_f.plot(df["Current, mA"], df["f3, GHz"], label="f3", marker="*")
    ax4_f.set_ylabel("Fitting parameters, GHz")
    ax4_f.set_xlabel("Current, mA")
    ax4_f.set_xlim(left=0, right=figure_max_current)
    ax4_f.set_ylim(bottom=0, top=figure_ec_fitpar_max)
    ax4_f.grid(which="both")
    ax4_f.minorticks_on()
    ax4_f.legend()

    # 2-nd row
    ax7_gamma = fig.add_subplot(445)
    ax7_gamma.set_title("ɣ from S21 approximation")
    ax7_gamma.plot(
        df["Current, mA"],
        df["gamma, 1/ns"],
        label="ɣ, 1/ns",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax7_gamma.plot(
            df["Current, mA"],
            df["gamma(f_p fixed), 1/ns"],
            label="ɣ(f_p fixed), 1/ns",
            alpha=0.5,
            marker="o",
        )
    ax7_gamma.set_ylabel("ɣ, 1/ns")
    ax7_gamma.set_xlabel("Current, mA")
    ax7_gamma.set_xlim(left=0, right=figure_max_current)
    ax7_gamma.set_ylim(bottom=0, top=figure_max_gamma)
    ax7_gamma.grid(which="both")
    ax7_gamma.minorticks_on()
    if fp_fixed:
        ax7_gamma.legend()

    ax8_fp = fig.add_subplot(446)
    ax8_fp.set_title(
        "Parasitic cut-off frequiencies from S21 approximation and equivalent circuit"
    )
    ax8_fp.plot(
        df["Current, mA"],
        df["f_p, GHz"],
        label="from S21",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax8_fp.plot(
            df["Current, mA"],
            df["f_p(fixed), GHz"],
            label="from equivalent circuit",
            marker="o",
            alpha=0.5,
        )
    ax8_fp.set_ylabel("f_p, GHz")
    ax8_fp.set_xlabel("Current, mA")
    ax8_fp.set_xlim(left=0, right=figure_max_current)
    ax8_fp.set_ylim(bottom=0, top=figure_max_freq)
    ax8_fp.grid(which="both")
    ax8_fp.minorticks_on()
    if fp_fixed:
        ax8_fp.legend()

    ax9_fr = fig.add_subplot(447)
    ax9_fr.set_title("Resonance frequencies from S21 approximation")
    ax9_fr.plot(
        df["Current, mA"],
        df["f_r, GHz"],
        label="f_r",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax9_fr.plot(
            df["Current, mA"],
            df["f_r(f_p fixed), GHz"],
            label="f_r(f_p fixed)",
            marker="o",
            alpha=0.5,
        )
    ax9_fr.set_ylabel("f_r, GHz")
    ax9_fr.set_xlabel("Current, mA")
    ax9_fr.set_xlim(left=0, right=figure_max_current)
    ax9_fr.set_ylim(bottom=0, top=figure_max_freq)
    ax9_fr.grid(which="both")
    ax9_fr.minorticks_on()
    if fp_fixed:
        ax9_fr.legend()

    ax10_f3db = fig.add_subplot(4, 4, 8)
    ax10_f3db.set_title("f3dB frequencies from S21 approximation")
    ax10_f3db.plot(
        df["Current, mA"],
        df["f_3dB, GHz"],
        label="f_3dB",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax10_f3db.plot(
            df["Current, mA"],
            df["f_3dB(f_p fixed), GHz"],
            label="f_3dB(f_p fixed)",
            marker="o",
            alpha=0.5,
        )
    ax10_f3db.set_ylabel("f_3dB, GHz")
    ax10_f3db.set_xlabel("Current, mA")
    ax10_f3db.set_xlim(left=0, right=figure_max_current)
    ax10_f3db.set_ylim(bottom=0, top=figure_max_freq)
    ax10_f3db.grid(which="both")
    ax10_f3db.minorticks_on()

    # 3-rd row
    ax11_sqrt_gamma = fig.add_subplot(4, 4, 9)
    ax11_sqrt_gamma.set_title("ɣ vs f_r^2")
    ax11_sqrt_gamma.plot(
        df["f_r, GHz"] ** 2,
        df["gamma, 1/ns"],
        label="ɣ, 1/ns",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax11_sqrt_gamma.plot(
            df["f_r(f_p fixed), GHz"] ** 2,
            df["gamma(f_p fixed), 1/ns"],
            label="ɣ(f_p fixed), 1/ns",
            marker="o",
            alpha=0.5,
        )
    ax11_sqrt_gamma.set_ylabel("ɣ, 1/ns")
    ax11_sqrt_gamma.set_xlabel("f_r^2, GHz^2")
    ax11_sqrt_gamma.set_xlim(left=0, right=figure_max_f_r2_for_gamma)
    ax11_sqrt_gamma.set_ylim(bottom=0, top=figure_max_gamma)
    ax11_sqrt_gamma.grid(which="both")
    ax11_sqrt_gamma.minorticks_on()
    if fp_fixed:
        ax11_sqrt_gamma.legend()

    ax13_fr_for_D = fig.add_subplot(4, 4, 11)
    ax13_fr_for_D.set_title("f_r vs sqrt(I-Ith) for D factor derivation")
    ax13_fr_for_D.plot(
        df["sqrt(I-I_th), sqrt(mA)"],
        df["f_r, GHz"],
        label=f"f_r",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax13_fr_for_D.plot(
            df["sqrt(I-I_th), sqrt(mA)"],
            df["f_r(f_p fixed), GHz"],
            label="f_r(f_p fixed)",
            marker="o",
            alpha=0.5,
        )
    ax13_fr_for_D.set_ylabel("f_r, GHz")
    ax13_fr_for_D.set_xlabel("sqrt(I-I_th), sqrt(mA)")
    if figure_max_current:
        ax13_fr_for_D.set_xlim(left=0, right=np.sqrt(figure_max_current))
    else:
        ax13_fr_for_D.set_xlim(left=0)
    ax13_fr_for_D.set_ylim(bottom=0, top=figure_max_freq)
    ax13_fr_for_D.grid(which="both")
    ax13_fr_for_D.minorticks_on()
    if fp_fixed:
        ax13_fr_for_D.legend()

    ax14_f3dB_for_MCEF = fig.add_subplot(4, 4, 12)
    ax14_f3dB_for_MCEF.set_title("f_3dB vs sqrt(I-Ith) for MCEF derivation")
    ax14_f3dB_for_MCEF.plot(
        df["sqrt(I-I_th), sqrt(mA)"],
        df["f_3dB, GHz"],
        label=f"f_3dB",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax14_f3dB_for_MCEF.plot(
            df["sqrt(I-I_th), sqrt(mA)"],
            df["f_3dB(f_p fixed), GHz"],
            label=f"f_3dB(f_p fixed)",
            marker="o",
            alpha=0.5,
        )
    ax14_f3dB_for_MCEF.set_ylabel("f_3dB, GHz")
    ax14_f3dB_for_MCEF.set_xlabel("sqrt(I-I_th), sqrt(mA)")
    if figure_max_current:
        ax14_f3dB_for_MCEF.set_xlim(left=0, right=np.sqrt(figure_max_current))
    else:
        ax14_f3dB_for_MCEF.set_xlim(left=0)
    ax14_f3dB_for_MCEF.set_ylim(bottom=0, top=figure_max_freq)
    ax14_f3dB_for_MCEF.grid(which="both")
    ax14_f3dB_for_MCEF.minorticks_on()
    if fp_fixed:
        ax14_f3dB_for_MCEF.legend()

    # 4-th row
    ax15_K = fig.add_subplot(4, 4, 13)
    ax15_K.set_title("K factor for different appoximation limits")
    ax15_K.plot(
        K_D_MCEF_df["f_r_2_max"],
        K_D_MCEF_df["K factor, ns"],
        label=f"K factor",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax15_K.plot(
            K_D_MCEF_df2["f_r_2_max"],
            K_D_MCEF_df2["K factor, ns"],
            label=f"K factor(f_p fixed)",
            marker="o",
            alpha=0.5,
        )
    ax15_K.set_ylabel("K factor, ns")
    ax15_K.set_xlabel("max f_r^2, GHz^2")
    ax15_K.set_xlim(left=0, right=figure_max_f_r2_for_gamma)
    ax15_K.set_ylim(bottom=0, top=figure_K_max)
    ax15_K.grid(which="both")
    ax15_K.minorticks_on()
    ax15_K.legend()

    ax16_gamma0 = fig.add_subplot(4, 4, 14)
    ax16_gamma0.set_title("ɣ0 for different appoximation limits")
    ax16_gamma0.plot(
        K_D_MCEF_df["f_r_2_max"],
        K_D_MCEF_df["gamma0, 1/ns"],
        label="ɣ_0, 1/ns",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax16_gamma0.plot(
            K_D_MCEF_df2["f_r_2_max"],
            K_D_MCEF_df2["gamma0, 1/ns"],
            label="ɣ_0(f_p fixed), 1/ns",
            marker="o",
            alpha=0.5,
        )

    ax16_gamma0.set_ylabel("ɣ_0, 1/ns")
    ax16_gamma0.set_xlabel("max f_r^2, GHz^2")
    ax16_gamma0.set_xlim(left=0, right=figure_max_f_r2_for_gamma)
    ax16_gamma0.set_ylim(bottom=0, top=figure_gamma0_max)
    ax16_gamma0.grid(which="both")
    ax16_gamma0.minorticks_on()
    ax15_K.legend()

    ax17_D = fig.add_subplot(4, 4, 15)
    ax17_D.set_title("D factor for different appoximation limits")
    ax17_D.plot(
        K_D_MCEF_df["max sqrt(I-I_th), sqrt(mA)"],
        K_D_MCEF_df["D factor"],
        label=f"D factor",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax17_D.plot(
            K_D_MCEF_df2["max sqrt(I-I_th), sqrt(mA)"],
            K_D_MCEF_df2["D factor"],
            label=f"D factor(f_p fixed)",
            marker="o",
            alpha=0.5,
        )

    ax17_D.set_ylabel("D factor, GHz/sqrt(mA)")
    ax17_D.set_xlabel("max sqrt(I-I_th), sqrt(mA)")
    if figure_max_current:
        ax17_D.set_xlim(left=0, right=np.sqrt(figure_max_current))
    else:
        ax17_D.set_xlim(left=0)
    ax17_D.set_ylim(bottom=0, top=figure_D_MCEF_max)
    ax17_D.grid(which="both")
    ax17_D.minorticks_on()
    ax17_D.legend()

    ax18_MCEF = fig.add_subplot(4, 4, 16)
    ax18_MCEF.set_title("MCEF for different appoximation limits")
    ax18_MCEF.plot(
        K_D_MCEF_df["max sqrt(I-I_th), sqrt(mA)"],
        K_D_MCEF_df["MCEF"],
        label="MCEF",
        marker="o",
        alpha=0.5,
    )
    if fp_fixed:
        ax18_MCEF.plot(
            K_D_MCEF_df2["max sqrt(I-I_th), sqrt(mA)"],
            K_D_MCEF_df2["MCEF"],
            label="MCEF(f_p fixed)",
            marker="o",
            alpha=0.5,
        )

    ax18_MCEF.set_ylabel("MCEF, GHz/sqrt(mA)")
    ax18_MCEF.set_xlabel("max sqrt(I-I_th), sqrt(mA)")
    if figure_max_current:
        ax18_MCEF.set_xlim(left=0, right=np.sqrt(figure_max_current))
    else:
        ax18_MCEF.set_xlim(left=0)
    ax18_MCEF.set_ylim(bottom=0, top=figure_D_MCEF_max)
    ax18_MCEF.grid(which="both")
    ax18_MCEF.minorticks_on()
    ax18_MCEF.legend()

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

    directory.mkdir(exist_ok=True)
    plt.savefig(directory / (name_from_dir + ".png"))  # save figure

    if additional_report_directory:
        if isinstance(additional_report_directory, str):
            additional_report_directory = Path(additional_report_directory)
        additional_report_directory.mkdir(exist_ok=True)
        plt.savefig(
            additional_report_directory / (name_from_dir + ".png")
        )  # save figure

    plt.close()


# for i, directory in enumerate(sys.argv[1:]):
#     le = len(sys.argv[1:])
#     print(f"[{i+1}/{le}] {directory}")
def analyze_ssm_function(directory, settings=None):
    if isinstance(directory, str):
        directory = Path(directory)
    if settings is None:
        with open(Path("templates") / "ssm.yaml") as fh:
            settings = yaml.safe_load(fh)
    print(directory)
    print(settings)

    title = settings["title"]
    additional_report_directory = settings["additional_report_directory"]
    S21_MSE_threshold = settings["S21_MSE_threshold"]
    probe_port = settings["probe_port"]
    fit_freqlimit = settings["fit_freqlimit"]
    i_th_fixed = settings["i_th_fixed"]
    fp_fixed = settings["fp_fixed"]
    threshold_decision_level = settings["threshold_decision_level"]
    figure_ec_ind_max = settings["figure_ec_ind_max"]
    figure_ec_res_max = settings["figure_ec_res_max"]
    figure_ec_cap_max = settings["figure_ec_cap_max"]
    figure_ec_fitpar_max = settings["figure_ec_fitpar_max"]
    figure_max_current = settings["figure_max_current"]
    figure_max_freq = settings["figure_max_freq"]
    figure_max_gamma = settings["figure_max_gamma"]
    figure_max_f_r2_for_gamma = settings["figure_max_f_r2_for_gamma"]
    figure_K_max = settings["figure_K_max"]
    figure_gamma0_max = settings["figure_gamma0_max"]
    figure_D_MCEF_max = settings["figure_D_MCEF_max"]
    L_bounds = settings["L_bounds"]
    R_p_high_bounds = settings["R_p_high_bounds"]
    R_m_bounds = settings["R_m_bounds"]
    R_a_bounds = settings["R_a_bounds"]
    C_p_low_bounds = settings["C_p_low_bounds"]
    C_a_bounds = settings["C_a_bounds"]
    f1_bounds = settings["f1_bounds"]
    f2_bounds = settings["f2_bounds"]
    f3_bounds = settings["f3_bounds"]
    f3_bounds = settings["f3_bounds"]
    f_r_bounds = settings["f_r_bounds"]
    f_p_bounds = settings["f_p_bounds"]
    gamma_bounds = settings["gamma_bounds"]
    c_bounds = settings["c_bounds"]
    S11_bounds = (
        [
            L_bounds[0],
            R_p_high_bounds[0],
            R_m_bounds[0],
            R_a_bounds[0],
            C_p_low_bounds[0],
            C_a_bounds[0],
            f1_bounds[0],
            f2_bounds[0],
            f3_bounds[0],
        ],
        [
            L_bounds[1],
            R_p_high_bounds[1],
            R_m_bounds[1],
            R_a_bounds[1],
            C_p_low_bounds[1],
            C_a_bounds[1],
            f1_bounds[1],
            f2_bounds[1],
            f3_bounds[1],
        ],
    )

    S21_bounds = (
        [
            f_r_bounds[0],
            f_p_bounds[0],
            gamma_bounds[0],
            c_bounds[0],
        ],
        [
            f_r_bounds[1],
            f_p_bounds[1],
            gamma_bounds[1],
            c_bounds[1],
        ],
    )

    photodiode_s2p = rf.Network("resources/T3K7V9_DXM30BF_U00162.s2p")

    for s2p in (True, False):
        print(f"s2p: {s2p}")
        df, dir, report_dir = analyze_ssm(
            directory,
            title=title,
            s2p=s2p,
            probe_port=probe_port,
            threshold_decision_level=threshold_decision_level,
            freqlimit=fit_freqlimit,
            S21_MSE_threshold=S21_MSE_threshold,
            i_th_fixed=i_th_fixed,
            fp_fixed=fp_fixed,
            photodiode_s2p=photodiode_s2p,
            S11_bounds=S11_bounds,
            S21_bounds=S21_bounds,
        )
        K_D_MCEF_df = collect_K_D_MCEF(
            df, col_f_r="f_r, GHz", col_f_3dB="f_3dB, GHz", col_gamma="gamma, 1/ns"
        )

        name_from_dir = name_from_directory(directory)

        if K_D_MCEF_df is not None:
            K_D_MCEF_df.to_csv(
                report_dir / (name_from_dir + "-K_D_MCEF.csv"), index=False
            )

        if fp_fixed:
            K_D_MCEF_df2 = collect_K_D_MCEF(
                df,
                col_f_r="f_r(f_p fixed), GHz",
                col_f_3dB="f_3dB(f_p fixed), GHz",
                col_gamma="gamma(f_p fixed), 1/ns",
            )
            if K_D_MCEF_df2 is not None:
                K_D_MCEF_df2.to_csv(
                    report_dir / (name_from_dir + "-K_D_MCEF(f_p fixed).csv"),
                    index=False,
                )
        else:
            K_D_MCEF_df2 = None
        makefigs(
            df,
            report_dir,
            K_D_MCEF_df=K_D_MCEF_df,
            K_D_MCEF_df2=K_D_MCEF_df2,
            title=title,
            additional_report_directory=additional_report_directory,
            figure_ec_ind_max=figure_ec_ind_max,
            figure_ec_res_max=figure_ec_res_max,
            figure_ec_cap_max=figure_ec_cap_max,
            figure_ec_fitpar_max=figure_ec_fitpar_max,
            figure_max_current=figure_max_current,
            figure_max_freq=figure_max_freq,
            figure_max_gamma=figure_max_gamma,
            figure_max_f_r2_for_gamma=figure_max_f_r2_for_gamma,
            figure_K_max=figure_K_max,
            figure_gamma0_max=figure_gamma0_max,
            figure_D_MCEF_max=figure_D_MCEF_max,
            fp_fixed=fp_fixed,
        )
        # if s2p:
        #     print(".s2p")
        # else:
        #     print("automatic system .csv")
        # print("K_D_MCEF_df")
        # print(K_D_MCEF_df)
        # print("\nK_D_MCEF_df2")
        # print(K_D_MCEF_df2)
