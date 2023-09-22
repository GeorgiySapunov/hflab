#!/usr/bin/env python3

import os
import skrf as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error


def one_file_approximation(directory, file_name, probe_port, freqlimit):
    full_path = directory + "/" + file_name
    if probe_port == 1:
        optical_port = 2
    elif probe_port == 2:
        optical_port = 1
    else:
        raise Exception("probe_port is not 1 or 2")

    vcsel_ntwk = rf.Network(full_path)
    photodiode = rf.Network("resources/T3K7V9_DXM30BF_U00162.s2p")
    # ntwk = rf.pna_csv_2_ntwks3(
    #     "745.csv"
    # )  # Read a CSV file exported from an Agilent PNA in dB/deg format
    vcsel_df = vcsel_ntwk.to_dataframe("s")
    pd_df = photodiode.to_dataframe("s")
    vcsel_df = vcsel_df[vcsel_df.index < freqlimit * 10**9]  # Frequency less then
    vcsel_df["s11_re"] = vcsel_df[f"s {probe_port}{probe_port}"].values.real
    vcsel_df["s11_im"] = vcsel_df[f"s {probe_port}{probe_port}"].values.imag
    vcsel_df["s21_re"] = vcsel_df[f"s {optical_port}{probe_port}"].values.real
    vcsel_df["s21_im"] = vcsel_df[f"s {optical_port}{probe_port}"].values.imag
    f = vcsel_df.index.values
    pd_df.index = pd_df.index.values * 10**9  # fixing index in photodiodes .s2p file

    #  ____  _ _
    # / ___|/ / |
    # \___ \| | |
    #  ___) | | |
    # |____/|_|_|
    # from Hui Li theses p.65
    def s11_func(f, R_m, R_j, C_p, C_m):
        z1 = R_m + 1 / (1 / R_j + 2 * np.pi * 1j * f * C_m * 10**-15)
        z2 = 1 / (2 * np.pi * f * C_p * 10**-15 * 1j)
        z = 1 / (1 / z1 + 1 / z2)
        return (z - 50) / (z + 50)

    # stacking S11 data to fit them simultaneously
    def s11_both_func(f, R_m, R_j, C_p, C_m):
        N = len(f)
        f_real = f[: N // 2]
        f_imag = f[N // 2 :]
        y_real = np.real(s11_func(f_real, R_m, R_j, C_p, C_m))
        y_imag = np.imag(s11_func(f_imag, R_m, R_j, C_p, C_m))
        return np.hstack([y_real, y_imag])

    # Split the measurements into a real and imaginary part
    S11_Real = vcsel_df["s11_re"]
    S11_Imag = vcsel_df["s11_im"]
    S11_Both = np.hstack([S11_Real, S11_Imag])

    # Find the best-fit solution
    poptBoth_S11, _ = curve_fit(
        s11_both_func,
        np.hstack([f, f]),
        S11_Both,
        # p0=[150, 150, 30, 150],
        bounds=(0, [500, 500, 5000, 5000]),
        maxfev=100000,
    )
    R_m, R_j, C_p, C_m = poptBoth_S11

    # Compute the best-fit solution and check the mean squared error
    S11_Fit = s11_func(f, *poptBoth_S11)
    S11_Fit_both = s11_both_func(np.hstack([f, f]), *poptBoth_S11)
    S11_Fit_MSE = mean_squared_error(S11_Both, S11_Fit_both)
    # print(f"S11_Fit_MSE={S11_Fit_MSE}")
    S11_Fit_MSE_real = mean_squared_error(S11_Real, np.real(S11_Fit))
    # print(f"S11_Fit_MSE_real={S11_Fit_MSE_real}")
    S11_Fit_MSE_imag = mean_squared_error(S11_Imag, np.imag(S11_Fit))
    # print(f"S11_Fit_MSE_imag={S11_Fit_MSE_imag}")

    # equivalent impedance (Hui Li thesis p.66)
    def Z(s11, z0=50):
        Z = z0 * (1 + s11) / (1 - s11)
        return Z

    z = Z(S11_Fit)
    zmag = np.log(np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2))

    # Normalised low pass magnitude
    # TODO

    #  ____ ____  _
    # / ___|___ \/ |
    # \___ \ __) | |
    #  ___) / __/| |
    # |____/_____|_|
    # from Michalzik p. 55
    def s21_func(f, f_r, f_p, gamma, c):
        f = f * 10**-9
        f_r = f_r
        f_p = f_p
        h2 = c + 20 * np.log(
            np.sqrt(
                f_r**4
                / (
                    ((f_r**2 - f**2) ** 2 + (f * gamma / (2 * np.pi)) ** 2)
                    * (1 + (f / f_p) ** 2)
                )
            )
        )
        return h2

    # Split the measurements into a real and imaginary part
    S21_Real = vcsel_df["s21_re"]
    S21_Imag = vcsel_df["s21_im"]
    S21_Magnitude = 10 * np.log(S21_Real**2 + S21_Imag**2)

    # substracting photodiode S21
    pd_df["pd_s21_re"] = pd_df["s 21"].values.real
    pd_df["pd_s21_im"] = pd_df["s 21"].values.imag
    vcsel_df = vcsel_df.join(pd_df[["pd_s21_re", "pd_s21_im"]], how="outer")
    vcsel_df["pd_s21_re"] = vcsel_df["pd_s21_re"].interpolate()
    vcsel_df["pd_s21_im"] = vcsel_df["pd_s21_im"].interpolate()
    vcsel_df = vcsel_df.dropna()
    pd_Real = vcsel_df["pd_s21_re"]
    pd_Imag = vcsel_df["pd_s21_im"]
    pd_Magnitude = 10 * np.log(pd_Real**2 + pd_Imag**2)
    S21_Magnitude_subpd = S21_Magnitude - pd_Magnitude

    # Find the best-fit solution for S21
    # f_r, f_p, gamma, c
    popt_S21, pcovBoth = curve_fit(
        s21_func,
        f,
        S21_Magnitude_subpd,
        # p0=[150, 150, 30, 150, 0],
        bounds=([0, 0, 0, -np.inf], [100, 100, np.inf, np.inf]),
        maxfev=100000,
    )
    f_r, f_p, gamma, c = popt_S21

    # Compute the best-fit solution and mean squered error for S21
    S21_Fit = s21_func(f, *popt_S21)
    S21_Fit_MSE = mean_squared_error(S21_Magnitude_subpd, S21_Fit)
    # print(f"S21_Fit_MSE={S21_Fit_MSE}")
    f3dB = f[np.where(S21_Fit < c - 3)[0][0]] * 10**-9  # GHz

    #  _____ _
    # |  ___(_) __ _ _   _ _ __ ___  ___
    # | |_  | |/ _` | | | | '__/ _ \/ __|
    # |  _| | | (_| | |_| | | |  __/\__ \
    # |_|   |_|\__, |\__,_|_|  \___||___/
    #          |___/
    fig = plt.figure(figsize=(20, 10))

    ax1_s11re = fig.add_subplot(221)
    ax1_s11re.plot(f * 10**-9, vcsel_df["s11_re"], "k.", label="Experiment S11 Real")
    ax1_s11re.plot(
        f * 10**-9,
        np.real(S11_Fit),
        label=f"Best fit S11 Real R_m={poptBoth_S11[0]:.2f}, R_j={poptBoth_S11[1]:.2f}, C_p={poptBoth_S11[2]:.2f}, C_m={poptBoth_S11[3]:.2f}",
    )
    ax1_s11re.set_ylabel("Re(S11), dB")
    ax1_s11re.set_xlabel("Frequency, GHz")
    ax1_s11re.legend()
    ax1_s11re.grid(which="both")
    ax1_s11re.minorticks_on()

    ax2_s11im = fig.add_subplot(222)
    ax2_s11im.plot(
        f * 10**-9,
        np.real(vcsel_df["s11_im"]),
        "k.",
        label="Experiment S11 Imaginary",
    )
    ax2_s11im.plot(
        f * 10**-9,
        np.imag(S11_Fit),
        label=f"Best fit S11 Imaginary R_m={poptBoth_S11[0]:.2f}, R_j={poptBoth_S11[1]:.2f}, C_p={poptBoth_S11[2]:.2f}, C_m={poptBoth_S11[3]:.2f}",
    )
    ax2_s11im.set_ylabel("Im(S11), dB")
    ax2_s11im.set_xlabel("Frequency, GHz")
    ax2_s11im.legend()
    ax2_s11im.grid(which="both")
    ax2_s11im.minorticks_on()

    # plot the S21 results
    ax3_s21 = fig.add_subplot(212)
    # plt.plot(f * 10**-9, S21_Magnitude, "k.", label="Experiment S21 (need to substrack the pd)")
    ax3_s21.plot(f * 10**-9, S21_Magnitude_subpd, "b.", label="Experimental S21")
    # plt.plot(f * 10**-9, pd_Magnitude, "y", label="PD")
    ax3_s21.plot(
        f * 10**-9,
        S21_Fit,
        label=f"Best fit S21 f_r={popt_S21[0]:.2f} GHz, f_p={popt_S21[1]:.2f} GHz, gamma={popt_S21[2]:.2f}, c={popt_S21[3]:.2f}, MSE={S21_Fit_MSE:.2f}, f3dB={f3dB:.2f}",
    )
    ax3_s21.set_ylabel("Magintude(S21), dB")
    ax3_s21.set_xlabel("Frequency, GHz")
    ax3_s21.set_xlim([0, min(freqlimit, 40)])
    ax3_s21.legend()
    ax3_s21.grid(which="both")
    ax3_s21.minorticks_on()
    ax3_s21.axhline(y=c - 3, color="r", linestyle="-")
    ax3_s21.axvline(x=f3dB, color="r", linestyle="-")

    # plt.tight_layout()
    # plt.show()
    if not os.path.exists(directory + "/reports/"):  # make directories
        os.makedirs(directory + "/reports/")
    plt.savefig(directory + "/reports/" + file_name[:-4] + ".png")  # save figure
    plt.close()

    # print and return the results
    print(
        f"R_m={R_m:.2f}\tR_j={R_j:.2f}\tC_p={C_p:.2f}\tC_m={C_m:.2f}\tf_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB:.2f}"
    )
    return R_m, R_j, C_p, C_m, f_r, f_p, gamma, c, f3dB


# one_file_approximation("data", "745.s2p", 2, 50)
