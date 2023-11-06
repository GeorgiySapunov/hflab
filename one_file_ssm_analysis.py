#!/usr/bin/env python3

import os
import skrf as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error

from pysmithplot_fork.smithplot import SmithAxes


def one_file_approximation(
    directory=None,
    report_directory=None,
    freqlimit=40,
    file_name=None,
    probe_port=None,
    waferid=None,
    wavelength=None,
    coordinates=None,
    current=None,
    temperature=25,
    frequency=None,
    s11mag=None,
    s11deg=None,
    s11deg_rad=None,
    s21mag=None,
    s21deg=None,
    S21_MSE_threshold=3,
    fp_fixed=True,
):
    if file_name is not None:
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
        vcsel_df = vcsel_df[
            vcsel_df.index <= freqlimit * 10**9
        ]  # Frequency equal or less then
        vcsel_df["s11_re"] = vcsel_df[f"s {probe_port}{probe_port}"].values.real
        vcsel_df["s11_im"] = vcsel_df[f"s {probe_port}{probe_port}"].values.imag
        vcsel_df["s21_re"] = vcsel_df[f"s {optical_port}{probe_port}"].values.real
        vcsel_df["s21_im"] = vcsel_df[f"s {optical_port}{probe_port}"].values.imag
        f = vcsel_df.index.values
        # fixing index in photodiodes .s2p file
        pd_df.index = pd_df.index.values * 10**9
        # Split the measurements into a real and imaginary part
        S21_Real = vcsel_df["s21_re"]
        S21_Imag = vcsel_df["s21_im"]
        S21_Magnitude = 10 * np.log10(S21_Real**2 + S21_Imag**2)

        # substracting photodiode S21
        pd_df["pd_s21_re"] = pd_df["s 21"].values.real
        pd_df["pd_s21_im"] = pd_df["s 21"].values.imag
        vcsel_df = vcsel_df.join(pd_df[["pd_s21_re", "pd_s21_im"]], how="outer")
        vcsel_df["pd_s21_re"] = vcsel_df["pd_s21_re"].interpolate()
        vcsel_df["pd_s21_im"] = vcsel_df["pd_s21_im"].interpolate()
        vcsel_df = vcsel_df.dropna()
        pd_Real = vcsel_df["pd_s21_re"]
        pd_Imag = vcsel_df["pd_s21_im"]
        pd_Magnitude = 10 * np.log10(pd_Real**2 + pd_Imag**2)
        S21_Magnitude_to_fit = S21_Magnitude - pd_Magnitude
        S11_Real = vcsel_df["s11_re"]
        S11_Imag = vcsel_df["s11_im"]
    else:  # working with automatic system data
        # For DB: let $mag = 10**($a/20), such that:
        # $complex = $mag*cos($b*pi()/180) + $mag*sin($b*pi()/180) j
        f = frequency
        if s11deg:
            s11deg_rad = s11deg * np.pi / 180
        # mag = 10 ** ((s11mag) / 20)  # usually
        mag = 10 ** (((s11mag) / 10) + 3)  # for automatic system
        S11_Real = np.cos(s11deg_rad) * mag
        S11_Imag = np.sin(s11deg_rad) * mag
        S21_Magnitude = s21mag  # usually
        # S21_Magnitude = 20 * ((s21mag / 10) + 3)  # for automatic system
        file_name = f"{waferid}-{wavelength}-{coordinates}-{temperature}°C-{current}mA"

        vcsel_df = pd.DataFrame(S11_Real, index=f)
        vcsel_df = vcsel_df[vcsel_df.index <= freqlimit * 10**9]
        # fixing index in photodiodes .s2p file
        # Substract PD
        photodiode = rf.Network("resources/T3K7V9_DXM30BF_U00162.s2p")
        pd_df = photodiode.to_dataframe("s")
        pd_df.index = pd_df.index.values * 10**9
        # substracting photodiode S21
        pd_df["pd_s21_re"] = pd_df["s 21"].values.real
        pd_df["pd_s21_im"] = pd_df["s 21"].values.imag
        vcsel_df = vcsel_df.join(pd_df[["pd_s21_re", "pd_s21_im"]], how="outer")
        vcsel_df["pd_s21_re"] = vcsel_df["pd_s21_re"].interpolate()
        vcsel_df["pd_s21_im"] = vcsel_df["pd_s21_im"].interpolate()
        vcsel_df = vcsel_df.dropna()
        pd_Real = vcsel_df["pd_s21_re"]
        pd_Imag = vcsel_df["pd_s21_im"]
        pd_Magnitude = 10 * np.log10(pd_Real**2 + pd_Imag**2)
        S21_Magnitude_to_fit = S21_Magnitude - pd_Magnitude

    #  ____  _ _
    # / ___|/ / |
    # \___ \| | |
    #  ___) | | |
    # |____/|_|_|
    # from Hui Li theses p.65
    # DOI:10.3390/app12126035
    def s11_func(f, L, R_p, R_m, R_a, C_p, C_a):
        Zsm = R_m + ((1 / R_a) + 1j * 2 * np.pi * f * C_a * 10**-15) ** -1
        Zt = (
            (1 / (Zsm + 1j * 2 * np.pi * f * L * 10**-12))
            + (R_p + (1 / (1j * 2 * np.pi * f * C_p * 10**-15))) ** -1
        ) ** -1
        # z1 = R_m + 1 / (1 / R_a + 2 * np.pi * 1j * f * C_a * 10**-15)
        # z2 = 1 / (2 * np.pi * f * C_p * 10**-15 * 1j)
        # z = 1 / (1 / z1 + 1 / z2)
        return (Zt - 50) / (Zt + 50)

    # stacking S11 data to fit them simultaneously
    def s11_both_func(f, L, R_p, R_m, R_a, C_p, C_a):
        N = len(f)
        f_real = f[: N // 2]
        f_imag = f[N // 2 :]
        y_real = np.real(s11_func(f_real, L, R_p, R_m, R_a, C_p, C_a))
        y_imag = np.imag(s11_func(f_imag, L, R_p, R_m, R_a, C_p, C_a))
        return np.hstack([y_real, y_imag])

    # Split the measurements into a real and imaginary part
    S11_Both = np.hstack([S11_Real, S11_Imag])

    # Find the best-fit solution
    poptBoth_S11, _ = curve_fit(
        s11_both_func,
        np.hstack([f, f]),
        S11_Both,
        # p0=[150, 150, 30, 150],
        bounds=(0, [1000, 1000, 1000, 1000, 5000, 5000]),
        maxfev=100000,
    )
    L, R_p, R_m, R_a, C_p, C_a = poptBoth_S11

    # Compute the best-fit solution and check the mean squared error
    S11_Fit = s11_func(f, *poptBoth_S11)
    S11_Fit_both = s11_both_func(np.hstack([f, f]), *poptBoth_S11)
    S11_Fit_MSE = mean_squared_error(S11_Both, S11_Fit_both)
    # print(f"S11_Fit_MSE={S11_Fit_MSE}")
    S11_Fit_MSE_real = mean_squared_error(S11_Real, np.real(S11_Fit))
    # print(f"S11_Fit_MSE_real={S11_Fit_MSE_real}")
    S11_Fit_MSE_imag = mean_squared_error(S11_Imag, np.imag(S11_Fit))
    # print(f"S11_Fit_MSE_imag={S11_Fit_MSE_imag}")

    # # equivalent impedance (Hui Li thesis p.66)
    # def Z(s11, z0=50):
    #     Z = z0 * (1 + s11) / (1 - s11)
    #     return Z

    # z = Z(S11_Fit)
    # zmag = np.log10(np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2))

    # # Normalised low pass magnitude
    # # https://electronicbase.net/low-pass-filter-calculator/
    # # https://electronics.stackexchange.com/questions/152159/deriving-2nd-order-passive-low-pass-filter-cutoff-frequency
    # A = 50 * R_m * C_p * C_a * 10**-30
    # B = 50 * C_p * 10**-15 + 50 * C_a * 10**-15 + R_m * C_a * 10**-15
    # w = 2 * np.pi * f
    # H_f = 1 / np.sqrt((w**4) * (A**2) + (w**2) * (B**2 - 2 * A) + 1)
    # H_f_dB = 20 * np.log10(H_f)
    # w_par_3dB = np.sqrt(
    #     1 / A
    #     - (B**2) / (2 * (A**2))
    #     + (np.sqrt(8 * (A**2) - 4 * A * (B**2) + B**4) / (2 * (A**2)))
    # )
    # f_par_Hz = w_par_3dB / (2 * np.pi)
    # f_par_GHz = f_par_Hz * 10**-9  # GHz
    def calc_H_ext(f, L, R_p, R_m, R_a, C_p, C_a):
        Z1 = R_a / (1 + 1j * 2 * np.pi * f * R_a * C_a * 10**-15)
        Z2 = Z1 + R_m + 1j * 2 * f * L * 10**-12
        Z3 = 1 / (1j * 2 * np.pi * f * C_p * 10**-15) + R_p
        Z4 = ((1 / Z2) + (1 / Z3)) ** -1
        H_ext = (Z1 * Z4 * 50) / (Z2 * (Z4 + 50) * R_a)
        return H_ext

    def calc_H_ext0(R_p, R_m, R_a):
        Z1 = R_a
        Z2 = Z1 + R_m
        # Z3 = np.inf
        Z4 = (1 / Z2) ** -1
        H_ext = (Z1 * Z4 * 50) / (Z2 * (Z4 + 50) * R_a)
        return H_ext

    f_h = np.linspace(0.01, 60 * 10**9, 6000)
    H_ext = calc_H_ext(f_h, L, R_p, R_m, R_a, C_p, C_a)
    H2_ext = np.abs(H_ext) ** 2
    H2_f_dB = 10 * np.log10(H2_ext)

    H_ext0 = calc_H_ext0(R_p, R_m, R_a)
    H2_ext0 = np.abs(H_ext0) ** 2
    H2_f_dB0 = 10 * np.log10(H2_ext0)

    f_par_Hz = f_h[np.where(H2_ext / H2_ext0 < 0.5)[0][0]]
    f_p2 = f_par_Hz * 10**-9

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
        h2 = c + 10 * np.log10(
            f_r**4
            / (
                ((f_r**2 - f**2) ** 2 + (f * gamma / (2 * np.pi)) ** 2)
                * (1 + (f / f_p) ** 2)
            )
        )
        return h2

    S21_Fit_MSE = np.inf
    s21_freqlimit = freqlimit
    new_f = np.array(f)
    new_S21_Magnitude_to_fit = np.array(S21_Magnitude_to_fit)
    while S21_Fit_MSE > S21_MSE_threshold and s21_freqlimit >= 5:
        new_S21_Magnitude_to_fit = new_S21_Magnitude_to_fit[
            new_f <= s21_freqlimit * 10**9
        ]
        new_f = new_f[new_f <= s21_freqlimit * 10**9]

        # Find the best-fit solution for S21
        # f_r, f_p, gamma, c
        popt_S21, pcovBoth = curve_fit(
            s21_func,
            new_f,
            new_S21_Magnitude_to_fit,
            # p0=[150, 150, 30, 150, 0],
            bounds=([0, 0, 0, -np.inf], [81, 81, 2001, np.inf]),
            maxfev=100000,
        )
        f_r, f_p, gamma, c = popt_S21

        # Compute the best-fit solution and mean squered error for S21
        S21_Fit = s21_func(new_f, *popt_S21)
        S21_Fit_MSE = mean_squared_error(new_S21_Magnitude_to_fit, S21_Fit)
        # print(f"S21_Fit_MSE={S21_Fit_MSE}")
        check_f3dB = np.where(S21_Fit < c - 3)[0]
        if check_f3dB.any():
            f3dB = f[np.where(S21_Fit < c - 3)[0][0]] * 10**-9  # GHz
        else:
            f3dB = None
        if S21_Fit_MSE > S21_MSE_threshold:
            print(f""" s21_freqlimit={s21_freqlimit} GHz """)
            # print(
            #     f"""
            # Trying to fit S21, MSE is too large.
            # s21_freqlimit={s21_freqlimit} GHz
            # MSE={S21_Fit_MSE:.2f}
            # f_r={popt_S21[0]:.2f} GHz, f_p={popt_S21[1]:.2f} GHz, ɣ={popt_S21[2]:.2f}, c={popt_S21[3]:.2f}, f3dB={f3dB:.2f}
            # """
            # )
        if S21_Fit_MSE > S21_MSE_threshold and s21_freqlimit >= 5:
            s21_freqlimit -= 1

    if S21_Fit_MSE > S21_MSE_threshold:
        print(f"\n                    Failed to fit S21\n")
    # if S21_Fit_MSE < 5:
    #     f_r, f_p, gamma, c, f3dB = None, None, None, None, None

    # fit with fp fixed at f_par_Hz
    if fp_fixed:

        def s21_func2(f, f_r, gamma, c):
            f_p = f_par_Hz
            f = f * 10**-9
            f_r = f_r
            f_p = f_p
            h2 = c + 10 * np.log10(
                f_r**4
                / (
                    ((f_r**2 - f**2) ** 2 + (f * gamma / (2 * np.pi)) ** 2)
                    * (1 + (f / f_p) ** 2)
                )
            )
            return h2

        S21_Fit_MSE2 = np.inf
        s21_freqlimit2 = freqlimit
        new_f2 = np.array(f)
        new_S21_Magnitude_to_fit2 = np.array(S21_Magnitude_to_fit)
        while S21_Fit_MSE2 > S21_MSE_threshold and s21_freqlimit2 >= 5:
            new_S21_Magnitude_to_fit2 = new_S21_Magnitude_to_fit2[
                new_f2 <= s21_freqlimit2 * 10**9
            ]
            new_f2 = new_f2[new_f2 <= s21_freqlimit2 * 10**9]

            # Find the best-fit solution for S21
            # f_r, gamma, c
            popt_S21_2, pcovBoth_2 = curve_fit(
                s21_func2,
                new_f2,
                new_S21_Magnitude_to_fit2,
                # p0=[150, 150, 30, 150, 0],
                bounds=([0, 0, -np.inf], [81, 2001, np.inf]),
                maxfev=100000,
            )
            f_r2, gamma2, c2 = popt_S21_2

            # Compute the best-fit solution and mean squered error for S21
            S21_Fit2 = s21_func2(new_f2, *popt_S21_2)
            S21_Fit_MSE2 = mean_squared_error(new_S21_Magnitude_to_fit2, S21_Fit2)
            # print(f"S21_Fit_MSE={S21_Fit_MSE}")
            check_f3dB2 = np.where(S21_Fit2 < c2 - 3)[0]
            if check_f3dB2.any():
                f3dB2 = f[np.where(S21_Fit2 < c2 - 3)[0][0]] * 10**-9  # GHz
            else:
                f3dB2 = None
            if S21_Fit_MSE2 > S21_MSE_threshold:
                print(
                    f"""f_par fixed at {f_p2:.2f} GHz, s21_freqlimit={s21_freqlimit2} GHz """
                )
                # print(
                #     f"""
                # Trying to fit S21, MSE is too large.
                # s21_freqlimit={s21_freqlimit} GHz
                # MSE={S21_Fit_MSE:.2f}
                # f_r={popt_S21[0]:.2f} GHz, f_p={popt_S21[1]:.2f} GHz, ɣ={popt_S21[2]:.2f}, c={popt_S21[3]:.2f}, f3dB={f3dB:.2f}
                # """
                # )
            if S21_Fit_MSE2 > S21_MSE_threshold and s21_freqlimit2 >= 5:
                s21_freqlimit2 -= 1

        if S21_Fit_MSE2 > S21_MSE_threshold:
            print(
                f"\n                    Failed to fit S21 with f_par fixed at {f_p2:.2f} GHz\n"
            )
        # if S21_Fit_MSE2 < 5:
        #     f_r2, gamma2, c2, f3dB2 = None, None, None, None
    else:
        f_r2, gamma2, c2, f3dB2 = None, None, None, None

    #  _____ _
    # |  ___(_) __ _ _   _ _ __ ___  ___
    # | |_  | |/ _` | | | | '__/ _ \/ __|
    # |  _| | | (_| | |_| | | |  __/\__ \
    # |_|   |_|\__, |\__,_|_|  \___||___/
    #          |___/
    fig = plt.figure(figsize=(2.5 * 11.69, 2.5 * 8.27))

    fig.suptitle(file_name)

    ax1_s11re = fig.add_subplot(321)
    ax1_s11re.plot(f * 10**-9, S11_Real, "k.", label="Experiment S11 Real", alpha=0.6)
    ax1_s11re.plot(
        f * 10**-9,
        np.real(S11_Fit),
        "b-.",
        label=f"""Best fit S11
        Real L={L:.2f} pH, R_p={R_p:.2f} Om, R_m={R_m:.2f} Om, R_a={R_a:.2f} Om, C_p={C_p:.2f} fF, C_a={C_a:.2f} fF
        MSE*10^4={S11_Fit_MSE*10**4:.2f}, MSE_real*10^4={S11_Fit_MSE_real*10**4:.2f}, fit to {freqlimit} GHz""",
        alpha=1,
    )
    ax1_s11re.set_ylabel("Re(S11)")
    ax1_s11re.set_xlabel("Frequency, GHz")
    ax1_s11re.legend()
    ax1_s11re.grid(which="both")
    ax1_s11re.minorticks_on()
    ax1_s11re.set_ylim([-1, 1])

    ax2_s11im = fig.add_subplot(322)
    ax2_s11im.plot(
        f * 10**-9,
        np.real(S11_Imag),
        "k.",
        label="Experiment S11 Imaginary",
        alpha=0.6,
    )
    ax2_s11im.plot(
        f * 10**-9,
        np.imag(S11_Fit),
        "b-.",
        label=f"""Best fit S11
        Imaginary L={L:.2f} pH, R_p={R_p:.2f} Om, R_m={R_m:.2f} Om, R_a={R_a:.2f} Om, C_p={C_p:.2f} fF, C_a={C_a:.2f} fF
        MSE*10^4={S11_Fit_MSE*10**4:.2f}, MSE_real*10^4={S11_Fit_MSE_real*10**4:.2f}, fit to {freqlimit} GHz""",
        alpha=1,
    )
    ax2_s11im.set_ylabel("Im(S11)")
    ax2_s11im.set_xlabel("Frequency, GHz")
    ax2_s11im.legend()
    ax2_s11im.grid(which="both")
    ax2_s11im.minorticks_on()
    ax2_s11im.set_ylim([-1, 1])

    # plot the S11 Smith Chart
    ax3_s11_smith = fig.add_subplot(
        323,
        projection="smith",
    )
    # ax3_s11_smith.update_scParams(axes_impedance=50)
    ax3_s11_smith.plot(
        S11_Real,
        np.real(S11_Imag),
        "k",
        label="S11 measured",
        alpha=0.6,
    )
    ax3_s11_smith.plot(
        np.real(S11_Fit), np.imag(S11_Fit), "b", label="S11 fit", alpha=0.6
    )
    ax3_s11_smith.set_title("S11 Smith chart")
    # ax3_s11_smith.set_ylabel("Im(S11)")
    # ax3_s11_smith.set_xlabel("Re(S11)")
    # ax3_s11_smith.legend()
    # ax3_s11_smith.grid(which="both")
    # ax3_s11_smith.minorticks_on()
    # ax3_s11_smith.set_ylim([-1, 1])
    # ax3_s11_smith.set_xlim([-1, 1])

    # H^2(f)
    ax4_h2 = fig.add_subplot(324)
    ax4_h2.plot(
        f_h * 10**-9,
        H2_f_dB - H2_f_dB0,
        "k",
        label="|H(f)|^2-|H(0)|^2, dB",
        alpha=0.6,
    )
    # ax6_vout.plot(f * 10**-9, H2_f_dB, "r", label="H^2(f)", alpha=0.6)

    ax4_h2.set_title("Equivalent circuit model |H(f)|^2-|H(0)|^2")
    ax4_h2.set_ylabel("|H(f)|^2-|H(0)|^2, dB")
    ax4_h2.set_xlabel("Frequency, GHz")
    ax4_h2.set_xlim([0, min(freqlimit, 40)])
    ax4_h2.set_ylim([-10, 1])
    ax4_h2.grid(which="both")
    ax4_h2.minorticks_on()
    ax4_h2.axhline(y=-3, color="y", linestyle="-.")
    # ax4_vout.axvline(
    #     x=f3dB_vout, color="y", linestyle="-.", label=f"f3dB_Vout={f3dB_vout:.2f}"
    # )
    ax4_h2.axvline(
        x=f_p2,
        color="r",
        linestyle=":",
        label=f"f3dB_parasitic={f_p2:.2f} GHz",
    )
    ax4_h2.legend()

    # plot the S21 results
    ax6_s21 = fig.add_subplot(326)
    # plt.plot(f * 10**-9, S21_Magnitude, "k.", label="Experiment S21 (need to substrack the pd)")
    ax6_s21.plot(
        f * 10**-9, S21_Magnitude_to_fit - c, "k", label="Experimental S21", alpha=0.6
    )
    # plt.plot(f * 10**-9, pd_Magnitude, "y", label="PD")
    ax6_s21.plot(
        new_f * 10**-9,
        S21_Fit - c,
        "b-.",
        label=f"""Best fit S21
        f_r={f_r:.2f} GHz, f_p={f_p:.2f} GHz, ɣ={gamma:.2f}, c={c:.2f}
        MSE={S21_Fit_MSE:.2f}, fit to {s21_freqlimit} GHz""",
        alpha=1,
    )
    if fp_fixed:
        ax6_s21.plot(
            new_f2 * 10**-9,
            S21_Fit2 - c2,
            "m:",
            label=f"""S21 fit with f_p_2={f_p2:.2f} GHz
            f_r={f_r2:.2f} GHz, ɣ={gamma2:.2f}, c={c2:.2f}
            MSE={S21_Fit_MSE2:.2f}, fit to {s21_freqlimit2} GHz""",
            alpha=1,
        )
    ax6_s21.set_title("S21")
    ax6_s21.set_ylabel("Magintude(S21), dB")
    ax6_s21.set_xlabel("Frequency, GHz")
    ax6_s21.set_xlim([0, min(freqlimit, 40)])
    ax6_s21.set_ylim([-40, 10])
    ax6_s21.grid(which="both")
    ax6_s21.minorticks_on()
    ax6_s21.axhline(y=-3, color="y", linestyle="-.")
    if f3dB:
        ax6_s21.axvline(x=f3dB, color="y", linestyle="-.", label=f"f3dB={f3dB:.2f} GHz")
    if f3dB2:
        ax6_s21.axvline(
            x=f3dB2, color="y", linestyle=":", label=f"f3dB2={f3dB2:.2f} GHz"
        )
    ax6_s21.axvline(x=f_p, color="r", linestyle="-.", label=f"f_p={f_p:.2f} GHz")
    ax6_s21.axvline(x=f_r, color="g", linestyle="-.", label=f"f_r={f_r:.2f} GHz")
    if fp_fixed:
        ax6_s21.axvline(x=f_p2, color="r", linestyle=":", label=f"f_p_2={f_p2:.2f} GHz")
        ax6_s21.axvline(
            x=f_r2,
            color="g",
            linestyle=":",
            label=f"f_r2={f_r2:.2f} GHz",
        )
    ax6_s21.legend(fontsize=10)

    # plt.tight_layout()
    # plt.show()
    if not os.path.exists(report_directory):  # make directories
        os.makedirs(report_directory)
    plt.savefig(
        report_directory + file_name.removesuffix(".s2p") + ".png"
    )  # save figure
    plt.close()

    # print and return the results
    if f3dB:
        print(
            f""" L={L:.2f} pH, R_p={R_p:.2f} Om, R_m={R_m:.2f}\tR_a={R_a:.2f}\tC_p={C_p:.2f}\tC_a={C_a:.2f}
            f_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB:.2f}
            MSE_S21={S21_Fit_MSE:.2f}, MSE_S11*10^4={S11_Fit_MSE*10**4:.2f}, MSE_imag*10^4={S11_Fit_MSE_imag*10**4:.2f}, MSE_real*10^4={S11_Fit_MSE_real*10**4:.2f}, fit to {freqlimit} GHz """
        )
    else:
        print(
            f""" L={L:.2f} pH, R_p={R_p:.2f} Om, R_m={R_m:.2f}\tR_a={R_a:.2f}\tC_p={C_p:.2f}\tC_a={C_a:.2f}
            f_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB}
            MSE_S21={S21_Fit_MSE:.2f}, MSE_S11*10^4={S11_Fit_MSE*10**4:.2f}, MSE_imag*10^4={S11_Fit_MSE_imag*10**4:.2f}, MSE_real*10^4={S11_Fit_MSE_real*10**4:.2f}, fit to {freqlimit} GHz """
        )
    if f_r > 80 or f_p > 80 or gamma > 2000:
        f_r, f_p, c, gamma, f3dB = None, None, None, None, None
        print(
            f"""
            f_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB}
            *Changed to None*
            """
        )
    if f_r2 > 80 or f_p2 > 80 or gamma2 > 2000:
        f_r2, f_p2, c2, gamma2, f3dB2 = None, None, None, None, None
        print(
            f"""
            f_r2={f_r:.2f}\tf_p2={f_p2:.2f}\tgamma2={gamma2:.2f}\tc2={c2:.2f}\tf3dB2={f3dB2}
            *Changed to None*
            """
        )
    return (
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
    )


# one_file_approximation("data", "745.s2p", 2, 50)
