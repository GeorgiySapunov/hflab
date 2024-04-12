#!/usr/bin/env python3

import os
import skrf as rf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from sklearn.metrics import mean_squared_error
from pathlib import Path
from termcolor import colored


def one_file_approximation(
    directory=None,
    report_directory=None,
    title=None,
    freqlimit=40,  # GHz
    file_path=None,
    probe_port=None,
    waferid=None,
    wavelength=None,
    coordinates=None,
    current=None,
    temperature=25,
    frequency=None,  # Hz
    s11re=None,
    s11im=None,
    s21mag=None,
    S21_MSE_threshold=3,
    fp_fixed=True,
    L_bounds=[0, np.inf],
    R_p_high_bounds=[0, np.inf],
    R_m_bounds=[0, np.inf],
    R_a_bounds=[0, np.inf],
    C_p_low_bounds=[0, np.inf],
    C_a_bounds=[0, np.inf],
    f1_bounds=[0, np.inf],
    f2_bounds=[0, np.inf],
    f3_bounds=[0, np.inf],
):
    if file_path is not None:
        if probe_port == 1:
            optical_port = 2
        elif probe_port == 2:
            optical_port = 1
        else:
            raise Exception("probe_port is unclear")

        if isinstance(report_directory, str):
            report_directory = Path(report_directory)
        report_directory.mkdir(exist_ok=True)

        file_name = file_path.stem
        vcsel_ntwk = rf.Network(file_path)
        # ntwk = rf.pna_csv_2_ntwks3(
        #     "745.csv"
        # )  # Read a CSV file exported from an Agilent PNA in dB/deg format
        vcsel_df = vcsel_ntwk.to_dataframe("s")
        vcsel_df = vcsel_df[
            vcsel_df.index <= freqlimit * 10**9
        ]  # Frequency equal or less then
        vcsel_df["s11_re"] = vcsel_df[f"s {probe_port}{probe_port}"].values.real
        vcsel_df["s11_im"] = vcsel_df[f"s {probe_port}{probe_port}"].values.imag
        vcsel_df["s21_re"] = vcsel_df[f"s {optical_port}{probe_port}"].values.real
        vcsel_df["s21_im"] = vcsel_df[f"s {optical_port}{probe_port}"].values.imag
        f = vcsel_df.index.values  # Hz
        # Split the measurements into a real and imaginary part
        S21_Magnitude = 10 * np.log10(S21_Real**2 + S21_Imag**2)

    else:  # working with automatic system data
        # For DB: let $mag = 10**($a/20), such that:
        # $complex = $mag*cos($b*pi()/180) + $mag*sin($b*pi()/180) j

        # TODO delete this line after fixing the automatic system!!!!!!!
        frequency = pd.read_csv("resources/801point_10MHz-40GHz.csv")["Frequency, Hz"]
        #

        f = pd.Series(frequency)  # Hz
        file_name = f"{waferid}-{wavelength}-{coordinates}-{temperature}°C-{current}mA"
        sdict = {"s11_re": s11re, "s11_im": s11im, "s21_mag": s21mag}
        # s11 = np.array(s11re) + 1j * np.array(s11im)
        # s21 = np.array(s21re) + 1j * np.array(s11im)
        # S = np.zeros((len(f), 2, 2), dtype=complex)
        # S[:, 0, 0] = s11
        # S[:, 1, 0] = s21
        # # S[:,0,1] = S12
        # # S[:,1,1] = S22
        # vcsel_ntwk = rf.Network(s=S, f=f, f_unit="Hz")

        vcsel_df = pd.DataFrame(sdict, index=f)
        vcsel_df = vcsel_df[vcsel_df.index <= freqlimit * 10**9]
        S11_Real = vcsel_df["s11_re"].values
        S11_Imag = vcsel_df["s11_im"].values
        S21_Magnitude = vcsel_df["s21_mag"].values

    # Subtract PD
    photodiode = rf.Network("resources/T3K7V9_DXM30BF_U00162.s2p")
    pd_df = photodiode.to_dataframe("s")

    # subtracting photodiode S21
    pd_df["pd_s21_re"] = pd_df["s 21"].values.real
    pd_df["pd_s21_im"] = pd_df["s 21"].values.imag
    pd_df["pd_s21_logmag"] = 10 * np.log10(
        pd_df["pd_s21_re"] ** 2 + pd_df["pd_s21_im"] ** 2
    )
    vcsel_df = vcsel_df.join(pd_df[["pd_s21_logmag"]], how="outer")
    vcsel_df["pd_s21_logmag"] = vcsel_df["pd_s21_logmag"].interpolate()
    vcsel_df = vcsel_df.dropna()
    pd_Magnitude = vcsel_df["pd_s21_logmag"].values
    S21_Magnitude_to_fit = S21_Magnitude - pd_Magnitude
    S11_Real = vcsel_df["s11_re"].values
    S11_Imag = vcsel_df["s11_im"].values

    #  ____  _ _
    # / ___|/ / |
    # \___ \| | |
    #  ___) | | |
    # |____/|_|_|
    # from Hui Li theses p.65
    # DOI:10.3390/app12126035
    def s11_func(f, L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3):
        f1 = f1 * 10**9
        f2 = f2 * 10**9
        f3 = f3 * 10**9
        C_p = C_p_low * np.exp(-((f**2) / (f1**2)))
        R_p = R_p_high * np.exp(-(((f - f2) ** 2) / (f3**2)))

        Zsm = R_m + ((1 / R_a) + 1j * 2 * np.pi * f * C_a * 10**-15) ** -1
        Z3C_p = 1 / (1j * 2 * np.pi * f * C_p * 10**-15)
        Z3 = Z3C_p + R_p
        Zt = ((1 / (Zsm + 1j * 2 * np.pi * f * L * 10**-12)) + Z3**-1) ** -1
        # z1 = R_m + 1 / (1 / R_a + 2 * np.pi * 1j * f * C_a * 10**-15)
        # z2 = 1 / (2 * np.pi * f * C_p * 10**-15 * 1j)
        # z = 1 / (1 / z1 + 1 / z2)
        return (Zt - 50) / (Zt + 50)

    # stacking S11 data to fit them simultaneously
    def s11_both_func(f, L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3):
        f1 = f1 * 10**9
        f2 = f2 * 10**9
        f3 = f3 * 10**9
        N = len(f)
        f_real = f[: N // 2]
        f_imag = f[N // 2 :]
        y_real = np.real(
            s11_func(f_real, L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3)
        )
        y_imag = np.imag(
            s11_func(f_imag, L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3)
        )
        return np.hstack([y_real, y_imag])

    # Split the measurements into a real and imaginary part
    S11_Both = np.hstack([S11_Real, S11_Imag])

    # Find the best-fit solution
    poptBoth_S11, _ = curve_fit(
        s11_both_func,
        np.hstack([f, f]),
        S11_Both,
        # p0=[150, 150, 30, 150],
        bounds=(
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
        ),
        maxfev=100000,
    )
    L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3 = poptBoth_S11

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
        z = z0 * (1 + s11) / (1 - s11)
        return z

    z_Fit = Z(S11_Fit)
    S11 = np.asarray(S11_Real) + (1j * np.asarray(S11_Imag))
    z = Z(S11)
    # zmag = np.log10(np.sqrt(np.real(z) ** 2 + np.imag(z) ** 2))

    def calc_H_ext(f, L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3):
        f1 = f1 * 10**9
        f2 = f2 * 10**9
        f3 = f3 * 10**9
        C_p = C_p_low * np.exp(-((f**2) / (f1**2)))
        R_p = R_p_high * np.exp(-(((f - f2) ** 2) / (f3**2)))
        Z1 = R_a / (1 + 1j * 2 * np.pi * f * R_a * C_a * 10**-15)
        Z2 = Z1 + R_m + 1j * 2 * f * L * 10**-12
        Z3 = 1 / (1j * 2 * np.pi * f * C_p * 10**-15) + R_p
        Z4 = ((1 / Z2) + (1 / Z3)) ** -1
        H_ext = (Z1 * Z4 * 50) / (Z2 * (Z4 + 50) * R_a)
        return H_ext

    def calc_H_ext0(R_p_high, R_m, R_a, f2, f3):
        f2 = f2 * 10**9
        f3 = f3 * 10**9
        R_p = R_p_high * np.exp(-(((f - f2) ** 2) / (f3**2)))
        Z1 = R_a
        Z2 = Z1 + R_m
        # Z3 = np.inf
        Z4 = (1 / Z2) ** -1
        H_ext = (Z1 * Z4 * 50) / (Z2 * (Z4 + 50) * R_a)
        return H_ext

    f_h = np.linspace(0.01, 60 * 10**9, 6000)
    H_ext = calc_H_ext(f_h, L, R_p_high, R_m, R_a, C_p_low, C_a, f1, f2, f3)
    H2_ext = np.abs(H_ext) ** 2
    H2_f_dB = 10 * np.log10(H2_ext)

    H_ext0 = calc_H_ext0(R_p_high, R_m, R_a, f2, f3)
    H2_ext0 = np.abs(H_ext0) ** 2
    H2_f_dB0 = 10 * np.log10(H2_ext0)

    if np.any(
        np.where((H2_ext / H2_ext0) < 0.5)
    ):  # TODO fix it! turn on and off S11 fitting?
        f_par_Hz = f_h[np.where((H2_ext / H2_ext0) < 0.5)[0][0]]
        f_p2 = f_par_Hz * 10**-9
    else:
        # f_par_Hz = 10 * 10**9
        # f_p2 = f_par_Hz * 10**-9
        f_par_Hz, f_p2 = None, None
        print(
            colored(
                "WARNING!: np.where(H2_ext / H2_ext0 < 0.5) == False\tf_parasitic set to None!",
                "yellow",
            )
        )

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

        # Compute the best-fit solution and mean squared error for S21
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
        if S21_Fit_MSE > S21_MSE_threshold and s21_freqlimit >= 5:
            s21_freqlimit -= 1

    if S21_Fit_MSE > S21_MSE_threshold:
        print(colored(f"\n\tFailed to fit S21\n", "red"))
    # if S21_Fit_MSE < 5:
    #     f_r, f_p, gamma, c, f3dB = None, None, None, None, None

    # fit with fp fixed at f_par_Hz
    if fp_fixed and f_par_Hz:

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

            # Compute the best-fit solution and mean squared error for S21
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
            if S21_Fit_MSE2 > S21_MSE_threshold and s21_freqlimit2 >= 5:
                s21_freqlimit2 -= 1

        if S21_Fit_MSE2 > S21_MSE_threshold:
            print(
                colored(
                    f"\n\tFailed to fit S21 with f_par fixed at {f_p2:.2f} GHz\n", "red"
                )
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

    if title:
        fig.suptitle(title, fontsize=40)
    else:
        fig.suptitle(file_name, fontsize=40)

    ax1_s11re = fig.add_subplot(421)
    ax1_s11re.plot(f * 10**-9, S11_Real, "k.", label="Experiment S11 Real", alpha=0.6)
    ax1_s11re.plot(
        f * 10**-9,
        np.real(S11_Fit),
        "b-.",
        label=f"""Best fit S11
        L={L:.2f} pH, R_m={R_m:.2f} Om, R_a={R_a:.2f} Om, C_a={C_a:.2f} fF,
        C_p_low={C_p_low:.2f} fF, f1={f1:.2f}GHz,
        R_p_high={R_p_high:.2f} Om, f2={f2:.2f}GHz, f3={f3:.2f} GHz,
        MSE={S11_Fit_MSE:.6f}, MSE_real={S11_Fit_MSE_real:.6f}, fit to {freqlimit} GHz""",
        alpha=1,
    )
    ax1_s11re.set_ylabel("Re(S11)")
    ax1_s11re.set_xlabel("Frequency, GHz")
    ax1_s11re.legend()
    ax1_s11re.grid(which="both")
    ax1_s11re.minorticks_on()
    ax1_s11re.set_ylim([-1, 1])

    ax2_s11im = fig.add_subplot(422)
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
        L={L:.2f} pH, R_m={R_m:.2f} Om, R_a={R_a:.2f} Om, C_a={C_a:.2f} fF,
        C_p_low={C_p_low:.2f} fF, f1={f1:.2f}GHz,
        R_p_high={R_p_high:.2f} Om, f2={f2:.2f}GHz, f3={f3:.2f} GHz,
        MSE={S11_Fit_MSE:.6f}, MSE_real={S11_Fit_MSE_real:.6f}, fit to {freqlimit} GHz""",
        alpha=1,
    )
    ax2_s11im.set_ylabel("Im(S11)")
    ax2_s11im.set_xlabel("Frequency, GHz")
    ax2_s11im.legend()
    ax2_s11im.grid(which="both")
    ax2_s11im.minorticks_on()
    ax2_s11im.set_ylim([-1, 1])

    # # plot the S11 Smith Chart
    # ax3_s11_smith = fig.add_subplot(
    #     423,
    #     projection="smith",
    # )
    # # SmithAxes.update_scParams(axes_impedance=50)
    # ax3_s11_smith.plot(
    #     S11_Real,
    #     np.real(S11_Imag),
    #     "k",
    #     label="S11 measured",
    #     alpha=0.6,
    # )
    # ax3_s11_smith.plot(
    #     np.real(S11_Fit), np.imag(S11_Fit), "b", label="S11 fit", alpha=0.6
    # )
    # ax3_s11_smith.set_title("S11 Smith chart")

    # plot the S11 Smith Chart
    ax3_s11_smith = fig.add_subplot(423)
    S11_complex = np.array(S11_Real) + 1j * np.array(S11_Imag)
    rf.plotting.plot_smith(
        s=S11_complex,
        label="S11 experiment",
        color="k",
        alpha=0.6,
        title="S11 Smith Chart",
        ax=ax3_s11_smith,
    )
    rf.plotting.plot_smith(
        s=S11_Fit,
        label="S11 fit",
        color="b",
        linestyle="-.",
        title="S11 Smith Chart",
        ax=ax3_s11_smith,
    )

    # H^2(f)
    ax4_h2 = fig.add_subplot(424)
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
    if f_p2:
        ax4_h2.axvline(
            x=f_p2,
            color="r",
            linestyle=":",
            label=f"f3dB_parasitic={f_p2:.2f} GHz",
        )
    ax4_h2.legend()

    # plot the Z Chart
    impedance_argmin = np.abs(z).argmin()
    impedance_argmax = np.abs(z).argmax()
    impedance_min = z[impedance_argmin]
    impedance_max = z[impedance_argmax]
    impedance_min_f = f[impedance_argmin] * 10**-9
    impedance_max_f = f[impedance_argmax] * 10**-9
    ax5_z_real = fig.add_subplot(425)
    ax5_z_imag = ax5_z_real.twinx()
    ax5_z_real.set_title("Impedance")
    lns1 = ax5_z_real.plot(
        f * 10**-9,
        np.real(z),
        "k.",
        label=f"Resistance, Ohm\nimpedance min {impedance_min:3.2f} Ohm at {impedance_min_f:3.2f} GHz\nimpedance max {impedance_max:3.2f} Ohm at {impedance_max_f:3.2f} GHz",
        alpha=0.6,
    )
    lns2 = ax5_z_imag.plot(
        f * 10**-9,
        np.imag(z),
        "y.",
        label=f"Reactance, Ohm\nimpedance min {impedance_min:3.2f} Ohm at {impedance_min_f:3.2f} GHz\nimpedance max {impedance_max:3.2f} Ohm at {impedance_max_f:3.2f} GHz",
        alpha=0.6,
    )
    lns3 = ax5_z_real.plot(
        f * 10**-9,
        np.real(z_Fit),
        "b-.",
        label="Resistance Fit, Ohm",
        alpha=1,
    )
    lns4 = ax5_z_imag.plot(
        f * 10**-9,
        np.imag(z_Fit),
        "r-.",
        label="Reactance Fit, Ohm",
        alpha=1,
    )
    ax5_z_real.set_ylabel("Resistance, Ohm", color="blue")
    ax5_z_imag.set_ylabel("Reactance, Ohm", color="red")
    ax5_z_real.set_xlabel("Frequency, GHz")
    lns = lns1 + lns2 + lns3 + lns4
    labs = [l.get_label() for l in lns]
    ax5_z_real.legend(lns, labs, loc="center right")
    ax5_z_real.grid(which="both")
    ax5_z_real.minorticks_on()
    ax5_z_real.set_ylim([0, 1000])
    ax5_z_imag.set_ylim([-900, 100])

    # plot the S21 results
    ax6_s21 = fig.add_subplot(426)
    # plt.plot(f * 10**-9, S21_Magnitude, "k.", label="Experiment S21 (need to substrack the pd)")
    ax6_s21.plot(
        f * 10**-9, S21_Magnitude_to_fit - c, "k", label="Experimental S21", alpha=0.6
    )
    # plt.plot(f * 10**-9, pd_Magnitude, "y", label="PD")
    ax6_s21.plot(
        new_f * 10**-9,
        S21_Fit - c,
        "b-.",
        label=f"Best fit S21\nf_r={f_r:.2f} GHz, f_p={f_p:.2f} GHz, ɣ={gamma:.2f}, c={c:.2f}\nMSE={S21_Fit_MSE:.2f}, fit to {s21_freqlimit} GHz",
        alpha=1,
    )
    if fp_fixed and f_par_Hz:
        ax6_s21.plot(
            new_f2 * 10**-9,
            S21_Fit2 - c2,
            "m:",
            label=f"S21 fit with f_p_2={f_p2:.2f} GHz\nf_r={f_r2:.2f} GHz, ɣ={gamma2:.2f}, c={c2:.2f}\nMSE={S21_Fit_MSE2:.2f}, fit to {s21_freqlimit2} GHz",
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
    if fp_fixed and f_par_Hz:
        ax6_s21.axvline(x=f_p2, color="r", linestyle=":", label=f"f_p_2={f_p2:.2f} GHz")
        ax6_s21.axvline(
            x=f_r2,
            color="g",
            linestyle=":",
            label=f"f_r2={f_r2:.2f} GHz",
        )
    ax6_s21.legend(fontsize=10)

    C_p = C_p_low * np.exp(-((f**2) / ((f1 * 10**9) ** 2)))
    R_p = R_p_high * np.exp(-(((f - (f2 * 10**9)) ** 2) / ((f3 * 10**9) ** 2)))
    ax7_C_p = fig.add_subplot(427)
    ax7_R_p = ax7_C_p.twinx()
    ax7_C_p.set_title("Pad capacitance and dielectric losses")
    lns1 = ax7_C_p.plot(
        f * 10**-9,
        C_p,
        "r.",
        label=f"Pad capacitance C_p, fF\nC_p_low={C_p_low:3.2f} fF",
        alpha=1,
    )
    lns2 = ax7_R_p.plot(
        f * 10**-9,
        R_p,
        "g.",
        label=f"Pad dielectric losses R_p, Ohm\nR_p_high={R_p_high:3.2f} Ohm\nR_p_max={R_p.max():3.2f} Ohm at {(f*10**-9)[R_p.argmax()]} GHz",
        alpha=1,
    )
    ax7_C_p.set_ylabel("C_p, fF", color="r")
    ax7_R_p.set_ylabel("R_p, Ohm", color="green")
    ax7_C_p.set_xlabel("Frequency, GHz")
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax7_C_p.legend(lns, labs, loc="center left")
    ax7_C_p.grid(which="both")
    ax7_C_p.minorticks_on()
    # ax7_C_p.set_ylim(bottom=0)
    # ax7_R_p.set_ylim(bottom=0)

    if report_directory:
        plt.savefig(
            report_directory / (file_name.removesuffix(".s2p") + ".png"), dpi=200
        )  # save figure
    plt.close()

    # print and return the results
    if f3dB:
        print(
            f"""
            L={L:.2f} pH, R_m={R_m:.2f} Om, R_a={R_a:.2f} Om, C_a={C_a:.2f} fF,
            C_p_low={C_p_low:.2f} fF, f1={f1:.2f}GHz,
            R_p_high={R_p_high:.2f} Om, f2={f2:.2f}GHz, f3={f3:.2f} GHz,
            f_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB:.2f}
            MSE_S21={S21_Fit_MSE:.2f}, MSE_S11={S11_Fit_MSE:.6f}, MSE_imag={S11_Fit_MSE_imag:.6f}, MSE_real={S11_Fit_MSE_real:.6f}, fit to {freqlimit} GHz
            """
        )
    else:
        print(
            f"""
            L={L:.2f} pH, R_m={R_m:.2f} Om, R_a={R_a:.2f} Om, C_a={C_a:.2f} fF,
            C_p_low={C_p_low:.2f} fF, f1={f1:.2f}GHz,
            R_p_high={R_p_high:.2f} Om, f2={f2:.2f}GHz, f3={f3:.2f} GHz,
            f_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB}
            MSE_S21={S21_Fit_MSE:.2f}, MSE_S11={S11_Fit_MSE:.6f}, MSE_imag={S11_Fit_MSE_imag:.6f}, MSE_real={S11_Fit_MSE_real:.6f}, fit to {freqlimit} GHz
            """
        )
    if f_r > 80 or f_p > 80 or gamma > 2000:
        print(
            f"f_r={f_r:.2f}\tf_p={f_p:.2f}\tgamma={gamma:.2f}\tc={c:.2f}\tf3dB={f3dB}\n*Changed to None*"
        )
        f_r, f_p, gamma, f3dB = None, None, None, None
    if fp_fixed and f_par_Hz:
        if f_r2 > 80 or f_p2 > 80 or gamma2 > 2000:
            print(
                f"f_r2={f_r2:.2f}\tf_p2={f_p2:.2f}\tgamma2={gamma2:.2f}\tc2={c2:.2f}\tf3dB2={f3dB2}\n*Changed to None*"
            )
            f_r2, f_p2, gamma2, f3dB2 = None, None, None, None
    return (
        f * 10**-9,
        S11_Real,
        S11_Imag,
        S21_Magnitude_to_fit,
        S21_Fit,
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
    )


# one_file_approximation("data", "745.s2p", 2, 50)
