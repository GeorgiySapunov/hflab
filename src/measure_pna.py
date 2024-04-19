#!/usr/bin/env python3
import sys
import os
import re
import time
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorama
import skrf as rf
from scipy.optimize import curve_fit
from configparser import ConfigParser
from termcolor import colored
from pathlib import Path

from src.analysis_ssm_one_file import remove_pd, s21_func


def simple_S21fit(
    s2p_file=None,
    photodiode_s2p=rf.Network("resources/T3K7V9_DXM30BF_U00162.s2p"),
    probe_port=1,
):
    vcsel_df = s2p_file.to_dataframe("s")
    f = vcsel_df.index.values  # Hz
    S21_Magnitude, S21_Magnitude_to_fit = remove_pd(
        vcsel_df=vcsel_df, photodiode_s2p=photodiode_s2p, probe_port=probe_port
    )
    # Find the best-fit solution for S21
    # f_r, f_p, gamma, c
    S21_bounds = ([0, 0, 0, -np.inf], [81, 81, 2001, np.inf])
    popt_S21, pcovBoth = curve_fit(
        s21_func,
        f,
        S21_Magnitude_to_fit,
        # p0=[150, 150, 30, 150, 0],
        bounds=S21_bounds,
        maxfev=100000,
    )
    f_r, f_p, gamma, c = popt_S21
    S21_Fit = s21_func(f, *popt_S21)
    check_f3dB = np.where(S21_Fit < c - 3)[0]
    if check_f3dB.any():
        f3dB = f[np.where(S21_Fit < c - 3)[0][0]] * 10**-9  # GHz
    else:
        f3dB = 0
    return f_r, f_p, gamma, c, f3dB


def check_maximum_current(livfile: Path):
    liv_dataframe = pd.read_csv(livfile)
    max_current = liv_dataframe.iloc[-1]["Current set, mA"]
    return max_current, liv_dataframe


def measure_pna(
    waferid,
    wavelength,
    coordinates,
    temperature,
    Keysight_B2901A=None,
    Keysight_N5247B=None,
    CoherentSolutions_MatrIQswitch=None,
    update_windows=True,
):
    colorama.init()
    config = ConfigParser()
    config.read("config.ini")
    # instruments_config = config["INSTRUMENTS"]
    pna_config = config["PNA"]
    current_increment_PNA = float(pna_config["current_increment_PNA"])
    probe_port = int(pna_config["probe_port"])
    averaging_PNA = int(pna_config["averaging_PNA"])
    optical_switch_port = int(pna_config["optical_switch_port"])
    photodiode_s2p = rf.Network(pna_config["photodiode_s2p"])

    if probe_port == 1:
        optical_port = 2
    elif probe_port == 2:
        optical_port = 1
    else:
        raise Exception("probe_port is unclear")

    if optical_switch_port:
        CoherentSolutions_MatrIQswitch.write(f"ROUT1:CHAN1:STATE {optical_switch_port}")
        time.sleep(0.3)

    alarm = False
    warnings = []
    if Keysight_N5247B:
        Keysight_N5247B_toggle = True
        pna = "Keysight_N5247B"
        Keysight_N5247B.timeout = 20000

    dirpath = (
        Path("data") / f"{waferid}-{wavelength}nm" / f"{coordinates}"
    )  # get the directory path
    pnadir = dirpath / "PNA"
    pnadir.mkdir(exist_ok=True, parents=True)

    # read LIV .csv file
    livdir = dirpath / "LIV"
    livfiles = sorted(
        livdir.glob(
            f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-*PM100USB.csv"
        ),
        reverse=True,
    )
    if len(livfiles) > 1:
        livfile = True
        print(colored(f"{len(livfiles)} LIV files found:", "red"))
        for fileindex, file in enumerate(livfiles, start=1):
            livfile_max_current, liv_dataframe = check_maximum_current(file)
            print(
                f"[{fileindex}/{len(livfiles)}] {file.stem}\tMax current: {livfile_max_current} mA"
            )
    else:
        livfile = False
        livfile_max_current = float(
            input("LIV file is not found! Please, input max_current (mA): ")
        )

    # make a list of currents for measurements
    pna_current_list = [0.0]
    round_to = max(0, int(np.ceil(np.log10(1 / current_increment_PNA))))
    while pna_current_list[-1] <= livfile_max_current - current_increment_PNA:
        pna_current_list.append(
            round(pna_current_list[-1] + current_increment_PNA, round_to)
        )
    livfile_max_current = pna_current_list[-1]

    # make a data frame for additional IV measurements (.csv file in OSA directory)
    iv = pd.DataFrame(
        columns=[
            "Current set, mA",
            "Current, mA",
            "Voltage, V",
            "Power consumption, mW",
        ]
    )
    f3dBmax = 0

    # initial setings for PNA
    Keysight_N5247B.write("*CLS")
    Keysight_N5247B.write("TRIG:SOUR MAN")
    S11name = f"ch1_s{probe_port}{probe_port}"
    S21name = f"ch1_s{optical_port}{probe_port}"
    if update_windows:
        Keysight_N5247B.write("CALC:PAR:DEL:ALL")
    else:
        catalog = Keysight_N5247B.query("CALC1:PAR:CAT? DEF")[1:-1].split(",")
        # e.g. catalog == "ch1_s22,S22,ch1_s12,S12,ch1_s21,S21"
        if S11name in catalog:
            Keysight_N5247B.write(f"CALC:PAR:DEL {S11name}")
        if S21name in catalog:
            Keysight_N5247B.write(f"CALC:PAR:DEL {S21name}")
    Keysight_N5247B.write("CALC1:PAR:DEF " + S11name + f", S{probe_port}{probe_port}")
    Keysight_N5247B.write("CALC1:PAR:DEF " + S21name + f", S{optical_port}{probe_port}")
    if update_windows:
        # https://coppermountaintech.com/help-r/calcform.html?q=smith
        # https://planarchel.ru/instruction/rvna/calcform.html
        Keysight_N5247B.write("DISP:WIND1:STATE ON")
        Keysight_N5247B.write("DISP:WIND2:STATE ON")
        Keysight_N5247B.write("DISP:WIND3:STATE ON")
        Keysight_N5247B.write("DISP:WIND4:STATE ON")
        Keysight_N5247B.write("DISP:WIND5:STATE ON")
        # Smith S11
        Keysight_N5247B.write(
            "CALC1:PAR:DEF " + S11name + f"Smith, S{probe_port}{probe_port}"
        )
        Keysight_N5247B.write(f"CALC1:PAR:SEL {S11name}Smith")
        Keysight_N5247B.write("CALC1:FORMat SMIT")
        Keysight_N5247B.write(f"DISP:WIND1:TRAC:FEED {S11name}Smith")
        # LogM S11
        Keysight_N5247B.write(
            "CALC1:PAR:DEF " + S11name + f"LogM, S{probe_port}{probe_port}"
        )
        Keysight_N5247B.write(f"CALC1:PAR:SEL {S11name}LogM")
        Keysight_N5247B.write("CALC1:FORMat MLOG")
        Keysight_N5247B.write(f"DISP:WIND2:TRAC:FEED {S11name}LogM")
        # Real S11
        Keysight_N5247B.write(
            "CALC1:PAR:DEF " + S11name + f"Real, S{probe_port}{probe_port}"
        )
        Keysight_N5247B.write(f"CALC1:PAR:SEL {S11name}Real")
        Keysight_N5247B.write("CALC1:FORMat REAL")
        Keysight_N5247B.write(f"DISP:WIND3:TRAC:FEED {S11name}Real")
        # Imag S11
        Keysight_N5247B.write(
            "CALC1:PAR:DEF " + S11name + f"Imag, S{probe_port}{probe_port}"
        )
        Keysight_N5247B.write(f"CALC1:PAR:SEL {S11name}Imag")
        Keysight_N5247B.write("CALC1:FORMat IMAG")
        Keysight_N5247B.write(f"DISP:WIND4:TRAC:FEED {S11name}Imag")
        # S21 LogM
        Keysight_N5247B.write(
            "CALC1:PAR:DEF " + S21name + f"LogM, S{optical_port}{probe_port}"
        )
        Keysight_N5247B.write(f"CALC1:PAR:SEL {S21name}LogM")
        Keysight_N5247B.write("CALC1:FORMat MLOG")
        Keysight_N5247B.write(f"DISP:WIND5:TRAC:FEED {S21name}LogM")
        Keysight_N5247B.write("INIT1:IMM")
        # Keysight_N5247B.write("DISP:WIND1:Y:AUTO")
        Keysight_N5247B.query("*OPC?")
        Keysight_N5247B.write("DISP:WIND2:Y:AUTO")
        Keysight_N5247B.write("DISP:WIND3:Y:AUTO")
        Keysight_N5247B.write("DISP:WIND4:Y:AUTO")
        Keysight_N5247B.write("DISP:WIND5:Y:AUTO")

        # The initial settings are applied by the *RST command
        # Keysight_B2901A.write(":SOUR:CURR:RANG:AUTO 1")
        Keysight_B2901A.write("*RST")
        Keysight_B2901A.write(
            ":SOUR:FUNC:MODE CURR"
        )  # Setting the Source Output Mode to current
        Keysight_B2901A.write(
            ":SENS:CURR:PROT 0.1"
        )  # Setting the Limit/Compliance Value 100 mA
        Keysight_B2901A.write(
            ":SENS:VOLT:PROT 10"
        )  # Setting the Limit/Compliance Value 10 V
        Keysight_B2901A.write(
            ":OUTP ON"
        )  # Measurement channel is enabled by the :OUTP ON command.

    # main loop for OSA measurements at different currents
    for current_set in pna_current_list:
        # Outputs i Ampere immediately
        Keysight_B2901A.write(":SOUR:CURR " + str(current_set / 1000))
        time.sleep(0.03)
        # measure Voltage, V
        voltage_measured_along_osa = float(Keysight_B2901A.query("MEAS:VOLT?"))
        # measure Current, mA
        current_measured = float(Keysight_B2901A.query("MEAS:CURR?")) * 1000

        # add current, measured current, voltage, power, power consumption to the DataFrame
        iv.loc[len(iv)] = [
            current_set,
            current_measured,
            voltage_measured_along_osa,
            voltage_measured_along_osa * current_measured,
        ]

        # print data to the terminal
        current_set_text = f"[{current_set:3.2f}/{livfile_max_current:3.2f} mA] {current_measured:10.5f} mA, {voltage_measured_along_osa:8.5f} V"

        S11_list = []
        S21_LinMagnitude_list = []
        s2pfilename = f"{waferid}-{wavelength}nm-{coordinates}-{current_set}mA-{temperature}°C-{pna}.s2p"
        for i in range(averaging_PNA):
            print(f"{current_set_text} [{i+1}/{averaging_PNA}]", end="\r")
            Keysight_N5247B.write("INIT1:IMM")
            Keysight_N5247B.query("*OPC?")

            Keysight_N5247B.write(f"CALC1:PAR:SEL {S11name}")
            S11 = list(
                map(
                    float, Keysight_N5247B.query("CALC1:DATA? SDATA").strip().split(",")
                )
            )
            S11_Real, S11_Imag = np.array(S11[0::2]).reshape(-1, 1), np.array(
                S11[1::2]
            ).reshape(-1, 1)
            Keysight_N5247B.write(f"CALC1:PAR:SEL {S21name}")
            S21 = list(
                map(
                    float, Keysight_N5247B.query("CALC1:DATA? SDATA").strip().split(",")
                )
            )
            S21_Real, S21_Imag = np.array(S21[0::2]).reshape(-1, 1), np.array(
                S21[1::2]
            ).reshape(-1, 1)
            S21_LinMagnitude = np.sqrt(S21_Real**2 + S21_Imag**2)
            S11 = S11_Real + 1j * S11_Imag
            # S21 = S21_Real + 1j * S21_Imag
            S11_list.append(S11)
            S21_LinMagnitude_list.append(S21_LinMagnitude)
        S11_mean = np.hstack(S11_list).mean(axis=1)
        S21_LinMagnitude_mean = np.hstack(S21_LinMagnitude_list).mean(axis=1)
        S21_mean = S21_LinMagnitude_mean + 1j * 0
        f = np.array(
            list(map(float, Keysight_N5247B.query("CALC1:X?").strip().split(",")))
        )
        S = np.zeros((len(f), 2, 2), dtype=complex)
        S[:, 0, 0] = S11_mean
        S[:, 1, 0] = S21_mean
        # S[:,0,1] = S12
        # S[:,1,1] = S22
        vcsel_ntwk = rf.Network(s=S, f=f, f_unit="Hz")
        vcsel_ntwk.write_touchstone(filename=s2pfilename, dir=pnadir)

        f_r, f_p, gamma, c, f3dB = simple_S21fit(
            s2p_file=vcsel_ntwk,
            photodiode_s2p=photodiode_s2p,
        )
        if f3dB > f3dBmax:
            f3dBmax = f3dB
            color = "green"
        else:
            color = "blue"
        print(
            colored(
                f"{current_set_text} | f_r={f_r:3.2f} GHz, f_p={f_p:3.2f} GHz, gamma={gamma:3.2f} GHz, f3dB={f3dB:3.2f} GHz, f3dBmax={f3dBmax:3.2f} GHz",
                color=color,
            ),
            end="\n",
        )

        # deal with set/measured current mismatch
        current_error = abs(current_set - current_measured)
        if round(current_measured, round_to) != current_set:
            warnings.append(
                f"Current set={current_set} mA, current measured={current_measured} mA"
            )
            print(
                colored(
                    f"WARNING! Current set is {current_set}, while current measured is {current_measured}",
                    "cyan",
                )
            )

        if livfile:
            voltage_measured_along_liv = float(
                (
                    liv_dataframe.loc[liv_dataframe["Current set, mA"] == current_set][
                        "Voltage, V"
                    ]
                ).iloc[0]
            )
            voltage_error = abs(voltage_measured_along_osa - voltage_measured_along_liv)
            if round(voltage_measured_along_osa, 2) != round(
                voltage_measured_along_liv, 2
            ):
                warnings.append(
                    f"Voltage measured along osa={voltage_measured_along_osa} V, voltage measured along liv={voltage_measured_along_liv} V"
                )
                print(
                    colored(
                        f"WARNING! Voltage measured along osa={voltage_measured_along_osa} V, voltage measured along liv={voltage_measured_along_liv} V",
                        "cyan",
                    )
                )

        if current_error >= 0.03:
            alarm = True
            print(
                colored(
                    f"ALARM! Current set is {current_set}, while current measured is {current_measured}\tBreaking the measurements!",
                    "red",
                )
            )
            break  # break the loop

        if livfile and voltage_error >= 0.1:
            alarm = True
            print(
                colored(
                    f"ALARM! Voltage measured along osa={voltage_measured_along_osa} V, voltage measured along liv={voltage_measured_along_liv} V\tBreaking the measurements!",
                    "red",
                )
            )
            break  # break the loop

    # Keysight_N5247B.write("*CLS")
    Keysight_N5247B.write(f"CALC1:PAR:DEL {S11name}")
    Keysight_N5247B.write(f"CALC1:PAR:DEL {S21name}")
    Keysight_N5247B.write("TRIG:SOUR IMM")
    if optical_switch_port:
        CoherentSolutions_MatrIQswitch.write(f"ROUT1:CHAN1:STATE 1")

    # slowly decrease current
    current_measured = (
        float(Keysight_B2901A.query("MEAS:CURR?")) * 1000
    )  # measure current in mA
    for current_set in np.arange(current_measured, 0, -0.1):
        Keysight_B2901A.write(
            ":SOUR:CURR " + str(current_set / 1000)
        )  # Outputs current_set mA immediately
        print(f"Current set: {current_set:3.1f} mA", end="\r")
        time.sleep(0.01)  # 0.01 sec for a step, 1 sec for 10 mA

    # Measurement is stopped by the :OUTP OFF command.
    Keysight_B2901A.write(":OUTP OFF")
    Keysight_B2901A.write(f":SOUR:CURR 0.001")

    timestr = time.strftime("%Y%m%d-%H%M%S")  # current time

    filename = f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{timestr}-{pna}"

    iv.to_csv(pnadir / (filename + "-IV.csv"), index=False)

    if warnings:
        print(colored(f"Warnings: {len(warnings)}", "cyan"))
        print(*[colored(warning, "cyan") for warning in warnings], sep="\n")

    return alarm
