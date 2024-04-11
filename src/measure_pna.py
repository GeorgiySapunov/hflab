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
from configparser import ConfigParser
from termcolor import colored
from pathlib import Path


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
):
    colorama.init()
    config = ConfigParser()
    config.read("config.ini")
    # instruments_config = config["INSTRUMENTS"]
    pna_config = config["PNA"]
    current_increment_PNA = float(pna_config["current_increment_PNA"])

    alarm = False
    warnings = []
    if Keysight_N5247B:
        YOKOGAWA_AQ6370D_toggle = True
        pna = "YOKOGAWA_AQ6370D"

    dirpath = (
        Path("data") / f"{waferid}-{wavelength}nm" / f"{coordinates}"
    )  # get the directory path
    pnadir = dirpath / "PNA"
    pnadir.mkdir(exist_ok=True)

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
        livfile_max_current = input(
            "LIV file is not found! Please, input max_current (mA):"
        )

    # make a list of currents for spectra measurements
    pna_current_list = [0.0]
    round_to = max(0, int(np.ceil(np.log10(1 / current_increment_PNA))))
    while pna_current_list[-1] <= livfile_max_current - current_increment_PNA:
        pna_current_list.append(
            round(pna_current_list[-1] + current_increment_PNA, round_to)
        )
    livfile_max_current = pna_current_list[-1]

    if livfile:
        # make a data frame for additional IV measurements (.csv file in OSA directory)
        iv = pd.DataFrame(
            columns=[
                "Current set, mA",
                "Current, mA",
                "Voltage, V",
                "Power consumption, mW",
            ]
        )

    # make a data frame for spectra measurements
    columns_spectra = ["Wavelength, nm"] + [
        f"Intensity at {i:.2f} mA, dBm" for i in pna_current_list
    ]
    spectra = pd.DataFrame(columns=columns_spectra, dtype="float64")

    # initial setings for PNA
    Keysight_N5247B.write("CALC1:PAR:DEFEXT 'ch1_s22', S22")
    Keysight_N5247B.write("CALC1:PAR:DEFEXT 'ch1_s12', S12")
    status = None

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
        print(
            f"[{current_set:3.2f}/{livfile_max_current:3.2f} mA] {current_measured:10.5f} mA, {voltage_measured_along_osa:8.5f} V",
            end="\n",
        )

        # YOKOGAWA_AQ6370D.write("*CLS")
        Keysight_N5247B.write("INIT1:IMM")

        status = Keysight_N5247B.query("OPC?")[0]

        # loop to check whether spectrum is aquired
        while status != "1":
            status = Keysight_N5247B.query("OPC?")[0]
            time.sleep(0.3)

        # Keysight_N5247B.write("*CLS")
        Keysight_N5247B.write("CALC1:PAR:SEL 'ch1_s22'")
        S11 = Keysight_N5247B.query("CALC1:DATA? SDATA").strip().split(",")
        S11real, S11imag = S11[0::2], S11[1::2]
        S11 = np.array(S11real) + 1j * np.array(S11imag)
        Keysight_N5247B.write("CALC1:PAR:SEL 'ch1_s12'")
        S21 = Keysight_N5247B.query("CALC1:DATA? SDATA").strip().split(",")
        S21real, S21imag = S21[0::2], S21[1::2]
        S21 = np.array(S21real) + 1j * np.array(S21imag)
        f = np.array(Keysight_N5247B.query("CALC1:X?").strip().split(","))
        S = np.zeros((len(f), 2, 2), dtype=complex)
        S[:, 0, 0] = S11
        S[:, 1, 0] = S21
        # S[:,0,1] = S12
        # S[:,1,1] = S22
        vcsel_ntwk = rf.Network(s=S, f=f, f_unit="Hz")
        s2pfilename = f"{waferid}-{wavelength}nm-{coordinates}-{current_set}mA-{temperature}°C-{timestr}-{pna}"
        vcsel_ntwk.write_touchstone(filename=s2pfilename, dir=pnadir)

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

    Keysight_N5247B.write("*CLS")

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
