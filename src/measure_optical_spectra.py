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
from configparser import ConfigParser
from termcolor import colored
from pathlib import Path


def check_maximum_current(livfile: Path):
    liv_dataframe = pd.read_csv(livfile)
    max_current = liv_dataframe.iloc[-1]["Current set, mA"]
    return max_current, liv_dataframe


def measure_osa(
    waferid,
    wavelength,
    coordinates,
    temperature,
    Keysight_B2901A=None,
    YOKOGAWA_AQ6370D=None,
):
    config = ConfigParser()
    config.read("config.ini")
    # instruments_config = config["INSTRUMENTS"]
    liv_config = config["LIV"]
    osa_config = config["OSA"]
    # other_config = config["OTHER"]
    max_current = float(liv_config["max_current"])
    osa_resolution = float(osa_config["osa_resolution"])
    osa_span = float(osa_config["osa_span"])
    # osa_points = float(osa_config["osa_points"]) # TODO del
    current_increment_OSA = float(osa_config["current_increment_OSA"])
    osa_force_wavelength = float(osa_config["osa_force_wavelength"])

    alarm = False
    warnings = []
    if YOKOGAWA_AQ6370D:
        YOKOGAWA_AQ6370D_toggle = True
        osa = "YOKOGAWA_AQ6370D"

    dirpath = (
        Path("data") / f"{waferid}-{wavelength}nm" / f"{coordinates}"
    )  # get the directory path

    # read LIV .csv file
    livdir = dirpath / "LIV"
    livfiles = sorted(
        livdir.glob(f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-*.csv"),
        reverse=True,
    )
    if len(livfiles) > 1:
        print(colored(f"{len(livfiles)} LIV files found:", "red"))
    for fileindex, file in enumerate(livfiles, start=1):
        max_current, liv_dataframe = check_maximum_current(file)
        print(
            f"[{fileindex}/{len(livfiles)}] {file.stem}\tMax current: {max_current} mA"
        )

    # make a list of currents for spectra measurements
    osa_current_list = [0.0]
    round_to = max(0, int(np.ceil(np.log10(1 / current_increment_OSA))))
    while osa_current_list[-1] <= max_current - current_increment_OSA:
        osa_current_list.append(
            round(osa_current_list[-1] + current_increment_OSA, round_to)
        )
    max_current = osa_current_list[-1]

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
        f"Intensity at {i:.2f} mA, dBm" for i in osa_current_list
    ]
    spectra = pd.DataFrame(columns=columns_spectra, dtype="float64")

    # initial setings for OSA
    YOKOGAWA_AQ6370D.write("*RST")
    YOKOGAWA_AQ6370D.write(":CALibration:ZERO once")
    YOKOGAWA_AQ6370D.write("FORMAT:DATA ASCII")
    YOKOGAWA_AQ6370D.write(":TRAC:ACT TRA")
    YOKOGAWA_AQ6370D.write(
        f":SENSe:BANDwidth:RESolution {osa_resolution}nm"
    )  # TODO it changes
    osa_force_wavelength
    # YOKOGAWA_AQ6370D.write(f":SENSe:SWEep:POINts {osa_points}")
    YOKOGAWA_AQ6370D.write(":SENs:SWEep:POINts:auto on")
    if osa_force_wavelength:
        YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:CENTer {osa_force_wavelength}nm")
    else:
        YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:CENTer {wavelength}nm")
    YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:CENTer {wavelength}nm")
    YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:SPAN {osa_span}nm")
    YOKOGAWA_AQ6370D.write(":SENSe:SENSe MID")
    YOKOGAWA_AQ6370D.write(":INITiate:SMODe SINGle")
    YOKOGAWA_AQ6370D.write("*CLS")
    status = None

    # main loop for OSA measurements at different currents
    for current_set in osa_current_list:
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
            f"[{current_set:3.2f}/{max_current:3.2f} mA] {current_measured:10.5f} mA, {voltage_measured_along_osa:8.5f} V"
        )

        # YOKOGAWA_AQ6370D.write("*CLS")
        YOKOGAWA_AQ6370D.write(":INITiate")

        status = YOKOGAWA_AQ6370D.query(":STATus:OPERation:EVENt?")[0]

        # loop to check whether spectrum is aquired
        while status != "1":
            status = YOKOGAWA_AQ6370D.query(":STATus:OPERation:EVENt?")[0]
            time.sleep(0.3)

        if not current_set:  # if i == 0.0:
            wavelength_list = YOKOGAWA_AQ6370D.query(":TRACE:X? TRA").strip().split(",")
            spectra["Wavelength, nm"] = (
                pd.Series(wavelength_list).astype("float64") * 10**9
            )
        YOKOGAWA_AQ6370D.write("*CLS")
        intensity = YOKOGAWA_AQ6370D.query(":TRACE:Y? TRA").strip().split(",")
        column_spectra = f"Intensity at {current_set:.2f} mA, dBm"
        spectra[column_spectra] = pd.Series(intensity).astype("float64")

        # deal with set/measured current mismatch
        current_error = abs(current_set - current_measured)
        if round(current_measured, round_to) != current_set:
            warnings.append(
                f"Current set={current_set} mA, current measured={current_measured} mA"
            )
            print(
                colored(
                    f"WARNING! Current set is {current_set}, while current measured is {current_measured}",
                    "yellow",
                )
            )

        voltage_measured_along_liv = float(
            (
                liv_dataframe.loc[liv_dataframe["Current set, mA"] == current_set][
                    "Voltage, V"
                ]
            ).iloc[0]
        )
        voltage_error = abs(voltage_measured_along_osa - voltage_measured_along_liv)
        if round(voltage_measured_along_osa, 2) != round(voltage_measured_along_liv, 2):
            warnings.append(
                f"Voltage measured along osa={voltage_measured_along_osa} V, voltage measured along liv={voltage_measured_along_liv} V"
            )
            print(
                colored(
                    f"WARNING! Voltage measured along osa={voltage_measured_along_osa} V, voltage measured along liv={voltage_measured_along_liv} V",
                    "yellow",
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

        if voltage_error >= 0.1:
            alarm = True
            print(
                colored(
                    f"ALARM! Voltage measured along osa={voltage_measured_along_osa} V, voltage measured along liv={voltage_measured_along_liv} V\tBreaking the measurements!",
                    "red",
                )
            )
            break  # break the loop

    YOKOGAWA_AQ6370D.write("*CLS")

    # slowly decrease current
    current_measured = (
        float(Keysight_B2901A.query("MEAS:CURR?")) * 1000
    )  # measure current in mA
    for current_set in np.arange(current_measured, 0, -0.1):
        Keysight_B2901A.write(
            ":SOUR:CURR " + str(current_set / 1000)
        )  # Outputs current_set mA immediately
        print(f"Current set: {current_set:3.1f} mA")
        time.sleep(0.01)  # 0.01 sec for a step, 1 sec for 10 mA

    # Measurement is stopped by the :OUTP OFF command.
    Keysight_B2901A.write(":OUTP OFF")
    Keysight_B2901A.write(f":SOUR:CURR 0.001")

    timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
    osadir = dirpath / "OSA"
    osadir.mkdir(exist_ok=True)

    filename = f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{timestr}-{osa}"

    iv.to_csv(osadir / (filename + "-IV.csv"), index=False)
    spectra.to_csv(osadir / (filename + "-OS.csv"), index=False)

    if warnings:
        print(colored(f"Warnings: {len(warnings)}", "yellow"))
        print(*[colored(warning, "yellow") for warning in warnings], sep="\n")

    return alarm
