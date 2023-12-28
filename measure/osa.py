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
    # liv_config = config["LIV"]
    osa_config = config["OSA"]
    # other_config = config["OTHER"]
    max_current = float(osa_config["max_current"])
    osa_span = float(osa_config["osa_span"])
    current_increment_OSA = float(osa_config["current_increment_OSA"])

    alarm = False
    warnings = []
    if YOKOGAWA_AQ6370D:
        YOKOGAWA_AQ6370D_toggle = True
        osa = "YOKOGAWA_AQ6370D"

    dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/"  # get the directory path

    # read LIV .csv file
    walk = list(os.walk(dirpath + "LIV"))
    string_for_re = (
        f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C".replace(".", "\.")
        + ".*\\.csv"
    )
    r = re.compile(string_for_re)
    files = walk[0][2]
    matched_files = list(filter(r.match, files))
    matched_files.sort(reverse=True)
    file = matched_files[0]
    liv_dataframe = pd.read_csv(dirpath + "LIV/" + file)

    # get maximum current from LIV file
    max_current = liv_dataframe.iloc[-1]["Current set, mA"]

    # make a list of currents for spectra measurements
    osa_current_list = [0.0]
    round_to = max(0, int(np.ceil(np.log10(1 / current_increment_OSA))))
    while osa_current_list[-1] <= max_current - current_increment_OSA:
        osa_current_list.append(
            round(osa_current_list[-1] + current_increment_OSA, round_to)
        )
    # osa_current_list = np.arange(
    #     0,
    #     max_current,
    #     current_increment_OSA,
    # )
    # round_to = max(0, int(np.ceil(np.log10(1 / current_increment_OSA))))
    # osa_current_list = np.array([round(i, round_to) for i in osa_current_list])
    print(f"to {osa_current_list[-1]} mA")

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
    YOKOGAWA_AQ6370D.write(":SENSe:BANDwidth:RESolution 0.032nm")  # TODO it changes
    YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:CENTer {wavelength}nm")
    YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:SPAN {osa_span}nm")
    YOKOGAWA_AQ6370D.write(":SENSe:SWEep:POINts 2000")
    # YOKOGAWA_AQ6370D.write(":sens:sweep:points:auto on")
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
            f"{current_set:3.2f} mA: {current_measured:10.5f} mA, {voltage_measured_along_osa:8.5f} V"
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
                f"WARNING! Current set is {current_set}, while current measured is {current_measured}"
            )

        voltage_measured_along_liv = liv_dataframe.loc[
            "Current set, mA" == current_set
        ]["Voltage, V"]
        voltage_error = abs(voltage_measured_along_osa - voltage_measured_along_liv)
        if round(voltage_measured_along_osa, 2) != round(voltage_measured_along_liv, 2):
            warnings.append(
                f"voltage measured along osa={voltage_measured_along_osa} mA, voltage measured along liv={voltage_measured_along_liv} mA"
            )
            print(
                f"WARNING! Voltage measured along osa={voltage_measured_along_osa} mA, voltage measured along liv={voltage_measured_along_liv} mA"
            )

        if current_error >= 0.03:
            alarm = True
            print(
                f"ALARM! Current set is {current_set}, while current measured is {current_measured}\tBreaking the measurements!"
            )
            break  # break the loop

        if voltage_error >= 0.1:
            alarm = True
            print(
                f"ALARM! Voltage measured along osa={voltage_measured_along_osa} mA, voltage measured along liv={voltage_measured_along_liv} mA\tBreaking the measurements!"
            )
            break  # break the loop

    YOKOGAWA_AQ6370D.write("*CLS")

    # slowly decrease current
    current_measured = float(Keysight_B2901A.query("MEAS:CURR?"))  # measure current
    for current_set in np.arange(current_measured, 0, -0.1):
        Keysight_B2901A.write(
            ":SOUR:CURR " + str(current_set / 1000)
        )  # Outputs i mA immediately
        print(f"Current set: {current_set:3.1f} mA")
        time.sleep(0.01)  # 0.01 sec for a step, 1 sec for 10 mA

    # Measurement is stopped by the :OUTP OFF command.
    Keysight_B2901A.write(":OUTP OFF")
    Keysight_B2901A.write(f":SOUR:CURR 0.001")

    timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
    if not os.path.exists(dirpath + "OSA"):  # make directories
        os.makedirs(dirpath + "OSA")

    # get a filepath and save .csv files
    filepath = (
        dirpath
        + "OSA/"
        + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{timestr}-{osa}"
    )

    iv.to_csv(filepath + "-IV.csv", index=False)
    spectra.to_csv(filepath + "-OS.csv", index=False)

    print(f"Warnings: {len(warnings)}")
    if warnings:
        print(*warnings, sep="\n")

    return alarm
