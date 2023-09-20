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

from settings import settings


# parameters
# equipment = sys.argv[1]
# waferid = sys.argv[2]
# wavelength = sys.argv[3]
# coordinates = sys.argv[4]
# temperature = sys.argv[5]


def measure_osa(
    waferid,
    wavelength,
    coordinates,
    temperature,
    Keysight_B2901A=None,
    YOKOGAWA_AQ6370D=None,
    #
    Keysight_B2901A_address=None,
    Thorlabs_PM100USB_address=None,
    Keysight_8163B_address=None,
    YOKOGAWA_AQ6370D_address=None,
    ATT_A160CMI_address=None,
    current_list=None,
    beyond_rollover_stop_cond=None,
    current_limit1=None,
    current_limit2=None,
    temperature_limit=None,
    osa_span=None,
):
    if YOKOGAWA_AQ6370D:
        YOKOGAWA_AQ6370D_toggle = True
        osa = "YOKOGAWA_AQ6370D"

    dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/"

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
    dataframe = pd.read_csv(dirpath + "LIV/" + file)

    max_current = dataframe.iloc[-1]["Current set, mA"]

    osa_current_list = [i / 1000000 for i in range(0, int(max_current * 1000), 100)]

    iv = pd.DataFrame(
        columns=[
            "Current set, mA",
            "Current, mA",
            "Voltage, V",
            "Power consumption, mW",
        ]
    )

    spectra = pd.DataFrame()

    YOKOGAWA_AQ6370D.write("*RST")
    YOKOGAWA_AQ6370D.write(":CALibration:ZERO once")
    YOKOGAWA_AQ6370D.write("FORMAT:DATA ASCII")
    YOKOGAWA_AQ6370D.write(":TRAC:ACT TRA")
    YOKOGAWA_AQ6370D.write(":SENSe:BANDwidth:RESolution 0.017nm")
    YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:CENTer {wavelength}nm")
    YOKOGAWA_AQ6370D.write(f":SENSe:WAVelength:SPAN {osa_span}nm")
    YOKOGAWA_AQ6370D.write(":SENSe:SWEep:POINts 2000")
    # YOKOGAWA_AQ6370D.write(":sens:sweep:points:auto on")
    YOKOGAWA_AQ6370D.write(":SENSe:SENSe MID")
    YOKOGAWA_AQ6370D.write(":INITiate:SMODe SINGle")
    YOKOGAWA_AQ6370D.write("*CLS")
    status = None

    # main loop for osa
    for i in osa_current_list:
        # Outputs i Ampere immediately
        Keysight_B2901A.write(":SOUR:CURR " + str(i))
        time.sleep(0.03)
        # measure Voltage, V
        voltage = float(Keysight_B2901A.query("MEAS:VOLT?"))
        # measure Current, A
        current = float(Keysight_B2901A.query("MEAS:CURR?"))

        # add current, measured current, voltage, power, power consumption to the DataFrame
        iv.loc[len(iv)] = [
            i * 1000,
            current * 1000,
            voltage,
            voltage * current * 1000,
        ]

        # print data to the terminal
        print(f"{i*1000:3.2f} mA: {current*1000:10.5f} mA, {voltage:8.5f} V")

        YOKOGAWA_AQ6370D.write("*CLS")
        YOKOGAWA_AQ6370D.write(":INITiate")

        status = YOKOGAWA_AQ6370D.query(":STATus:OPERation:EVENt?")[0]

        while status != "1":
            status = YOKOGAWA_AQ6370D.query(":STATus:OPERation:EVENt?")[0]
            time.sleep(0.1)

        if not i:  # if i == 0.0:
            wavelength_list = YOKOGAWA_AQ6370D.query(":TRACE:X? TRA").strip().split(",")
            spectra["Wavelength, nm"] = pd.Series(wavelength_list) * 10**9
        YOKOGAWA_AQ6370D.write("*CLS")
        intensity = YOKOGAWA_AQ6370D.query(":TRACE:Y? TRA").strip().split(",")
        column_spectra = f"Intensity at {i*1000:.2f} mA, dBm"
        spectra[column_spectra] = pd.Series(intensity)

    YOKOGAWA_AQ6370D.write("*CLS")

    # slowly decrease current
    current = float(Keysight_B2901A.query("MEAS:CURR?"))  # measure current
    # e.g. 5 mA to 50 and 1 is a step
    for i in range(int(current * 10000), 0, -1):
        i /= 10000  # makes 0.1 mA steps
        Keysight_B2901A.write(":SOUR:CURR " + str(i))  # Outputs i A immediately
        print(f"Current set: {i*1000:3.1f} mA")
        time.sleep(0.01)  # 0.01 sec for a step, 1 sec for 10 mA

    # Measurement is stopped by the :OUTP OFF command.
    Keysight_B2901A.write(":OUTP OFF")

    timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
    if not os.path.exists(dirpath + "OSA"):  # make directories
        os.makedirs(dirpath + "OSA")

    filepath = (
        dirpath
        + "OSA/"
        + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{timestr}-{osa}"
    )

    iv.to_csv(filepath + "-IV.csv")
    spectra.to_csv(filepath + "-OS.csv")
