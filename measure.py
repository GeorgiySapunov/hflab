#!/usr/bin/env python3
"""
Equipment list:
- Keysight B2901A Precision Source/Measure Unit
- Thorlabs PM100USB Power and energy meter
- Keysight 8163B Lightwave Multimeter
- YOKOGAWA AQ6370D Optical Spectrum Analyzer
- Advanced Temperature Test Systems Chuck System A160 CMI
- Keysight N5247B PNA-X network analyzer
- CoherentSolutions MatrIQswitch
"""

import sys
import os
import re
import time
import datetime
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorama
from configparser import ConfigParser
from pathlib import Path

from src.measure_liv import measure_liv
from src.measure_optical_spectra import measure_osa
from src.measure_pna import measure_pna


def update_att_temperature(set_temperature, ATT_A160CMI=None):
    config = ConfigParser()
    config.read("config.ini")
    other_config = config["OTHER"]
    if not ATT_A160CMI:
        instruments_config = config["INSTRUMENTS"]
        rm = pyvisa.ResourceManager()
        # rm = pyvisa.ResourceManager('@py') # for pyvisa-py
        ATT_A160CMI = rm.open_resource(
            instruments_config["ATT_A160CMI_address"],
            write_termination="\r\n",
            read_termination="\n",
        )
    temp_for_att = ""
    if set_temperature >= 0 and set_temperature < 10:
        temp_for_att = "+00" + str(int(round(set_temperature, ndigits=2) * 100))
    elif set_temperature >= 10 and set_temperature < 100:
        temp_for_att = "+0" + str(int(round(set_temperature, ndigits=2) * 100))
    elif set_temperature >= 100 and set_temperature <= float(
        other_config["temperature_limit"]
    ):
        temp_for_att = "+" + str(int(round(set_temperature, ndigits=2) * 100))
    else:
        ATT_A160CMI.write("TS=+02500")
        Exception("Temperature is set too high!")
    # while len(temp_for_att) < 6:  # TODO check whether we need it
    #     temp_for_att = temp_for_att + "0"
    ATT_A160CMI.write(f"TS={temp_for_att}")
    stable = False
    counter_stability = 0
    sign = 0
    while not stable:
        time.sleep(10)
        current_temperature_str = str(ATT_A160CMI.query("TA?"))
        if current_temperature_str[3] == "+":
            sign = 1
        elif current_temperature_str[3] == "-":
            sign = -1
        current_temperature = sign * (
            float(current_temperature_str[4:7])
            + float(current_temperature_str[7:9]) / 100
        )
        error = abs(current_temperature - set_temperature)
        if error < 0.04:
            counter_stability += 1
        else:
            counter_stability = 0
        if counter_stability == 4:
            stable = True
        print(
            f"Temperature set to {set_temperature},\t measured {current_temperature},\t stabilizing [{counter_stability}/4]\r"
        )


def main():
    config = ConfigParser()
    config.read("config.ini")
    instruments_config = config["INSTRUMENTS"]
    # liv_config = config["LIV"]
    # osa_config = config["OSA"]
    other_config = config["OTHER"]
    pna_config = config["PNA"]
    # if python got less then or more then 6 parameters
    if len(sys.argv) not in (6, 8):
        # initiate pyvisa
        rm = pyvisa.ResourceManager()
        # rm = pyvisa.ResourceManager('@py') # for pyvisa-py
        print("List of resources:")
        print(rm.list_resources())
        print()

        # check visa addresses
        for addr in rm.list_resources():
            try:
                print(addr, "-->", rm.open_resource(addr).query("*IDN?").strip())
            except pyvisa.VisaIOError:
                pass

        print(
            f"""Make sure addresses in the programm are correct!
            Keysight_B2901A_address is set to       {instruments_config['Keysight_B2901A_address']}
            Thorlabs_PM100USB_address is set to     {instruments_config['Thorlabs_PM100USB_address']}
            Keysight_8163B_address is set to        {instruments_config['Keysight_8163B_address']}
            Yokogawa_AQ6370D_adress is set to       {instruments_config['YOKOGAWA_AQ6370D_address']}
            ATT_A160CMI_address is set to           {instruments_config['ATT_A160CMI_address']}

            following arguments are needed:
            Equipment_choice WaferID Wavelength(nm) Coordinates Temperature(°C)
            e.g. run 'python measure.py k2 gs15 1550 00C9 25'

            for equipment choice use:
            t    for Thorlabs PM100USB Power and energy meter
            k1   for Keysight 8163B Lightwave Multimeter port 1
            k2   for Keysight 8163B Lightwave Multimeter port 2
            y    for YOKOGAWA AQ6370D Optical Spectrum Analyzer
            s

            for multiple temperature you need to specify start, stop and step temperature values:
            Equipment_choice WaferID Wavelength(nm) Coordinates Start_Temperature(°C) Stop_Temperature(°C) Temperature_Increment(°C)
            '-' is not allowed!
            e.g. run 'python measure.py t gs15 1550 00C9 25 85 40'
            in this case you will get LIVs for 25, 65 and 85 degrees
            """
        )

    elif len(sys.argv) == 6 or len(sys.argv) == 8:
        # parameters
        equipment = sys.argv[1]
        waferid = sys.argv[2]
        wavelength = sys.argv[3]
        coordinates = sys.argv[4]
        temperature_start = sys.argv[5]
        temperature_list = [float(temperature_start)]
        temp_list_len = 1
        if len(sys.argv) == 8:
            temperature_end = sys.argv[6]
            temperature_increment = sys.argv[7]
            temperature_list = [float(temperature_start)]
            while temperature_list[-1] < float(temperature_end) - float(
                temperature_increment
            ):
                temperature_list.append(
                    round(float(temperature_list[-1] + float(temperature_increment)), 2)
                )
            temperature_list.append(float(temperature_end))
            temperature_list = [
                t
                for t in temperature_list
                if t <= float(other_config["temperature_limit"])
            ]
            temp_list_len = len(temperature_list)
            temperature_list = sorted(list(set(temperature_list)))
            print(f"temperature list: {temperature_list}")

        for arg in sys.argv[1:]:
            if "-" in arg:
                raise Exception(
                    "\nCharacter '-' is not allowed! It breaks parsing file names.\n"
                )

        dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/"

        alarm = False
        pm100_toggle = False
        Keysight_8163B_toggle = False
        k_port = None
        YOKOGAWA_AQ6370D_toggle = False
        Keysight_N5247B_toggle = False

        powermeter = None
        osa = None
        PM100USB = None
        Keysight_8163B = None
        YOKOGAWA_AQ6370D = None
        pna = None
        CoherentSolutions_MatrIQswitch = None

        if equipment == "t":
            pm100_toggle = True  # toggle Thorlabs PM100USB Power and energy meter
        elif equipment == "k1":
            Keysight_8163B_toggle = True  # toggle Keysight 8163B Lightwave Multimeter
            k_port = "1"
        elif equipment == "k2":
            Keysight_8163B_toggle = True  # toggle Keysight 8163B Lightwave Multimeter
            k_port = "2"
        elif equipment == "y":
            YOKOGAWA_AQ6370D_toggle = True  # toggle Keysight 8163B Lightwave Multimeter
        elif equipment == "p":
            Keysight_N5247B_toggle = True  # PNA

        # initiate pyvisa
        rm = pyvisa.ResourceManager()
        # set addresses for devices
        if any((pm100_toggle, Keysight_8163B_toggle, YOKOGAWA_AQ6370D_toggle)):
            Keysight_B2901A = rm.open_resource(
                instruments_config["Keysight_B2901A_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
        elif Keysight_N5247B_toggle:
            Keysight_B2901A = rm.open_resource(
                instruments_config["Keysight_B2901A_2_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if pm100_toggle:
            PM100USB = rm.open_resource(
                instruments_config["Thorlabs_PM100USB_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            powermeter = "PM100USB"
        elif Keysight_8163B_toggle:
            Keysight_8163B = rm.open_resource(
                instruments_config["Keysight_8163B_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            powermeter = "Keysight_8163B_port" + str(k_port)
        elif YOKOGAWA_AQ6370D_toggle:
            YOKOGAWA_AQ6370D = rm.open_resource(
                instruments_config["YOKOGAWA_AQ6370D_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            osa = "YOKOGAWA_AQ6370D"
        elif Keysight_N5247B_toggle:
            Keysight_N5247B = rm.open_resource(
                instruments_config["Keysight_N5247B_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            pna = "Keysight_N5247B"
            optical_switch_port = int(pna_config["optical_switch_port"])
            if optical_switch_port:
                CoherentSolutions_MatrIQswitch = rm.open_resource(
                    instruments_config["CoherentSolutions_MatrIQswitch_address"],
                    write_termination="\r\n",
                    read_termination="\n",
                )
                optical_switch = "CoherentSolutions_MatrIQswitch"

        if temp_list_len != 1:
            ATT_A160CMI = rm.open_resource(
                instruments_config["ATT_A160CMI_address"],
                write_termination="\r\n",
                read_termination="\n",
            )

        for i, set_temperature in enumerate(temperature_list):
            print(f"[{i+1}/{len(temperature_list)}] {set_temperature} degree Celsius")
            if temp_list_len != 1:
                update_att_temperature(set_temperature, ATT_A160CMI=ATT_A160CMI)
            if powermeter:
                filepath, filename, alarm = measure_liv(
                    waferid,
                    wavelength,
                    coordinates,
                    set_temperature,
                    Keysight_B2901A=Keysight_B2901A,
                    PM100USB=PM100USB,
                    Keysight_8163B=Keysight_8163B,
                    k_port=k_port,
                )
                if len(temperature_list) == 1:
                    # show figure
                    image = mpimg.imread(filepath / (filename + "-everything.png"))
                    plt.imshow(image)
                    plt.axis("off")
                    plt.show()

            elif osa:
                alarm = measure_osa(
                    waferid,
                    wavelength,
                    coordinates,
                    set_temperature,
                    Keysight_B2901A=Keysight_B2901A,
                    YOKOGAWA_AQ6370D=YOKOGAWA_AQ6370D,
                )
            elif pna:
                alarm = measure_pna(
                    waferid,
                    wavelength,
                    coordinates,
                    set_temperature,
                    Keysight_B2901A=Keysight_B2901A,
                    Keysight_N5247B=Keysight_N5247B,
                    CoherentSolutions_MatrIQswitch=CoherentSolutions_MatrIQswitch,
                )

            if alarm and len(temperature_list) != 1:
                time.sleep(1)
                ATT_A160CMI.write("TS=+02500")
                break

        if len(temperature_list) != 1:
            time.sleep(1)
            ATT_A160CMI.write("TS=+02500")


# Run main when the script is run by passing it as a command to the Python interpreter (just a good practice)
if __name__ == "__main__":
    colorama.init()
    start_time = datetime.datetime.now()
    main()
    end_time = datetime.datetime.now()
    print(f"Duration: {end_time - start_time}")
