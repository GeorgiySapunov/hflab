#!/usr/bin/env python3
# Equipment list:
# - Keysight B2901A Precision Source/Measure Unit
# - Thorlabs PM100USB Power and energy meter
# - Keysight 8163B Lightwave Multimeter
# - YOKOGAWA AQ6370D Optical Spectrum Analyzer
# - Advanced Temperature Test Systems Chuck System A160 CMI

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

from measure.liv import measure_liv
from measure.osa import measure_osa


def main():
    config = ConfigParser()
    config.read("config.ini")
    instruments_config = config["INSTRUMENTS"]
    # liv_config = config["LIV"]
    # osa_config = config["OSA"]
    other_config = config["OTHER"]
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

        print()
        print("Make sure addresses in the programm are correct!")
        print(
            f"Keysight_B2901A_address is set to       {instruments_config['Keysight_B2901A_address']}"
        )
        print(
            f"Thorlabs_PM100USB_address is set to     {instruments_config['Thorlabs_PM100USB_address']}"
        )
        print(
            f"Keysight_8163B_address is set to        {instruments_config['Keysight_8163B_address']}"
        )
        print(
            f"Yokogawa_AQ6370D_adress is set to       {instruments_config['YOKOGAWA_AQ6370D_address']}"
        )
        print(
            f"ATT_A160CMI_address is set to           {instruments_config['ATT_A160CMI_address']}"
        )
        print()
        print("following arguments are needed:")
        print("Equipment_choice WaferID Wavelength(nm) Coordinates Temperature(°C)")
        print("e.g. run 'python measure.py k2 gs15 1550 00C9 25'")
        print()
        print("for equipment choice use:")
        print("t    for Thorlabs PM100USB Power and energy meter")
        print("k1   for Keysight 8163B Lightwave Multimeter port 1")
        print("k2   for Keysight 8163B Lightwave Multimeter port 2")
        print("y    for YOKOGAWA AQ6370D Optical Spectrum Analyzer")
        print()
        print(
            "for multiple temperature you need to specify start, stop and step temperature values:"
        )
        print(
            "Equipment_choice WaferID Wavelength(nm) Coordinates Start_Temperature(°C) Stop_Temperature(°C) Temperature_Increment(°C)"
        )
        print("'-' is not allowed!")
        print("e.g. run 'python measure.py t gs15 1550 00C9 25 85 40'")
        print("in this case you will get LIVs for 25, 65 and 85 degrees")
        print()

    elif len(sys.argv) == 6:
        # parameters
        equipment = sys.argv[1]
        waferid = sys.argv[2]
        wavelength = sys.argv[3]
        coordinates = sys.argv[4]
        temperature_start = sys.argv[5]
        temperature_list = [float(temperature_start)]
        if len(sys.argv) == 8:
            temperature_end = sys.argv[6]
            temperature_increment = sys.argv[8]
            temperature_list = [float(temperature_start)]
            while temperature_list[-1] < float(temperature_end) - float(
                temperature_increment
            ):
                temperature_list.append(
                    float(temperature_list[-1] + float(temperature_increment))
                )
            temperature_list.append(float(temperature_end))
            temperature_list = [
                t
                for t in temperature_list
                if t <= float(other_config["temperature_limit"])
            ]

        for arg in sys.argv[1:]:
            if "-" in arg:
                raise Exception(
                    "\nCharacter '-' is not allowed! It breaks parsing file names.\n"
                )

        dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/"

        alarm = False
        pm100_toggle = False
        keysight_8163B_toggle = False
        k_port = None
        YOKOGAWA_AQ6370D_toggle = False

        powermeter = None
        osa = None
        PM100USB = None
        Keysight_8163B = None
        YOKOGAWA_AQ6370D = None

        if equipment == "t":
            pm100_toggle = True  # toggle Thorlabs PM100USB Power and energy meter
        elif equipment == "k1":
            keysight_8163B_toggle = True  # toggle Keysight 8163B Lightwave Multimeter
            k_port = "1"
        elif equipment == "k2":
            keysight_8163B_toggle = True  # toggle Keysight 8163B Lightwave Multimeter
            k_port = "2"
        elif equipment == "y":
            YOKOGAWA_AQ6370D_toggle = True  # toggle Keysight 8163B Lightwave Multimeter

        # initiate pyvisa
        rm = pyvisa.ResourceManager()
        # set addresses for devices
        Keysight_B2901A = rm.open_resource(
            instruments_config["Keysight_B2901A_address"],
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
        elif keysight_8163B_toggle:
            Keysight_8163B = rm.open_resource(
                instruments_config["Keysight_8163B_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            powermeter = "Keysight_8163B_port" + k_port
        elif YOKOGAWA_AQ6370D_toggle:
            YOKOGAWA_AQ6370D = rm.open_resource(
                instruments_config["YOKOGAWA_AQ6370D_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            osa = "YOKOGAWA_AQ6370D"
        if len(temperature_list) != 1:
            ATT_A160CMI = rm.open_resource(
                instruments_config["ATT_A160CMI_address"],
                write_termination="\r\n",
                read_termination="\n",
            )

        for i, set_temperature in enumerate(temperature_list):
            print(f"[{i}/{len(temperature_list)}] {set_temperature} degree Celsius")
            if len(temperature_list) != 1:  # TODO
                ATT_A160CMI_address.write("RS=1")
                ATT_A160CMI_address.write(f"TA=+{set_temperature:3.2f}")
                stable = False
                counter_stability = 0
                sign = 0
                time.sleep(120)
                while not stable:
                    time.sleep(3)
                    current_temperature_str = str(ATT_A160CMI_address.query("TA?"))
                    if current_temperature_str[4] == "+":
                        sign = 1
                    elif current_temperature_str[4] == "-":
                        sign = -1
                    current_temperature = sign * (
                        float(current_temperature_str[4:6])
                        + float(current_temperature_str[7:8]) / 100
                    )
                    error = abs(current_temperature - set_temperature)
                    if error < 0.05:
                        counter_stability += 1
                    else:
                        counter_stability = 0
                    if counter_stability == 10:
                        stable = True
            if powermeter:
                filepath, alarm = measure_liv(
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
                    image = mpimg.imread(filepath + "-all.png")
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

            if alarm:
                time.sleep(1)
                ATT_A160CMI_address.write("TA=+02500")
                break

        if len(temperature_list) != 1:
            time.sleep(1)
            ATT_A160CMI_address.write("TA=+02500")


# Run main when the script is run by passing it as a command to the Python interpreter (just a good practice)
if __name__ == "__main__":
    main()
