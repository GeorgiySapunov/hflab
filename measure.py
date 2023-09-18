#!/usr/bin/env python3
# Equipment list:
# - Keysight B2901A Precision Source/Measure Unit
# - Thorlabs PM100USB Power and energy meter
# - Keysight 8163B Lightwave Multimeter
# - YOKOGAWA AQ6370D Optical Spectrum Analyzer TODO
# - Advanced Temperature Test Systems Chuck System A160 CMI TODO

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
from liv import measure_liv
from osa import measure_osa


def main():
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
            f"Keysight_B2901A_address is set to       {settings['Keysight_B2901A_address']}"
        )
        print(
            f"Thorlabs_PM100USB_address is set to     {settings['Thorlabs_PM100USB_address']}"
        )
        print(
            f"Keysight_8163B_address is set to        {settings['Keysight_8163B_address']}"
        )
        print(
            f"Yokogawa_AQ6370D_adress is set to       {settings['YOKOGAWA_AQ6370D_address']}"
        )
        print(
            f"ATT_A160CMI_address is set to           {settings['ATT_A160CMI_address']}"
        )
        print()
        print("following arguments are needed:")
        print("Equipment_choice WaferID Wavelength(nm) Coordinates Temperature(°C)")
        print("e.g. run 'python measure.py k2 gs15 1550 00C9 25'")
        print()
        print("for equipment choise use:")
        print("t    for Thorlabs PM100USB Power and energy meter")
        print("k1   for Keysight 8163B Lightwave Multimeter port 1")
        print("k2   for Keysight 8163B Lightwave Multimeter port 2")
        print("y    for YOKOGAWA AQ6370D Optical Spectrum Analyzer")
        print()
        print(
            "for multiple temperature you need to specify start, stop and step tempertature values:"
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
        temperature1 = sys.argv[5]
        temperature_list = [float(temperature1)]
        if len(sys.argv) == 8:
            temperature2 = sys.argv[6]
            temperature3 = sys.argv[8]
            temperature_list = list(
                range(float(temperature1), float(temperature2), float(temperature3))
            ).append(float(temperature2))

        for arg in sys.argv[1:]:
            if "-" in arg:
                raise Exception(
                    "\nCharacter '-' is not allowed! It breaks parsing file names.\n"
                )

        dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/"

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
        Keysight_B2901A = rm.open_resource(settings["Keysight_B2901A_address"])
        if pm100_toggle:
            PM100USB = rm.open_resource(settings["Thorlabs_PM100USB_address"])
            powermeter = "PM100USB"
        elif keysight_8163B_toggle:
            Keysight_8163B = rm.open_resource(settings["Keysight_8163B_address"])
            powermeter = "Keysight_8163B_port" + k_port
        elif YOKOGAWA_AQ6370D_toggle:
            YOKOGAWA_AQ6370D = rm.open_resource(settings["YOKOGAWA_AQ6370D_address"])
            osa = "YOKOGAWA_AQ6370D"
        if len(temperature_list) != 1:
            ATT_A160CMI = rm.open_resource(
                ATT_A160CMI_address, write_termination="\r\n", read_termination="\n"
            )

        if powermeter:
            for temperature in temperature_list:
                # if len(temperature_list) != 1:  # TODO
                #     temp_for_att = ""
                #     if temperature >= 0 and temperature < 10:
                #         temp_for_att = "+00" + str(round(temperature, ndigits=2))
                #     elif temperature >= 10 and temperature < 100:
                #         temp_for_att = "+0" + str(round(temperature, ndigits=2))
                #     elif temperature >= 100 and temperature <= temperature_limit:
                #         temp_for_att = "+" + str(round(temperature, ndigits=2))
                #     else:
                #         Exception("Temperature is too high!")
                #     ATT_A160CMI_address.write("RS=1")
                #     ATT_A160CMI_address.write("TA=+" + temp_for_att)
                #     current_temperature = float(ATT_A160CMI_address.query("TA?"))
                #     pass
                #     # TODO
                filepath = measure_liv(
                    waferid,
                    wavelength,
                    coordinates,
                    temperature,
                    Keysight_B2901A=Keysight_B2901A,
                    PM100USB=PM100USB,
                    Keysight_8163B=Keysight_8163B,
                    k_port=k_port,
                    **settings,
                )
                if len(temperature_list) == 1:
                    # show figure
                    image = mpimg.imread(filepath + "-all.png")
                    plt.imshow(image)
                    plt.show()

        elif osa:
            for temperature in temperature_list:
                if len(temperature_list) != 1:
                    pass
                    # TODO
                measure_osa(
                    waferid,
                    wavelength,
                    coordinates,
                    temperature,
                    Keysight_B2901A=Keysight_B2901A,
                    YOKOGAWA_AQ6370D=YOKOGAWA_AQ6370D,
                    **settings,
                )


# Run main when the script is run by passing it as a command to the Python interpreter (just a good practice)
if __name__ == "__main__":
    main()
