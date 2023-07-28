#!/usr/bin/env python3
# Equipment list:
# - Keysight B2901A Precision Source/Measure Unit
# - Thorlabs PM100USB Power and energy meter
# - Keysight 8163B Lightwave Multimeter TODO
# - YOKOGAWA AQ6370D Optical Spectrum Analyzer TODO

import sys
import os
import time
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

#           _   _   _
#          | | | | (_)
#  ___  ___| |_| |_ _ _ __   __ _ ___
# / __|/ _ \ __| __| | '_ \ / _` / __|
# \__ \  __/ |_| |_| | | | | (_| \__ \
# |___/\___|\__|\__|_|_| |_|\__, |___/
#                            __/ |
#                           |___/
Keysight_B2901A_address = "GPIB0::23::INSTR"
Thorlabs_PM100USB_address = "USB0::0x1313::0x8072::1923257::INSTR"
Keysight_8163B_address = ""


pm100_toggle = True  # toggle Thorlabs PM100USB Power and energy meter
Keysight_8163B_toggle = False  # TODO toggle Keysight 8163B Lightwave Multimeter

current_list = [
    i / 100000 for i in range(0, 5000, 1)
]  # list of current to measure (from 0 to 50 mA)
# current_list = [i/1000 for i in [**put a list of currents here**]] # put values and uncomment for arbitrary list of currents to measure
stop_cond = 0.8  # stop if power lower then 80% of max power
#
#
#


def main():
    # if python got less then or more then 5 parameters
    if len(sys.argv) != 5:
        # initiate pyvisa
        rm = pyvisa.ResourceManager()
        # rm = pyvisa.ResourceManager('@py') # for pyvisa-py
        print("List of resources:")
        print(rm.list_resources())

        # check visa addresses
        for addr in rm.list_resources():
            try:
                print(addr, "-->", rm.open_resource(addr).query("*IDN?").strip())
            except pyvisa.VisaIOError:
                pass

        print("Make sure addressees in the programm are correct!")
        print(f"Keysight_B2901A_address set to    {Keysight_B2901A_address}")
        print(f"Thorlabs_PM100USB_address set to           {Thorlabs_PM100USB_address}")
        print(f"Keysight_8163B_address set to           {Keysight_8163B_address}")
        print()
        print("following arguments are needed:")
        print("WaferID Wavelength(nm) Coordinates Temperature(°C)")
        print("e.g. run 'python measure.py gs15 1550 0000 25'")

    else:
        notes = input("Input notes: ")
        if notes:
            notes_withspace = " " + notes
        else:
            notes_withspace = ""

        # parameters
        waferid = sys.argv[1]
        wavelength = sys.argv[2]
        coordinates = sys.argv[3]
        temperature = sys.argv[4]

        # initiate pyvisa
        rm = pyvisa.ResourceManager()
        # set addresses for devices
        Keysight_B2901A = rm.open_resource(Keysight_B2901A_address)
        if pm100_toggle:
            PM100USB = rm.open_resource(Thorlabs_PM100USB_address)
            powermeter = "PM100USB"
        if Keysight_8163B_toggle:
            Keysight_8163B = rm.open_resource(Keysight_8163B_address)
            powermeter = "Keysight_8163B"

        # initiate pandas Data Frame
        df = pd.DataFrame(
            columns=[
                "Current set, mA",
                "Current, mA",
                "Voltage, V",
                "Output power, mW",
                "Power consumption, mW",
            ]
        )

        Keysight_B2901A.write(
            "*RST"
        )  # The initial settings are applied by the *RST command

        if pm100_toggle:
            PM100USB.write("sense:corr:wav " + wavelength)  # set wavelength
            PM100USB.write("power:dc:unit W")  # set power units
        elif Keysight_8163B_toggle:  # TODO
            Keysight_8163B.write("*cls'")
            Keysight_8163B.write(
                "SENS1:CHAN2:POW:WAV:VAL" + wavelength + "nm"
            )  # set wavelength
            Keysight_8163B.write("SENS1:CHAN2:POW:UNIT W")  # set power units

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

        # initate power and max power variables with 0
        max_output_power = 0
        output_power = 0

        # Creating figure
        fig = plt.figure(figsize=(8, 5))
        ax1 = fig.add_subplot(221)  # subplot for set current
        ax12 = ax1.twinx()

        ax2 = fig.add_subplot(222)  # subplot for measured current
        ax22 = ax2.twinx()

        ax3 = fig.add_subplot(223)  # subplot for power

        # plt.title(
        #    waferid
        #    + " "
        #    + wavelength
        #    + " nm "
        #    + coord
        #    + notes_withspace
        #    + " "
        #    + temperature
        #    + " °C"
        # )  # Adding title

        # Adding legend
        # ax1.legend(loc=0)
        # ax12.legend(loc=0)
        # ax2.legend(loc=0)
        # ax22.legend(loc=0)

        ax1.grid()  # adding grid
        ax2.grid()  # adding grid
        ax3.grid()  # adding grid

        # Adding labels
        ax1.set_xlabel("Current set, mA")
        ax2.set_xlabel("Current measured, mA")
        ax3.set_xlabel("Current measured, mA")

        ax1.set_ylabel("Output power, mW")
        ax2.set_ylabel("Output power, mW")
        ax3.set_ylabel("Power, mW")

        ax12.set_ylabel("Voltage, V", color="red")
        ax22.set_ylabel("Voltage, V", color="red")

        # Setting Y limits
        # ax1.set_ylim(0, 40) # Power
        # ax12.set_ylim(0, 10) # Voltage
        # ax2.set_ylim(0, 40) # Power
        # ax22.set_ylim(0, 10) # Voltage

        # functions to build graphs
        def buildplt_all(dataframe=df):
            # select columns in the Data Frame
            seti = dataframe["Current set, mA"]
            i = dataframe["Current, mA"]
            v = dataframe["Voltage, V"]
            l = dataframe["Output power, mW"]
            p = dataframe["Power consumption, mW"]

            # Plotting
            ax1.plot(seti, l, "-", label="Output power, mW", marker="o")
            ax2.plot(i, l, "-", label="Output power, mW", marker="o")
            ax3.plot(i, l, "-", label="Output power, mW", marker="o")
            # Creating Twin axes
            ax12.plot(seti, v, "-r", label="Voltage, V", marker="o")
            ax22.plot(i, v, "-r", label="Voltage, V", marker="o")
            ax3.plot(i, p, "-b", label="Power consumption, mW", marker="o")
            ax3.legend(loc=0)
            plt.draw()
            plt.pause(0.01)

        def buildplt_tosave(dataframe=df):
            # Creating figure
            fig = plt.figure()
            ax = fig.add_subplot(111)
            ax2 = ax.twinx()

            plt.title(
                waferid
                + " "
                + wavelength
                + " nm "
                + coordinates
                + notes_withspace
                + " "
                + temperature
                + " °C"
            )  # Adding title

            # Adding legend
            # ax.legend(loc=0)
            # ax2.legend(loc=0)

            ax.grid()  # adding grid

            # Adding labels
            ax.set_xlabel("Current, mA")
            ax.set_ylabel("Output power, mW")
            ax2.set_ylabel("Voltage, V", color="red")

            # Setting Y limits
            # ax.set_ylim(0, 40) # Power
            # ax2.set_ylim(0, 10) # Voltage

            # select columns in the Data Frame
            seti = dataframe["Current set, mA"]
            i = dataframe["Current, mA"]
            # i = seti # uncoment this line to use "Current set, mA" column to plot graphs
            v = dataframe["Voltage, V"]
            l = dataframe["Output power, mW"]

            # Plotting dataset_2
            ax.plot(i, l, "-", label="Output power, mW", marker="o")
            # Creating Twin axes for dataset_1
            ax2.plot(i, v, "-r", label="Voltage, V", marker="o")

        #                 (_)       | |
        #  _ __ ___   __ _ _ _ __   | | ___   ___  _ __
        # | '_ ` _ \ / _` | | '_ \  | |/ _ \ / _ \| '_ \
        # | | | | | | (_| | | | | | | | (_) | (_) | |_) |
        # |_| |_| |_|\__,_|_|_| |_| |_|\___/ \___/| .__/
        #                                         | |
        #                                         |_|
        for i in current_list:
            # time.sleep(0.05)
            Keysight_B2901A.write(
                ":SOUR:CURR " + str(i)
            )  # Outputs i Ampere immediately
            time.sleep(0.03)
            voltage = float(Keysight_B2901A.query("MEAS:VOLT?"))  # measure Voltage, V
            # time.sleep(0.06)  # TODO do I need this?
            current = float(Keysight_B2901A.query("MEAS:CURR?"))  # measure Current, mA
            if pm100_toggle:
                output_power = float(
                    PM100USB.query("measure:power?")
                )  # measure output power, W
                output_power *= 1000
                # PM100USB.query("*OPC?") # synchronization TODO is it working?
                # print(PM100USB.query("*OPC?")) # TODO
                if output_power > max_output_power:  # track max power
                    max_output_power = output_power
            elif Keysight_8163B_toggle:  # TODO test
                output_power = float(
                    Keysight_8163B.query("FETC1:CHAN2:POW?")
                )  # measure output power, W
                output_power *= 1000
                # PM100USB.query("*OPC?") # synchronization TODO is it working?
                # print(PM100USB.query("*OPC?")) # TODO
                if output_power > max_output_power:  # track max power
                    max_output_power = output_power

            df.loc[len(df)] = [
                i * 1000,
                current * 1000,
                voltage,
                None,
                None,
            ]  # add current, measured current, voltage, and power to the DataFrame
            if (
                pm100_toggle or Keysight_8163B_toggle
            ):  # add power data if pm100toggle is set to True TODO test
                df.iloc[-1]["Output power, mW"] = output_power
                df.iloc[-1]["Power consumption, mW"] = output_power * current * 1000

            print(
                f"{i*1000:3.2f} mA: {current*1000:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW"
            )
            # buildplt_all()  # plot the data

            if i > 0.003:  # if current is more then 3 mA
                if (
                    output_power <= max_output_power * stop_cond or output_power <= 0.01
                ):  # check conditions to stop the measurements
                    break  # break the loop

        # slowly decrease current
        current = float(Keysight_B2901A.query("MEAS:CURR?"))  # measure current
        for i in range(int(current * 10000), 0, -1):  # e.g. 5 mA to 50 and 1 is a step
            i /= 10000  # makes 0.1 mA steps
            Keysight_B2901A.write(":SOUR:CURR " + str(i))  # Outputs i A immediately
            print(f"Current set: {i*1000:3.1f} mA")
            time.sleep(0.1)  # 0.2 sec for a step

        Keysight_B2901A.write(
            ":OUTP OFF"
        )  # Measurement is stopped by the :OUTP OFF command.

        # print(df.head()) # print first 5 lines of Data Frame

        if notes:  # to add dash in directory/file names
            notes = "-" + notes
        dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}{notes}/liv"
        filepath = (
            f"data/{waferid}-{wavelength}nm/{coordinates}{notes}/liv/"
            + "{waferid}-{wavelength}nm-{coord}{notes}-{temperature}c-{powermeter}-{timestr}"
        )

        if not os.path.exists(dirpath):  # make directories
            os.makedirs(dirpath)

        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
        df.to_csv(filepath + ".csv")  # save DataFrame to csv file

        # save figures
        buildplt_all()
        plt.savefig(filepath + "-all")  # save figure
        buildplt_tosave()
        plt.savefig(filepath)  # save figure
        plt.show()


# Run main when the script is run by passing it as a command to the Python interpreter (just a good practice)
if __name__ == "__main__":
    main()
