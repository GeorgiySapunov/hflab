#!/usr/bin/env python3
# Equipment list:
# - Keysight B2901A Precision Source/Measure Unit
# - Thorlabs PM100USB Power and energy meter
# - Keysight 8163B Lightwave Multimeter
# - YOKOGAWA AQ6370D Optical Spectrum Analyzer TODO

import sys
import os
import time
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#           _   _   _
#          | | | | (_)
#  ___  ___| |_| |_ _ _ __   __ _ ___
# / __|/ _ \ __| __| | '_ \ / _` / __|
# \__ \  __/ |_| |_| | | | | (_| \__ \
# |___/\___|\__|\__|_|_| |_|\__, |___/
#                            __/ |
#                           |___/
# Keysight_B2901A_address = "GPIB0::23::INSTR"
Keysight_B2901A_address = "USB0::0x0957::0x8B18::MY51143485::INSTR"
Thorlabs_PM100USB_address = "USB0::0x1313::0x8072::1923257::INSTR"
Keysight_8163B_address = "GPIB0::10::INSTR"
YOKOGAWA_AQ6370D_address = ""


current_list = [
    i / 1000000 for i in range(0, 50000, 10)
]  # list of current to measure (from 0 to 50 mA, 0.01 mA steps)
beyond_rollover_stop_cond = 0.9  # stop if power lower then 90% of max output power
current_limit1 = 4  # mA, stop measuremet if current above limit1 (mA) and output power less then 0.01 mW
current_limit2 = 10  # mA, stop measuremet if current above limit2 (mA) and maximum output power less then 0.5 mW
#
#
#


def main():
    # if python got less then or more then 6 parameters
    if len(sys.argv) != 6:
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

        print("Make sure addressees in the programm are correct!")
        print(f"Keysight_B2901A_address set to    {Keysight_B2901A_address}")
        print(f"Thorlabs_PM100USB_address set to           {Thorlabs_PM100USB_address}")
        print(f"Keysight_8163B_address set to           {Keysight_8163B_address}")
        print()
        print("following arguments are needed:")
        print("Equipment_choice WaferID Wavelength(nm) Coordinates Temperature(°C)")
        print("e.g. run 'python measure.py k2 gs15 1550 00C9 25'")
        print()
        print("for equipment choise use:")
        print("t   for Thorlabs PM100USB Power and energy meter")
        print("k1   for Keysight 8163B Lightwave Multimeter port 1")
        print("k2   for Keysight 8163B Lightwave Multimeter port 2")
        print("y   for YOKOGAWA AQ6370D Optical Spectrum Analyzer")

    else:
        # parameters
        equipment = sys.argv[1]
        waferid = sys.argv[2]
        wavelength = sys.argv[3]
        coordinates = sys.argv[4]
        temperature = sys.argv[5]

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
        powermeter = None
        osa = None
        rm = pyvisa.ResourceManager()
        # set addresses for devices
        Keysight_B2901A = rm.open_resource(Keysight_B2901A_address)
        if pm100_toggle:
            PM100USB = rm.open_resource(Thorlabs_PM100USB_address)
            powermeter = "PM100USB"
        elif keysight_8163B_toggle:
            Keysight_8163B = rm.open_resource(Keysight_8163B_address)
            powermeter = "Keysight_8163B_port" + k_port
        elif YOKOGAWA_AQ6370D_toggle:
            osa = "YOKOGAWA_AQ6370D"

        #  _     _____     __
        # | |   |_ _\ \   / /
        # | |    | | \ \ / /
        # | |___ | |  \ V /
        # |_____|___|  \_/
        if powermeter:
            print("Measuring LIV using " + powermeter)
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

            # The initial settings are applied by the *RST command
            Keysight_B2901A.write("*RST")

            if pm100_toggle:
                PM100USB.write("sense:corr:wav " + wavelength)  # set wavelength
                PM100USB.write("power:dc:unit W")  # set power units
            elif keysight_8163B_toggle:
                Keysight_8163B.write("*RST")  # reset
                # set wavelength
                Keysight_8163B.write(
                    "SENS1:CHAN" + k_port + ":POW:WAV " + wavelength + "nm"
                )
                # set power units
                Keysight_8163B.write("SENS1:CHAN" + k_port + ":POW:UNIT W")

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

            #  _____ _
            # |  ___(_) __ _ _   _ _ __ ___  ___
            # | |_  | |/ _` | | | | '__/ _ \/ __|
            # |  _| | | (_| | |_| | | |  __/\__ \
            # |_|   |_|\__, |\__,_|_|  \___||___/
            #          |___/
            def annotate_max(x, y, ax=None):
                xmax = x[np.argmax(y)]
                ymax = y.max()
                text = f"current={xmax:.2f}, optical power={ymax:.2f}"
                if not ax:
                    ax = plt.gca()
                bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
                arrowprops = dict(
                    arrowstyle="->", connectionstyle="angle,angleA=0,angleB=60"
                )
                kw = dict(
                    xycoords="data",
                    textcoords="axes fraction",
                    arrowprops=arrowprops,
                    bbox=bbox_props,
                    ha="right",
                    va="top",
                )
                ax.annotate(text, xy=(xmax, ymax), xytext=(0.94, 0.96), **kw)
                return xmax

            def annotate_threshhold(x, y, ax=None):  # TODO
                first_der = np.gradient(y, x)
                second_der = np.gradient(first_der, x)

            fig = plt.figure(figsize=(20, 10))
            ax1 = fig.add_subplot(221)  # subplot for set current
            ax12 = ax1.twinx()

            ax2 = fig.add_subplot(222)  # subplot for measured current
            ax22 = ax2.twinx()

            ax3 = fig.add_subplot(223)  # subplot for power

            # Adding legend
            # ax1.legend(loc=0)
            # ax12.legend(loc=0)
            # ax2.legend(loc=0)
            # ax22.legend(loc=0)

            ax1.grid(which="both")  # adding grid
            ax2.grid(which="both")  # adding grid
            ax3.grid(which="both")  # adding grid

            # Adding labels
            ax1.set_xlabel("Current set, mA")
            ax2.set_xlabel("Current measured, mA")
            ax3.set_xlabel("Current measured, mA")

            ax1.set_ylabel("Output power, mW", color="blue")
            ax2.set_ylabel("Output power, mW", color="blue")
            ax3.set_ylabel("Power, mW")

            ax12.set_ylabel("Voltage, V", color="red")
            ax22.set_ylabel("Voltage, V", color="red")

            ax1.minorticks_on()
            ax2.minorticks_on()
            ax12.minorticks_on()
            ax22.minorticks_on()
            ax3.minorticks_on()

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
                ax1.plot(seti, l, "-", label="Output power, mW")
                ax2.plot(i, l, "-", label="Output power, mW")
                ax3.plot(i, l, "-", label="Output power, mW")
                # Creating Twin axes
                ax12.plot(seti, v, "-r", label="Voltage, V")
                ax22.plot(i, v, "-r", label="Voltage, V")
                ax3.plot(i, p, "-b", label="Power consumption, mW")
                ax3.legend(loc=0)
                # annotate maximum output power
                annotate_max(seti, l, ax=ax1)
                annotate_max(i, l, ax=ax2)

            def buildplt_tosave(dataframe=df):
                # Creating figure
                fig = plt.figure(figsize=(20, 10))
                ax = fig.add_subplot(111)
                ax2 = ax.twinx()

                plt.title(
                    waferid
                    + " "
                    + wavelength
                    + " nm "
                    + coordinates
                    + " "
                    + temperature
                    + " °C"
                    # + " "
                    # + powermeter
                )  # Adding title

                # Adding legend
                # ax.legend(loc=0)
                # ax2.legend(loc=0)

                ax.grid(which="both")  # adding grid

                # Adding labels
                ax.set_xlabel("Current, mA")
                ax.set_ylabel("Output power, mW", color="blue")
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
                ax.plot(i, l, "-", label="Output power, mW")
                # Creating Twin axes for dataset_1
                ax2.plot(i, v, "-r", label="Voltage, V")

                ax.minorticks_on()
                ax2.minorticks_on()

                # annotate maximum output power
                i_rollover = annotate_max(i, l, ax=ax)
                return i_rollover

            #                 (_)       | |
            #  _ __ ___   __ _ _ _ __   | | ___   ___  _ __
            # | '_ ` _ \ / _` | | '_ \  | |/ _ \ / _ \| '_ \
            # | | | | | | (_| | | | | | | | (_) | (_) | |_) |
            # |_| |_| |_|\__,_|_|_| |_| |_|\___/ \___/| .__/
            #                                         | |
            #                                         |_|
            for i in current_list:
                # Outputs i Ampere immediately
                Keysight_B2901A.write(":SOUR:CURR " + str(i))
                time.sleep(0.03)
                # measure Voltage, V
                voltage = float(Keysight_B2901A.query("MEAS:VOLT?"))
                # measure Current, A
                current = float(Keysight_B2901A.query("MEAS:CURR?"))

                # measure output power
                if pm100_toggle:
                    # measure output power, W
                    output_power = float(PM100USB.query("measure:power?"))
                elif keysight_8163B_toggle:
                    # measure output power, W
                    output_power = float(Keysight_8163B.query("FETC1:CHAN2:POW?"))

                output_power *= 1000
                if output_power > max_output_power:  # track max power
                    max_output_power = output_power

                # add current, measured current, voltage, power, power conlumption to the DataFrame
                df.loc[len(df)] = [
                    i * 1000,
                    current * 1000,
                    voltage,
                    None,
                    voltage * current * 1000,
                ]

                # add power data if pm100toggle is set to True
                if pm100_toggle or keysight_8163B_toggle:
                    df.iloc[-1]["Output power, mW"] = output_power

                # print data to the terminal
                if voltage * current == 0 or output_power >= (voltage * current * 1000):
                    print(
                        f"{i*1000:3.2f} mA: {current*1000:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, 0 %"
                    )
                else:
                    print(
                        f"{i*1000:3.2f} mA: {current*1000:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, {output_power/(voltage*current*10):8.2f} %"
                    )

                # breaking conditions
                if i > current_limit1 / 1000:  # if current is more then limit1 mA
                    if (
                        output_power <= max_output_power * beyond_rollover_stop_cond
                        or output_power <= 0.01
                    ):  # check conditions to stop the measurements
                        break  # break the loop
                if max_output_power <= 0.5:
                    if i > current_limit2 / 1000:  # if current is more then limit2 mA
                        break  # break the loop

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
            dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/liv"
            filepath = (
                f"data/{waferid}-{wavelength}nm/{coordinates}/liv/"
                + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}c-{powermeter}-{timestr}"
            )

            if not os.path.exists(dirpath):  # make directories
                os.makedirs(dirpath)

            df.to_csv(filepath + ".csv")  # save DataFrame to csv file

            # save figures
            buildplt_all()
            plt.savefig(filepath + "-all")  # save figure
            i_rollover = buildplt_tosave()
            plt.savefig(filepath + f"_Iro={i_rollover:.2f}")  # save figure
            # plt.show()
            plt.close("all")
            # show figure
            image = mpimg.imread(filepath + "-all" + ".png")
            plt.imshow(image)
            plt.show()

        #  / _ \/ ___|  / \
        # | | | \___ \ / _ \
        # | |_| |___) / ___ \
        #  \___/|____/_/   \_\
        elif osa:
            pass


# Run main when the script is run by passing it as a command to the Python interpreter (just a good practice)
if __name__ == "__main__":
    main()
