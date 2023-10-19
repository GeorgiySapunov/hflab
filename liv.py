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


def measure_liv(
    waferid,
    wavelength,
    coordinates,
    temperature,
    Keysight_B2901A=None,
    PM100USB=None,
    Keysight_8163B=None,
    k_port=None,
    # settings
    Keysight_B2901A_address=None,
    Thorlabs_PM100USB_address=None,
    Keysight_8163B_address=None,
    YOKOGAWA_AQ6370D_address=None,
    ATT_A160CMI_address=None,
    current_increment_LIV=0.01,
    max_current=50,
    beyond_rollover_stop_cond=0.9,
    current_limit1=4,
    current_limit2=10,
    temperature_limit=110,
    osa_span=30,
    current_increment_OSA=0.3,
    spectra_dpi=100,
):
    current_list = [
        i / 10**5
        for i in range(0, max_current * 10**4, int(current_increment_LIV * 100))
    ]

    pm100_toggle = False
    keysight_8163B_toggle = False
    if PM100USB:
        pm100_toggle = True
        powermeter = "PM100USB"
    if Keysight_8163B:
        keysight_8163B_toggle = True
        powermeter = "Keysight_8163B_port" + str(k_port)

    dirpath = f"data/{waferid}-{wavelength}nm/{coordinates}/"

    print(f"Measuring LIV using {powermeter}")
    # initiate pandas Data Frame
    iv = pd.DataFrame(
        columns=[
            "Current set, mA",
            "Current, mA",
            "Voltage, V",
            "Output power, mW",
            "Power consumption, mW",
            "Power conversion efficiency, %",
        ]
    )

    # The initial settings are applied by the *RST command
    Keysight_B2901A.write("*RST")

    if pm100_toggle:
        PM100USB.write(f"sense:corr:wav {wavelength}")  # set wavelength
        PM100USB.write("power:dc:unit W")  # set power units
    elif keysight_8163B_toggle:
        Keysight_8163B.write("*RST")  # reset
        # set wavelength
        Keysight_8163B.write(f"SENS1:CHAN{k_port}:POW:WAV {wavelength}nm")
        # set power units
        Keysight_8163B.write(f"SENS1:CHAN{k_port}:POW:UNIT W")

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
    def annotate_max_L(x, y, ax=None):
        xmax = x[np.argmax(y)]
        ymax = y.max()
        text = f"I_ro={xmax:.2f} mA, optical power={ymax:.2f} mW"
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90")
        kw = dict(
            xycoords="data",
            textcoords="axes fraction",
            arrowprops=arrowprops,
            bbox=bbox_props,
            ha="left",
            va="top",
        )
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.2, 0.99), **kw)
        return xmax

    def annotate_max_ef(x, y, threshold=0, ax=None):  # TODO
        thresholdx = int(threshold / current_increment_LIV)
        xmax = x[np.argmax(y[thresholdx:]) + thresholdx]  # TODO
        ymax = y[thresholdx:].max()  # TODO
        text = f"I={xmax:.2f} mA, PCE={ymax:.2f}, %"
        if not ax:
            ax = plt.gca()
        bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
        arrowprops = dict(arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90")
        kw = dict(
            xycoords="data",
            textcoords="axes fraction",
            arrowprops=arrowprops,
            bbox=bbox_props,
            ha="left",
            va="top",
        )
        ax.annotate(text, xy=(xmax, ymax), xytext=(0.4, 0.99), **kw)
        return xmax

    def annotate_threshold(x, y, ax=None):  # TODO
        first_der = np.gradient(y, x)
        second_der = np.gradient(first_der, x)
        # print(second_der)
        print(second_der.max())
        if second_der.max() >= 10:
            x_threshold = x[np.argmax(second_der >= 10)]  # decision level
            y_threshold = y[np.argmax(second_der >= 10)]

            text = f"I_th={x_threshold:.2f} mA"
            if not ax:
                ax = plt.gca()
            bbox_props = dict(boxstyle="square,pad=0.3", fc="w", ec="k", lw=0.72)
            arrowprops = dict(
                arrowstyle="->", connectionstyle="angle,angleA=0,angleB=90"
            )
            kw = dict(
                xycoords="data",
                textcoords="axes fraction",
                arrowprops=arrowprops,
                bbox=bbox_props,
                ha="left",
                va="top",
            )
            ax.annotate(text, xy=(x_threshold, y_threshold), xytext=(0.01, 0.99), **kw)
        else:
            x_threshold = 0.0
        return x_threshold

    fig = plt.figure(figsize=(20, 10))
    fig.suptitle(f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{powermeter}")
    ax1 = fig.add_subplot(221)  # subplot for set current
    ax12 = ax1.twinx()

    ax2 = fig.add_subplot(222)  # subplot for measured current
    ax22 = ax2.twinx()

    ax3 = fig.add_subplot(223)  # subplot for power
    ax32 = ax3.twinx()

    ax4 = fig.add_subplot(224)  # subplot for power
    ax42 = ax4.twinx()

    # Adding legend
    # ax1.legend(loc=0)
    # ax12.legend(loc=0)
    # ax2.legend(loc=0)
    # ax22.legend(loc=0)

    ax1.grid(which="both")  # adding grid
    ax2.grid(which="both")  # adding grid
    ax3.grid(which="both")  # adding grid
    ax4.grid(which="both")  # adding grid

    # Adding labels
    ax1.set_xlabel("Current set, mA")
    ax2.set_xlabel("Current measured, mA")
    ax3.set_xlabel("Current measured, mA")
    ax4.set_xlabel("Current measured, mA")

    ax1.set_ylabel("Output power, mW", color="blue")
    ax2.set_ylabel("Output power, mW", color="blue")
    ax3.set_ylabel("Power, mW")
    ax4.set_ylabel("dP_out/dI, mW/mA", color="blue")

    ax12.set_ylabel("Voltage, V", color="red")
    ax22.set_ylabel("Voltage, V", color="red")
    ax32.set_ylabel("Power conversion efficiency, %", color="red")
    ax42.set_ylabel("dV/dI, V/mA", color="red")

    ax1.minorticks_on()
    ax2.minorticks_on()
    ax12.minorticks_on()
    ax22.minorticks_on()
    ax3.minorticks_on()
    ax32.minorticks_on()
    ax4.minorticks_on()
    ax42.minorticks_on()

    # functions to build graphs
    def buildplt_all(dataframe=iv):
        # select columns in the Data Frame
        seti = dataframe["Current set, mA"]
        i = dataframe["Current, mA"]
        v = dataframe["Voltage, V"]
        l = dataframe["Output power, mW"]
        p = dataframe["Power consumption, mW"]
        e = dataframe["Power conversion efficiency, %"]

        # Plotting
        lns11 = ax1.plot(seti, l, "-", label="Output power, mW")
        lns21 = ax2.plot(i, l, "-", label="Output power, mW")
        lns41 = ax4.plot(i, np.gradient(l, i), "-", label="dP_out/dI, mW/mA")
        # Creating Twin axes
        lns12 = ax12.plot(seti, v, "-r", label="Voltage, V")
        lns22 = ax22.plot(i, v, "-r", label="Voltage, V")
        lns42 = ax42.plot(i, np.gradient(v, i), "-r", label="dV/dI, V/mA")
        # annotate maximum output power
        annotate_max_L(seti, l, ax=ax1)
        annotate_max_L(i, l, ax=ax2)
        annotate_threshold(seti, l, ax=ax1)
        threshold = annotate_threshold(i, l, ax=ax2)

        lns31 = ax3.plot(i, l, "-", label="Output power, mW")
        lns32 = ax3.plot(i, p, "-b", label="Power consumption, mW")
        lns33 = ax32.plot(
            i[threshold:], e[threshold:], "-r", label="Power conversion efficiency, %"
        )
        annotate_max_ef(i, e, threshold=threshold, ax=ax32)

        # legends
        lns = lns11 + lns12
        labs = [l.get_label() for l in lns]
        ax1.legend(lns, labs, loc=4)
        lns = lns21 + lns22
        labs = [l.get_label() for l in lns]
        ax2.legend(lns, labs, loc=4)
        lns = lns31 + lns32 + lns33
        labs = [l.get_label() for l in lns]
        ax3.legend(lns, labs, loc=7)
        lns = lns41 + lns42
        labs = [l.get_label() for l in lns]
        ax4.legend(lns, labs, loc=4)

        # Setting limits
        ax1.set_ylim(bottom=0)  # Power
        ax1.set_xlim(left=0)  # Current set
        ax12.set_ylim(bottom=0)  # Voltage
        ax2.set_ylim(bottom=0)  # Power
        ax2.set_xlim(left=0)  # Current
        ax22.set_ylim(bottom=0)  # Voltage
        ax3.set_ylim(bottom=0)  # Voltage
        ax3.set_xlim(left=0)  # Current
        ax32.set_ylim(0, 100)  # Power conversion efficiency

    def buildplt_tosave(dataframe=iv):
        # Creating figure
        fig = plt.figure(figsize=(20, 10))
        ax = fig.add_subplot(111)
        ax2 = ax.twinx()

        plt.title(
            str(waferid)
            + " "
            + str(wavelength)
            + " nm "
            + str(coordinates)
            + " "
            + str(temperature)
            + " °C"
            # + " "
            # + powermeter
        )  # Adding title

        ax.grid(which="both")  # adding grid

        # Adding labels
        ax.set_xlabel("Current, mA")
        ax.set_ylabel("Output power, mW", color="blue")
        ax2.set_ylabel("Voltage, V", color="red")

        # select columns in the Data Frame
        seti = dataframe["Current set, mA"]
        i = dataframe["Current, mA"]
        # i = seti # uncoment this line to use "Current set, mA" column to plot graphs
        v = dataframe["Voltage, V"]
        l = dataframe["Output power, mW"]

        # Plotting dataset_2
        lns1 = ax.plot(i, l, "-", label="Output power, mW")
        # Creating Twin axes for dataset_1
        lns2 = ax2.plot(i, v, "-r", label="Voltage, V")

        # Setting Y limits
        ax.set_ylim(bottom=0)  # Power
        ax.set_xlim(left=0)  # Current
        ax2.set_ylim(bottom=0)  # Voltage

        # Adding legend
        # ax.legend(loc=0)
        # ax2.legend(loc=0)

        ax.minorticks_on()
        ax2.minorticks_on()

        # annotate maximum output power
        i_rollover = annotate_max_L(i, l, ax=ax)
        i_threshold = annotate_threshold(i, l, ax=ax)
        # legend
        lns = lns1 + lns2
        labs = [l.get_label() for l in lns]
        ax.legend(lns, labs, loc=4)
        return i_threshold, i_rollover

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
            output_power = float(Keysight_8163B.query(f"FETC1:CHAN{k_port}:POW?"))

        output_power *= 1000
        if output_power > max_output_power:  # track max power
            max_output_power = output_power

        # add current, measured current, voltage, output power, power consumption, power conversion efficiency to the DataFrame
        iv.loc[len(iv)] = [
            i * 1000,
            current * 1000,
            voltage,
            None,
            voltage * current * 1000,
            0,
        ]

        # add power data if pm100toggle is set to True
        if pm100_toggle or keysight_8163B_toggle:
            iv.iloc[-1]["Output power, mW"] = output_power

        # print data to the terminal
        if voltage * current == 0 or output_power >= (voltage * current * 1000):
            print(
                f"{i*1000:3.2f} mA: {current*1000:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, 0 %"
            )
        else:
            print(
                f"{i*1000:3.2f} mA: {current*1000:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, {output_power/(voltage*current*10):8.2f} %"
            )
            iv.iloc[-1]["Power conversion efficiency, %"] = output_power / (
                voltage * current * 10
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
        Keysight_B2901A.write(f":SOUR:CURR {str(i)}")  # Outputs i A immediately
        print(f"Current set: {i*1000:3.1f} mA")
        time.sleep(0.01)  # 0.01 sec for a step, 1 sec for 10 mA

    # Measurement is stopped by the :OUTP OFF command.
    Keysight_B2901A.write(":OUTP OFF")
    Keysight_B2901A.write(f":SOUR:CURR 0.001")

    timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
    filepath = (
        dirpath
        + "LIV/"
        + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{timestr}-{powermeter}"
    )

    if not os.path.exists(dirpath + "LIV"):  # make directories
        os.makedirs(dirpath + "LIV")

    iv.to_csv(filepath + ".csv")  # save DataFrame to csv file

    # save figures
    buildplt_all()
    plt.savefig(filepath + "-all.png", dpi=300)  # save figure
    i_threshold, i_rollover = buildplt_tosave()
    plt.savefig(
        filepath + f"_Ith={i_threshold:.2f}_Iro={i_rollover:.2f}.png", dpi=300
    )  # save figure
    # plt.show()
    plt.close("all")
    # show figure
    # image = mpimg.imread(filepath + "-all.png")
    # plt.imshow(image)
    # plt.show()

    return filepath
