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

# from settings import settings


# parameters
# equipment = sys.argv[1]
# waferid = sys.argv[2]
# wavelength = sys.argv[3]
# coordinates = sys.argv[4]
# temperature = sys.argv[5]


#  _____ _
# |  ___(_) __ _ _   _ _ __ ___  ___
# | |_  | |/ _` | | | | '__/ _ \/ __|
# |  _| | | (_| | |_| | | |  __/\__ \
# |_|   |_|\__, |\__,_|_|  \___||___/
#          |___/


def annotate_threshold(x, y, ax=None):  # TODO
    decision_level = 2
    first_der = np.gradient(y, x)
    second_der = np.gradient(first_der, x)
    # print(f"max second der = {second_der.max()}")
    if second_der.max() >= decision_level:
        x_threshold = x[np.argmax(second_der >= decision_level)]
        y_threshold = y[np.argmax(second_der >= decision_level)]

        text = f"I_th={x_threshold:.2f} mA"
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
        ax.annotate(text, xy=(x_threshold, y_threshold), xytext=(0.01, 0.99), **kw)
    else:
        x_threshold = 0.0
    return x_threshold


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
    ax.annotate(text, xy=(xmax, ymax), xytext=(0.3, 0.99), **kw)
    return xmax, ymax


def annotate_max_ef(x, y, threshold=0, ax=None):
    # thresholdx = int(threshold / current_increment_LIV)  # TODO
    thresholdx = np.argmax(x == threshold)
    xmax = x[np.argmax(y[thresholdx:]) + thresholdx]
    ymax = y[thresholdx:].max()
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


# functions to build graphs
def buildplt_all(
    dataframe,
    waferid,
    wavelength,
    coordinates,
    temperature,
    powermeter,
):
    fig = plt.figure(figsize=(1.8 * 11.69, 1.8 * 8.27))
    fig.suptitle(f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{powermeter}")
    ax1 = fig.add_subplot(221)  # subplot for set current
    ax12 = ax1.twinx()
    ax2 = fig.add_subplot(222)  # subplot for L/P
    ax22 = ax2.twinx()  # derivative
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
    ax1.set_ylabel("Output power, mW", color="blue")
    ax12.set_ylabel("Voltage, V", color="red")

    ax2.set_xlabel("Power consumption, mW")
    ax2.set_ylabel("Output power, mW", color="blue")
    ax22.set_ylabel("dP_out/dP_in", color="red")
    # ax22.set_ylabel("P_out/P_in", color="red")

    ax3.set_xlabel("Current measured, mA")
    ax3.set_ylabel("Power, mW")
    ax32.set_ylabel("Power conversion efficiency, %", color="red")

    ax4.set_xlabel("Current measured, mA")
    ax4.set_ylabel("dP_out/dI, mW/mA", color="blue")
    ax42.set_ylabel("dV/dI, V/mA", color="red")

    ax1.minorticks_on()
    ax2.minorticks_on()
    ax12.minorticks_on()
    ax22.minorticks_on()
    ax3.minorticks_on()
    ax32.minorticks_on()
    ax4.minorticks_on()
    ax42.minorticks_on()

    # select columns in the Data Frame
    seti = dataframe["Current set, mA"]
    i = dataframe["Current, mA"]
    v = dataframe["Voltage, V"]
    l = dataframe["Output power, mW"]
    p = dataframe["Power consumption, mW"]
    if "Power conversion efficiency, %" in dataframe.columns:
        e = dataframe["Power conversion efficiency, %"]
    else:
        dataframe["Power conversion efficiency, %"] = 100 * l / (v * i)
        e = dataframe["Power conversion efficiency, %"]

    # Plotting
    lns11 = ax1.plot(i, l, "-", label="Output power, mW")
    lns12 = ax12.plot(i, v, "-r", label="Voltage, V")
    # annotate maximum output power
    annotate_max_L(i, l, ax=ax1)
    # annotate_max_L(i, l, ax=ax2)
    threshold = annotate_threshold(i, l, ax=ax1)
    print(f"threshold {threshold:.2f}")

    # above_threshold = threshold + 0.02
    # zbul = seti >= above_threshold
    # z = (l.loc[zbul] - l.loc[zbul].iloc[0]) / (p.loc[zbul] - p.loc[zbul].iloc[0])
    z = np.gradient(l.loc[i >= threshold], p.loc[i >= threshold])

    lns21 = ax2.plot(p, l, "-", label="Output power, mW")
    lns22 = ax22.plot(
        p.loc[i >= threshold],
        # p.loc[zbul],
        z,
        "-r",
        # label="(P_out-P_out(@threshold))/(P_in-P_in(@threshold))",
        label="dP_out/dP_in",
    )

    lns31 = ax3.plot(i, l, "-", label="Output power, mW")
    lns32 = ax3.plot(i, p, "-b", label="Power consumption, mW")
    lns33 = ax32.plot(
        i.loc[i >= threshold],
        e.loc[i >= threshold],
        "-r",
        label="Power conversion efficiency, %",
    )
    annotate_max_ef(i, e, threshold=threshold, ax=ax32)

    lns41 = ax4.plot(i, np.gradient(l, i), "-", label="dP_out/dI, mW/mA")
    lns42 = ax42.plot(
        i.loc[i >= threshold],
        np.gradient(v.loc[i >= threshold], i.loc[i >= threshold]),
        "-r",
        label="dV/dI, V/mA",
    )

    # Setting limits
    ax1.set_xlim(left=0)  # Current
    ax1.set_ylim(bottom=0)  # Power
    ax12.set_ylim(bottom=0)  # Voltage

    ax2.set_xlim(left=0)  # Power input
    ax2.set_ylim(bottom=0)  # Power output
    ax22.set_ylim(0, 1)  # Derivative

    ax3.set_xlim(left=0)  # Current
    ax3.set_ylim(bottom=0)  # Voltage
    ax32.set_ylim(0, 100)  # Power conversion efficiency

    ax4.set_xlim(left=0)  # Current
    ax4.set_ylim(bottom=0)  # Current
    # ax42.set_ylim((0, 0.4))  # Current

    # legends
    lns = lns11 + lns12
    labs = [l.get_label() for l in lns]
    ax1.legend(lns, labs, loc=4)

    lns = lns21 + lns22
    labs = [l.get_label() for l in lns]
    ax2.legend(lns, labs, loc=7)

    lns = lns31 + lns32 + lns33
    labs = [l.get_label() for l in lns]
    ax3.legend(lns, labs, loc=7, prop={"size": 8})

    lns = lns41 + lns42
    labs = [l.get_label() for l in lns]
    ax4.legend(lns, labs, loc=0)


def buildplt_tosave(
    dataframe, waferid, wavelength, coordinates, temperature, powermeter
):
    # Creating figure
    fig = plt.figure(figsize=(11.69, 8.27))
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
        + " °C "
        # + " "
        + str(powermeter)
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
    i_rollover, l_rollover = annotate_max_L(i, l, ax=ax)
    i_threshold = annotate_threshold(i, l, ax=ax)
    # legend
    lns = lns1 + lns2
    labs = [l.get_label() for l in lns]
    ax.legend(lns, labs, loc=4)
    return i_threshold, i_rollover


# ___  ___                                _     _____ _   _
# |  \/  |                               | |   |_   _| | | |
# | .  . | ___  __ _ ___ _   _ _ __ ___  | |     | | | | | |
# | |\/| |/ _ \/ _` / __| | | | '__/ _ \ | |     | | | | | |
# | |  | |  __/ (_| \__ \ |_| | | |  __/ | |_____| |_\ \_/ /
# \_|  |_/\___|\__,_|___/\__,_|_|  \___| \_____/\___/ \___/
def measure_liv(
    waferid,
    wavelength,
    coordinates,
    temperature,
    Keysight_B2901A=None,
    PM100USB=None,
    Keysight_8163B=None,
    k_port=None,
):
    config = ConfigParser()
    config.read("config.ini")
    # instruments_config = config["INSTRUMENTS"]
    liv_config = config["LIV"]
    # osa_config = config["OSA"]
    # other_config = config["OTHER"]
    current_increment_LIV = float(liv_config["current_increment_LIV"])
    max_current = float(liv_config["max_current"])
    beyond_rollover_stop_cond = float(liv_config["beyond_rollover_stop_cond"])
    current_limit1 = float(liv_config["current_limit1"])
    current_limit2 = float(liv_config["current_limit2"])

    current_list = [0.0]
    round_to = max(0, int(np.ceil(np.log10(1 / current_increment_LIV))))
    while current_list[-1] <= max_current - current_increment_LIV:
        current_list.append(round(current_list[-1] + current_increment_LIV, round_to))
    # current_list = np.arange(
    #     0,
    #     max_current + current_increment_LIV,
    #     current_increment_LIV,
    #     dtype=np.float64,
    # )  # mA
    # round_to = max(0, int(np.ceil(np.log10(1 / current_increment_LIV))))
    # current_list = np.array([round(i, round_to) for i in current_list])

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
        sleep_time = 0
    elif keysight_8163B_toggle:
        Keysight_8163B.write("*RST")  # reset
        # set wavelength
        Keysight_8163B.write(f"SENS1:CHAN{k_port}:POW:WAV {wavelength}nm")
        # set power units
        Keysight_8163B.write(f"SENS1:CHAN{k_port}:POW:UNIT W")
        sleep_time = 0.2

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
    warnings = []
    alarm = False

    #                 (_)       | |
    #  _ __ ___   __ _ _ _ __   | | ___   ___  _ __
    # | '_ ` _ \ / _` | | '_ \  | |/ _ \ / _ \| '_ \
    # | | | | | | (_| | | | | | | | (_) | (_) | |_) |
    # |_| |_| |_|\__,_|_|_| |_| |_|\___/ \___/| .__/
    #                                         | |
    #                                         |_|
    for current_set in current_list:
        # Outputs {current_set} mA immediately
        Keysight_B2901A.write(":SOUR:CURR " + str(current_set / 1000))
        time.sleep(sleep_time)
        # measure Voltage, V
        voltage = float(Keysight_B2901A.query("MEAS:VOLT?"))
        # measure Current, A
        current_measured = float(Keysight_B2901A.query("MEAS:CURR?")) * 1000  # mA

        # measure output power
        if pm100_toggle:
            # measure output power, mW
            output_power = float(PM100USB.query("measure:power?")) * 1000
        elif keysight_8163B_toggle:
            # measure output power, mW
            output_power = (
                float(Keysight_8163B.query(f"FETC1:CHAN{k_port}:POW?")) * 1000
            )

        if output_power > max_output_power:  # track max power
            max_output_power = output_power

        # add current, measured current, voltage, output power, power consumption, power conversion efficiency to the DataFrame
        iv.loc[len(iv)] = [
            current_set,
            current_measured,
            voltage,
            None,
            voltage * current_measured,
            0,
        ]

        # add power data if pm100toggle is set to True
        if pm100_toggle or keysight_8163B_toggle:
            iv.iloc[-1]["Output power, mW"] = output_power

        # print data to the terminal
        if voltage * current_measured == 0 or output_power >= (
            voltage * current_measured
        ):
            print(
                f"{current_set:3.2f} mA: {current_measured:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, 0 %"
            )
        else:
            print(
                f"{current_set:3.2f} mA: {current_measured:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, {(100*output_power)/(voltage*current_measured):8.2f} %"
            )
            iv.iloc[-1]["Power conversion efficiency, %"] = (100 * output_power) / (
                voltage * current_measured
            )

        # deal with set/measured current mismatch
        current_error = abs(current_set - current_measured)
        if np.float64(round(current_measured, round_to)) != np.float64(
            round(current_set, round_to)
        ):
            warnings.append(
                f"Current set={current_set} mA, current measured={current_measured} mA"
            )
            print(
                f"WARNING! Current set is {current_set} mA, while current measured is {current_measured} mA"
            )

        if current_error >= 0.03:
            alarm = True
            print(
                f"ALARM! Current set is {current_set}, while current measured is {current_measured}\tBreaking the measurements!"
            )
            break  # break the loop

        # breaking conditions
        if current_set > current_limit1:  # if current is more then limit1 mA
            if (
                output_power <= max_output_power * beyond_rollover_stop_cond
                or output_power <= 0.01
            ):  # check conditions to stop the measurements
                if output_power <= 0.01:
                    print(
                        f"Current reached {current_limit1} mA, but the output_power is less then 0.01 mW\tbreaking the loop"
                    )
                    alarm = True
                break  # break the loop
        if max_output_power <= 0.5:
            if current_set > current_limit2:  # if current is more then limit2 mA
                print(
                    f"Current reached {current_limit2} mA, but the output_power is less then 0.5 mW\tbreaking the loop"
                )
                alarm = True
                break  # break the loop

    # slowly decrease current
    current_measured = (
        float(Keysight_B2901A.query("MEAS:CURR?")) * 1000
    )  # measure current
    for current_set in np.arange(current_measured, 0, -0.1):
        Keysight_B2901A.write(
            f":SOUR:CURR {str(current_set/1000)}"
        )  # Outputs i A immediately
        print(f"Current set: {current_set:3.1f} mA")
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

    iv.to_csv(filepath + ".csv", index=False)  # save DataFrame to csv file

    # save figures
    buildplt_all(
        dataframe=iv,
        waferid=waferid,
        wavelength=wavelength,
        coordinates=coordinates,
        temperature=temperature,
        powermeter=powermeter,
    )
    plt.savefig(filepath + "-all.png", dpi=300)  # save figure
    i_threshold, i_rollover = buildplt_tosave(
        dataframe=iv,
        waferid=waferid,
        wavelength=wavelength,
        coordinates=coordinates,
        temperature=temperature,
        powermeter=powermeter,
    )
    plt.savefig(
        filepath + f"_Ith={i_threshold:.2f}_Iro={i_rollover:.2f}.png", dpi=300
    )  # save figure
    # plt.show()
    plt.close("all")
    # show figure
    # image = mpimg.imread(filepath + "-all.png")
    # plt.imshow(image)
    # plt.show()
    print(f"Warnings: {len(warnings)}")
    if warnings:
        print(*warnings, sep="\n")

    return filepath, alarm
