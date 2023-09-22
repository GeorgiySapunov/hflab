#!/usr/bin/env python3

import sys
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from scipy.signal import find_peaks as fp

from settings import settings


def analyse(dirpath):
    if dirpath[-1] != "/":  # TODO check it
        dirpath = dirpath + "/"
    photodiode = "PM100USB"

    # get filenames for temperature set
    walk = list(os.walk(dirpath + "OSA"))
    string_for_re = ".*\\-OS.csv"
    r = re.compile(string_for_re)
    files = walk[0][2]
    matched_os_files = list(filter(r.match, files))
    waferid_wavelength, coordinates = os.path.normpath(dirpath).split("/")[-2:]
    waferid, wavelength_withnm = waferid_wavelength.split("-")
    wavelength = wavelength_withnm[:-2]

    # get temperature set
    temperatures = set()
    for file in matched_os_files:
        file_name_parser = file.split("-")
        r2 = re.compile(".*°C")
        temperature = list(filter(r2.match, file_name_parser))[0]
        temperature = float(temperature.removesuffix("°C"))
        temperatures.add(float(temperature))
    print(f"temperature set from OSA:\n{temperatures}")

    df_Pdis_T = (
        pd.DataFrame(columns=[*temperatures])
        .set_index("Dissipated power, mW")
        .rename_axis("Temperature, °C", axis=1)
    )

    dict_of_filenames_liv = {}
    dict_of_filenames_os = {}

    for temperature in temperatures:
        # 1. take liv files
        walk = list(os.walk(dirpath + "LIV"))
        string_for_re = (
            f"{waferid_wavelength}-{coordinates}-{temperature}°C-.*-{photodiode}\\.csv"
        )
        r = re.compile(string_for_re)
        files = walk[0][2]
        matched_files = list(filter(r.match, files))
        matched_files.sort(reverse=True)
        livfile = matched_files[0]
        dict_of_filenames_liv[temperature] = livfile
        print(dict_of_filenames_liv)

        # 2. take osa file
        walk = list(os.walk(dirpath + "OSA"))
        string_for_re = (
            f"{waferid_wavelength}-{coordinates}-{temperature}°C-.*-OS\\.csv"
        )
        r = re.compile(string_for_re)
        files = walk[0][2]
        matched_files = list(filter(r.match, files))
        matched_files.sort(reverse=True)
        osfile = matched_files[0]
        dict_of_filenames_os[temperature] = osfile
        print(dict_of_filenames_os)

        # 3. get last peak lambdas and Pdis
        # read files
        osdf = pd.read_csv(dirpath + "OSA/" + osfile, index_col=0)
        livdf = pd.read_csv(dirpath + "LIV/" + livfile, index_col=0)
        columns = osdf.columns.values.tolist()
        # get a list of currents from "Intensity for 0.00 mA, dBm" columns
        currents = [float(col.split()[2]) for col in columns]
        # find peaks
        for current, column in zip(currents, columns):  # itterate by current
            # get Pdis
            row = livdf[livdf["Current set, mA"] == current]
            pdis = float(
                row["Current, mA"] * row["Voltage, V"] - row["Output power, mW"]
            )  # mW
            # get peak wavelength
            peak_indexes, _ = fp(x=osdf[column], height=-65, prominence=2, distance=10)
            if len(peak_indexes):
                last_peak_index = peak_indexes[-1]  # get last peak index
                wl_peak = osdf.index.values[last_peak_index]  # get last peak wl
            else:
                wl_peak = None
            print(
                f"{temperature} °C\t{current} mA\tPdis: {pdis} mW\tpeak: {wl_peak} nm\t{column}"
            )

            # TODO 4. make a plot and save .png
            # Creating figure
            fig = plt.figure()
            # Plotting dataset
            ax = fig.add_subplot(111)
            ax.plot(
                osdf.index,
                osdf[column],
                "-",
                alpha=0.5,
                label=f"{temperature} °C, {current} mA, dissipated power: {pdis} mW, peak: {wl_peak} nm",
            )
            ax.scatter(
                osdf.iloc[peak_indexes].index.values,
                osdf[column].iloc[peak_indexes],
                alpha=0.5,
            )
            # Adding title
            plt.title(
                f"{waferid}-{wavelength}nm-{coordinates}, {temperature} °C, {current} mA"
            )
            # adding grid
            ax.grid()
            # Adding labels
            ax.set_xlabel("Wavelength, nm")
            ax.set_ylabel("Intensity, dBm")
            # Adding legend
            ax.legend(loc=0, prop={"size": 4})
            # Setting limits (xlim1 and xlim2 will be also used in saved csv)
            # xlim1 = 930
            # xlim2 = 955
            # ax.set_xlim(xlim1, xlim2)
            ax.set_ylim(-75, -10)

            filepath_t = (
                dirpath
                + f"OSA/figures/temperature/{temperature}°C"
                + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{current}mA"
            )
            filepath_i = (
                dirpath
                + f"OSA/figures/current/{current}mA"
                + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{current}mA"
            )

            if not os.path.exists(dirpath + f"OSA/figures/current/{current}mA"):
                os.makedirs(dirpath + f"OSA/figures/current/{current}mA")
            if not os.path.exists(dirpath + f"OSA/figures/temperature/{temperature}°C"):
                os.makedirs(dirpath + f"OSA/figures/temperature/{temperature}°C")

            plt.savefig(filepath_t + ".png")
            plt.savefig(filepath_i + ".png")
            plt.close()

            # 5. fill Pdis, temperature, lambda df
            df_Pdis_T[temperature].loc[pdis] = wl_peak

    # 6. sort and interpolate
    df_Pdis_T.interpolate(method="linear", limit_area="inside")

    # 7. transpose the data to T, lambdas at diff Pdis
    # df_T_Pdis = df_Pdis_T.T

    # 8. TODO plot lineplots
    # Creating figure
    fig = plt.figure()
    # Plotting dataset
    ax = fig.add_subplot(121)
    for col in df_Pdis_T.columns:
        ax.plot(
            df_Pdis_T.index,
            df_Pdis_T[col],
            "-",
            alpha=0.5,
            label=f"{col} °C",
        )
    # Adding title
    plt.title(f"{waferid}-{wavelength}nm-{coordinates}")
    # adding grid
    ax.grid()
    # Adding labels
    ax.set_xlabel("Dissipated power, mW")
    ax.set_ylabel("Peak wavelength, nm")
    # Adding legend
    ax.legend(loc=0, prop={"size": 4})

    ax2 = fig.add_subplot(121)
    for col in df_Pdis_T.T.columns:
        ax2.plot(
            df_Pdis_T.T.index,
            df_Pdis_T.T[col],
            "-",
            alpha=0.5,
            # label=f"Dissipated power {col} mW",
        )
    # Adding title
    # plt.title(f"{waferid}-{wavelength}nm-{coordinates}")
    # adding grid
    ax2.grid()
    # Adding labels
    ax2.set_xlabel("Temperature, °C")
    ax2.set_ylabel("Peak wavelength, nm")
    # Adding legend
    # ax.legend(loc=0, prop={"size": 4})

    filepath = (
        dirpath + f"OSA/figures/" + f"{waferid}-{wavelength}nm-{coordinates}-lineplot"
    )

    if not os.path.exists(dirpath + f"OSA/figures/"):
        os.makedirs(dirpath + f"OSA/figures/")

    plt.savefig(filepath + ".png")
    plt.close()

    # 9. TODO plot heatmaps (T, lambda, Pdis and Pdis, lambda, T)
    # heatdf = df_Pdis_T.pivot(columns="")
    # 10. calculate dT/dPdis


for i, directory in enumerate(sys.argv[1:]):
    num = len(sys.argv[1:])
    print(f"[{i+1}/{num}] {directory}")
    df = analyse(directory)
    analyse(directory)
