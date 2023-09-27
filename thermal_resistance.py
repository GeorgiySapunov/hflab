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
    temperatures = list(temperatures)
    temperatures.sort()
    print(f"temperature set from OSA:\n{temperatures}")

    df_Pdis_T = (
        pd.DataFrame(columns=[*temperatures])
        # .set_index("Dissipated power, mW") # TODO
        .rename_axis("Temperature, °C", axis=1)
    )

    dict_of_filenames_liv = {}
    dict_of_filenames_os = {}

    for i, temperature in enumerate(temperatures):
        print(f"{i+1}/{len(temperatures)} temperature {temperature} °C")
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
        print(f"LIV file {dict_of_filenames_liv}")

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
        print(f"OS file {dict_of_filenames_os}")

        # 3. get last peak lambdas and Pdis
        # read files
        osdf = pd.read_csv(dirpath + "OSA/" + osfile, index_col=0)
        livdf = pd.read_csv(dirpath + "LIV/" + livfile, index_col=0)
        columns = [
            col
            for col in osdf.columns.values.tolist()
            if col.startswith("Intensity at")
        ]
        # get a list of currents from "Intensity for 0.00 mA, dBm" columns
        currents = [
            float(col.split()[2]) for col in columns if col.startswith("Intensity at")
        ]
        # if Wavelength in meters
        if "Wavelength, nm" not in osdf.columns.values.tolist():
            osdf["Wavelength, nm"] = osdf["Wavelength, m"] * 10**9
        # find peaks
        # itterate by current
        for current, column in zip(currents, columns):
            # get Pdis
            row = livdf.loc[
                round(current / settings["current_increment_LIV"], 2)
            ]  # TODO ЧЗХ
            pdis = float(
                row["Current, mA"] * row["Voltage, V"] - row["Output power, mW"]
            )  # mW
            # get peak wavelength
            peak_indexes, _ = fp(x=osdf[column], height=-65, prominence=2, distance=10)
            if len(peak_indexes):
                last_peak_index = peak_indexes[-1]  # get last peak index
                # get last peak wl
                wl_peak = osdf["Wavelength, nm"].iloc[last_peak_index]
                print(
                    f"{temperature} °C\t{current} mA\tPdis: {pdis:.3f} mW\tpeak: {wl_peak:.3f} nm\t{column}"
                )
            else:
                wl_peak = None
                print(
                    f"{temperature} °C\t{current} mA\tPdis: {pdis:.3f} mW\tpeak: {wl_peak}\t\t{column}"
                )

            # 4. fill Pdis, temperature, lambda df
            df_Pdis_T.at[pdis, temperature] = wl_peak

            # 5. make a plot and save .png
            # Creating figure
            fig = plt.figure(figsize=(20, 10))
            # Plotting dataset
            ax = fig.add_subplot(111)
            ax.plot(
                osdf["Wavelength, nm"],
                osdf[column],
                "-",
                alpha=0.5,
                label=f"{temperature} °C, {current} mA, dissipated power: {pdis} mW, peak: {wl_peak} nm",
            )
            ax.scatter(
                osdf["Wavelength, nm"].iloc[peak_indexes],
                osdf[column].iloc[peak_indexes],
                alpha=0.5,
            )
            # Adding title
            plt.title(
                f"{waferid}-{wavelength}nm-{coordinates}, {temperature} °C, {current} mA"
            )
            # adding grid
            ax.grid(which="both")  # adding grid
            ax.minorticks_on()
            # Adding labels
            ax.set_xlabel("Wavelength, nm")
            ax.set_ylabel("Intensity, dBm")
            # Adding legend
            # ax.legend(loc=0, prop={"size": 4})
            # Setting limits (xlim1 and xlim2 will be also used in saved csv)
            # xlim1 = 930
            # xlim2 = 955
            # ax.set_xlim(xlim1, xlim2)
            ax.set_ylim(-80, -10)

            filepath_t = (
                dirpath
                + f"OSA/figures/temperature/{temperature}°C/"
                + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{current}mA"
            )
            filepath_i = (
                dirpath
                + f"OSA/figures/current/{current}mA/"
                + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{current}mA"
            )

            if not os.path.exists(dirpath + f"OSA/figures/current/{current}mA"):
                os.makedirs(dirpath + f"OSA/figures/current/{current}mA")
            if not os.path.exists(dirpath + f"OSA/figures/temperature/{temperature}°C"):
                os.makedirs(dirpath + f"OSA/figures/temperature/{temperature}°C")

            plt.savefig(filepath_t + ".png")
            plt.savefig(filepath_i + ".png")
            plt.close()

    # 6. sort and interpolate
    df_Pdis_T = df_Pdis_T.sort_index().astype("float64")
    df_Pdis_T.to_csv(
        dirpath
        + f"OSA/figures/"
        + f"{waferid}-{wavelength}nm-{coordinates}-withNaN.csv"
    )
    df_Pdis_T_int = df_Pdis_T.interpolate(method="values", limit_area="inside", axis=0)

    # 7. transpose the data to T, lambdas at diff Pdis
    # df_T_Pdis = df_Pdis_T_int.T

    # 8. plot lineplots
    # Creating figure
    fig = plt.figure(figsize=(20, 10))
    plt.suptitle(f"{waferid}-{wavelength}nm-{coordinates}")
    # Plotting dataset
    ax = fig.add_subplot(121)
    for col in df_Pdis_T_int.columns:
        ax.plot(
            df_Pdis_T_int.index,
            df_Pdis_T_int[col],
            "-",
            alpha=0.5,
            label=f"{col} °C",
        )
        ax.scatter(
            df_Pdis_T.index,
            df_Pdis_T[col],
            alpha=0.2,
        )
    # Adding title
    # adding grid
    ax.grid(which="both")  # adding grid
    ax.minorticks_on()
    # Adding labels
    ax.set_xlabel("Dissipated power, mW")
    ax.set_ylabel("Peak wavelength, nm")
    # Adding legend
    ax.legend(loc=0, prop={"size": 4})

    ax2 = fig.add_subplot(122)
    for col in df_Pdis_T_int.T.columns:
        ax2.plot(
            df_Pdis_T_int.T.index,
            df_Pdis_T_int.T[col],
            "-",
            alpha=0.5,
            # label=f"Dissipated power {col} mW",
        )
        ax2.scatter(
            df_Pdis_T.T.index,
            df_Pdis_T.T[col],
            alpha=0.2,
        )
    # Adding title
    # plt.title(f"{waferid}-{wavelength}nm-{coordinates}")
    # adding grid
    ax2.grid(which="both")  # adding grid
    ax2.minorticks_on()
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
    df_Pdis_T_int.to_csv(
        dirpath + f"OSA/figures/" + f"{waferid}-{wavelength}nm-{coordinates}.csv"
    )

    # 9. TODO plot heatmaps (T, lambda, Pdis and Pdis, lambda, T)
    # heatdf = df_Pdis_T_int.pivot(columns="")
    # 10. calculate dλ/dT at Pdis==0
    #

    # 11. calculate dT/dPdis


for i, directory in enumerate(sys.argv[1:]):
    num = len(sys.argv[1:])
    print(f"[{i+1}/{num}] {directory}")
    analyse(directory)
