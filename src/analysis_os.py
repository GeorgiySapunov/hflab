#!/usr/bin/env python3

import sys
import os
import re
import yaml
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import seaborn as sns
from scipy.signal import find_peaks as fp
from sklearn import linear_model
from itertools import cycle
from pathlib import Path


def find_liv_threshold(x, y, liv_threshold_decision_level=2):
    first_der = np.gradient(y, x)
    second_der = np.gradient(first_der, x)
    if second_der.max() >= liv_threshold_decision_level:
        x_threshold = x[np.argmax(second_der >= liv_threshold_decision_level)]
        text = f"I_th={x_threshold:.2f} mA"
    else:
        x_threshold = 0.0
    return float(x_threshold)


def find_liv_rollower(x, y):
    xmax = x[np.argmax(y)]
    ymax = y.max()
    text = f"I_ro={xmax:.2f} mA, optical power={ymax:.2f} mW"
    return float(xmax)


def peak_threshold_height(osdf_column):
    x = np.array(osdf_column.values)[np.where(np.asarray(osdf_column.values) > -80)]
    mask = np.abs((x - x.mean(0)) / x.std(0)) < 2
    x = x[np.where(mask)]
    peak_threshold_height = x.mean() + 2 * x.std() + 2
    return peak_threshold_height


def analyze_os_function(directory, settings=None):
    if isinstance(directory, str):
        directory = Path(directory)
    osa_directory = directory / "OSA"
    liv_directory = directory / "LIV"
    figure_directory = osa_directory / "figures"
    if settings is None:
        with open(Path("templates") / "os.yaml") as fh:
            settings = yaml.safe_load(fh)
    print(directory)
    print(settings)

    title = settings["title"]
    photodiode = settings["photodiode"]
    liv_threshold_decision_level = settings["liv_threshold_decision_level"]
    plot_optical_spectra = settings["plot_optical_spectra"]
    spectra_figsize = settings["spectra_figsize"]
    spectra_xlim = settings["spectra_xlim"]
    optical_spectra_dpi = int(settings["optical_spectra_dpi"])
    increment_current_heatmap = settings["increment_current_heatmap"]
    if not photodiode:
        photodiode = "PM100USB"

    # get files for temperature set
    matched_os_files = sorted(osa_directory.glob("*-OS.csv"))
    if matched_os_files:
        figure_directory.mkdir(exist_ok=True)
    else:
        print("Can't find optical spectra files, skipping")
        return
    waferid_wavelength, coordinates = list(directory.parts)[-2:]
    waferid, wavelength_withnm = waferid_wavelength.split("-")
    wavelength = wavelength_withnm.removesuffix("nm")
    native_title = f"{waferid}-{wavelength}nm-{coordinates}"
    if not title:
        title = native_title

    # get temperature set
    temperatures = set()
    for file in matched_os_files:
        file_name_parser = file.stem.split("-")
        r2 = re.compile(".*°C")
        temperature = list(filter(r2.match, file_name_parser))[0]
        temperature = float(temperature.removesuffix("°C"))
        temperatures.add(temperature)
    temperatures = list(temperatures)
    temperatures.sort()
    print(f"temperature set from OSA:\n{temperatures}")

    df_Pdis_T = (
        pd.DataFrame(columns=temperatures)
        .rename_axis("Dissipated power, mW", axis=0)
        .rename_axis("Temperature, °C", axis=1)
    )

    df_I_T = (
        pd.DataFrame(columns=[*temperatures])
        .rename_axis("Current set, mA", axis=0)
        .rename_axis("Temperature, °C", axis=1)
    )

    df_I_T_smsr = (
        pd.DataFrame(columns=[*temperatures])
        .rename_axis("Current set, mA", axis=0)
        .rename_axis("Temperature, °C", axis=1)
    )

    df_I_T_highest_peak = (
        pd.DataFrame(columns=[*temperatures])
        .rename_axis("Current set, mA", axis=0)
        .rename_axis("Temperature, °C", axis=1)
    )

    dict_of_filenames_liv = {}
    dict_of_filenames_os = {}

    # itterate by temperature
    for i, temperature in enumerate(temperatures):
        print(f"{i+1}/{len(temperatures)} temperature {temperature} °C")
        # 1. take liv files
        matched_files = list(
            liv_directory.glob(f"{native_title}-{temperature}°C-*-{photodiode}.csv")
        )
        matched_files.sort(reverse=True)
        livfile = matched_files[0]
        dict_of_filenames_liv[temperature] = livfile
        print(f"LIV file: {dict_of_filenames_liv[temperature].stem}")

        # 2. take osa file
        matched_files = list(
            osa_directory.glob(f"{native_title}-{temperature}°C-*-OS.csv")
        )
        matched_files.sort(reverse=True)
        osfile = matched_files[0]
        dict_of_filenames_os[temperature] = osfile
        print(f"Optical Spectra file: {dict_of_filenames_os[temperature].stem}")

        # 3. get last peak lambdas and Pdis
        # read files
        osdf = pd.read_csv(osfile)
        livdf = pd.read_csv(livfile)

        I_th = round(
            find_liv_threshold(
                livdf["Current set, mA"],
                livdf["Output power, mW"],
                liv_threshold_decision_level=liv_threshold_decision_level,
            ),
            2,
        )
        I_ro = round(
            find_liv_rollower(livdf["Current set, mA"], livdf["Output power, mW"]), 2
        )
        # get a list of currents
        columns = [
            col
            for col in osdf.columns.values.tolist()
            if col.startswith("Intensity at")
            if float(col.split()[2]) >= (I_th - 0.05)
            and float(col.split()[2]) <= (I_ro + 0.05)
        ]
        currents = [
            float(col.split()[2]) for col in columns if col.startswith("Intensity at")
        ]
        if i == 0:
            larg_current = livdf["Current set, mA"].iloc[-1]
        # adjust if Wavelength in meters
        if "Wavelength, nm" not in osdf.columns.values.tolist():
            osdf["Wavelength, nm"] = osdf["Wavelength, m"] * 10**9

        # find peaks
        # itterate by current
        for current, column in zip(currents, columns):
            # get Pdis
            current = round(current, 6)
            row = livdf.loc[np.round(livdf["Current set, mA"], 10) == current]
            pdis = float(
                (
                    row["Current set, mA"] * row["Voltage, V"] - row["Output power, mW"]
                ).iloc[0]
            )  # mW
            # find peaks wavelength
            peak_indexes, _ = fp(
                x=osdf[column],
                height=peak_threshold_height(osdf[column]),
                prominence=2,
                distance=1,
            )
            if len(peak_indexes):
                last_peak_index = peak_indexes[-1]  # get last peak index
                wl_peak = osdf["Wavelength, nm"].iloc[last_peak_index]
                print(
                    f"{temperature} °C\t{current} mA\tPdis: {pdis:.3f} mW\tpeak: {wl_peak:.3f} nm\t{column}"
                )
            else:
                last_peak_index = None
                wl_peak = None
                print(
                    f"{temperature} °C\t{current} mA\tPdis: {pdis:.3f} mW\tpeak: {wl_peak}\t\t{column}"
                )
            peak_indexes2, properties = fp(
                x=osdf[column], height=-80, prominence=2, distance=1
            )
            argsort_peaks = np.argsort(
                properties["peak_heights"]
            )  # TODO partial argsort should be better in terms of speed
            highest_peak_hight = properties["peak_heights"][argsort_peaks[-1]]
            highest_peak_index = peak_indexes2[argsort_peaks[-1]]
            second_highest_peak_hight = properties["peak_heights"][argsort_peaks[-2]]
            second_highest_peak_index = peak_indexes2[argsort_peaks[-2]]
            smsr = highest_peak_hight - second_highest_peak_hight

            # 4. fill Pdis, temperature, lambda df
            df_Pdis_T.at[pdis, temperature] = wl_peak
            df_I_T.at[current, temperature] = wl_peak
            df_I_T_smsr.at[current, temperature] = smsr
            df_I_T_highest_peak.at[current, temperature] = osdf["Wavelength, nm"].iloc[
                highest_peak_index
            ]

            # 5. make spectra plots and save .png
            if plot_optical_spectra:
                # Make spectra figures
                fig = plt.figure(figsize=spectra_figsize)
                ax = fig.add_subplot(111)
                # spectrum line
                ax.plot(
                    osdf["Wavelength, nm"],
                    osdf[column],
                    "-",
                    alpha=0.5,
                    label=f"{temperature} °C, {current} mA, dissipated power: {pdis} mW",
                )
                # peaks scatterplot
                ax.scatter(
                    osdf["Wavelength, nm"].iloc[peak_indexes],
                    osdf[column].iloc[peak_indexes],
                    alpha=0.5,
                )
                ax.scatter(
                    osdf["Wavelength, nm"].iloc[highest_peak_index],
                    highest_peak_hight,
                    alpha=0.5,
                    c="r",
                    label=f"the highest peak at {osdf['Wavelength, nm'].iloc[highest_peak_index]} nm, {highest_peak_hight} dBm, SMSR={smsr}",
                )
                ax.scatter(
                    osdf["Wavelength, nm"].iloc[second_highest_peak_index],
                    second_highest_peak_hight,
                    alpha=0.5,
                    c="g",
                    label=f"the second highest peak at {osdf['Wavelength, nm'].iloc[second_highest_peak_index]} nm, {second_highest_peak_hight} dBm",
                )
                if wl_peak:
                    ax.axvline(
                        x=wl_peak,
                        linestyle=":",
                        alpha=0.5,
                        label=f"far right peak at {wl_peak} nm",
                    )
                plt.title(f"{title}, {temperature} °C, {current} mA")
                ax.grid(which="both")
                ax.minorticks_on()
                ax.set_xlabel("Wavelength, nm")
                ax.set_ylabel("Intensity, dBm")
                ax.legend(loc=1)
                ax.set_ylim(-80, 10)
                ax.set_xlim(spectra_xlim)

                filestem = f"{native_title}-{temperature}°C-{current}mA"
                filepath_t = figure_directory / "temperature" / f"{temperature}°C"
                filepath_i = figure_directory / "current" / f"{current}mA"
                filepath_t.mkdir(parents=True, exist_ok=True)
                filepath_i.mkdir(parents=True, exist_ok=True)

                plt.savefig(filepath_t / (filestem + ".png"), dpi=optical_spectra_dpi)
                plt.savefig(filepath_i / (filestem + ".png"), dpi=optical_spectra_dpi)
                plt.close()

    # 6. sort and interpolate
    df_Pdis_T = df_Pdis_T.sort_index().astype("float64")
    df_Pdis_T_drop = df_Pdis_T.dropna()  # delete rows with empty cells
    df_Pdis_T_int = df_Pdis_T.interpolate(method="values", limit_area="inside", axis=0)
    df_Pdis_T.to_csv(
        figure_directory / f"{native_title}-withNaN.csv",
        index=False,
    )
    df_Pdis_T_int.to_csv(
        figure_directory / f"{native_title}.csv",
        index=False,
    )
    df_Pdis_T_int_drop = df_Pdis_T_int.dropna()  # delete rows with empty cells

    df_I_T.to_csv(
        figure_directory / f"{native_title}-Current-Temperature.csv",
        index=False,
    )

    df_I_T_smsr.to_csv(
        figure_directory / f"{native_title}-Current-Temperature-SMSR.csv",
        index=False,
    )

    df_I_T_highest_peak.to_csv(
        figure_directory / f"{native_title}-Current-Temperature-highestpeak.csv",
        index=False,
    )

    # 7. fill dλ/dP_dis and dλ/dT
    dldp = pd.DataFrame(columns=["Temperature, °C", "dλ/dP_dis", "intercept"])
    dldt = pd.DataFrame(columns=["Dissipated power, mW", "dλ/dT", "intercept"])
    dldi = pd.DataFrame(columns=["Temperature, °C", "dλ/dI", "intercept"])

    # 8.1 Current approximation
    fig = plt.figure(figsize=(11.69, 8.27))
    plt.suptitle(f"{title}\nλ(I) at different temperatures")
    ax1 = fig.add_subplot(111)  # λ(I) at different temperatures
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    # Creating figure
    for col_temp in df_I_T.columns:  # columns are temperatures
        color = next(colors)
        df_I_T_drop = df_I_T[col_temp].dropna()
        # linear approximation
        model = linear_model.LinearRegression()
        X = df_I_T_drop.index.values.reshape(-1, 1)
        y = df_I_T_drop  # column [col_temp]
        model.fit(X, y)
        slope = model.coef_[0]
        # save fit parameters to a DataFrame
        dldi.loc[len(dldi)] = [col_temp, slope, model.intercept_]

        ax1.plot(
            df_I_T.index,
            df_I_T[col_temp],
            "-",
            # marker="o",
            c=color,
            alpha=0.6,
            label=f"{col_temp} °C",
        )
        ax1.plot(
            df_I_T.index,
            model.predict(df_I_T.index.values.reshape(-1, 1)),
            "-.",
            c=color,
            alpha=0.2,
            label=f"fit {col_temp} °C, dλ/dI={slope:.3f}, intercept={model.intercept_:.3f}",
        )
    # adding grid
    ax1.grid(which="both")  # adding grid
    ax1.minorticks_on()
    # Adding labels
    ax1.set_xlabel("Current set, mA")
    ax1.set_ylabel("Peak wavelength, nm")
    # Adding legend
    ax1.legend(loc=0)
    ax1.set_xlim(left=0)

    # save files
    plt.savefig(figure_directory / f"{native_title}-Lambda-Current.png", dpi=300)
    plt.close()

    # 8.2 highest peak (current approximation)
    fig = plt.figure(figsize=(11.69, 8.27))
    plt.suptitle(f"{title}\nthe highest peak at different temperatures/currents")
    ax1 = fig.add_subplot(111)  # λ(I) at different temperatures
    colors = cycle(plt.rcParams["axes.prop_cycle"].by_key()["color"])
    # Creating figure
    for col_temp in df_I_T_highest_peak.columns:  # columns are temperatures
        color = next(colors)
        ax1.plot(
            df_I_T_highest_peak.index,
            df_I_T_highest_peak[col_temp],
            "-",
            # marker="o",
            alpha=0.3,
            color=color,
            label=f"{col_temp} °C",
        )
        ax1.plot(
            df_I_T_highest_peak[col_temp].loc[df_I_T_smsr[col_temp] >= 30].index,
            df_I_T_highest_peak[col_temp].loc[df_I_T_smsr[col_temp] >= 30],
            "-",
            color=color,
            alpha=0.8,
        )
    ax1.grid(which="both")  # adding grid
    ax1.minorticks_on()
    ax1.set_xlabel("Current set, mA")
    ax1.set_ylabel("Peak wavelength, nm")
    ax1.legend(loc=0)
    ax1.set_xlim(left=0)

    plt.savefig(figure_directory / f"{native_title}-highestpeak.png", dpi=300)
    plt.close()

    # 8. plot λ(P_dis), λ(T), dλ\dT(P_dis), R_th(T) lineplots
    # Creating figure
    fig = plt.figure(figsize=(2 * 11.69, 2 * 8.27))
    plt.suptitle(f"{title}")
    # Plotting dataset
    ax1 = fig.add_subplot(221)  # λ(P_dis) at different temperatures
    # iteration for left plot ax
    for col_temp in df_Pdis_T_int.columns:  # columns are temperatures
        # linear approximation
        model = linear_model.LinearRegression()
        X = df_Pdis_T_int_drop.index.values.reshape(-1, 1)
        y = df_Pdis_T_int_drop[col_temp]
        model.fit(X, y)
        slope = model.coef_[0]
        # save fit parameters to a DataFrame
        dldp.loc[len(dldp)] = [col_temp, slope, model.intercept_]

        ax1.plot(
            df_Pdis_T_int.index,
            df_Pdis_T_int[col_temp],
            "-",
            alpha=0.8,
            label=f"{col_temp} °C",
        )
        ax1.scatter(
            df_Pdis_T.index,
            df_Pdis_T[col_temp],
            alpha=0.2,
        )
        ax1.plot(
            df_Pdis_T_int.index,
            model.predict(df_Pdis_T_int.index.values.reshape(-1, 1)),
            "-.",
            alpha=0.2,
            label=f"fit {col_temp} °C, dλ/dP_dis={slope:.3f}, intercept={model.intercept_:.3f}",
        )
    # Adding title
    plt.title("λ(P_dis) at different temperatures")
    # adding grid
    ax1.grid(which="both")  # adding grid
    ax1.minorticks_on()
    # Adding labels
    ax1.set_xlabel("Dissipated power, mW")
    ax1.set_ylabel("Peak wavelength, nm")
    # Adding legend
    ax1.legend(loc=0, prop={"size": 4})

    ax2 = fig.add_subplot(222)  # λ(T) at different dissipated power
    for col_pdis in df_Pdis_T_int.T.columns:  # columns are dissipated power
        if col_pdis in df_Pdis_T_int_drop.T.columns:
            # linear approximation
            model = linear_model.LinearRegression()
            X = df_Pdis_T_int_drop.T.index.values.reshape(-1, 1)
            y = df_Pdis_T_int_drop.T[col_pdis]
            model.fit(X, y)
            slope = model.coef_[0]
            # save fit parameters to a DataFrame
            dldt.loc[len(dldt)] = [col_pdis, slope, model.intercept_]

        ax2.plot(
            df_Pdis_T_int.T.index,
            df_Pdis_T_int.T[col_pdis],
            "-",
            alpha=0.4,
            # label=f"Dissipated power {col} mW",
        )
        ax2.scatter(
            df_Pdis_T.T.index,
            df_Pdis_T.T[col_pdis],
            alpha=0.2,
        )
    # Adding title
    plt.title("λ(T) at different dissipated power")
    ax2.grid(which="both")  # adding grid
    ax2.minorticks_on()
    ax2.set_xlabel("Temperature, °C")
    ax2.set_ylabel("Peak wavelength, nm")
    # ax.legend(loc=0, prop={"size": 4})

    # need to get dλ/dT at 0 mW dissipated power
    # make a linear approximation TODO make it linear/polinomial choice
    model = linear_model.LinearRegression()
    X = dldt["Dissipated power, mW"].values.reshape(-1, 1)
    y = dldt["dλ/dT"]
    model.fit(X, y)
    model.coef_[0]
    dldt_zero = model.intercept_  # dλ/dT at 0 mW dissipated power

    ax3 = fig.add_subplot(223)  # dλ/dT(P_dis)
    ax3.scatter(
        dldt["Dissipated power, mW"],
        dldt["dλ/dT"],
        alpha=0.2,
    )
    ax3.plot(
        np.vstack([[0], dldt["Dissipated power, mW"].values.reshape(-1, 1)]),
        model.predict(
            np.vstack([[0], dldt["Dissipated power, mW"].values.reshape(-1, 1)])
        ),
        "-.",
        alpha=0.6,
        label=f"fit slope={model.coef_[0]:.6f}, intercept(dλ/dT(0))={model.intercept_:.6f}, ",
    )
    plt.title("dλ/dT(P_dis)")
    ax3.grid(which="both")  # adding grid
    ax3.minorticks_on()
    ax3.set_xlabel("Dissipated power, mW")
    ax3.set_ylabel("dλ/dT")
    ax3.legend(loc=0, prop={"size": 10})
    ax3.set_ylim(bottom=0)

    # make a thermal resistance DataFrame
    R_th = pd.DataFrame(columns=["Temperature, °C", "R_th, K/mW"])
    for i, temperature in enumerate(temperatures):
        # populate DataFrame with data
        R_th.loc[i] = [temperature, dldp["dλ/dP_dis"].iloc[i] / dldt_zero]
    # save DataFrame to .csv
    R_th.to_csv(
        figure_directory / f"{native_title}-R_th.csv",
        index=False,
    )

    # linear approximation of R_th(T)
    R_th_model = linear_model.LinearRegression()
    X = R_th["Temperature, °C"].values.reshape(-1, 1)
    y = R_th["R_th, K/mW"]
    R_th_model.fit(X, y)

    ax4 = fig.add_subplot(224)  # R_th(T)
    ax4.scatter(
        R_th["Temperature, °C"],
        R_th["R_th, K/mW"],
        alpha=0.2,
    )
    ax4.plot(
        np.linspace(temperatures[0], temperatures[-1], 100),
        R_th_model.predict(
            np.linspace(temperatures[0], temperatures[-1], 100).reshape(-1, 1)
        ),
        "-.",
        alpha=0.6,
        label=f"fit R_th",
    )
    # Adding title
    plt.title("R_th(T)=dT/dP_dis=(dλ/dP_dis)/(dλ/dT)")
    # adding grid
    ax4.grid(which="both")  # adding grid
    ax4.minorticks_on()
    # Adding labels
    ax4.set_xlabel("Temperature, °C")
    ax4.set_ylabel("R_th, K/mW")
    # Adding legend
    ax4.legend(loc=0, prop={"size": 12})
    ax4.set_ylim(bottom=0)

    plt.savefig(
        figure_directory / f"{waferid}-{wavelength}nm-{coordinates}-lineplot.png",
        dpi=300,
    )
    plt.close()

    dldp.to_csv(
        figure_directory / f"{waferid}-{wavelength}nm-{coordinates}-dλdP_dis_fit.csv",
        index=False,
    )
    dldt.to_csv(
        figure_directory / f"{waferid}-{wavelength}nm-{coordinates}-dλdT_fit.csv",
        index=False,
    )

    # 9. plot heatmaps
    T_act = (
        pd.DataFrame(columns=[*temperatures])
        .rename_axis("Current set, mA", axis=0)
        .rename_axis("Temperature, °C", axis=1)
    )
    P_out = (
        pd.DataFrame(columns=[*temperatures])
        .rename_axis("Current set, mA", axis=0)
        .rename_axis("Temperature, °C", axis=1)
    )
    mask = (
        pd.DataFrame(columns=[*temperatures])
        .rename_axis("Current set, mA", axis=0)
        .rename_axis("Output power, mW", axis=1)
    )
    current_axis_values = [
        i * increment_current_heatmap
        for i in range(0, int(larg_current / increment_current_heatmap + 1), 1)
    ]
    for temperature in temperatures:
        livdf = pd.read_csv(dict_of_filenames_liv[temperature])
        currents = [
            current
            for current in current_axis_values
            if np.float64(current) in livdf["Current set, mA"].astype(float).tolist()
        ]
        for current in currents:
            row = livdf.loc[livdf["Current set, mA"] == current]
            if (
                float(
                    (
                        row["Current set, mA"] * row["Voltage, V"]
                        - row["Output power, mW"]
                    ).iloc[0]
                )
            ) > 0:
                pdis = float(
                    (
                        row["Current set, mA"] * row["Voltage, V"]
                        - row["Output power, mW"]
                    ).iloc[0]
                )  # mW
            else:
                pdis = float((row["Current set, mA"] * row["Voltage, V"]).iloc[0])  # mW
            deltaT = float(
                (
                    R_th["R_th, K/mW"][R_th["Temperature, °C"] == temperature] * pdis
                ).iloc[0]
            )
            T_act.at[current, temperature] = deltaT
            P_out.at[current, temperature] = float(row["Output power, mW"].iloc[0])
            mask.at[current, temperature] = False
            T_act.sort_index(ascending=False, inplace=True)
            P_out.sort_index(ascending=False, inplace=True)
            mask.sort_index(ascending=False, inplace=True)

    T_act.to_csv(
        figure_directory / f"{waferid}-{wavelength}nm-{coordinates}-T_act.csv",
    )
    P_out.to_csv(figure_directory / f"{waferid}-{wavelength}nm-{coordinates}-P_out.csv")

    T_act = T_act.astype(float)
    P_out = P_out.astype(float)
    mask = mask.astype(bool).fillna(True)

    # fig = plt.figure(figsize=(2 * 11.69, 2 * 8.27)) # TODO make variables for figsizes
    fig = plt.figure(figsize=(10, 5))
    fig.suptitle(f"{title}")
    ax1 = fig.add_subplot(121)
    sns.heatmap(T_act, annot=True, fmt="3.2f", ax=ax1, mask=mask)
    ax1.set_title("ΔT of active area, °C")
    ax2 = fig.add_subplot(122)
    sns.heatmap(P_out, annot=True, fmt="3.2f", ax=ax2, mask=mask)
    ax2.set_title("Output power, mW")

    plt.savefig(
        figure_directory / f"{waferid}-{wavelength}nm-{coordinates}-T_active_area.png",
        dpi=300,
    )
