#!/usr/bin/env python3

import sys
import re
import os

import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from math import ceil
from pathlib import Path
from builtins import str


def name_from_directory(directory):
    name_from_dir = list(directory.parts)
    if name_from_dir[0] == "data":
        del name_from_dir[0]
    if name_from_dir[-1] == "RIN":
        del name_from_dir[-1]
    name_from_dir = "-".join(name_from_dir)
    return name_from_dir


def analyze_rin(
    directory,
    pxa_csv=True,
    freqlimit=[0, 40],
):
    if isinstance(directory, str):
        directory = Path(directory)
    start_directory = directory
    report_dir = start_directory / "RIN_reports"
    print(report_dir)
    # get filenames and currents
    walk = list(os.walk(directory))
    if not pxa_csv:
        # first check if you have a csv file from automatic system
        matched_csv_files = sorted(
            directory.glob("*.csv")
        )  # TODO don't forget to iterate through files!
        matched_csv_files_stems = (
            f"{n}: " + i.stem for n, i in enumerate(matched_csv_files, start=1)
        )
        auto_file_path = matched_csv_files[0]
        print(
            f"Matched .csv files: {len(matched_csv_files)}",
            *matched_csv_files_stems,
            sep="\n",
        )
        print(f"Processing .csv file: {auto_file_path.stem}")
        if not matched_csv_files:
            return None, report_dir
    elif pxa_csv:
        pass  # TODO fix and del

    name_from_dir = name_from_directory(directory)

    if pxa_csv:
        pass  # TODO fix and del
    elif not pxa_csv:  # automatic system csv file parsing and processing
        report_dir = start_directory / "RIN_reports(auto)"
        report_dir.mkdir(exist_ok=True)
        auto_file = pd.read_csv(auto_file_path, header=[0, 1, 2], sep="\t")
        # print(auto_file.head())
        sourcing_currents = np.append(
            auto_file["ESA Current"].iloc[0].values.reshape(-1),
            auto_file["ESA Current"]
            .iloc[1:][auto_file["ESA Current"].iloc[1:] > 0]
            .dropna()
            .values.reshape(-1),
        )
        pd_currents = np.append(
            auto_file["ESA PD Current"].iloc[0].values.reshape(-1),
            auto_file["ESA PD Current"]
            .iloc[1:][auto_file["ESA PD Current"].iloc[1:] > 0]
            .dropna()
            .values.reshape(-1),
        )

        sourcing_current_columns = [f"Current {i}, mA" for i in sourcing_currents]
        sourcing_current_columns_with_pd = [
            f"Sourcing current {c}, mA; PD current {p}, μA"
            for c, p in zip(sourcing_currents, pd_currents)
        ]

        spectra = auto_file["ESA Intensity"].values.reshape(-1)
        rin = auto_file["ESA RIN"].values.reshape(-1)
        points = np.where(spectra == -999999999)[0][0]
        frequency = auto_file["ESA Frequency"].values.reshape(-1)[0:points]
        waferid_wl, coordinates, _ = auto_file_path.stem.split("_")
        waferid, wavelength = waferid_wl.split("-")
        coordinates = coordinates[:2] + coordinates[3:]
        temperature = 25.0

        df_all_spectra = pd.DataFrame(
            index=frequency, columns=sourcing_current_columns_with_pd
        )
        df_all_rin = pd.DataFrame(index=frequency, columns=sourcing_current_columns)
        for i in (df_all_spectra, df_all_rin):
            i.index.name = "Frequency, Hz"

        for i, (c, p) in enumerate(zip(sourcing_currents, pd_currents)):
            print(f"Sourcing current {c}, mA; PD current {p}, μA")
            start = i * points + i
            stop = (i + 1) * points + i
            spectrum_col = f"Sourcing current {c}, mA; PD current {p}, μA"
            rin_col = f"Current {c}, mA"
            df_all_spectra[spectrum_col] = spectra[start:stop]
            if c > 0:
                df_all_rin[rin_col] = rin[start:stop]

        df_all_spectra = df_all_spectra[
            (df_all_spectra.index >= freqlimit[0] * 10**9)
            & (df_all_spectra.index <= freqlimit[1] * 10**9)
        ]

        df_all_rin = df_all_rin[
            (df_all_rin.index > freqlimit[0] * 10**9)
            & (df_all_rin.index < freqlimit[1] * 10**9)
        ]

        if not os.path.exists(report_dir):  # make directories
            os.makedirs(report_dir)
        df_all_spectra.to_csv(report_dir / (name_from_dir + "-spectra(auto).csv"))
        df_all_rin.to_csv(report_dir / (name_from_dir + "-rin(auto).csv"))

    return df_all_spectra, df_all_rin, report_dir


#  _____ _
# |  ___(_) __ _ _   _ _ __ ___  ___
# | |_  | |/ _` | | | | '__/ _ \/ __|
# |  _| | | (_| | |_| | | |  __/\__ \
# |_|   |_|\__, |\__,_|_|  \___||___/
#          |___/
def makefigs(
    spectra,
    rin,
    directory,
    report_directory,
    title=None,
    pxa_csv=False,
    columns_in_figure=5,
    xlim=None,
    ylim=None,
    figsize=(18, 12),
):
    if spectra is None and rin is None:
        return

    if isinstance(directory, str):
        directory = Path(directory)
    name_from_dir = name_from_directory(directory)
    if not title:
        title = name_from_dir

    for i in (spectra, rin):
        if i.index.name == "Frequency, Hz":
            i.index = i.index * (10 ** (-9))
            i.index.name = "Frequency, GHz"

    # plot every spectra
    spectra.plot(
        title=title + f" All Spectra",
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
    )
    plt.minorticks_on()
    plt.grid(which="both")
    plt.savefig(
        report_directory / (name_from_dir + f"-spectra[all].png")
    )  # save figure
    plt.close()

    rin.plot(
        title=title + f" All RIN",
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
    )
    plt.minorticks_on()
    plt.grid(which="both")
    plt.savefig(report_directory / (name_from_dir + f"-RIN[all].png"))  # save figure
    plt.close()

    numfig = ceil(len(spectra.columns) / columns_in_figure)
    mincol = [min(columns_in_figure * i, len(spectra.columns)) for i in range(numfig)]
    maxcol = [
        min(columns_in_figure * (i + 1), len(spectra.columns)) for i in range(numfig)
    ]

    # plot in small chunks
    for i, (mincol, maxcol) in enumerate(zip(mincol, maxcol)):
        print(f"i:{i}, mincol {mincol}, maxcol {maxcol}")
        spectra.iloc[:, mincol:maxcol].plot(
            title=title + f" Spectra [{i+1}/{numfig}]",
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
        )
        plt.minorticks_on()
        plt.grid(which="both")
        plt.savefig(
            report_directory / (name_from_dir + f"-spectra[{i+1}of{numfig}].png")
        )
        plt.close()

        rin.iloc[:, mincol:maxcol].plot(
            title=title + f" RIN [{i+1}/{numfig}]",
            figsize=figsize,
            xlim=xlim,
            ylim=ylim,
        )
        plt.minorticks_on()
        plt.grid(which="both")
        plt.savefig(report_directory / (name_from_dir + f"-RIN[{i+1}of{numfig}].png"))
        plt.close()

    # plot only cols_in_fig
    spectra.iloc[:, ::numfig].plot(
        title=title + f" Spectra [every_{numfig}]",
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
    )
    plt.minorticks_on()
    plt.grid(which="both")
    plt.savefig(report_directory / (name_from_dir + f"-spectra[every_{numfig}].png"))
    plt.close()

    rin.iloc[:, ::numfig].plot(
        title=title + f" RIN [every {numfig}]",
        figsize=figsize,
        xlim=xlim,
        ylim=ylim,
    )
    plt.minorticks_on()
    plt.grid(which="both")
    plt.savefig(report_directory / (name_from_dir + f"-RIN[every_{numfig}].png"))
    plt.close()


def analyze_rin_function(directory, settings=None):
    if isinstance(directory, str):
        directory = Path(directory)
    if settings is None:
        with open(Path("templates") / "rin.yaml") as fh:
            settings = yaml.safe_load(fh)
    print(directory)
    print(settings)

    title = settings["title"]
    fit_freqlimit = settings["fit_freqlimit"]
    columns_in_figure = settings["columns_in_figure"]
    xlim = settings["xlim"]
    ylim = settings["ylim"]
    figure_size = settings["figure_size"]

    # automatic system .csv
    df_all_spectra, df_all_rin, report_directory = analyze_rin(
        directory,
        pxa_csv=False,
        freqlimit=fit_freqlimit,
    )

    makefigs(
        df_all_spectra,
        df_all_rin,
        directory,
        report_directory,
        title=title,
        pxa_csv=False,
        columns_in_figure=columns_in_figure,
        xlim=xlim,
        ylim=ylim,
        figsize=figure_size,
    )
    print("RIN for auto .csv is done")
