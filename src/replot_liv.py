#!/usr/bin/env python3
import sys
import os
import re
import time
import pyvisa
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from pathlib import Path
from builtins import str

from src.measure_liv import buildplt_everything, buildplt_liv


def replot_liv_function(directory, settings=None):
    if isinstance(directory, str):
        directory = Path(directory)
    directory = directory / "LIV"
    if not directory.exists() or not directory.is_dir():
        print(f"Can't find LIV data in {directory}")
        return
    if settings is None:
        with open(Path("templates") / "replot_liv.yaml") as fh:
            settings = yaml.safe_load(fh)
    print(directory)
    print(settings)

    title = settings["title"]
    title_fontsize = settings["title_fontsize"]
    threshold_decision_level = settings["threshold_decision_level"]
    figure_dpi = settings["figure_dpi"]
    everithing_figsize = settings["everithing_figsize"]
    pce_legendsize = settings["pce_legendsize"]
    liv_figsize = settings["liv_figsize"]

    matched_files = sorted(directory.glob("*.csv"))
    matched_files_stems = (
        f"{n}: " + i.stem for n, i in enumerate(matched_files, start=1)
    )
    print(f"Matched .csv files: {len(matched_files)}", *matched_files_stems, sep="\n")

    for i, file in enumerate(matched_files, start=1):
        print(f"[{i}/{len(matched_files)}] {file}")
        iv = pd.read_csv(file)
        file_stem = file.stem
        (
            waferid,
            wavelength,
            coordinates,
            temperature,
            date,
            time,
            powermeter,
        ) = file_stem.split("-")
        temperature = temperature.removesuffix("°C")
        # timestr = date + "-" + time
        wavelength = wavelength.removesuffix("nm")
        # filepath = directory / file_stem
        # save figures
        buildplt_everything(
            dataframe=iv,
            waferid=waferid,
            wavelength=wavelength,
            coordinates=coordinates,
            temperature=temperature,
            powermeter=powermeter,
            threshold_decision_level=threshold_decision_level,
            title=title,
            title_fontsize=title_fontsize,
            everithing_figsize=everithing_figsize,  # (1.8 * 11.69, 1.8 * 8.27)
            pce_legendsize=pce_legendsize,
        )
        plt.savefig(
            directory / (file_stem + "-everything.png"), dpi=figure_dpi
        )  # save figure
        i_threshold, i_rollover = buildplt_liv(
            dataframe=iv,
            waferid=waferid,
            wavelength=wavelength,
            coordinates=coordinates,
            temperature=temperature,
            powermeter=powermeter,
            threshold_decision_level=threshold_decision_level,
            title=title,
            liv_figsize=liv_figsize,
        )
        plt.savefig(
            directory
            / (file_stem + f"_Ith={i_threshold:.2f}_Iro={i_rollover:.2f}.png"),
            dpi=figure_dpi,
        )  # save figure
        plt.close("all")
