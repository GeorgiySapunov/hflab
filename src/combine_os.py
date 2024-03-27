#!/usr/bin/env python3
import os
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from builtins import str


def get_temperature(file):
    if isinstance(file, str):
        file = Path(file)
    file_name_parser = file.stem.split("-")
    r2 = re.compile(".*°C")
    temperature = list(filter(r2.match, file_name_parser))[0]
    temperature = float(temperature.removesuffix("°C"))
    return temperature


def combine_os_function(path, settings=None):
    if not settings:
        return
    report_directory = path.parent
    if settings["title"]:
        title = settings["title"]
    else:
        title = path.stem
    list_of_files = settings["list_of_files"]
    spectra_xlim_left, spectra_xlim_right = settings["spectra_xlim"]
    figure_dpi = int(settings["figure_dpi"])

    # figure_size = settings["figure_size"]
    # legend_fontsize = settings["legend_fontsize"]
    # label_size = settings["label_size"]
    # title_size = settings["title_size"]
    # legend_location = settings["legend_location"]
    # alpha = settings["alpha"]

    fig = plt.figure(figsize=(11.69, 0.5 * 8.27))
    ax = fig.add_subplot(111)
    ax.set_title(title)
    for fn, file in enumerate(list(list_of_files.keys())):
        filepath = Path(file)
        osdf = pd.read_csv(filepath)
        if list_of_files[file]:
            if "currents" in list_of_files[file].keys():
                currents = list(list_of_files[file]["currents"])
                columns = [f"Intensity at {i:.2f} mA, dBm" for i in currents]

                temperature = get_temperature(filepath)
                if fn == 0:
                    line = "-"
                else:
                    line = "-."
                for i, current_column in enumerate(zip(currents, columns)):
                    current, column = current_column
                    label = None
                    if "labels" in list(list_of_files[file].keys()):
                        if list_of_files[file]["labels"]:
                            label = list_of_files[file]["labels"][i]
                    if not label:
                        label = f"{current} mA at {temperature} °C"
                    # spectrum line
                    ax.plot(
                        osdf["Wavelength, nm"],
                        osdf[column],
                        line,
                        alpha=0.5,
                        label=label,
                    )
    # Adding title
    # adding grid
    ax.grid()  # adding grid
    ax.minorticks_on()
    # Adding labels
    ax.set_xlabel("Wavelength, nm")
    ax.set_ylabel("Intensity, dBm")
    # Adding legend
    # ax.legend(loc=0, prop={"size": 4})
    ax.legend(loc=2)
    ax.set_ylim(-80, 0)
    ax.set_xlim(spectra_xlim_left, spectra_xlim_right)

    plt.savefig(report_directory / (title + ".png"), dpi=figure_dpi)
    plt.close()
