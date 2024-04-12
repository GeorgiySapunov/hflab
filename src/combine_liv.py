#!/usr/bin/env python3

import sys
import re
import os
import yaml
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from builtins import str


def combine_liv_function(path, settings=None):
    if not settings:
        return
    report_directory = path.parent
    if settings["title"]:
        title = settings["title"]
    else:
        title = path.stem
    legend_dict = settings["legend_dict"]
    power_limit = settings["power_limit"]
    voltage_limit = settings["voltage_limit"]
    current_limit = settings["current_limit"]
    figure_size = settings["figure_size"]
    legend_fontsize = settings["legend_fontsize"]
    label_size = settings["label_size"]
    title_size = settings["title_size"]
    legend_location = settings["legend_location"]
    alpha = settings["alpha"]
    figure_dpi = int(settings["figure_dpi"])

    legend_list = list(legend_dict.keys())
    liv_files = [Path(file) for file in legend_dict.values()]

    if report_directory:
        if isinstance(report_directory, str):
            report_directory = Path(report_directory)
    else:
        report_directory = Path("reports")
        report_directory.mkdir(exist_ok=True)

    plt.rc("axes", labelsize=label_size)  # fontsize of the x and y labels
    plt.rc("axes", titlesize=title_size)  # fontsize of the axes title
    fig = plt.figure(figsize=figure_size)
    ax = fig.add_subplot(111)
    # Creating figure
    ax2 = ax.twinx()
    if title:
        plt.title(title)

    labs = []
    lns = []

    for num, file in enumerate(liv_files):
        le = len(liv_files)
        print(f"[{num+1}/{le}] {file}")
        leg = legend_list[num]
        dfiv = pd.read_csv(file)

        ax.set_xlabel("Current, mA")
        ax.set_ylabel("Output power, mW", color="blue")
        ax2.set_ylabel("Voltage, V", color="red")

        # select columns in the Data Frame
        seti = dfiv["Current set, mA"]
        i = dfiv["Current, mA"]
        v = dfiv["Voltage, V"]
        l = dfiv["Output power, mW"]

        lns1 = ax.plot(i, l, "-", label=f"{leg} Output power, mW", alpha=alpha)
        # Creating Twin axes for dataset_1
        lns2 = ax2.plot(i, v, ":", label=f"{leg} Voltage, V", alpha=alpha)

        # legend
        lns.extend(lns1 + lns2)
        labs.extend([l.get_label() for l in lns1 + lns2])

    ax.set_ylim(power_limit)
    ax.set_xlim(current_limit)
    ax2.set_ylim(voltage_limit)

    ax.minorticks_on()
    ax2.minorticks_on()
    ax.grid(which="both")
    ax.legend(lns, labs, loc=legend_location, fontsize=legend_fontsize)

    plt.savefig(report_directory / (title + ".png"), dpi=figure_dpi)
