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
from liv import (
    # annotate_max_L,
    # annotate_max_ef,
    # annotate_threshold,
    buildplt_all,
    buildplt_tosave,
)


def makefigs(directory):
    if directory[-1] != "/":  # TODO check it
        directory = directory + "/"
    directory = directory + "LIV"
    if not os.path.exists(directory):
        print(f"Can't find LIV data in {directory}")
        return
    # get filenames and currents
    walk = list(os.walk(directory))
    # first check if you have a csv file
    string_for_re = ".*\\.csv$"
    r = re.compile(string_for_re)
    files = walk[0][2]
    matched_files = list(filter(r.match, files))
    matched_files.sort()
    print(f"Matched .csv files: {matched_files}")

    for file in matched_files:
        print(file)
        iv = pd.read_csv(directory + "/" + file)
        file = file.removesuffix("/").removesuffix(".csv")
        (
            waferid,
            wavelength,
            coordinates,
            temperature,
            powermeter,
            date,
            time,
        ) = file.split("-")
        temperature = temperature.removesuffix("°C")
        timestr = date + "-" + time
        wavelength = wavelength.removesuffix("nm")
        filepath = (
            directory
            + "/"
            + f"{waferid}-{wavelength}nm-{coordinates}-{temperature}°C-{timestr}-{powermeter}"
        )
        # save figures
        buildplt_all(
            dataframe=iv,
            waferid=waferid,
            wavelength=wavelength,
            coordinates=coordinates,
            temperature=temperature,
            powermeter=powermeter,
            current_increment_LIV=settings["current_increment_LIV"],
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


for i, directory in enumerate(sys.argv[1:]):
    le = len(sys.argv[1:])
    print(f"[{i+1}/{le}] {directory}")
    makefigs(directory)
