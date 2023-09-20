#!/usr/bin/env python3

import sys
import os
import re
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

from settings import settings


def analyse(directory):
    # get filenames for temperature set
    walk = list(os.walk(directory + "OSA/"))
    string_for_re = ".*\\-OS.csv"
    r = re.compile(string_for_re)
    files = walk[0][2]
    matched_os_files = list(filter(r.match, files))
    # waferid, wavelength, coordinates, *_ = matched_os_files[0]
    # get temperature set
    temperatures = set()
    for file in matched_os_files:
        file_name_parser = file.split("-")
        r2 = re.compile(".*°C")
        temperature = list(filter(r2.match, file_name_parser))[0]
        temperature = float(temperature.removesuffix("°C"))
        temperatures.add(float(temperature))
    print(f"temperature set from OSA: {temperatures}")

    df1 = pd.DataFrame(
        columns=[
            "Dissipated power, mW",
            *temperatures
            # "Current, mA",
            # "Voltage, V",
            # "Power consumption, mW",
        ]
    )
    for temperature in temperatures:
        # 1. take liv file
        # 2. take osa file
        # 3. get the last peak lambda
        # 3. make a Pdis, lambda df
        # 4. merge them together on Pdis
        pass

    # 6. sort and interpolate
    # 7. reshape the data to (T, lambdas at diff Pdis)
    # 8. plot heatmaps (T, lambda, Pdis and Pdis, lambda, T)
    # 9. plot the same lineplots
    # 10. calculate dT/dPdis


for i, directory in enumerate(sys.argv[1:]):
    num = len(sys.argv[1:])
    print(f"[{i+1}/{num}] {directory}")
    df = analyse(directory)
    makefigs(df, directory)
