#!/usr/bin/env python3

import sys
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

legend_list = [
    "25 °C",
    "35 °C",
    "45 °C",
    "55 °C",
    "65 °C",
    "75 °C",
    "85 °C",
]
# legend_list = None

fig = plt.figure(figsize=(11.69, 8.27))
ax = fig.add_subplot(111)
# Creating figure
ax2 = ax.twinx()
plt.title("1550nm 57M6GSG")
# plt.title(
#     str(waferid)
#     + " "
#     + str(wavelength)
#     + " nm "
#     + str(coordinates)
#     + " "
#     + str(temperature)
#     + " °C"
#     # + " "
#     # + powermeter
# )  # Adding title

labs = []
lns = []
names = []
vcsel = []
# max_current = 20

for num, file in enumerate(sys.argv[1:]):
    directory = file.split("/")

    le = len(sys.argv[1:])
    print(f"[{num+1}/{le}] {directory}")

    if legend_list:
        leg = legend_list[num]
    else:
        name_from_dir = (
            file.replace("/", "-")
            .removesuffix("-")
            .removesuffix("-PNA")
            .removeprefix("data-")
        )
        leg = name_from_dir

    # names.append(name_from_dir)
    # vcsel.append(name_from_dir.split("-")[-1])

    dfiv = pd.read_csv(file, index_col=0)

    # Adding labels
    ax.set_xlabel("Current, mA")
    ax.set_ylabel("Output power, mW", color="blue")
    ax2.set_ylabel("Voltage, V", color="red")

    # select columns in the Data Frame
    seti = dfiv["Current set, mA"]
    i = dfiv["Current, mA"]
    # i = seti # uncoment this line to use "Current set, mA" column to plot graphs
    v = dfiv["Voltage, V"]
    l = dfiv["Output power, mW"]

    # Plotting dataset_2
    lns1 = ax.plot(i, l, "-", label=f"{leg} Output power, mW")
    # Creating Twin axes for dataset_1
    lns2 = ax2.plot(i, v, "-.", label=f"{leg} Voltage, V")

    # legend
    lns.extend(lns1 + lns2)
    labs.extend([l.get_label() for l in lns1 + lns2])

# Setting Y limits
ax.set_ylim(bottom=0)  # Power
ax.set_xlim(left=0)  # Current
ax2.set_ylim(bottom=0)  # Voltage

ax.minorticks_on()
ax2.minorticks_on()
ax.grid(which="both")  # adding grid

# Adding legend
ax.legend(lns, labs, loc=4, fontsize=8)
# ax2.legend(loc=0)

# fig.suptitle("-".join(name_from_dir.split("-")[:-1]) + " " + "-".join(vcsel))

if not os.path.exists("reports/"):  # make directories
    os.makedirs("reports/")
# plt.savefig("reports/" + "-".join(vcsel) + ".png", dpi=600)  # save figure
plt.savefig("reports/" + "LIV" + ".png", dpi=600)  # save figure
