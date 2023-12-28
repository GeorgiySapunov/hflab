#!/usr/bin/env python3

import sys
import re
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# 57M6GSG

legend_list = [  # 1550nm 57M6GSG
    # python plot_liv.py data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-25.0°C-20230824-121432-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-35.0°C-20230824-121845-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-45.0°C-20230824-122213-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-55.0°C-20230824-123119-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-65.0°C-20230824-123512-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-75.0°C-20230824-133308-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-85.0°C-20230824-133709-PM100USB.csv
    "25 °C",
    "35 °C",
    "45 °C",
    "55 °C",
    "65 °C",
    "75 °C",
    "85 °C",
]
title = "M-type 6 μm BTJ 1550 nm VCSEL"

# legend_list = [  # MuHA2B 940nm classical VCSELs
# python plot_liv.py data/MuHA2B-940nm/1004/LIV/MuHA2B-940nm-1004-25.0°C-20231019-083902-PM100USB.csv data/MuHA2B-940nm/1005/LIV/MuHA2B-940nm-1005-25.0°C-20231019-083549-PM100USB.csv data/MuHA2B-940nm/1007/LIV/MuHA2B-940nm-1007-25.0°C-20231019-084410-PM100USB.csv data/MuHA2B-940nm/1008/LIV/MuHA2B-940nm-1008-25.0°C-20231019-085141-PM100USB.csv data/MuHA2B-940nm/1009/LIV/MuHA2B-940nm-1009-25.0°C-20231019-085515-PM100USB.csv data/MuHA2B-940nm/100A/LIV/MuHA2B-940nm-100A-25.0°C-20231019-085926-PM100USB.csv data/MuHA2B-940nm/100B/LIV/MuHA2B-940nm-100B-25.0°C-20231019-090420-PM100USB.csv data/MuHA2B-940nm/100C/LIV/MuHA2B-940nm-100C-25.0°C-20231019-090952-PM100USB.csv data/MuHA2B-940nm/100D/LIV/MuHA2B-940nm-100D-25.0°C-20231019-091402-PM100USB.csv data/MuHA2B-940nm/100E/LIV/MuHA2B-940nm-100E-25.0°C-20231019-092200-PM100USB.csv
#     "2.5μm",
#     "2.5μm",
#     "3μm",
#     "3.5μm",
#     "3.5μm",
#     "4μm",
#     "4μm",
#     "4.5μm",
#     "5μm",
#     "10μm",
# ]
# title = "MuHA2B 940nm classical VCSELs"

# legend_list = [  # 1550nm 5μm BTJ
#     # python plot_liv.py data/PW2024/46S5GSG/LIV/spb1-1550nm-46S5GSG-25.0°C-PM100USB-202308-0.csv data/PW2024/57M5GSG/LIV/spb1-1550nm-57M5GSG-25.0°C-PM100USB-20230817-082640.csv data/PW2024/57L5GSG/LIV/spb1-1550nm-57L5GSG-25.0°C-PM100USB-20230817-082007.csv
#     # data/PW2024/46S5GSG 57M5GSG 57L5GSG
#     "S-type 5 μm BTJ",
#     "M-type 5 μm BTJ",
#     "L-type 5 μm BTJ",
# ]
# title = "5 μm BTJ 1550 nm VCSELs"

# legend_list = [  # 1550nm 6μm BTJ
#     # python plot_liv.py data/PW2024-1550nm/36S6GSG/LIV/spb1-1550nm-36S6GSG-25.0°C-20231114-062932-PM100USB.csv data/PW2024-1550nm/57M6GSG/LIV/PW2024-1550nm-57M6GSG-25.0°C-20230824-121432-PM100USB.csv data/PW2024-1550nm/46L6GSG/LIV/spb1-1550nm-46L6GSG-25.0°C-202308-0-PM100USB.csv
#     # data/PW2024/36S6GSG 57M6GSG 46L6GSG
#     "S-type 6 μm BTJ",
#     "M-type 6 μm BTJ",
#     "L-type 6 μm BTJ",
# ]
# title = "6 μm BTJ 1550 nm VCSELs"

# legend_list = [  # 1550nm 8μm BTJ
#     # python plot_liv.py data/PW2024-1550nm/36S8GSG/LIV/spb1-1550nm-36S8GSG-25.0°C-20231114-065049-PM100USB.csv data/PW2024-1550nm/36M8GSG/LIV/spb1-1550nm-36M8GSG-25.0°C-20231114-064733-PM100USB.csv data/PW2024-1550nm/36L8GSG/LIV/spb1-1550nm-36L8GSG-25.0°C-20231114-064346-PM100USB.csv
#     # data/PW2024/36S8GSG 36M8GSG 36L8GSG
#     "S-type 8 μm BTJ",
#     "M-type 8 μm BTJ",
#     "L-type 8 μm BTJ",
# ]
# title = "8 μm BTJ 1550 nm VCSELs"

# 47M6GSG
#
# legend_list = [  # 1550nm 47M6GSG
#     # python plot_liv.py data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-25.0°C-20231211-120450-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-35.0°C-20231211-044948-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-45.0°C-20231211-050036-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-55.0°C-20231211-051000-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-65.0°C-20231211-051850-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-75.0°C-20231211-053353-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-85.0°C-20231211-054511-PM100USB.csv
#     "25 °C",
#     "35 °C",
#     "45 °C",
#     "55 °C",
#     "65 °C",
#     "75 °C",
#     "85 °C",
# ]
# title = "M-type 6 μm BTJ 1550 nm VCSEL"

# legend_list = [  # MuHA2B 940nm classical VCSELs
# python plot_liv.py data/MuHA2B-940nm/1004/LIV/MuHA2B-940nm-1004-25.0°C-20231019-083902-PM100USB.csv data/MuHA2B-940nm/1005/LIV/MuHA2B-940nm-1005-25.0°C-20231019-083549-PM100USB.csv data/MuHA2B-940nm/1007/LIV/MuHA2B-940nm-1007-25.0°C-20231019-084410-PM100USB.csv data/MuHA2B-940nm/1008/LIV/MuHA2B-940nm-1008-25.0°C-20231019-085141-PM100USB.csv data/MuHA2B-940nm/1009/LIV/MuHA2B-940nm-1009-25.0°C-20231019-085515-PM100USB.csv data/MuHA2B-940nm/100A/LIV/MuHA2B-940nm-100A-25.0°C-20231019-085926-PM100USB.csv data/MuHA2B-940nm/100B/LIV/MuHA2B-940nm-100B-25.0°C-20231019-090420-PM100USB.csv data/MuHA2B-940nm/100C/LIV/MuHA2B-940nm-100C-25.0°C-20231019-090952-PM100USB.csv data/MuHA2B-940nm/100D/LIV/MuHA2B-940nm-100D-25.0°C-20231019-091402-PM100USB.csv data/MuHA2B-940nm/100E/LIV/MuHA2B-940nm-100E-25.0°C-20231019-092200-PM100USB.csv
#     "2.5μm",
#     "2.5μm",
#     "3μm",
#     "3.5μm",
#     "3.5μm",
#     "4μm",
#     "4μm",
#     "4.5μm",
#     "5μm",
#     "10μm",
# ]
# title = "MuHA2B 940nm classical VCSELs"


# legend_list = [  # 1550nm 6μm BTJ
#     # python plot_liv.py data/PW2024-1550nm/36S6GSG/LIV/spb1-1550nm-36S6GSG-25.0°C-20231114-062932-PM100USB.csv data/spb1-1550nm/47M6GSG/LIV/spb1-1550nm-47M6GSG-25.0°C-20231211-120450-PM100USB.csv data/PW2024-1550nm/46L6GSG/LIV/spb1-1550nm-46L6GSG-25.0°C-202308-0-PM100USB.csv
#     # data/PW2024/36S6GSG 47M6GSG 46L6GSG
#     "S-type 6 μm BTJ",
#     "M-type 6 μm BTJ",
#     "L-type 6 μm BTJ",
# ]
# title = "6 μm BTJ 1550 nm VCSELs"

# legend_list = [  # 1550nm 8μm BTJ
#     # python plot_liv.py data/PW2024-1550nm/36S8GSG/LIV/spb1-1550nm-36S8GSG-25.0°C-20231114-065049-PM100USB.csv data/PW2024-1550nm/36M8GSG/LIV/spb1-1550nm-36M8GSG-25.0°C-20231114-064733-PM100USB.csv data/PW2024-1550nm/36L8GSG/LIV/spb1-1550nm-36L8GSG-25.0°C-20231114-064346-PM100USB.csv
#     # data/PW2024/36S8GSG 36M8GSG 36L8GSG
#     "S-type 8 μm BTJ",
#     "M-type 8 μm BTJ",
#     "L-type 8 μm BTJ",
# ]
# title = "8 μm BTJ 1550 nm VCSELs"

# legend_list = None

# TITLE:
# title = "1550nm 5μm BTJ"
plot_title = True

# limits
powerlim = [0, 5.5]
voltagelim = [0, 3.2]
currentlim = [0, 35]

# legendfontsize = 5
legendfontsize = 7
# legendfontsize = 5.7
legendloc = 0

fig = plt.figure(figsize=(11.69 / 2, 8.27 / 2))
ax = fig.add_subplot(111)
# Creating figure
ax2 = ax.twinx()
if plot_title:
    plt.title(title)

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

    dfiv = pd.read_csv(file)

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
    lns1 = ax.plot(i, l, "-", label=f"{leg} Output power, mW", alpha=0.6)
    # Creating Twin axes for dataset_1
    lns2 = ax2.plot(i, v, ":", label=f"{leg} Voltage, V", alpha=0.6)

    # legend
    lns.extend(lns1 + lns2)
    labs.extend([l.get_label() for l in lns1 + lns2])

# Setting Y limits
ax.set_ylim(powerlim)  # Power
ax.set_xlim(currentlim)  # Current
ax2.set_ylim(voltagelim)  # Voltage

ax.minorticks_on()
ax2.minorticks_on()
ax.grid(which="both")  # adding grid

# Adding legend
ax.legend(lns, labs, loc=legendloc, fontsize=legendfontsize)
# ax2.legend(loc=0)

# fig.suptitle("-".join(name_from_dir.split("-")[:-1]) + " " + "-".join(vcsel))

if not os.path.exists("reports/"):  # make directories
    os.makedirs("reports/")
# plt.savefig("reports/" + "-".join(vcsel) + ".png", dpi=600)  # save figure
plt.savefig("reports/" + title + ".png", dpi=600)  # save figure
