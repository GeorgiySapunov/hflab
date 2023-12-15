#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# list_of_files = [
#     "PW2024-1550nm-57M6GSG-25.0°C-20231114-074029-YOKOGAWA_AQ6370D-OS.csv",
#     "PW2024-1550nm-57M6GSG-35.0°C-20231114-075221-YOKOGAWA_AQ6370D-OS.csv",
#     # 47m6gsg
#     "PW2024-1550nm-57M6GSG-45.0°C-20231130-045357-YOKOGAWA_AQ6370D-OS.csv",
#     "PW2024-1550nm-57M6GSG-55.0°C-20231130-050235-YOKOGAWA_AQ6370D-OS.csv",
#     # "PW2024-1550nm-57M6GSG-65.0°C-20231130-051353-YOKOGAWA_AQ6370D-OS.csv",
#     "PW2024-1550nm-57M6GSG-65.0°C-20231130-052227-YOKOGAWA_AQ6370D-OS.csv",
#     # "PW2024-1550nm-57M6GSG-75.0°C-20231130-053127-YOKOGAWA_AQ6370D-OS.csv",
#     # 57m6gsg
#     # "PW2024-1550nm-57M6GSG-45.0°C-20231114-080258-YOKOGAWA_AQ6370D-OS.csv",
#     # "PW2024-1550nm-57M6GSG-55.0°C-20231114-081326-YOKOGAWA_AQ6370D-OS.csv",
#     # "PW2024-1550nm-57M6GSG-65.0°C-20231115-073447-YOKOGAWA_AQ6370D-OS.csv",
#     "PW2024-1550nm-57M6GSG-75.0°C-20231114-083322-YOKOGAWA_AQ6370D-OS.csv",
#     "PW2024-1550nm-57M6GSG-85.0°C-20231114-084201-YOKOGAWA_AQ6370D-OS.csv",
# ]
# list_of_temp = [25, 35, 45, 55, 65, 75, 85]

# list_of_currents = [
#     [3, 4, 5, 8, 10, 12, 15, 18],
#     [15],
#     [15],
#     [15],
#     [15],
#     [15],
#     [],
# ]

# # 47M6GSG
# list_of_files = [
#     "spb1-1550nm-47M6GSG-25.0°C-20231212-122155-YOKOGAWA_AQ6370D-OS.csv",
#     "spb1-1550nm-47M6GSG-35.0°C-20231212-123423-YOKOGAWA_AQ6370D-OS.csv",
#     "spb1-1550nm-47M6GSG-45.0°C-20231212-124321-YOKOGAWA_AQ6370D-OS.csv",
#     "spb1-1550nm-47M6GSG-55.0°C-20231212-125526-YOKOGAWA_AQ6370D-OS.csv",
#     "spb1-1550nm-47M6GSG-65.0°C-20231212-130505-YOKOGAWA_AQ6370D-OS.csv",
# ]
# list_of_temp = [25, 35, 45, 55, 65, 75, 85]

# list_of_currents = [
#     [3, 4, 5, 8, 10, 12, 15, 18],
#     [15],
#     [15],
#     [15],
#     [15],
# ]

# directory = "data/spb1-1550nm/47M6GSG/OSA/"
# spectra_xlim_left = 1563
# spectra_xlim_right = 1577

# 47M6GSGlens
list_of_files = [
    "spb1-1550nm-47M6GSG-25.0°C-20231211-130146-YOKOGAWA_AQ6370D-OS.csv",
    "spb1-1550nm-47M6GSG-35.0°C-20231211-132211-YOKOGAWA_AQ6370D-OS.csv",
    "spb1-1550nm-47M6GSG-45.0°C-20231211-134300-YOKOGAWA_AQ6370D-OS.csv",
    "spb1-1550nm-47M6GSG-55.0°C-20231211-140126-YOKOGAWA_AQ6370D-OS.csv",
    "spb1-1550nm-47M6GSG-65.0°C-20231211-141357-YOKOGAWA_AQ6370D-OS.csv",
]
list_of_temp = [25, 35, 45, 55, 65, 75, 85]

list_of_currents = [
    [3, 4, 5, 8, 10, 12, 15, 18],
    [15],
    [15],
    [15],
    [],
]

directory = "data/spb1-1550nm/47M6GSG-lens/OSA/"
spectra_xlim_left = 1563
spectra_xlim_right = 1577


# Make spectra figures
# Creating figure
fig = plt.figure(figsize=(11.69, 0.5 * 8.27))
# Plotting dataset
ax = fig.add_subplot(111)
ax.set_title("M-type 6 μm BTJ 1550 nm VCSEL")
for fn, file in enumerate(list_of_files):
    osdf = pd.read_csv(directory + file)
    currents = list_of_currents[fn]
    columns = [f"Intensity at {i:.2f} mA, dBm" for i in currents]
    temperature = list_of_temp[fn]
    if fn == 0:
        line = "-"
    else:
        line = "-."
    for current, column in zip(currents, columns):
        # spectrum line
        ax.plot(
            osdf["Wavelength, nm"],
            osdf[column],
            line,
            alpha=0.5,
            label=f"{current} mA at {temperature} °C",
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

filepath = "reports/"

if not os.path.exists("reports/"):
    os.makedirs("reports/")

plt.savefig(filepath + "optical_spectra.png", dpi=600)
plt.close()
