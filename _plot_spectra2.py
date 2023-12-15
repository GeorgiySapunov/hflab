#!/usr/bin/env python3
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

file1 = "PW2024-1550nm-57M6GSG-25.0°C-20231114-074029-YOKOGAWA_AQ6370D-OS.csv"
file2 = "PW2024-1550nm-57M6GSG-55.0°C-20231114-081326-YOKOGAWA_AQ6370D-OS.csv"
file3 = "PW2024-1550nm-57M6GSG-85.0°C-20231114-084201-YOKOGAWA_AQ6370D-OS.csv"
directory = "data/PW2024-1550nm/57M6GSG/OSA/"
currents1 = [3, 4, 5, 8, 10, 12, 15, 17]
currents2 = [4, 5, 8, 10, 12, 15]
currents3 = [8, 10, 12]
spectra_xlim_left = 1565
spectra_xlim_right = 1577


osdf1 = pd.read_csv(directory + file1)
osdf2 = pd.read_csv(directory + file2)
osdf3 = pd.read_csv(directory + file3)
columns1 = [f"Intensity at {i:.2f} mA, dBm" for i in currents1]
columns2 = [f"Intensity at {i:.2f} mA, dBm" for i in currents2]
columns3 = [f"Intensity at {i:.2f} mA, dBm" for i in currents3]

# 5. make spectra plots and save .png
# Make spectra figures
# Creating figure
fig = plt.figure(figsize=(11.69, 8.27))
# Plotting dataset
ax = fig.add_subplot(311)
ax.set_title("6 μm BTJ 1550 nm VCSELs at 25 °C")
for current, column in zip(currents1, columns1):
    # spectrum line
    ax.plot(
        osdf1["Wavelength, nm"],
        osdf1[column],
        "-",
        alpha=0.5,
        label=f"{current} mA",
    )
# Adding title
# adding grid
ax.grid(which="both")  # adding grid
ax.minorticks_on()
# Adding labels
# ax.set_xlabel("Wavelength, nm")
ax.set_ylabel("Intensity, dBm")
# Adding legend
# ax.legend(loc=0, prop={"size": 4})
ax.legend(loc=2)
ax.set_ylim(-80, 0)
ax.set_xlim(spectra_xlim_left, spectra_xlim_right)
#
ax2 = fig.add_subplot(312)
ax2.set_title("6 μm BTJ 1550 nm VCSELs at 55 °C")
for current, column in zip(currents2, columns2):
    # spectrum line
    ax2.plot(
        osdf2["Wavelength, nm"],
        osdf2[column],
        "-",
        alpha=0.5,
        label=f"{current} mA",
    )
# Adding title
# adding grid
ax2.grid(which="both")  # adding grid
ax2.minorticks_on()
# Adding labels
# ax2.set_xlabel("Wavelength, nm")
ax2.set_ylabel("Intensity, dBm")
# Adding legend
# ax.legend(loc=0, prop={"size": 4})
ax2.legend(loc=2)
ax2.set_ylim(-80, 0)
ax2.set_xlim(spectra_xlim_left, spectra_xlim_right)
#
ax3 = fig.add_subplot(313)
ax3.set_title("6 μm BTJ 1550 nm VCSELs at 85 °C")
for current, column in zip(currents3, columns3):
    # spectrum line
    ax3.plot(
        osdf3["Wavelength, nm"],
        osdf3[column],
        "-",
        alpha=0.5,
        label=f"{current} mA",
    )
# Adding title
# adding grid
ax3.grid(which="both")  # adding grid
ax3.minorticks_on()
# Adding labels
ax3.set_xlabel("Wavelength, nm")
ax3.set_ylabel("Intensity, dBm")
# Adding legend
# ax.legend(loc=0, prop={"size": 4})
ax3.legend(loc=2)
ax3.set_ylim(-80, 0)
ax3.set_xlim(spectra_xlim_left, spectra_xlim_right)

filepath = "reports/"

if not os.path.exists("reports/"):
    os.makedirs("reports/")

plt.savefig(filepath + "optical_spectra.png", dpi=600)
plt.close()
