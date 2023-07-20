#!/usr/bin/env python3
# Keysight 8163B Lightwave Multimeter
# Keysight B2901A Precision Source/Measure Unit
# Thorlabs PM100USB Power and energy meter

# import sys
# import time
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame()

# choose GPIB Channel 23 as Drain-Source
rm = pyvisa.ResourceManager()
rm.list_resources()
Keysight_B2901A = rm.open_resource('GPIB0::23::INSTR')
Keysight_B2901A.write("*RST") # The initial settings are applied by the *RST command
Keysight_B2901A.timeout = 50 # units: ms

# Keysight_B2901A.write("")

Keysight_B2901A.write(":SOUR:FUNC:MODE CURR") # Setting the Source Output Mode to current
Keysight_B2901A.write(":SENS:CURR:PROT 0.1") # Setting the Limit/Compliance Value 100 mA
Keysight_B2901A.write(":SENS:VOLT:PROT 10") # Setting the Limit/Compliance Value 10 V

Keysight_B2901A.write(":OUTP ON") # Measurement channel is enabled by the :OUTP ON command.

Keysight_B2901A.query("*OPC?") # synchronization
# Keysight_B2901A.write("")

for i in range(0, 0.05, 0.00001):
    Keysight_B2901A.write(":SOUR:CURR " + str(i)) # Outputs i mA immediately
    Keysight_B2901A.query("*OPC?") # synchronization
    voltage = Keysight_B2901A.query_ascii_values("MEAS:VOLT?")
    Keysight_B2901A.query("*OPC?") # synchronization
    current = Keysight_B2901A.query_ascii_values("MEAS:CURR?")
    Keysight_B2901A.query("*OPC?") # synchronization
    current = Keysight_B2901A.query_ascii_values("MEAS:CURR?")
    Keysight_B2901A.query("*OPC?") # synchronization

    df.lock[i]["Voltage, V"] = voltage
    df.lock[i]["Current, V"] = current*1000

    print(f"curret set at {i*1000} mA, current: {current*1000} mA, voltage: {voltage}")
    if float(current) != float(i):
        print("WARNING: i is not equal to current!")

Keysight_B2901A.write(":OUTP OFF") # Measurement is stopped by the :OUTP OFF command.

print(df.head())
