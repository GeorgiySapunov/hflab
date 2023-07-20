#!/usr/bin/env python3
# Keysight 8163B Lightwave Multimeter
# Keysight B2901A Precision Source/Measure Unit
# Thorlabs PM100USB Power and energy meter

# import sys
import time
import pyvisa
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

df = pd.DataFrame()

# choose GPIB Channel 23 as Drain-Source
rm = pyvisa.ResourceManager()
# rm = pyvisa.ResourceManager('@py') # for pyvisa-py
print(rm.list_resources())

# check visa address
for addr in rm.list_resources():
    try:
        print(addr, '-->', rm.open_resource(addr).query('*IDN?').strip())
    except visa.VisaIOError:
        pass

input() # check visa

Keysight_B2901A = rm.open_resource('GPIB0::23::INSTR')
PM100USB = rm.open_resource('') # TODO input

Keysight_B2901A.write("*RST") # The initial settings are applied by the *RST command
Keysight_B2901A.timeout = 50 # units: ms

PM100USB.write('sense:corr:wav ' + srt(1550)) # set wavelengh
PM100USB.write('power:dc:unit W')

# Keysight_B2901A.write("")

Keysight_B2901A.write(":SOUR:FUNC:MODE CURR") # Setting the Source Output Mode to current
Keysight_B2901A.write(":SENS:CURR:PROT 0.1") # Setting the Limit/Compliance Value 100 mA
Keysight_B2901A.write(":SENS:VOLT:PROT 10") # Setting the Limit/Compliance Value 10 V

Keysight_B2901A.write(":OUTP ON") # Measurement channel is enabled by the :OUTP ON command.

Keysight_B2901A.query("*OPC?") # synchronization

max_power = 0

for i in range(0, 0.05, 0.00001):
    Keysight_B2901A.write(":SOUR:CURR " + str(i)) # Outputs i mA immediately
    Keysight_B2901A.query("*OPC?") # synchronization
    print(Keysight_B2901A.query("*OPC?")) # TODO
    voltage = float(Keysight_B2901A.query_ascii_values("MEAS:VOLT?"))
    Keysight_B2901A.query("*OPC?") # synchronization
    current = float(Keysight_B2901A.query_ascii_values("MEAS:CURR?"))
    Keysight_B2901A.query("*OPC?") # synchronization
    power = float(PM100USB.query('measure:power?'))
    PM100USB.query("*OPC?") # synchronization
    print(PM100USB.query("*OPC?")) # TODO
    if power > max_power:
        max_power = power


    df.lock[i]["Voltage, V"] = voltage
    df.lock[i]["Current, V"] = current*1000
    df.lock[i]["Power, W"] = power

    print(f"curret set at {i*1000} mA, current: {current*1000} mA, voltage: {voltage}")
    if current != float(i):
        print("WARNING: i is not equal to current!")

    # TODO
    if power < max_power*0.8 and i > 0.0003:
        break

# slowly decrease current
for i in range(current, 0, 0.0001):
    Keysight_B2901A.write(":SOUR:CURR " + str(i)) # Outputs i mA immediately
    time.sleep(0.2)

Keysight_B2901A.write(":OUTP OFF") # Measurement is stopped by the :OUTP OFF command.

print(df.head())
