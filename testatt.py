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
from liv import measure_liv
from osa import measure_osa

ATT_A160CMI_address = settings["ATT_A160CMI_address"]
# ATT_A160CMI_address = "USB0::0x0403::0x6001::FTDFPZD9::INSTR"
print(ATT_A160CMI_address)

# if python got less then or more then 6 parameters
# initiate pyvisa
rm = pyvisa.ResourceManager()
# rm = pyvisa.ResourceManager('@py') # for pyvisa-py
print("List of resources:")
print(rm.list_resources())
print()

# check visa addresses
for addr in rm.list_resources():
    try:
        print(addr, "-->", rm.open_resource(addr).query("*IDN?").strip())
    except pyvisa.VisaIOError:
        pass

print(
    ATT_A160CMI_address,
    "-->",
    rm.open_resource(ATT_A160CMI_address).query("TA?\r\n").strip(),
)

print(
    ATT_A160CMI_address,
    "-->",
    rm.open_resource(ATT_A160CMI_address).query("TS?\r\n").strip(),
)
