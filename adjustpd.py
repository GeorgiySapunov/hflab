#!/usr/bin/env python3

import time
import sys
import os
import pyvisa
import numpy as np
import pyfiglet
from rich import print

from settings import settings


wavelength = sys.argv[1]

rm = pyvisa.ResourceManager()
PM100USB = rm.open_resource(
    settings["Thorlabs_PM100USB_address"],
    write_termination="\r\n",
    read_termination="\n",
)
PM100USB.write(f"sense:corr:wav {wavelength}")  # set wavelength
PM100USB.write("power:dc:unit W")  # set power units


while True:
    output_power = float(PM100USB.query("measure:power?")) * 1000  # mW
    time.sleep(0.03)
    os.system("cls" if os.name == "nt" else "clear")
    power = pyfiglet.figlet_format(f"{output_power:3.6f}", font="epic")
    color = "blue"
    if output_power < 0.01:
        color = "blue"
    elif output_power < 0.1 and output_power >= 0.01:
        color = "yellow"
    elif output_power < 1 and output_power >= 0.1:
        color = "red"
    elif output_power > 1:
        color = "green"
    print(f"[{color}]{power}[/{color}]")
