#!/usr/bin/env python3

import time
import sys
import os
import pyvisa
import numpy as np
import pyfiglet
from rich import print
from configparser import ConfigParser
import tkinter as tk
from tkinter import ttk

comand_line = True

config = ConfigParser()
config.read("config.ini")
instruments_config = config["INSTRUMENTS"]

if comand_line:
    wavelength = sys.argv[1]
    rm = pyvisa.ResourceManager()
    PM100USB = rm.open_resource(
        instruments_config["Thorlabs_PM100USB_address"],
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
        print(f"\n{output_power:3.6f} mW")
        print("Ctrl-C to cancel")
else:
    rm = pyvisa.ResourceManager()
    PM100USB = rm.open_resource(
        instruments_config["Thorlabs_PM100USB_address"],
        write_termination="\r\n",
        read_termination="\n",
    )
    PM100USB.write("power:dc:unit W")  # set power units
    if len(sys.argv) != 1:
        wavelength = sys.argv[1]
        PM100USB.write(f"sense:corr:wav {wavelength}")  # set wavelength

    window = tk.Tk()
    window.title("Powermeter")
    frm = ttk.Frame(window, padding=10)
    frm.grid()
    lbltext = tk.StringVar("0.000000 mW")
    lbl = ttk.Label(frm, textvariable=lbltext, font=("Arial Bold", 50)).grid(
        column=0, row=0
    )
    window.geometry("1000x200")

    # row 1: start stop quit
    def stop_loop():
        window.poll = False

    loop_trigger = False

    def loop():
        global loop_trigger
        if loop_trigger:
            output_power = float(PM100USB.query("measure:power?")) * 1000  # mW
            lbltext.set(f"{output_power:3.6f} mW")
            window.after(30, loop)
            window.update()

    def start_loop():
        if window.poll:
            print("Polling")
            output_power = float(PM100USB.query("measure:power?")) * 1000  # mW
            lbl.configure(text=f"{output_power:3.6f} mW")
            window.after(30, start_loop)
        else:
            print("Stopped long running process.")

    startButton = ttk.Button(frm, text="Start", command=start_loop).grid(
        column=1, row=1
    )
    stopButton = ttk.Button(window, text="Pause", command=stop_loop).grid(
        column=2, row=1
    )
    quitButton = ttk.Button(frm, text="Quit", command=window.destroy).grid(
        column=3, row=1
    )

    # row 2: wavelength
    def set_wavelength(wl):
        PM100USB.write(f"sense:corr:wav {wl}")  # set wavelength
        lbl.configure(text=f"{wl} nm")

    wlone = ttk.Button(window, text="850 nm", command=lambda: set_wavelength(850)).grid(
        column=1, row=2
    )
    wltwo = ttk.Button(window, text="940 nm", command=lambda: set_wavelength(940)).grid(
        column=2, row=2
    )
    wlthree = ttk.Button(
        window, text="1300 nm", command=lambda: set_wavelength(1300)
    ).grid(column=3, row=2)
    wlfour = ttk.Button(
        window, text="1550 nm", command=lambda: set_wavelength(1550)
    ).grid(column=4, row=2)

    startButton.pack()
    stopButton.pack()
    quitButton.pack()
    wlone.pack()
    wltwo.pack()
    wlthree.pack()
    wlfour.pack()

    window.mainloop()
