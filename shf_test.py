#!/usr/bin/env python3

import pyvisa
import sys
import re
import os
import yaml
import numpy as np
import pandas as pd
import click
import colorama
import datetime
from pathlib import Path

from src.shf import SHF


def print_help():
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    ctx.exit()


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "-t",
    "--test",
    is_flag=True,
    show_default=True,
    default=False,
    help="test the system",
)
@click.option(
    "-l",
    "--liv",
    is_flag=True,
    show_default=True,
    default=False,
    help="Measure LIV with attenuator",
)
@click.option(
    "-o",
    "--osa",
    is_flag=True,
    show_default=True,
    default=False,
    help="Measure optical spectra with OSA",
)
@click.argument("arguments", nargs=-1)
def analyze(test, liv, osa, arguments):
    WaferID, Wavelength, Coordinates, Temperature = arguments
    print("WaferID: ", WaferID)
    print("Wavelength: ", Wavelength, " nm")
    print("Coordinates: ", Coordinates)
    print("Temperature: ", Temperature, " Deg. Celsius")
    if not any((test, liv, osa)):
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
        print_help()
        print("for LIV e.g. run 'python shf_test.py -l waferid 1550 00C9 25'")
        exit()
    shfclass = SHF(
        waferid=WaferID,
        wavelength=Wavelength,
        coordinates=Coordinates,
        temperature=Temperature,
    )
    try:
        if test:
            shfclass.test()
        elif liv:
            shfclass.rst_current_source()
            print("rst_current_source done")
            assert shfclass.rst_attenuator() == True
            print("rst_attenuator done")
            shfclass.measure_liv_with_attenuator()
        elif osa:
            from src.measure_optical_spectra import measure_osa
            from configparser import ConfigParser

            config = ConfigParser()
            config.read("config.ini")
            instruments_config = config["INSTRUMENTS"]
            rm = pyvisa.ResourceManager()
            YOKOGAWA_AQ6370D = rm.open_resource(
                instruments_config["YOKOGAWA_AQ6370D_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            Keysight_B2901A = rm.open_resource(
                instruments_config["Keysight_B2901A_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            alarm = measure_osa(
                WaferID,
                Wavelength,
                Coordinates,
                Temperature,
                Keysight_B2901A=Keysight_B2901A,
                YOKOGAWA_AQ6370D=YOKOGAWA_AQ6370D,
            )
    finally:
        shfclass.gently_apply_current(0)
        shfclass.save_logs()


if __name__ == "__main__":
    start_time = datetime.datetime.now()
    analyze()
    end_time = datetime.datetime.now()
    print(f"Duration: {end_time - start_time}")
