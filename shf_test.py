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
import time
from pathlib import Path
from configparser import ConfigParser

from src.measure_optical_spectra import measure_osa
from src.shf import SHF


def update_att_temperature(set_temperature, ATT_A160CMI=None):
    config = ConfigParser()
    config.read("config.ini")
    other_config = config["OTHER"]
    if not ATT_A160CMI:
        instruments_config = config["INSTRUMENTS"]
        rm = pyvisa.ResourceManager()
        # rm = pyvisa.ResourceManager('@py') # for pyvisa-py
        ATT_A160CMI = rm.open_resource(
            instruments_config["ATT_A160CMI_address"],
            write_termination="\r\n",
            read_termination="\n",
        )
    temp_for_att = ""
    if set_temperature >= 0 and set_temperature < 10:
        temp_for_att = "+00" + str(int(round(set_temperature, ndigits=2) * 100))
    elif set_temperature >= 10 and set_temperature < 100:
        temp_for_att = "+0" + str(int(round(set_temperature, ndigits=2) * 100))
    elif set_temperature >= 100 and set_temperature <= float(
        other_config["temperature_limit"]
    ):
        temp_for_att = "+" + str(int(round(set_temperature, ndigits=2) * 100))
    else:
        ATT_A160CMI.write("TS=+02500")
        Exception("Temperature is set too high!")
    # while len(temp_for_att) < 6:  # TODO check whether we need it
    #     temp_for_att = temp_for_att + "0"
    ATT_A160CMI.write(f"TS={temp_for_att}")
    stable = False
    counter_stability = 0
    sign = 0
    while not stable:
        time.sleep(10)
        current_temperature_str = str(ATT_A160CMI.query("TA?"))
        if current_temperature_str[3] == "+":
            sign = 1
        elif current_temperature_str[3] == "-":
            sign = -1
        current_temperature = sign * (
            float(current_temperature_str[4:7])
            + float(current_temperature_str[7:9]) / 100
        )
        error = abs(current_temperature - set_temperature)
        if error < 0.04:
            counter_stability += 1
        else:
            counter_stability = 0
        if counter_stability == 4:
            stable = True
        print(
            f"Temperature set to {set_temperature},\t measured {current_temperature},\t stabilizing [{counter_stability}/4]\r"
        )


def print_help():
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    ctx.exit()


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "-p",
    "--piezo",
    is_flag=True,
    show_default=True,
    default=False,
    help="Adjust fiber by piezo KCubes",
)
@click.option(
    "-q",
    "--quest",
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
@click.option(
    "-c",
    "--combined",
    is_flag=True,
    show_default=True,
    default=False,
    help="Adjust fiber + measure LIV + open attenuator + measure optical spectra with OSA + close attenuator",
)
@click.option(
    "-t",
    "--temperature",
    is_flag=True,
    show_default=True,
    default=False,
    help="Combined measurements at different temperatures",
)
@click.argument("arguments", nargs=-1)
def analyze(quest, liv, osa, piezo, combined, temperature, arguments):
    start_time = datetime.datetime.now()
    WaferID, Wavelength, Coordinates, Temperature = arguments
    print("WaferID:    ", WaferID)
    print("Wavelength: ", Wavelength, "nm")
    print("Coordinates:", Coordinates)
    print("Temperature:", Temperature, "°C")
    if not any((quest, liv, osa, piezo, combined, temperature)):
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
        print("For example for LIV run 'python shf_test.py -l waferid 1550 00C9 25'")
        exit()
    shfclass = SHF(
        waferid=WaferID,
        wavelength=Wavelength,
        coordinates=Coordinates,
        temperature=float(Temperature),
    )
    try:
        if quest:
            shfclass.test()
        elif combined:
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

            shfclass.rst_current_source()
            assert shfclass.rst_attenuator() == True
            shfclass.connect_kcubes()
            shfclass.start_optimizing_fiber()
            shfclass.gently_apply_current(0)
            shfclass.measure_liv_with_attenuator()
            shfclass.attenuator_command(":CONT:MODE ATT")
            shfclass.attenuator_command(":INP:ATT MIN")
            shfclass.open_attenuator_shutter()
            alarm = measure_osa(
                WaferID,
                Wavelength,
                Coordinates,
                float(Temperature),
                Keysight_B2901A=Keysight_B2901A,
                YOKOGAWA_AQ6370D=YOKOGAWA_AQ6370D,
            )
        elif temperature:
            temperature_list = [25, 35, 45, 55, 65, 75, 85]
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
            ATT_A160CMI = rm.open_resource(
                instruments_config["ATT_A160CMI_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
            shfclass.rst_current_source()
            assert shfclass.rst_attenuator() == True
            shfclass.connect_kcubes()
            for i, set_temperature in enumerate(temperature_list, start=1):
                print(f"[{i}/{len(temperature_list)}] {set_temperature} degree Celsius")
                update_att_temperature(set_temperature, ATT_A160CMI=ATT_A160CMI)
                shfclass.temperature = float(set_temperature)
                if i == 1:
                    ask = True
                else:
                    ask = False
                shfclass.start_optimizing_fiber(ask=ask)
                shfclass.gently_apply_current(0)
                shfclass.measure_liv_with_attenuator()
                shfclass.attenuator_command(":CONT:MODE ATT")
                shfclass.attenuator_command(":INP:ATT MIN")
                shfclass.attenuator_shutter("open")
                alarm = measure_osa(
                    WaferID,
                    Wavelength,
                    Coordinates,
                    float(Temperature),
                    Keysight_B2901A=Keysight_B2901A,
                    YOKOGAWA_AQ6370D=YOKOGAWA_AQ6370D,
                )
                shfclass.attenuator_shutter("close")
                if alarm:
                    break
        elif piezo:
            shfclass.rst_current_source()
            assert shfclass.rst_attenuator() == True
            shfclass.connect_kcubes()
            shfclass.start_optimizing_fiber()
        elif liv:
            shfclass.rst_current_source()
            assert shfclass.rst_attenuator() == True
            shfclass.measure_liv_with_attenuator()
        elif osa:
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
                float(Temperature),
                Keysight_B2901A=Keysight_B2901A,
                YOKOGAWA_AQ6370D=YOKOGAWA_AQ6370D,
            )
    finally:
        if temperature:
            time.sleep(1)
            ATT_A160CMI.write("TS=+02500")
        shfclass.attenuator_shutter("close")
        shfclass.shf_turn_off()
        shfclass.gently_apply_current(0)
        shfclass.save_logs()
        end_time = datetime.datetime.now()
        print(f"Duration: {end_time - start_time}")


if __name__ == "__main__":
    analyze()
