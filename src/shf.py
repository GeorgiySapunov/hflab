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
import colorama
from configparser import ConfigParser
from pathlib import Path
from termcolor import colored


class SHF:
    current = 0
    clksrc_frequency = 20  # GHz
    clksrc_output = 0  # OFF
    bpg_pattern = "PRBS7"
    bpg_output = {
        "channel1": 0,
        "channel2": 0,
        "channel3": 0,
        "channel4": 0,
        "channel5": 0,
        "channel6": 0,
    }  # OFF
    bpg_amplitude = {
        "channel1": 650,
        "channel2": 650,
        "channel3": 650,
        "channel4": 650,
        "channel5": 650,
        "channel6": 650,
    }  # mV
    bpg_preemphasis = {"TAP0": 0, "TAP2": 0, "TAP3": 0}  # TAP1 is Main
    dac_amplitude = 150  # mV
    dac_output = 0  # OFF
    attenuator_powerin = 0
    attenuator_powerout = 0

    def __init__(
        self,
        waferid,
        wavelength,
        coordinates,
        temperature,
        current_source=None,
        attenuator=None,
        bpg=None,
        dac=None,
        clksrc=None,
        ea=None,
        amplifier_power=1,
        pam4=1,
        mainframe=1,
    ):
        self.waferid = waferid
        self.wavelength = wavelength
        self.coordinates = coordinates
        self.temperature = temperature

        colorama.init()
        config = ConfigParser()
        config.read("config.ini")
        self.dirpath = (
            Path.cwd() / "data" / f"{waferid}-{wavelength}nm" / f"{coordinates}"
        )

        instruments_config = config["INSTRUMENTS"]
        rm = pyvisa.ResourceManager()
        if current_source:
            self.current_source = current_source
        else:
            self.current_source = rm.open_resource(
                instruments_config["Keysight_B2901A_address"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if attenuator:
            self.attenuator = attenuator
        else:
            self.attenuator = rm.open_resource(
                instruments_config["Attenuator_EXPO_LTB1"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if bpg:
            self.bpg = bpg
        else:
            self.bpg = rm.open_resource(
                instruments_config["BPG_SHF12105A"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if dac:
            self.dac = dac
        else:
            self.dac = rm.open_resource(
                instruments_config["DAC_SHF614C"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if clksrc:
            self.clksrc = clksrc
        else:
            self.clksrc = rm.open_resource(
                instruments_config["CLKSRC_SHF78122B"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if ea:
            self.ea = ea
        else:
            self.ea = rm.open_resource(
                instruments_config["EA_SHF11104A"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if amplifier_power:
            self.amplifier_power = amplifier_power
        else:
            self.amplifier_power = rm.open_resource(
                instruments_config["PowerSource_RS_HMP2000"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if not pam4:
            self.pam4 = pam4
        else:
            self.pam4 = rm.open_resource(
                instruments_config["PAM4_SHF616C"],
                write_termination="\r\n",
                read_termination="\n",
            )
        if mainframe:
            self.mainframe = mainframe
        else:
            self.mainframe = rm.open_resource(
                instruments_config["Mainframe_SHF10000A"],
                write_termination="\r\n",
                read_termination="\n",
            )

    def shf_command(self, command: str):
        """Sends a commands to a relative SHF equipment and querry the result"""
        if command.startswith("BPG:"):
            responce = str(self.bpg.query(command))
        elif command.startswith("DAC:"):
            responce = str(self.dac.query(command))
        elif command.startswith("CLKSRC:"):
            responce = str(self.clksrc.query(command))
        elif command.startswith("EA:"):
            responce = str(self.ea.query(command))
        if command != responce:
            print(command + "\t" + responce)
        time.sleep(0.01)  # TODO: do we need this?
        return responce

    def rst_current_source(self):
        """Reset and apply initial settings to the current source (Keysight_B2901A)"""
        self.current_source.write("*RST")
        self.current_source.write(":SOUR:FUNC:MODE CURR")
        self.current_source.write(":SENS:CURR:PROT 0.1")
        self.current_source.write(":SENS:VOLT:PROT 10")
        self.current_source.write(":OUTP OFF")
        self.current_source.write(":SOUR:CURR 0")
        self.current = 0

    def gently_apply_current(self, target_current_mA: float):
        """Gradually apply current.
        Turn off the source at 0 mA. Turn on the source automatically (if self.current == 0 at the start).
        It Reads/saves the current value in self.current.
        TODO: can we get the output status directly from the current source?"""
        if self.current == 0 and target_current_mA > 0:
            self.current_source.write(f":SOUR:CURR 0.01")
            self.current_source.write(":OUTP ON")
            self.current = 1
            voltage = float(self.current_source.query("MEAS:VOLT?"))
            if voltage > 9.8:
                self.set_alarm("The contact is bad. Please, check the probe")
        elif self.current == 0 and target_current_mA == 0:
            self.current_source.write(":OUTP OFF")
            self.current_source.write(":SOUR:CURR 0.001")
            return

        current_measured = float(self.current_source.query("MEAS:CURR?")) * 1000  # mA
        if current_measured > target_current_mA:
            step = -0.1
        elif current_measured < target_current_mA:
            step = 0.1
        for current_set in np.arange(current_measured, target_current_mA, step):
            self.current_source.write(f":SOUR:CURR {str(current_set/1000)}")
            print(f"Current set: {current_set:3.1f} mA", end="\r")
            time.sleep(0.01)
        self.current_source.write(f":SOUR:CURR {str(target_current_mA/1000)}")

        if target_current_mA == 0:
            self.current_source.write(":OUTP OFF")
            self.current = 0
            self.current_source.write(":SOUR:CURR 0.001")
            return
        else:
            current_measured = (
                float(self.current_source.query("MEAS:CURR?")) * 1000
            )  # mA
            current_error = abs(target_current_mA - current_measured)
            if current_error < 0.03:
                self.current = current_measured
            else:
                self.set_alarm(
                    f"Current set: {target_current_mA:.2f} mA\tCurrent measured: {current_measured:.2f} mA"
                )

    def set_alarm(self, message: str):
        self.alarm = message
        print(colored("ALARM: " + message), "red")
        self.gently_apply_current(0)
        exit()

    def shf_init(self):
        bpg_commands = [
            "BPG:PREEMPHASIS=ENABLE:OFF;",
            "BPG:FIRFILTER=ENABLE:OFF;",
            "BPG:ALLOWNEWCONNECTIONS=ON;",
            "BPG:OUTPUTLEVEL=0;",
            "BPG:AMPLITUDE=Channel1:650 mV,Channel2:650 mV,Channel3:650 mV,Channel4:650 mV,Channel5:650 mV,Channel6:650 mV,Channel7:500 mV,Channel8:500 mV;",
            "BPG:OUTPUT=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF:",
            "BPG:PATTERN=Channel1:PRBS7,Channel2:PRBS7,Channel3:PRBS7,Channel4:PRBS7,Channel5:PRBS7,Channel6:PRBS7;Channel7:PRBS7,Channel8:PRBS7;",
            "BPG:ERRORINJECTION=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF;",
            "BPG:DUTYCYCLEADJUST=Channel1:0,Channel2:0,Channel3:0,Channel4:0,Channel5:0,Channel6:0,Channel7:0,Channel8:0;",
            "BPG:CLOCKINPUT=FULL;",
            "BPG:SELECTABLECLOCK=4;",
            "BPG:SELECTABLEOUTPUT=SELECTABLECLOCK"
            "BPG:USERSETTINGS=SCC.PATTERN TYPE:PRBS7"
            "BPG:FIRFILTER=GO:!PRBS7,G1:!PRBS7;",
        ]
        dac_commands = [
            "DAC:SYMMETRY=VALUE:0.500;",
            "DAC:OUTPUT=STATE:DISABLED;",
        ]
        clksrc_commands = [
            "CLKSRC:OUTPUT=OFF;",
            "CLKSRC:AMPLITUDE=3.0;",
            "CUKSRC:FREQUENCY=20000000 Hz;",
            "CLKSRC:TRIGGER=MODE:CLKDIV2,MAX;",
            "CLKSRC:REFERENCE=SOURCE:INTERNAL;",
            "CLKSRC:SSCMODE=MODE:OFF;",
            "CLKSRC:SSCDEVIATION=VALUE:0.00;",
            "CUKSRC:SSCFREQUENCY=VALUE:20000;",
            "CLKSRC:JITTERSOURCE=STATE:OFF;",
        ]
        for command in bpg_commands + dac_commands + clksrc_commands:
            self.shf_command(command)
        self.bpg_output = {
            "channel1": 0,
            "channel2": 0,
            "channel3": 0,
            "channel4": 0,
            "channel5": 0,
            "channel6": 0,
        }  # OFF
        self.bpg_amplitude = {
            "channel1": 650,
            "channel2": 650,
            "channel3": 650,
            "channel4": 650,
            "channel5": 650,
            "channel6": 650,
        }  # mV
        self.dac_output = 0
        self.clksrc_frequency = 20  # GHz
        self.clksrc_output = 0  # OFF

    def shf_nrzsetup(self):
        clksrc_commands = [
            "CLKSRC:FREQUENCY=20000000000 Hz;",
            "CLKSRC:OUTPUT=ON;",
        ]
        bpg_commands = [
            "BPG:SKEW=Channel1:7 ps,Channel2:6 ps,Channel3:7 ps,Channel4:5 ps,Channel5:1 ps,Channel6:4 ps;",
            "BPG:AMPLITUDE=Channel1:650 mV,Channel2:650 mV,Channel3:650 mV,Channel4:650 mV,Channel5:650 mV,Channel6:650 mV;",
            "BPG:USERSETTINGS=SCC.GRAYCODING:12;",
            "BPG:USERSETTINGS=SCC.PATTERNTYPE:PRBS7;",
            "BPG:PREEMPHASIS=ENABLE:OFF,PAMLEVELS:NONE;",
            "BPG:PATTERN=Channel1:PRBS7,Channel2:PRBS7,Channel3:PRBS7,Channel4:PRBS7,Channel5:PRBS7,Channel6:PRBS7;",
            "BPG:FIRFILTER=ENABLE:OFF;",
            "BPG:PAMLEVELS=SCC.PAMORDER:2;",
            "BPG:PAMLEVELS=SCC.L0:0.00%,SCC.L1:100.00%;",
            "BPG:PREEMPHASIS=DAC:SCC,PAMLEVELS:SCC;",
            "BPG:FIRFILTER=G0:!PRBS7,G1:!PRBS7;",
            "BPG:FIRFILTER=G0:!PRBS7,G1:!PRBS7;",
            "BPG:FIRFILTER=FUNCTION:1*y+0;",
            "BPG:PREEMPHASIS=ENABLE:ON;",
            "BPG:OUTPUT=Channel1:ON,Channel2:ON,Channel3:ON,Channel4:ON,Channel5:ON,Channel6:ON;",
        ]
        for command in clksrc_commands + bpg_commands:
            self.shf_command(command)
        self.clksrc_frequency = 20  # GHz
        self.clksrc_output = 1  # ON
        self.bpg_output = {
            "channel1": 1,
            "channel2": 1,
            "channel3": 1,
            "channel4": 1,
            "channel5": 1,
            "channel6": 1,
        }  # ON
        self.bpg_amplitude = {
            "channel1": 650,
            "channel2": 650,
            "channel3": 650,
            "channel4": 650,
            "channel5": 650,
            "channel6": 650,
        }  # mV

    def shf_set_preemphasis(self, tap_index: int = 0, value: int = 0):
        """tap_index should be 0, 2 or 3
        TAP1 is MAIN"""
        if tap_index in (0, 2, 3):
            self.shf_command(f"BPG:PREEMPHASIS=TAP{tap_index}:{value} %;")
            tap_str = f"TAP{tap_index}"
            self.bpg_preemphasis[tap_str] = value
        else:
            print("tap_index should be 0, 2 or 3; TAP1 is MAIN")

    def shf_set_amplitude(self, target_amplitude: int):
        D0_onemV = 0.01587302
        D0 = float(target_amplitude * D0_onemV)
        clksrc_commands = [
            f"DAC:SIGNAL=ALIAS:D0,VALUE:{D0:.4f};",
            f"DAC:SIGNAL=ALIAS:D1,VALUE:{D0*2:4f};",
            f"DAC:SIGNAL=ALIAS:D2,VALUE:{D0*4:4f};",
            f"DAC:SIGNAL=ALIAS:D3,VALUE:{D0*8:4f};",
            f"DAC:SIGNAL=ALIAS:D4,VALUE:{D0*16:4f};",
            f"DAC:SIGNAL=ALIAS:D5,VALUE:{D0*32:4f};",
            "DAC:OUTPUT=STATE:ENABLED;",
        ]
        for command in clksrc_commands:
            self.shf_command(command)
        self.dac_amplitude = target_amplitude
        self.dac_output = 1

    def shf_turn_off(self):
        clksrc_commands = [
            "CLKSRC:FREQUENCY=20000000000 Hz.",
            "CLKSRC:OUTPUT=OFF",
        ]
        bpg_commands = [
            "BPG:PREEMPHASIS=TAP0:0 %;",
            "BPG:PREEMPHASIS=TAP2:0 %;",
            "BPG:PREEMPHASIS=TAP3:0 %;",
            "BPG:OUTPUT=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF:",
        ]
        dac_commands = [
            "DAC:OUTPUT=STATE:DISABLED;",
        ]
        for command in clksrc_commands + bpg_commands + dac_commands:
            self.shf_command(command)
        self.clksrc_frequency = 20  # GHz
        self.clksrc_output = 0  # OFF
        self.bpg_output = {
            "channel1": 0,
            "channel2": 0,
            "channel3": 0,
            "channel4": 0,
            "channel5": 0,
            "channel6": 0,
        }  # OFF
        self.bpg_preemphasis = {"TAP0": 0, "TAP2": 0, "TAP3": 0}  # TAP1 is Main
        self.dac_output = 0
