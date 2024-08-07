# /usr/bin/env python3
import sys
import os
import re
import time
import clr  # for Thorlabs K-cube
import pyvisa
import csv
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import colorama
from configparser import ConfigParser
from pathlib import Path
from termcolor import colored

from src.measure_liv import buildplt_everything, buildplt_liv

ThorlabsCLI = [
    r"Thorlabs.MotionControl.DeviceManagerCLI.dll",
    r"Thorlabs.MotionControl.GenericMotorCLI.dll",
    r"ThorLabs.MotionControl.KCube.PiezoCLI.dll",
]
path = r"C:\Program Files\Thorlabs\Kinesis"
for dllfile in ThorlabsCLI:
    clr.AddReference(os.path.join(path, dllfile))
from Thorlabs.MotionControl.DeviceManagerCLI import *
from Thorlabs.MotionControl.GenericMotorCLI import *
from Thorlabs.MotionControl.KCube.PiezoCLI import *
from System import Decimal  # necessary for real world units


class SHF:
    """Equipment list:
    - Keysight B2901A Precision Source/Measure Unit
    - Attenuator EXPO LTB1
    - BPG SHF12105A
    - DAC SHF614C
    - CLKSRC SHF78122B
    - PAM4 SHF616C
    - EA SHF11104A
    - Mainframe SHF10000A
    - PAM4 Decoder SHF11220A
    ?- Amplifier power source?
    """

    skew = pd.read_csv(
        (Path("resources") / "skew.csv"), header=0, index_col=0
    ).transpose()
    attenuator_shutter_min_timeinterval = 3
    attenuator_shutter_prev_time = time.time()

    logs = []
    errorlogs = []

    attenuator_lins = "LINS1"
    shf_amplifier = 8  # dB
    test_current = 2  # mA
    max_optical_powermW = 9  # mW

    shf_connected = False
    current = 0
    attenuator_locked = 0
    attenuator_powerin = 0  # mW
    attenuator_powerout = 0  # dBm
    attenuator_shutter_open = None

    piezo_voltages = [-1.0, -1.0, -1.0]
    shf_connected = None
    bpg_output = {
        "channel1": None,
        "channel2": None,
        "channel3": None,
        "channel4": None,
        "channel5": None,
        "channel6": None,
    }
    bpg_amplitude = {
        "channel1": None,
        "channel2": None,
        "channel3": None,
        "channel4": None,
        "channel5": None,
        "channel6": None,
    }  # mV
    bpg_preemphasis = {"TAP0": None, "TAP2": None, "TAP3": None}  # TAP1 is Main
    bpg_pattern = None
    clksrc_frequency = None
    clksrc_output = None
    dac_amplitude = None
    dac_output = None

    ea_initiated = None

    kcube_devices = [None, None, None]

    def __init__(
        self,
        waferid,
        wavelength,
        coordinates,
        temperature,
        current_source=None,
        attenuator=None,
    ):
        self.waferid = waferid
        self.wavelength = wavelength
        self.coordinates = coordinates
        self.temperature = float(temperature)

        colorama.init()
        self.config = ConfigParser()
        self.config.read("config.ini")
        self.dirpath = (
            Path.cwd() / "data" / f"{waferid}-{wavelength}nm" / f"{coordinates}"
        )

        instruments_config = self.config["INSTRUMENTS"]
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

    def log_state(self):
        state_dict = {
            "Time": time.strftime("%Y%m%d-%H%M%S"),
            "Wafer ID": self.waferid,
            "Wavelength, nm": self.wavelength,
            "Coordinates": self.coordinates,
            "Temperature, °C": self.temperature,
            "Piezo coordinates": self.piezo_voltages,
            "Is SHF equipment connected?": self.shf_connected,
            "Current, mA": self.current,
            "Is attenuator locked?": self.attenuator_locked,
            "Input power to the attenuator, dBm": self.attenuator_powerin,
            "Output power from attemuator (set), dBm": self.attenuator_powerout,
            "Is attenuator shutter open?": self.attenuator_shutter_open,
            "Is SHF connected?": self.shf_connected,
            "BPG output": self.bpg_output,
            "BPG channel amplitude, mW": self.bpg_amplitude,
            "BPG preemphasis": self.bpg_preemphasis,
            "BPG patter": self.bpg_pattern,
            "Clock frequency, GHz": self.clksrc_frequency,
            "Clock output": self.clksrc_output,
            "DAC amplitude, mV": self.dac_amplitude,
            "RF amplification, dB": self.shf_amplifier,
            "DAC output": self.dac_output,
        }
        return json.dumps(state_dict)

    def save_logs(self, note: str = ""):
        if note:
            note = note + "_"
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
        shfdir = self.dirpath / "SHF"
        shfdir.mkdir(parents=True, exist_ok=True)
        with open(shfdir / f"{timestr}_{note}logs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.logs)
        with open(shfdir / f"{timestr}_{note}errorlogs.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerows(self.errorlogs)

    def set_alarm(self, message: str):
        """Print alarm message, turn off shf and current source, save logs, exit()"""
        self.alarm = message
        print(colored("ALARM: " + message, "red"))
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
        self.logs.append([timestr, "ALARM: " + message, self.log_state()])
        self.errorlogs.append([timestr, "ALARM: " + message, self.log_state()])
        self.shf_turn_off()
        time.sleep(0.2)  # can we remove it?
        self.gently_apply_current(0)
        self.save_logs()
        exit()

    def mW_to_dBm(self, mW: float):
        if mW == 0:
            return -np.inf
        dBm = 10 * np.log10(mW)
        return dBm

    def dBm_to_mW(self, dBm: float):
        if dBm == -np.inf:
            return 0
        mW = 10 ** (dBm / 10)
        return mW

    def Vpp_to_dBm(self, Vpp: float):
        "for sinusoidal signal and Z=50 Ohm"
        Z = 50  # Ohm
        Vrms = (1 / 2 * np.sqrt(2)) * Vpp  # sinusoidal
        mW = 10 * np.log10((Vrms**2) / Z) + 30
        return self.mW_to_dBm(mW)

    def dBm_to_Vpp(self, dBm: float):
        "for sinusoidal signal and Z=50 Ohm"
        Z = 50  # Ohm
        Vrms = np.sqrt(Z / 1000) * (10 ** (dBm / 20))
        Vpp = Vrms * 2 * np.sqrt(2)  # sinusoidal
        return Vpp

    # current source functions
    def rst_current_source(self):
        """Reset and apply initial settings to the current source (Keysight_B2901A)"""
        self.current_source.write("*RST")
        self.current_source.write(":SOUR:FUNC:MODE CURR")
        self.current_source.write(":SENS:CURR:PROT 0.1")
        self.current_source.write(":SENS:VOLT:PROT 10")
        self.current_source.write(":OUTP OFF")
        self.current_source.write(":SOUR:CURR 0")
        self.current = 0
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "Current source settings reset is finished",
            ]
        )

    def gently_apply_current(self, target_current_mA: float):
        """Gradually apply current.
        Turn off the source at 0 mA. Turn on the source automatically (if self.current == 0 at the start).
        It Reads/saves the current value in self.current.
        TODO: can we get the output status directly from the current source?"""
        oldcurrent = float(self.current)
        if target_current_mA < 0:
            self.set_alarm(f"Target current can't be negative.")
            return
        if self.current == 0 and target_current_mA > 0:
            self.current_source.write(f":SOUR:CURR 0.001")
            self.current_source.write(":OUTP ON")
            self.current = 1
            time.sleep(0.01)
            voltage = float(self.current_source.query("MEAS:VOLT?"))
            if voltage > 9.8:
                self.set_alarm("The contact is bad. Please, check the probe")
                return  # don't need it, but let it be for clarity
        elif self.current == 0 and target_current_mA == 0:
            self.current_source.write(":OUTP OFF")
            self.current_source.write(":SOUR:CURR 0.001")
            self.logs.append(
                [
                    time.strftime("%Y%m%d-%H%M%S"),
                    f"{oldcurrent} -> {target_current_mA} mA",
                    f"0 mA",
                    f"0 V",
                ]
            )
            print(f"{oldcurrent} -> {target_current_mA} mA")
            return
        current_measured = float(self.current_source.query("MEAS:CURR?")) * 1000  # mA
        voltage = float(self.current_source.query("MEAS:VOLT?"))
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"{oldcurrent} -> {target_current_mA} mA",
                f"{current_measured} mA",
                f"{voltage} V",
            ]
        )  # Initial current
        if current_measured > target_current_mA:
            step = -0.1
        elif current_measured <= target_current_mA:
            step = 0.1
        for current_set in np.arange(current_measured, target_current_mA, step):
            self.current_source.write(f":SOUR:CURR {str(current_set/1000)}")
            print(f"Current set: {current_set:3.1f} mA", end="\r")
            time.sleep(0.01)
        self.current_source.write(f":SOUR:CURR {str(target_current_mA/1000)}")
        current_measured = float(self.current_source.query("MEAS:CURR?")) * 1000  # mA
        voltage = float(self.current_source.query("MEAS:VOLT?"))
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"{oldcurrent} -> {target_current_mA} mA",
                f"{current_measured} mA",
                f"{voltage} V",
            ]
        )  # Final current
        print(" " * 80, end="\r")
        print(
            f"{oldcurrent:3.3f} -> {target_current_mA:3.3f} mA,\t{current_measured:3.3f} mA,\t{voltage:3.3f} V"
        )
        if target_current_mA == 0:
            self.current_source.write(":OUTP OFF")
            self.current = 0
            self.current_source.write(":SOUR:CURR 0.001")
            self.logs.append(
                [
                    time.strftime("%Y%m%d-%H%M%S"),
                    f"{oldcurrent} -> {target_current_mA} mA",
                    f"0 mA",
                    f"0 V",
                ]
            )
            print(f"Current output is turned off")
            return
        elif target_current_mA > 0:
            current_measured = (
                float(self.current_source.query("MEAS:CURR?")) * 1000
            )  # mA
            current_error = abs(target_current_mA - current_measured)
            if current_error < 0.03:
                self.current = target_current_mA
                voltage = float(self.current_source.query("MEAS:VOLT?"))
            else:
                voltage = float(self.current_source.query("MEAS:VOLT?"))
                self.set_alarm(
                    f"Current set: {target_current_mA:.2f} mA\tCurrent measured: {current_measured:.2f} mA"
                )

    # Attenuator functions
    def check_attenuator_timeout(self):
        self.attenuator_status = "Unknown"
        counter = 0
        while self.attenuator_status != "READY":
            counter += 1
            self.attenuator_status = self.attenuator.query(
                self.attenuator_lins + ":STAT?"
            ).strip()
            if self.attenuator_status == "READY":
                return
            time.sleep(0.5)
            if counter == 20:
                self.set_alarm("Attenuator command timeout")

    def update_attenuator_powerin(self, sleep: float = 0):
        "get the input power measured by the attenuator in mW"
        time.sleep(sleep)
        responce = self.query_attenuator_command(":READ:SCAL:POW:DC?")  # mW
        if "(Underrange)" in responce:
            self.attenuator_powerin = 0.0
        else:
            self.attenuator_powerin = float(responce) * 10**3
        return self.attenuator_powerin

    def attenuator_command(self, command):
        "optical attenuator SCPI write"
        self.check_attenuator_timeout()
        self.attenuator.write(self.attenuator_lins + command)
        self.logs.append([time.strftime("%Y%m%d-%H%M%S"), command])
        self.check_attenuator_timeout()

    def query_attenuator_command(self, command):
        "optical attenuator SCPI query"
        self.check_attenuator_timeout()
        responce = self.attenuator.query(self.attenuator_lins + command).rstrip()
        self.logs.append([time.strftime("%Y%m%d-%H%M%S"), command, responce])
        self.check_attenuator_timeout()
        return responce

    def attenuator_shutter(self, status: str):
        """open to open without power check
        (see self.open_attenuator_shutter() to open with input power check),
        close to close attenuator shuttter"""
        while (
            time.time() - self.attenuator_shutter_prev_time
        ) < self.attenuator_shutter_min_timeinterval:
            time.sleep(1)
        if status == "open":
            self.attenuator_command(":OUTP:STAT ON")
            self.attenuator_shutter_open = True
        elif status == "close":
            self.attenuator_command(":OUTP:STAT OFF")
            self.attenuator_shutter_open = False
        self.attenuator_shutter_prev_time = time.time()

    def rst_attenuator(self):
        """Reset and apply initial settings to the attenuator
        Standard attenuator_lins = "LINS1"
        Returns True if Attenuator is Ready and not Locked.
        """
        self.attenuator_shutter("close")
        self.attenuator_command(":RST")
        time.sleep(3)  # don't open/close the shutter too often
        self.attenuator_command(f":INP:WAV {self.wavelength} NM")
        self.attenuator_command(":CONT:MODE POW")
        self.attenuator_command(":OUTP:ALC ON")  # Power tracking
        self.attenuator_command(":OUTP:APM REF")  # Reference mode.
        self.attenuator_command(
            f":OUTP:POW {self.max_optical_powermW:.3f}"
        )  # TODO Is it working? How about ":OUTP:POW MAX"?
        self.attenuator_powerout = float(
            self.query_attenuator_command(":OUTP:POW?")
        )  # dBm
        self.update_attenuator_powerin()
        self.attenuator_locked = int(
            self.query_attenuator_command(":OUTP:LOCK:STAT?")
        )  # locked state of the instrument API (1 or 0)
        if self.attenuator_locked:
            self.set_alarm("Attenuator is locked!")
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "Attenuator settings reset is finished",
                str(self.attenuator_status),
            ]
        )
        return not self.attenuator_locked and self.attenuator_status == "READY"

    def set_attenuation(self, target_value: float):
        """sets the optical attenuation"""
        self.update_attenuator_powerin()
        if target_value > self.max_optical_powermW:
            print(
                f"Power is larger then self.max_optical_powerdBm, target value is changed to {self.mW_to_dBm(self.max_optical_powermW):.3f} dBm"
            )
            target_value = self.mW_to_dBm(self.max_optical_powermW)
        if target_value > self.mW_to_dBm(self.attenuator_powerin):
            timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
            self.logs.append(
                [
                    timestr,
                    f"power in: {self.mW_to_dBm(self.attenuator_powerin)} dBm",
                    f"attenuation target: {target_value} dBm",
                    f"Optical attenuator shutter is open: {self.attenuator_shutter_open}",
                ]
            )
            self.errorlogs.append(
                [
                    timestr,
                    f"power in: {self.mW_to_dBm(self.attenuator_powerin)} dBm",
                    f"attenuation target: {target_value} dBm",
                    f"Optical attenuator shutter is open: {self.attenuator_shutter_open}",
                ]
            )
            print("Target attenuation can't be reached")
            print(f"powerin: {self.mW_to_dBm(self.attenuator_powerin)} dBm")
            print(f"target:  {target_value} dBm")
        self.attenuator_command(f":OUTP:POW {target_value:.3f} DBM")
        self.update_attenuation_data()

    def update_attenuation_data(self):
        """updates self.attenuator_powerout containing optical output power"""
        self.update_attenuator_powerin()
        self.attenuator_powerout = float(
            self.query_attenuator_command(":OUTP:POW?")
        )  # dBm

    def open_attenuator_shutter(self):
        """open to open with input power check"""
        self.attenuator_powerout = float(
            self.query_attenuator_command(":OUTP:POW?")
        )  # dBm
        if self.attenuator_powerout > self.mW_to_dBm(self.max_optical_powermW):
            print(
                f"Power is larger then {self.max_optical_powermW} mW, can't open the attenuator shutter"
            )
            self.set_attenuation(9)
            timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
            self.logs.append(
                [
                    timestr,
                    f"power in: {self.attenuator_powerin} dBm",
                    f"power out: {self.attenuator_powerout} dBm",
                    f"Optical attenuator shutter is open: {self.attenuator_shutter_open}",
                ]
            )
            self.errorlogs.append(
                [
                    timestr,
                    f"power in: {self.attenuator_powerin} dBm",
                    f"power out: {self.attenuator_powerout} dBm",
                    f"Optical attenuator shutter is open: {self.attenuator_shutter_open}",
                ]
            )
        self.attenuator_shutter("open")
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"power in: {self.attenuator_powerin} dBm",
                f"power out: {self.attenuator_powerout} dBm",
                f"Optical attenuator shutter is open: {self.attenuator_shutter_open}",
            ]
        )
        print(f"Output power: {self.attenuator_powerout} dBm, shutter is open")

    # SHF commands
    def connect_shf(
        self,
        bpg=None,
        dac=None,
        clksrc=None,
        ea=None,
        det=None,
        amplifier_power=1,
        pam4=1,
        mainframe=1,
    ):
        instruments_config = self.config["INSTRUMENTS"]
        rm = pyvisa.ResourceManager()
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
        # if det:
        #    self.det = det
        # else:
        #    self.det = rm.open_resource(
        #        instruments_config["DET_SHF11220A"],
        #        write_termination="\r\n",
        #        read_termination="\n",
        #    )
        if amplifier_power:
            self.amplifier_power = amplifier_power
        else:
            self.amplifier_power = rm.open_resource(
                instruments_config["PowerSource_RS_HMP2000"],
                write_termination="\r\n",
                read_termination="\n",
            )
        # if not pam4:
        #    self.pam4 = pam4
        # else:
        #    self.pam4 = rm.open_resource(
        #        instruments_config["PAM4_SHF616C"],
        #        write_termination="\r\n",
        #        read_termination="\n",
        #    )
        if mainframe:
            self.mainframe = mainframe
        else:
            self.mainframe = rm.open_resource(
                instruments_config["Mainframe_SHF10000A"],
                write_termination="\r\n",
                read_termination="\n",
            )
        self.shf_connected = True
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "SHF equipment is connected",
            ]
        )

    def shf_command(self, command: str):
        """Sends a commands to a relative SHF equipment and query the result"""
        #print()
        #print("command:  " + command)
        if not self.shf_connected:
            self.logs.append(
                [
                    time.strftime("%Y%m%d-%H%M%S"),
                    "SHF is not connected, connecting...",
                ]
            )
            self.connect_shf()
            self.shf_init()
        if command.startswith("BPG:"):
            responce = str(self.bpg.query(command))
        elif command.startswith("DAC:"):
            responce = str(self.dac.query(command))
        elif command.startswith("CLKSRC:"):
            responce = str(self.clksrc.query(command))
        elif command.startswith("EA:"):
            responce = str(self.ea.query(command))
        elif command.startswith("DET:"):
            responce = str(self.det.query(command))
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
        command = command.strip()
        responce = responce.strip()
        self.logs.append([timestr, command, responce])
        if command != responce:
            #print(timestr)
            #print("command:  " + command)
            print("responce:", command, responce)
            self.errorlogs.append(
                [
                    timestr,
                    command,
                    responce,
                ]
            )
            # self.errorlogs.append([timestr, command, responce, self.log_state()])
        time.sleep(0.03)  # do we need this?
        return responce

    def shf_init(self):
        bpg_commands = [
            "BPG:PREEMPHASIS=ENABLE:OFF;",
            "BPG:FIRFILTER=ENABLE:OFF;",
            "BPG:ALLOWNEWCONNECTIONS=ON;",
            "BPG:OUTPUTLEVEL=0;",
            "BPG:AMPLITUDE=Channel1:650 mV,Channel2:650 mV,Channel3:650 mV,Channel4:650 mV,Channel5:650 mV,Channel6:650 mV,Channel7:500 mV,Channel8:500 mV;",
            "BPG:OUTPUT=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF;",
            "BPG:PATTERN=Channel1:PRBS7,Channel2:PRBS7,Channel3:PRBS7,Channel4:PRBS7,Channel5:PRBS7,Channel6:PRBS7,Channel7:PRBS7,Channel8:PRBS7;",
            "BPG:ERRORINJECTION=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF;",
            "BPG:DUTYCYCLEADJUST=Channel1:0,Channel2:0,Channel3:0,Channel4:0,Channel5:0,Channel6:0,Channel7:0,Channel8:0;",
            "BPG:CLOCKINPUT=FULL;",
            "BPG:SELECTABLECLOCK=4;",
            "BPG:SELECTABLEOUTPUT=SELECTABLECLOCK;",
            "BPG:USERSETTINGS=SCC.PATTERN TYPE:PRBS7;",
            "BPG:FIRFILTER=G0:!PRBS7,G1:!PRBS7;",
        ]
        D0_onemV = 0.01587302
        D0 = float(150 * D0_onemV)
        dac_commands = [
            # "DAC:SYMMETRY=VALUE:?;",
            "DAC:OUTPUT=STATE:DISABLED;",
            f"DAC:SIGNAL=ALIAS:D0,VALUE:{D0:.2f};",
            f"DAC:SIGNAL=ALIAS:D1,VALUE:{(D0*2):.2f};",
            f"DAC:SIGNAL=ALIAS:D2,VALUE:{(D0*4):.2f};",
            f"DAC:SIGNAL=ALIAS:D3,VALUE:{(D0*8):.2f};",
            f"DAC:SIGNAL=ALIAS:D4,VALUE:{(D0*16):.2f};",
            f"DAC:SIGNAL=ALIAS:D5,VALUE:{(D0*32):.2f};",
        ]
        clksrc_commands = [
            "CLKSRC:OUTPUT=OFF;",
            "CLKSRC:AMPLITUDE=3.0;",
            "CLKSRC:FREQUENCY=20000000000 Hz;",
            "CLKSRC:TRIGGER=MODE:CLKDIV4;",
            # "CLKSRC:TRIGGER=MODE:?",
            # "CLKSRC:TRIGGER=MODE:CLKDIV2,MAX:?",
            "CLKSRC:REFERENCE=SOURCE:INTERNAL;",
            "CLKSRC:SSCMODE=MODE:OFF;",
            "CLKSRC:SSCDEVIATION=VALUE:0.00;",
            "CLKSRC:SSCFREQUENCY=VALUE:20000;",
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
        self.bpg_preemphasis = {"TAP0": 0, "TAP2": 0, "TAP3": 0}  # TAP1 is Main
        self.bpg_pattern = "PRBS7"
        self.clksrc_frequency = 20  # GHz
        self.clksrc_output = False  # OFF
        self.dac_amplitude = 150  # mV
        self.dac_output = 0
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "SHF initialization is finished",
            ]
        )

    def ea_init(self):
        self.ea.timeout = 60000
        ea_commands = [
            "EA:CLOCKINPUT=FULL;",
            "EA:PATTERN=?;",
            "EA:PATTERN=CHANNEL1:!PRBS7;",  # ! -- means inverted
            "EA:MEASUREMENTMODE=?;",
            "EA:THRESHOLDMODE=?;",
            "EA:THRESHOLDMODE=CHANNEL1:INVERTED;",
            "EA:MEASUREMENTMODE=CHANNEL1:SINGLE;",
            "EA:MEASUREMENTPERIOD=CHANNEL1:?;",
            "EA:MEASUREMENTPERIOD=CHANNEL1:20 s;",  # TODO
            # "EA:MEASUREMENTPERIOD=CHANNEL1:5 m;",
        ]
        for command in ea_commands:
            self.shf_command(command)
        self.ea_initiated = True
        time.sleep(5)

    def shf_patternsetup(self, pattern: str):
        """Configure PRBS7 and PRBS7Q pattern, open channels, set amplitude to 150 and open DAC output
        Implemented patterns: PAM4 and NRZ"""
        self.shf_command("DAC:OUTPUT=STATE:DISABLED;")
        self.shf_set_clksrc_frequency(20)
        PRBS7commands = [
            "BPG:USERSETTINGS=SCC.GRAYCODING:12;",
            "BPG:USERSETTINGS=SCC.PATTERNTYPE:PRBS7;",
            "BPG:PREEMPHASIS=ENABLE:OFF,PAMLEVELS:NONE;",
            "BPG:PATTERN=Channel1:PRBS7,Channel2:PRBS7,Channel3:PRBS7,Channel4:PRBS7,Channel5:PRBS7,Channel6:PRBS7;",
            "BPG:FIRFILTER=ENABLE:OFF;",
            "BPG:PAMLEVELS=SCC.PAMORDER:2;",
            "BPG:PAMLEVELS=SCC.L0:0%,SCC.L1:100%;",
            "BPG:PREEMPHASIS=DAC:SCC,PAMLEVELS:SCC;",
            # "BPG:FIRFILTER=G0:!PRBS7,G1:!PRBS7;",
            # "BPG:FIRFILTER=FUNCTION:1*y+0;",
        ]
        PRBS7Qcommands = [
            "BPG:USERSETTINGS=SCC.GRAYCODING:13;",
            "BPG:USERSETTINGS=SCC.PATTERNTYPE:PRBS7Q;",
            "BPG:PREEMPHASIS=ENABLE:OFF,PAMLEVELS:NONE;",
            "BPG:FIRFILTER=ENABLE:OFF;",
            "BPG:PAMLEVELS=SCC.PAMORDER:4;",
            "BPG:PAMLEVELS=SCC.L0:0%,SCC.L1:33.33%,SCC.L2:66.66%,SCC.L3:100%;",
            "BPG:PREEMPHASIS=DAC:SCC,PAMLEVELS:SCC;",
            "BPG:PATTERN=Channel1:PAMX,Channel2:PAMX,Channel3:PAMX,Channel4:PAMX,Channel5:PAMX,Channel6:PAMX;",
            # "BPG:FIRFILTER=G0:!PRBS7+60,G1:!PRBS7;",
            # "BPG:FIRFILTER=FUNCTION:1*y+0;",
        ]
        if pattern.lower() in ["pam2", "nrz", "prbs7"]:
            bpg_pattern_commands = PRBS7commands
            pattern = "PRBS7"
        elif pattern.lower() in ["pam4", "prbs7q"]:
            bpg_pattern_commands = PRBS7Qcommands
            pattern = "PRBS7Q"
        else:
            self.set_alarm(f"The pattern {pattern} is not implemented")
        bpg_amplitude_command = [
            "BPG:AMPLITUDE=Channel1:650 mV,Channel2:650 mV,Channel3:650 mV,Channel4:650 mV,Channel5:650 mV,Channel6:650 mV;",
        ]
        bpg_preemphasis_commands = [
            "BPG:PREEMPHASIS=ENABLE:ON;",
            "BPG:PREEMPHASIS=TAP0:0%;",
            "BPG:PREEMPHASIS=TAP2:0%;",
            "BPG:PREEMPHASIS=TAP3:0%;",
        ]
        bpg_channelouptput_commands = [
            "BPG:OUTPUT=Channel1:ON,Channel2:ON,Channel3:ON,Channel4:ON,Channel5:ON,Channel6:ON;",
        ]
        bpg_commands = (
            bpg_amplitude_command
            + bpg_pattern_commands
            + bpg_preemphasis_commands
            + bpg_channelouptput_commands
        )
        for command in bpg_commands:
            self.shf_command(command)
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
        self.bpg_preemphasis = {"TAP0": 0, "TAP2": 0, "TAP3": 0}  # TAP1 is Main
        self.bpg_pattern = pattern
        self.clksrc_frequency = 20  # GHz
        self.clksrc_output = True  # ON
        self.shf_set_amplitude(150)
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"{pattern} setup is finished",
            ]
        )

    def shf_set_preemphasis(self, tap_index: int = 0, value: int = 0):
        """tap_index should be 0, 2 or 3
        value should be from -100 to 100
        TAP1 is MAIN"""
        if abs(value) > 100:
            self.set_alarm(
                f"Preemphasis value: {value}. It should be from -100 to 100."
            )
        if tap_index in (0, 2, 3):
            self.shf_command(f"BPG:PREEMPHASIS=TAP{tap_index}:{value}%;")
            tap_str = f"TAP{tap_index}"
            self.bpg_preemphasis[tap_str] = value
            print(f"Preemphasis: Tap{tap_index}: {value}%")
        else:
            print("tap_index should be 0, 2 or 3; TAP1 is MAIN")

    def shf_set_amplitude(self, target_amplitude: int):
        real_target_amplitude = self.dBm_to_Vpp(
            self.shf_amplifier + self.Vpp_to_dBm(target_amplitude)
        )
        D0_onemV = 0.01587302
        D0 = float(target_amplitude * D0_onemV)
        dac_commands = [
            f"DAC:SIGNAL=ALIAS:D0,VALUE:{D0:.2f};",
            f"DAC:SIGNAL=ALIAS:D1,VALUE:{(D0*2):.2f};",
            f"DAC:SIGNAL=ALIAS:D2,VALUE:{(D0*4):.2f};",
            f"DAC:SIGNAL=ALIAS:D3,VALUE:{(D0*8):.2f};",
            f"DAC:SIGNAL=ALIAS:D4,VALUE:{(D0*16):.2f};",
            f"DAC:SIGNAL=ALIAS:D5,VALUE:{(D0*32):.2f};",
            "DAC:OUTPUT=STATE:ENABLED;",
        ]
        for command in dac_commands:
            self.shf_command(command)
        print(f"DAC amplitude set: {target_amplitude} mV")
        self.dac_amplitude = target_amplitude
        self.dac_output = 1
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"DAC amplitude: {target_amplitude} mV",
                f"after {self.shf_amplifier} dB amplification: {real_target_amplitude} mV",
            ]
        )

    def shf_set_clksrc_frequency(self, target_frequency: int):  # GHz
        if f"{target_frequency} Gbaud" in self.skew.columns:
            skews = self.skew[f"{target_frequency} Gbaud"].values.ravel()
        else:
            self.set_alarm("Skews are not calibreated for this frequency")
        bpg_skew_commands = [
            f"BPG:SKEW=Channel1:{skews[0]} ps,Channel2:{skews[1]} ps,Channel3:{skews[2]} ps,Channel4:{skews[3]} ps,Channel5:{skews[4]} ps,Channel6:{skews[5]} ps;"
        ]
        clksrc_commands = [
            "CLKSRC:OUTPUT=ON;",
            f"CLKSRC:FREQUENCY={target_frequency*1000000000} Hz;",
        ]
        if target_frequency >= 45:
            dac_frequency = 60
            dac_databias_value = -0.6
        else:
            dac_frequency = 32
            dac_databias_value = -0.4
        dac_band = [f"DAC:FREQUENCY=VALUE:{dac_frequency};"]
        dac_databias = [f"DAC:DATABIAS=VALUE:{dac_databias_value:.2f};"]
        for command in bpg_skew_commands + dac_band + dac_databias + clksrc_commands:
            self.shf_command(command)
        self.clksrc_output = True  # ON
        self.clksrc_frequency = target_frequency  # GHz
        self.logs.append(
            [time.strftime("%Y%m%d-%H%M%S"), f"Clock is set to {target_frequency} GHz"]
        )
        time.sleep(2)

    def shf_turn_off(self):
        if not self.shf_connected:
            self.logs.append(
                [
                    time.strftime("%Y%m%d-%H%M%S"),
                    "SHF is not connected",
                ]
            )
            return
        dac_commands = [
            "DAC:OUTPUT=STATE:DISABLED;",
        ]
        bpg_commands = [
            "BPG:PREEMPHASIS=TAP0:0%;",
            "BPG:PREEMPHASIS=TAP2:0%;",
            "BPG:PREEMPHASIS=TAP3:0%;",
            "BPG:OUTPUT=Channel1:OFF,Channel2:OFF,Channel3:OFF,Channel4:OFF,Channel5:OFF,Channel6:OFF,Channel7:OFF,Channel8:OFF;",
        ]
        clksrc_commands = [
            "CLKSRC:FREQUENCY=20000000000 Hz;",
            "CLKSRC:OUTPUT=OFF;",
        ]
        for command in dac_commands + bpg_commands + clksrc_commands:
            self.shf_command(command)
        self.bpg_output = {
            "channel1": 0,
            "channel2": 0,
            "channel3": 0,
            "channel4": 0,
            "channel5": 0,
            "channel6": 0,
        }  # OFF
        self.clksrc_frequency = 20  # GHz
        self.clksrc_output = 0  # OFF
        self.bpg_preemphasis = {"TAP0": 0, "TAP2": 0, "TAP3": 0}  # TAP1 is Main
        self.dac_output = 0
        self.shf_connected = False
        self.logs.append(
            [time.strftime("%Y%m%d-%H%M%S"), "DAC, BPG and Clock output is closed"]
        )

    # LIV
    def measure_liv_with_attenuator(self):
        if self.dac_output == 1:
            self.shf_turn_off()
        if self.current != 0:
            self.gently_apply_current(0)
        liv_config = self.config["LIV"]
        current_increment_LIV = float(liv_config["current_increment_LIV"])
        max_current = float(liv_config["max_current"])
        beyond_rollover_stop_cond = float(liv_config["beyond_rollover_stop_cond"])
        current_limit1 = float(liv_config["current_limit1"])
        current_limit2 = float(liv_config["current_limit2"])
        threshold_decision_level = float(liv_config["threshold_decision_level"])
        liv_dpi = int(liv_config["liv_dpi"])
        current_list = [0.0]
        round_to = max(0, int(np.ceil(np.log10(1 / current_increment_LIV))))
        while current_list[-1] <= max_current - current_increment_LIV:
            current_list.append(
                round(current_list[-1] + current_increment_LIV, round_to)
            )
        powermeter = "EXPOLTB1"
        dirpath = (
            Path.cwd()
            / "data"
            / f"{self.waferid}-{self.wavelength}nm"
            / f"{self.coordinates}"
        )
        print(f"Measuring LIV using {powermeter}")
        # initiate pandas Data Frame
        iv = pd.DataFrame(
            columns=[
                "Current set, mA",
                "Current, mA",
                "Voltage, V",
                "Output power, mW",
                "Output power, dBm",
                "Power consumption, mW",
                "Power conversion efficiency, %",
            ]
        )
        # The initial settings are applied by the *RST command
        self.attenuator_shutter("close")
        self.attenuator_command(":RST")
        self.attenuator_command(f":INP:WAV {self.wavelength} NM")
        # self.attenuator_command(":CONT:MODE POW")
        self.attenuator_command(":CONT:MODE ATT")
        self.attenuator_command(":OUTP:ALC ON")  # Power tracking
        self.attenuator_command(":OUTP:APM REF")  # Reference mode.
        self.rst_current_source()
        self.current_source.write(
            ":OUTP ON"
        )  # Measurement channel is enabled by the :OUTP ON command.
        # initate power and max power variables with 0
        max_output_power = 0
        output_power = 0
        warnings = []
        for current_set in current_list:
            # Outputs {current_set} mA immediately
            self.current_source.write(":SOUR:CURR " + str(current_set / 1000))
            # time.sleep(0) # TODO del
            # measure Voltage, V
            voltage = float(self.current_source.query("MEAS:VOLT?"))
            # measure Current, A
            current_measured = (
                float(self.current_source.query("MEAS:CURR?")) * 1000
            )  # mA
            self.current = current_measured
            # measure output power
            output_power = self.update_attenuator_powerin(sleep=0.5)
            output_power_dBm = self.mW_to_dBm(output_power)

            if output_power > max_output_power:  # track max power
                max_output_power = output_power
                color = "light_green"
            else:
                color = "light_yellow"
            # add current, measured current, voltage, output power (mW), power consumption, power conversion efficiency, output power (dBm) to the DataFrame
            iv.loc[len(iv)] = [
                current_set,
                current_measured,
                voltage,
                None,
                None,
                voltage * current_measured,
                0,
            ]

            iv.iloc[-1, iv.columns.get_loc("Output power, mW")] = output_power
            iv.iloc[-1, iv.columns.get_loc("Output power, dBm")] = output_power_dBm
            # print data to the terminal
            if voltage * current_measured == 0 or output_power >= (
                voltage * current_measured
            ):
                print(
                    colored(
                        f"{current_set:3.2f} mA: {current_measured:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW, 0 %",
                        color,
                    )
                )
            else:
                print(
                    colored(
                        f"{current_set:3.2f} mA: {current_measured:10.5f} mA, {voltage:8.5f} V, {output_power:8.5f} mW ({output_power_dBm:3.3f} dBm), {(100*output_power)/(voltage*current_measured):8.2f} %",
                        color,
                    )
                )
                iv.iloc[-1, iv.columns.get_loc("Power conversion efficiency, %")] = (
                    100 * output_power
                ) / (voltage * current_measured)
            # deal with set/measured current mismatch
            current_error = abs(current_set - current_measured)
            if np.float64(round(current_measured, round_to)) != np.float64(
                round(current_set, round_to)
            ):
                warnings.append(
                    [
                        time.strftime("%Y%m%d-%H%M%S"),
                        f"Current set={current_set} mA, current measured={current_measured} mA, current error: {current_error} mA",
                    ]
                )
                print(
                    colored(
                        f"WARNING! Current set is {current_set} mA, while current measured is {current_measured} mA",
                        "yellow",
                    )
                )
            if current_error >= 0.03:
                print(
                    colored(
                        f"ALARM! Current set is {current_set}, while current measured is {current_measured}\tBreaking the measurements!",
                        "red",
                    )
                )
                warnings.append(
                    [
                        time.strftime("%Y%m%d-%H%M%S"),
                        f"Alarm! Breaking the LIV measurements!",
                    ]
                )
                break  # break the loop
            # breaking conditions
            if current_set > current_limit1:  # if current is more then limit1 mA
                if (
                    output_power <= max_output_power * beyond_rollover_stop_cond
                    or output_power <= 0.01
                ):  # check conditions to stop the measurements
                    if output_power <= 0.01:
                        print(
                            colored(
                                f"Current reached {current_limit1} mA, but the output_power is less then 0.01 mW\tbreaking the loop",
                                "red",
                            )
                        )
                        warnings.append(
                            [
                                time.strftime("%Y%m%d-%H%M%S"),
                                f"Current reached {current_limit1} mA, but the output_power is less then 0.01 mW. Breaking the loop",
                            ]
                        )
                    break  # break the loop
            if max_output_power <= 0.5:
                if current_set > current_limit2:  # if current is more then limit2 mA
                    print(
                        colored(
                            f"Current reached {current_limit2} mA, but the output_power is less then 0.5 mW\tbreaking the loop",
                            "red",
                        )
                    )
                    warnings.append(
                        [
                            time.strftime("%Y%m%d-%H%M%S"),
                            f"Current reached {current_limit2} mA, but the output_power is less then 0.5 mW. Breaking the loop",
                        ]
                    )
                    break  # break the loop
        # slowly decrease current
        current_measured = (
            float(self.current_source.query("MEAS:CURR?")) * 1000
        )  # measure current
        self.current = current_measured
        for current_set in np.arange(current_measured, 0, -0.1):
            self.current_source.write(
                f":SOUR:CURR {str(current_set/1000)}"
            )  # Outputs i A immediately
            print(f"Current set: {current_set:3.1f} mA", end="\r")
            time.sleep(0.01)  # 0.01 sec for a step, 1 sec for 10 mA
        # Measurement is stopped by the :OUTP OFF command.
        self.current = 0
        self.current_source.write(":OUTP OFF")
        self.current_source.write(f":SOUR:CURR 0.001")
        self.attenuator_command(":CONT:MODE POW")
        timestr = time.strftime("%Y%m%d-%H%M%S")  # current time
        livdirpath = dirpath / "LIV"
        livdirpath.mkdir(exist_ok=True, parents=True)
        filepath = dirpath / "LIV"
        filename = f"{self.waferid}-{self.wavelength}nm-{self.coordinates}-{self.temperature}°C-{timestr}-{powermeter}"
        iv.to_csv(
            filepath / (filename + ".csv"), index=False
        )  # save DataFrame to csv file
        # save figures
        buildplt_everything(
            dataframe=iv,
            waferid=self.waferid,
            wavelength=self.wavelength,
            coordinates=self.coordinates,
            temperature=self.temperature,
            powermeter=powermeter,
            threshold_decision_level=threshold_decision_level,
        )
        plt.savefig(
            filepath / (filename + "-everything.png"), dpi=liv_dpi
        )  # save figure
        i_threshold, i_rollover = buildplt_liv(
            dataframe=iv,
            waferid=self.waferid,
            wavelength=self.wavelength,
            coordinates=self.coordinates,
            temperature=self.temperature,
            powermeter=powermeter,
            threshold_decision_level=threshold_decision_level,
        )
        plt.savefig(
            (
                filepath
                / (filename + f"_Ith={i_threshold:.2f}_Iro={i_rollover:.2f}.png")
            ),
            dpi=liv_dpi,
        )  # save figure
        plt.close("all")
        if warnings:
            print(colored(f"Warnings: {len(warnings)}", "yellow"))
            print(*[colored(warning[1], "yellow") for warning in warnings], sep="\n")
            self.logs.extend(warnings)
            self.errorlogs.extend(warnings)
            self.save_logs(note="LIV")
        print(f"Directory: {dirpath}")
        return

    # def cubes commands
    def connect_kcubes(self):
        "Populates self.kcube_devices and moves fiber to x,y,z=37.5"
        # https://github.com/Thorlabs/Motion_Control_Examples/blob/main/Python/Modular%20Rack/mmr_pythonnet.py
        # https://github.com/Thorlabs/Motion_Control_Examples/blob/main/Python/KCube/KPZ101/kpz101_pythonnet.py
        DeviceManagerCLI.BuildDeviceList()
        # create new device
        instruments_config = self.config["INSTRUMENTS"]
        X_Kcube_sn = instruments_config["X_Kcube"]
        Y_Kcube_sn = instruments_config["Y_Kcube"]
        Z_Kcube_sn = instruments_config["Z_Kcube"]
        deviceX = KCubePiezo.CreateKCubePiezo(X_Kcube_sn)
        deviceY = KCubePiezo.CreateKCubePiezo(Y_Kcube_sn)
        deviceZ = KCubePiezo.CreateKCubePiezo(Z_Kcube_sn)
        self.kcube_devices = [deviceX, deviceY, deviceZ]
        deviceX.Connect(X_Kcube_sn)
        deviceY.Connect(Y_Kcube_sn)
        deviceZ.Connect(Z_Kcube_sn)
        device_infoX = deviceX.GetDeviceInfo()
        device_infoY = deviceY.GetDeviceInfo()
        device_infoZ = deviceZ.GetDeviceInfo()
        print(device_infoX.Description)
        print(device_infoY.Description)
        print(device_infoZ.Description)
        # Start polling and enable
        for device in self.kcube_devices:
            device.StartPolling(250)  # 250ms polling rate
        for device in self.kcube_devices:
            device.EnableDevice()
        time.sleep(0.25)  # Wait for device to enable
        # Ensure that the device settings have been initialized
        for device in self.kcube_devices:
            if not device.IsSettingsInitialized():
                device.WaitForSettingsInitialized(10000)  # 10 second timeout
                assert device.IsSettingsInitialized() is True
        device_config = deviceX.GetPiezoConfiguration(X_Kcube_sn)
        device_config = deviceY.GetPiezoConfiguration(Y_Kcube_sn)
        device_config = deviceZ.GetPiezoConfiguration(Z_Kcube_sn)
        print("Setting Zero Point")
        #
        # Set the Zero point of the device
        # for device in self.kcube_devices:
        #     device.SetZero()  # TODO do we need this?
        # self.piezo_voltages = [0.0, 0.0, 0.0]
        #
        # Get the maximum voltage output of the KPZ
        for device in self.kcube_devices:
            self.piezo_max_voltage = (
                device.GetMaxOutputVoltage()
            )  # This is stored as a .NET decimal
            print(f"Piezo max voltage: {self.piezo_max_voltage}")
        self.move_fiber_sim([37.5, 37.5, 37.5])
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "Kcubes are connected",
            ]
        )

    def start_optimizing_fiber(self, ask=True):
        self.move_fiber_sim([37.5, 37.5, 37.5])
        print(
            "Piezo voltages set to 37.5. Please manually adjust the fiber position using nobs."
        )
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "Kcubes are moved to voltage 37.5",
            ]
        )
        timeout = 300
        self.gently_apply_current(self.test_current)
        start_time = time.time()  # start time for timeout
        while (time.time() - start_time) < timeout:
            if ask:
                prompt = f"The fiber optimization will be canceled after {timeout} seconds. Start optimizing the fiber position? Y/n"
                answer = input(prompt)
            else:
                answer = None
            if answer:
                if answer in ("Y", "y"):
                    self.simple_optimize_fiber()
                    # self.optimize_fiber()
                else:
                    self.set_alarm("Fiber reposition declined.")
                return
            else:
                self.simple_optimize_fiber()
                return
        self.set_alarm("Timeout. Fiber reposition declined.")

    def simple_optimize_fiber(self):
        "optimize the fiber position using Thorlabs KCubes with simple algorithm"
        self.update_fiber_position()
        self.attenuator_shutter("close")
        self.attenuator_command(f":INP:WAV {self.wavelength} NM")
        self.attenuator_command(":CONT:MODE ATT")
        self.attenuator_command(":OUTP:ALC ON")  # Power tracking
        self.attenuator_command(":OUTP:APM REF")  # Reference mode.
        self.gently_apply_current(self.test_current)
        self.move_fiber_sim([37.5, 37.5, 37.5])
        self.update_attenuator_powerin()
        start_powerin = self.attenuator_powerin
        print(f"start_powerin: {start_powerin:.6f} mW")
        list_of_steps = [10] * 5 + [3] * 5 + [1] * 5 + [0.1] * 7
        print("Optimizing fiber position")
        for i, step in enumerate(list_of_steps, start=1):
            print(" " * 80, end="\r")
            print(f"[{i}/{len(list_of_steps)}] step: {step}" + " " * 50)
            max_powerin, max_powerin_position, border = self.optim_fiber_itteration(
                step=step
            )
            if border:
                return
        powerin = self.update_attenuator_powerin()
        print(" " * 80, end="\r")
        print(
            f"{max_powerin_position}\tPowerin: {start_powerin:3.3f} -> {self.attenuator_powerin:3.3f} mW ({self.mW_to_dBm(start_powerin):3.3f} -> {self.mW_to_dBm(self.attenuator_powerin):3.3f} dBm)"
        )
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"Fiber position is optimized. {max_powerin_position}\tPowerin: {start_powerin} -> {self.attenuator_powerin} mW",
            ]
        )
        self.attenuator_command(":CONT:MODE POW")

    def optim_fiber_itteration(self, step: float = 1, piezo_sleep: float = 0.5):
        border = False
        self.update_fiber_position()
        max_powerin = self.update_attenuator_powerin()
        max_powerin_position = self.piezo_voltages[:]
        for axis in (0, 1, 2):
            previous_power = self.update_attenuator_powerin()
            go_positive = True
            while go_positive:
                voltages = self.piezo_voltages[:]
                voltages[axis] += step
                if voltages[axis] >= 74.7:
                    print(" " * 100, end="\r")
                    print(voltages)
                    self.fiber_at_border(axis)
                    border = True
                    return max_powerin, max_powerin_position, border
                power = self.fiber_position_to_power(voltages, sleep=piezo_sleep)
                print(" " * 80, end="\r")
                print(
                    f"{self.piezo_voltages}\tpower={power:3.3f} mW ({(self.mW_to_dBm(power)):3.3f} dBm)",
                    end="\r",
                )
                if power >= previous_power:
                    max_powerin = self.attenuator_powerin
                    previous_power = max_powerin
                    max_powerin_position = self.piezo_voltages[:]
                    # print(
                    #    f"{max_powerin_position},\tmoving positive\t step {step} \t max_powerin={max_powerin:3.3f} mW ({(self.mW_to_dBm(max_powerin)):3.3f} dBm)",
                    #    # end="\r",
                    # )
                elif power < previous_power:
                    go_positive = False
                    self.move_fiber(
                        axis=axis,
                        target_voltage=max_powerin_position[axis],
                        sleep=piezo_sleep,
                    )
            while not go_positive:
                voltages = self.piezo_voltages[:]
                voltages[axis] -= step
                if voltages[axis] <= 0.3:
                    print(" " * 100, end="\r")
                    print(voltages)
                    self.fiber_at_border(axis)
                    border = True
                    return max_powerin, max_powerin_position, border
                power = self.fiber_position_to_power(voltages, sleep=piezo_sleep)
                print(" " * 80, end="\r")
                print(
                    f"{self.piezo_voltages}\tpower={power:3.3f} mW ({(self.mW_to_dBm(power)):3.3f} dBm)",
                    end="\r",
                )
                if power >= previous_power:
                    max_powerin = self.attenuator_powerin
                    previous_power = max_powerin
                    max_powerin_position = self.piezo_voltages[:]
                    # print(
                    #    f"{max_powerin_position},\tmoving negative\t step {step} \t max_powerin={max_powerin:3.3f} mW ({(self.mW_to_dBm(max_powerin)):3.3f} dBm)",
                    #    # end="\r",
                    # )
                elif power < previous_power:
                    go_positive = True
                    self.move_fiber(
                        axis=axis,
                        target_voltage=max_powerin_position[axis],
                        sleep=piezo_sleep,
                    )
            self.move_fiber_sim(max_powerin_position)
        return max_powerin, max_powerin_position, border

    # def optimize_fiber(self):
    #     "optimize the fiber position using Thorlabs KCubes"
    #     self.update_fiber_position()
    #     self.attenuator_shutter("close")
    #     self.attenuator_command(f":INP:WAV {self.wavelength} NM")
    #     self.attenuator_command(":CONT:MODE POW")
    #     self.attenuator_command(":OUTP:ALC ON")  # Power tracking
    #     self.attenuator_command(":OUTP:APM REF")  # Reference mode.
    #     self.update_attenuator_powerin()
    #     start_powerin = self.attenuator_powerin
    #     self.gently_apply_current(self.test_current)
    #     theta = [37.5, 37.5, 37.5]  # initial voltages
    #     self.move_fiber_sim(theta)
    #     eta = 2  # optimization rate
    #     n_epochs = 150
    #     sleep = 0.1
    #     delta = 0.2
    #     for epoch in range(n_epochs):
    #         eta *= 0.988
    #         # if sleep < 0.5:
    #         #     sleep *= 1.01
    #         # else:
    #         #     sleep = 0.5
    #         gradient = self.gradient_power(delta=delta, sleep=sleep)
    #         mode = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2)
    #         if eta * mode > 5:
    #             coeff = mode / 5
    #             gradient = [
    #                 gradient[0] / coeff,
    #                 gradient[1] / coeff,
    #                 gradient[2] / coeff,
    #             ]
    #         mode = np.sqrt(gradient[0] ** 2 + gradient[1] ** 2 + gradient[2] ** 2)
    #         X = round(theta[0] + eta * gradient[0], 2)
    #         Y = round(theta[1] + eta * gradient[1], 2)
    #         Z = round(theta[2] + eta * gradient[2], 2)
    #         theta = [X, Y, Z]
    #         self.move_fiber_sim(theta)
    #         print(f"X: {X}\tY: {Y}\tZ: {Z},\tlast step: {eta * mode}", end="\r")
    #         if eta * mode < 0.1:
    #             print(f"Fiber position is optimized in {epoch+1} epochs")
    #             break
    #     print(
    #         f"The last epoch. Derivatives: X:{gradient[0]}, Y:{gradient[1]}, Z:{gradient[2]}"
    #     )
    #     self.attenuator_powerin = self.fiber_position_to_power(theta)
    #     print(
    #         f"X: {X}\tY: {Y}\tZ: {Z}\tPowerin: {start_powerin} -> {self.attenuator_powerin} mW"
    #     )
    #     self.logs.append(
    #         [
    #             time.strftime("%Y%m%d-%H%M%S"),
    #             f"Fiber position is optimized. X: {X}\tY: {Y}\tZ: {Z}\tPowerin: {start_powerin} -> {self.attenuator_powerin} mW",
    #         ]
    #     )

    # def gradient_power(self, delta: float = 0.1, sleep: float = 0.5):
    #     "measure power and calculate gradient. You need to keep piezo voltages 2*delta away from the borders"
    #     self.update_fiber_position()
    #     mid_power = self.update_attenuator_powerin()
    #     old_piezo_voltages = self.piezo_voltages[:]
    #     gradient = [0.0, 0.0, 0.0]
    #     for axis, start_voltage in enumerate(old_piezo_voltages):
    #         if start_voltage < delta and start_voltage > 75 - delta:
    #             self.fiber_at_border(axis)
    #         piezo_voltages_tmp = old_piezo_voltages[:]
    #         piezo_voltages_tmp[axis] = start_voltage - delta
    #         plus_power = (
    #             self.fiber_position_to_power(piezo_voltages_tmp, sleep=sleep)
    #             / mid_power
    #         )
    #         piezo_voltages_tmp = old_piezo_voltages[:]
    #         piezo_voltages_tmp[axis] = start_voltage + delta
    #         minus_power = (
    #             self.fiber_position_to_power(piezo_voltages_tmp, sleep=sleep)
    #             / mid_power
    #         )
    #         gradient[axis] = (plus_power - minus_power) / (2 * delta)
    #     self.move_fiber_sim(old_piezo_voltages)
    #     return gradient

    def fiber_at_border(self, axis: int):
        "Handle the piezo borders using timeout input"
        if axis == 0:
            axis_name = "X"
        elif axis == 1:
            axis_name = "Y"
        elif axis == 2:
            axis_name = "Z"
        print(" " * 100, end="\r")
        print(
            colored(
                f"Piezo axis {axis_name} reached {self.piezo_voltages[axis]}",
                "yellow",
                "on_black",
            )
        )
        timestr = time.strftime("%Y%m%d-%H%M%S")
        self.logs.append(
            [
                timestr,
                "Piezo axis {axis_name} reached {self.piezo_voltages[axis]}",
            ]
        )
        self.errorlogs.append(
            [
                timestr,
                "Piezo axis {axis_name} reached {self.piezo_voltages[axis]}",
            ]
        )
        self.start_optimizing_fiber()

    def move_fiber_sim(self, target_voltages: list, sleep: float = 0.5):
        "move all 3 axes to voltage positiones"
        self.update_fiber_position()
        moving_from = f"moving from {self.piezo_voltages}"
        moving_to = f"moving to {target_voltages}"
        if self.piezo_voltages != target_voltages:
            for axis, voltage in enumerate(target_voltages):
                if voltage != self.piezo_voltages[axis]:
                    if voltage >= 75:
                        voltage = 75
                    device = self.kcube_devices[axis]
                    dev_voltage = Decimal(voltage)
                    device.SetOutputVoltage(dev_voltage)
            time.sleep(sleep)
            self.update_fiber_position
            x = 1
            z = 0
            for i in (0, 1, 2):
                x += abs(self.piezo_voltages[i] - target_voltages[i]) > 0.1
            if x:
                while x:
                    x = 0
                    z += 1
                    time.sleep(0.1)
                    self.update_fiber_position()
                    for i in (0, 1, 2):
                        x += abs(self.piezo_voltages[i] - target_voltages[i]) > 0.1
                    if z >= 5:
                        break
        moved_to = f"moved to {self.piezo_voltages}"
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "Kcubes are" + moving_to,
                moving_from,
                moved_to,
            ]
        )

    def move_fiber(self, axis: int, target_voltage: float, sleep: float = 0.5):
        "x:0, y:1, z:2. Voltage from 0 to 75"
        self.update_fiber_position()
        moving_from = f"moving from {self.piezo_voltages}"
        moving_to = f"moving to {target_voltage}"
        if self.piezo_voltages[axis] != target_voltage:
            if target_voltage >= 75:
                target_voltage = 75
            device = self.kcube_devices[axis]
            dev_voltage = Decimal(target_voltage)
            device.SetOutputVoltage(dev_voltage)
            time.sleep(sleep)
            self.piezo_voltages[axis] = round(float(str(device.GetOutputVoltage())), 2)
            x = 1
            z = 0
            if abs(self.piezo_voltages[axis] - target_voltage) > 0.1:
                while x:
                    z += 1
                    time.sleep(0.1)
                    self.piezo_voltages[axis] = round(
                        float(str(device.GetOutputVoltage())), 2
                    )
                    x = abs(self.piezo_voltages[axis] - target_voltage) > 0.1
                    if z >= 5:
                        break
        moved_to = f"moved to {self.piezo_voltages}"
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                "Kcubes are" + moving_to,
                moving_from,
                moved_to,
            ]
        )

    def update_fiber_position(self):
        "updates self.piezo_voltages"
        device = self.kcube_devices[0]
        if not device:
            "Kcubes are disconnected, connecting..."
            self.connect_kcubes()
        for axis, device in enumerate(self.kcube_devices):
            self.piezo_voltages[axis] = round(float(str(device.GetOutputVoltage())), 2)

    def kcubes_disconnect(self):
        for device in self.kcube_devices:
            device.SetZero()
            device.StopPolling()
            device.Disconnect()
        self.piezo_voltages = [0.0, 0.0, 0.0]
        self.kcube_devices = [None, None, None]

    def fiber_position_to_power(self, voltages: list, sleep: float = 0.5):
        self.move_fiber_sim(voltages, sleep=sleep)
        self.update_attenuator_powerin()
        return self.attenuator_powerin

    # TODO EA
    # def autosearch_eye(self):  # TODO DET doesn't connect
    #     if self.bpg_pattern == "PRBS7Q":
    #         self.shf_command("DET:OUTPUT=STATE:ON;")
    #     value = self.shf_command("DET:AUTOSEARCH=VALUE:RUN;")
    #     epoch = 0
    #     while value not in ("FINISHED", "ABORTED"):
    #         time.sleep(1)
    #         epoch += 1
    #         value = (
    #             self.shf_command("DET:AUTOSEARCH=VALUE:?;").strip(";").split(":")[-1]
    #         )
    #         if epoch > 30:
    #             break
    #     if value == "FINISHED":
    #         result = self.shf_command(
    #             "DET:AUTOSEARCH=DELAY:?,THRESHOLDMIN:?,THRESHOLDMAX:?;"
    #         )
    #         return True
    #     elif value == "ABORTED":
    #         result = "ABORTED"
    #         self.errorlogs.append(
    #             [time.strftime("%Y%m%d-%H%M%S"), "Autosearch aborted"]
    #         )
    #         return False

    def ea_sync(self):
        "Checks EA sync"
        if not self.ea_initiated:
            self.ea_init()
        responce = self.shf_command("EA:SYNC=?;")
        if responce == "EA:SYNC=CHANNEL1:100.00 %;":
            return True
        else:
            for i in range(10):
                time.sleep(3)
                responce = self.shf_command("EA:SYNC=?;")
                if responce == "EA:SYNC=CHANNEL1:100.00 %;":
                    return True
            self.set_alarm(f"Error Analyser is unsynced: {responce}")
            return False

    # for ea_autosearch
    def remove_duplicates(self, list):
        new_list = []
        for elem in list:
            if elem not in new_list:
                new_list.append(elem)
        return new_list

    def calculate_eye_area(self, array):
        "Use Green's theorem to compute the area enclosed by the given contour."
        x = array[:, 0]
        y = array[:, 1]
        area = 0.5 * np.sum(y[:-1] * np.diff(x) - x[:-1] * np.diff(y))
        area = np.abs(area)
        return area

    def handle_units(self, unit: str):
        if len(unit) == 1:
            prefix = 1
        elif len(unit) == 2:
            prefix = unit[0]
            if prefix == "p":
                prefix = 10**-12
            elif prefix == "n":
                prefix = 10**-9
            elif prefix == "u":
                prefix = 10**-6
            elif prefix == "m":
                prefix = 10**-3
            else:
                print(f"Can't parce unit: {unit}")
                self.errorlogs.append(
                    [time.strftime("%Y%m%d-%H%M%S"), f"Can't parce unit: {unit}"]
                )
                prefix = 0
        return prefix

    def process_points_in_eyecontour(self, line):
        line = line[1:-1].split(",")
        xvalue, xunit = line[0].split()
        x = round(float(xvalue), 4) * self.handle_units(xunit)
        yvalue, yunit = line[1].split()
        y = round(float(yvalue), 4) * self.handle_units(yunit)
        return [x, y]

    # def hullpoints(self, eyecontour):
    #     "convex hull for elliptic autosearch"
    #     from scipy.spatial import ConvexHull
    #
    #     hull = ConvexHull(eyecontour).simplices
    #     result = []
    #     for x in hull:
    #         for y, z in hull:
    #             if y not in result:
    #                 result.append(y)
    #             if z not in result:
    #                 result.append(z)
    #     contour = eyecontour[sorted(result)]
    #     contour_points = np.vstack((contour, contour[0]))
    #     area = self.calculate_eye_area(contour_points)
    #     return contour_points, area

    def parce_eyecontour(self, responce: str):
        "parce successfull autosearch responce"
        if "success:true" in responce.lower():
            eyecontour_raw = responce.split("CHANNEL1.EYECONTOUR:")[-1][1:-2].split(
                "--"
            )
            eyecontour_open = np.array(
                self.remove_duplicates(
                    [self.process_points_in_eyecontour(x) for x in eyecontour_raw]
                )
            )
            eyecontour = np.vstack((eyecontour_open, eyecontour_open[0]))
            area = self.calculate_eye_area(eyecontour)
            return round(area, 18), eyecontour
        else:
            return 0, False

    def get_ea_bitrate(self):
        "Gbit/s"
        responce = self.shf_command("EA:BITRATEAPPROX=CHANNEL1:?;")
        bitrate, unit = responce.lstrip("EA:BITRATEAPPROX=CHANNEL1:").split()
        if unit == "Gbit/s;":
            assert float(self.clksrc_frequency) == float(bitrate)
            return float(bitrate)
        if unit == "Mbit/s;":
            assert float(self.clksrc_frequency) == float(bitrate) / 1000
            return float(bitrate) / 1000

    def ea_autosearch(self, type: str = "SIMPLEBER", logber: int = -6):
        # TODO elliptic has a problems with correct area calculation due to outliers
        self.ea_sync()
        if type.upper() in ("SIMPLE", "SIMPLEBER", "ELLIPTIC"):
            responce = self.shf_command(
                f"EA:AUTOSEARCH=CHANNEL1.TYPE:{type.upper()},CHANNEL1.BER:1e{int(logber)};"
            )
            if "SUCCESS:FALSE" in responce:
                bitrate = self.get_ea_bitrate()
                self.logs.append(
                    [
                        time.strftime("%Y%m%d-%H%M%S"),
                        responce,
                        f"No eye contour, bitrate: {bitrate:3.3f} Gbit/s",
                    ]
                )
                print(
                    f"No eye contour,\tbitrate: {bitrate:3.3f} Gbit/s"
                )
                return 0, 0, [], bitrate
            while True:
                responce = str(self.ea.read_raw().strip(), "utf-8")
                if "SUCCESS:TRUE" in responce:
                    bitrate = self.get_ea_bitrate()
                    area, eyecontour = self.parce_eyecontour(responce)
                    specific_area_mV = area * bitrate * 10**12
                    self.logs.append(
                        [
                            time.strftime("%Y%m%d-%H%M%S"),
                            responce,
                            f"eye contour area: {area*10**14:3.3f}E-14 s*V, specific eye contour area: {specific_area_mV:3.3f} mV, bitrate: {bitrate:3.3f} Gbit/s",
                        ]
                    )
                    print(
                        f"eye contour area: {area*10**14:3.3f}E-14 s*V,\tspecific eye contour area: {specific_area_mV:3.3f} mV,\tbitrate: {bitrate:3.3f} Gbit/s"
                    )
                    # in case you need it:
                    # self.shf_command("EA:DELAY=CHANNEL1:?;") # "EA:DELAY=CHANNEL1:51.7 ps;"
                    # self.shf_command("EA:DELAYRANGE=CHANNEL1:?;") # "EA:DELAYRANGE=CHANNEL1:0 s..79.1 ps;"
                    # self.shf_command("EA:THRESHOLD=CHANNEL1:?;") # "EA:THRESHOLD=CHANNEL1:1.5 mV;"
                    # self.shf_command("EA:THRESHOLDRANGE=CHANNEL1:?;") # "EA:THRESHOLDRANGE=CHANNEL1:-412 mV..389 mV;"
                    return area, specific_area_mV, eyecontour, bitrate
                elif "SUCCESS:FALSE" in responce:
                    bitrate = self.get_ea_bitrate()
                    self.logs.append(
                        [
                            time.strftime("%Y%m%d-%H%M%S"),
                            responce,
                            f"No eye contour, bitrate: {bitrate:3.3f} Gbit/s",
                        ]
                    )
                    print(
                        f"No eye contour,\tbitrate: {bitrate:3.3f} Gbit/s"
                    )
                    return 0, 0, [], bitrate
        else:
            self.set_alarm("Autosearch fail")  # TODO what about closed eye?
            return 0, 0, [], 0

    def test_ea_job(self):
        if not self.ea_initiated:
            self.ea_init()
        timeout = 600
        start_time = time.time()  # start time for timeout
        while (time.time() - start_time) < timeout:
            job = self.shf_command("EA:CURRENTJOB=?;")
            if job.lower() == "EA:CURRENTJOB=NONE;".lower():
                return
            else:
                time.sleep(2)
        self.shf_command("EA:ABORT;")
        self.set_alarm("Error Analyzer timeout")

    def parce_ber_responce(self, responce: str):
        "returns: BER, total Gbits countered so far, total number of errors, final result or not"
        pattern = r"CHANNEL1\.BER:.*,,CHANNEL1\.BITS:"
        ber = float(
            re.findall(pattern, responce)[0]
            .lstrip("CHANNEL1.BER:")
            .rstrip("CHANNEL1.BITS:")
            .rstrip(",,")
        )
        pattern = r"CHANNEL1\.BITS:.*,CHANNEL1\.INS:"
        gbits = (
            int(
                re.findall(pattern, responce)[0]
                .lstrip("CHANNEL1.BITS")
                .lstrip(":")
                .rstrip(",CHANNEL1.INS:")
            )
            * 10**-9
        )
        pattern = r"EA:RESULT=CHANNEL1.FINAL:.*,CHANNEL1.BER:"
        final_result = (
            re.findall(pattern, responce)[0]
            .lstrip("EA:RESULT=CHANNEL1")
            .lstrip(".FINAL")
            .lstrip(":")
            .rstrip(",CHANNEL1.BER:")
        )
        if final_result == "YES":
            final = True
        elif final_result == "NO":
            final = False
        else:
            final = None
        pattern = r"CHANNEL1.SUM:.*;"
        number_of_errors = int(
            re.findall(pattern, responce)[0]
            .lstrip("CHANNEL1.SUM")
            .lstrip(":")
            .rstrip(";")
        )
        return ber, gbits, final, number_of_errors

    def measure_ber(self):
        "measure BER using EA"
        self.test_ea_job()
        self.ea_autosearch()
        self.shf_command("EA:MEASUREMENT=CHANNEL1:ON;")
        while True:
            time.sleep(10)
            responce = self.shf_command("EA:RESULT=CHANNEL1:?;")
            ber, gbits, final, number_of_errors = self.parce_ber_responce(responce)
            if "final:yes" in responce.lower():
                print(f"BER: {ber}")
                break
        return ber, gbits, final, number_of_errors

    def parce_qfactor_responce(self, responce: str):
        if ".qfactor:" in responce.lower():
            pattern = r"CHANNEL1.QFACTOR:.*;"
            qfactor = float(
                re.findall(pattern, responce)[0].lstrip("CHANNEL1.QFACTOR").lstrip(":").rstrip(";")
            )
            return qfactor
        else:
            return -1

    def measure_qfactor(self, logbermin=-9, logbermax=-4, exact=True):
        self.test_ea_job()
        self.ea_autosearch()
        self.shf_command(
            f"EA:QFACTOR=CHANNEL1.BERMIN:1e{logbermin},CHANNEL1.BERMAX:1e{logbermax},CHANNEL1.EXACT:{exact};"
        )
        qfactor = -1
        while True:
            responce = str(self.ea.read_raw(), "utf-8").strip()
            #print(responce)
            self.logs.append([time.strftime("%Y%m%d-%H%M%S"), responce])
            if ".qfactor:" in responce.lower():
                qfactor = self.parce_qfactor_responce(responce)
                print(f"Qfactor: {qfactor}")
                break
        return qfactor

    def optimize_preemphasis(self, logber: int = -9):
        start_taps = self.bpg_preemphasis.copy()
        area, specific_area_start, eyecontour, bitrate = self.ea_autosearch(
            type="SIMPLEBER", logber=logber
        )
        list_of_steps = [20, 10, 10, 10, 5, 5, 5, 1, 1, 1]
        print("Optimizing preemphasis")
        for i, step in enumerate(list_of_steps):
            print(" " * 80, end="\r")
            print(f"[{i}/{len(list_of_steps)}] step: {step}" + " " * 50)
            max_area_taps, max_specific_area = self.optim_preemphasis_itteration(
                step=step, logber=logber
            )
        print(" " * 80, end="\r")
        print(
            max_area_taps, f"{specific_area_start:3.3f} -> {max_specific_area:3.3f} mV"
        )
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"Preemphasis is optimized. {max_area_taps} {specific_area_start:3.3f} -> {max_specific_area:3.3f} mV",
            ]
        )

    def optim_preemphasis_itteration(self, step: int = 10, logber: int = -2):
        max_specific_area = 0
        max_area_taps = self.bpg_preemphasis.copy()
        for tap_index in (0, 2, 3):
            plus_specific_area = -1
            minus_specific_area = -1
            tap_str = f"TAP{tap_index}"
            start_area, start_specific_area, _, _ = self.ea_autosearch(
                type="SIMPLEBER", logber=logber
            )
            if start_specific_area >= max_specific_area:
                max_specific_area = start_specific_area
                max_area_taps = self.bpg_preemphasis.copy()
                print(" " * 80, end="\r")
                print(max_area_taps, f"{max_specific_area:3.3f} mV", end="\r")
            start_position = self.bpg_preemphasis[tap_str]
            plus_position = start_position + step
            if start_position == 100:
                plus_specific_area = 0
                plus_position = None
            if plus_position and plus_position > 100:
                plus_position = 100
            if plus_position:
                self.shf_set_preemphasis(tap_index, plus_position)
                plus_area, plus_specific_area, _, _ = self.ea_autosearch(
                    type="SIMPLEBER", logber=logber
                )
                if plus_specific_area >= max_specific_area:
                    max_specific_area = plus_specific_area
                    max_area_taps = self.bpg_preemphasis.copy()
                    print(" " * 80, end="\r")
                    print(max_area_taps, f"{max_specific_area:3.3f} mV", end="\r")
            minus_position = start_position - step
            if start_position == -100:
                minus_specific_area = 0
                minus_position = None
            if minus_position and minus_position < -100:
                minus_position = -100
            if minus_position:
                self.shf_set_preemphasis(tap_index, minus_position)
                minus_area, minus_specific_area, _, _ = self.ea_autosearch(
                    type="SIMPLEBER", logber=logber
                )
                if minus_specific_area >= max_specific_area:
                    max_specific_area = minus_specific_area
                    max_area_taps = self.bpg_preemphasis.copy()
                    print(" " * 80, end="\r")
                    print(max_area_taps, f"{max_specific_area:3.3f} mV", end="\r")
            print(f"\t\t\tstart_specific_area: {start_specific_area:3.3f}\tplus_specific_area: {plus_specific_area:3.3f}\tminus_specific_area: {minus_specific_area:3.3f}")
            if plus_specific_area >= start_specific_area:
                prev_specific_area = plus_specific_area
                prev_position = plus_position
                while True:
                    new_position = prev_position + step
                    self.shf_set_preemphasis(tap_index, new_position)
                    new_area, new_specific_area, _, _ = self.ea_autosearch(
                        type="SIMPLEBER", logber=logber
                    )
                    if new_specific_area < prev_specific_area:
                        break
                    else:
                        prev_specific_area = new_specific_area
                        prev_position = new_position
                        if new_specific_area >= max_specific_area:
                            max_specific_area = new_specific_area
                            max_area_taps = self.bpg_preemphasis.copy()
                            print(" " * 80, end="\r")
                            print(
                                max_area_taps, f"{max_specific_area:3.3f} mV", end="\r"
                            )
            if minus_specific_area >= start_specific_area:
                prev_specific_area = minus_specific_area
                prev_position = minus_position
                while True:
                    new_position = prev_position - step
                    self.shf_set_preemphasis(tap_index, new_position)
                    new_area, new_specific_area, _, _ = self.ea_autosearch(
                        type="SIMPLEBER", logber=logber
                    )
                    if new_specific_area < prev_specific_area:
                        break
                    else:
                        prev_specific_area = new_specific_area
                        prev_position = new_position
                        if new_specific_area >= max_specific_area:
                            max_specific_area = new_specific_area
                            max_area_taps = self.bpg_preemphasis.copy()
                            print(" " * 80, end="\r")
                            print(
                                max_area_taps, f"{max_specific_area:3.3f} mV", end="\r"
                            )
            self.shf_set_preemphasis(tap_index, max_area_taps[tap_str])
        return max_area_taps, max_specific_area

    def optimize_amplitude(self, logbermin=-9, logbermax=-4, exact=True):
        start_qfactor_amplitude = self.dac_amplitude
        start_qfactor = self.measure_qfactor(
            logbermin=logbermin, logbermax=logbermax, exact=exact
        )
        list_of_steps = [100, 100, 50, 50, 10, 10, 5, 5]
        print("Optimizing DAC amplitude")
        for i, step in enumerate(list_of_steps):
            print(" " * 80, end="\r")
            print(f"[{i}/{len(list_of_steps)}] step: {step}" + " " * 50)
            max_qfactor = self.optim_amplitude_itteration(
                step=step, logbermin=logbermin, logbermax=logbermax, exact=exact
            )
        print(" " * 80, end="\r")
        print(
            f"Amplitude: {self.dac_amplitude} mV,\tQfactor: {start_qfactor:3.3f} -> {max_qfactor:3.3f}"
        )
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"DAC Amplitude is optimized. Amplitude: {self.dac_amplitude} mV, Qfactor: {start_qfactor:3.3f} -> {max_qfactor:3.3f}",
            ]
        )

    def optim_amplitude_itteration(
        self,
        lowest_amplitude: int = 100,
        largest_amplitude: int = 1000,
        step: int = 10,
        logbermin=-9,
        logbermax=-2,
        exact=True,
    ):
        max_qfactor = 0
        max_qfactor_amplitude = self.dac_amplitude
        plus_qfactor = -1
        minus_qfactor = -1
        start_qfactor = self.measure_qfactor(
            logbermin=logbermin, logbermax=logbermax, exact=exact
        )
        if start_qfactor >= max_qfactor:
            max_qfactor = start_qfactor
            max_qfactor_amplitude = self.dac_amplitude
            print(" " * 80, end="\r")
            print(
                f"Amplitude: {max_qfactor_amplitude:3.3f} mV, Qfactor: {max_qfactor:3.3f}",
                end="\r",
            )
        start_position = self.dac_amplitude
        plus_position = start_position + step
        if start_position == largest_amplitude:
            plus_qfactor = -1
            plus_position = None
        if plus_position and plus_position > largest_amplitude:
            plus_position = largest_amplitude
        if plus_position:
            self.shf_set_amplitude(plus_position)
            plus_qfactor = self.measure_qfactor(
                logbermin=logbermin, logbermax=logbermax, exact=exact
            )
            if plus_qfactor >= max_qfactor:
                max_qfactor = plus_qfactor
                max_qfactor_amplitude = self.dac_amplitude
                print(" " * 80, end="\r")
                print(
                    f"Amplitude: {max_qfactor_amplitude:3.3f} mV, Qfactor: {max_qfactor:3.3f}",
                    end="\r",
                )
        minus_position = start_position - step
        if start_position == lowest_amplitude:
            minus_qfactor = -1
            minus_position = None
        if minus_position and minus_position < lowest_amplitude:
            minus_position = lowest_amplitude
        if minus_position:
            self.shf_set_amplitude(minus_position)
            minus_qfactor = self.measure_qfactor(
                logbermin=logbermin, logbermax=logbermax, exact=exact
            )
            if minus_qfactor >= max_qfactor:
                max_qfactor = minus_qfactor
                max_qfactor_amplitude = self.dac_amplitude
                print(" " * 80, end="\r")
                print(
                    f"Amplitude: {max_qfactor_amplitude:3.3f} mV, Qfactor: {max_qfactor:3.3f}",
                    end="\r",
                )
        if plus_qfactor >= start_qfactor:
            prev_qfactor = plus_qfactor
            prev_position = plus_position
            while True:
                new_position = prev_position + step
                self.shf_set_amplitude(new_position)
                new_qfactor = self.measure_qfactor(
                    logbermin=logbermin, logbermax=logbermax, exact=exact
                )
                if new_qfactor < prev_qfactor:
                    break
                else:
                    prev_qfactor = new_qfactor
                    prev_position = new_position
                    if new_qfactor >= max_qfactor:
                        max_qfactor = new_qfactor
                        max_qfactor_amplitude = self.dac_amplitude
                        print(" " * 80, end="\r")
                        print(
                            f"Amplitude: {max_qfactor_amplitude:3.3f} mV, Qfactor: {max_qfactor:3.3f}",
                            end="\r",
                        )
        if minus_qfactor >= start_qfactor:
            prev_qfactor = minus_qfactor
            prev_position = minus_position
            while True:
                new_position = prev_position - step
                self.shf_set_amplitude(new_position)
                new_qfactor = self.measure_qfactor(
                    logbermin=logbermin, logbermax=logbermax, exact=exact
                )
                if new_qfactor < prev_qfactor:
                    break
                else:
                    prev_qfactor = new_qfactor
                    prev_position = new_position
                    if new_qfactor >= max_qfactor:
                        max_qfactor = new_qfactor
                        max_qfactor_amplitude = self.dac_amplitude
                        print(" " * 80, end="\r")
                        print(
                            f"Amplitude: {max_qfactor_amplitude:3.3f} mV, Qfactor: {max_qfactor:3.3f}",
                            end="\r",
                        )
        self.shf_set_amplitude(max_qfactor_amplitude)
        return max_qfactor

    def optimize_current(
        self,
        lowest_current: float = 1,
        largest_current: float = 2,
        logbermin=-9,
        logbermax=-4,
        exact=True,
    ):
        start_qfactor_amplitude = self.dac_amplitude
        start_qfactor = self.measure_qfactor(
            logbermin=logbermin, logbermax=logbermax, exact=exact
        )
        currentdelta = largest_current - lowest_current
        list_of_steps = []
        if currentdelta >= 35:
            list_of_steps.append(7)
            list_of_steps.append(7)
        if currentdelta >= 15:
            list_of_steps.append(3)
            list_of_steps.append(3)
        if currentdelta >= 10:
            list_of_steps.append(2)
            list_of_steps.append(2)
        if currentdelta >= 5:
            list_of_steps.append(1)
            list_of_steps.append(1)
        if currentdelta >= 2:
            list_of_steps.append(0.5)
            list_of_steps.append(0.5)
        list_of_steps.append(0.1)
        list_of_steps.append(0.1)
        print("Optimizing current")
        for i, step in enumerate(list_of_steps):
            print(" " * 80, end="\r")
            print(f"[{i}/{len(list_of_steps)}] step: {step}" + " " * 50)
            max_qfactor = self.optim_current_itteration(
                step=step,
                lowest_current=lowest_current,
                largest_current=largest_current,
                logbermin=logbermin,
                logbermax=logbermax,
                exact=exact,
            )
        print(" " * 80, end="\r")
        print(
            f"Current: {self.current} mA,\tQfactor: {start_qfactor:3.3f} -> {max_qfactor:3.3f}"
        )
        self.logs.append(
            [
                time.strftime("%Y%m%d-%H%M%S"),
                f"Current is optimized. Current: {self.current} mA, Qfactor: {start_qfactor:3.3f} -> {max_qfactor:3.3f}",
            ]
        )

    def optim_current_itteration(
        self,
        lowest_current: float = 1,
        largest_current: float = 2,
        step: int = 10,
        logbermin=-9,
        logbermax=-2,
        exact=True,
    ):
        max_qfactor = 0
        max_qfactor_current = self.current
        start_qfactor = self.measure_qfactor(
            logbermin=logbermin, logbermax=logbermax, exact=exact
        )
        if start_qfactor >= max_qfactor:
            max_qfactor = start_qfactor
            max_qfactor_current = self.current
            print(" " * 80, end="\r")
            print(
                f"Current {max_qfactor_current:3.3f} mA, Qfactor: {max_qfactor:3.3f}",
                end="\r",
            )
        start_position = self.current
        plus_position = start_position + step
        if start_position == largest_current:
            plus_qfactor = -1
            plus_position = None
        if plus_position and plus_position > largest_current:
            plus_position = largest_current
        if plus_position:
            self.gently_apply_current(plus_position)
            plus_qfactor = self.measure_qfactor(
                logbermin=logbermin, logbermax=logbermax, exact=exact
            )
            if plus_qfactor >= max_qfactor:
                max_qfactor = plus_qfactor
                max_qfactor_current = self.current
                print(" " * 80, end="\r")
                print(
                    f"Current {max_qfactor_current:3.3f} mA, Qfactor: {max_qfactor:3.3f}",
                    end="\r",
                )
        minus_position = start_position - step
        if start_position == lowest_current:
            minus_qfactor = -1
            minus_position = None
        if minus_position and minus_position < lowest_current:
            minus_position = lowest_current
        if minus_position:
            self.gently_apply_current(minus_position)
            minus_qfactor = self.measure_qfactor(
                logbermin=logbermin, logbermax=logbermax, exact=exact
            )
            if minus_qfactor >= max_qfactor:
                max_qfactor = minus_qfactor
                max_qfactor_current = self.current
                print(" " * 80, end="\r")
                print(
                    f"Current {max_qfactor_current:3.3f} mA, Qfactor: {max_qfactor:3.3f}",
                    end="\r",
                )
        if plus_qfactor >= start_qfactor:
            prev_qfactor = plus_qfactor
            prev_position = plus_position
            while True:
                new_position = prev_position + step
                self.gently_apply_current(new_position)
                new_qfactor = self.measure_qfactor(
                    logbermin=logbermin, logbermax=logbermax, exact=exact
                )
                if new_qfactor < prev_qfactor:
                    break
                else:
                    prev_qfactor = new_qfactor
                    prev_position = new_position
                    if new_qfactor >= max_qfactor:
                        max_qfactor = new_qfactor
                        max_qfactor_current = self.current
                        print(" " * 80, end="\r")
                        print(
                            f"Current {max_qfactor_current:3.3f} mA, Qfactor: {max_qfactor:3.3f}",
                            end="\r",
                        )
        if minus_qfactor >= start_qfactor:
            prev_qfactor = minus_qfactor
            prev_position = minus_position
            while True:
                new_position = prev_position - step
                self.gently_apply_current(new_position)
                new_qfactor = self.measure_qfactor(
                    logbermin=logbermin, logbermax=logbermax, exact=exact
                )
                if new_qfactor < prev_qfactor:
                    break
                else:
                    prev_qfactor = new_qfactor
                    prev_position = new_position
                    if new_qfactor >= max_qfactor:
                        max_qfactor = new_qfactor
                        max_qfactor_current = self.current
                        print(" " * 80, end="\r")
                        print(
                            f"Current {max_qfactor_current:3.3f} mA, Qfactor: {max_qfactor:3.3f}",
                            end="\r",
                        )
        self.gently_apply_current(max_qfactor_current)
        return max_qfactor

    def measure_eye_diagrams(self):  # TODO
        "get an eye diagram picture"
        pass

    def calculate_edr(self):  # TODO
        "calculate EDR (from an eye diagram or BER)?"
        pass

    def optimize_bitrate(self):  # TODO
        "optimize parameters for best bitrate at particular current"
        best_bitrate = 0  # GHz
        best_parameters = {
            "curent": 0,
        }
        return best_bitrate, best_parameters

    def test_ea(self):
        self.ea_init()
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(35)
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(40)
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(45)
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(50)
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(55)
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(60)
        #self.ea_autosearch()
        #self.shf_set_clksrc_frequency(25)
        #self.ea_autosearch()
        #self.ea_autosearch()
        #self.ea_autosearch()
        self.measure_qfactor()
        self.measure_qfactor()
        self.measure_qfactor()
        self.measure_ber()
        #input("optimize preemphasis?")
        #self.optimize_preemphasis()
        #self.measure_qfactor()
        #self.ea_autosearch()
        #self.measure_ber()
        #input("optimize amplitude?")
        self.optimize_amplitude()
        self.measure_qfactor()
        self.ea_autosearch()
        self.measure_ber()
        #input("optimize current?")
        self.optimize_current(lowest_current=4, largest_current=14)
        # self.shf_command("EA:PAMMEASUREMENTPOINTS=CHANNEL1:?;")
        # self.test_ea_job()
        # self.shf_command("EA:RESULT=CHANNEL1:?;")
        # self.shf_command("EA:MEASUREBERPAM=CHANNEL1:ON;")
        # self.test_ea_job()
        # elf.shf_command("EA:RESULT=CHANNEL1:?;")
        # self.shf_command("EA:MEASUREBERPAM=CHANNEL1:ON;")
        # self.test_ea_job()
        # self.shf_command("EA:RESULT=CHANNEL1:?;")

    def test(self):
        self.rst_current_source()
        print("rst_current_source done")
        assert self.rst_attenuator() == True
        print("rst_attenuator done")
        self.gently_apply_current(12)
        self.update_attenuation_data()
        self.shf_init()
        self.shf_patternsetup("nrz")
        self.shf_set_amplitude(400)
        self.shf_set_preemphasis(0, 0)
        self.shf_set_preemphasis(2, 0)
        self.shf_set_preemphasis(3, 0)
        self.shf_set_clksrc_frequency(35)
        self.set_attenuation(-2)
        self.open_attenuator_shutter()
        self.test_ea()
        self.attenuator_shutter("close")
