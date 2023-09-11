#!/usr/bin/env python3

#           _   _   _
#          | | | | (_)
#  ___  ___| |_| |_ _ _ __   __ _ ___
# / __|/ _ \ __| __| | '_ \ / _` / __|
# \__ \  __/ |_| |_| | | | | (_| \__ \
# |___/\___|\__|\__|_|_| |_|\__, |___/
#                            __/ |
#                           |___/
# Keysight_B2901A_address = "GPIB0::23::INSTR"
Keysight_B2901A_address = "USB0::0x0957::0x8B18::MY51143485::INSTR"
Thorlabs_PM100USB_address = "USB0::0x1313::0x8072::1923257::INSTR"
Keysight_8163B_address = "GPIB0::10::INSTR"
#YOKOGAWA_AQ6370D_address = "TCPIP0::169.254.5.10::10001::SOCKET"
YOKOGAWA_AQ6370D_address = "GPIB0::1::INSTR"
ATT_A160CMI_address = ""


# list of current to measure (from 0 to 50 mA, 0.01 mA steps)
current_list = (i / 1000000 for i in range(0, 50000, 10))
beyond_rollover_stop_cond = 0.9  # stop if power lower then 90% of max output power
current_limit1 = 4  # mA, stop measuremet if current above limit1 (mA) and output power less then 0.01 mW
current_limit2 = 10  # mA, stop measuremet if current above limit2 (mA) and maximum output power less then 0.5 mW

temperature_limit = 110

settings = {
    # Keysight_B2901A_address: "GPIB0::23::INSTR",
    "Keysight_B2901A_address": "USB0::0x0957::0x8B18::MY51143485::INSTR",
    "Thorlabs_PM100USB_address": "USB0::0x1313::0x8072::1923257::INSTR",
    "Keysight_8163B_address": "GPIB0::10::INSTR",
    "YOKOGAWA_AQ6370D_address": "GPIB0::1::INSTR",
    "ATT_A160CMI_address": "",
    # list of current to measure (from 0 to 50 mA, 0.01 mA steps)
    "current_list": (i / 1000000 for i in range(0, 50000, 10)),
    "beyond_rollover_stop_cond": 0.9,  # stop if power lower then 90% of max output power
    "current_limit1": 4,  # mA, stop measuremet if current above limit1 (mA) and output power less then 0.01 mW
    "current_limit2": 10,  # mA, stop measuremet if current above limit2 (mA) and maximum output power less then 0.5 mW
    "temperature_limit": 110,
}
