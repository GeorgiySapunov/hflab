#!/usr/bin/env python3
import sys
import os
import re
import time
import csv
import json
import numpy as np
import pandas as pd
from pathlib import Path
import xml.etree.ElementTree as ET

skew_dict = {}

dir = Path("resources") / "DeviceStates"
for gb in range(70):
    files = list(dir.glob(f"{gb}G*.xml"))
    if files:
        if len(files) > 1:
            print(files)
        skew_dict[f"{gb} Gbaud"] = [None, None, None, None, None, None]
        with open(files[0], "r") as file:
            file = file.read()
            file = file.replace("\x08", "").replace("\x0c", "").replace(":", "")
            # print(file)
            tree = ET.fromstring(file)
            # root = tree.getroot()
            # tree.find("Channel1")
            for i in range(1, 7):
                skew = int(
                    float(
                        tree.find("modules")
                        .find("device")
                        .find("Channels")
                        .find(f"Channel{i}")
                        .find("Skew")
                        .attrib["Value"]
                    )
                    * 10**12
                )

                skew_dict[f"{gb} Gbaud"][i - 1] = skew

# df = pd.DataFrame(skew_dict)
df = pd.DataFrame(
    skew_dict,
    index=[
        "Channel 1",
        "Channel 2",
        "Channel 3",
        "Channel 4",
        "Channel 5",
        "Channel 6",
    ],
).transpose()
print(df)
df.to_csv((Path("resources") / "skew.csv"))
