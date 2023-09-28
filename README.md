1. Scripts for LIV and optical spectra measurements using following equipment:
    - Keysight B2901A Precision Source/Measure Unit
    - Thorlabs PM100USB Power and energy meter
    - Keysight 8163B Lightwave Multimeter
    - YOKOGAWA AQ6370D Optical Spectrum Analyzer
2. Script to approximate small signal modulation signal (S-parameters) at
   different currents stored in multiple .s2p files
3. Script to calculate and plot thermal resistance of a VCSEL

# How to use:
## *To measure LIV* run
```zsh
python measure.py
```
You will get a small instruction on what arguments to provide. settings.py file
contains a dictionary with visa addresses and settings.

following arguments are needed: Equipment_choice WaferID Wavelength(nm) Coordinates Temperature(°C)
e.g. run
```zsh
python measure.py k2 gs15 1550 00C9 25'
```
 > **_NOTE:_** Character '-' is not allowed! It breaks filename parsing.

for equipment choice use:

- t    for Thorlabs PM100USB Power and energy meter
- k1   for Keysight 8163B Lightwave Multimeter port 1
- k2   for Keysight 8163B Lightwave Multimeter port 2
- y    for YOKOGAWA AQ6370D Optical Spectrum Analyzer

<!-- for multiple temperature you need to specify start, stop and step temperature values: Equipment_choice WaferID Wavelength(nm) Coordinates Start_Temperature(°C) Stop_Temperature(°C) Temperature_Increment(°C) -->

<!-- e.g. run -->
<!-- ```zsh -->
<!-- python measure.py t gs15 1550 00C9 25 85 40 -->
<!-- ``` -->
<!-- in this case you will get LIVs for 25, 65 and 85 degrees -->

## *To approximate S-parameters* run 
```zsh
python ssm_analysis.py *directories with .s2p files (or with PNA directory with .s2p files)*
```
for example: 
```zsh
python smm_analysis.py data/test-1550nm/0000/PNA/
```
or 
```zsh
python smm_analysis.py data/test-1550nm/*
```
Filenames should be \{WaferID\}-\{Wavelength\}-\{Coordinates\}-\{Current\}mA.s2p, e.g.
Sample1-850-0022-1.0mA.s2p

## *To calculate thermal resistance* run 
```zsh 
python thermal_resistance.py *directories with LIV and OSA directories inside*
``` 
for example:
```zsh
python thermal_resistance.py data/test-1550nm/0000/
```
or 
```zsh
python thermal_resistance.py data/test-1550nm/*
```

# Needed python libraries
```zsh
pip install numpy scikit-rf pandas scipy scikit-learn matplotlib pyvisa pyvisa-py pyusb pyserial psutil zeroconf
```
### Install from requirements.txt
```zsh
pip install -r requirements.txt
```
### Make virtual environment
```zsh
python -m venv env
```
(note: "env" is the name of a directory with a virtual environment)
### Activate an environment (Unix)
```zsh
source env\bin\activate
```
### Activate an environment (Windows)
```powershell
env\Scripts\Activate.Ps1
```
# note: PyVISA installation
If you are told your version of pip is out of date, you might as well update it
with
```zsh
python -m pip install –upgrade pip

```
Once you have that installed, you will need to install pyvisa. Under Windows,
that is as simple as running
```zsh
pip install pyvisa

```
Finally, as pyvisa is really only a “wrapper” for a VISA layer, you will need to
install a supported VISA layer for it to talk to. If you are running Windows,
realistically, you should use the National Instruments NI-VISA package. It’s a
bulky piece of software, but it is the most well-supported VISA layer generally
speaking – a version of it may already be installed on your computer by the
“bundled” software that comes with your instrument. It is possible to avoid
using NI-VISA on Windows, but then you will need to cook up your own drivers for
each instrument and self-sign them and it’s just a whole heap of hassle.

On Linux, however, you will also need to install pyvisa-py which provides a
Python-based VISA layer, as well as pyusb (for USB-TMC instruments) and pyserial
(for Serial connected instruments). This can be done by using the following
command (both with and without sudo)
```zsh
pip install pyvisa-py pyusb pyserial
pip install psutil zeroconf
```
1. https://goughlui.com/2021/03/28/tutorial-introduction-to-scpi-automation-of-test-equipment-with-pyvisa/
2. https://youtu.be/1HQxnz3P9P4
3. https://github.com/instrumentkit/InstrumentKit
4. https://instrumentkit.readthedocs.io/en/latest/_modules/instruments/yokogawa/yokogawa6370.html
