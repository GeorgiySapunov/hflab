Script fot LIV measurements using following equipment:
- Keysight 8163B Lightwave Multimeter
- Keysight B2901A Precision Source/Measure Unit
- Thorlabs PM100USB Power and energy meter

If you are told your version of pip is out of date, you might as well update it with

    python -m pip install –upgrade pip

Once you have that installed, you will need to install pyvisa. Under Windows, that is as simple as running

    pip install pyvisa

Finally, as pyvisa is really only a “wrapper” for a VISA layer, you will need to install a supported VISA layer for it to talk to. If you are running Windows, realistically, you should use the National Instruments NI-VISA package. It’s a bulky piece of software, but it is the most well-supported VISA layer generally speaking – a version of it may already be installed on your computer by the “bundled” software that comes with your instrument. It is possible to avoid using NI-VISA on Windows, but then you will need to cook up your own drivers for each instrument and self-sign them and it’s just a whole heap of hassle.

On Linux, however, you will also need to install pyvisa-py which provides a Python-based VISA layer, as well as pyusb (for USB-TMC instruments) and pyserial (for Serial connected instruments). This can be done by using the following command (both with and without sudo)

    pip install pyvisa-py pyusb pyserial

1. https://goughlui.com/2021/03/28/tutorial-introduction-to-scpi-automation-of-test-equipment-with-pyvisa/
2. https://youtu.be/1HQxnz3P9P4
