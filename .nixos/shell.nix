# shell.nix
{ pkgs ? import <nixpkgs> {} }:
let
  my-python-packages = ps: with ps; [
    scikit-rf
    numpy
    pandas
    scipy
    scikit-learn
    matplotlib
    seaborn
    pyvisa
    pyvisa-py
    pyusb
    pyserial
    psutil
    zeroconf
    pyfiglet
    rich
    # # other python packages
    # # ...
    # (
    #   buildPythonPackage rec {
    #     pname = "pysmithplot-fork";
    #     version = "0.2.1";
    #     src = builtins.fetchurl {
    #       url = "https://files.pythonhosted.org/packages/9a/95/4e09c1d90da71a5a3045f64f4ba690faa6cf96e2acd7ef7c3c4fd7b86d35/pysmithplot_fork-0.2.1.tar.gz";
    #       sha256 = "4a11146f53f825d2b5859379fad3fac02b0242b7aef1dcb7d284c3366e851bd6";
    #     };
    #     doCheck = false;
    #     propagatedBuildInputs = [
    #       # Specify dependencies
    #       pkgs.python3Packages.matplotlib
    #       pkgs.python3Packages.scipy
    #     ];
    #   }
    # )
  ];
  my-python = pkgs.python311.withPackages my-python-packages;
in my-python.env
