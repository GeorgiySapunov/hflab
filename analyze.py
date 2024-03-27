#!/usr/bin/env python3

import sys
import re
import os
import yaml
import numpy as np
import pandas as pd
import click
from pathlib import Path

from src.replot_liv import replot_liv_function
from src.analysis_ssm import analyze_ssm_function
from src.analysis_rin import analyze_rin_function
from src.analysis_os import analyze_os_function
from src.combine_liv import combine_liv_function
from src.combine_os import combine_os_function
from src.combine_ssm_reports import combine_ssm_reports_function


def print_help():
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    ctx.exit()


@click.command(context_settings={"ignore_unknown_options": True})
@click.option(
    "-l",
    "--replot_liv",
    is_flag=True,
    show_default=True,
    default=False,
    help="Replot LIV figures",
)
@click.option(
    "-s",
    "--ssm",
    is_flag=True,
    show_default=True,
    default=False,
    help="Analyze small signal modulation data (.s2p files or automatic system reports)",
)
@click.option(
    "-r",
    "--rin",
    is_flag=True,
    show_default=True,
    default=False,
    help="Analyze RIN data (automatic system reports)",
)
@click.option(
    "-o",
    "--optical_spectra",
    is_flag=True,
    show_default=True,
    default=False,
    help="Plot optical spectra",
)
@click.option(
    "-y",
    "--yaml_project",
    is_flag=True,
    show_default=True,
    default=False,
    help="Read and analize according to settings in YAML files",
)
@click.argument("paths", nargs=-1)
def analyze(replot_liv, ssm, rin, optical_spectra, yaml_project, paths):
    if not any((replot_liv, ssm, rin, optical_spectra, yaml_project)):
        print_help()
    for i, path in enumerate(map(Path, paths), start=1):
        print(f"[{i}/{len(paths)}] {path}")
        if path.suffix in (".png", ".zip", ".txt", ".org", ".md", ".doc", ".docx"):
            pass
        if replot_liv:
            replot_liv_function(path)
        if ssm:
            analyze_ssm_function(path)
        if rin:
            analyze_rin_function(path)
        if optical_spectra:
            analyze_os_function(path)
        if yaml_project:
            with open(path) as fh:
                settings = yaml.safe_load(fh)
                print(settings)
            algorithm_name = settings["algorithm_name"]
            if algorithm_name in ("replot_liv", "ssm", "rin", "optical_spectra"):
                directories = settings["directories"]
                for i, path_in_yaml in enumerate(map(Path, directories), start=1):
                    print(f"{i}: [{i}/{len(directories)}] {path_in_yaml}")
                    if algorithm_name == "replot_liv":
                        replot_liv_function(path_in_yaml, settings=settings)
                    if algorithm_name == "ssm":
                        analyze_ssm_function(path_in_yaml, settings=settings)
                    if algorithm_name == "rin":
                        analyze_rin_function(path_in_yaml, settings=settings)
                    if algorithm_name == "optical_spectra":
                        analyze_os_function(path_in_yaml, settings=settings)
            else:
                if algorithm_name == "combine_liv":
                    combine_liv_function(path, settings=settings)
                if algorithm_name == "combine_os":
                    combine_os_function(path, settings=settings)
                if algorithm_name == "combine_ssm":
                    combine_ssm_reports_function(path, settings=settings)

        print(f"[{i}/{len(paths)}] {path} is done")


if __name__ == "__main__":
    analyze()
