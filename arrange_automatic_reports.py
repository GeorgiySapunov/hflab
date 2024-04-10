#!/usr/bin/env python3

import click
import datetime
import hashlib
from pathlib import Path
from termcolor import colored


def print_help():
    ctx = click.get_current_context()
    click.echo(ctx.get_help())
    ctx.exit()


def parce_time(time=None, file=None):
    if file:
        filestem = file.stem
        waferid_wavelength, cellxy_coordinatesxy, time = filestem.split("_")
    if not time:
        return
    time = datetime.datetime(
        year=int(time[:4]),
        month=int(time[4:6]),
        day=int(time[6:8]),
        hour=int(time[8:10]),
        minute=int(time[10:12]),
        second=int(time[12:]),
    )
    # print(time.strftime("%m/%d/%Y, %H:%M:%S"))
    return time


def move_file(file):
    filestem = file.stem
    waferid_wavelength, cellxy_coordinatesxy, time = filestem.split("_")
    # waferid, wavelength = waferid_wavelength.split("-")
    cellxy, coordinatesxy = cellxy_coordinatesxy.split("-")
    data_dir = Path("data")
    data_dir.mkdir(exist_ok=True)
    wafer_dir = data_dir / f"{waferid_wavelength}"
    wafer_dir.mkdir(exist_ok=True)
    vcsel_report_dir = wafer_dir / f"{cellxy}{coordinatesxy}"
    vcsel_report_dir.mkdir(exist_ok=True)
    newpath = vcsel_report_dir / (filestem + ".csv")
    other_automatic_files = sorted(
        list(
            vcsel_report_dir.glob(
                f"{waferid_wavelength}_{cellxy_coordinatesxy}_" + "*.csv"
            )
        )
    )
    for other_automatic_file in other_automatic_files:
        hash_file = hashlib.md5(open(file, "rb").read()).hexdigest()
        hash_other_file = hashlib.md5(
            open(other_automatic_file, "rb").read()
        ).hexdigest()
        if hash_file == hash_other_file:
            file_time = parce_time(file=file).strftime("%Y/%m/%d, %H:%M:%S")
            other_file_time = parce_time(file=other_automatic_file).strftime(
                "%Y/%m/%d, %H:%M:%S"
            )
            print(
                colored(
                    f"old timestemp: {other_file_time}\nnew timestemp: {file_time}\nfile exists, skipping",
                    "red",
                )
            )
            return
    file.rename(newpath)
    print(f"{newpath}")


@click.command(context_settings={"ignore_unknown_options": True})
@click.argument("paths", nargs=-1)
def arrange_automatic_reports(paths):
    if not paths:
        print_help()
    for i, path in enumerate(map(Path, paths), start=1):
        print(f"[{i}/{len(paths)}] {path}")
        if path.suffix == ".csv":
            move_file(path)
        if path.is_dir():
            files = sorted(path.glob("*.csv"))
            for j, file in enumerate(files, start=1):
                print(f"[{i}/{len(paths)} {j}/{len(files)}] {file}")
                move_file(file)


if __name__ == "__main__":
    arrange_automatic_reports()
