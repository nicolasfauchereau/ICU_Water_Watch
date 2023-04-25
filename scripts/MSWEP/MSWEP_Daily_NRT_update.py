#!/usr/bin/env python
# coding: utf-8

import argparse
from ICU_Water_Watch import MSWEP

def main():

    parser = argparse.ArgumentParser(
        prog="MSWEP_Daily_NRT_update.py",
        description="""update the MSWEP daily NRT dataset locally, from the glo2ho FTP server""",
    )

    parser.add_argument(
        "-c",
        "--credentials",
        type=str,
        default='./MSWEP_credentials.txt',
        help="""Text file with login and password for data.gloh2o.org\
        \ndefault `MSWEP_credentials.txt`""",
    )

    parser.add_argument(
        "-o",
        "--opath",
        type=str,
        default='/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/NRT/Daily/',
        help="""Path to the local daily MSWEP netcdf files (one file per day, NRT version)
        \ndefault `/media/nicolasf/END19101/ICU/data/glo2ho/MSWEP280/NRT/Daily/`""",
    )

    args = parser.parse_args()

    credentials = args.credentials
    opath = args.opath

    MSWEP.update(credentials=credentials, opath=opath)

if __name__ == "__main__":
    main()


