#! /usr/bin/env python3
# -*- coding: utf-8 -*-
"""Transform original installation data files into a unified raw data file."""

import os
import pandas as pd
from src.config import (
    DATA_ORIG_DIR,
    DATA_RAW_DIR,
    DATA_RAW_FILENAME,
    INSTALLATION_DATA_FILENAME,
)
from src.data import gather_and_transform_data

if __name__ == "__main__":
    installation_metadata = pd.read_csv(
        os.path.join(DATA_ORIG_DIR, INSTALLATION_DATA_FILENAME),
        sep=";",
        index_col="installation",
    )

    all_power_data = gather_and_transform_data(
        installation_metadata=installation_metadata, orig_data_dir_name=DATA_ORIG_DIR
    )

    raw_data_pathfilename: str = os.path.join(DATA_RAW_DIR, DATA_RAW_FILENAME)

    if not all_power_data.empty:
        # Ensure the raw data directory exists
        os.makedirs(os.path.dirname(raw_data_pathfilename), exist_ok=True)

        all_power_data.to_csv(raw_data_pathfilename, index=False, sep=";")
