"""
Transform the solstice data to a simple format

This script transforms the solstice and equinox data into a more usable format by parsing the date strings and normalizing the timezone information.
"""

import os
import pandas as pd
from src.config import DATA_ORIG_DIR
import dateparser

df = pd.read_csv(os.path.join(DATA_ORIG_DIR, "solstice,equinox.csv"), sep=";")


df["date"] = pd.to_datetime(
    df["Date"].map(lambda s: dateparser.parse(s, languages=["en"])), utc=True
)
df["date"] = df["date"].dt.tz_localize(None)
df = df.drop(columns=["Date"])
df.to_csv(
    os.path.join(DATA_ORIG_DIR, "solstice_equinox_berlin.csv"), sep=";", index=False
)
