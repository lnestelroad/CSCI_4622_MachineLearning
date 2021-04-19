#!/usr/bin/env python3

import pathlib
import pandas as pd
import numpy as np

FILE_PATH = str(pathlib.Path(__file__).parent.absolute())
CROSS_SECTIONAL = FILE_PATH + '/oasis_cross-sectional.csv'
LONGITUDINAL = FILE_PATH + '/oasis_longitudinal.csv'



data_l = pd.read_csv(LONGITUDINAL)
data_cs = pd.read_csv(CROSS_SECTIONAL)
