# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:03:15 2023

@author: basil
"""

import pandas as pd
import os.path

def loadFile(filename, delimiter):
    return pd.read_csv(filename, delimiter = delimiter)

def fileExists(path):
    return os.path.isfile(path)