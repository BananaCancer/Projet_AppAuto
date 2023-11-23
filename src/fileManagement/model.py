# -*- coding: utf-8 -*-
"""
Created on Thu Nov 23 14:02:59 2023

@author: basil
"""

from joblib import dump, load

def saveModel(model, filename):
    dump(model, filename)
    
def loadModel(filename):
    return load(filename)