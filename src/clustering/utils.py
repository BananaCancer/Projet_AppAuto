# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:15:34 2023

@author: basil
"""

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
import pandas as pd
import itertools

def doPCA(df, components):
    if components > 0:
        pca = PCA(n_components = components) # Keep 2 or 3 components
        df = pca.fit_transform(df)
        df = pd.DataFrame(df)
        scaler = StandardScaler()
        df_standardized = scaler.fit_transform(df)
        df = pd.DataFrame(df_standardized, columns=df.columns)
    return df

def makeDictionnary(*params, existing = None):
    if existing == None:
        result = {}
    else:
        result = existing
    for values in itertools.product(*params):
        current_dict = result
        for value in values[:-1]:
            current_dict = current_dict.setdefault(value, {})
        
        # Set default values for the last key
        current_dict[values[-1]] = {
            "silhouette": None,
            "calinski-harabasz": None,
            "davies-bouldin": None,
            "clusters": None,
            "model": None
        }

    return result

def getClusters(model, df, components = -1):
    df = doPCA(df, components)
    return model.predict(df)