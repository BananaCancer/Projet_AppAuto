# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:16:32 2023

@author: basil
"""
import numpy as np
from sklearn.metrics import silhouette_score, calinski_harabasz_score, davies_bouldin_score

def score(df, clusters, clusterer, scoreDic, min_clust = 0, file = None, verbose = False):
    if len(np.unique(clusters)) > 1:
        # The silhouette_score gives the average value for all the samples.
        # This gives a perspective into the density and separation of the formed
        # clusters
        silhouette_avg = silhouette_score(df, clusters)
        calinski_harabasz = calinski_harabasz_score(df, clusters)
        davies_bouldin = davies_bouldin_score(df, clusters)
        unique_values, counts = np.unique(clusters, return_counts=True)
        if (silhouette_avg > 0.1 and len(unique_values) > min_clust):
            if verbose:
                print(f"\tNb clusters: {len(np.unique(clusters))}\n")
            unique_values, counts = np.unique(clusters, return_counts=True)
            if (len(unique_values) <= 5):
                for value, count in zip(unique_values, counts):
                    if verbose:
                        print(f"\t\t{value}: {count} occurrences\n")
            scoreDic["silhouette"] = silhouette_avg
            scoreDic["calinski-harabasz"] = calinski_harabasz
            scoreDic["davies-bouldin"] = davies_bouldin
            scoreDic["clusters"] = clusters
            scoreDic["model"] = clusterer
            if verbose:
                print(f"\tThe average silhouette_score is: {silhouette_avg}\n"
                      f"\tThe calinski-harabasz score is: {calinski_harabasz} (the higher the better)\n"
                      f"\tThe davies-bouldin score is: {davies_bouldin} (the lower the better)\n")        
        return silhouette_avg
    else:
        return None
