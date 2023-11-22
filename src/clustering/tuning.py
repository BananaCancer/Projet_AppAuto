# -*- coding: utf-8 -*-
"""
Created on Wed Nov 22 15:14:05 2023

@author: basil
"""
from sklearn.cluster import DBSCAN, AgglomerativeClustering, KMeans, SpectralClustering
from clustering.scoring import score
from clustering.plotting import plotSilhouette, plotScatterKmeans, plotScatterDBSCAN
import itertools
from sklearn.mixture import GaussianMixture
from clustering.utils import doPCA, makeDictionnary

def clusterKmeans(df, params, verbose):
    k = params["k"]
    
    results = makeDictionnary(k)
    for n_clusters in results:
        if verbose:
            print(f"Kmeans: Current parameters: {n_clusters}:\n")

        clusterer = KMeans(n_clusters=n_clusters, n_init="auto", random_state=10)
        labels = clusterer.fit_predict(df)
        
        #Scoring
        silhouette_avg = score(df, labels, clusterer, results[n_clusters], verbose = verbose)
        
        # Plots
        plotSilhouette(df, labels, n_clusters, silhouette_avg)
        plotScatterKmeans(df, labels, n_clusters, clusterer)
        
    return results

def clusterDBSCAN(df, params, verbose):
    epslist = params["eps"]
    min_sampleslist = params["min_samples"]
    all_triplets = list(itertools.product(epslist, min_sampleslist))
    
    results = makeDictionnary(epslist, min_sampleslist)
    for eps, min_samples in all_triplets:
        if verbose:
            print(f"DBSCAN: Current parameters: {eps}, {min_samples}")
        
        # Clustering
        clusterer = DBSCAN(eps=eps, min_samples=min_samples).fit(df)
        labels = clusterer.labels_
        
        # Scoring
        score(df, labels, clusterer, results[eps][min_samples], min_clust = 2, verbose = verbose)
        
        # Plots
        plotScatterDBSCAN(df, clusterer)
    
    return results

def clusterHierarchical(df, params, verbose):
    n_clusters = params["k"]
    metrics = params["metrics"]
    linkages = params["linkage"]
    all_triplets = list(itertools.product(n_clusters, metrics, linkages))
    
    results = makeDictionnary(n_clusters, metrics, linkages)
    
    for k, metric, linkage in all_triplets:
        if verbose:
            print(f"Hclust: Current parameters: {k}, {metric}, {linkage}")
        
        # Cluster
        clusterer = AgglomerativeClustering(n_clusters = k, metric = metric, 
                                            linkage = linkage).fit(df)
        labels = clusterer.labels_
        
        # Scoring
        score(df, labels, clusterer, results[k][metric][linkage], verbose = verbose)
    
    if params["ward"]:
        metrics = ["euclidean"]
        linkages = ["ward"]
        all_triplets = list(itertools.product(n_clusters, metrics, linkages))
        
        results = makeDictionnary(n_clusters, metrics, linkages, existing = results)
        
        for k, metric, linkage in all_triplets:
            if verbose:
                print(f"Hclust: Current parameters: {k}, {metric}, {linkage}")
            
            # Cluster
            clusterer = AgglomerativeClustering(n_clusters = k, metric = metric, 
                                                linkage = linkage).fit(df)
            labels = clusterer.labels_
            
            # Scoring
            score(df, labels, clusterer, results[k][metric][linkage], verbose = verbose)
    
    return results

def clusterGMM(df, params, verbose):
    n_components_range = params["k"]
    covariance_types = params["cov_type"]
    all_triplets = list(itertools.product(n_components_range, covariance_types))
    
    results = makeDictionnary(n_components_range, covariance_types)

    for k, covariance_type in all_triplets:
        if verbose:
            print(f"GMM: Current parameters: {k}, {covariance_type}")
        clusterer = GaussianMixture(n_components=k, 
                                    covariance_type=covariance_type, 
                                    random_state=42)
        clusterer.fit(df)
        labels = clusterer.predict(df)
        score(df, labels, clusterer, results[k][covariance_type], verbose = verbose)
    
    return results

def clusterSpectral(df, params, verbose):    
    n_clusters = params["k"]
    eigen_solvers = params["eigen_solvers"]
    n_neighbors = params["n_neighbors"]
    results = {}
    
    if "rbf" in params["affinities"]:
        results["rbf"] = makeDictionnary(n_clusters, eigen_solvers)
        all_triplets = list(itertools.product(n_clusters, eigen_solvers))
        for n_cluster, eigen_solver in all_triplets:
            if verbose:
                print(f"Current parameters: {n_cluster}, {eigen_solver}, rbf")
            clusterer = SpectralClustering(n_clusters=n_cluster, 
                                        eigen_solver=eigen_solver, 
                                        random_state=42,
                                        affinity = "rbf")
            clusterer.fit(df)
            labels = clusterer.labels_
            score(df, labels, clusterer, results["rbf"][n_cluster][eigen_solver], verbose = verbose)

    if "nearest_neighbors" in params["affinities"]:
        results["nearest_neighbors"] = makeDictionnary(n_clusters, eigen_solvers, n_neighbors)
        all_triplets = list(itertools.product(n_clusters, eigen_solvers, n_neighbors))
        for n_cluster, eigen_solver, n_neighbor in all_triplets:
            if verbose:
                print(f"Current parameters: {n_cluster}, {eigen_solver}, nearest_neighbors, {n_neighbor}")
            clusterer = SpectralClustering(n_clusters=n_cluster, 
                                        eigen_solver=eigen_solver, 
                                        random_state=42,
                                        affinity = "nearest_neighbors",
                                        n_neighbors = n_neighbor)
            clusterer.fit(df)
            labels = clusterer.labels_
            score(df, labels, clusterer, results["nearest_neighbors"][n_cluster][eigen_solver][n_neighbor], verbose = verbose)
    
    return results

algorithms = {
    "Kmeans": clusterKmeans,
    "DBSCAN": clusterDBSCAN,
    "Hclust": clusterHierarchical,
    "GMM": clusterGMM,
    "Spectral": clusterSpectral
}

def tuneAlgorithms(df, listAlgorithms, components = -1, verbose = True):
    df = doPCA(df, components)
    results = {}
    for algorithm in listAlgorithms:
        results[algorithm] = algorithms[algorithm](df, listAlgorithms[algorithm], verbose = verbose)
        
    return results