#!/usr/bin/env python3
"""
Module containing the Agglomerative function for hierarchical clustering.
"""
import scipy.cluster.hierarchy as sch
from sklearn import cluster


def Agglomerative(X, n_clusters):
    """
    Performs agglomerative hierarchical clustering on tabular data and
    plots the corresponding dendrogram.

    Parameters:
    - X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
    - n_clusters (int): Number of clusters to find

    Returns:
    - numpy.ndarray: Cluster labels for each data point of shape (n_samples,)
    """
    # Create the link matrix using Ward's linkage method
    linkage_matrix = sch.linkage(X, method='ward')

    # Plot the dendrogram using scipy
    sch.dendrogram(linkage_matrix)

    # Fit the agglomerative clustering model
    agg_model = cluster.AgglomerativeClustering(
        n_clusters=n_clusters,
        linkage='ward'
    )
    labels = agg_model.fit_predict(X)

    return labels
