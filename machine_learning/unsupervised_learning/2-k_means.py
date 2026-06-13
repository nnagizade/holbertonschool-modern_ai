#!/usr/bin/env python3
"""
Module containing the K_Means function for clustering tabular data.
"""
from sklearn import cluster


def K_Means(X, n_clusters, random_state):
    """
    Creates and fits a K-Means clustering model using Scikit-learn.

    Parameters:
    - X (numpy.ndarray): Tabular data of shape (n_samples, n_features)
    - n_clusters (int): Number of clusters to find
    - random_state (int): Random seed for reproducibility

    Returns:
    - sklearn.cluster.KMeans: The fitted KMeans instance.
    """
    kmeans = cluster.KMeans(n_clusters=n_clusters, random_state=random_state)
    kmeans.fit(X)
    return kmeans
