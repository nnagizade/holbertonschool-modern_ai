#!/usr/bin/env python3
"""
Module containing the Standardize function for tabular data preprocessing.
"""
from sklearn import preprocessing


def Standardize(X):
    """
    Standardizes tabular data using Scikit-learn's StandardScaler.

    Parameters:
    - X (numpy.ndarray): Tabular data of shape (n_samples, n_features)

    Returns:
    - numpy.ndarray: The standardized version of the input data,
                     with the same shape as X.
    """
    scaler = preprocessing.StandardScaler()
    return scaler.fit_transform(X)
