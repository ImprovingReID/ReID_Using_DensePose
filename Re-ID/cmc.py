"""Calculates the cmc curve of a dataset consisting of images of N different people.
"""
import random

import numpy as np


def cmc(
    D: np.ndarray,
    G: np.ndarray,
    P: np.ndarray,
    repetitions=100,
) -> tuple((np.ndarray, np.ndarray)):
    """Overhead function for calculating the cmc.
    The gallery set G is made to only include unique ids.
    The process is repeated to get an average."""
    gtypes = np.array(list(set(G)))
    ngtypes = len(gtypes)
    C = np.zeros(shape=(ngtypes, repetitions))
    for t in range(repetitions):
        subdist = np.zeros(shape=(ngtypes, D.shape[1]))
        for i in range(ngtypes):
            j = np.nonzero(G == gtypes[i])[0]
            k = random.choice(j)
            subdist[i, :] = D[k, :]
        C[:, t], matches = cmc_core(subdist, gtypes, P)
    C = np.mean(C, axis=1)
    return C, matches


def cmc_core(
    D: np.ndarray, G: np.ndarray, P: np.ndarray
) -> tuple((np.ndarray, np.ndarray)):
    n, m = D.shape
    order = np.argsort(D, axis=0)
    matches = G[order] == np.tile(P, (n, 1))
    C = np.cumsum(np.sum(matches, 1) / m)
    return C, matches
