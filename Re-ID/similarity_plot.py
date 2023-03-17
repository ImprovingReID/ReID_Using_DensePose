"""
Compare different variants of the VOD tracker.
"""

import argparse
from pathlib import Path
from itertools import combinations, product
import random

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
import pandas as pd
import seaborn as sns
from scipy.spatial import distance

from re_id.feature_extractors.color_hist_3d import ColorHist3D
from re_id.feature_extractors.fsnetq import FsNetQ
from common.utils import read_image


def cmc(D, g, p, repetitions=100):
    gtypes = np.array(list(set(g)))
    ngtypes = len(gtypes)

    C = np.zeros(shape=(ngtypes, repetitions))
    for t in range(repetitions):
        subdist = np.zeros(shape=(ngtypes, D.shape[1]))
        for i in range(ngtypes):
            j = np.nonzero(g == gtypes[i])[0]
            k = random.choice(j)
            subdist[i, :] = D[k, :]

        C[:, t] = cmc_core(subdist, gtypes, p)

    C = np.mean(C, axis=1)
    return C


def cmc_core(D, g, p):
    n, m = D.shape

    P = np.repeat(p.reshape((1, m)), n, axis=0)
    G = np.repeat(g.reshape((n, 1)), m, axis=1)

    order = np.argsort(D, axis=0)
    S = np.take_along_axis(G, order, axis=0)

    return np.cumsum((P == S).sum(1) / m)


def partition_samples(IDs):
    """"""
    edges = np.array(
        [next(i for i, ID in enumerate(IDs) if ID == j) for j in set(IDs)])
    edges = np.r_[edges, [len(IDs)]]

    gallery_indices = []
    query_indices = []

    for i in range(len(edges) - 1):
        start = edges[i]
        stop = edges[i + 1]
        mid = (start + stop) // 2

        gallery_indices.extend(list(range(start, mid)))
        query_indices.extend(list(range(mid, stop)))

    return (np.array(gallery_indices, dtype=int),
            np.array(query_indices, dtype=int),
            edges[:-1])


def analyze_cmc(path, extractor):
    """Auxiliary function."""
    def _extract_features(crop_path: Path):
        img = read_image(crop_path)
        return extractor.extract_features(img)

    features = [(ID, person.stem, _extract_features(crop))
                for ID, person in enumerate(path.iterdir())
                for crop in person.iterdir()]

    # features = [(ID, person.stem, row[5:])
    #             for ID, person in enumerate(path.iterdir())
    #             for row in np.loadtxt(person / 'features.txt', delimiter=',')]

    # pairwise cosine simliarities
    n = len(features)
    d = np.zeros((n, n))

    for i, j in combinations(range(n), 2):
        # f1 = features[i][2]
        # f2 = features[j][2]
        # dist = distance.euclidean(f1/np.linalg.norm(f1), f2/np.linalg.norm(f2))
        # sim = 1 - dist
        sim = 1 - distance.cosine(features[i][2], features[j][2])
        d[i, j] = sim
        d[j, i] = sim

    # construct D matrix
    IDs = np.array([ID for ID, *_ in features])
    gallery_indices, query_indices, start_indices = partition_samples(IDs)

    g = IDs[gallery_indices]
    p = IDs[query_indices]
    n, m = len(g), len(p)

    D = np.array([-d[j, k] for (j, k) in
                  product(gallery_indices, query_indices)]).reshape((n, m))
    cmc_score = cmc(D, g, p)

    return features, start_indices, cmc_score, d


def plot_simplified_heatmap(features, d, extractor_name):
    labels = [f"{f[0]}: {f[1].split('_')[0]}" for f in features]
    lut = dict(zip(sorted(set(labels)), sns.hls_palette(len(set(labels)))))
    row_colors = pd.DataFrame(labels)[0].map(lut)

    g = sns.clustermap(d, col_colors=[row_colors],
                       row_cluster=False, col_cluster=False)
    handles = [Patch(facecolor=lut[name]) for name in lut]
    plt.legend(handles, lut, title='ID', bbox_to_anchor=(1, 1),
               bbox_transform=plt.gcf().transFigure, loc='best')
    g.savefig(f'sim_{extractor_name}.png')


def plot_cmc_curve(cmc_scores, extractor_names):
    id_count = len(cmc_scores[0])
    x = np.arange(id_count) + 1
    fig1, ax1 = plt.subplots()
    for c in cmc_scores:
        ax1.plot(x, c, marker='s')

    ax1.legend(extractor_names)
    ax1.set_xlabel('rank')
    ax1.set_ylabel('Accuracy')
    ax1.set_title('CMC curve')
    fig1.savefig('cmc_curve.png')


def run(path: Path, background_path: Path):
    # add all features and along with their identifiers
    extractors = [ColorHist3D(), FsNetQ()]
    cmc_scores = []

    for extractor in extractors:
        # features, start_indices = _extract_features(path, extractor)
        # d = _compute_similarities(features)
        # cmc_acc = _compute_cmc_curve(features, d)

        features, start_indices, cmc_score, d = analyze_cmc(path, extractor)
        cmc_scores.append(cmc_score)
        print(cmc_score)

        plot_simplified_heatmap(features, d, extractor.name)

    plot_cmc_curve(cmc_scores, [extractor.name for extractor in extractors])


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "-d",
        "--detections-path",
        type=Path,
        help="Input .txt file (MOTChallenge format) containing detections.",
        required=True,
    )

    parser.add_argument(
        "-b",
        "--background-path",
        type=Path,
        help="Path to background image.",
        required=True,
    )

    args = parser.parse_args()
    # test_cmc()
    run(args.detections_path, args.background_path)


def test_cmc():
    D = np.array([
        [1, 1, 1, 1],
        [2, 2, 2, 2],
        [2, 2, 2, 2],
        [3, 3, 3, 3],
        [3, 3, 3, 3]
    ])

    g = np.array([1, 2, 2, 3, 3])
    p = np.array([1, 2, 3, 2])

    res = cmc(D, g, p)
    print(res)
