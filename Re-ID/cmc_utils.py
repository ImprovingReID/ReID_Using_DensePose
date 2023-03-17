"""Runs CMC evaluation on a dataset consisting of
a set of objects. For each object there are mutiple
images from different view angles
"""

import itertools
import pathlib
import random
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np

from evaluation.cmc import cmc


def split_data(
    data: defaultdict, split_factor=0.5
) -> tuple((defaultdict(list), defaultdict(list))):
    split_indices = defaultdict(tuple)
    for ID, images in data.items():
        index_range = range(len(images))
        split1 = random.sample(index_range, int(split_factor * len(images)))
        split2 = [i for i in index_range if i not in split1]
        split_indices[ID] = (split1, split2)

    return split_indices


def split_query_target(
    data: defaultdict, split_indices: defaultdict
) -> tuple((defaultdict(list), defaultdict(list))):
    query = defaultdict(list)
    target = defaultdict(list)
    for ID, images in data.items():
        query_indices, target_indices = split_indices[ID]
        query[ID] = [images[i] for i in query_indices]
        target[ID] = [images[i] for i in target_indices]

    return query, target


def create_D(
    query: defaultdict(list),
    target: defaultdict(list),
    distance_function: defaultdict(list),
) -> tuple((np.ndarray, np.ndarray, np.ndarray)):
    """Created the distance matrix consisting of all samples in the
    query set compared to all samples in the target set.
    """
    n = sum([len(images) for images in query.values()])
    m = sum([len(images) for images in target.values()])

    G = {ID: [e for e in embeddings] for ID, embeddings in query.items()}
    P = {ID: [e for e in embeddings] for ID, embeddings in target.items()}

    G_keys = [ID for ID, feat in G.items() for _ in feat]
    P_keys = [ID for ID, feat in P.items() for _ in feat]

    G_values = [feat for feature_list in G.values() for feat in feature_list]
    P_values = [feat for feature_list in P.values() for feat in feature_list]

    D = np.empty(shape=(n * m))
    for idx, (feat1, feat2) in enumerate(itertools.product(G_values, P_values)):
        D[idx] = distance_function(feat1, feat2)

    D = D.reshape(n, m)
    return D, np.array(G_keys), np.array(P_keys)


def plot_cmc(C: np.ndarray) -> None:
    x = np.arange(len(C)) + 1
    fig, ax = plt.subplots(figsize=(6, 6))
    ax.plot(x, C, linewidth=2.0, marker="s")
    ax.set_xticks(x)
    ax.set_xlabel("rank")
    ax.set_ylabel("accuracy")
    ax.set_title("CMC curve")
    plt.show()


def run_cmc(data, distance_function):
    random.seed(100)
    split_indices = split_data(data)
    query, target = split_query_target(data, split_indices)
    D, G_keys, P_keys = create_D(query, target, distance_function)
    return cmc(D, G_keys, P_keys)


def run_eval(model, input_path: pathlib.Path):
    import torch
    import torchvision.transforms as transforms
    from PIL import Image
    from scipy.spatial import distance

    trans = transforms.Compose(
        [
            transforms.Resize((96, 96)),  # should be model input size
            transforms.ToTensor(),
            transforms.Normalize((0.486, 0.459, 0.408), (0.229, 0.224, 0.225)),
        ]
    )
    cmc_embeddings = defaultdict(list)
    model.eval()
    with torch.no_grad():
        for image_path in input_path.glob("*.jpg"):
            img = Image.open(image_path)
            x = trans(img).cuda()
            fv = model(x[None, :])
            fv = fv.cpu().numpy().flatten()

            pid = int(image_path.stem.split("_")[0])
            cmc_embeddings[pid].append(fv)

    c, _ = run_cmc(cmc_embeddings, distance.euclidean)
    return c


def hist3d(im, bin_count=64):
    img = im.reshape(-1, 3)
    stride = 256 / bin_count
    edges = [i * stride for i in range(bin_count)] + [255.0]
    edges3d = [edges, edges, edges]

    bins, _ = np.histogramdd(img, bins=edges3d)
    return bins.flatten()
