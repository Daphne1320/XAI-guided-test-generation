import numpy as np

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from sklearn.metrics import mean_squared_error


def normalize_maxmin(arr):
    """
    Normalize a numpy array so that its sum is 1.
    """
    max_, min_ = max(arr), min(arr)
    arr = arr - min_ / max_ - min_
    return arr


def normalize_as_probability_distribution(arr):
    """
    Normalize a numpy array so that its sum is 1.
    """
    max_, min_ = max(arr), min(arr)
    arr = arr - min_ / max_ - min_
    arr /= arr.sum()
    return arr


def cross_entropy(img1, img2):
    # element-wise relative entropy (KL divergence)
    img1 = normalize_maxmin(img1.flatten())
    img2 = normalize_maxmin(img2.flatten())
    img1 = img1[img2 != 0]
    img2 = img2[img2 != 0]
    return np.sum(rel_entr(img1, img2))


def mse_loss(img1, img2):
    img1 = normalize_maxmin(img1.flatten())
    img2 = normalize_maxmin(img2.flatten())
    return mean_squared_error(img1, img2)


def kl_divergence(img1, img2):
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    img1 = img1[img2 != 0]
    img2 = img2[img2 != 0]
    return entropy(img1, img2)


def js_divergence(img1, img2):
    # jessen divergence
    return (kl_divergence(img1, img2) + kl_divergence(img2, img1)) / 2


def ws_distance(img1, img2):
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    return wasserstein_distance(img1, img2)