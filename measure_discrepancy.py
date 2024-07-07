import numpy as np

from scipy.stats import entropy
from scipy.stats import wasserstein_distance
from scipy.special import rel_entr
from sklearn.metrics import mean_squared_error


def normalize_maxmin(arr):
    """
    Normalize a numpy array so that its sum is 1.
    """
    # assert not np.any(np.isinf(arr)), "Array contains inf values"
    # assert not np.any(np.isnan(arr)), "Array contains nan values"
    max_, min_ = max(arr), min(arr)
    arr = (arr - min_) / (max_ - min_) if max_ - min_ != 0 else arr - min_
    return arr


def normalize_as_probability_distribution(arr):
    """
    Normalize a numpy array so that its sum is 1.
    """
    # assert not np.any(np.isinf(arr)), "Array contains inf values"
    # assert not np.any(np.isnan(arr)), "Array contains nan values"
    max_, min_ = max(arr), min(arr)
    arr = (arr - min_) / (max_ - min_) if max_ - min_ != 0 else arr - min_
    arr /= arr.sum()
    return arr


def mse_loss(img1, img2):
    img1 = normalize_maxmin(img1.flatten())
    img2 = normalize_maxmin(img2.flatten())

    # Create a mask for non-nan values in both images
    mask = ~np.isnan(img1) & ~np.isnan(img2)

    # Filter out nan values using the mask
    img1 = img1[mask]
    img2 = img2[mask]

    return mean_squared_error(img1, img2)


def cross_entropy(img1, img2):
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    img1 = img1[img2 != 0]
    img2 = img2[img2 != 0]
    return -np.sum(img1 * np.log(img2))


def kl_divergence_org(img1, img2):
    "orginal kl_divergence"
    img1 = img1.flatten()
    img2 = img2.flatten()
    img1 = img1[img2 != 0]
    img2 = img2[img2 != 0]
    return entropy(img1, img2)


def kl_divergence(img1, img2):
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    img1 = img1[img2 != 0]
    img2 = img2[img2 != 0]
    return np.sum(rel_entr(img1, img2))


def js_divergence(img1, img2):
    # jessen divergence
    return (kl_divergence(img1, img2) + kl_divergence(img2, img1)) / 2


def ws_distance(img1, img2):
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    return wasserstein_distance(img1, img2)