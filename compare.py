# -*- coding: utf-8 -*-

from __future__ import print_function

import os

from scipy.special import rel_entr
from sklearn.metrics import mean_squared_error

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.stats import wasserstein_distance

import tensorflow as tf

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import InputLayer

from data.mnist import mnist_data
from data.utils import sample_and_categorize

from contrib.DLFuzz.utils import load_image


def flatten_model(nested_model):
    """
    Flatten a nested Keras model into a single sequential model.

    Parameters:
    nested_model (tf.keras.Model): The nested model to flatten.

    Returns:
    tf.keras.Sequential: A flattened sequential model.
    """
    flat_model = Sequential()

    # Add an input layer explicitly
    flat_model.add(InputLayer(input_shape=nested_model.input_shape[1:]))

    def add_layers(layers):
        for layer in layers:
            if isinstance(layer, tf.keras.Model):
                add_layers(layer.layers)  # Recursively add nested layers
            else:
                # Skip cloning the InputLayer, just add it directly
                if isinstance(layer, InputLayer):
                    flat_model.add(layer)
                else:
                    # Clone the layer configuration
                    cloned_layer = layer.__class__.from_config(layer.get_config())
                    # Build layer to initialize weights
                    cloned_layer.build(layer.input_shape)
                    # Copy weights
                    cloned_layer.set_weights(layer.get_weights())
                    # print(f"Adding layer: {layer.name}")
                    flat_model.add(cloned_layer)

    add_layers(nested_model.layers)

    return flat_model


def plot_image_comparison(images, titles, cmap='gray'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


def load_samples_for_test(number):
    train_images, train_labels, test_images, test_labels = mnist_data()
    return sample_and_categorize(test_images, test_labels, number=number)


def load_samples_for_test_folder(img_dir):
    img_paths = os.listdir(img_dir)
    samples_test = np.reshape([load_image(os.path.join(img_dir, img_path)) for img_path in img_paths], (-1, 28, 28, 1))
    sample_labels_test = [int(img_path.split("_")[-1][0]) for img_path in img_paths]
    return samples_test, sample_labels_test


def get_h_lava_via_one_grid_step(h, gradient, step=0.5):
    max_grad_index = np.argmax(np.abs(gradient))
    h_lava = np.copy(h)  # _lava means latent variants
    h_lava[0, max_grad_index] += step * np.sign(gradient[max_grad_index])
    return h_lava


def get_h_lava_via_one_step(h, gradient, step=0.5):
    h_lava = np.copy(h)  # _lava means latent variants
    h_lava += step * gradient
    return h_lava


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
    # Kullback-Leibler (KL)
    # If the KL divergence is small, it indicates that the distributions represented by the two images are similar.
    # If the KL divergence is large, it indicates that the distributions represented by the two images are different.
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    img1 = img1[img2 != 0]
    img2 = img2[img2 != 0]
    return entropy(img1, img2)


def js_divergence(img1, img2):
    # jessen divergence
    # Smaller JS Divergence is Better: Similar to KL divergence,
    # a smaller JS divergence indicates that the two distributions are more similar.
    return (kl_divergence(img1, img2) + kl_divergence(img2, img1)) / 2


def ws_distance(img1, img2):
    # distribution?? ele - ele
    # Smaller values indicate that the distributions are more similar.
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    return wasserstein_distance(img1, img2)