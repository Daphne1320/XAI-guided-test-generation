# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import numpy as np
import matplotlib.pyplot as plt

from data.mnist import mnist_data
from data.utils import sample_and_categorize
from contrib.DLFuzz.utils import load_image


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
