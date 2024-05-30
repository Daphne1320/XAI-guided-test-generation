# -*- coding: utf-8 -*-

from __future__ import print_function

from PIL import Image
import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from scipy.stats import entropy
from scipy.stats import wasserstein_distance

from tensorflow.keras.models import clone_model, load_model
import tensorflow as tf
from tensorflow.keras import backend as K
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import InputLayer

from data.mnist import mnist_data
from data.utils import sample_and_categorize
from model.XAI_classifier import xai_model
from model.vae import VAE

from xai import gradient_of_x

from contrib.DLFuzz.dlfuzz import DLFuzz
from contrib.DLFuzz.utils import clear_up_dir, load_image, deprocess_image, get_signature


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


def normalize_as_probability_distribution(arr):
    """
    Normalize a numpy array so that its sum is 1.
    """
    max_, min_ = max(arr), min(arr)
    arr = arr - min_ / max_ - min_
    arr /= arr.sum()
    return arr


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
    _img1 = normalize_as_probability_distribution(img1.flatten())
    _img2 = normalize_as_probability_distribution(img2.flatten())
    img1 = _img1[_img2 != 0]
    img2 = _img2[_img2 != 0]
    kl1 = entropy(img1, img2)
    img1 = _img1[_img1 != 0]
    img2 = _img2[_img1 != 0]
    kl2 = entropy(img2, img1)
    return (kl1 + kl2) / 2


def ws_distance(img1, img2):
    # Smaller values indicate that the distributions are more similar.
    img1 = normalize_as_probability_distribution(img1.flatten())
    img2 = normalize_as_probability_distribution(img2.flatten())
    return wasserstein_distance(img1, img2)


if __name__ == "__main__":

    vae = VAE.load("trained_models")
    cnn = load_model("trained_models/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))

    # input images
    # samples_test, sample_labels_test = load_samples_for_test_folder(img_dir='./contrib/DLFuzz/MNIST/seeds_50')
    samples_test, sample_labels_test = load_samples_for_test(200)

    # prepare
    view_samples = samples_test
    view_sample_labels = sample_labels_test
    # view_samples = samples
    # view_sample_labels = sample_labels

    x_view = np.reshape(view_samples, (-1, 784))
    y_view_onehot = tf.one_hot(tf.constant(view_sample_labels), depth=10).numpy()
    h_view = vae.encoder.predict(x_view)

    K.set_learning_phase(0)
    dlfuzz = DLFuzz(cnn)

    # start
    for i in tqdm(range(len(x_view))):

        # calculate fuzz image
        image_org = np.array([view_samples[i]], dtype="float32")
        label_org = view_sample_labels[i]
        image_gen_fuzz = dlfuzz.generate_fuzzy_image(image_org)
        # of shape (28, 28, 1)

        # calculate latent variant image
        x = np.array([h_view[i]])
        y = np.array([y_view_onehot[i]])
        g = gradient_of_x(x, y, xai)
        g_npy = np.squeeze(g.numpy())

        var = 0.5
        h_lava = get_h_lava_via_one_step(x, g_npy, step=var)
        image_gen_lava = vae.decoder.predict(h_lava)[0].reshape((28, 28, 1))

        label_fuzz = np.argmax(cnn.predict(np.array([image_gen_fuzz]))[0])
        label_lava = np.argmax(cnn.predict(np.array([image_gen_lava]))[0])

        # List of images and their titles
        images = [image_org, image_gen_fuzz, image_gen_lava]
        titles = [f'image_org_{label_org}', f'image_gen_fuzz_{label_fuzz}', f'image_gen_lava_{label_lava}']

        # Plot the images
        if label_lava != label_org:
            plot_image_comparison(images, titles)

            # in latent space
            h_fuzz = vae.encoder.predict(np.array([image_gen_fuzz.reshape((784,))]))[0]
            d_fuzz = np.linalg.norm(x - h_fuzz)
            d_lava = np.linalg.norm(x - h_lava)
            print(f"d_fuzz: {d_fuzz}\nd_lava: {d_lava}")

            # in image space
            kl_fuzz = kl_divergence(image_org, image_gen_fuzz)
            kl_lava = kl_divergence(image_org, image_gen_lava)
            print(f"kl_fuzz: {kl_fuzz}\nkl_lava: {kl_lava}")

            ws_fuzz = ws_distance(image_org, image_gen_fuzz)
            ws_lava = ws_distance(image_org, image_gen_lava)
            print(f"ws_fuzz: {ws_fuzz}\nws_lava: {ws_lava}")