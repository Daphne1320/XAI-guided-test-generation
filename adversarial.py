# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from tqdm import tqdm

import numpy as np
import matplotlib.pyplot as plt

from tensorflow.keras.models import clone_model, load_model
import tensorflow as tf
from tensorflow.keras import backend as K

from data.mnist import mnist_data
from data.utils import sample_and_categorize
from model.XAI_classifier import xai_model
from model.vae import VAE

from xai import gradient_of_x
from discrepancy_measure import *

from contrib.DLFuzz.dlfuzz import DLFuzz
from contrib.DLFuzz.utils import load_image
from contrib.Cleverhans.cleverhans import CleverHans


def predict_label(cnn, image):
    return np.argmax(cnn.predict(np.array([image]))[0])


def load_samples_for_test(number):
    train_images, train_labels, test_images, test_labels = mnist_data()
    return sample_and_categorize(test_images, test_labels, number=number)


def load_samples_for_test_from_folder(img_dir):
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


def generate_adversarial_lava(h, y_onehot, vae, xai, var=0.5):
    # calculate latent variant image
    x = np.array([h])
    y = np.array([y_onehot])
    g = gradient_of_x(x, y, xai)
    g_npy = np.squeeze(g.numpy())
    h_lava = get_h_lava_via_one_step(x, g_npy, step=var)
    print(h_lava.shape)

    image_adv = vae.decoder.predict(h_lava)[0].reshape((28, 28, 1))
    return image_adv, h_lava


def generate_adversial_lava_step_by_step(h, y_onehot, vae, xai, cnn, alpha=0.01, num_iterations=100):
    def refiner_gradient_descent(h, alpha, label_true):
        for iteration in range(num_iterations):
            h_lava, image_adv = generate_adversarial_lava(h, y_onehot, vae, xai, var=alpha)
            if predict_label(cnn, image_adv) != label_true:
                return h_lava, image_adv, iteration
            h = h_lava
        return None, None, None

    label_true = np.argmax(y_onehot)
    for iteration in range(num_iterations):
        h_lava, image_adv = generate_adversarial_lava(h, y_onehot, vae, xai, var=alpha)

        if predict_label(cnn, image_adv) != label_true:
            inner_alpha = alpha / num_iterations
            h_lava, image_adv, inner_iteration = refiner_gradient_descent(h, inner_alpha, label_true)
            if h_lava is not None:
                print(f"Misclassification achieved at iteration {iteration}, inner iteration {inner_iteration}")
                return h_lava, image_adv

        h = h_lava

    return image_adv, h_lava


def plot_image_comparison_two(images, titles, cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    vae = VAE.load("trained_models")
    cnn = load_model("trained_models/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))

    # input images
    # samples_test, sample_labels_test = load_samples_for_test_folder(img_dir='./contrib/DLFuzz/MNIST/seeds_50')
    samples_test, sample_labels_test = load_samples_for_test(200)

    # prepare
    samples_view = samples_test
    sample_labels_view = sample_labels_test
    # samples_view = samples
    # sample_labels_view = sample_labels

    x_view = np.reshape(samples_view, (-1, 784))
    y_onehot_view = tf.one_hot(tf.constant(sample_labels_view), depth=10).numpy()

    K.set_learning_phase(0)
    # dlfuzz = DLFuzz(cnn)
    clhans = CleverHans(cnn)

    # start exp
    for i in tqdm(range(len(x_view))):
        # get original image
        image_org = samples_view[i]
        label_org = sample_labels_view[i]

        # generate dlfuzz image
        # image_adv = dlfuzz.generate_adversarial_image(image_org)  # of shape (28, 28, 1)
        image_adv = clhans.generate_adversarial_image(image_org)  # of shape (28, 28, 1)
        label = np.argmax(cnn.predict(np.array([image_adv]))[0])

        # List of images and their titles
        images = [image_org, image_adv]
        titles = [f'image_org_{label_org}', f'image_adv_{label}']

        # Plot the images
        if label != label_org:
            plot_image_comparison_two(images, titles)

            kl_fuzz = kl_divergence(image_org, image_adv)
            print(f"kl: {kl_fuzz}")

            ws_fuzz = ws_distance(image_org, image_adv)
            print(f"ws: {ws_fuzz}")