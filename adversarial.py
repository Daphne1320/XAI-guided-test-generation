# -*- coding: utf-8 -*-

from __future__ import print_function

import os
from tqdm.notebook import tqdm

import matplotlib.pyplot as plt

from tensorflow.keras.models import load_model
import tensorflow as tf
from tensorflow.keras import backend as K

from adversarial_methods.fgsm import FGSM
from data.mnist import mnist_data
from data.utils import sample_and_categorize
from model.XAI_classifier import xai_model
from model.vae import VAE

from discrepancy_measure import *


def predict_label(cnn, image):
    return np.argmax(cnn.predict(np.array([image]))[0])


def load_samples_for_test(number, return_indices=False):
    train_images, train_labels, test_images, test_labels = mnist_data()
    return sample_and_categorize(test_images, test_labels, number=number, return_indices=return_indices)


def load_samples_for_test_from_folder(img_dir):
    from contrib.DLFuzz.utils import load_image

    img_paths = os.listdir(img_dir)
    samples_test = np.reshape([load_image(os.path.join(img_dir, img_path)) for img_path in img_paths], (-1, 28, 28, 1))
    sample_labels_test = [int(img_path.split("_")[-1][0]) for img_path in img_paths]
    return samples_test, sample_labels_test


def plot_image_comparison_two(images, titles, cmap='gray'):
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":

    vae = VAE.load("trained_models/VAE")
    cnn = load_model("trained_models/CNN/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))

    # input shape: (28, 28, 1)

    # input images
    fgsm = FGSM(cnn)
    # dlfuzz = DLFuzz(cnn)
    # clhans = CleverHans(cnn)

    # vae = VAE.load("trained_models")
    # lava = Lava(cnn, vae.encoder, vae.decoder)

    # input images
    samples, sample_labels = load_samples_for_test(200)

    # x = np.reshape(samples, (-1, 784))
    y_onehot = tf.one_hot(tf.constant(sample_labels), depth=10).numpy()

    K.set_learning_phase(0)

    # start exp
    for i in tqdm(range(len(samples))):
        # get original image
        image_org, label_org = samples[i], sample_labels[i]

        image_adv = fgsm.generate_adversarial_image(image_org, y_onehot[i])
        # image_adv = dlfuzz.generate_adversarial_image(image_org)  # of shape (28, 28, 1)
        # image_adv = clhans.generate_adversarial_image(image_org)  # of shape (28, 28, 1)
        # image_adv = lava.generate_adversarial_image(image_org, y_onehot[i])

        label_adv = np.argmax(cnn.predict(np.array([image_adv]))[0])

        # List of images and their titles
        images = [image_org, image_adv]
        titles = [f'image_org_{label_org}', f'image_adv_{label_adv}']

        # Plot the images
        if label_adv != label_org:
            plot_image_comparison_two(images, titles)

            print(f"kl: {kl_divergence(image_org, image_adv)}")
            # print(f"ws: {ws_divergence(image_org, image_adv)}")
            # print(f"js: {js_divergence(image_org, image_adv)}")