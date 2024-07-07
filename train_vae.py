import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import clone_model, load_model

from model.vae import VAE
from model.cae import CAE
from data.mnist import mnist_data
from data.utils import sample_and_categorize
from model.utils import clone_encoder


def pretrain_vae(x_train, x_train_samples):
    # x_train and x_train_samples are reshaped to 1-d vectors

    model_pre = VAE()

    dummy_eps_input = np.zeros((len(x_train), model_pre.latent_dim))

    # Fit the model. Note that the 'eps' input is ignored because it is an Input tensor.
    history = model_pre.model.fit([x_train, dummy_eps_input], x_train, shuffle=True, epochs=50, batch_size=100)
    # plot_learning_curve(history)

    classifier = model_pre.classifier()
    encodings_pretrain = model_pre.encoder.predict(x_train_samples)
    image_encoder = clone_encoder(model_pre.image_encoder())

    return classifier, encodings_pretrain, image_encoder


def clone_encoder(encoder):
    cloned_encoder = clone_model(encoder)
    cloned_encoder.set_weights(encoder.get_weights())
    cloned_encoder.trainable = False
    return cloned_encoder


def plot_learning_curve(history):
    plt.plot(history.history['loss'])
    plt.title('Model loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend(['Train'], loc='upper right')
    plt.savefig('trained_models/learning_curve.pdf')
    plt.show()


def plot_denoiser_examples(denoiser, samples, n=10):
    plt.figure(figsize=(20, 4))

    # Generate noisy samples
    noise_factor = 0.5  # You can adjust this value
    samples_noisy = samples + noise_factor * np.random.normal(loc=0.0, scale=1.0, size=samples.shape)
    samples_noisy = np.clip(samples_noisy, 0., 1.)

    # Denoise using the autoencoder
    denoised_samples = denoiser.predict(samples)

    for i in range(n):
        # Display original
        ax = plt.subplot(3, n, i + 1)
        plt.imshow(samples[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display noisy version
        ax = plt.subplot(3, n, i + 1 + n)
        plt.imshow(samples_noisy[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

        # Display reconstruction
        ax = plt.subplot(3, n, i + 1 + 2 * n)
        plt.imshow(denoised_samples[i].reshape(28, 28))
        plt.gray()
        ax.get_xaxis().set_visible(False)
        ax.get_yaxis().set_visible(False)

    plt.show()


def plot_encodings2d_with_labels(encodings, labels):
    plt.figure(figsize=(8, 8))

    unique_labels = np.unique(labels)  # Identify unique class labels
    colormap = plt.cm.jet  # or another suitable colormap like 'viridis', 'plasma', etc.

    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)
        color = colormap(i / len(unique_labels))  # Choose color from the colormap
        plt.scatter(encodings[indices, 0], encodings[indices, 1], color=color, label=str(label))

    plt.legend()
    plt.title('2D t-SNE of Encodings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()


if __name__ == "__main__":
    # 1. data
    # load data
    train_images, train_labels, test_images, test_labels = mnist_data()
    # samples, sample_labels = sample_and_categorize(train_images, train_labels, number=3000)
    samples, sample_labels = sample_and_categorize(train_images, train_labels)
    samples_test, sample_labels_test = sample_and_categorize(test_images, test_labels)
    print(samples.shape)
    print(sample_labels.shape)

    # reshape data
    x_train = np.reshape(samples, (-1, 784))
    # x_train = np.reshape(train_images, (-1, 784))
    print(x_train.shape)

    # 2. model: train a VAE
    model_pre = VAE(latent_dim=12, name="vae2")

    # Fit the model. Note that the 'eps' input is ignored because it is an Input tensor.
    dummy_eps_input = np.zeros((len(x_train), model_pre.latent_dim))
    history = model_pre.model.fit([x_train, dummy_eps_input], x_train, shuffle=True, epochs=50, batch_size=100)
    plot_learning_curve(history)

    # show example
    x_test = np.reshape(samples_test, (-1, 784))
    for i in range(10):
        embed = model_pre.encoder(x_test[i])
        img = model_pre.decoder(embed)
        plt.imshow(np.reshape(x_test[i], (28, 28)))
        plt.imshow(np.reshape(img, (28, 28)))
        print("-"*64)

    model_pre.save()
