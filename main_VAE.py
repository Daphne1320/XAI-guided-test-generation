import numpy as np

from sklearn.manifold import TSNE

from VAE import VAE
from NoisyDataGenerator import NoisyDataGenerator
from plot import plot_encodings2d_with_labels
from utils import get_mnist_data, sample_and_categorize, timer


@timer
def pretrain_vae(x_train, x_train_samples):
    # x_train and x_train_samples are reshaped to 1-d vectors

    data_gen = NoisyDataGenerator(x_train, batch_size=128)
    model_pre = VAE()

    # %%time
    history = model_pre.model.fit(data_gen, epochs=10)
    # plot_learning_curve(history)
    # plot_denoiser_examples(model_pre.model, x_train_samples)

    classifier = model_pre.classifier()
    encodings_pretrain = model_pre.encoder.predict(x_train_samples)
    image_encoder = model_pre.clone_encoder()

    return classifier, encodings_pretrain, image_encoder


if __name__ == "__main__":
    # get data
    train_images, train_labels, test_images, test_labels = get_mnist_data(True, True)
    samples, sample_labels = sample_and_categorize(train_images, train_labels, number=3000)  # for run_with_visual

    # data reshape for model
    x_train = np.reshape(train_images, (-1, 784))
    x_train_samples = np.reshape(samples, (-1, 784))

    # save model to local & import next time
    clf, encodings_samples, encoder = pretrain_vae(x_train, x_train_samples)

    # plot
    tsne = TSNE(n_components=2, random_state=42)
    encodings_samples_2d = tsne.fit_transform(encodings_samples)
    plot_encodings2d_with_labels(encodings_samples_2d, sample_labels)
