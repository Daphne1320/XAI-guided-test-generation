import numpy as np
import matplotlib.pyplot as plt

from model.cae import CAE
from data.mnist import mnist_data
from data.utils import sample_and_categorize
from train_vae import plot_learning_curve

if __name__ == "__main__":
    # 1. data
    # load data
    train_images, train_labels, test_images, test_labels = mnist_data()
    samples, sample_labels = sample_and_categorize(train_images, train_labels, number=3000)
    # samples, sample_labels = sample_and_categorize(train_images, train_labels)
    samples_test, sample_labels_test = sample_and_categorize(test_images, test_labels)
    print(samples.shape)
    print(sample_labels.shape)

    # reshape data
    y_train = np.reshape(samples, (-1, 784))
    x_train = y_train + np.random.normal(0, 0.01, size=y_train.shape)
    print(x_train.shape)
    print(y_train.shape)

    # 2. model: train a CAE
    model_pre = CAE(latent_dim=12, name="cae2")

    # Fit the model. Note that the 'eps' input is ignored because it is an Input tensor.
    history = model_pre.model.fit(x_train, y_train, epochs=50, batch_size=100)
    plot_learning_curve(history)

    # show example
    x_test = np.reshape(samples_test, (-1, 784))
    for i in range(10):
        # Extract the current sample and add batch dimension
        sample = np.expand_dims(x_test[i], axis=0)

        embed = model_pre.encoder(sample)
        img = model_pre.decoder(embed)
        plt.imshow(np.reshape(x_test[i], (28, 28)))
        plt.imshow(np.reshape(img, (28, 28)))
        print("-"*64)

    model_pre.save()
