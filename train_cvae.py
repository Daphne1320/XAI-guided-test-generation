import numpy as np
import matplotlib.pyplot as plt
from tensorflow.data import Dataset
from tensorflow.keras.utils import to_categorical

from model.cvae import C_VAE
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
    x_test = np.reshape(samples_test, (-1, 784))
    x_train = np.reshape(samples, (-1, 784))
    x_train_label = to_categorical(sample_labels, 10)
    x_test_label = to_categorical(sample_labels_test, 10)
    print(x_train.shape)
    print(x_train_label.shape)

    # 2. model: train a CAE
    model_pre = C_VAE(latent_dim=12, model_name="cvae")

    # Fit the model. Note that the 'eps' input is ignored because it is an Input tensor.
    train_gen = (Dataset.from_tensor_slices((x_train, x_train_label)).shuffle(len(x_train)).batch(batch_size=100))
    test_gen = (Dataset.from_tensor_slices((x_test, x_test_label)).shuffle(len(x_test)).batch(batch_size=100))
    history = model_pre.fit(train_gen, test_gen, epochs=100)
    plot_learning_curve(history)
    model_pre.save()

    exit()
    # show example
    for i in range(10):
        embed, _ = model_pre.encode(x_test[i], x_test_label[i])
        img = model_pre.decode(embed, x_test_label, True)
        plt.imshow(np.reshape(x_test[i], (28, 28)))
        plt.imshow(np.reshape(img, (28, 28)))
        plt.show()
        print("-"*64)
