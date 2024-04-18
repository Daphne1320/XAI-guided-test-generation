import tensorflow as tf
import numpy as np
from sklearn.manifold import TSNE

from model.cvae import CVAE
from utils.plot import plot_encodings2d_with_labels
from utils.utils import get_mnist_data, timer, preprocess_images


@timer
def pretrain_cvae():
    # get data
    train_images, train_labels, test_images, test_labels = get_mnist_data(True, True)

    # data reshape
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                     .shuffle(60000).batch(batch_size=128))
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                    .shuffle(10000).batch(batch_size=128))

    model_pre = CVAE()

    # %%time
    history = model_pre.fit(train_dataset, test_dataset, epochs=1)

    # save
    model_pre.save("./trained_model/CVAE")

    encodings_pretrain = model_pre.encoder.predict([train_images[:1000, :], train_labels[:1000, :]])

    # plot
    tsne = TSNE(n_components=2, random_state=42)
    encodings_samples_2d = tsne.fit_transform(encodings_pretrain)
    plot_encodings2d_with_labels(encodings_samples_2d, np.argmax(train_labels[:1000, :], axis=-1))

    return encodings_pretrain, model_pre.encoder


if __name__ == "__main__":

    # save model to local & import next time
    encodings_samples, encoder = pretrain_cvae()

