import numpy as np
import tensorflow as tf

from sklearn.manifold import TSNE

from CVAE import CVAE
from plot import plot_encodings2d_with_labels
from utils import get_mnist_data, sample_and_categorize, timer, preprocess_images


@timer
def pretrain_cvae():
    # get data
    train_images, train_labels, test_images, test_labels = get_mnist_data(True, True)
    train_labels = tf.one_hot(train_labels, depth=10)
    test_labels = tf.one_hot(test_labels, depth=10)

    # data reshape
    train_images = preprocess_images(train_images)
    test_images = preprocess_images(test_images)

    train_dataset = (tf.data.Dataset.from_tensor_slices((train_images, train_labels))
                     .shuffle(60000).batch(batch_size=128))
    test_dataset = (tf.data.Dataset.from_tensor_slices((test_images, test_labels))
                    .shuffle(10000).batch(batch_size=128))

    model_pre = CVAE()

    # %%time
    history = model_pre.model.fit(train_dataset, test_dataset, epochs=10)

    encodings_pretrain = model_pre.encoder.predict(train_dataset)
    image_encoder = model_pre.clone_encoder()

    return encodings_pretrain, image_encoder


if __name__ == "__main__":

    # save model to local & import next time
    encodings_samples, encoder = pretrain_cvae()

    # plot
    #tsne = TSNE(n_components=2, random_state=42)
    #encodings_samples_2d = tsne.fit_transform(encodings_samples)

