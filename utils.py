import tensorflow as tf
import numpy as np
import functools
import time

from keras.utils import to_categorical


def get_mnist_data(reshape=True, categorical=True):
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train, x_test = x_train / 255.0, x_test / 255.0

    if reshape:
        # Reshape data
        x_train = x_train.reshape((x_train.shape[0], 28, 28, 1))
        x_test = x_test.reshape((x_test.shape[0], 28, 28, 1))

    if categorical:
        # Convert class vectors to binary class matrices
        y_train = to_categorical(y_train, 10)
        y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


def sample_and_categorize(train_images, train_labels, number=3000):
    # sample data
    indices = np.random.choice(train_images.shape[0], size=number, replace=False)
    samples, sample_labels = train_images[indices], train_labels[indices]

    # y to labels (in category format)
    sample_labels = np.argmax(sample_labels, axis=-1)
    return samples, sample_labels


# model each pixel with a Bernoulli distribution
def preprocess_images(images):
    images = images.reshape((images.shape[0], 28, 28, 1)) / 255.
    return images.astype('float32')

def gradient_of_x(x, y, model):
    # Convert the numpy arrays to TensorFlow tensors
    input_data = tf.convert_to_tensor(x, dtype=tf.float32)
    true_labels = tf.convert_to_tensor(y, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_data)  # Explicitly watch the input tensor

        # Now directly feeding `input_data` to the model, so TensorFlow automatically tracks operations
        predictions = model(input_data, training=False)

        # Compute the categorical cross-entropy loss
        loss = tf.keras.losses.categorical_crossentropy(true_labels, predictions)

    # Compute the gradient of the loss with respect to the input
    return tape.gradient(loss, input_data)


def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        value = func(*args, **kwargs)
        end_time = time.perf_counter()
        run_time = end_time - start_time
        print("Finished {} in {} secs".format(repr(func.__name__), round(run_time, 3)))
        return value

    return wrapper
