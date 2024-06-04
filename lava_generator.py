from tensorflow.keras.models import load_model
from tqdm.notebook import tqdm
from tensorflow.keras import backend as K

from compare import get_h_lava_via_one_step, kl_divergence, ws_distance, plot_image_comparison, load_samples_for_test
from contrib.DLFuzz.dlfuzz import DLFuzz
from model.XAI_classifier import xai_model
from model.vae import VAE
from xai import gradient_of_x

import numpy as np
import tensorflow as tf

def generate_adversarial_with_gradient_descent(h_view, label, alpha, num_iterations):
    # h_view: latent space
    # load models
    vae = VAE.load("trained_models/VAE")
    cnn = load_model("trained_models/CNN/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))

    # prepare data
    x = np.array([h_view])
    y = np.array([label])

    # Adversarial example generation
    for iteration in range(num_iterations):
        gradient = gradient_of_x(x, y, xai)  # Compute gradient
        x = x + alpha * gradient  # Update the encoded image with the gradient
        decoded_image = vae.decoder.predict(x).reshape((28, 28, 1))  # Decode the perturbed image

        # Check if the decoded image is misclassified by the CNN
        prediction = cnn.predict(np.array([decoded_image]))[0]
        predicted_label = np.argmax(prediction)

        if predicted_label != np.argmax(y):
            print(f"Misclassification achieved at iteration {iteration}")
            break
    return [decoded_image, predicted_label]


def predict_label(cnn, image):
    return np.argmax(cnn.predict(np.array([image]))[0])


def generate_adversarial_lava(h, y_onehot, vae, xai, var=0.5):
    # calculate latent variant image
    x = np.array([h])
    y = np.array([y_onehot])
    g = gradient_of_x(x, y, xai)
    g_npy = np.squeeze(g.numpy())
    h_lava = get_h_lava_via_one_step(x, g_npy, step=var)
    print(h_lava.shape)

    image_gen = vae.decoder.predict(h_lava)[0].reshape((28, 28, 1))
    return image_gen, h_lava


def generate_adversarial_lava_step_by_step(h, y_onehot, vae, xai, cnn, alpha=0.01, num_iterations=100):
    def refiner_gradient_descent(h, alpha, label_true):
        for iteration in range(num_iterations):
            h_lava, image_gen = generate_adversarial_lava(h, y_onehot, vae, xai, var=alpha)
            if predict_label(cnn, image_gen) != label_true:
                return h_lava, image_gen, iteration
            h = h_lava
        return None, None, None

    label_true = np.argmax(y_onehot)
    for iteration in range(num_iterations):
        h_lava, image_gen = generate_adversarial_lava(h, y_onehot, vae, xai, var=alpha)

        if predict_label(cnn, image_gen) != label_true:
            inner_alpha = alpha / num_iterations
            h_lava, image_gen, inner_iteration = refiner_gradient_descent(h, inner_alpha, label_true)
            if h_lava is not None:
                print(f"Misclassification achieved at iteration {iteration}, inner iteration {inner_iteration}")
                return h_lava, image_gen

        h = h_lava

    return image_gen, h_lava


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
    h_view = vae.encoder.predict(x_view)

    K.set_learning_phase(0)
    dlfuzz = DLFuzz(cnn)

    # start
    for i in tqdm(range(len(x_view))):

        # calculate fuzz image
        image_org = samples_view[i]
        label_org = sample_labels_view[i]

        # generate dlfuzz image
        image_gen_fuzz = dlfuzz.generate_fuzzy_image(image_org)  # of shape (28, 28, 1)

        # generate latent variant image
        image_gen_lava, h_lava = generate_adversarial_lava(h_view[i], y_onehot_view[i], vae, xai)

        label_fuzz = np.argmax(cnn.predict(np.array([image_gen_fuzz]))[0])
        label_lava = np.argmax(cnn.predict(np.array([image_gen_lava]))[0])

        # List of images and their titles
        images = [image_org, image_gen_fuzz, image_gen_lava]
        titles = [f'image_org_{label_org}', f'image_gen_fuzz_{label_fuzz}', f'image_gen_lava_{label_lava}']

        # Plot the images
        if label_lava != label_org:
            plot_image_comparison(images, titles)

            # in latent space
            h = np.array[h_view[i]]
            h_fuzz = vae.encoder.predict(np.array([image_gen_fuzz.reshape((784,))]))[0]

            d_fuzz = np.linalg.norm(h - h_fuzz)
            d_lava = np.linalg.norm(h - h_lava)
            print(f"d_fuzz: {d_fuzz}\nd_lava: {d_lava}")

            # in image space
            kl_fuzz = kl_divergence(image_org, image_gen_fuzz)
            kl_lava = kl_divergence(image_org, image_gen_lava)
            print(f"kl_fuzz: {kl_fuzz}\nkl_lava: {kl_lava}")

            ws_fuzz = ws_distance(image_org, image_gen_fuzz)
            ws_lava = ws_distance(image_org, image_gen_lava)
            print(f"ws_fuzz: {ws_fuzz}\nws_lava: {ws_lava}")