from tensorflow.keras.models import load_model

from compare import get_h_lava_via_one_step
from model.XAI_classifier import xai_model
from model.vae import VAE
from xai import gradient_of_x

import numpy as np

def generate_adversarial_with_gradient_descent(h_view, label, alpha, num_iterations):
    # h_view: latent space
    # load models
    vae = VAE.load("trained_models/VAE")
    cnn = load_model("trained_models/CNN/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(10,))

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
