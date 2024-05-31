from tensorflow.keras.models import load_model

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


def generate_adversarial_with_gradient_descent_improved(h_view, label, alpha, num_iterations):
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
        x_tmp = x + alpha * gradient  # Update the encoded image with the gradient
        decoded_image = vae.decoder.predict(x_tmp).reshape((28, 28, 1))  # Decode the perturbed image

        # Check if the decoded image is misclassified by the CNN
        prediction = cnn.predict(np.array([decoded_image]))[0]
        predicted_label = np.argmax(prediction)

        if predicted_label != np.argmax(y):
            # step back and try finer alpha
            inner_alpha = alpha / num_iterations
            for inner_iteration in range(num_iterations):
                gradient = gradient_of_x(x, y, xai)  # Compute gradient
                inner_x_tmp = x + inner_alpha * gradient  # Update the encoded image with the gradient
                decoded_image = vae.decoder.predict(inner_x_tmp).reshape((28, 28, 1))  # Decode the perturbed image

                # Check if the decoded image is misclassified by the CNN
                prediction = cnn.predict(np.array([decoded_image]))[0]
                predicted_label = np.argmax(prediction)

                if predicted_label != np.argmax(y):
                    print(f"Misclassification achieved at iteration {iteration},  inner iteration {inner_iteration}")
                    break
                x = inner_x_tmp

        if predicted_label != np.argmax(y):
            break
        x = x_tmp;
    return [decoded_image, predicted_label]
