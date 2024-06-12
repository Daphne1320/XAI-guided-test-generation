# -*- coding: utf-8 -*-

import pandas as pd
import matplotlib as plt
from adversarial import *
from adversarial_methods.latent_variation import Lava, LavaMultiSteps
from contrib.DLFuzz.dlfuzz import DLFuzz


def plot_image_comparison(images, titles, cmap='gray'):
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    for ax, img, title in zip(axes, images, titles):
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.set_title(title)
        ax.axis('off')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    cnn = load_model("trained_models/CNN/classifier.h5")
    # input shape: (28, 28, 1)

    fgsm = FGSM(cnn)
    dfuz = DLFuzz(cnn)

    vae = VAE.load("trained_models/VAE")
    lava = Lava(cnn, vae.encoder, vae.decoder)
    lavam = LavaMultiSteps(cnn, vae.encoder, vae.decoder, verbose=True)

    # input images
    samples, sample_labels, sample_indices = load_samples_for_test(200, return_indices=True)

    y_onehot = tf.one_hot(tf.constant(sample_labels), depth=10).numpy()

    K.set_learning_phase(0)

    results = []
    # Start experiment
    for i in tqdm(range(len(samples))):
        # Get original image and label
        image_org, label_org = samples[i], sample_labels[i]

        # Generate adversarial images of shape (28, 28, 1)
        adversarial_images = {
            "fgsm": fgsm.generate_adversarial_image(image_org, y_onehot[i]),
            "dlfuzz": dfuz.generate_adversarial_image(image_org),
            "lava": lava.generate_adversarial_image(image_org, y_onehot[i])[0],
            "lavam": lavam.generate_adversarial_image(image_org, y_onehot[i], alpha=0.05, num_iterations=10)[0],
        }
        # adversarial_images["lava"], h_lava = lava.generate_adversarial_image(image_org, y_onehot[i])

        # Predict labels for adversarial images
        predicted_labels = {
            key: np.argmax(cnn.predict(np.array([img]))[0])
            for key, img in adversarial_images.items()
        }

        # Plot the images if original label differs from all adversarial labels
        if label_org not in predicted_labels.values():
            images = [image_org] + list(adversarial_images.values())
            labels = [label_org] + list(predicted_labels.values())
            markers = ["org"] + list(adversarial_images.keys())
            plot_image_comparison(images, labels, markers)

            # In image space
            divergence_functions = {
                "kl": kl_divergence,
                "ws": ws_distance,
                "js": js_divergence,
                # "mse": mse_loss,
                # "xentropy", cross_entropy,
            }
            result = dict()
            result["index"] = i
            result["index_in_dataset"] = sample_indices[i]
            for name, func in divergence_functions.items():
                divergences = {key: func(image_org, img) for key, img in adversarial_images.items()}
                for key, value in divergences.items():
                    result[f"{name}_{key}"] = value
                    print(f"{name}_{key}: {value}")
                print("-"*64)

            results.append(result)

            """
            # In latent space
            h = vae.encoder.predict([np.reshape([image_org], (784, ))])[0]
            latent_h = {
                key: vae.encoder.predict([np.reshape([img], (784, ))])[0]
                for key, img in adversarial_images.items()
            }
            latent_h["lava"] = h_lava  # Use h_lava directly for lava

            distances = {
                key: np.linalg.norm(h - latent_h[key])
                for key in latent_h.keys()
            }

            for key, dist in distances.items():
                print(f"d_{key}: {dist}")
            """

    results_df = pd.DataFrame(results)

    # Save DataFrame to a CSV file
    results_df.to_csv('results.csv', index=False)