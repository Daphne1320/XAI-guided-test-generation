import pandas as pd
import json
import time

from adversarial import *
from adversarial_methods.DLFuzz.dlfuzz import DLFuzz
from adversarial_methods.fgsm import FGSM


def plot_image_comparison(images, labels, markers, cmap='gray', save_path=None):
    fig, axes = plt.subplots(1, len(images), figsize=(15, 5))
    for ax, img, label, marker in zip(axes, images, labels, markers):
        ax.imshow(img.squeeze(), cmap=cmap)
        ax.set_title(f"image_{marker}_{label}")
        ax.axis('off')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path)
        plt.close()
    else:
        plt.show()


def calculate_success_ratio(predicted_labels, true_labels):
    assert len(predicted_labels) == len(true_labels)
    success_counts = {key: 0 for key in predicted_labels[0].keys() if key != "org"}
    total_samples = len(predicted_labels)

    for i, pred in enumerate(predicted_labels):
        for key in success_counts.keys():
            if pred[key] != true_labels[i]:
                success_counts[key] += 1

    success_ratios = {key: count / total_samples for key, count in success_counts.items()}
    return success_ratios


if __name__ == "__main__":
    num_test_samples = 200
    fgsm_step = 0.1

    cnn = load_model("trained_models/classifier.h5")
    # input shape: (28, 28, 1)

    # vae = VAE.load("trained_models")
    alg_instances = {
        "fgsm": FGSM(cnn),
        "dlfuzz": DLFuzz(cnn),
        # "lava": Lava(cnn, vae.encoder, vae.decoder),
        # "lavam": LavaMultiSteps(cnn, vae.encoder, vae.decoder, verbose=True),
    }

    # input images
    samples, sample_labels, sample_indices = load_samples_for_test(num_test_samples, return_indices=True)

    y_onehot = tf.one_hot(tf.constant(sample_labels), depth=10).numpy()

    K.set_learning_phase(0)

    res_folder = f'results/n_{num_test_samples}_fgsm({fgsm_step})'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    predicted_labels_list = []
    true_labels_list = []
    results = []
    # Start experiment
    for i in tqdm(range(len(samples))):
        try:
            result = dict()
            result["index"] = i
            result["index_in_dataset"] = sample_indices[i]

            # Get original image and label
            image_org, label_org = samples[i], sample_labels[i]

            # Generate adversarial images of shape (28, 28, 1)
            adversarial_images = {}
            for key, mtd in tqdm(alg_instances.items(), desc="methods"):
                start = time.time()

                if key == "fgsm":
                    image = mtd.generate_adversarial_image(image_org, y_onehot[i], step=fgsm_step)
                elif key == "dlfuzz":
                    image = mtd.generate_adversarial_image(image_org)
                elif key == "lava":
                    image = mtd.generate_adversarial_image(image_org, y_onehot[i])[0]
                elif key == "lavam":
                    image = mtd.generate_adversarial_image(image_org, y_onehot[i], alpha=0.05, num_iterations=10)[0]
                else:
                    image = image_org

                end = time.time()
                result[f"t_{key}"] = end - start
                adversarial_images.update({key: image})

            # Predict labels for adversarial images
            predicted_labels = {
                key: np.argmax(cnn.predict(np.array([img]))[0])
                for key, img in adversarial_images.items()
            }
            predicted_labels_list.append(predicted_labels)
            true_labels_list.append(sample_labels[i])

            # In image space
            divergence_functions = {
                "kl": kl_divergence,
                "ws": ws_distance,
                "js": js_divergence,
                "mse": mse_loss,
                # "xentropy", cross_entropy,
            }
            # Iterate over divergence functions
            for d_name, func in divergence_functions.items():
                # Compute divergences only for matching labels
                for mtd, img in tqdm(adversarial_images.items(), desc="metrics for all images"):
                    if predicted_labels[mtd] != label_org:
                        divergence = func(image_org, img)
                        result[f"{d_name}_{mtd}"] = divergence
                    else:
                        result[f"{d_name}_{mtd}"] = None

            # Append the result to results
            if any(value is not None for key, value in result.items() if key.startswith(f"kl_")):
                results.append(result)

            # Plot the images if original label differs from all adversarial labels
            if label_org not in predicted_labels.values():
                images = [image_org] + list(adversarial_images.values())
                labels = [label_org] + list(predicted_labels.values())
                markers = ["org"] + list(adversarial_images.keys())
                num_subfigures = len(images)
                save_filename = os.path.join(res_folder, f"{i}_index{sample_indices[i]}_{num_subfigures}.png")

                for j in range(1, len(images)):
                    subdir = os.path.join(res_folder, markers[j])
                    if not os.path.exists(subdir):
                        os.makedirs(subdir)
                    plt.imshow(images[j], cmap="gray")
                    plt.savefig(
                        os.path.join(subdir, f"org{label_org}_pred{labels[j]}_{i}_index{sample_indices[i]}.png"))

                plot_image_comparison(images, labels, markers, save_path=save_filename)

                # for key, value in result.items():
                #     print(f"{key}: {value}")
                # print("-" * 64)

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
            if i % 10 == 0:
                # Calculate success ratios
                success_ratios = calculate_success_ratio(predicted_labels_list, true_labels_list)
                print(success_ratios)
                with open(os.path.join(res_folder, f'success_ratio.json'), 'w') as file:
                    json.dump(success_ratios, file, indent=4)

                # Save metrics
                results_df = pd.DataFrame(results)
                results_df.to_csv(os.path.join(res_folder, f'results.csv'), index=False)

        except:
            continue

    # Calculate success ratios
    success_ratios = calculate_success_ratio(predicted_labels_list, true_labels_list)
    print(success_ratios)
    with open(os.path.join(res_folder, f'success_ratio.json'), 'w') as file:
        json.dump(success_ratios, file, indent=4)

    # Save metrics
    results_df = pd.DataFrame(results)
    results_df.to_csv(os.path.join(res_folder, f'results.csv'), index=False)
