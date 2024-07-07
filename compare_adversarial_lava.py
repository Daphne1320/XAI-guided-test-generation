from compare_adversarial import *

if __name__ == "__main__":
    alpha = 0.5
    num_iterations = 10
    num_test_samples = 200

    validator = VAEValidation()
    cnn = load_model("trained_models/classifier.h5")
    # input shape: (28, 28, 1)

    # Load models
    vae = VAE.load("trained_models/epoch_200")
    cae = CAE.load("trained_models/epoch_200")
    cvae = C_VAE.load("trained_models/epoch_100")

    models = [vae, cae, cvae]
    # models = [cvae]
    # model_names = ["cvae"]
    model_names = ["vae", "cae", "cvae"]
    adversarial_methods = ["forward_gradient", "bp_gradient", "vanilla"]
    # adversarial_methods = ["forward_gradient"]

    # Create LavaMultiSteps instances
    lava_instances = {}
    for model, prefix in zip(models, model_names):
        for method in adversarial_methods:
            key = f"lava_{prefix}_{method}"
            lava_instances[key] = LavaMultiSteps(cnn, model.encoder, model.decoder, adversarial_method=method,
                                                 verbose=True)

    # Load input images
    samples, sample_labels, sample_indices = load_samples_for_test(num_test_samples, return_indices=True)
    y_onehot = tf.one_hot(tf.constant(sample_labels), depth=10).numpy()

    tf.keras.backend.set_learning_phase(0)

    res_folder = f'results/alpha{alpha}_ni{num_iterations}_n{num_test_samples}'
    if not os.path.exists(res_folder):
        os.makedirs(res_folder)

    predicted_labels_list = []
    true_labels_list = []
    results = []
    # Start experiment
    for i in tqdm(range(len(samples)), desc="samples"):
        try:
            result = dict()
            result["index"] = i
            result["index_in_dataset"] = sample_indices[i]

            image_org, label_org = samples[i], sample_labels[i]

            # Generate adversarial image
            adversarial_images = {}
            for key, lava in tqdm(lava_instances.items(), desc="lavam methods"):
                start = time.time()
                image = \
                    lava.generate_adversarial_image(image_org, y_onehot[i], alpha=alpha, num_iterations=num_iterations)[
                        0]
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
                        try:
                            divergence = func(image_org, img)
                        except:
                            raise Exception(f"{d_name}{mtd}")
                        result[f"{d_name}_{mtd}"] = divergence
                    else:
                        result[f"{d_name}_{mtd}"] = None

            for mtd, img in tqdm(adversarial_images.items(), desc="validation for all images"):
                if predicted_labels[mtd] != label_org:
                    result[f"valid_{mtd}"] = validator.validate(img)
                else:
                    result[f"valid_{mtd}"] = None

            # Append the result to results
            if any(value is not None for key, value in result.items() if key.startswith(f"kl_")):
                results.append(result)

            # Initialize empty lists for filtered images, labels, and markers
            filtered_images = []
            filtered_labels = []
            filtered_markers = []

            # Iterate through predicted_labels and their corresponding adversarial images and markers
            for marker, (img, lbl) in zip(adversarial_images.keys(),
                                          zip(adversarial_images.values(), predicted_labels.values())):
                if lbl != label_org:
                    filtered_images.append(img)
                    filtered_labels.append(lbl)
                    filtered_markers.append(marker)

            # Include the original image, label, and marker if there are any filtered labels
            if filtered_labels:
                images = [image_org] + filtered_images
                labels = [label_org] + filtered_labels
                markers = ["org"] + filtered_markers
                num_subfigures = len(images)
                save_filename = os.path.join(res_folder, f"{i}_index{sample_indices[i]}_{num_subfigures}.png")

                for j in range(1, len(images)):
                    subdir = os.path.join(res_folder, markers[j])
                    if not os.path.exists(subdir):
                        os.makedirs(subdir)
                    plt.imshow(images[j], cmap="gray")
                    plt.savefig(os.path.join(subdir, f"org{label_org}_pred{labels[j]}_{i}_index{sample_indices[i]}.png"))

                if len(images) == 10:
                    fig, axes = plt.subplots(4, 3, figsize=(10, 10))
                    for ax, img, label, marker in zip(np.ravel(axes)[:10], images, labels, markers):
                        ax.imshow(img.squeeze(), cmap="gray")
                        ax.set_title(f"image_{marker}_{label}")
                        ax.axis('off')
                    plt.tight_layout()
                    if save_filename:
                        plt.savefig(save_filename)
                        plt.close()
                    else:
                        plt.show()

                elif len(images) == 9:
                    fig, axes = plt.subplots(3, 3, figsize=(10, 10))
                    for ax, img, label, marker in zip(np.ravel(axes), images, labels, markers):
                        ax.imshow(img.squeeze(), cmap="gray")
                        ax.set_title(f"image_{marker}_{label}")
                        ax.axis('off')
                    plt.tight_layout()
                    if save_filename:
                        plt.savefig(save_filename)
                        plt.close()
                    else:
                        plt.show()
                elif len(images) == 6:
                    fig, axes = plt.subplots(3, 2, figsize=(10, 8))
                    for ax, img, label, marker in zip(np.ravel(axes), images, labels, markers):
                        ax.imshow(img.squeeze(), cmap="gray")
                        ax.set_title(f"image_{marker}_{label}")
                        ax.axis('off')
                    plt.tight_layout()
                    if save_filename:
                        plt.savefig(save_filename)
                        plt.close()
                    else:
                        plt.show()

                else:
                    plot_image_comparison(images, labels, markers, save_path=save_filename)

                # Print the results
                # for key, value in result.items():
                #    print(f"{key}: {value}")
                # print("-" * 64)

            # Plot the images if original label differs from all adversarial labels
            """
            if label_org not in predicted_labels.values():
                images = [image_org] + list(adversarial_images.values())
                labels = [label_org] + list(predicted_labels.values())
                markers = ["org"] + list(adversarial_images.keys())
                plot_image_comparison(images, labels, markers, save_path=os.path.join(res_folder, f"{i}.png"))
    
                for key, value in result.items():
                    print(f"{key}: {value}")
                print("-" * 64)
    
            """

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
