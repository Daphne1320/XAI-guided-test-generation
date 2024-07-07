# -*- coding: utf-8 -*-
import h5py
from compare_adversarial import *
from model.XAI_classifier import xai_model

DATASET = 'mnist_images/janus_dataset_comparison.h5'
# from contrib.DeepJanus.mnist import JanusDataset


class AdversarialImage:
    def __init__(self):
        self.image = None
        self.label = None
        self.label_predicted = None
        self.seed = None


def load_samples_from_janusdeep(data_path="mnist_images/np_data"):
    """
    Enumerate all .npy files and load them into a list of AdversarialImage instances.

    Parameters:
    data_path (str): The path to the directory containing the .npy files.

    Returns:
    list: A list of AdversarialImage instances loaded from the .npy files.
    """
    npy_files = [f for f in os.listdir(data_path) if f.endswith('.npy')]
    adversarial_images = []

    for file in npy_files:
        # Parse filename to extract seed, true label, and predicted label
        parts = file.split('_')
        seed = int(parts[1])
        label = int(parts[2][1:])
        label_predicted = int(parts[3][1:-4])  # Remove the .npy extension

        # Load the numpy data
        file_path = os.path.join(data_path, file)
        data = np.load(file_path)

        # Create an AdversarialImage instance and populate its fields
        adv_image = AdversarialImage()
        adv_image.image = data.reshape((28, 28, 1))
        adv_image.label = label
        adv_image.label_predicted = label_predicted
        adv_image.seed = seed

        # Add the AdversarialImage instance to the list
        adversarial_images.append(adv_image)

    return adversarial_images


if __name__ == "__main__":

    vae = VAE.load("../trained_models")
    cnn = load_model("../trained_models/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))

    # input images
    # janus_dataset = JanusDataset()
    samples = np.array(h5py.File(DATASET, 'r').get('xn'))

    adversarial_images = load_samples_from_janusdeep("mnist_images/np_data/mnist_dj")
    samples_gen = [ad.image for ad in adversarial_images]
    # samples_view = [janus_dataset.generate_digit(ad.seed) for ad in adversarial_images]
    samples_view = [samples[int(ad.seed)] for ad in adversarial_images]
    sample_labels_view = [ad.label for ad in adversarial_images]

    x_view = np.reshape(samples_view, (-1, 784))
    x_gen = np.reshape(samples_gen, (-1, 784))
    y_onehot_view = tf.one_hot(tf.constant(sample_labels_view), depth=10).numpy()
    h_view = vae.encoder.predict(x_view)

    # start
    for i in tqdm(range(len(x_view))):

        # calculate fuzz image
        image_org = samples_view[i]
        image_gen = samples_gen[i]

        label_org = sample_labels_view[i]

        # calculate latent variant image
        image_gen_lava, _ = generate_adversarial_lava(h_view[i], y_onehot_view[i], vae, xai)

        label_gen = np.argmax(cnn.predict(np.array([image_gen]))[0])
        label_lava = np.argmax(cnn.predict(np.array([image_gen_lava]))[0])

        # List of images and their titles
        images = [image_org, image_gen, image_gen_lava]
        titles = [f'image_org_{label_org}', f'image_gen_{label_gen}', f'image_gen_lava_{label_lava}']

        # Plot the images
        if label_lava != label_org:
            plot_image_comparison(images, titles)

            # in image space
            kl_fuzz = kl_divergence(image_org, image_gen)
            kl_lava = kl_divergence(image_org, image_gen_lava)
            print(f"kl_fuzz: {kl_fuzz}\nkl_lava: {kl_lava}")

            ws_fuzz = ws_distance(image_org, image_gen)
            ws_lava = ws_distance(image_org, image_gen_lava)
            print(f"ws_fuzz: {ws_fuzz}\nws_lava: {ws_lava}")
