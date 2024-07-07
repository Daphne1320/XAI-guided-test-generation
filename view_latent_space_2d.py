from utils import *
from data.utils import value_analysis


def create_mesh(dimension=12, mesh_size_x=10, mesh_size_y=10):
    # Define the mesh grid parameters again correctly
    x_start, x_end = -2, 2
    y_start, y_end = -2, 2
    num_total_dimensions = dimension

    # Generate mesh grid for the first two dimensions
    x_values = np.linspace(x_start, x_end, mesh_size_x)
    y_values = np.linspace(y_start, y_end, mesh_size_y)
    xx, yy = np.meshgrid(x_values, y_values, indexing='ij')

    # Initialize a numpy array for the entire dataset with zeros
    mesh_data_reshaped = np.zeros((mesh_size_x, mesh_size_y, num_total_dimensions))

    # Fill the first two dimensions with the mesh grid values
    mesh_data_reshaped[..., 0] = xx
    mesh_data_reshaped[..., 1] = yy

    return mesh_data_reshaped


if __name__ == "__main__":
    # 1. data
    # load data
    train_images, train_labels, test_images, test_labels = mnist_data()
    samples, sample_labels = sample_and_categorize(train_images, train_labels, number=3000)
    samples_test, sample_labels_test = sample_and_categorize(test_images, test_labels, number=len(test_labels))
    print(samples.shape)
    print(sample_labels.shape)

    # reshape data
    x_train = np.reshape(samples, (-1, 784))
    # x_train = np.reshape(train_images, (-1, 784))
    x_train_samples = np.reshape(samples, (-1, 784))
    print(x_train.shape)
    print(x_train_samples.shape)

    vae = VAE.load("trained_models")

    classifier = vae.classifier()
    encodings_pretrain = vae.encoder.predict(x_train_samples)
    image_encoder = clone_encoder(vae.image_encoder())

    encodings_samples = vae.encoder.predict(x_train_samples)

    tsne = TSNE(n_components=2, random_state=42)
    encodings_samples_2d = tsne.fit_transform(encodings_samples)
    plot_encodings2d_with_labels(encodings_samples_2d, sample_labels)

    cnn = load_model("trained_models/classifier.h5")
    xai = xai_model(vae.decoder, cnn)

    value_analysis(encodings_samples[:, 0])
    value_analysis(encodings_samples[:, 1])

    mesh_data_reshaped = create_mesh()

    # Plot setup
    fig, axs = plt.subplots(10, 10, figsize=(20, 20))

    labels_pred = []
    for i in range(10):
        for j in range(10):
            # Decode each point to a 2D matrix
            z = np.reshape(mesh_data_reshaped[i, j, :], (1, 12))
            matrix_data = vae.decoder(z)
            img = np.reshape(matrix_data[0], (28, 28))
            label = np.argmax(xai(z)[0])
            labels_pred.append(label)

            # Plot the matrix
            axs[i, j].imshow(img)
            axs[i, j].axis('off')  # Hide the axis
            axs[i, j].set_title("{} ({:.2f},{:.2f})".format(label, z[0, 0], z[0, 1]))

    plt.tight_layout()
    plt.show()

    print(set(labels_pred))
