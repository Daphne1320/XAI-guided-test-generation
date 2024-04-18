import tensorflow as tf
from train_vae import *


def gradient_of_x(x, y, model, before_softmax=False):

    # Check if the last layer is a Dense layer with softmax activation
    if before_softmax and isinstance(model.layers[-1], tf.keras.layers.Dense) and \
            getattr(model.layers[-1], 'activation') == tf.keras.activations.softmax:
        # Modify the last layer to have a linear activation
        model_clone = tf.keras.models.clone_model(model)
        model_clone.set_weights(model.get_weights())
        model_clone.layers[-1].activation = tf.keras.activations.linear
        model = tf.keras.Model(inputs=model_clone.inputs, outputs=model_clone.layers[-1].output)

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

# x as correct predicted input data
def saliency_of_x(x, model, before_softmax=False):

    # Check if the last layer is a Dense layer with softmax activation
    if before_softmax and isinstance(model.layers[-1], tf.keras.layers.Dense) and \
            getattr(model.layers[-1], 'activation') == tf.keras.activations.softmax:
        # Modify the last layer to have a linear activation
        model_clone = tf.keras.models.clone_model(model)
        model_clone.set_weights(model.get_weights())
        model_clone.layers[-1].activation = tf.keras.activations.linear
        model = tf.keras.Model(inputs=model_clone.inputs, outputs=model_clone.layers[-1].output)

    # Convert the numpy array to a TensorFlow tensor
    input_data = tf.convert_to_tensor(x, dtype=tf.float32)

    with tf.GradientTape() as tape:
        tape.watch(input_data)  # Explicitly watch the input tensor

        # Feeding `input_data` to the model and tracking operations for gradient computation
        predictions = model(input_data, training=False)

        # Use the maximum of the predictions as the target for the gradient
        target = tf.reduce_max(predictions, axis=1)

    # Compute the gradient of the target with respect to the input
    return tape.gradient(target, input_data)


def plot_image_and_gradient(img, gradient, title='Gradient of Hidden vector'):
    # Plotting setup with adjusted aspect ratio for a narrower gradient plot
    fig, axs = plt.subplots(1, 2, figsize=(12, 4), gridspec_kw={'width_ratios': [1, 0.2]})

    # Original Image
    ax = axs[0]
    ax.imshow(img, cmap='gray')
    ax.set_title('Original Image')

    # 1D Gradient Vertically and Narrow
    ax = axs[1]
    ax.barh(range(len(gradient)), gradient, color='blue')  # Using a single color for simplicity
    ax.set_title(title)
    ax.invert_yaxis()  # Inverting the y-axis

    plt.tight_layout()
    plt.show()


def latent_space_display(h, decoder, highlight_dim=0, n_variation=9):
    """
    Visualizes how variations in each dimension of a latent vector affect the generated images.
    This version includes 9 variations per dimension, with the center being the original vector.

    Parameters:
    - h: The base latent vector from which variations are generated.
    - decoder: The decoder model that generates images from latent vectors.
    - highlight_dim: The index of the dimension to highlight with a different colormap.
    """
    num_dims = h.shape[1]  # Assuming h is of shape [1, latent_dim]
    # Generate 9 variations with the center being no variation
    variation_range = np.linspace(-1.0, 1.0, n_variation)

    # Adjust plot settings for 9 variations
    fig, axs = plt.subplots(num_dims, n_variation, figsize=(18, 2 * num_dims))

    for dim in range(num_dims):
        for i, var in enumerate(variation_range):
            # Use the original h for the center variation
            modified_h = np.copy(h)
            if var != 0:  # Apply variation only if var is not zero
                modified_h[0, dim] += var  # Vary the current dimension

            generated_image = decoder.predict(modified_h)
            generated_image_reshaped = np.reshape(generated_image, (28, 28))

            ax = axs[dim] if num_dims == 1 else axs[dim, i]
            ax.imshow(generated_image_reshaped, cmap='gray' if dim != highlight_dim else 'cool')

            # center column
            if i == len(variation_range) // 2:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(10)

            ax.set_xticks([])  # Hide x-axis tick marks
            ax.set_yticks([])  # Hide y-axis tick marks
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            if i == 0:
                ax.set_ylabel(f'Dim {dim}')

    plt.tight_layout()
    plt.show()


def latent_space_display_mark(h, y, decoder, predictor, highlight_dim=0, n_variation=9):
    """
    Visualizes how variations in each dimension of a latent vector affect the generated images.
    This version includes 9 variations per dimension, with the center being the original vector and marks variations
    leading to a different prediction with a red border.

    Parameters:
    - h: The base latent vector from which variations are generated.
    - y: The actual label of the input data.
    - decoder: The decoder model that generates images from latent vectors.
    - predictor: The model that predicts the label from the latent vector or generated image.
    - highlight_dim: The index of the dimension to highlight with a different colormap.
    - n_variation: Number of variations per dimension to generate.
    """
    num_dims = h.shape[1]  # Assuming h is of shape [1, latent_dim]
    # Generate 9 variations with the center being no variation
    variation_range = np.linspace(-1.0, 1.0, n_variation)

    # Adjust plot settings for 9 variations
    fig, axs = plt.subplots(num_dims, n_variation, figsize=(18, 2 * num_dims))

    for dim in range(num_dims):
        for i, var in enumerate(variation_range):
            # Use the original h for the center variation
            modified_h = np.copy(h)
            if var != 0:  # Apply variation only if var is not zero
                modified_h[0, dim] += var  # Vary the current dimension

            generated_image = decoder.predict(modified_h)
            generated_image_reshaped = np.reshape(generated_image, (28, 28))
            y_pred = np.argmax(predictor.predict(modified_h))

            ax = axs[dim] if num_dims == 1 else axs[dim, i]
            ax.imshow(generated_image_reshaped, cmap='gray' if dim != highlight_dim else 'cool')

            # center column
            if i == len(variation_range) // 2:
                for spine in ax.spines.values():
                    spine.set_edgecolor('gray')
                    spine.set_linewidth(10)

            # variated label
            if y != y_pred:
                # Set the border of the subplot to be red
                for spine in ax.spines.values():
                    spine.set_edgecolor('red')
                    spine.set_linewidth(6)

            ax.set_xticks([])  # Hide x-axis tick marks
            ax.set_yticks([])  # Hide y-axis tick marks
            ax.spines['top'].set_visible(True)
            ax.spines['right'].set_visible(True)
            ax.spines['bottom'].set_visible(True)
            ax.spines['left'].set_visible(True)

            if i == 0:
                ax.set_ylabel(f'Dim {dim}')
            if dim == num_dims - 1:
                ax.set_xlabel(f"{var}")

    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
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
    # x_train_samples = np.reshape(samples, (-1, 784))
    print(x_train.shape)
    # print(x_train_samples.shape)

    # XAI framework
    vae = VAE.load("trained_models")
    cnn = load_model("trained_models/classifier.h5")
    xai = xai_model(vae.decoder, cnn, input_shape=(12,))
    xai.summary()

    view_samples = samples_test
    view_sample_labels = sample_labels_test
    # view_samples = samples
    # view_sample_labels = sample_labels

    x_view = np.reshape(view_samples, (-1, 784))
    y_view_onehot = tf.one_hot(tf.constant(view_sample_labels), depth=10).numpy()
    h_view = vae.encoder.predict(x_view)

    count = 10
    for i in range(len(x_view)):
        x = np.array([h_view[i]])
        y = np.array([y_view_onehot[i]])
        # g = gradient_of_x(x, y, xai)
        g = saliency_of_x(x, xai)

        g_npy = np.squeeze(g.numpy())
        # plot_image_and_gradient(np.reshape(x_view[i], (28, 28)), g_npy, title="Gradient of hidden vector")
        plot_image_and_gradient(np.reshape(x_view[i], (28, 28)), g_npy, title="Saliency of hidden vector")

        # Identify the maximum gradient entry
        max_grad_index = np.argmax(np.abs(g_npy))
        # latent_space_display(x, vae.decoder, highlight_dim=max_grad_index)
        latent_space_display_mark(x, view_sample_labels[i], vae.decoder, xai, highlight_dim=int(max_grad_index))

        if count <= 0:
            break
        count -= 1