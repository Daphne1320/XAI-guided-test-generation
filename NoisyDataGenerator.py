import numpy as np
import matplotlib.pyplot as plt

from keras.utils import Sequence


class NoisyDataGenerator(Sequence):
    def __init__(self, images, batch_size=32, clip_normalize=True, noise_scale=1.0):
        self.images = images
        self.indices = np.arange(len(self.images))
        self.batch_size = batch_size
        self.epoch_count = 0

        self.clip_normalize = clip_normalize
        self.noise_scale = noise_scale

    def on_epoch_end(self):
        """This function gets called at the end of each epoch."""
        self.epoch_count += 1

        np.random.shuffle(self.indices)
        self.images = self.images[self.indices]

    def __len__(self):
        """This denotes the number of batches per epoch."""
        return int(np.ceil(len(self.images) / self.batch_size))

    def __getitem__(self, index):
        """Generate a batch of data."""
        start_index = index * self.batch_size
        end_index = (index + 1) * self.batch_size

        batch_images = self.images[start_index:end_index]

        # Add noise based on the current epoch
        # For demonstration purposes, we'll use the epoch number to modulate the noise intensity
        noise_factor = 0.5 + (0.5 * (self.epoch_count % 10) / 10)  # Varying between 0.5 to 1.0
        noisy_images = batch_images + noise_factor * np.random.normal(loc=0.0, scale=self.noise_scale,
                                                                      size=batch_images.shape)

        if self.clip_normalize:
            noisy_images = np.clip(noisy_images, 0., 1.)

        return noisy_images, batch_images

    def view(self, count=5, dimension=1):
        if dimension == 1:
            self.view_signal(count=count)
        else:
            self.view_image(count=count)

    def view_image(self, count=5):
        """Visualize the first two pairs of original and noised images for the first `count` batches."""
        for batch_index in range(count):
            noisy_images, original_images = self[batch_index]
            img_index = 0

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

            ax1.imshow(noisy_images[img_index], cmap='gray')
            ax1.set_title(f'Noisy Image from Batch {batch_index + 1}')
            ax1.axis('off')

            ax2.imshow(original_images[img_index], cmap='gray')
            ax2.set_title(f'Original Image from Batch {batch_index + 1}')
            ax2.axis('off')

            plt.show()

    def view_signal(self, count=5):
        """Visualize the first two pairs of original and noised signal for the first `count` batches.
        blue one is noisy input, black is output"""
        for batch_index in range(count):
            noisy_signals, original_signals = self[batch_index]
            signal_index = 0
            channel_index = 0

            fig, ax = plt.subplots(figsize=(20, 5))

            ax.plot(noisy_signals[signal_index, :, channel_index], color='blue', label='Noisy Signal')
            ax.plot(original_signals[signal_index, :, channel_index], color='black', label='Original Signal')

            ax.set_title(f'Batch {batch_index + 1}: Noisy vs Original Signal')
            ax.grid(True)
            ax.legend()

            plt.tight_layout()
            plt.show()
