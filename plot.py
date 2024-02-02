import numpy as np
import matplotlib.pyplot as plt


def plot_encodings2d_with_labels(encodings, labels):
    plt.figure(figsize=(8, 8))

    unique_labels = np.unique(labels)  # Identify unique class labels
    colormap = plt.cm.jet  # or another suitable colormap like 'viridis', 'plasma', etc.

    for i, label in enumerate(unique_labels):
        indices = np.where(labels == label)
        color = colormap(i / len(unique_labels))  # Choose color from the colormap
        plt.scatter(encodings[indices, 0], encodings[indices, 1], color=color, label=str(label))

    plt.legend()
    plt.title('2D t-SNE of Encodings')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.show()
