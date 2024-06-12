import numpy as np
import matplotlib.pyplot as plt


def sample_and_categorize(train_images, train_labels, seed=2024, number=3000, return_indices=False):
    # sample data
    np.random.seed(seed)
    indices = np.random.choice(train_images.shape[0], size=number, replace=False)
    samples, sample_labels = train_images[indices], train_labels[indices]

    # y to labels (in category format)
    sample_labels = np.argmax(sample_labels, axis=-1)
    if return_indices:
        return samples, sample_labels, indices
    else:
        return samples, sample_labels


def value_analysis(scores, varname="varname"):
    mean = np.mean(scores)
    median = np.median(scores)
    std = np.std(scores)

    plt.figure(figsize=(10, 6))
    plt.hist(scores, bins=30, alpha=0.7, label='Distribution')
    plt.axvline(mean, color='r', linestyle='dashed', linewidth=2, label=f'Mean: {mean:.2f}')
    plt.axvline(median, color='g', linestyle='dashed', linewidth=2, label=f'Median: {median:.2f}')
    plt.axvline(mean + std, color='b', linestyle='dotted', linewidth=2, label=f'Mean + Std: {mean + std:.2f}')
    plt.axvline(mean - std, color='b', linestyle='dotted', linewidth=2, label=f'Mean - Std: {mean - std:.2f}')
    plt.title(f'Distribution with Mean, Median, and Std Deviation of {varname}')
    plt.xlabel('Value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.show()