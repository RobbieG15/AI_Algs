import os
import re
import shutil
import tempfile

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import multivariate_normal, norm
from sklearn.decomposition import PCA
from tqdm import tqdm


# === Initialization ===
def initialize_parameters(data, k):
    n, d = data.shape
    indices = np.random.choice(n, k, replace=False)
    means = data[indices].copy()
    if d == 1:
        covariances = np.array([np.var(data)] * k)
    else:
        covariances = np.array([np.cov(data.T) for _ in range(k)])
    pis = np.ones(k) / k
    return means, covariances, pis


# === Gaussian PDF ===
def gaussian_pdf(x, mean, cov):
    if x.ndim == 1:
        x = x[np.newaxis, :]  # shape (1, d)

    if len(mean) == 1:  # 1D case
        return norm.pdf(x.ravel(), loc=mean[0], scale=np.sqrt(cov))
    else:  # Multivariate
        return multivariate_normal.pdf(x[0], mean=mean, cov=cov)


# === E-Step ===
def expectation_step(data, means, covariances, pis):
    n, k = data.shape[0], means.shape[0]
    responsibilities = np.zeros((n, k))
    for i in range(k):
        for j in range(n):
            if data.shape[1] == 1:
                responsibilities[j, i] = pis[i] * norm.pdf(
                    data[j, 0], loc=means[i, 0], scale=np.sqrt(covariances[i])
                )
            else:
                responsibilities[j, i] = pis[i] * gaussian_pdf(
                    data[j], means[i], covariances[i]
                )
    responsibilities /= responsibilities.sum(axis=1, keepdims=True)
    return responsibilities


# === M-Step ===
def maximization_step(data, responsibilities):
    n, d = data.shape
    k = responsibilities.shape[1]
    Nk = responsibilities.sum(axis=0)
    means = np.zeros((k, d))
    pis = Nk / n

    if d == 1:
        covariances = np.zeros(k)
        for i in range(k):
            means[i] = (responsibilities[:, i, np.newaxis] * data).sum(axis=0) / Nk[i]
            diff = data - means[i]
            covariances[i] = (responsibilities[:, i] * (diff[:, 0] ** 2)).sum() / Nk[i]
    else:
        covariances = np.zeros((k, d, d))
        for i in range(k):
            means[i] = (responsibilities[:, i, np.newaxis] * data).sum(axis=0) / Nk[i]
            diff = data - means[i]
            covariances[i] = (
                sum(
                    responsibilities[j, i] * np.outer(diff[j], diff[j])
                    for j in range(n)
                )
                / Nk[i]
            )

    return means, covariances, pis


# === EM Algorithm ===
def em_mixture_model(d, num_iterations, data, k, save_gif=False):
    assert data.shape[1] == d, "Data dimensionality mismatch"
    means, covariances, pis = initialize_parameters(data, k)

    temp_dir = tempfile.mkdtemp()

    for iteration in tqdm(range(num_iterations), desc="EM Iteration"):
        responsibilities = expectation_step(data, means, covariances, pis)
        means, covariances, pis = maximization_step(data, responsibilities)

        if save_gif:
            save_plot(
                data, means, covariances, responsibilities, k, temp_dir, iteration
            )

    if d == 1:
        stds = np.sqrt(covariances)[:, np.newaxis]
    else:
        stds = np.array([np.sqrt(np.diag(cov)) for cov in covariances])

    if save_gif:
        create_gif(temp_dir)

    # Clean up the temporary directory
    shutil.rmtree(temp_dir)

    return means, stds, covariances, responsibilities


# === Plotting (1D) ===
def plot_1d(data, means, covariances, responsibilities, k, show):
    cluster_ids = np.argmax(responsibilities, axis=1)
    colors = plt.get_cmap("tab10", k)
    plt.figure(figsize=(8, 4))

    # Histogram of data, colored by cluster
    for i in range(k):
        cluster_data = data[cluster_ids == i, 0]
        plt.hist(
            cluster_data,
            bins=30,
            density=True,
            alpha=0.5,
            label=f"Cluster {i + 1}",
            color=colors(i),
        )

    # Plot Gaussians
    x = np.linspace(np.min(data) - 2, np.max(data) + 2, 500)
    for i in range(k):
        y = norm.pdf(x, loc=means[i, 0], scale=np.sqrt(covariances[i]))
        plt.plot(x, y, color=colors(i), linewidth=2)

    plt.title("1D Gaussian Mixture Model")
    plt.xlabel("X")
    plt.ylabel("Density")
    plt.legend()
    plt.grid(True)
    if show:
        plt.show()


# === Plotting for 2D or PCA projection ===
def plot_gaussian_density(data, means, covariances, cluster_ids, k, show):
    plt.figure(figsize=(8, 6))
    colors = plt.get_cmap("tab10", k)

    for i in range(k):
        cluster_data = data[cluster_ids == i]
        plt.scatter(
            cluster_data[:, 0],
            cluster_data[:, 1],
            s=10,
            alpha=0.6,
            label=f"Cluster {i + 1}",
            color=colors(i),
        )

    x = np.linspace(np.min(data[:, 0]) - 1, np.max(data[:, 0]) + 1, 300)
    y = np.linspace(np.min(data[:, 1]) - 1, np.max(data[:, 1]) + 1, 300)
    X, Y = np.meshgrid(x, y)
    pos = np.dstack((X, Y))

    for i in range(k):
        rv = multivariate_normal(mean=means[i], cov=covariances[i])
        Z = rv.pdf(pos)
        plt.contour(X, Y, Z, levels=5, colors=[colors(i)], linewidths=1.5)

    plt.title("Gaussian Mixture Model (Density Contours)")
    plt.xlabel("X1")
    plt.ylabel("X2")
    plt.legend()
    plt.grid(True)
    plt.axis("equal")
    if show:
        plt.show()


# === Wrapper: handles 1D, 2D, >2D ===
def plot_gaussians_projected(data, means, covariances, responsibilities, k, show=True):
    d = data.shape[1]
    if d == 1:
        plot_1d(data, means, covariances, responsibilities, k, show)
    elif d == 2:
        cluster_ids = np.argmax(responsibilities, axis=1)
        plot_gaussian_density(data, means, covariances, cluster_ids, k, show)
    else:
        pca = PCA(n_components=2)
        data_2d = pca.fit_transform(data)
        means_2d = pca.transform(means)
        covariances_2d = [
            pca.components_ @ cov @ pca.components_.T for cov in covariances
        ]
        cluster_ids = np.argmax(responsibilities, axis=1)
        plot_gaussian_density(data_2d, means_2d, covariances_2d, cluster_ids, k, show)


# === Save Plot to Temporary Directory ===
def save_plot(data, means, covariances, responsibilities, k, temp_dir, iteration):
    plot_gaussians_projected(data, means, covariances, responsibilities, k, show=False)

    # Save the figure to the temp directory
    plt.savefig(os.path.join(temp_dir, f"iteration_{iteration}.png"))
    plt.close()


# === Create GIF from Saved Figures ===
def create_gif(temp_dir):
    def extract_iteration_number(filename):
        match = re.search(r"iteration_(\d+)\.png", filename)
        return int(match.group(1)) if match else -1

    # Get all the image files in the temp directory
    image_files = [f for f in os.listdir(temp_dir) if f.endswith(".png")]
    sorted_files = sorted(image_files, key=extract_iteration_number)
    images = [imageio.imread(os.path.join(temp_dir, f)) for f in sorted_files]

    # Save the images as a GIF
    gif_output_path = os.path.join("outputs", "gmm_iterations.gif")
    imageio.mimsave(gif_output_path, images, duration=1.5)
    print(f"GIF saved to {gif_output_path}")


# === Example Usage ===
if __name__ == "__main__":
    np.random.seed(69)

    d = 5
    k = 3
    n = 300
    num_iterations = 20

    # Generate synthetic data
    if d == 1:
        data = np.concatenate(
            [
                np.random.normal(-5, 1, (n, 1)),
                np.random.normal(0, 1, (n, 1)),
                np.random.normal(5, 1, (n, 1)),
            ]
        )
    else:
        mean_list = [np.ones(d) * i for i in range(k)]
        cov = np.eye(d)
        data = np.vstack(
            [np.random.multivariate_normal(mean, cov, n) for mean in mean_list]
        )

    means, stds, covariances, responsibilities = em_mixture_model(
        d, num_iterations, data, k, save_gif=True
    )

    print("Final Means:\n", means)
    print("Final STDs:\n", stds)

    plot_gaussians_projected(data, means, covariances, responsibilities, k)
