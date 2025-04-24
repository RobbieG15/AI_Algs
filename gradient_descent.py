import os
import tempfile
from pathlib import Path

import imageio.v2 as imageio
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
from tqdm import tqdm


class GradientDescentVisualizer:
    def __init__(
        self,
        X,
        y,
        learning_rate=0.1,
        num_iterations=50,
        output_dir="outputs",
        cost_fn=None,
        gradient_fn=None,
    ):
        self.X = X
        self.y = y
        self.learning_rate = learning_rate
        self.num_iterations = num_iterations
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.m, self.n = X.shape
        self.theta = np.random.randn(self.n)
        self.theta_history = []
        self.cost_history = []

        # Allow custom cost and gradient functions
        self.cost_fn = cost_fn if cost_fn else self._default_cost
        self.gradient_fn = gradient_fn if gradient_fn else self._default_gradient

    def _default_cost(self, theta):
        predictions = self.X @ theta
        return (1 / (2 * self.m)) * np.sum((predictions - self.y) ** 2)

    def _default_gradient(self, theta):
        return (1 / self.m) * self.X.T @ (self.X @ theta - self.y)

    def fit(self):
        for _ in tqdm(range(self.num_iterations), desc="Gradient Descent Iterations"):
            self.cost_history.append(self.cost_fn(self.theta))
            self.theta_history.append(self.theta.copy())
            self.theta -= self.learning_rate * self.gradient_fn(self.theta)
        self.theta_history = np.array(self.theta_history)

    def project_theta_history(self):
        self.pca = PCA(n_components=2)
        return self.pca.fit_transform(self.theta_history)

    def generate_cost_surface(self, theta_pca):
        x_vals = np.linspace(np.min(theta_pca[:, 0]), np.max(theta_pca[:, 0]), 50)
        y_vals = np.linspace(np.min(theta_pca[:, 1]), np.max(theta_pca[:, 1]), 50)
        X_grid, Y_grid = np.meshgrid(x_vals, y_vals)
        Z_grid = np.zeros_like(X_grid)

        for i in range(X_grid.shape[0]):
            for j in range(X_grid.shape[1]):
                low_dim = np.array([X_grid[i, j], Y_grid[i, j]])
                high_dim = self.pca.inverse_transform(low_dim)
                Z_grid[i, j] = self.cost_fn(high_dim)

        return X_grid, Y_grid, Z_grid

    def save_gif(self, filename="projected_gif.gif"):
        theta_pca = self.project_theta_history()
        X_grid, Y_grid, Z_grid = self.generate_cost_surface(theta_pca)
        gif_path = Path(self.output_dir).joinpath(filename).as_posix()

        with tempfile.TemporaryDirectory() as tmpdir:
            filenames = []
            for i in range(len(theta_pca)):
                fig = plt.figure(figsize=(8, 6))
                ax = fig.add_subplot(111, projection="3d")
                ax.plot_surface(X_grid, Y_grid, Z_grid, cmap="viridis", alpha=0.6)
                ax.plot(
                    theta_pca[: i + 1, 0],
                    theta_pca[: i + 1, 1],
                    self.cost_history[: i + 1],
                    color="red",
                    marker="o",
                )
                ax.set_xlabel("PCA 1")
                ax.set_ylabel("PCA 2")
                ax.set_zlabel("Cost")
                ax.set_title(f"Iteration {i} | Cost: {self.cost_history[i]:.3f}")
                fname = os.path.join(tmpdir, f"proj_frame_{i:03d}.png")
                plt.savefig(fname)
                filenames.append(fname)
                plt.close()

            with imageio.get_writer(gif_path, mode="I", duration=0.3) as writer:
                for fname in filenames:
                    writer.append_data(imageio.imread(fname))

        print(f"GIF saved to {gif_path}")

        return gif_path

    def save_final_fit_plot(self, filename="final_regression_fit.png"):
        filepath = Path(self.output_dir).joinpath(filename).as_posix()
        if self.X.shape[1] == 2:
            x_raw = self.X[:, 1]
            y_pred = self.X @ self.theta
            plt.figure(figsize=(6, 4))
            plt.scatter(x_raw, self.y, label="Data")
            plt.plot(x_raw, y_pred, color="red", label="Fitted Line")
            plt.xlabel("x")
            plt.ylabel("y")
            plt.title("Final Regression Fit")
            plt.legend()
            plt.grid(True)
            plt.savefig(filepath)
            plt.close()
            print(f"Final regression plot saved as: {filepath}")
        else:
            print(
                "Skipped regression fit plot — X must have exactly one feature (plus bias)."
            )

        return filepath


if __name__ == "__main__":
    np.random.seed(0)
    X = np.random.rand(100, 1)
    X = np.hstack([np.ones((100, 1)), X])  # add bias term
    true_theta = np.array([1, 2])
    y = X @ true_theta + np.random.randn(100) * 0.5

    gdv = GradientDescentVisualizer(
        X, y, learning_rate=0.1, num_iterations=50, output_dir="outputs"
    )
    gdv.fit()
    gdv.save_gif("example_gradient_desc.gif")
    gdv.save_final_fit_plot("example_gradient_desc_fit.png")
