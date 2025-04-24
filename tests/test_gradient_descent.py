import tempfile
from pathlib import Path

import numpy as np
import pytest

from gradient_descent import GradientDescentVisualizer


# --- Fixtures ---
@pytest.fixture
def dummy_data():
    np.random.seed(42)
    n_samples, n_features = 1, 1  # Set to 1 feature for final line fit test
    X = np.random.rand(n_samples, n_features)
    X = np.hstack([np.ones((n_samples, 1)), X])  # bias term
    true_theta = np.array([1] + [2] * n_features)
    y = X @ true_theta + np.random.randn(n_samples) * 0.3
    return X, y


@pytest.fixture
def visualizer(dummy_data):
    X, y = dummy_data
    return GradientDescentVisualizer(X, y, num_iterations=10)


# --- Tests ---
def test_fit_reduces_cost(visualizer):
    initial_cost = visualizer.cost_fn(visualizer.theta)
    visualizer.fit()
    final_cost = visualizer.cost_fn(visualizer.theta)
    assert final_cost < initial_cost, "Gradient descent should reduce the cost"


def test_theta_shape(visualizer):
    visualizer.fit()
    assert visualizer.theta.shape[0] == visualizer.X.shape[1], "Theta shape mismatch"


def test_gif_saved_tempdir(visualizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer.output_dir = tmpdir
        visualizer.fit()
        gif_path = visualizer.save_gif("test_gif.gif")
        assert Path(gif_path).exists(), f"GIF not found at {gif_path}"


def test_final_plot_saved_tempdir(visualizer):
    with tempfile.TemporaryDirectory() as tmpdir:
        visualizer.output_dir = tmpdir
        visualizer.fit()
        plot_path = visualizer.save_final_fit_plot("test_plot.png")
        if visualizer.X.shape[1] == 2:
            assert Path(plot_path).exists(), f"Final plot not found at {plot_path}"
        else:
            assert plot_path is None, (
                "Should not attempt to save plot if X has >1 feature"
            )


# --- Custom Cost Function (MAE) ---
def mae_cost(theta, X, y):
    return np.mean(np.abs(X @ theta - y))


def mae_grad(theta, X, y):
    return (1 / len(y)) * X.T @ np.sign(X @ theta - y)


@pytest.fixture
def mae_visualizer(dummy_data):
    X, y = dummy_data
    return GradientDescentVisualizer(
        X,
        y,
        num_iterations=10,
        learning_rate=0.05,
        cost_fn=lambda theta: mae_cost(theta, X, y),
        gradient_fn=lambda theta: mae_grad(theta, X, y),
    )


def test_custom_cost_fn(mae_visualizer):
    initial = mae_visualizer.cost_fn(mae_visualizer.theta)
    mae_visualizer.fit()
    final = mae_visualizer.cost_fn(mae_visualizer.theta)
    assert final < initial, "Custom cost function should still reduce error"
