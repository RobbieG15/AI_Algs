import numpy as np
import pytest

from gmm_em import em_mixture_model


@pytest.mark.parametrize(
    "d, n_samples, k",
    [
        (1, 300, 3),
        (2, 300, 3),
        (5, 500, 4),
    ],
)
def test_em_gmm_runs_and_shapes(d, n_samples, k):
    np.random.seed(69)

    # Generate synthetic data with d dimensions and k clusters
    data = []
    for i in range(k):
        mean = np.random.rand(d) * 10
        cov = np.eye(d) * np.random.rand()  # diagonal covariance
        samples = np.random.multivariate_normal(mean, cov, size=n_samples // k)
        data.append(samples)

    data = np.vstack(data)
    np.random.shuffle(data)

    # Run EM
    means, stds, covariances, responsibilities = em_mixture_model(
        d=d, num_iterations=10, data=data, k=k
    )

    # Assertions
    assert means.shape == (k, d)
    assert stds.shape == (k, d if d > 1 else 1)
    assert len(covariances) == k
    assert responsibilities.shape == (data.shape[0], k)

    # Check that responsibilities sum to ~1 per sample
    row_sums = responsibilities.sum(axis=1)
    assert np.allclose(row_sums, np.ones_like(row_sums), atol=1e-5)
