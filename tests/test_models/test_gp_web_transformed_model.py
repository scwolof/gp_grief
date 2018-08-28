
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.stats import multivariate_normal as mvn
from scipy.stats import chi2

from gp_grief.models import GPwebTransformedModel


class TestGPwebTransformedModel:

    def test_gp_web_transformed_model (self):
        np.random.seed(0)
        X = np.random.randn(100, 4)
        X[:, 0] = 1. # Intercept column.
        Y = np.dot(X, [0.5, 0.1, 0.25, 1.]) + 0.1 * np.random.randn(X.shape[0])

        m            = GPwebTransformedModel(Phi=X, y=Y)
        tmp          = np.random.rand(*m.parameters.shape) + 1e-6
        m.parameters = tmp
        ll_gpweb     = m.log_likelihood()

        # now compute manually
        w    = m.kern.parameters.reshape((1,-1))
        sig2 = m.noise_var
        Phit = np.linalg.svd(X, full_matrices=False, compute_uv=True)[0]
        K    = Phit.dot(np.diag(w.squeeze()).dot(Phit.T)) \
                + sig2*np.identity(Phit.shape[0])
        ll_exact = mvn.logpdf(Y.squeeze(), mean=np.zeros(Phit.shape[0]), cov=K)

        assert_array_almost_equal(ll_gpweb, ll_exact)

        m.checkgrad()