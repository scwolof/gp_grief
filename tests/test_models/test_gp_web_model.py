
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.stats import multivariate_normal as mvn
from scipy.stats import chi2

from gp_grief.models import GPwebModel


class TestGPwebModel:

    def test_gp_web_model (self):
        np.random.seed(0)
        X = np.random.randn(100, 4)
        X[:, 0] = 1. # Intercept column.
        Y = np.dot(X, [0.5, 0.1, 0.25, 1.]) + 0.1 * np.random.randn(X.shape[0])

        m            = GPwebModel(Phi=X, y=Y)
        tmp          = np.random.rand(*m.parameters.shape) + 1e-6
        m.parameters = tmp
        ll_gpweb     = m.log_likelihood()

        # now compute manually
        w    = m.kern.parameters.reshape((1,-1))
        sig2 = m.noise_var
        Phi  = X
        K    = Phi.dot(np.diag(w.squeeze()).dot(Phi.T)) \
                + sig2*np.identity(Phi.shape[0])
        ll_exact = mvn.logpdf(Y.squeeze(), mean=np.zeros(Phi.shape[0]), cov=K)

        assert_array_almost_equal(ll_gpweb, ll_exact)

        m.checkgrad()