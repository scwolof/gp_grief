
import pytest
import numpy as np
from numpy.testing import assert_almost_equal, assert_array_almost_equal
from scipy.stats import multivariate_normal as mvn

from gp_grief.grid import InducingGrid
from gp_grief.kern import RBF, GriefKernel
from gp_grief.models import GPGriefModel


class TestGPGriefModel:

    def test_gp_grief_model (self):
        np.random.seed(0)
        d = 5
        n = 57
        x = np.random.rand(n,d)
        y = np.random.rand(n,1)

        grid = InducingGrid(x)
        kern = RBF(1, lengthscale=0.5)
        kern = GriefKernel(kern_list=[kern,]*d, grid=grid, n_eigs=50)
        m    = GPGriefModel(x,y,kern,noise_var=0.1)
        lml  = m._compute_log_likelihood( m.parameters )
        K    = m._mv_cov( np.identity(n) )

        # check the linear system solve accuracy
        alp = m._mv_cov_inv(y)
        alp_exact = np.linalg.solve(K, y)
        assert_array_almost_equal(alp, alp_exact, decimal=6)

        # check the log determinant
        log_det = m._cov_log_det()
        log_det_exact = np.linalg.slogdet(K)[1]
        assert_almost_equal(log_det, log_det_exact, decimal=6)

        # check the LML accuacy
        lml_exact = mvn.logpdf(x=y.squeeze(), mean=np.zeros(n), cov=K)
        assert_almost_equal(lml, lml_exact, decimal=6)

        m.optimize()
        xnew  = np.random.rand(13, d)
        mu,s2 = m.predict(xnew)