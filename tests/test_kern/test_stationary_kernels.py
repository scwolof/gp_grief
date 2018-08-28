
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import GPy.kern
from gp_grief.kern import RBF, Exponential, Matern32, Matern52


class TestStationaryKernels:

    def test_stationary_kernels (self):
        for d in [1, 10]:
            np.random.seed(0)
            N = 100
            x = np.random.uniform(size=(N,d)) # generate dataset

            dd = {'lengthscale':0.5, 'variance':0.5}
            # RBF
            np.testing.assert_array_almost_equal(
                RBF(         d, **dd).K(x),
                GPy.kern.RBF(d, **dd).K(x))
            
            # Exponential
            np.testing.assert_array_almost_equal(
                Exponential(         d, **dd).K(x),
                GPy.kern.Exponential(d, **dd).K(x))
            
            # Matern32
            np.testing.assert_array_almost_equal(
                Matern32(         d, **dd).K(x),
                GPy.kern.Matern32(d, **dd).K(x))
            
            # Matern52
            np.testing.assert_array_almost_equal(
                Matern52(         d, **dd).K(x),
                GPy.kern.Matern52(d, **dd).K(x))
