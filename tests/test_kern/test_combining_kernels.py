
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal

import GPy.kern
from gp_grief.kern import GPyKernel, RBF
from gp_grief.models import GPRegressionModel


def rastrigin (x, lin_term=None):
    """
    Rastrigin test function
    input should be in range [-5.12, 5.12]
    if x in range [0,1], can transform by rastrigin((x*2-1)*5.12)
    if lin_term is not None then will add a linear term to the first dimension. 
    This helps to make the function non-symetric wrt the input dimensions
    """
    assert x.ndim == 2
    d = x.shape[1]
    f = 10*d
    for i in range(d):
        f = f+(np.power(x[:,i,None],2) - 10*np.cos(2*np.pi*x[:,i,None]));
    if lin_term is not None:
        f += lin_term*x[:,(0,)]
    return f


class TestStationaryKernels:

    def test_inhouse_kernels (self):
        # generate data
        np.random.seed(0)
        d = 5
        N = 100
        x = np.random.uniform(size=(N,d)) # generate dataset
        y = rastrigin((x*2-1)*5.12)

        dd = {'lengthscale':0.5, 'variance':0.5}
        def ddf (i):
            t = 0.5*i+0.5
            return {'lengthscale':t, 'variance':t, 'name':'k%d' % i}

        # consider cases where the base kernels are the same and different
        for case in ['constant_bases', 'different_bases']:
            
            # first define the base kernels
            if case == 'constant_bases':
                kb_gpy = [ GPy.kern.RBF(d, **dd) for _ in range(4) ]
                kb_kml = [ RBF(         d, **dd) for _ in range(4) ]
            elif case == 'different_bases':
                kb_gpy = [ GPy.kern.RBF(d, **ddf(i)) for i in range(4) ]
                kb_kml = [ RBF(         d, **ddf(i)) for i in range(4) ]
            else:
                assert False

            # then combine the base kernels
            k_gpy = ((kb_gpy[0] * kb_gpy[1]) + kb_gpy[2]) * kb_gpy[3]
            k_kml = ((kb_kml[0] * kb_kml[1]) + kb_kml[2]) * kb_kml[3]

            # check to ensure they give the same kernel covariance matrix
            np.testing.assert_array_almost_equal( k_kml.cov(x), k_gpy.K(x) )
            
            # test training
            m = GPRegressionModel(x, y, k_kml)
            m.optimize( max_iters=5 )
            m.checkgrad()

    def test_wrapped_gpy_kernels (self):
        # generate data
        np.random.seed(0)
        d = 5
        N = 100
        x = np.random.uniform(size=(N,d)) # generate dataset
        y = rastrigin((x*2-1)*5.12)

        def ddf (i):
            t = 0.5*i+0.5
            return {'lengthscale':t, 'variance':t, 'name':'k%d' % i}

        # get base kernels
        kb_gpy = [ GPy.kern.RBF(d, **ddf(i)) for i in range(4) ]
        k_gpy  = ((kb_gpy[0] * kb_gpy[1]) + kb_gpy[2]) * kb_gpy[3]
        # Wrap the GPy kernel
        k_kml = GPyKernel(d, kernel=k_gpy)

        # check to ensure they give the same kernel covariance matrix
        np.testing.assert_array_almost_equal( k_kml.cov(x), k_gpy.K(x) )

        # test training
        m = GPRegressionModel(x, y, k_kml)
        m.optimize(max_iters=5)
        m.checkgrad()