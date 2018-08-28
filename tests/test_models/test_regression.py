
import pytest
import numpy as np
import warnings
from numpy.testing import assert_array_almost_equal

import GPy
from gp_grief.kern import RBF
from gp_grief.models import GPRegressionModel


def rastrigin(x, lin_term=None):
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


class TestRegression:

    def test_gp_grief_model_all_active (self):
        for d in [1, 10]:
            # generate data
            np.random.seed(0)
            N  = 100
            x  = np.random.uniform(size=(N,d)) # generate dataset
            y  = rastrigin((x*2-1)*5.12, lin_term=1.)
            xx = np.random.uniform(size=(N+1,d)) # generate test set

            # initialize GPy.models.GPRegression
            kern = GPy.kern.RBF(d)
            #with pytest.warns(UserWarning):
            mg   = GPy.models.GPRegression(x, y, kern)
            #print('hej2')
            #warnings.warn("my warning", UserWarning)
            # initialize gp_grief model
            kern = RBF(d)
            m    = GPRegressionModel(x, y, kern)
            
            m.checkgrad() 
            
            ll1, ll2 = mg.log_likelihood(), m.log_likelihood()
            assert_array_almost_equal(ll1, ll2, decimal=3) 
            
            m.fit()
            alpha_gpy = mg.posterior.woodbury_vector
            assert_array_almost_equal(alpha_gpy, m._alpha, decimal=3)
            
            yyh     = m.predict(xx, compute_var='full')
            yyh_gpy = mg.predict(xx,full_cov=True)
            assert_array_almost_equal(yyh_gpy[0], yyh[0], decimal=3)
            assert_array_almost_equal(yyh_gpy[1], yyh[1], decimal=2)
            
            m.optimize()

    def test_gp_grief_model_some_active (self):
        d     = 10
        adims = [0,2,4,6,7]

        # generate data
        np.random.seed(0)
        N  = 100
        x  = np.random.uniform(size=(N,d)) # generate dataset
        y  = rastrigin((x*2-1)*5.12, lin_term=1.)
        xx = np.random.uniform(size=(N+1,d)) # generate test set

        # initialize GPy.models.GPRegression
        kern = GPy.kern.RBF(len(adims), active_dims=adims)
        with pytest.warns(UserWarning):
            mg = GPy.models.GPRegression(x, y, kern)
        # initialize gp_grief model
        kern = RBF(d, active_dims=adims)
        m    = GPRegressionModel(x, y, kern)
        
        m.checkgrad() 
        
        ll1, ll2 = mg.log_likelihood(), m.log_likelihood()
        assert_array_almost_equal(ll1, ll2, decimal=3) 
        
        m.fit()
        alpha_gpy = mg.posterior.woodbury_vector
        assert_array_almost_equal(alpha_gpy, m._alpha, decimal=3)
        
        yyh     = m.predict(xx, compute_var='full')
        yyh_gpy = mg.predict(xx,full_cov=True)
        assert_array_almost_equal(yyh_gpy[0], yyh[0], decimal=3)
        assert_array_almost_equal(yyh_gpy[1], yyh[1], decimal=2)
        
        m.optimize()

    def test_combining_kernels (self):
        # generate data
        np.random.seed(1)
        N = 100
        d = 5
        x = np.random.uniform(size=(N,d)) # generate dataset
        y = rastrigin((x*2-1)*5.12, lin_term=1.)
        xx = np.random.uniform(size=(N+1,d)) # generate test set
            
        def dd (i): 
            l = 0.5*i+0.5
            return {'lengthscale':l, 'variance':l, 'name':'k%d'%i} 
        # first define the base kernels
        kb_gpy = [GPy.kern.RBF(d, **dd(i)) for i in range(4)]
        kb_kml = [RBF(         d, **dd(i)) for i in range(4)]

        # then combine the base kernels
        k_gpy = ((kb_gpy[0] * kb_gpy[1]) + kb_gpy[2]) * kb_gpy[3]
        k_kml = ((kb_kml[0] * kb_kml[1]) + kb_kml[2]) * kb_kml[3]
        assert_array_almost_equal( k_kml.cov(x), k_gpy.K(x) )

        # construct the models
        m = dict()
        m['gpy'] = GPy.models.GPRegression(x, y, k_gpy)
        m['kml'] = GPRegressionModel(x, y, k_kml)
        m['gpy'].mul.sum.mul.k1.variance.fix()
        m['gpy'].mul.k3.variance.fix()

        assert m['kml'].checkgrad()

        ll1, ll2 = m['gpy'].log_likelihood(), m['kml'].log_likelihood()
        assert_array_almost_equal(ll1, ll2, decimal=3) 

        yyh = dict()
        yyh['gpy'] = m['gpy'].predict(xx, full_cov=True)
        yyh['kml'] = m['kml'].predict(xx, compute_var='full')
        assert_array_almost_equal(*[yyh[key][0] for key in m], decimal=2) 
        assert_array_almost_equal(*[yyh[key][1] for key in m], decimal=2) 

        wv1, wv2 = m['gpy'].posterior.woodbury_vector, m['kml']._alpha
        assert_array_almost_equal(wv1, wv2, decimal=3)

        with pytest.warns(RuntimeWarning):
            m['gpy'].optimize()
        m['kml'].optimize()
        ll1, ll2 = m['gpy'].log_likelihood(), m['kml'].log_likelihood()
        assert_array_almost_equal(ll1, ll2, decimal=-1) 

T = TestRegression()
T.test_gp_grief_model_some_active()