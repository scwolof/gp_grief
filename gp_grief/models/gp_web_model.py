
from ..kern import WEBKernel
from gp_grief.models import BaseModel

# numpy/scipy stuff 
import numpy as np
from scipy.linalg import cho_factor, cho_solve

# development stuff
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)


class GPwebModel (BaseModel):
    """ 
    GP with weighted basis function kernel (WEB) with O(p^3) computations
    """

    def __init__(self, Phi, y, noise_var=1.):
        # call init of the super method
        super(GPwebModel, self).__init__()

        # initialize counts and stuff
        self.n = y.shape[0]
        y      = y.reshape((self.n,1))
        assert Phi.shape[0] == self.n
        self.p = Phi.shape[1]

        # precompute some stuff
        self.r   = Phi.T.dot(y)
        self.yTy = y.T.dot(y)
        self.A   = Phi.T.dot(Phi)

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # initialize the weights
        self.kern        = WEBKernel(initial_weights=np.ones(self.p))
        self.grad_method = 'adjoint' #  'finite_difference'

        # add to dependent attributes
        self.dependent_attributes = np.unique(np.concatenate(
            (self.dependent_attributes,
             [
                 '_P', '_Pchol', '_Pinv_r',
                 '_alpha_p', # precomputed W * Phi.T * alpha
             ])))


    def _compute_log_likelihood(self, parameters):
        # unpack the parameters and initialize some stuff
        self.parameters = parameters # set the internal state

        # precompute some stuff if not already done
        if self._P is None: # compute the P matrix and factorize
            w            = self.kern.parameters
            self._P      = self.A + np.diag(self.noise_var/w)
            self._Pchol  = cho_factor(self._P)
            self._Pinv_r = cho_solve(self._Pchol,self.r)

        # compute the log likelihood
        w              = self.kern.parameters
        sig2           = self.noise_var
        datafit        = (self.yTy - self.r.T.dot(self._Pinv_r)) / sig2
        complexity     = 2. * np.sum(np.log(np.diag(self._Pchol[0])))\
                         + np.sum(np.log(w)) + float(self.n-self.p)*np.log(sig2)
        log_likelihood = -0.5 * (complexity + datafit + self.n*np.log(2.*np.pi))
        return log_likelihood.squeeze()


    def _adjoint_gradient(self,parameters):
        """
        compute the log likelihood and the gradient wrt the hyperparameters
        using the adjoint method
        """
        assert isinstance(parameters,np.ndarray)

        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]
        gradient  = np.zeros(parameters.shape) + np.nan # initialize this

        # Compute log like at current point. Internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # precompute terms
        Pinv_A = cho_solve(self._Pchol, self.A)
        w      = self.kern.parameters
        sig2   = self.noise_var

        # get the gradients wrt the eigenfunction weights
        data_fit_grad   = -np.power((self.r-self.A.dot(self._Pinv_r))/sig2, 2)
        complexity_grad = (self.A.diagonal()-(self.A * Pinv_A).sum(axis=0))/sig2
        # place the gradients in the vector (it goes last in the list)
        gradient[1:]    = -0.5*data_fit_grad.squeeze()\
                          - 0.5*complexity_grad.squeeze()

        # get the noise var gradient
        data_fit_grad   = -(self.yTy - 2. * self.r.T.dot(self._Pinv_r)\
                        + self._Pinv_r.T.dot(self.A.dot(self._Pinv_r)))/(sig2**2)
        complexity_grad = (float(self.n) - np.trace(Pinv_A))/sig2
        gradient[0]     = -0.5*(data_fit_grad.squeeze()+complexity_grad.squeeze())

        # check to make sure no gradient was missed
        assert not np.any(np.isnan(gradient[free_inds])), "gradient missed!"
        return log_like, gradient


    def predict(self, Phi_new):
        """
        make predictions at new points
        """
        logger.debug('Predicting model at new points.')
        assert Phi_new.ndim == 2
        assert Phi_new.shape[1] == self.p
        parameters = self.parameters # ensure internal state is consistent!
        if self._alpha_p is None: # compute and save for future use
            if self._Pinv_r is None: # compute LML to compute this stuff
                self._compute_log_likelihood(parameters);
            w = self.kern.parameters.reshape((-1,1))
            self._alpha_p = (self.r-self.A.dot(self._Pinv_r))*w/self.noise_var

        # perform the posterior mean computation
        Yhat    = Phi_new.dot(self._alpha_p)

        # predict the variance at the test points
        Yhatvar = self.noise_var*Phi_new.dot(cho_solve(self._Pchol, Phi_new.T))\
                  + self.noise_var * np.eye(Phi_new.shape[0]) # see 2.11 of GPML
        return Yhat,Yhatvar

