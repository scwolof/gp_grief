
from ..kern import WEBKernel
from gp_grief.models import BaseModel

# numpy/scipy stuff 
import numpy as np

# development stuff
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)


class GPwebTransformedModel (BaseModel):
    """ 
    GP with weighted basis function kernel (WEB) with basis functions
    transformed to give O(p) computations
    """

    def __init__(self, Phi, y, noise_var=1.):
        # call init of the super method
        super(GPwebTransformedModel, self).__init__()

        # initialize counts and stuff
        self.n = y.shape[0]
        y = y.reshape((self.n,1))
        assert Phi.shape[0] == self.n
        self.p_orig = Phi.shape[1]

        # factorize the Phi matrix and determine which transformed bases to keep
        Phit, self.singular_vals, VT = np.linalg.svd(\
                                    Phi, full_matrices=False, compute_uv=True)
        ikeep = self.singular_vals > 1e-7 # eliminate tiny singular values
        Phit = Phit[:,ikeep]
        self.singular_vals = self.singular_vals[ikeep]
        VT = VT[ikeep,:]
        self.V = VT.T
        self.p = Phit.shape[1]
        if self.p < self.p_orig:
            logger.info("Num Bases decreased from p=%d to p=%d."\
                +"Only a subspace can now be searched." % (self.p_orig, self.p))

        # precompute some other stuff
        self.PhitT_y = Phit.T.dot(y).squeeze()
        self.PhitT_y_2 = np.power(self.PhitT_y,2)
        self.yTy = np.power(y,2).sum()

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # initialize the weights
        self.kern = WEBKernel(initial_weights=np.ones(self.p))
        self.grad_method = 'adjoint' #  'finite_difference'


    def _compute_log_likelihood(self, parameters):
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # compute the log likelihood
        w              = self.kern.parameters
        sig2           = self.noise_var
        Pdiag          = sig2/w + 1.
        datafit        = (self.yTy - np.sum(self.PhitT_y_2/Pdiag))/sig2
        complexity     = np.sum(np.log(Pdiag)) + np.sum(np.log(w))\
                            + (self.n-self.p)*np.log(sig2)
        log_likelihood = -0.5 * (complexity + datafit + self.n*np.log(2.*np.pi))
        return log_likelihood


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
        log_like  = self._compute_log_likelihood(parameters)

        # get the gradients wrt the eigenfunction weights
        w    = self.kern.parameters
        sig2 = self.noise_var
        # compute the data fit gradient
        data_fit_grad   = -self.PhitT_y_2 / np.power(sig2 + w, 2)
        # compute the complexity term gradient
        complexity_grad = 1./(sig2 + w)
        # place the gradients in the vector (it goes last in the list)
        gradient[1:] = -0.5*(data_fit_grad.squeeze()+complexity_grad.squeeze())

        # get the noise var gradient (see notebook dec 23 & 28, 2017)
        data_fit_grad   = (-self.yTy + np.sum(self.PhitT_y_2 * w * \
                            (sig2*2. + w) / np.power(sig2 + w, 2))) / sig2**2
        complexity_grad = float(self.n-self.p)/sig2 + np.sum(1./(sig2+w))
        gradient[0]   = -0.5*(data_fit_grad.squeeze()+complexity_grad.squeeze())

        # check to make sure no gradient was missed
        assert not np.any(np.isnan(gradient[free_inds])), "gradient missed!"
        return log_like, gradient


    def predict(self, Phi_new):
        """
        make predictions at new points
        """
        logger.debug('Predicting model at new points.')
        assert Phi_new.ndim == 2
        assert Phi_new.shape[1] == self.p_orig
        self.parameters # ensure that the internal state is consistent!

        # perform the posterior mean computation (see june 4, 2018 notes)
        w = self.kern.parameters
        sig2 = self.noise_var
        Pdiag = sig2/w + 1.
        alpha_p = (self.PhitT_y - self.PhitT_y/Pdiag)  * w / sig2
        Yhat = Phi_new.dot(self.V.dot(alpha_p/self.singular_vals)).reshape((-1,1))

        # predict the variance at the test points
        Yhatvar = self.noise_var * Phi_new.dot(Phi_new.T/Pdiag.reshape((-1,1)))\
                  + self.noise_var * np.eye(Phi_new.shape[0]) # see 2.11 of GPML
        return Yhat,Yhatvar


