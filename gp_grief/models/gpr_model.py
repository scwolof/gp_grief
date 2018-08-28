
from ..linalg import solve_schur, solve_chol 
from gp_grief.kern import BaseKernel
from gp_grief.models import BaseModel

# numpy/scipy stuff
import numpy as np

# development stuff
from logging import getLogger
logger = getLogger(__name__)


class GPRegressionModel (BaseModel):
    """
    general GP regression model
    """

    def __init__(self, X, Y, kernel, noise_var=1.):
        # call init of the super method
        super(GPRegressionModel, self).__init__()
        # check inputs
        assert X.ndim == 2
        assert Y.ndim == 2
        self.X = np.asarray(X)
        self.Y = np.asarray(Y)
        assert not np.any(np.isnan(Y))
        self.num_data, self.input_dim = self.X.shape
        if Y.shape[0] != self.num_data:
            raise ValueError('X and Y sizes are inconsistent')
        self.output_dim = self.Y.shape[1]
        if self.output_dim != 1:
            raise RuntimeError('this only deals with 1 response for now')
        assert isinstance(kernel, BaseKernel)
        self.kern = kernel

        # add the noise_var internally
        self.noise_var = np.float64(noise_var)

        # set some defaults
        self.grad_method = 'finite_difference chol'
        return


    def fit(self):
        """ finds the weight vector alpha """
        logger.debug('Fitting; determining weight vector.')
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then need to find the new alpha
            self._alpha = np.linalg.solve(self.kern.cov(x=self.X) + \
                                self.noise_var*np.eye(self.num_data), self.Y)


    def predict(self,Xnew,compute_var=None):
        """
        make predictions at new points
        Inputs:
            Xnew : (M,d) numpy array of points to predict at
            compute_var : whether to compute the variance at the test points
                * None (default) : don't compute variance
                * 'diag' : return diagonal of the covariance matrix, size (M,1)
                * 'full' : return the full covariance matrix of size (M,M)
        Outputs:
            Yhat : (M,1) numpy array predictions at Xnew
            Yhatvar : only returned if compute_var is not None. See compute_var
        """
        logger.debug('Predicting model at new points.')
        assert Xnew.ndim == 2
        assert Xnew.shape[1] == self.input_dim
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then I need to train
            self.fit()

        # get cross covariance between training and testing points
        Khat = self.kern.cov(x=Xnew, z=self.X)

        # predict the mean at the test points
        Yhat = Khat.dot(self._alpha)

        # predict the variance at the test points
        # TODO: make this more efficient, especially for diagonal predictions
        if compute_var is not None:
            Yhatvar = self.kern.cov(x=Xnew)+self.noise_var*np.eye(Xnew.shape[0])\
                    - Khat.dot(np.linalg.solve(self.kern.cov(x=self.X) + \
                                self.noise_var*np.eye(self.num_data), Khat.T))
            if compute_var == 'diag':
                Yhatvar = Yhatvar.diag().reshape((-1,1))
            elif compute_var != 'full':
                raise ValueError('Unknown compute_var = %s' % repr(compute_var))
            return Yhat,Yhatvar
        else: # just return the mean
            return Yhat


    def _compute_log_likelihood(self, parameters):
        """
        helper function to compute log likelihood
        Inputs:
            parameters : 1d array
                whose first element is the gaussian noise and the other elements 
                are the kernel parameters
        Outputs:
            log_likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state

        # compute the new covariance
        K = self.kern.cov(self.X)

        # compute the log likelihood
        if 'svd' in self.grad_method: # then compute using svd
            (Q,eig_vals) = np.linalg.svd(K, full_matrices=0, compute_uv=1)[:2]
            log_like = -0.5*np.sum(np.log(eig_vals+self.noise_var))\
                        - 0.5*np.dot(self.Y.T, solve_schur(\
                                    Q,eig_vals,self.Y,shift=self.noise_var))\
                        - 0.5*self.num_data*np.log(np.pi*2)
        if 'chol' in self.grad_method: # then compute using cholesky factorization
            U = np.linalg.cholesky(K + self.noise_var * np.eye(self.num_data)).T
            log_like = -np.sum(np.log(np.diagonal(U,offset=0,axis1=-1,axis2=-2)))\
                        - 0.5*np.dot(self.Y.T, solve_chol(U,self.Y))\
                        - 0.5*self.num_data*np.log(np.pi*2)
        else: # just use logpdf from scipy 
            log_like = mvn.logpdf(self.Y.squeeze(),
                                  np.zeros(self.num_data),
                                  K + self.noise_var * np.eye(self.num_data))
        return log_like
