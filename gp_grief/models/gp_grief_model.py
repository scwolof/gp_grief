
from ..kern import BaseKernel, GriefKernel
from gp_grief.models import BaseModel

# numpy/scipy stuff 
import numpy as np
from scipy.linalg import cho_factor,cho_solve

# development stuff
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)


class GPGriefModel (BaseModel):
    """ 
    GP-GRIEF (GP with GRId-structured Eigen Functions) 
    """

    def __init__(self, X, Y, kern, noise_var=1.):
        """
        GP-GRIEF (GP with GRId-structured Eigen Functions)

        Inputs:
        """
        # call init of the super method
        super(GPGriefModel, self).__init__()
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

        # check the kernel
        assert isinstance(kern, GriefKernel)
        assert np.ndim(kern.kern_list) == 1 # can only be a 1d array of objects
        for i,ki in enumerate(kern.kern_list):
            assert isinstance(ki, BaseKernel) # ensure it is a kernel
            assert ki.n_dims == 1, "currently only 1-dimensional grids allowed"
        self.kern = kern

        # set noise_var internally
        self.noise_var = np.float64(noise_var)

        # add to dependent attributes
        self.dependent_attributes = np.unique(np.concatenate(
            (self.dependent_attributes,
             [
                 '_P', '_Pchol',
                 '_alpha_p', # precomputed W * Phi.T * alpha
             ])))
        if self.kern.opt_kernel_params:
            self.dependent_attributes = np.unique(np.concatenate(
                (self.dependent_attributes,
                 [
                     '_A', '_Phi', # change when base kernel hyperparams change
                     '_X_last_pred', '_Phi_last_pred', # saved from last
                 ])))
        else:
            self._A = None
            self._Phi_last_pred = None

        # set some other default stuff
        if self.kern.opt_kernel_params:
            self.grad_method = 'finite_difference'
        else:
            self.grad_method = ['adjoint', 'finite_difference'][0]
        return


    def fit(self, **kwargs):
        """
        finds the weight vector alpha
        """
        # compute using the matrix inversion lemma
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is not None: # check if alpha is already computed
            return
        self._cov_setup()
        self._alpha = self._mv_cov_inv(self.Y)

    def predict_precompute (self, Xnew):
        logger.debug('Predicting model at new points.')
        assert Xnew.ndim == 2
        assert Xnew.shape[1] == self.input_dim
        self.parameters # ensure that the internal state is consistent!
        if self._alpha is None: # then I need to train
            self.fit()
        if self._alpha_p is None: # compute so future calcs can be done in O(p)
            self._alpha_p = self._Phi.T.dot(self._alpha)*self.kern.w.reshape((-1,1))

    def predict (self, Xnew):
        """
        make predictions at new points

        Inputs:
            Xnew : (M,d) numpy array of points to predict at

        Outputs:
            Yhat : (M,1) numpy array predictions at Xnew
            Yhatvar : only returned if compute_var is not None. See compute_var
        """
        self.predict_precompute(Xnew)

        # get cross covariance between training and testing points
        if self._Phi_last_pred is None or not np.array_equal(Xnew,self._X_last_pred):
            logger.debug("computing Phi at new prediction points")
            self._Phi_last_pred = self.kern.cov(x=Xnew)[0]
            self._X_last_pred = Xnew

        # predict the mean at the test points
        Yhat = self._Phi_last_pred.dot(self._alpha_p)

        # predict the variance at the test points
        Yhatvar = self.noise_var * self._Phi_last_pred.dot(\
                            cho_solve(self._Pchol, self._Phi_last_pred.T))\
                  + self.noise_var * np.eye(Xnew.shape[0]) # see 2.11 of GPML
        return Yhat, Yhatvar

    def d_Yhat_d_x (self, Xnew, dim):
        """
        Computes d Yhat / d x{:,dim}
        """
        self.predict_precompute(Xnew)
        dPhi = self.kern.cov_grad(Xnew, dim)
        # predict the mean at the test points
        return dPhi.dot(self._alpha_p)


    def _cov_setup(self):
        """
        setup the covariance matrix
        """
        if self._P is not None: # then already computed so return
            return
        # get the weights
        self._w = self.kern.w

        # get the p x p matrix A if ness
        if self._A is None: # then compute, note this is expensive
            self._Phi = self.kern.cov(self.X)[0]
            self._A = self._Phi.T.dot(self._Phi) # O(np^2) operation!

        # compute the P matrix and factorize
        self._P = self._A + np.diag(self.noise_var/self._w)
        self._Pchol = cho_factor(self._P)


    def _adjoint_gradient(self,parameters):
        """ 
        compute the log likelihood and the gradient wrt the hyperparameters 
        using the adjoint method 
        """
        assert isinstance(parameters,np.ndarray)

        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]
        gradient = np.zeros(parameters.shape) + np.nan # initialize this

        # Compute log like at current point. Internal state will be set here
        log_like = self._compute_log_likelihood(parameters)

        # get gradients wrt eigenfunction weights
        if self.kern.reweight_eig_funs:
            # compute the data fit gradient
            data_fit_grad = 0.5*np.power(self._Phi.T.dot(self._alpha),2).squeeze()
            # compute the complexity term gradient
            Pinv_A = cho_solve(self._Pchol, self._A)
            complexity_grad = -0.5*(self._A.diagonal() -\
                            (self._A * Pinv_A).sum(axis=0))/(self.noise_var)
            # place the gradients in the vector (it goes last in the list)
            gradient[-self.kern.n_eigs:] = data_fit_grad.squeeze()\
                                           + complexity_grad.squeeze()
        else:
            Pinv_A = None # specify that this hasn't been computed.

        # get the noise var gradient
        if self.noise_var_constraint != 'fixed':
            if Pinv_A is None:
                Pinv_A = cho_solve(self._Pchol, self._A)
            data_fit_grad = 0.5*(self._alpha.T.dot(self._alpha))
            complexity_grad = -0.5*(float(self.num_data)\
                                - np.trace(Pinv_A))/self.noise_var
            gradient[0] = data_fit_grad.squeeze() + complexity_grad.squeeze()

        # compute the gradient with respect to the kernel parameters
        if self.kern.opt_kernel_params:
            raise NotImplementedError("adjoint method not implemented for"\
                            +"kernel parameter optimiszation, just weights")

        # check to make sure not gradient was missed
        assert not np.any(np.isnan(gradient[free_inds])), "gradient missed!"
        return log_like, gradient


    def _compute_log_likelihood(self, parameters):
        """
        compute log likelihood
        """
        # unpack the parameters
        self.parameters = parameters # set the internal state
        # fit the model 
        self.fit()
        # compute the log likelihood
        log_like = -0.5 * (self.Y.T.dot(self._alpha) + self._cov_log_det()\
                            + self.num_data*np.log(np.pi*2))
        return log_like


    def _mv_cov(self, x):
        """ 
        matrix vector product with shifted covariance matrix. 
        To get full cov, do `_mv_cov(np.identity(n))` 
        """
        assert x.shape[0] == self.num_data
        assert self._Phi is not None, "cov has not been setup"
        return self._Phi.dot(self._Phi.T.dot(x) * self._w.reshape((-1,1)))\
                + x * self.noise_var


    def _mv_cov_inv(self, x):
        """ 
        matrix vector product with shifted covariance inverse 
        """
        assert x.shape[0] == self.num_data
        assert self._Pchol is not None, "cov has not been setup"
        return (x - self._Phi.dot(cho_solve(self._Pchol, self._Phi.T.dot(x))))\
                /self.noise_var


    def _cov_log_det(self):
        """ 
        compute covariance log determinant 
        """
        assert self._Pchol is not None, "cov has not been setup"
        return 2.*np.sum(np.log(np.diag(self._Pchol[0])))\
            + np.sum(np.log(self._w))\
            + float(self.num_data-self.kern.n_eigs) * np.log(self.noise_var)
