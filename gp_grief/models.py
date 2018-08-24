from .kern import BaseKernel, GridKernel, GriefKernel, WEBKernel
from .linalg import solve_schur,solve_chol,solver_counter,LogexpTransformation
from .grid import InducingGrid
from .stats import norm, lognorm, RandomVariable, StreamMeanVar

# numpy/scipy stuff 
import numpy as np
from scipy.linalg import cho_factor,cho_solve
from scipy.optimize import fmin_l_bfgs_b

# development stuff
from numpy.linalg.linalg import LinAlgError
from numpy.testing import assert_array_almost_equal
from traceback import format_exc
from pdb import set_trace
from logging import getLogger
from warnings import warn
logger = getLogger(__name__)

class BaseModel(object):
    # for pos. or neg. constrained problems, as close as it can get to zero
    # by default this is not used
    param_shift = {'+ve':1e-200, '-ve':-1e-200} 
    _transformations = {'+ve':LogexpTransformation()}

    def __init__(self):
        """ initialize a few instance variables """
        logger.debug('Initializing %s model.' % self.__class__.__name__)
        self.dependent_attributes = ['_alpha',
                                     '_log_like',
                                     '_gradient','_K',
                                     '_log_det']
        self._previous_parameters = None # previous parameters from last call
        self.grad_method = None # could be {'finite_difference','adjoint'}
        self.noise_var_constraint = '+ve' # Gaussian noise variance constraint
        return


    def log_likelihood(self, return_gradient=False):
        """
        computes the log likelihood and the gradient (if not already computed).

        If return_gradient then gradient will be returned as second arguement.
        """
        p = self.parameters # this has to be called first

        # check if I need to recompute anything
        if return_gradient and (self._gradient is None):
            # compute the log likelihood and gradient wrt the parameters
            if 'adjoint' in self.grad_method:
                (self._log_like, self._gradient) = self._adjoint_gradient(p)
            elif 'finite_difference' in self.grad_method:
                (self._log_like, self._gradient) = self._finite_diff_gradient(p)
            else:
                raise RuntimeError('unknown grad_method %s' % repr(self.grad_method))
        elif self._log_like is None: # compute the log-likelihood without gradient
            self._log_like = self._compute_log_likelihood(p)
        else: # everything is already computed
            pass

        if return_gradient: # return both
            return self._log_like, self._gradient
        else: # just return likelihood
            return self._log_like


    def optimize(self, max_iters=1e3, messages=True, use_counter=False,\
                factr=1e7, pgtol=1e-05):
        """
        maximize the log likelihood

        Inputs:
            max_iters : int
                maximum number of optimization iterations
            factr, pgtol : lbfgsb convergence criteria, see fmin_l_bfgs_b help
                use factr of 1e12 for low accuracy, 10 for extremely high accuracy 
                (default 1e7)
        """
        logger.debug('Beginning MLE to optimize hyperparams. grad_method=%s'\
                    % self.grad_method)

        # setup the optimization
        try:
            x0 = self._transform_parameters(self.parameters)
            assert np.all(np.isfinite(x0))
        except:
            logger.error('Transformation failed for initial values. '\
                + 'Ensure constraints are met or the value is not too small.')
            raise

        # filter out the fixed parameters
        free = np.logical_not(self._fixed_indicies)
        x0 = x0[free]

        # setup the counter
        if use_counter:
            self._counter = solver_counter(disp=True)
        else:
            self._counter = None

        # run the optimization
        try:
            x_opt, f_opt, opt = fmin_l_bfgs_b(func=self._objective_grad, x0=x0,\
                factr=factr, pgtol=pgtol, maxiter=max_iters, disp=messages)
        except (KeyboardInterrupt,IndexError):
            logger.info('Keyboard interrupt raised. Cleaning up...')
            if self._counter is not None and self._counter.backup is not None:
                self.parameters = self._counter.backup[1]
                logger.info('will return best parameter set with'\
                    + 'log-likelihood = %.4g' % self._counter.backup[0])
        else:
            logger.info('Function Evals: %d. Exit status: %s' % (f_opt, opt['warnflag']))
            # extract the optimal value and set the parameters to this
            transformed_parameters = self._previous_parameters 
            transformed_parameters[free] = x_opt
            self.parameters = self._untransform_parameters(transformed_parameters)
        return opt


    def checkgrad(self, decimal=3, raise_if_fails=True):
        """
        checks the gradient and raises if does not pass
        """
        grad_exact = self._finite_diff_gradient(self.parameters)[1]
        grad_exact[self._fixed_indicies] = 1 # gradients of fixed variables
        grad_analytic = self.log_likelihood(return_gradient=True)[1]
        grad_analytic[self._fixed_indicies] = 1 # gradients of fixed variables

        # first protect from nan values incase both analytic and exact are small
        protected_nan = np.logical_and(np.abs(grad_exact) < 1e-8, \
                                       np.abs(grad_analytic) < 1e-8)

        # protect against division by zero
        protected_div0 = np.abs(grad_exact-grad_analytic) < 1e-5
        # artificially change these protected values
        grad_exact[np.logical_or(protected_nan, protected_div0)] = 1.
        grad_analytic[np.logical_or(protected_nan, protected_div0)] = 1.

        try:
            assert_array_almost_equal(grad_exact / grad_analytic,\
                                      np.ones(grad_exact.shape),decimal=decimal)
        except:
            logger.info('Gradient check failed.')
            logger.debug('[[Finite-Diff Gradient], [Analytic Gradient]]:\n%s\n'\
                % repr(np.asarray([grad_exact,grad_analytic])))
            if raise_if_fails:
                raise
            else:
                logger.info(format_exc()) # print the output
                return False
        else:
            logger.info('Gradient check passed.')
            return True

    @property
    def parameters(self):
        """
        this gets the parameters from the object attributes
        """
        parameters = np.concatenate( (np.ravel(self.noise_var),\
                                      self.kern.parameters), axis=0)

        # check if the parameters have changed
        if not np.array_equal(parameters, self._previous_parameters):
            # remove the internal variables that rely on the parameters
            for attr in self.dependent_attributes:
                setattr(self, attr, None)
            # update the previous parameter array
            self._previous_parameters = parameters.copy()
        return parameters.copy()

    @parameters.setter
    def parameters(self,parameters):
        """
        takes optimization variable parameters and sets the internal state of
        self to make it consistent with the variables
        """
        # set the parameters internally
        self.noise_var       = parameters[0]
        self.kern.parameters = parameters[1:]

        # check if the parameters have changed
        if not np.array_equal(parameters, self._previous_parameters):
            # remove the internal variables that rely on the parameters
            for attr in self.dependent_attributes:
                setattr(self, attr, None)
            # update the previous parameter array
            self._previous_parameters = parameters.copy()
        return parameters

    @property
    def constraints(self):
        """ returns the model parameter constraints as a list """
        constraints = np.concatenate( (np.ravel(self.noise_var_constraint), 
                                       self.kern.constraints), axis=0)
        return constraints


    def predict(self, Xnew, compute_var=None):
        """
        make predictions at new points
        MUST begin with call to parameters property
        """
        raise NotImplementedError('')


    def fit(self):
        """
        determines the weight vector _alpha
        MUST begin with a call to parameters property
        """
        raise NotImplementedError('')


    def _objective_grad(self, transformed_free_parameters):
        """ 
        determines the objective and gradients in the transformed input space 
        """
        # get the fixed indices and add to the transformed parameters
        free = np.logical_not(self._fixed_indicies)
        transformed_parameters = self._previous_parameters
        transformed_parameters[free] = transformed_free_parameters
        try:
            # untransform and internalize parameters
            self.parameters = self._untransform_parameters(transformed_parameters)
            # compute objective and gradient in untransformed space
            (objective, gradient) = self.log_likelihood(return_gradient=True)
            objective = -objective # since we want to minimize
            gradient =  -gradient
            # ensure the values are finite
            if not np.isfinite(objective):
                logger.debug('objective is not finite')
            if not np.all(np.isfinite(gradient[free])):
                logger.debug('some derivatives are non-finite')
            # transform the gradient 
            gradient = self._transform_gradient(self.parameters, gradient)
        except (LinAlgError, ZeroDivisionError, ValueError):
            logger.error('numerical issue computing log-likelihood or gradient')
            logger.debug('Model where failure occured:\n' + self.__str__())
            raise
        # get rid of the gradients of the fixed parameters
        free_gradient = gradient[free]

        # call the counter if ness
        if self._counter is not None:
            msg='log-likelihood=%.4g, gradient_norm=%.2g'\
                % (-objective, np.linalg.norm(gradient))
            if self._counter.backup is None or self._counter.backup[0]<-objective:
                self._counter(msg=msg,store=(-objective,self.parameters.copy()))
            else: # don't update backup
                self._counter(msg=msg)
        return objective, free_gradient

    @property
    def _fixed_indicies(self):
        """ 
        returns a bool array specifiying where the indicies are fixed 
        """
        fixed_inds = self.constraints == 'fixed'
        return fixed_inds

    @property
    def _free_indicies(self):
        """ 
        returns a bool array specifiying where the indicies are free 
        """
        return np.logical_not(self._fixed_indicies)


    def _transform_parameters(self, parameters):
        """
        applies a transformation to the parameters based on a constraint
        """
        constraints = self.constraints
        assert parameters.size == np.size(constraints) # check if sizes correct
        transformed_parameters = np.zeros(parameters.size)
        for i,(param,constraint) in enumerate(zip(parameters,constraints)):
            if constraint is None or constraint == 'fixed' or constraint == '':
                transformed_parameters[i] = param
            else: # I need to transform the parameters
                transformed_parameters[i] =\
                    self._transformations[constraint].transform(\
                                          param - self.param_shift[constraint])

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(transformed_parameters)):
            logger.debug('transformation led to non-finite value')
        return transformed_parameters


    def _transform_gradient(self, parameters, gradients):
        """
        see _transform parameters
        """
        constraints = self.constraints
        assert parameters.size == gradients.size == np.size(constraints)
        transformed_grads      = np.zeros(parameters.size)
        ziplist                = zip(parameters,gradients,constraints)
        for i,(param,grad,constraint) in enumerate(ziplist):
            if constraint is None or constraint == '': # no transformation
                transformed_grads[i] = grad
            elif constraint != 'fixed': # apply transformation
                transformed_grads[i] =\
                    self._transformations[constraint].transform_grad(\
                                    param - self.param_shift[constraint],grad)

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(transformed_grads)):
            logger.debug('transformation led to non-finite value')
        return transformed_grads


    def _untransform_parameters(self, transformed_parameters):
        """ 
        applies a reverse transformation to the parameters given constraints
        """
        assert transformed_parameters.size == np.size(self.constraints)
        parameters = np.zeros(transformed_parameters.size)
        zipist     = zip(transformed_parameters,self.constraints)
        for i,(t_param,constraint) in enumerate(ziplist):
            if constraint is None or constraint == 'fixed' or constraint == '':
                parameters[i] = t_param
            else:
                parameters[i] = \
                    self._transformations[constraint].inverse_transform(t_param)\
                    + self.param_shift[constraint]

        # check to ensure transformation led to finite value
        if not np.all(np.isfinite(parameters)):
            logger.debug('transformation led to non-finite value')
        return parameters


    def _finite_diff_gradient(self, parameters):
        """
        helper function to compute function gradients by finite difference.

        Inputs:
            parameters : 1d array
                whose first element is the Gaussian noise and the other 
                elements are the kernel parameters
            log_like : float
                log likelihood at the current point

        Outputs:
            log_likelihood
        """
        assert isinstance(parameters,np.ndarray)
        # get the free indicies
        free_inds = np.nonzero(np.logical_not(self._fixed_indicies))[0]

        # first take a forward step in each direction
        step = 1e-6 # finite difference step
        log_like_fs = np.zeros(free_inds.size)
        for i,param_idx in enumerate(free_inds):
            p_fs = parameters.copy()
            p_fs[param_idx] += step # take a step forward
            log_like_fs[i] = self._compute_log_likelihood(p_fs)

        # compute the log likelihood at current point
        log_like = self._compute_log_likelihood(parameters)

        # compute the gradient
        gradient = np.zeros(parameters.shape) # default gradient is zero
        gradient[free_inds] = (log_like_fs-log_like)
        gradient[free_inds] = gradient[free_inds]/step # divide by step length
        return log_like, gradient


    def _compute_log_likelihood(self, parameters):
        """
        helper function to compute log likelihood.
        Inputs:
            parameters : 1d array
                whose first element is the Gaussian noise and the other 
                elements are the kernel parameters

        Outputs:
            log_likelihood
        """
        raise NotImplementedError('')


    def _adjoint_gradient(self,parameters):
        raise NotImplementedError('')
        return log_like, gradient


    def __str__(self):
        from tabulate import tabulate
        s = '\n%s Model\n' % self.__class__.__name__

        # print the  noise_var stuff
        s += str(tabulate([['noise_var',self.noise_var,self.noise_var_constraint]],
                headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl'))+'\n'

        # print the kernel stuff
        s += str(self.kern)
        return s


class GPGrief(BaseModel):
    """ 
    GP-GRIEF (GP with GRId-structured Eigen Functions) 
    """

    def __init__(self, X, Y, kern, noise_var=1.):
        """
        GP-GRIEF (GP with GRId-structured Eigen Functions)

        Inputs:
        """
        # call init of the super method
        super(GPGrief, self).__init__()
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


    def predict(self, Xnew):
        """
        make predictions at new points

        Inputs:
            Xnew : (M,d) numpy array of points to predict at

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
        if self._alpha_p is None: # compute so future calcs can be done in O(p)
            self._alpha_p = self._Phi.T.dot(self._alpha)*self.kern.w.reshape((-1,1))

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
        return Yhat,Yhatvar


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


class GPweb(BaseModel):
    """ 
    GP with weighted basis function kernel (WEB) with O(p^3) computations
    """

    def __init__(self, Phi, y, noise_var=1.):
        # call init of the super method
        super(GPweb, self).__init__()

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


class GPweb_transformed(BaseModel):
    """ 
    GP with weighted basis function kernel (WEB) with basis functions
    transformed to give O(p) computations
    """

    def __init__(self, Phi, y, noise_var=1.):
        # call init of the super method
        super(GPweb_transformed, self).__init__()

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


