
from ..linalg import solver_counter, LogexpTransformation

# numpy/scipy stuff 
import numpy as np
from numpy.linalg.linalg import LinAlgError
from numpy.testing import assert_array_almost_equal
from scipy.optimize import fmin_l_bfgs_b


class BaseModel (object):
    # for pos. or neg. constrained problems, as close as it can get to zero
    # by default this is not used
    param_shift = {'+ve':1e-200, '-ve':-1e-200} 
    _transformations = {'+ve':LogexpTransformation()}

    def __init__(self):
        """ initialize a few instance variables """
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


    def optimize(self, max_iters=1e3, messages=False, use_counter=False,\
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
        # setup the optimization
        x0 = self._transform_parameters(self.parameters)
        assert np.all(np.isfinite(x0))

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
            if self._counter is not None and self._counter.backup is not None:
                self.parameters = self._counter.backup[1]
        else:
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
            if raise_if_fails:
                raise
            else:
                return False
        else:
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
            # transform the gradient 
            gradient = self._transform_gradient(self.parameters, gradient)
        except (LinAlgError, ZeroDivisionError, ValueError):
            print('numerical issue computing log-likelihood or gradient')
            print('Model where failure occured:\n' + self.__str__())
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
        return transformed_grads


    def _untransform_parameters(self, transformed_parameters):
        """ 
        applies a reverse transformation to the parameters given constraints
        """
        assert transformed_parameters.size == np.size(self.constraints)
        parameters = np.zeros(transformed_parameters.size)
        ziplist    = zip(transformed_parameters,self.constraints)
        for i,(t_param,constraint) in enumerate(ziplist):
            if constraint is None or constraint == 'fixed' or constraint == '':
                parameters[i] = t_param
            else:
                parameters[i] = \
                    self._transformations[constraint].inverse_transform(t_param)\
                    + self.param_shift[constraint]
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


    def __str__(self):
        from tabulate import tabulate
        s = '\n%s Model\n' % self.__class__.__name__

        # print the  noise_var stuff
        s += str(tabulate([['noise_var',self.noise_var,self.noise_var_constraint]],
                headers=['Name', 'Value', 'Constraint'], tablefmt='orgtbl'))+'\n'

        # print the kernel stuff
        s += str(self.kern)
        return s
