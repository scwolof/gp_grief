
import numpy as np
import GPy.kern
from gp_grief.kern import BaseKernel


class GPyKernel (BaseKernel):
    def __init__ (self, n_dims, kernel=None, name=None, **kwargs):
        """
        Use kernel from the GPy library

        Inputs:
            n_dims : int
                number of input dimensions
            kernel : GPy.kern.Kern
                GPy kernel object
        """
        assert isinstance(kernel, GPy.kern.Kern) # check if its a GPy object
        if name is None:
            name = "GPy - " + repr(kernel)
        d = {'n_dims':n_dims, 'active_dims':None, 'name':name}
        super(GPyKernel, self).__init__(**d)
        self.kern = kernel
        # Constrain parameters
        self.constraint_list = [ ['+ve',]*np.size(param.values) for param in \
                                            self.kern.flattened_parameters ]

    def K (self, x, z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        K = self.kern.K(x, z)
        K = self._apply_children(K, x, z)
        return K

    @property
    def parameters (self):
        """
        Returns the kernel parameters as a 1d array
        """
        # first get the parent's parameters
        params = [np.ravel(p.values) for p in self.kern.flattened_parameters]

        # now add the children's parameters
        params += [child.parameters for _,child in self._children]

        # now concatenate into an array
        if len(params) > 0:
            params = np.concatenate(params, axis=0)
        else:
            params = np.array([])
        return params

    @parameters.setter
    def parameters (self, value):
        """
        Setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d
        i0 = 0 # counter of current position in value

        # set the parent's parameters
        for ip in range(np.size(self.kern.flattened_parameters)):
            old = self.kern.flattened_parameters[ip]
            try:
                self.kern.flattened_parameters[ip][:] = \
                            value[i0:i0+np.size(old)].reshape(np.shape(old))
            except:
                raise
            i0 += np.size(old) # increment counter

        # set the children's parameters
        for _,child in self._children:
            old = getattr(child, 'parameters') # old value
            setattr(child, 'parameters', \
                            value[i0:i0+np.size(old)].reshape(np.shape(old)))
            i0 += np.size(old) # increment counter

    @property
    def constraints (self):
        """ 
        Get constraints. over ride of inherited property 
        """
        # first get the parent's constraints
        constraints = [np.ravel(c) for c in self.constraint_list]

        # now add the children's constraints
        constraints += [child.constraints for _,child in self._children]

        # now concatenate into an array
        if len(constraints) > 0:
            constraints = np.concatenate(constraints, axis=0)
        else:
            constraints = np.array([])
        return constraints


    def fix_variance (self):
        """ 
        Apply fixed constraint to the variance 
        """
        # look for the index of each occurance of variance
        i_var = np.where(['variance' in param._name.lower() for param in \
                            self.kern.flattened_parameters])[0]
        # check if none or multiple found
        if np.size(i_var) == 0:
            raise RuntimeError("No variance parameter found")
        # constrain it
        self.constraint_list[i_var[0]][0] = 'fixed'

