
import numpy as np


class BaseKernel (object):
    """ 
    Base class for all kernel functions 
    """

    def __init__ (self, n_dims, active_dims, name):
        self.n_dims = n_dims
        if active_dims is None: # then all dims are active
            active_dims = np.arange(self.n_dims)
        else: # active_dims has been specified
            active_dims = np.ravel(active_dims) # ensure 1d array
            assert 'int' in active_dims.dtype.type.__name__ # ensure it is an int array
            assert active_dims.min() >= 0 # less than zero is not a valid index
            assert active_dims.max() <  self.n_dims # max it can be is n_dims-1
        self.active_dims = active_dims
        if name is None: # then set a default name
            name = self.__class__.__name__
        self.name = name

        # initialize a few things that need to be implemented for new kernels
        self.parameter_list = None # list of parameter attribute names as strings
        self.constraint_map = None # dict with elements in parameter_list as keys
        self._children = [] # contains the children kernels (form __mul__ and __add__)


    def K (self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix
        """
        raise NotImplementedError

    @property
    def parameters (self):
        """
        returns the kernel parameters as a 1d array
        """
        # check if implemented
        if self.parameter_list is None:
            raise NotImplementedError('Need to specify kern.parameter_list')

        # first get the parent's parameters
        parameters = [np.ravel(getattr(self, name)) for name in self.parameter_list]

        # now add the children's parameters
        parameters += [child.parameters for _,child in self._children]

        # now concatenate into an array
        if len(parameters) > 0:
            parameters = np.concatenate(parameters, axis=0)
        else:
            parameters = np.array([])
        return parameters

    @parameters.setter
    def parameters (self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d
        i0 = 0 # counter of current position in value

        # set the parent's parameters
        for name in self.parameter_list:
            old = getattr(self, name) # old value
            setattr(self, name, value[i0:i0+np.size(old)].reshape(np.shape(old)))
            i0 += np.size(old) # increment counter

        # set the children's parameters
        for _,child in self._children:
            old = getattr(child, 'parameters') # old value
            setattr(child, 'parameters', value[i0:i0+np.size(old)].reshape(np.shape(old)))
            i0 += np.size(old) # increment counter

    @property
    def constraints (self):
        """ returns the constraints for all parameters """
        # check if implemented
        if self.constraint_map is None:
            raise NotImplementedError('Need to specify kern.constraint_map')

        # first get the parent's parameters
        constraints = [np.ravel(self.constraint_map[name]) for name in self.parameter_list]

        # now add the children's parameters
        constraints += [child.constraints for _,child in self._children]

        # now concatenate into an array
        if len(constraints) > 0:
            constraints = np.concatenate(constraints, axis=0)
        else:
            constraints = np.array([])
        return constraints


    def is_stationary (self):
        """ 
        check if stationary 
        """
        return isinstance(self, Stationary)


    def _process_cov_inputs (self, x, z):
        """
        function for processing inputs to the cov function

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d). If none then will assume z=x

        Outputs:
            x,z
        """
        assert x.ndim == 2
        assert x.shape[1] == self.n_dims
        if z is None:
            z = x
        else:
            assert z.ndim == 2
            assert z.shape[1] == self.n_dims,\
                        "should be %d dims, not %d" % (self.n_dims,z.shape[1])
        return x,z


    def _apply_children (self, K, x, z=None):
        """
        apply the children to the parents covariance matrix
        This MUST be called right at the end of the cov routine

        Inputs:
            K : parent's covariance matrix
            x,z : same as cov
        """
        for operation,child in self._children:
            if operation == 'mul':
                K = np.multiply(K, child.K(x, z))
            elif operation == 'add':
                K = np.add(     K, child.K(x, z))
            else:
                raise ValueError('Unknown kernel operation %s' % repr(operation))
        return K

    def __mul__ (k1, k2):
        """
        multiply kernel k1 with kernel k2
        returns k = k1 * k2
        note that explicit copies are formed in the process
        """
        assert isinstance(k2, BaseKernel)
        assert k2.n_dims == k1.n_dims # ensure the kernel has the same number of dimensions
        # copy the kernels
        parent = k1.copy()
        child  = k2.copy()

        # since multiplying, set the childs variance to fixed
        if np.size(child.constraint_map['variance']) > 1:
            child.constraint_map['variance'][0] = 'fixed'
        else:
            child.constraint_map['variance'] = 'fixed'

        # add the child to the parent
        parent._children.append( ('mul', child) ) # add the kernel the children
        return parent


    def __add__ (k1, k2):
        """
        adds kernel k1 with kernel k2
        returns k = k1 + k2
        note that explicit copies are formed in the process
        """
        assert isinstance(k2, BaseKernel), 'k2 must be a kernel'
        # copy the kernels
        parent = k1.copy()
        child  = k2.copy()

        # add the child to the parent
        parent._children.append( ('add', child) )
        return parent


    def copy (self):
        """ 
        return a deepcopy 
        """
        from copy import deepcopy
        # first create a deepcopy
        self_copy = deepcopy(self)
        # then create a copy of all children 
        self_copy._children = [(deepcopy(operation),child.copy()) \
                                    for operation,child in self_copy._children]
        return self_copy
