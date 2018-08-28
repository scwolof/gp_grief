
import numpy as np
from gp_grief.kern import BaseKernel


class Stationary (BaseKernel):
    """ base class for stationary kernels """

    def distances_squared (self, x, z=None, lengthscale=None):
        """
        Evaluate the distance between points squared.

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional)

        Outputs:
            k : matrix of distances of shape shape (N, M)
        """
        x, z = self._process_cov_inputs(x, z) # process inputs

        # reshape the matricies correctly for broadcasting
        N = x.shape[0]
        M = z.shape[0]
        d = self.active_dims.size # the number of active dimensions
        x = np.asarray(x)[:,self.active_dims].reshape((N,1,d))
        z = np.asarray(z)[:,self.active_dims].reshape((1,M,d))

        # Code added to use different lengthscales for each dimension
        if lengthscale is None:
            lengthscale = np.ones(d,dtype='d')
        elif isinstance(lengthscale,float):
            lengthscale = lengthscale*np.ones(d,dtype='d')
        else:
            lengthscale = np.asarray(lengthscale).flatten()
            assert len(lengthscale) == d

        # now compute the distances
        return np.sum(np.power((x-z)/lengthscale.reshape((1,1,d)),2),
                      axis=2, keepdims=False)


    def distances (self, x, z=None, lengthscale=None):
        """
        Evaluate the distance between points along each dimension

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional)

        Outputs:
            k : matrix of distances of shape (N, M, d)
        """
        x, z = self._process_cov_inputs(x, z) # process inputs

        # reshape the matricies correctly for broadcasting
        N = x.shape[0]
        M = z.shape[0]
        d = self.active_dims.size # the number of active dimensions
        x = np.asarray(x)[:,self.active_dims].reshape((N,1,d))
        z = np.asarray(z)[:,self.active_dims].reshape((1,M,d))

        # Code added to use different lengthscales for each dimension
        if lengthscale is None:
            lengthscale = np.ones(d,dtype='d')
        elif isinstance(lengthscale,float):
            lengthscale = lengthscale*np.ones(d,dtype='d')
        else:
            lengthscale = np.asarray(lengthscale).flatten()
            assert len(lengthscale) == d

        # now compute the distances
        return (x-z)/lengthscale.reshape((1,1,d))


class RBF (Stationary):
    """
    squared exponential kernel with the same shape parameter in each dimension
    """
    def __init__ (self,n_dims,variance=1.,lengthscale=1.,active_dims=None,name=None):
        """
        squared exponential kernel

        Inputs: (very much the same as in GPy.kern.RBF)
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : all dims active by default, subset can be specified
        """
        super(RBF, self).__init__(\
                    n_dims=n_dims, active_dims=active_dims, name=name)

        # deal with the parameters
        assert np.size(variance) == 1
        assert np.size(lengthscale) == 1
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def K (self,x,z=None,lengthscale=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). Default: z = x
            lengthscale : a vector of length scales for each dimension

        Outputs:
            k : matrix of shape (N, M)
        """
        if self.lengthscale < 1e-6: # then make resiliant to division by zero
            K = self.variance * (self.distances_squared(x=x,z=z)==0)
        else: # then compute the nominal way
            if lengthscale is None:
                K = self.variance*np.exp( -0.5*self.distances_squared(x=x,z=z)\
                                            / self.lengthscale**2 )
            else:
                lengthscale = np.asarray(lengthscale).flatten()
                assert len(lengthscale) == self.active_dims.size
                K = self.variance * np.exp( -0.5 * \
                        self.distances_squared(x=x,z=z,lengthscale=lengthscale))
        K = self._apply_children(K, x, z)
        return K


class Exponential (Stationary):
    def __init__ (self,n_dims,variance=1.,lengthscale=1.,active_dims=None,name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : all dims active by default, subset can be specified
        """
        super(Exponential, self).__init__(\
                    n_dims=n_dims, active_dims=active_dims, name=name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def K (self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r = np.sqrt(self.distances_squared(x=x,z=z)) / self.lengthscale
        K = self.variance * np.exp( -r )
        K = self._apply_children(K, x, z)
        return K


class Matern32 (Stationary):
    def __init__ (self,n_dims,variance=1.,lengthscale=1.,active_dims=None,name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : all dims active by default, subset can be specified
        """
        super(Matern32, self).__init__(\
                    n_dims=n_dims, active_dims=active_dims, name=name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def K (self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified then will assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        r = np.sqrt(self.distances_squared(x=x,z=z)) / self.lengthscale
        K = self.variance * (1.+np.sqrt(3.)*r) * np.exp(-np.sqrt(3.)*r)
        K = self._apply_children(K, x, z)
        return K


class Matern52 (Stationary):
    def __init__ (self,n_dims,variance=1.,lengthscale=1.,active_dims=None,name=None):
        """
        squared exponential kernel

        Inputs:
            n_dims : number of dimensions
            variance : kernel variance
            lengthscale : kernel lengthscale
            active_dims : all dims active by default, subset can be specified
        """
        super(Matern52, self).__init__(\
                    n_dims=n_dims, active_dims=active_dims, name=name)

        # deal with the parameters
        self.variance = np.float64(variance)
        self.lengthscale = np.float64(lengthscale)
        self.parameter_list = ['variance','lengthscale']

        # deal with default constraints
        self.constraint_map = {'variance':'+ve', 'lengthscale':'+ve'}


    def K (self,x,z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). Default: z = x

        Outputs:
            k : matrix of shape (N, M)
        """
        r2 = self.distances_squared(x=x,z=z) / self.lengthscale**2
        r  = np.sqrt(r2)
        K  = self.variance * (1.+np.sqrt(5.)*r+(5./3)*r2)*np.exp(-np.sqrt(5.)*r)
        K  = self._apply_children(K, x, z)
        return K
