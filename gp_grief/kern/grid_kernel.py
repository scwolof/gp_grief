
import numpy as np
from itertools import product
from gp_grief.tensors import KronMatrix, KhatriRaoMatrix


class GridKernel (object):
    """ 
    Simple kernel wrapper for GridRegression which is a product of 1d kernels 
    """
    def __init__(self, kern_list, radial_kernel=False):
        """
        Kernel for gridded inducing point methods and structured problems

        Inputs:
            kern_list : list or 1d array of kernels
            radial_kernel : bool
                If true then will use the same kernel along each dimension. 
                Will just grab kernel from the first dimension to use for all.
        """
        # initialize the kernel list
        self.kern_list = kern_list

        # add the dimension of the grid
        self.grid_dim = len(kern_list)

        # check if radial kernel
        assert isinstance(radial_kernel, bool)
        self.radial_kernel = radial_kernel
        if self.radial_kernel:
            for kern in self.kern_list:
                assert kern.n_dims == self.kern_list[0].n_dims,\
                        "number of grid dims must be equal for all slices"
            self.kern_list = [self.kern_list[0],]*np.size(kern_list)
        else:
            # set the variance as fixed for all but the first kernel. 
            for i in range(1,self.grid_dim):
                if hasattr(self.kern_list[i], 'fix_variance'):
                    self.kern_list[i].fix_variance()
                elif np.size(self.kern_list[i].constraint_map['variance']) > 1:
                    self.kern_list[i].constraint_map['variance'][0] = 'fixed'
                else:
                    self.kern_list[i].constraint_map['variance'] = 'fixed'

        # the the total number of dims
        self.n_dims = np.sum([kern.n_dims for kern in self.kern_list])
        return


    def K_grid(self, x, z=None, dim_noise_var=None, use_toeplitz=False):
        """
        generate matrix, creates covariance matrix mapping between x1 and x2.

        Inputs:
          x : numpy.ndarray of shape (self.grid_dim,)
          z : (optional) numpy.ndarray of shape (self.grid_dim,) if None x2=x1
              for both x1 and x2:
              the ith element in the array must be a matrix of size 
              [n_mesh_i,n_dims_i] where n_dims_i is the number of dimensions 
              in the ith kronecker pdt matrix and n_mesh_i is the number of 
              points along the ith dimension of the grid.
              Note that for spatial temporal datasets, n_dims_i is probably 1
              but for other problems this might be of much higher dimensions.
          dim_noise_var : float (optional)
              diagonal term to use to shift the diagonal of each dimension to 
              improve conditioning

        Outputs:
          K : gp_grief.tensors.KronMatrix of size determined by x and z 
                (prod(n_mesh1(:)), prod(n_mesh2(:))
              covariance matrix
        """
        assert dim_noise_var is not None, "dim_noise_var must be specified"
        # toeplitz stuff
        if isinstance(use_toeplitz, bool):
            use_toeplitz = [use_toeplitz,] * self.grid_dim
        else:
            assert np.size(use_toeplitz) == self.grid_dim
        if np.any(use_toeplitz):
            assert z is None, "toeplitz can only be used where the (square)"\
                                + "covariance matrix is being computed"

        # check inputs
        assert len(x) == self.grid_dim # ensure 1st dim is same as grid dim
        if z is None:
            cross_cov = False
            z = [None,] * self.grid_dim # array of None
        else:
            cross_cov = True
            assert len(z) == self.grid_dim # ensure 1st dim is same as grid dim

        # get the 1d covariance matricies
        K = []
        # loop through and generate the covariance matricies
        for i,(kern, toeplitz) in enumerate(zip(self.kern_list, use_toeplitz)): 
            if toeplitz and z[i] is None:
                K.append(kern.K_toeplitz(x=x[i]))
            else:
                K.append(kern.K(x=x[i],z=z[i]))

        # now create a KronMatrix instance
        # reverse order and set as symmetric only if the two lists are identical
        K = KronMatrix(K[::-1], sym=(z[0] is None)) 

        # shift the diagonal of the sub-matricies if required
        if dim_noise_var != 0.:
            assert not cross_cov, "not implemented for cross covariances yet"
            K = K.sub_shift(shift=dim_noise_var)
        return K


    def K (self, x, z=None):
        """
        Evaluate covariance kernel at points to form a covariance matrix

        Inputs:
            x : array of shape (N, d)
            z : array of shape (M, d) (optional). If not specified assume z=x

        Outputs:
            k : matrix of shape (N, M)
        """
        # loop through dimensions, compute cov and perform hadamard product
        i_cur = 0
        zi = None # set default value
        for i,kern in enumerate(self.kern_list):
            xi = x[:,i_cur:(i_cur+kern.n_dims)] # just grab subset of dimensions
            if z is not None:
                zi = z[:,i_cur:(i_cur+kern.n_dims)]
            i_cur += kern.n_dims

            # compute cov of subset of dimensions and multipy with other dimensions
            if i == 0:
                K = kern.K(x=xi,z=zi)
            else: # perform hadamard product
                K = np.multiply(K, kern.K(x=xi,z=zi))
        return K


    def K_kr (self, x, z, form_kr=True):
        """
        Evaluate covariance kernel at points to form a covariance matrix in 
        row partitioned Khatri-Rao form

        Inputs:
            x : array of shape (N, d)
            z : numpy.ndarray of shape (d,)
              the ith element in the array must be a matrix of size [n_mesh_i,1]
              where n_mesh_i is the number of points along the ith dimension
              of the grid.
            form_kr : if True will form the KhatriRao matrix, 
                      else will just return a list of arrays

        Outputs:
            k : row partitioned Khatri-Rao matrix of shape (N, prod(n_mesh))
        """
        N, d = x.shape
        assert self.grid_dim == d, "currently works for 1-dimensional grids"

        # loop through dimensions and compute covariance matricies
        # and compute the covariance of the subset of dimensions
        Kxz = [kern.K(x=x[:,(i,)],z=z[i]) for i,kern in enumerate(self.kern_list)]

        # flip the order
        Kxz = Kxz[::-1]

        # convert to a Khatri-Rao Matrix
        if form_kr:
            Kxz = KhatriRaoMatrix(A=Kxz, partition=0) # row partitioned
        return Kxz

    def dK_kr_dX (self, x, z, grad_dim):
        """
        Gradient of K_kr wrt grad_dim'th dimension of x
        """
        N, d = x.shape
        assert self.grid_dim == d

        kern_list = m_grief.kern.kern_list
        # loop through each dimension and compute the 1-dimensional covariance 
        # matricies and compute the covariance of the subset of dimensions
        Kxz = []
        for i, k in enumerate(kern_list):
            if i == grad_dim:
                Ui, n = z[i], z[i].shape[0]
                t = np.zeros((N,n))
                for j in range(n):
                    t[:,[j]] = k.kern.gradients_X(1, x[:,(i,)], Ui[[j]])
            else:
                t = k.K(x=x[:,(i,)],z=z[i])
            Kxz.append(t)
            
        # flip the order
        Kxz = Kxz[::-1]
        return Kxz

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        if self.radial_kernel:
            parameters = np.ravel(self.kern_list[0].parameters)
        else:
            parameters = np.concatenate(\
                [np.ravel(kern.parameters) for kern in self.kern_list], axis=0)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        assert isinstance(value, np.ndarray) # must be a numpy array
        assert value.ndim == 1 # must be 1d

        # set the parameters
        if self.radial_kernel:
            self.kern_list[0].parameters = value
            self.kern_list = [self.kern_list[0],]*np.size(self.kern_list)
        else:
            i0 = 0 # counter of current position in value
            for kern in self.kern_list:
                # get the old parameters to check the size
                old = kern.parameters
                # set the parameters
                kern.parameters = value[i0:i0+np.size(old)].reshape(np.shape(old))
                i0 += np.size(old) # increment counter

    @property
    def constraints(self):
        """
        returns the kernel parameters' constraints as a 1d array
        """
        if self.radial_kernel:
            constraints = np.ravel(self.kern_list[0].constraints)
        else:
            constraints = np.concatenate(\
                [np.ravel(kern.constraints) for kern in self.kern_list], axis=0)
        return constraints

    @property
    def diag_val(self):
        """ 
        return diagonal value of covariance matrix. 
        Note that it's assumed the kernel is stationary 
        """
        return self.K(np.zeros((1,self.n_dims))).squeeze()