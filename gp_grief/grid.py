from .tensors import KhatriRaoMatrix, BlockMatrix
from .linalg import uniquetol
import numpy as np
import logging
import warnings
from pdb import set_trace
logger = logging.getLogger(__name__)


def nd_grid(*xg):
    """
    This mimics the behaviour of nd_grid in matlab.
    (np.mgrid behaves similarly however I don't get how to call it clearly.)
    """
    grid_shape = [np.shape(xg1d)[0] for xg1d in xg] # shape of the grid
    d          = np.size(grid_shape)
    N          = np.product(grid_shape)
    X_mesh     = np.empty(d, dtype=object)
    for i, xg1d in enumerate(xg): # for each 1d component
        if np.ndim(xg1d) > 1:
            assert np.shape(xg1d)[1] == 1, "currently supports 1d grid dims"
        n = np.shape(xg1d)[0] # number of points along dimension of grid
        slice_shape    = np.ones(d, dtype=int)
        slice_shape[i] = n
        stack_shape    = np.copy(grid_shape)     
        stack_shape[i] = 1
        X_mesh[i]      = np.tile(xg1d.reshape(slice_shape), stack_shape)
    return X_mesh


def grid2mat(*xg):
    """
    transforms a grid to a numpy matrix of points

    Inputs:
        *xg : ith input is a 1D array of grid locations along the ith dimension

    Outputs:
        x : (N,d) numpy matrix, where N = len(xg[0])*...*len(xg[d-1])
    """
    X_mesh = nd_grid(*xg) # this is the meshgrid, all I have to do is flatten it
    d = X_mesh.shape[0]
    N = X_mesh[0].size
    x = np.zeros((N, d)) # initialize
    for i, X1d in enumerate(X_mesh): # for each 1d component of the mesh
        x[:,i] = X1d.reshape(-1, order='C') # reshape it into a vector
    return x


class InducingGrid(object):
    """ inducing point grid for structured kernel interpolation """

    def __init__(self, x=None, mbar=10, eq=True, 
                    mbar_min=1, xg=None, beyond_domain=None):
        """
        Generates an inducing point grid.

        Inputs to generate an inducing grid from scattered data:
            x : (n_train,n_dims) scattered training points
            mbar : if mbar > 1 then indicates the number of points desired 
                along each dim. Else if in (0,1] then the number of points will 
                be that fraction of unique values. If mbar is a tuple, then the 
                value applys for that specific dimension
            eq : if true forces evenly spaced grid, else will use a kmeans 
                algorithm which tries to get an even number of points closest 
                to each inducing point.
            mbar_min : the minimum number of points per dimension. Note that k 
                will be changed to no greater than the number of unique points
                per dimension if not eq. This parameter sets a lower bound for
                this value which is useful for eg when SKI is employed.
            beyond_domain : None or float
                if None, then no effect. If a value then will go that fraction 
                beyond bounds along each dimension. Note that the number of 
                points specifed won't be violated and a point will still be
                placed right on the edge of the bounds if eq=True

        Inputs to generate an instance from a user specified grid:
            xg : list specifying the grid in each dimension. Each element in xg
                must be the array of points along each grid dimension, eg.
                    xg[i].shape = (grid_shape[i], grid_sub_dim[i])
                Points along each grid dimension should be sorted ascending if 
                the demension is 1d. No other inputs are nessessary and if 
                specified then they'll be ignored
        """
        logger.debug('Initializing inducing grid.')
        k = mbar; del mbar # mbar is an alias
        k_min = mbar_min; del mbar_min # mbar_min is an alias
        if xg is None: # then generate a grid from the scattered points x
            # deal with inputs
            assert isinstance(x,np.ndarray)
            assert x.ndim == 2
            self.eq = eq
            if not isinstance(k,(tuple,list,np.ndarray)):
                k = (k,)*x.shape[1]

            # get some statistics and counts (assuming 1d along each dimension)
            (n_train, self.grid_dim) = x.shape
            self.grid_sub_dim        = np.ones(self.grid_dim, dtype=int)
            self.input_dim           = np.sum(self.grid_sub_dim)
            self.grid_shape          = np.zeros(self.grid_dim, dtype=int)
            x_rng                    = np.vstack((np.amin(x,axis=0), 
                                                  np.amax(x,axis=0), 
                                                  np.ptp(x,axis=0))).T
            n_unq                    = np.array([np.unique(x[:,i]).size for \
                                                 i in range(self.grid_dim)])
            if not np.all(n_unq >= 2):
                logger.debug('some dimension have < 2 unique points')
            for i,ki in enumerate(k):
                if ki <= 1:
                    self.grid_shape[i] = np.int32(np.maximum(\
                                            np.ceil(ki*n_unq[i]), k_min) )
                else:
                    assert np.mod(ki,1) == 0, "if k > 1 then k must be integer"
                    # don't allow the number of points to be greater than n_unq
                    self.grid_shape[i] = np.int32(np.maximum(\
                                            np.minimum(ki, n_unq[i]), k_min))
            self.num_data = np.prod(np.float64(self.grid_shape))

            # if bounds are to be added, then want to call recursively
            if beyond_domain is not None:
                assert np.all(self.grid_shape >= 2), "need >=2 points per dim"
                # get the grid with no bounds but 2 less points per dimension
                xg = InducingGrid(x=x, k=self.grid_shape-2, eq=eq, \
                    to_plot=False, k_min=0, xg=None, beyond_domain=None).xg
                for i in range(x.shape[1]):
                    xg[i] = np.vstack((x_rng[i,0]-beyond_domain*x_rng[i,2], \
                            xg[i], x_rng[i,1]+beyond_domain*x_rng[i,2]))
                # since xg is now specified, it will be added to the grid below
            else:
                #figure out if the grid should be on unique points
                on_unique = self.grid_shape == n_unq

                # create the grid
                # self.xg (list, length n_dims) specifies grid along each dim
                self.xg = np.empty(self.grid_dim, dtype=object)
                for i_d in range(self.grid_dim):
                    if on_unique[i_d]: # place the grid on the unique values
                        self.xg[i_d] = np.unique(x[:,i_d]).reshape((-1,1))
                    elif self.eq: # equally spaced grid points
                        self.xg[i_d] = np.linspace(x_rng[i_d,0],x_rng[i_d,1], \
                            num=self.grid_shape[i_d]).reshape((-1,1))
                    elif self.grid_shape[i_d] == 2: # then place on the ends
                        self.xg[i_d] = x_rng[i_d,:2].reshape((-1,1))
                    else: # non equally spaced grid points
                        raise NotImplementedError
        if xg is not None: # a grid has already been specified, use this instead
            self.xg        = np.asarray(xg)
            self.grid_dim = self.xg.shape[0] # number of grid dimensions
            self.grid_shape   = np.zeros(self.grid_dim, dtype=int) 
            self.grid_sub_dim = np.zeros(self.grid_dim, dtype=int)
            for i,X in enumerate(self.xg): # loop over grid dimensions
                assert X.ndim == 2, "each element in xg must be a 2d array"
                self.grid_sub_dim[i] = X.shape[1]
                self.grid_shape[i]   = X.shape[0]
            self.input_dim = np.sum(self.grid_sub_dim) # total number of dims
            self.num_data = np.prod(np.float64(self.grid_shape)) 
            self.eq = None


    def __getitem__(self, key):
        """ so you can get self[key] """
        return self.xg[key]


    def __setitem__(self, key, value):
        """ so you can set self[key] = value """
        self.xg[key] = value


