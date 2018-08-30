
import numpy as np

from ..grid import InducingGrid
from gp_grief.tensors import KronMatrix, SelectionMatrixSparse,\
                            RowColKhatriRaoMatrix, expand_SKC
from gp_grief.kern import GridKernel, GPyKernel

import logging
logger = logging.getLogger(__name__)


class GriefKernel (GridKernel):
    """ 
    Kernel composed of grid-structured eigenfunctions 
    """
    def __init__(self, kern_list, grid, n_eigs=1000, reweight_eig_funs=True, \
                    opt_kernel_params=False, w=1., dim_noise_var=1e-12,\
                    log_KRrowcol=True, **kwargs):
        """
        Inputs:
            kern_list : list of 1d kernels
            grid : inducing point grid
            n_eigs : number of eigenvalues to use
            reweight_eig_funs : whether the eigenfunctions should be reweighted
            opt_kernel_params : optimise kernel hyperparameters
            w : initial basis function weights (default is unity)
        """
        self.reweight_eig_funs = bool(reweight_eig_funs)
        self.opt_kernel_params = bool(opt_kernel_params)
        super(GriefKernel, self).__init__(kern_list=kern_list, **kwargs)
        assert isinstance(grid,InducingGrid), "must be an InducingGrid"
        assert grid.input_dim==self.n_dims, "number of dimensions do not match"
        self.grid = grid
        self.dim_noise_var = float(dim_noise_var)
        self.n_eigs = int(min(n_eigs, self.grid.num_data))

        # set the contraints for the base kernel hyperparameters
        if not self.opt_kernel_params: # then fix everything
            for i,kern in enumerate(self.kern_list):
                if isinstance(kern, GPyKernel):
                    self.kern_list[i].constraint_list = \
                                np.tile('fixed', np.shape(kern.constraint_list))
                else:
                    for key in kern.constraint_map:
                        self.kern_list[i].constraint_map[key] = \
                            np.tile('fixed', np.shape(kern.constraint_map[key]))

        # set the constraints for the weights
        if self.reweight_eig_funs:
            self.w_constraints = np.array(['+ve',] * self.n_eigs, dtype='|S10')
        else:
            self.w_constraints = np.array(['fixed',] * self.n_eigs, dtype='|S10')


        if w == 1.:
            self.w = np.ones(self.n_eigs)
        else:
            assert w.shape == (self.n_eigs,)
            assert np.all(w > 0.), "w's must be positive"
            self.w = w

        # initialize some stuff
        self._old_base_kern_params = None
        self.log_KRrowcol = log_KRrowcol


    def cov(self, x, z=None):
        """
        Returns everything needed for covariance matrix and its inverse, etc.
        Note that z should generally not be specified, you can save work by just 
        computing Phi_L or Phi_R, see how the computation is done below.

        Outputs:
            Phi_L : left basis function or coeff. matrix
            w : basis function weights
            Phi_R : right basis function or coeff. matrix (Phi_L if z is none)

        Notes:
            ```
            from scipy.sparse import dia_matrix
            from gp_grief.tensors import TensorProduct
            W = dia_matrix((w/lam, 0),shape=(w.size,)*2)
            K = TensorProduct([Phi_L, W, Phi_R.T])
            ```
        """
        assert x.shape[1] == self.n_dims
        if z is not None: # then computing cross cov matrix
            Phi_L = self.cov(x=x)[0]
            Phi_R = self.cov(x=z)[0]
        else:
            # setup inducing covariance matrix and eigenvals/vecs
            self._setup_inducing_cov()
            # compute the left coefficient matrix
            # first get the cross covariance matrix
            Kxu = super(GriefKernel,self).cov_kr(x=x,z=self.grid.xg,form_kr=False)
            Kux = [k.T for k in Kxu]

            # form the RowColKhatriRaoMatrix 
            SKC    = {'S':self._Sp, 'K':self._Quu.T.K, 'C':Kux}
            loglam = self._log_lam.reshape((1,-1))
            if self.log_KRrowcol: # form and rescale in a numerically stable manner
                log_matrix, sign = expand_SKC(**SKC, logged=True)
                Phi_L = sign.T * np.exp(log_matrix.T - 0.5*loglam)
            else:
                Phi_L = expand_SKC(**SKC, logged=False).T/np.sqrt(np.exp(loglam))

            # compute the left coefficient matrix (which is identical)
            Phi_R = Phi_L
        return Phi_L, self.w, Phi_R

    def cov_grad (self, x, grad_dim):
        """
        Computes d Phi_L / d x_{:,grad_dim}
        """
        self._setup_inducing_cov()
        # compute the left coefficient matrix
        # first get the cross covariance matrix
        dKxu = self.cov_kr_grad(x, self.grid.xg, grad_dim)
        dKux = [k.T for k in dKxu]

        # form the RowColKhatriRaoMatrix 
        log_matrix, sign = expand_SKC(S=self._Sp, K=self._Quu.T.K, C=dKux)
        dPhi = sign.T * np.exp(log_matrix.T - 0.5*self._log_lam.reshape((1,-1)))
        return dPhi

    @property
    def parameters(self):
        """
        returns the kernel parameters as a 1d array
        """
        # first get the regular parameters
        parameters = super(GriefKernel, self).parameters
        # then add the eigenfunction weights
        parameters = np.concatenate([parameters, self.w], axis=0)
        return parameters

    @parameters.setter
    def parameters(self, value):
        """
        setter for parameters property
        """
        # get the number of base kernel hyperparameters
        n_theta = value.size-self.n_eigs
        # set the parameters of the base kernel
        super(GriefKernel, self.__class__).parameters.fset(self, value[:n_theta]) # call setter in the super method, this
        # now set the weights
        self.w = value[n_theta:]

    @property
    def constraints(self):
        """
        returns the kernel parameters' constraints as a 1d array
        """
        # first get the regular constraints
        constraints = super(GriefKernel, self).constraints
        # then add the eigenfunction weights if nessessary
        constraints = np.concatenate([constraints, self.w_constraints], axis=0)
        return constraints

    @property
    def diag_val(self):
        """ return diagonal value of covariance matrix. Note that it's assumed the kernel is stationary """
        raise NotImplementedError('')


    def _setup_inducing_cov(self):
        """ setup the covariance matrix on the inducing grid, factorize and find largest eigvals/vecs """
        # determine if anything needs to be recomputed
        base_kern_params = super(GriefKernel, self).parameters
        if self._old_base_kern_params is not None and np.array_equal(self._old_base_kern_params, base_kern_params):
            return # then no need to recompute

        # get the covariance matrix on the grid
        Kuu = self.cov_grid(self.grid.xg, dim_noise_var=self.dim_noise_var)

        # compute svd of Kuu
        (self._Quu,T) = Kuu.schur()
        all_eig_vals = T.diag()

        # get the biggest eigenvalues and eigenvectors
        n_eigs = int(min(self.n_eigs, all_eig_vals.shape[0])) # can't use more eigenvalues then all of them
        eig_pos, self._log_lam = all_eig_vals.find_extremum_eigs(n_eigs=n_eigs,mode='largest',log_expand=True)[:2]

        # create a Khatri-Rao selection matrix, Sp 
        self._Sp = [SelectionMatrixSparse((col, Kuu.K[i].shape[0])) for i,col in enumerate(eig_pos.T)]

        # save the parameters
        self._old_base_kern_params = base_kern_params
