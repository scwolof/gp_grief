
from ..linalg import log_kron
import numpy as np
import scipy.linalg as la
import scipy.linalg.blas as blas
from numpy.linalg.linalg import LinAlgError


class KronMatrix (object):
	"""
	Tensor class which is a Kronecker product of matricies.

	Trefor Evans
	"""

	def __init__(self, K, sym=False):
		"""
		Inputs:
			K  : is a list of numpy arrays
			sym : bool (default False)
				specify true it submatricies in K are symmetric
		"""
		self._K = K # shallow copy
		self.n  = len(self.K)
		# sizes of the sub matrices
		self.sshape = np.vstack([np.shape(Ki) for Ki in self.K]) 
		# shape of the big matrix
		self.shape = np.atleast_1d(np.prod(np.float64(self.sshape),axis=0)) 
		if np.all(self.shape < np.iinfo(np.uint64).max): 
			# then can be stored as an int else proceed with shape as a float
			self.shape = np.uint64(self.shape)
		self.ndim = self.shape.size
		assert self.ndim <= 2, "kron matrix cannot be more than 2d"
		self.square = self.ndim==2 and self.shape[0]==self.shape[1]
		self.sym = sym
		if sym:
			assert np.array_equal(self.sshape[:,0],self.sshape[:,1]),\
				'this matrix cannot be symmetric: it is not square'
			self = self.ensure_fortran()

	@property
	def K(self):
		return self._K
	@K.setter
	def K(self,K):
		raise AttributeError("Attribute is Read only.")


	def kronvec_prod(self, x):
		"""
		Computes K*x where K is a kronecker product matrix and x is a column 
		vector which is a numpy array

		Inputs:
			self : (N,M) KronMatrix
			x : (M,1) matrix

		Outputs:
			y : (N,1) matrix

		this routine ensures all matricies are in fortran contigous format 
		before using blas routines

		it also takes advantage of matrix symmetry (if sym flag is set)
		"""
		if x.shape != (self.shape[1],1):
			raise ValueError('x is the wrong shape, must be (%d,1), not %s'\
							 % (self.shape[1],repr(x.shape)))

		y = x
		for i,Ki in reversed(list(enumerate(self.K))):
			y = np.reshape(y, (self.sshape[i,1], -1),order='F')
			if isinstance(Ki, np.ndarray): # use optimized blas routines
				if np.isfortran(Ki):
					a = Ki
				else:
					a = Ki.T
				if self.sym:
					if np.isfortran(y):
						y = blas.dsymm(alpha=1, a=a, b=y  , side=0).T
					else:
						# just switch the order
						y = blas.dsymm(alpha=1, a=a, b=y.T, side=1) 
				else:
					if np.isfortran(y):
						b = y
					else:
						b = y.T
					y = blas.dgemm(alpha=1, a=a, b=b, trans_a=(
						not np.isfortran(Ki)), trans_b=(not np.isfortran(y))).T
			else: # use __mul__ routine
				y = (Ki*y).T
		y = y.reshape((-1,1),order='F') # reshape to a column vector
		return y


	def __mul__(self,x):
		""" overloaded * operator """
		return self.kronvec_prod(x)


	def kronkron_prod(self, X):
		"""
		computes K*X where K,X is a kronecker product matrix
		"""
		if not isinstance(X,KronMatrix):
			raise TypeError("X is not a KronMatrix")
		elif X.n != self.n:
			raise TypeError('inconsistent kron structure')
		elif not np.array_equal(X.sshape[1],self.sshape[0]):
			raise TypeError("Dimensions of X submatricies are not consistent")
		# perform the product
		return KronMatrix([self.K[i].dot(X.K[i]) for i in range(self.n)])


	def kronvec_div(self, x):
		"""
		Computes y = K \ x where K is a kron prod matrix and x a column matrix

		Inputs:
			self : (N,N) triangular cholesky KronMatrix from chol
			x : (N,1) matrix

		Outputs:
			y : (N,1) matrix
		"""
		assert self.ndim == 2
		if x.shape != (self.shape[0],1):
			raise ValueError('x wrong shape, must be (%d,1)' % self.shape[0])

		y = x # I don't care to actually copy this explicitly
		for i,Ki in enumerate(self.K):
			y = np.reshape(y, (-1, self.sshape[i,0]),order='F')
			if hasattr(Ki, "solve"):
				y = Ki.solve(b=y.T)
			else:
				y = la.solve(a=Ki, b=y.T, sym_pos=self.sym, overwrite_b=True)
		y = y.reshape((-1,1), order='F')
		return y


	def chol(self):
		"""
		performs cholesky factorization, returning upper triangular matrix

		see equivalent in matlab for details
		"""
		assert self.square
		C = np.empty(self.n, dtype=object)
		for i,Ki in enumerate(self.K):
			if hasattr(Ki, "chol"):
				C[i] = Ki.chol() # assumed will return upper triangular!
			else:
				C[i] = np.linalg.cholesky(Ki).T
		return KronMatrix(C)


	def schur(self):
		""" compute schur decomposition. Outputs (Q,T). """
		assert self.square
		T = np.empty(self.n, dtype=object)
		Q = np.empty(self.n, dtype=object)
		for i,Ki in enumerate(self.K):
			if hasattr(Ki, "schur"):
				T[i],Q[i] = Ki.schur()
			else:
				T[i],Q[i] = la.schur(Ki)
		return KronMatrix(Q), KronMatrix(T)


	def svd(self):
		"""
		singular value decomposition

		NOTE: it is currently assumed that self is PSD. 
		This can easily be changed though

		Returns (Q,eig_vals) where Q is a KronMatrix whose columns are 
		eigenvalues and eig_vals is a 1D array
		"""
		assert self.square, "matrix must be square for current implementation"
		# first get a list of the 1D matricies
		Q = np.empty(self.n, dtype=object)
		eig_vals = np.empty(self.n, dtype=object)
		for i,Ki in enumerate(self.K):
			if hasattr(Ki, "svd"):
				(Q[i],eig_vals[i]) = Ki.svd()
			else:
				(Q[i],eig_vals[i]) = np.linalg.svd(Ki,full_matrices=0,compute_uv=1)[:2]
		return KronMatrix(Q), KronMatrix(eig_vals)


	def transpose(self):
		""" 
		Transpose the kronecker product matrix. 
		Won't copy the matricies but will return a view 
		"""
		assert self.ndim == 2
		if self.sym:
			return self
		else:
			return KronMatrix([Ki.T for Ki in self.K]) # transpose submatrices
	T = property(transpose) # calling self.T will do the same thing as transpose

	def expand(self, log_expansion=False):
		"""
		expands the kronecker product matrix explicitly. Expensive!

		Inputs:
			log_expansion : if used then performs numerically stable expansion.
				If specified then the output will be the log of the value.
		"""
		if log_expansion:
			Kb = np.array([0.])
			for Ki in self.K:
				if hasattr(Ki, "expand"):
					Kb = log_kron(a=Kb, b=Ki.expand(), a_logged=True)
				else:
					Kb = log_kron(a=Kb, b=Ki, a_logged=True)
		else:
			Kb = 1.
			for Ki in self.K:
				if hasattr(Ki, "expand"):
					Kb = np.kron(Kb, Ki.expand())
				else:
					Kb = np.kron(Kb, Ki)
		return Kb.reshape(np.int32(self.shape))


	def inv(self):
		""" invert matrix """
		assert self.square
		I = np.empty(self.n, dtype=object)
		for i,Ki in enumerate(self.K):
			if hasattr(Ki, "inv"):
				I[i] = Ki.inv()
			else:
				I[i] = np.linalg.inv(Ki)
		return KronMatrix(I)


	def diag(self):
		"""
		returns the diagonal of the kronecker product matrix as 1d KronMatrix
		"""
		assert self.ndim == 2
		D = np.empty(self.n, dtype=object)
		for i,Ki in enumerate(self.K):
			if hasattr(Ki, "diag"):
				D[i] = Ki.diag()
			else:
				D[i] = np.diag(Ki)
		return KronMatrix(D)


	def sub_cond(self):
		""" 
		return the condition number of each of the sub matricies 
		"""
		assert self.square
		return [np.linalg.cond(Ki) for Ki in self.K]


	def sub_shift(self,shift=1e-6):
		""" 
		apply diagonal correction term to sub matrices to improve conditioning 
		"""
		if not np.array_equal(self.sshape[:,0],self.sshape[:,1]):
			raise RuntimeError('can only apply sub_shift for square matricies')
		for i,Ki in enumerate(self.K):
			self.K[i] = Ki + shift*np.identity(self.sshape[i,0])
		if self.sym:
			self = self.ensure_fortran()
		return self


	def ensure_fortran(self):
		""" ensures that the submatricies are fortran contiguous """
		for i,Ki in enumerate(self.K):
			if isinstance(Ki, np.ndarray):
				self.K[i] = np.asarray(Ki, order='F')
		return self


	def solve_chol(U,x):
		"""
		solves y = U \ (U' \ x)

		Inputs:
			U : (N,N) triangular cholesky KronMatrix from chol
			x : (N,1) matrix

		Outputs:
			y : (N,1) matrix
		"""
		if x.shape != (U.shape[0],1):
			raise ValueError('x wrong shape, must be (%d,1)' % U.shape[0])

		y = x # I don't care to actually copy this explicitly
		for i,Ui in enumerate(U.K):
			y = np.reshape(y, (-1, U.sshape[i,0]),order='F')
			if hasattr(Ui, "solve_chol"):
				y = Ui.solve_chol(y.T)
			else:
				lt = {'a':Ui, 'lower':False, 'overwrite_b':True}
				# y = Ui' \ y'
				#y = la.solve_triangular(a=Ui, b=y.T, trans='T', lower=False, overwrite_b=True) 
				y = la.solve_triangular(b=y.T, trans='T', **lt) 
				# Ui \ y
				#y = la.solve_triangular(a=Ui, b=y,   trans='N', lower=False, overwrite_b=True) 
				y = la.solve_triangular(b=y,   trans='N', **lt) 
		y = y.reshape((-1,1), order='F')
		return y


	def solve_schur(Q,t,x,shift=0.0):
		"""
		solves shifted linear system of equations (K+lam*I)y=x using the 
		schur decomposition of K

		Inputs:
			Q : (N,N) unitary KronMatrix containing the eigenvectors of K
			t : (N,) array containing corresponding eigenvalues of K.
				This can be computed from the diagonal matrix T returned by 
				schur (such that Q*T*Q.T = K) as t = T.diag().expand()
				If it's a KronMatrix then it will be assumed that T was passed
			x :   (N,1) matrix
			shape : float corresponding to shift to be applied to the system

		Outputs:
			y : (N,1) matrix
		"""
		if x.shape != (Q.shape[0],1):
			raise ValueError('x wrong shape, must be (%d,1)' % Q.shape[0])
		if isinstance(t,KronMatrix): # then assume that the T was passed
			t = t.diag().expand()
		y = (Q.T)*x
		y = y / np.reshape(t+shift, y.shape) # solve the diagonal linear system
		y = Q*y
		return y


	def eig_vals(self):
		""" returns the eigenvalues of the matrix """
		assert self.ndim == 2
		eigs = np.empty(self.n, dtype=object)
		for i,Ki in enumerate(self.K):
			if hasattr(Ki, "eig_vals"):
				eigs[i] = Ki.eig_vals()
			elif self.sym:
				eigs[i] = np.linalg.eigvalsh(Ki)
			else:
				eigs[i] = np.linalg.eigvals(Ki)
		return KronMatrix(eigs)


	def find_extremum_eigs(eigs, n_eigs, mode='largest', log_expand=False, 
							sort=True, compute_global_loc=False):
		"""
		returns position of n_eigs largest eigenvalues in KronMatrix vector eigs

		Inputs:
			n_eigs : number of eigenvalues to find
			mode : largest or smallest eigenvalues
			log_expand : if true then will return the log of the eigenvalues.
				Note that this is numerically more stable if problem is high dim
			sort : if true then sort returned eigenvalues in descending order

		Notes:
		This function only requires at most O( (d-1)pm ) time where d is the 
		number of dimensions, p is n_eigs and m is the size of each submatrix 
		in eigs (such that eigs has size m^d).
		"""
		assert eigs.ndim == 1, "eigs must be a 1D KronMatrix"
		assert isinstance(n_eigs, (int, np.int32)),\
			"n_eigs=%s must be an integer" % repr(n_eigs)
		assert n_eigs >= 1, "must use at least 1 eigenvalue"
		assert n_eigs <= eigs.shape[0], "n_eigs > number of eigenvalues"
		assert mode == 'largest' or mode == 'smallest'

		# Return n_eigs extremum values and indices of those values
		def get_extremum(vec):
			if np.size(vec) <= n_eigs: # then return all
				return np.arange(n_eigs), vec
			# now make the partition. this partition runs in linear O(n) time
			if mode == 'largest':
				ind = np.argpartition(vec, -n_eigs)[-n_eigs:]
			elif mode == 'smallest':
				ind = np.argpartition(vec, n_eigs)[:n_eigs]
			return ind, vec[ind]

		# break the problem up into a sequence of 2d kronecker products
		eig_loc, eig_vals = get_extremum(eigs.K[0]) # first initialize
		eig_loc = eig_loc.reshape((-1,1))
		if log_expand: # then take the log of these values
			eig_vals = np.log(eig_vals)
		for i in range(1, eigs.n):
			# perform the kronecker product and get the n_eigs extreme values
			if log_expand:
				inds, eig_vals = get_extremum(log_kron(
										a=eig_vals, b=eigs.K[i], a_logged=True))
			else:
				inds, eig_vals = get_extremum(np.kron(eig_vals, eigs.K[i]))
			# now update eig_loc
			el1 = eig_loc[np.int32(np.floor_divide(inds, eigs.K[i].size)),:]
			el2 = np.int32(np.mod(inds, eigs.K[i].size))
			eig_loc = np.hstack(
				[
				el1.reshape((inds.size,-1)), # position from previous eig_vals
				el2.reshape((-1,1)) # position from current dimension eig_vals
				])

		# now compute the global location of the eigenvalue 
		# (if your were to expand eigs, the index of each)
		if compute_global_loc:
			global_loc = np.zeros(n_eigs, dtype=int)
			# initialize size of vector being constructed from previous loop
			cum_size = 1 
			for i in reversed(range(eigs.n)): # loop backwards
				global_loc = cum_size * eig_loc[:,i] + global_loc
				cum_size *= eigs.K[i].size # increment the vector size
		else:
			global_loc = None

		# now sort if ness
		if sort:
			order = np.argsort(eig_vals)[::-1] # descending order
			eig_vals = eig_vals[order]
			eig_loc = eig_loc[order]
			if compute_global_loc:
				global_loc = global_loc[order]
		return eig_loc, eig_vals, global_loc


	def get_col(self, pos):
		"""
		returns the expanded column as a KronMatrix

		pos should be a tuple of length self.n whose elements specify the 
		column of the sub-matrix involved in the expansion

		to return the row, do K.T.get_col(...)
		"""
		assert len(pos) == self.n
		assert np.size(pos[0]) == 1
		assert isinstance(pos[0], int)
		assert self.ndim == 2
		return KronMatrix([self.K[i][:,j].reshape((-1,1)) 
							for i,j in enumerate(pos)])


	def log_det(eig_vals):
		""" compute the log determinant in an efficient manner """
		assert eig_vals.ndim == 1 # must be 1d KronMatrix
		ldet = 0
		for i,eigs in enumerate(eig_vals.K):
			# number of times this sum term is used in the log det expansion
			repetition = np.prod(np.delete(eig_vals.sshape,i)) 
			ldet += repetition * np.sum(np.log(eigs))
		return ldet
