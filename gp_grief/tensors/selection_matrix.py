
import numpy as np
import scipy.sparse as sparse


class SelectionMatrix:
	""" 
	allows efficient multiplication with a selection matrix and its transpose 
	"""
	ndim = 2

	def __init__(self, indicies):
		"""
		creates a selection matrix with one nonzero entry per row

		Inputs:
			indicies : bool array or tuple
				specifies the location of the non-zero in each row.
				if bool:
					Each the index of each True element will be on its own row
				if tuple:
					must be (selection_inds, size) where selection inds is a 
					1d int array and size is an int
		"""
		if isinstance(indicies, tuple):
			assert len(indicies) == 2
			assert indicies[0].ndim == 1
			self.shape = [indicies[0].size, indicies[1]]
			int_idx = indicies[0]
		else:
			assert indicies.ndim == 1
			assert indicies.dtype == bool
			self.shape = [np.count_nonzero(indicies), indicies.size]
			int_idx = np.nonzero(indicies)[0]

		nnz = self.shape[0]
		self.sel = sparse.csr_matrix(
			(np.ones(nnz,dtype=bool),(np.arange(nnz),int_idx)), 
			shape=self.shape, dtype=bool)
		# testing shows precomputing the transpose saves lots of time
		self.sel_T = self.sel.T 
		return


	def mul(self,x):
		""" matrix-vector product """
		return self.sel * x


	def mul_T(self,x):
		""" matrix-vector product with the transpose """
		return self.sel_T * x


class SelectionMatrixSparse:
	"""
	allows efficient multiplication with a selection matrix and its transpose 
	where we never explictly form a vector of full
	"""
	ndim = 2

	def __init__(self, indicies):
		"""
		creates a selection matrix with one nonzero entry per row

		Inputs:
			indicies : bool array or tuple
				specifies the location of the non-zero in each row.
				must be (selection_inds, size) where selection inds is a 
				1d int array and size is an int
		"""
		assert isinstance(indicies, tuple)
		assert len(indicies) == 2
		assert indicies[0].ndim == 1
		self.shape = [indicies[0].size, indicies[1]]
		self.indicies = indicies[0]
		# these are for doing matvecs with unique
		self.unique, self.unique_inverse = np.unique(
										self.indicies, return_inverse=True) 


	def mul(self,x):
		""" matrix product """
		assert x.ndim == 2
		return x[self.indicies,:]
	dot = __mul__ = mul # make this do the same thing


	def mul_unique(self, x):
		"""
		matrix product with the unique sliced elements of x
		after mul_unique has been called, to recover the full, 
		non-unique entires then `full = unique[S.unique_inverse]`
		"""
		assert x.ndim == 2
		return x[self.unique,:]


	def mul_T(self,x):
		""" matrix-vector product with the transpose """
		raise NotImplementedError('Not finished')

	def __getitem__(self,key):
		if isinstance(key,tuple): # only care about first index
			key = key[0]
		return SelectionMatrixSparse(
				indicies=(np.atleast_1d(self.indicies[key]),self.shape[1]))
