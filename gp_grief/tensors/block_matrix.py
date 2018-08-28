
import numpy as np

class BlockMatrix (object):
	""" 
	Create Block matrix 
	"""

	def __init__(self, A):
		"""
		Builds block matrix with which matrix-vector multiplication can be done.

		Inputs:
			A : numpy object array of blocks of size (h, w)
				i.e. A = np.array([[ A_11, A_12, ... ],
								   [ A_21, A_22, ... ], ... ]
				Each block in A must have the methods
				* shape
				* __mul__
				* T (transpose property)
				* expand (only nessessary if is to be used)
		"""
		assert A.ndim == 2, 'A must be 2d'
		self.A = A # shallow copy

		# get the shapes of the matricies
		self.block_shape = self.A.shape # shape of the block matrix
		self._partition_shape = ([A_i0.shape[0] for A_i0 in self.A[:,0]], 
			[A_0i.shape[1] for A_0i in self.A[0,:]]) # shape of each partition
		self.shape = tuple([np.sum(self._partition_shape[i]) for i in range(2)])

		# ensure the shapes are consistent for all partitions
		for i in range(self.block_shape[0]):
			for j in range(self.block_shape[1]):
				assert np.all(A[i,j].shape == self.partition_shape(i,j)),\
					"A[%d,%d].shape should be %s, not %s"\
					% (i,j,repr(self.partition_shape(i,j)),repr(A[i,j].shape))

		# define how passed vector should be split for matrix vector product
		self.vec_split =  np.cumsum([0,] + self._partition_shape[1], dtype='i')


	def partition_shape(self, i, j):
		""" returns the shape of A[i,j] """
		return (self._partition_shape[0][i],self._partition_shape[1][j])


	def __mul__( self, x ):
		""" matrix vector multiplication """
		assert x.shape == (self.shape[1], 1)

		# first split the vector x so I don't have to make so many slices
		xs = [x[self.vec_split[j]:self.vec_split[j+1],:] 
				for j in range(self.block_shape[1])]

		# loop through each block row and perform the matrix-vector product
		y = np.empty(self.block_shape[0], dtype=object)
		for i in range(self.block_shape[0]):
			y[i] = 0 # initialize
			for j in range(self.block_shape[1]): # loop accross the row
				y[i] += self.A[i,j] * xs[j]

		# concatenate results
		y = np.concatenate(y,axis=0)
		return y


	def transpose(self):
		""" 
		transpose kronecker product matrix. 
		This currently copies the matricies explicitly 
		"""
		A = self.A.copy()

		# first transpose each block individually
		for i in range(self.block_shape[0]):
			for j in range(self.block_shape[1]):
				A[i,j] = A[i,j].T

		# then, transpose globally
		A = A.T

		# then return a new instance of the object
		return self.__class__(A=A)
	T = property(transpose)

	def expand(self):
		""" expands each block matrix to form a big, full matrix """
		Abig = np.zeros(np.asarray(self.shape, dtype='i'))
		row_split = np.cumsum([0,] + self._partition_shape[0], dtype='i')
		col_split = np.cumsum([0,] + self._partition_shape[1], dtype='i')
		for i in range(int(round(self.block_shape[0]))):
			for j in range(int(round(self.block_shape[1]))):
				Abig[row_split[i]:row_split[i+1], col_split[j]:col_split[j+1]] \
					= self.A[i,j].expand()
		return Abig
