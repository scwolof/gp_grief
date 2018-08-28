
import numpy as np
import scipy.sparse as sparse
from gp_grief.tensors import KronMatrix, SelectionMatrix,\
							SelectionMatrixSparse, BlockMatrix


class KhatriRaoMatrix (BlockMatrix):
	""" a Khatri-Rao Matrix (block Kronecker Product matrix) """

	def __init__(self, A, partition=None):
		"""
		Khatri-Rao Block Matrix.

		Inputs:
			A : list of sub matricies or 2d array of KronMatricies. 
				If the latter then partition is ignored.
			partition : int specifying the direction that the 
				Khatri-Rao Matrix is partitioned:
				0 : row partitioned
				1 : column partitioned
				If A is an array of KronMatricies then this has now effect.
		"""
		# determine whether KronMatricies have been formed from the partitions
		if np.ndim(A)==2 and isinstance(A[0,0], KronMatrix): 
			# all the work is done
			super(KhatriRaoMatrix, self).__init__(A)
			return

		# else I need to create KronMatrices from each partition
		# get the number of blocks that will be needed
		assert partition in range(2)
		if partition == 0:
			block_shape = (A[0].shape[0], 1)
		elif partition == 1:
			block_shape = (1,A[0].shape[1])
		else:
			raise ValueError('unknown partition')

		# form the KronMatricies
		Akron = np.empty(max(block_shape), dtype=object) # make 1d, reshape later
		for i in range(max(block_shape)):
			if partition == 0:
				Akron[i] = KronMatrix([Aj[(i,),:] for Aj in A], sym=False)
			elif partition == 1:
				Akron[i] = KronMatrix([Aj[:,(i,)] for Aj in A], sym=False)
		Akron = Akron.reshape(block_shape)

		# Create a BlockMatrix from this
		super(KhatriRaoMatrix, self).__init__(Akron)


class RowColKhatriRaoMatrix (object):
	""" 
	matrix formed by R K C allowing memory efficient matrix-vector products 
	"""

	def __init__(self,R,K,C,nGb=1.):
		"""
		Inputs:
			R : list or np.ndarray
			K : list or np.ndarray, K can be None if there is no K
			C : list or np.ndarray
			nGb : number of gigabytes of memory that shouldn't be exceeded when
				performing mvproducts. If specified the multiple rows will be
				computed at once which allows for use of BLAS level-3 routines
		Note:
			* by default, KC will be merged therefore R should be sparse if any. 
			If C is sparse then you should either:
				use the ...Transposed class below
		"""
		self.shape = (R[0].shape[0],C[0].shape[1])
		self.d = len(R)
		if K is not None:
			K = np.asarray(K)
			assert len(K) == len(C) == self.d, "number of dims inconsistent"

			# merge K and C into self.C (row-partitioned Khatri-Rao prod matrix)
			self.R = R
			self.C = np.empty(self.d,dtype=object)
			for i in range(self.d): # ensure submatricies are consistent
				assert K[i].shape[0] == K[i].shape[1] == R[i].shape[1],\
					"K must be a square Kronecker product matrix,"\
					+"and must be consistent with R"
				self.C[i] = K[i].dot(C[i])
		else: # there is no K
			self.R = R
			self.C = C

		# figure out how many rows should be computed at once during matvec
		self.n_rows_at_once = 1 # default one at a time
		if nGb is not None:
			self.n_rows_at_once = \
					max(1,np.int32(np.floor(nGb*1e9/(8*self.shape[1]))))
		self.nGb = nGb

	@property
	def T(self):
		""" transpose operation """
		if isinstance(self.R[0], (SelectionMatrix,SelectionMatrixSparse)):
			# Don't want to transpose explicitly
			return RowColKhatriRaoMatrixTransposed(
									R=self.R, K=None, C=self.C, nGb=self.nGb)
		else: 
			# Compute the transpose
			return RowColKhatriRaoMatrix(R=[Ci.T for Ci in self.C], K=None, 
									C=[Ri.T for Ri in self.R], nGb=self.nGb)


	def get_rows(self, i_rows, logged=False):
		"""
		compute i_rows rows of the packed matrix
		a can be a vector or a slice object (ie. i_rows=slice(None) 
			will return the whole matrix)

		if logged, return the log of the rows, which is numerically more stable
		"""
		if logged:
			rows = 0.
			sign = 1.
		else:
			rows = 1.
		for i_d in range(self.d):
			if sparse.issparse(self.C[i_d]): 
				# have to treat this differently as of numpy 1.7
				# see http://stackoverflow.com/questions/31040188/
				rows_1d = self.R[i_d][i_rows,:]  *  self.C[i_d]
			else:
				rows_1d = self.R[i_d][i_rows,:].dot(self.C[i_d])
			if logged:
				sign *= np.int32(np.sign(rows_1d))
				rows_1d[sign == 0] = 1.
				rows += np.log(np.abs(rows_1d))
			else:
				rows *= rows_1d
		if logged:
			return rows, sign
		else:
			return rows


	def expand(self, logged=False):
		"""
		expand the product of matricies.
		this is very similar to the get_rows routine since that just computes 
		the matrix a few rows at a time whereas here we do it all in one shot 
		since we assume it is practical to store the entire matrix in memory

		if logged, return the log of the matrix which is numerically more stable
		"""
		return self.get_rows(i_rows=slice(None), logged=logged)


	def __mul__(self,x):
		"""
		memory efficient way to compute a matrix-vector product with a row and 
		column partitioned Khatri-Rao product matrix.
		This is the same as mvKRrowcol from the ICML submission
		"""
		assert x.shape == (self.shape[1],1)
		i = 0 # initialize counter
		y = np.zeros((self.shape[0],1))
		while i < self.shape[0]: # loop accross rows of the matrix
			i_rows = np.arange(i,min(i+self.n_rows_at_once,self.shape[0]))
			y[i_rows,:] = self.get_rows(i_rows).dot(x) # compute matvec
			i = i_rows[-1] + 1 # increment counter
		return y


class RowColKhatriRaoMatrixTransposed (RowColKhatriRaoMatrix):
	""" 
	thin wrapper for when the row KR is sparse before transposing 
	"""
	def __init__(self,*args,**kwargs):
		super(RowColKhatriRaoMatrixTransposed, self).__init__(*args,**kwargs)
		self.shape = self.shape[::-1] # flip this

		# need to redo, since now really slicing columns (and then transposing)
		self.n_rows_at_once = 1 # default one at a time
		if self.nGb is not None:
			self.n_rows_at_once = \
					max(1,np.int32(np.floor(self.nGb*1e9/(8*self.shape[1]))))


	def get_rows(self,i_rows):
		"""
		compute i_rows columns of the packed matrix and then transpose
		"""
		if sparse.issparse(self.C[0]): 
			# have to treat this differently as of numpy 1.7
			# see http://stackoverflow.com/questions/31040188/
			cols = self.R[0] * self.C[0][:,i_rows]
		else:
			cols = self.R[0].dot(self.C[0][:,i_rows])
		for i_d in range(1,self.d):
			if sparse.issparse(self.C[i_d]): 
				# have to treat this differently as of numpy 1.7
				cols *= self.R[i_d]  *  self.C[i_d][:,i_rows]
			else:
				cols *= self.R[i_d].dot(self.C[i_d][:,i_rows])
		return cols.T

	@property
	def T(self):
		""" transpose operation """
		# since its already a transposed type matrix, I just need to 
		# respecify as the non-transposed class
		return RowColKhatriRaoMatrix(R=self.R, K=None, C=self.C, nGb=self.nGb)


