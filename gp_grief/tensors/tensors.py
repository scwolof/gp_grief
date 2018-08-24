
import numpy as np

class TensorProduct(object):
	""" 
	class for performing matrix-vector product with a product of 
	several tensors without expansion 
	"""

	def __init__(self, tensor_list):
		"""
		all tensors must be 2d and have attribues:
			* shape

		and must have methods:
			* __mul__ (for vectors)
			* T (transpose)
		"""
		self.tensors = tensor_list
		self.n_tensors = len(tensor_list)
		self.shape = (self.tensors[0].shape[0], self.tensors[-1].shape[1])

		# check to ensure the shapes are compatible
		for i in range(self.n_tensors-1):
			assert self.tensors[i].shape[1] == self.tensors[i+1].shape[0]

	@property
	def T(self):
		raise NotImplementedError('easy to do this')

	def __mul__(self,x):
		assert x.shape == (self.shape[1], 1), "vector is wrong shape"
		y = x
		for i in reversed(range(self.n_tensors)):
			y = self.tensors[i] * y
		return y


class TensorSum(object):
	""" 
	class for performing matrix-vector product with a sum of 
	several tensors without expansion 
	"""

	def __init__(self, tensor_list):
		"""
		all tensors must be 2d and have attribues:
			* shape

		and must have methods:
			* __mul__ (for vectors)
			* T (transpose)
		"""
		self.tensors = tensor_list
		self.n_tensors = len(tensor_list)
		self.shape = self.tensors[0].shape

		# check to ensure the shapes are compatible
		for i in range(self.n_tensors-1):
			assert self.tensors[i].shape == self.tensors[i+1].shape

	@property
	def T(self):
		raise NotImplementedError('easy to do this')


	def __mul__(self,x):
		assert x.shape == (self.shape[1], 1), "vector is wrong shape"
		y = np.zeros((self.shape[0],1))
		for tensor in self.tensors:
			y += tensor * x
		return y


class Array:
	""" 
	simple lightweight wrapper for numpy.ndarray (or others) that will 
	work with tensor objects 
	"""
	def __init__(self,A):
		self.A = A
		self.shape = A.shape

	def __mul__(self,x):
		return self.A.dot(x)

	@property
	def T(self):
		return Array(self.A.T)

	def expand(self):
		return self.A


def expand_SKC(S, K, C, logged=True):
    """
    Expand selection matrix * kron matrix * column-partitioned KR matrix

    Inputs:
        S : list, row KR matrix of selection matricies
        K : list, kron matix
        C : list, column partitioned Khatri-Rao matrix
        logged : if logged, return log of rows, which is numerically more stable
    """
    assert isinstance(S, (list,np.ndarray))
    assert isinstance(S[0], SelectionMatrixSparse)
    assert isinstance(K, (list,np.ndarray))
    assert isinstance(C, (list,np.ndarray))
    if logged:
        log_prod = 0.
        sign = 1.
    else:
        prod = 1.
    for s,k,c in zip(S, K, C):
    	# just compute the unique rows of the product
        x_unique = s.mul_unique(k).dot(c)
        if logged:
            sign *= np.int32(np.sign(x_unique))[s.unique_inverse]
            x_unique[x_unique == 0] = 1.
            log_prod += np.log(np.abs(x_unique))[s.unique_inverse]
        else:
            prod *= x_unique[s.unique_inverse]
    if logged:
        return log_prod, sign
    else:
        return prod

