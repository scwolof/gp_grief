
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal
from gp_grief.tensors import BlockMatrix, Array

class TestBlockMatrix:
	def test_block_matrix (self):
		np.random.seed(0)
		# initialize random matricies
		a  = np.random.rand(2,3)
		b  = np.random.rand(2,2)
		c  = np.random.rand(3,3)
		d  = np.random.rand(3,2)
		A  = np.vstack((np.hstack((a,b)), np.hstack((c,d))))
		Ab = BlockMatrix(A=np.array([[Array(a),Array(b)],[Array(c),Array(d)]]))

		# test the transpose and expansion operations
		assert_array_almost_equal(A,   Ab.expand()  , decimal=8)
		assert_array_almost_equal(A.T, Ab.T.expand(), decimal=8)

		# initialize random vectors
		x = np.random.rand(A.shape[1],1)
		z = np.random.rand(A.shape[0],1)

		# test matrix vector products
		assert_array_almost_equal(A.dot(x),   Ab  *x, decimal=8)
		assert_array_almost_equal(A.T.dot(z), Ab.T*z, decimal=8)