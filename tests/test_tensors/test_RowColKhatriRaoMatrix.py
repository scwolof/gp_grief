
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal
from gp_grief.tensors import KronMatrix, KhatriRaoMatrix
from gp_grief.tensors import RowColKhatriRaoMatrix, RowColKhatriRaoMatrixTransposed

class TestRowColKhatriRaoMatrix:
	def test_row_col_khatri_rao_matrix (self):
		np.random.seed(0)
		N, p, d    = 5, 6, 3
		grid_shape = np.random.randint(low=2,high=15,size=d)
		R    = np.empty(d,dtype=object)
		R[:] = [np.random.rand(p,m)-0.5 for m in grid_shape]
		K    = np.empty(d,dtype=object)
		K[:] = [np.random.rand(m,m)-0.5 for m in grid_shape]
		C    = np.empty(d,dtype=object)
		C[:] = [np.random.rand(m,N)-0.5 for m in grid_shape]
		for i in range(d):
		    R[i][0,:] = 0. # set this to zero so there's a zero in the final matrix
		vec  = np.random.rand(N,1) - 0.5
		vecT = np.random.rand(p,1) - 0.5

		# initialize RowKronColKhatriRaoMatrix
		A  = RowColKhatriRaoMatrix(R=R, K=K, C=C)
		# initialize RowColKhatriRaoMatrixTransposed
		AT = RowColKhatriRaoMatrixTransposed(R=R, K=K, C=C)
		# initialize KhatriRaoMatrix's and KronMatrix to test
		R  = KhatriRaoMatrix(R,partition=0)
		C  = KhatriRaoMatrix(C,partition=1)
		K  = KronMatrix(K)

		# test matvec
		assert_array_almost_equal(A*vec, R*(K*(C*vec)))

		# test matvec with transpose
		assert_array_almost_equal(A.T*vecT, C.T*(K.T*(R.T*vecT)))

		# now try with RowColKhatriRaoMatrixTransposed
		assert_array_almost_equal(AT*vecT, C.T*(K.T*(R.T*vecT)))

		# test the expand method to compute the whole matrix
		RKC = R.expand().dot(K.expand().dot(C.expand()))
		assert_array_almost_equal(A.expand(), RKC)

		# test the log expand
		log_A, sign = A.expand(logged=True)
		assert_array_almost_equal(sign*np.exp(log_A), RKC)
