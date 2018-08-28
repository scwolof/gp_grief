
import pytest
import numpy as np 
from numpy.testing import assert_array_equal
from gp_grief.tensors import SelectionMatrix, SelectionMatrixSparse

class TestSelectionMatrix:
	def test_selection_matrix (self):
		np.random.seed(0)
		A = np.random.rand(20,20)
		sel = np.random.choice(A.shape[0], size=30)

		# check SelectionMatrix
		S = SelectionMatrix((sel, A.shape[0]))
		assert_array_equal(A[sel], S.mul(A)) 

		# check SelectionMatrixSparse
		S = SelectionMatrixSparse((sel, A.shape[0]))
		assert_array_equal(A[sel], S.mul(A)) 

		# check if able to perform unique subset then expand
		assert_array_equal(A[sel], S.mul_unique(A)[S.unique_inverse])