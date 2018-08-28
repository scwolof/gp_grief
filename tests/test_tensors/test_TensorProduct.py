
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal
from gp_grief.tensors import TensorProduct, Array

class TestTensorProduct:
	def test_tensor_product (self):
		np.random.seed(0)
		A     = Array(np.random.rand(5,3))
		B     = Array(np.random.rand(3,8))
		C     = Array(np.random.rand(8,16))
		D     = Array(np.random.rand(16,12))
		vec   = np.random.rand(12,1)
		exact = A * ( B* (C * D.A) )
		arr   = TensorProduct([A,B,C,D])
		assert_array_almost_equal(exact.dot(vec), arr*vec, decimal=8)