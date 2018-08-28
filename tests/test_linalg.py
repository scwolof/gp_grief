
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from gp_grief.linalg import log_kron

class TestLinAlg:
	def test_log_kron (self):
		np.random.seed(1)
		a = np.random.rand(10)
		b = np.random.rand(4)
		assert_array_almost_equal(np.log(np.kron(a,b)), log_kron(a,b))