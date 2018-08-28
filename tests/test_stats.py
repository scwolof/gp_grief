
import pytest
import numpy as np
from numpy.testing import assert_array_almost_equal
from gp_grief.stats import StreamMeanVar

class TestStats:
	def test_stats (self):
		np.random.seed(0)
		N    = 100
		data = np.random.rand(N,10)
		ov   = StreamMeanVar(ddof=0)
		for d in data:
		    ov.include(d)
		assert_array_almost_equal(data.std(axis=0), ov.std, decimal=10)
		assert_array_almost_equal(data.var(axis=0), ov.variance, decimal=10)
		assert_array_almost_equal(data.mean(axis=0), ov.mean, decimal=10)