
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal
from gp_grief.tensors import KhatriRaoMatrix

class TestKhatriRaoMatrix:
	def test_khatri_rao_matrix (self):
		np.random.seed(0)
		n_rows    = 5
		n_cols    = (2,3,5)
		partition = 0 # row partitioned

		# generate random matricies and initialize Khatri-Rao Matrix
		Araw    = np.empty(len(n_cols),dtype=object)
		Araw[:] = [np.random.rand(n_rows,nc) for nc in n_cols]
		Akr     = KhatriRaoMatrix(A=Araw, partition=partition)

		# expand the Khatri-Rao matrix to use for testing
		Abig = Akr.expand()

		# initialize randome vectors for matrix vector products
		x = np.random.rand(Abig.shape[1],1); 
		z = np.random.rand(Abig.shape[0],1);

		# test matrix vector products
		assert_array_almost_equal(Abig.dot(x),   Akr  *x, decimal=8)
		assert_array_almost_equal(Abig.T.dot(z), Akr.T*z, decimal=8)