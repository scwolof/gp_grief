
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal
from gp_grief.tensors import KronMatrix


"""
SET UP
"""
np.random.seed(0)
d, n = 3, 5
N    = n**d
A    = [np.array(np.random.rand(n,n),order='F') for i in range(d)]


Ab   = 1
for i in range(d):
	Ab = np.kron(Ab,A[i])
Abnorm = np.linalg.norm(Ab)

K     = KronMatrix(A, sym=False)
x     = np.matrix(np.random.rand(n**d,1))
xnorm = np.linalg.norm(x)
lam   = 1e-3

"""
class TestKronMatrixNotSym:

	def test_expansion (self):
		error = np.linalg.norm(K.expand()-Ab)/Abnorm
		assert error <= 1e-10

	def test_transpose (self):
		error = np.linalg.norm(K.T.expand()-Ab.T)/Abnorm
		assert error <= 1e-10

	def test_matrix_vector_product (self):
		error = np.linalg.norm(K*x-Ab.dot(x))/np.linalg.norm(Ab.dot(x))
		assert error <= 1e-10

	def test_solving_linear_system (self):
		error = np.linalg.norm(Ab.dot(K.kronvec_div(x))-x)/xnorm
		assert error <= 1e-10
		# verify consistency
		error = np.linalg.norm(K*(K.kronvec_div(x))-x)/xnorm
		assert error <= 1e-10

	def test_chol (self):
		C     = K.chol()
		# try to reconstruct K
		error =  np.linalg.norm((C.T).kronkron_prod(C).expand() - Ab)/Abnorm
		assert error <= 1e-10
		# solve linear system
		error = np.linalg.norm(K*(C.solve_chol(x))-x)/xnorm
		assert error <= 1e-10

	def test_schur (self):
		Q,T   = K.schur()
		# try to reconstruct K
		Kt    = Q.kronkron_prod(T).kronkron_prod(Q.T).expand()
		error = np.linalg.norm(Kt - Ab)/Abnorm 
		assert error <= 1e-10
		# solve linear system
		error = np.linalg.norm(K*(Q.solve_schur(T,x))-x)/xnorm
		assert error <= 1e-10
		# solve a shifted linear system
		y     = Q.solve_schur(T,x,shift=lam)
		error = np.linalg.norm(K*y + lam*y - x)/xnorm
		assert error <= 1e-10

	def test_svd (self):
		Q,eig_vals = K.svd()
		# reconstruct K
		Kt = Q.expand().dot(np.diag(eig_vals.expand()).dot(Q.T.expand()))
		assert_array_almost_equal(Kt, Ab)
		# solve shifted linear system
		y = Q.solve_schur(eig_vals.expand(),x,shift=lam)
		assert_array_almost_equal(K*y + lam*y, x)
"""