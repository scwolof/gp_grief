
import pytest
import numpy as np 
from numpy.testing import assert_array_almost_equal
from gp_grief.tensors import KronMatrix


"""
SET UP
"""
np.random.seed(1)
d = 10
n = 3
eigs = KronMatrix([np.random.rand(n) for i in range(d)])
all_eigs = eigs.expand() # compute all the eigenvalues for comparison
n_eigs = 5 # this is the number of largest/smallest that I want to find

"""
TESTS
"""
class TestKronEigenvalues:

	def test_False_largest (self):
		log_expand = False
		mode       = 'largest'
		# get the n_eigs largest/smallest
		eig_order, extreme_eigs, global_loc = eigs.find_extremum_eigs(
									n_eigs,mode=mode,log_expand=log_expand,
									sort=True, compute_global_loc=True)
		# check if extreme_eigs is being computed correctly
		assert_array_almost_equal(extreme_eigs,[np.prod([eigs.K[i][eig_order[j,i]] 
								for i in range(d)]) for j in range(n_eigs)])
		# ensure global_loc was computed correctly
		assert_array_almost_equal(extreme_eigs, all_eigs[global_loc], decimal=15)
		# then compare with the brute force expansion
		extreme_eigs_exact = np.sort(all_eigs)[::-1][:n_eigs]
		assert_array_almost_equal(extreme_eigs[::-1],\
								np.sort(extreme_eigs_exact),decimal=15) 

	def test_False_smallest (self):
		log_expand = False
		mode       = 'smallest'
		# get the n_eigs largest/smallest
		eig_order, extreme_eigs, global_loc = eigs.find_extremum_eigs(
									n_eigs,mode=mode,log_expand=log_expand,
									sort=True, compute_global_loc=True)
		# check if extreme_eigs is being computed correctly
		assert_array_almost_equal(extreme_eigs,[np.prod([eigs.K[i][eig_order[j,i]] 
								for i in range(d)]) for j in range(n_eigs)])
		# ensure global_loc was computed correctly
		assert_array_almost_equal(extreme_eigs, all_eigs[global_loc], decimal=15)
		# then compare with the brute force expansion
		extreme_eigs_exact = np.sort(all_eigs)[:n_eigs]
		assert_array_almost_equal(extreme_eigs[::-1],\
								np.sort(extreme_eigs_exact),decimal=15) 

	def test_True_largest (self):
		log_expand = True
		mode       = 'largest'
		# get the n_eigs largest/smallest
		eig_order, extreme_eigs, global_loc = eigs.find_extremum_eigs(
									n_eigs,mode=mode,log_expand=log_expand,
									sort=True, compute_global_loc=True)
		extreme_eigs = np.exp(extreme_eigs)
		# check if extreme_eigs is being computed correctly
		assert_array_almost_equal(extreme_eigs,[np.prod([eigs.K[i][eig_order[j,i]] 
								for i in range(d)]) for j in range(n_eigs)])
		# ensure global_loc was computed correctly
		assert_array_almost_equal(extreme_eigs, all_eigs[global_loc], decimal=15)
		# then compare with the brute force expansion
		extreme_eigs_exact = np.sort(all_eigs)[::-1][:n_eigs]
		assert_array_almost_equal(extreme_eigs[::-1],\
								np.sort(extreme_eigs_exact),decimal=15) 

	def test_True_smallest (self):
		log_expand = True
		mode       = 'smallest'

		# get the n_eigs largest/smallest
		eig_order, extreme_eigs, global_loc = eigs.find_extremum_eigs(
									n_eigs,mode=mode,log_expand=log_expand,
									sort=True, compute_global_loc=True)
		extreme_eigs = np.exp(extreme_eigs)
		# check if extreme_eigs is being computed correctly
		assert_array_almost_equal(extreme_eigs,[np.prod([eigs.K[i][eig_order[j,i]] 
								for i in range(d)]) for j in range(n_eigs)])
		# ensure global_loc was computed correctly
		assert_array_almost_equal(extreme_eigs, all_eigs[global_loc], decimal=15)
		# then compare with the brute force expansion
		extreme_eigs_exact = np.sort(all_eigs)[:n_eigs]
		assert_array_almost_equal(extreme_eigs[::-1],\
								np.sort(extreme_eigs_exact),decimal=15)

	def test_log_det (self):
		for sym in [True,False]:
			np.random.seed(0)
			A   = [np.random.rand(5,5)+np.eye(5) for i in range(2)]
			A   = [Ai.dot(Ai.T)+1e-6*np.eye(5) for Ai in A] # make it SPD
			A   = KronMatrix(A,sym=sym)
			eig = A.eig_vals()
			assert_array_almost_equal(eig.log_det(),\
									np.linalg.slogdet(A.expand())[1])

	def test_flipping_shuffling (self):
		np.random.seed(0)
		shapes = [(2,3), (2,2), (5,2)] # sizes of submatricies
		d      = len(shapes)

		# first do the exact computation
		K = KronMatrix([np.random.rand(*shape) for shape in shapes])
		x = np.random.rand(K.shape[1], 1)
		y = K*x

		# now shuffle K and the vector x and try to recover y
		for i in range(1,d): # i is the index which should go first
			# do the forward shuffle
			shuffle = np.concatenate(([i,], np.delete(np.arange(d), i)))
			K_s     = KronMatrix([K.K[axis] for axis in shuffle])
			X       = x.reshape(np.asarray(shapes).T[1]) 
			x_s     = np.transpose(X, shuffle).reshape((-1,1)) 
			y_s     = K_s * x_s

			# now reverse the shuffle in y
			new_shapes = [shapes[j] for j in shuffle]
			reverse    = np.squeeze([np.where(shuffle==j)[0] for j in range(d)])
			Y_s        = y_s.reshape(np.asarray(new_shapes).T[0])
			yy         = np.transpose(Y_s, reverse).reshape((-1,1))
			assert_array_almost_equal(yy,y)	