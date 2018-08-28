
import numpy as np

import logging
logger = logging.getLogger(__name__)


class RBF_RFF (object):
    """ random fourier features for an RBF kernel """
    def __init__(self, d, log_lengthscale=0, n_rffs=1000, \
                    dtype=np.float64, tune_len=True):
        """
        squared exponential kernel

        Input:
            d : number of input dims
            n_rffs : number of random features (will actually used twice this value)
        """
        # TODO: add ability to be non ARD
        logger.info("initializing RBF kernel")
        self.d            = int(d)
        self.n_rffs       = int(n_rffs)
        self.n_features   = 2*n_rffs # each random feature is broken into two
        self.dtype        = dtype
        self.freq_weights = np.asarray(\
                                np.random.normal(size=(self.d,self.n_rffs),\
                                loc=0, scale=1.), dtype=self.dtype)
        self.bf_scale     = 1./np.sqrt(self.n_rffs)

        # Set the lengthscale variable
        if np.size(log_lengthscale)==1 and log_lengthscale == 0:
            log_lengthscale = np.zeros((d,1), dtype=self.dtype)
        else:
            log_lengthscale = np.asarray(log_lengthscale, dtype=self.dtype)
            log_lengthscale = log_lengthscale.reshape((d,1))
        self.log_ell = log_lengthscale


    def Phi(self, x):
        """
        Get the basis function matrix

        Inputs:
            x : (n, d) input postions

        Outputs:
            Phi : (n, 2*n_features)
        """
        # scale the frequencies by the lengthscale and multiply with the inputs
        Xfreq = np.dot(x, self.freq_weights/np.exp(self.log_ell)) 
        waves = np.concatenate([np.cos(Xfreq), np.sin(Xfreq)], axis=1)
        return self.bf_scale * waves


