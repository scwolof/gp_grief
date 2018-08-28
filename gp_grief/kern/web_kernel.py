
import numpy as np

class WEBKernel (object):
    """ 
    simple class for parametrizing the weighted basis function kernel 
    """
    def __init__(self, initial_weights):
        assert isinstance( initial_weights, np.ndarray )
        assert np.ndim( initial_weights ) == 1
        self.p           = np.size( initial_weights )
        self.parameters  = initial_weights
        self.constraints = ['+ve',]*self.p