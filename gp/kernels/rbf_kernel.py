import numpy as np

class RBF_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0):
        self.l = length_scale
        self.sf = sigma_f

    def __call__(self, X1, X2):
        """
        Radial Basis Function kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """

        # Calculate the squared distance between X1_i and X2_j
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)  # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
        return (self.sf**2) * np.exp(-0.5 / self.l**2 * sqdist)  # (n1, n2)
