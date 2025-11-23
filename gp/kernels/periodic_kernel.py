import numpy as np

class Periodic_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, w=1.0):
        self.l = length_scale
        self.sf = sigma_f
        self.w = w

    def __call__(self, X1, X2):
        """
        Periodic kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """
        # Calculate the pairwise Euclidean distance between X1_i and X2_j
        pairwise_dists = np.linalg.norm((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2)  # (n1, n2)
        return self.sf**2 * np.exp(-1 * (np.sin(np.pi * pairwise_dists / self.w)**2) / (self.l**2))  # (n1, n2)
