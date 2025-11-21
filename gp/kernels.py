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


class Matern_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, nu=1.5):
        self.l = length_scale
        self.sf = sigma_f
        self.nu = nu

    def __call__(self, X1, X2):
        """
        Matern kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """
        # Calculate the pairwise Euclidean distance between X1_i and X2_j
        pairwise_dists = np.linalg.norm((X1[:, np.newaxis, :] - X2[np.newaxis, :, :]), axis=2) / self.l  # (n1, n2)

        if self.nu == 0.5:
            K = self.sf**2 * np.exp(-pairwise_dists)
        elif self.nu == 1.5:
            sqrt3_dists = np.sqrt(3) * pairwise_dists
            K = self.sf**2 * (1 + sqrt3_dists) * np.exp(-sqrt3_dists)
        elif self.nu == 2.5:
            sqrt5_dists = np.sqrt(5) * pairwise_dists
            K = self.sf**2 * (1 + sqrt5_dists + (5/3) * pairwise_dists**2) * np.exp(-sqrt5_dists)

        return K


class RationalQuadratic_kernel:
    def __init__(self, length_scale=1.0, sigma_f=1.0, alpha=1.0):
        self.l = length_scale
        self.sf = sigma_f
        self.alpha = alpha

    def __call__(self, X1, X2):
        """
        Rational Quadratic kernel
        X1 : (n1, d)
        X2 : (n2, d)
        -------------
        returns:
        K : (n1, n2) - kernel matrix
        -------------
        """
        # Calculate the squared distance between X1_i and X2_j
        sqdist = np.sum(X1**2, 1).reshape(-1, 1) + np.sum(X2**2, 1) - 2 * np.dot(X1, X2.T)  # (n1, 1) + (n1, n2) + (n1, n2) = (n1, n2)
        return self.sf**2 * (1 + sqdist / (2 * self.alpha * self.l**2)) ** (-self.alpha)  # (n1, n2)


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