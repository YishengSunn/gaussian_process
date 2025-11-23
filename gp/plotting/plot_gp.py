import numpy as np
import matplotlib.pyplot as plt


def plot_gp(gp, X_s, Y_s, mu_s, cov_s):
    """
    Plot the results
    -------------
    gp : GaussianProcess object
    X_s : (n2, d) - test points
    Y_s : (n2, 1) - true function values at test points
    mu_s : (n2, 1) - predicted mean
    cov_s : (n2, n2) - predicted covariance
    -------------
    """

    std_s = np.sqrt(np.diag(cov_s))

    plt.figure(figsize = (10, 6))
    plt.plot(gp.X_train.ravel(), gp.y_train.ravel(), 'ro', label="Training Points")
    plt.plot(X_s, Y_s, 'y-', label="Real Curve")

    plt.plot(X_s, mu_s, 'b-', label="Predicted Curve")
    plt.fill_between(X_s.ravel(), mu_s.ravel()-3*std_s, mu_s.ravel()+3*std_s, alpha=0.2, label="Confidence Level (3σ)")

    plt.title("Gaussian Process Regression")
    plt.xlabel("x")
    plt.ylabel("f(x)")
    plt.legend()
