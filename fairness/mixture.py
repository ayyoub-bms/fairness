import numpy as np
from scipy.stats import multivariate_normal, norm


def generate_samples(n, d, p, p0, p1, mu00, mu01, mu10, mu11, cov0, cov1):
    """Simulate the following data
        S ~ B(p)
        Y | S = 1 ~ B(p1)
        Y | S = 0 ~ B(p0)

        X | Y = 1, S = 1 ~ N(μ11, Σ1)
        X | Y = 1, S = 0 ~ N(μ10, Σ0)
        X | Y = 0, S = 1 ~ N(μ01, Σ1)
        X | Y = 0, S = 0 ~ N(μ00, Σ0)

    Parameters
    ----------
    n: int
        The number of samples
    d: ind
        The dimension of features (the random vector X)
    p: float
        The probability that S = 1
    p0: float
        The conditional probability of Y = 1 given S = 0
    p1: float
        The conditional probability of Y = 1 given S = 1
    mu00: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=0, Y=0
    mu01: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=1, Y=0
    mu10: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=0, Y=1
    mu11: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=1, Y=1
    cov0: Numpy 2D array of shape (d, d)
        The covariance matrix of the conditional distribution X|Y, S=0
    cov1: Numpy 2D array of shape (d, d)
        The covariance matrix of the conditional distribution X|Y, S=1

    Returns
    -------
        a tuple (features, target, sensitive) of random samples
        from the mixture model.
    """
    s = np.random.choice(2, n, p=(1 - p, p))
    y = np.zeros(n, dtype='int')
    x = np.zeros((n, d))

    s_mask = (s == 1)

    m = sum(s_mask)

    y[s_mask] = np.random.choice(2, m, p=(1 - p1, p1))
    y[~s_mask] = np.random.choice(2, n - m, p=(1 - p0, p0))
    y_mask = (y == 1)

    mask11 = np.where(y_mask & s_mask)[0]
    mask01 = np.where(~y_mask & s_mask)[0]
    mask10 = np.where(y_mask & ~s_mask)[0]
    mask00 = np.where(~y_mask & ~s_mask)[0]

    x[mask11] = multivariate_normal(mu11, cov1).rvs(len(mask11))
    x[mask01] = multivariate_normal(mu01, cov1).rvs(len(mask01))
    x[mask10] = multivariate_normal(mu10, cov0).rvs(len(mask10))
    x[mask00] = multivariate_normal(mu00, cov0).rvs(len(mask00))

    return x, s, y


def estimate(X, y, s):
    mu00, mu01, mu10, mu11 = _class_means(X, y, s)
    cov0, cov1 = _class_cov(X, s)
    p0 = y[np.where(s == 0)[0]].mean()
    p1 = y[np.where(s == 1)[0]].mean()
    args = (p0, p1, mu00, mu01, mu10, mu11, cov0, cov1)
    return args


def regression_function(X, s, p0, p1, mu00, mu01, mu10, mu11, cov0, cov1):
    """ Returns the regression fuction

        .. math::
            \\eta(x, s)

        Parameters
        ----------
        X: Numpy 2D array  of shape (n, d)
            The random samples
        s: Numpy 1D array of shape (n, 1)
            The samples of the sensitive feature
        p0: float
            The conditional probability of Y = 1 given S = 0
        p1: float
            The conditional probability of Y = 1 given S = 1
        mu00: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=0, Y=0
        mu10: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=0, Y=1
        mu01: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=1, Y=0
        mu11: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=1, Y=1
        cov0: Numpy 2D array of shape (d, d)
            The covariance matrix of the conditional distribution X|Y, S=0
        cov1: Numpy 2D array of shape (d, d)
            The covariance matrix of the conditional distribution X|Y, S=0
        Returns
        -------
            The regression function (posterior/class probabilities)
            for the sensitive and non-sensitive group
        """
    reg_function = np.zeros(X.shape[0])
    s_mask = s == 1
    reg_function[~s_mask] = _dist(X[~s_mask], p0, mu00, mu10, cov0)
    reg_function[s_mask] = _dist(X[s_mask], p1, mu01, mu11, cov1)
    return reg_function


def bayes_classifier(X, s, p0, p1, mu00, mu01, mu10, mu11, cov0, cov1):
    """ Returns the bayes classifier

        .. math::
            g^\\star(x, s) = \\mathbb{1}_{\\eta(x, s) \\geq 1/2}

        Parameters
        ----------
        X: Numpy 2D array  of shape (n, d)
            The random samples
        s: Numpy 1D array of shape (n, 1)
            The samples of the sensitive feature
        p0: float
            The conditional probability of Y = 1 given S = 0
        p1: float
            The conditional probability of Y = 1 given S = 1
        mu00: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=0, Y=0
        mu10: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=0, Y=1
        mu01: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=1, Y=0
        mu11: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=1, Y=1
        cov0: Numpy 2D array of shape (d, d)
            The covariance matrix of the conditional distribution X|Y, S=0
        cov1: Numpy 2D array of shape (d, d)
            The covariance matrix of the conditional distribution X|Y, S=0

        Returns
        -------
            A numpy array of ints corresponding to classes of
            the Bayes classifier for the  sensitive and non-sensitive group
        """
    r = regression_function(X, s, p0, p1, mu00, mu01, mu10, mu11, cov0, cov1)
    return (r > .5).astype(int)


def equal_opportunity(p0, p1, mu00, mu01, mu10, mu11, cov0, cov1):
    """ Computs the unfairness using the Equal Opportunity measure:

    .. math::

        | \\mathbb{P}(g(X, 1) = 1 | Y = 1, S = 1)
        - \\mathbb{P}(g(X, 1) = 1 | Y = 1, S = 0)|

    Parameters
    ----------
    p0: float
        Represents the probability: P(Y=1|S=0)
    p1: float
        Represents the probability: P(Y=1|S=1)
    mu00: np.array
        represents the mean vector of the rv: X | S=0, Y=0
    mu01: np.array
        represents the mean vector of the rv: X | S=1, Y=0
    mu10: np.ndarray
        represents the mean vector of the rv: X | S=0, Y=1
    mu11: np.ndarray
        represents the mean vector of the rv: X | S=1, Y=1
    cov0: np.ndarray
        The covariance matrix of the conditional distribution X|Y, S=0
    cov1: np.ndarray
        The covariance matrix of the conditional distribution X|Y, S=1

    Returns
    -------
        The Equal opportunity as a measure of unfairness associated with
        the Bayes classifier.
    """
    a1 = np.log(p1 / (1 - p1))
    a0 = np.log(p0 / (1 - p0))

    cov0_inv = np.linalg.pinv(cov0)
    cov1_inv = np.linalg.pinv(cov1)

    m0 = (mu00 - mu10).T @ cov0_inv
    m1 = (mu01 - mu11).T @ cov1_inv

    mu0 = m0 @ mu10
    mu1 = m1 @ mu11

    s0 = m0 @ (mu00 - mu10)
    s1 = m1 @ (mu01 - mu11)

    x0 = a0 + 0.5 * mu11.T @ cov1_inv @ (mu01 + mu11)
    x1 = a1 + 0.5 * mu10.T @ cov0_inv @ (mu10 + mu00)

    return np.abs(norm.cdf(x1, loc=mu1, scale=np.sqrt(s1)) -
                  norm.cdf(x0, loc=mu0, scale=np.sqrt(s0)))


def _dist(x, p, mu0, mu1, cov):
    f1 = multivariate_normal(mu1, cov).pdf(x)
    f0 = multivariate_normal(mu0, cov).pdf(x)
    return p * f1 / (p * f1 + (1 - p) * f0)


def _class_means(X, y, s):
    y_mask = y == 1
    s_mask = s == 1

    mask11 = np.where(y_mask & s_mask)
    mask01 = np.where(~y_mask & s_mask)
    mask10 = np.where(y_mask & ~s_mask)
    mask00 = np.where(~y_mask & ~s_mask)

    mu00 = np.mean(X[mask00], axis=0)
    mu01 = np.mean(X[mask01], axis=0)
    mu10 = np.mean(X[mask10], axis=0)
    mu11 = np.mean(X[mask11], axis=0)
    return mu00, mu01, mu10, mu11


def _class_cov(X, s):
    s_mask = s == 1
    cov1 = np.cov(X[s_mask].T)
    cov0 = np.cov(X[~s_mask].T)
    return cov0, cov1
