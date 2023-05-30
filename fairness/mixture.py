import numpy as np
from scipy.stats import multivariate_normal, norm


def simulate(n, d, p, p0, p1, mu00, mu01, mu10, mu11, cov0, cov1):
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

    """
    s = np.random.choice(2, n, p=(1 - p, p))
    y = np.zeros(n, dtype='int')
    x = np.zeros((n, d))

    s_mask = (s == 1)

    m = sum(s_mask)

    y[s_mask] = np.random.choice(2, m, p=(1 - p1, p1))
    y[~s_mask] = np.random.choice(2, n - m, p=(1 - p0, p0))
    y_mask = (y == 1)

    mask11 = y_mask & s_mask
    mask01 = ~y_mask & s_mask
    mask10 = y_mask & ~s_mask
    mask00 = ~y_mask & ~s_mask

    x[mask11] = multivariate_normal(mu11, cov1).rvs(sum(mask11)).reshape(-1, d)
    x[mask01] = multivariate_normal(mu01, cov1).rvs(sum(mask01)).reshape(-1, d)
    x[mask10] = multivariate_normal(mu10, cov0).rvs(sum(mask10)).reshape(-1, d)
    # noinspection PyTypeChecker
    x[mask00] = multivariate_normal(mu00, cov0).rvs(sum(mask00)).reshape(-1, d)
    return x, s, y


def eta0(x, p0, mu00, mu10, cov0):
    """ Returns the regression fuction

    .. math::
        \\eta(x, 0)

    Parameters
    ----------
    x: Array like
        The random vector realization
    p0: float
        The conditional probability of Y = 1 given S = 0
    mu00: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=0, Y=0
    mu10: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=0, Y=1
    cov0: Numpy 2D array of shape (d, d)
        The covariance matrix of the conditional distribution X|Y, S=0
    Returns
    -------
        the regression function (posterior/class probabilities)
        for the non-sensitive group
    """
    return _dist(x, p0, mu00, mu10, cov0)


def eta1(x, p1, mu01, mu11, cov1):
    """ Returns the regression fuction

        .. math::
            \\eta(x, 1)

        Parameters
        ----------
        x: Array like
            The random vector realization
        p1: float
            The conditional probability of Y = 1 given S = 1
        mu01: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=1, Y=0
        mu11: Numpy 1D array of shape (d, 1)
            represents the mean vector of the rv: X | S=1, Y=1
        cov1: Numpy 2D array of shape (d, d)
            The covariance matrix of the conditional distribution X|Y, S=0
        Returns
        -------
            the regression function (posterior/class probabilities)
            for the sensitive group
        """
    return _dist(x, p1, mu01, mu11, cov1)


def unfairness(p0, p1, mu00, mu01, mu10, mu11, cov0, cov1):
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


def g0(x, p0, mu00, mu10, cov0):
    """ Returns the bayes classifier

    .. math::
        g^\\star(x, 0) = \\mathbb{1}_{\\eta(x, 0) \\geq 1/2}

    Parameters
    ----------
    x: Array like
        The random vector realization
    p0: float
        The conditional probability of Y = 1 given S = 0
    mu00: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=0, Y=0
    mu10: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=0, Y=1
    cov0: Numpy 2D array of shape (d, d)
        The covariance matrix of the conditional distribution X|Y, S=0

    Returns
    -------
        int or an array of ints corresponding to classes of
        the Bayes classifier for the non-sensitive group
    """
    return int(eta0(x, p0, mu00, mu10, cov0) > .5)


def g1(x, p1, mu01, mu11, cov1):
    """ Returns the bayes classifier for the sensitive group

    .. math::
        g^\\star(x, 1) = \\mathbb{1}_{\\eta(x, 1) \\geq 1/2}

    Parameters
    ----------
    x: Array like
        The random vector realization
    p1: float
        The conditional probability of Y = 1 given S = 1
    mu01: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=1, Y=0
    mu11: Numpy 1D array of shape (d, 1)
        represents the mean vector of the rv: X | S=1, Y=1
    cov1: Numpy 2D array of shape (d, d)
        The covariance matrix of the conditional distribution X|Y, S=0
    Returns
    -------
        int or an array of ints corresponding to classes of
        the Bayes classifier for the sensitive group
    """
    return int(eta1(x, p1, mu01, mu11, cov1) > .5)


def _dist(x, p, mu0, mu1, cov):
    f1 = multivariate_normal(mu1, cov).pdf(x)
    f0 = multivariate_normal(mu0, cov).pdf(x)
    return p * f1 / (p * f1 + (1 - p) * f0)
