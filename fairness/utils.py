import numpy as np


def empirical_unfairness(y, y_pred, sensitive):
    pos_tpr = (y & y_pred & sensitive).sum() / (y & sensitive).sum()
    neg_tpr = (y & y_pred & ~sensitive).sum() / (y & ~sensitive).sum()
    return abs(pos_tpr - neg_tpr)


def objective(t, y_pred, y_prob, s):
    opt_pred = optimal_decision(t, y_pred, y_prob, s)

    s_mask = s == 1
    y_prob_s1 = y_prob[s_mask]
    y_prob_s0 = y_prob[~s_mask]
    g_theta_s1 = opt_pred[s_mask]
    g_theta_s0 = opt_pred[~s_mask]

    p10 = np.sum(y_pred[~s_mask]) / np.sum(~s_mask)
    p11 = np.sum(y_pred[s_mask]) / np.sum(s_mask)

    return (
        np.mean(y_prob_s0 * g_theta_s0) / p10 -
        np.mean(y_prob_s1 * g_theta_s1) / p11
    )


def recalibrate_predictions(y_pred, y_prob, s, max_iter=100, eps=1e-5, verbose=False):
    theta_min = -2
    theta_max = 2
    theta = 0
    prev_min = 0
    prev_max = 0
    value = objective(theta, y_pred, y_prob, s)
    for i in range(max_iter):
        if abs(value) < eps:
            if verbose:
                print(f'Calibration finished after {i} iterations')
            return theta, value, optimal_decision(theta, y_pred, y_prob, s)
        if value * objective(theta_min, y_pred, y_prob, s) < 0:
            theta_max = theta
        else:
            theta_min = theta

        if prev_min == theta_min and prev_max == theta_max:
            if verbose:
                print('Optimal value reached: '
                      'Extremal values are not updating anymore')
            return theta, value, optimal_decision(theta, y_pred, y_prob, s)
        prev_min = theta_min
        prev_max = theta_max
        theta = 0.5 * (theta_min + theta_max)
        value = objective(theta, y_pred, y_prob, s)

    print('[WARNING]: Maximum number of iterations exceeded.')
    return theta, value, optimal_decision(theta, y_pred, y_prob, s)


def optimal_decision(theta, y_pred, y_prob, s):
    y_opt = np.zeros(len(s), dtype=int)
    s_mask = s == 1
    y_prob_s1 = y_prob[s_mask]
    y_prob_s0 = y_prob[~s_mask]

    prob_y1s1 = np.mean(y_pred & s)
    prob_y1s0 = np.mean(y_pred & ~s)

    y_opt[s_mask] = ((1 - y_prob_s1 * (2 - theta / prob_y1s1)) <= 0).astype(int)
    y_opt[~s_mask] = ((1 - y_prob_s0 * (2 + theta / prob_y1s0)) <= 0).astype(int)
    return y_opt
