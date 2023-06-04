import numpy as np


def empirical_unfairness(y, y_pred, sensitive):
    pos_tpr = (y & y_pred & sensitive).sum() / (y & sensitive).sum()
    neg_tpr = (y & y_pred & ~sensitive).sum() / (y & ~sensitive).sum()
    return abs(pos_tpr - neg_tpr)


def objective(t, y_prob, s):
    opt_pred = optimal_decision(t, y_prob, s)

    s_mask = s == 1
    y_prob_s1 = y_prob[s_mask]
    y_prob_s0 = y_prob[~s_mask]
    g_theta_s1 = opt_pred[s_mask]
    g_theta_s0 = opt_pred[~s_mask]


    return (
        np.mean(y_prob_s0 * g_theta_s0) / np.mean(y_prob_s0) -
        np.mean(y_prob_s1 * g_theta_s1) / np.mean(y_prob_s1)
    )


def recalibrate_predictions(y_prob, s, max_iter=100, eps=1e-5, verbose=False):
    
    theta_min = -2
    theta_max = 2
    theta = 0
    prev_min = 0
    prev_max = 0
    value = objective(theta, y_prob, s)
    stop = False
    it = 1

    while not stop:

        if it == max_iter:
            print('[WARNING]: Maximum number of iterations exceeded.')
            stop = True

        if abs(value) < eps:
            stop = True
            if verbose:
                print(f'Calibration finished after {it} iterations')

        if round(prev_min, 6) == round(theta_min, 6) and round(prev_max, 6) == round(theta_max, 6):
            if verbose:
                print('Extremal values are not updating anymore')
            stop = True
        
        prev_min = theta_min
        prev_max = theta_max
        
        if value * objective(theta_min, y_prob, s) < 0:
            theta_max = theta
        else:
            theta_min = theta

        theta = 0.5 * (theta_min + theta_max)
        value = objective(theta, y_prob, s)
        
        it += 1

    if abs(theta) < 1e-10:
        theta = 0
        value = objective(theta, y_prob, s)

    return theta, value, optimal_decision(theta, y_prob, s)


def optimal_decision(theta, y_prob, s):
    y_opt = np.zeros(len(s), dtype=int)
    p1 = np.mean(s)
    s_mask = s == 1

    y_prob_s1 = y_prob[s_mask]
    y_prob_s0 = y_prob[~s_mask]

    y_opt[s_mask] = ((1 - y_prob_s1 * (2 - theta / (p1 * np.mean(y_prob_s1)))) <= 0).astype(int)
    y_opt[~s_mask] = ((1 - y_prob_s0 * (2 + theta / ((1-p1)*np.mean(y_prob_s0)))) <= 0).astype(int)

    return y_opt


def get_npv_optimal_params(param_list, accuracies, equal_opportunities, threshold=.9):
    max_accuracy = np.max(accuracies)
    indices = np.where(accuracies > threshold * max_accuracy)[0]
    index = np.argmin(equal_opportunities[indices])
    return param_list[indices][index]
