# Classes for visualizing results
# Much of this is adapted from Gianmario Spacagna
# https://github.com/gm-spacagna/deep-ttf

import matplotlib.pyplot as plt
import numpy as np
from math import max

def weibull_quantile(alpha, beta, p):
    return alpha*(-np.log(1-p))**(1/beta)

def weibull_pdf(alpha, beta, t):
    return (beta/alpha) * (t/alpha)**(beta-1)*np.exp(- (t/alpha)**beta)

def weibull_median(alpha, beta):
    return weibull_quantile(alpha, beta, 0.5)

def weibull_mean(alpha, beta):
    return alpha * math.gamma(1 + 1/beta)

def weibull_mode(alpha, beta):
    assert np.all(beta > 1)
    return alpha * ((beta-1)/beta)**(1/beta)

def plot_weibull_predictions(results_df, moment='mode'):
    """Plot the distributions vs. actual, mode vs. actual, and distribution of such residuals
    :param results_df: pandas.DataFrame with fields `rul`, `alpha`, and `beta`
    Adapted nearly verbatim from Gianmario Spacagna https://github.com/gm-spacagna/deep-ttf
    """

    if moment == 'mode':
        fun, desc = weibull_mode, 'Mode'
    elif moment == 'mean':
        fun, desc = weibull_mean, 'Mean'
    elif moment == 'median':
        fun, desc = weibull_median, 'Median'
    else:
        raise ValueError('moment must be one of: "mode", "mean", "median"')

    fig, axs = plt.subplots(3)

    # Get the linspace based on 1.5x the max predicted or actual TTE
    max_moment = results_df.apply(lambda r: fun(r['alpha'], r['beta']), axis=1).max()
    max_actual = results_df['rul'].max()
    t = np.arange(0, 1.5*max(max_moment, max_actual))

    nobs = results_df.shape[0]
    palette = sns.color_palette("RdBu_r", nobs + 1)
    color_dict = dict(enumerate(palette))

    # Subplot 1: evolution of prediction distributions over rows
    ax = axs[0]
    for i, (index, row, _) in enumerate(results_df.iterrows()):
        alpha = row['alpha']
        beta  = row['beta']
        rul   = row['rul']
        color = color_dict[i]
        yhat = fun(alpha, beta)
        ymax = weibull_pdf(alpha, beta, yhat)    

        _ = ax.plot(t, weibull_pdf(alpha, beta, t), color=color)
        _ = ax.scatter(rul, weibull_pdf(alpha,beta, rul), color=color, s=100)
        _ = ax.vlines(yhat, ymin=0, ymax=ymax, colors=color, linestyles='--')
        _ = ax.set_xlabel('Time-to-event')
        _ = ax.set_ylabel('Density')
        _ = ax.set_title('Weibull distributions of predictions')
    
    # Subplot 2: scatterplot of prediction point estimate vs. actual
    ax = axs[1]
    yhat = results_df.apply(lambda r: fun(r['alpha'], r['beta']), axis=1)
    _ = ax.scatter(results_df['rul'], yhat, label=desc)
    _ = ax.set_xlabel('Actual time-to-event')
    _ = ax.set_ylabel('{} predicted time-to-event'.format(desc))
    _ = ax.set_title('{} prediction vs. true time-to-event'.format(desc))
    
    # Subplot 3: distribution of residuals
    ax = axs[2]
    _ = ax.hist(results_df['rul'] - yhat)
    _ = ax.set_xlabel('Actual minus {} predicted time-to-event'.format(desc))
    _ = ax.set_title('Distribution of errors')
    plt.show()

def plot_predictions_over_time(df_results, moment='mode'):
    """Plot the distributions vs. actual, mode vs. actual, and distribution of such residuals
    :param results_df: pandas.DataFrame with fields `rul`, `alpha`, and `beta`
    Inspired by Gianmario Spacagna https://github.com/gm-spacagna/deep-ttf
    """

    if moment == 'mode':
        fun, desc = weibull_mode, 'Mode'
    elif moment == 'mean':
        fun, desc = weibull_mean, 'Mean'
    elif moment == 'median':
        fun, desc = weibull_median, 'Median'
    else:
        raise ValueError('moment must be one of: "mode", "mean", "median"')

    df = results_df[['rul','alpha','beta']].sort_values('rul', ascending=False).reset_index(drop=True)
    
    y = df['rul']
    yhat = df.apply(lambda r: fun(r['alpha'], r['beta']), axis=1)
    p10  = df.apply(lambda r: weibull_quantile(r['alpha'], r['beta']), axis=1)
    p90  = df.apply(lambda r: weibull_quantile(r['alpha'], r['beta']), axis=1)

    _ = plt.plot(y, label='Actual')
    _ = plt.plot(yhat, label='{} predicted'.format(desc))
    _ = plt.fill_between(df.index.values, p10.values, p90.values, color='gray', alpha=0.25,
                         label='P90-P10 interval')
    _ = plt.set_xlabel('Observation')
    _ = plt.set_ylabel('Time-to-event')
    _ = plt.set_title('{} predictions and actuals over sequence'.format(desc))
    _ = plt.legend()
    plt.show()
