import scipy as sc
import numpy as np

def get_stats(co):
    mean = np.mean(co)
    sd = np.std(co)
    var = np.var(co)
    kurt = sc.stats.kurtosis(co)
    skewness = sc.stats.skew(co)
    return [mean, sd, var, kurt, skewness]

def get_full_stats(co):
    min1 = np.min(co)
    max1 = np.max(co)
    first_stats = get_stats(co)
    first_stats.extend([min1, max1])
    return first_stats