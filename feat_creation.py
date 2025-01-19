import scipy as sc
import numpy as np

def get_stats(co):
    min1 = np.min(co)
    max1 = np.max(co)
    mean = np.mean(co)
    sd = np.std(co)
    var = np.var(co)
    kurt = sc.stats.kurtosis(co)
    skewness = sc.stats.skew(co)
    return [mean, sd, var, kurt, skewness]