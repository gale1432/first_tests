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

def root_sum_square(co):
    result = 0
    for x in co:
        #sum += x**2
        result += np.power(x, 2)
    result = np.sqrt(result)
    return result

def peak_to_peak(rss):
    return 2*np.sqrt(2*rss)

def peak_calc(co):
    rss = root_sum_square(co)
    pp = peak_to_peak(rss)
    return [rss, pp]

def get_stats_with_power(co):
    statistical_stats = get_full_stats(co)
    statistical_stats.pop(2)
    statistical_stats.extend(peak_calc(co))
    return statistical_stats