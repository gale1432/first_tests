import pandas as pd
import numpy as np

def unpack_labels(labels):
    new_series = list()
    for label in labels:
        new_series.append(label[0])
    return pd.Series(new_series)