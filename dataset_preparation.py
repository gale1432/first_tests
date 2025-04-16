import pandas as pd
import numpy as np
from glob import glob
import os

def unpack_labels(labels):
    new_series = list()
    for label in labels:
        new_series.append(label[0])
    return pd.Series(new_series)

def get_all_files_summary():
    CED_PATH = os.path.join(os.path.expanduser('~'), 'sda1', 'Documents', 'tuh_eeg', 'edf')
    train_file = glob(CED_PATH + '/train/**/*.csv', recursive=True)
    dev_file = glob(CED_PATH + '/dev/**/*.csv', recursive=True)
    eval_file = glob(CED_PATH + '/eval/**/*.csv', recursive=True)
    pd.DataFrame(get_file_summary(train_file)).to_csv('train_summary.csv')
    pd.DataFrame(get_file_summary(dev_file)).to_csv('dev_summary.csv')
    pd.DataFrame(get_file_summary(eval_file)).to_csv('eval_summary.csv')

def get_file_summary(file_list):
    summary_dict = list()
    for file in file_list:
        file_dataframe = pd.read_csv(file, header=5)
        summary_dict.append([file, file_dataframe['label'].unique()])
    return summary_dict