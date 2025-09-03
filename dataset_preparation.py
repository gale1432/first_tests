import pandas as pd
import numpy as np
from glob import glob
import os

"""
The methods here are used to get a summary of the labels
employed in the TUSZ database, as well as separate files
depending on their recording circumstances.
"""

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

def delete_ar_files(file_list, name):
    suffix = '03_tcp_ar_a'
    files = pd.read_csv(file_list)
    new_file_list = list()
    kk = 0
    for index, row in files.iterrows():
        if row['0'].find(suffix) != -1:
            new_file_list.append([row['0'].replace('csv', 'edf'), row['1']])
            kk = kk + 1
            print(kk)
    pd.DataFrame(new_file_list).to_csv(name)
    return 'done'

def delete_not_ar_files(file_list, name):
    suffix = '03_tcp_ar_a'
    files = pd.read_csv(file_list)
    new_file_list = list()
    for index, row in files.iterrows():
        if row['0'].find(suffix) == -1:
            new_file_list.append([row['0'].replace('csv', 'edf'), row['1']])
    pd.DataFrame(new_file_list).to_csv(name)
    return 'done'

def delete_bckg_files(file_list, name):
    suffix = 'bckg'
    files = pd.read_csv(file_list)
    new_file_list = list()
    for index, row in files.iterrows():
        if len(row['1']) < 9 and row['1'].find(suffix) != -1:
            continue
        new_file_list.append([row['0'].replace('csv', 'edf'), row['1']])
    pd.DataFrame(new_file_list).to_csv(name)
    return 'done'

def get_file_list(csv_file):
    df = pd.read_csv(csv_file)
    file_list = list()
    for index, row in df.iterrows():
        file_list.append(row['0'])
    return file_list

def get_files_by_seizure(csv_file, sz_list, name='new_seizure_list.csv'):
    df = pd.read_csv(csv_file)
    file_list = list()
    for index, row in df.iterrows():
        for sz in sz_list:
            if row['1'].find(sz) != -1:
                file_list.append([row['0'].replace('csv', 'edf'), row['1']])
                break
    pd.DataFrame(file_list).to_csv(name)
    return 'done'

def delete_files_by_seizure(csv_file, seizure, name='new_seizure_list.csv'):
    df = pd.read_csv(csv_file)
    file_list = list()
    for index, row in df.iterrows():
        if row['1'].find(seizure) == -1:
            file_list.append([row['0'].replace('csv', 'edf'), row['1']])
    pd.DataFrame(file_list).to_csv(name)
    return 'done'