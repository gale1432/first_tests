from glob import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path as pt
import joblib
import scipy as sp

from mne.conftest import event_id
from sklearn.preprocessing import LabelEncoder

#mne.viz.set_browser_backend("qt")

EVENT_DICT = {
    'fnsz': 1,
    'gnsz': 2,
    'spsz': 3,
    'cpsz': 4,
    'absz': 5,
    'tnsz': 6,
    'cnsz': 7,
    'tcsz': 8,
    'atsz': 9,
    'mysz': 10,
    'nesz': 11,
    'bckg': 12
}
CED_PATH='C:\\Users\\arsms\\Documents\\tuh_eeg\\edf'
HOME_PATH='E:\\Files\\tuh_eeg\\edf\\train'

train_file = glob(CED_PATH+'\\train\\**/*.edf', recursive=True)
print(len(train_file))

def read_data(file_path):
    data = mne.io.read_raw_edf(file_path, preload=True)
    data.set_eeg_reference('average')
    data = data.resample(250.0)
    #data = data.filter(52, 48)
    annotations = read_annotation(file_path)
    annotations_obj = create_annotations(annotations)
    data.set_annotations(annotations_obj)
    events, event_id = mne.events_from_annotations(raw=data, event_id=EVENT_DICT)
    print(events.shape)
    print(events)
    try:
        epochs = mne.make_fixed_length_epochs(raw=data, duration=1.8)
        epochs.events = events
        epochs.event_id = event_id
        epochs.selection = np.arange(len(events))
        epochs.drop_log = tuple(() if k in epochs.selection else ("IGNORED",) for k in range(600))
        """print(len(epochs.drop_log))
        print(len(epochs.selection))"""
        print(epochs.drop_log)
        epoch_dataframe = epochs.to_data_frame()
    except ValueError:
        print("This file could not be used!")
        raise ValueError
    #print(epoch_dataframe)
    segmented_data = epochs.get_data()
    #check_unique_conditions_per_epoch(epoch_dataframe)
    fft_df, fft_labels = fft_matrix_creation(epoch_dataframe)
    #print(np.shape(fft_df))
    if np.shape(fft_df)[1]:
        fft_columns = [i for i in range(np.shape(fft_df)[1])]
    else:
        fft_columns = [i for i in range(np.shape(fft_df)[0])]
    #file_df = pd.DataFrame(data=fft_df, columns=fft_columns)
    return fft_df, fft_labels, fft_columns
    #data.plot(duration=20, n_channels=31, bgcolor='white', scalings='auto')
    #print(events)

def read_annotation(file_path):
    an_file = file_path.replace('.edf', '.csv')
    annotations = pd.read_csv(an_file, skiprows=5)
    #annotations = annotations[annotations['label'] != 'bckg']
    return annotations

def create_annotations(annotations):
    """ch_names = [list(annotations['channel']),]
    for index in range(len(annotations['start_time'])-1):
        ch_names.append([])""" #ch_names is also the name of the Annotations obj paramenter; channel names do not coincide between raw data and annotation file
    obj = mne.Annotations(annotations['start_time'],calculate_duration(annotations['start_time'], annotations['stop_time']), annotations['label'])
    return obj

def calculate_duration(time1, time2):
    return time2 - time1

def rename_signal_channels(raw, annotations):
    channel_dict = {}
    raw_names = raw.ch_names
    ann_names = annotations['channel'].unique()
    for index in range(len(ann_names)):
        channel_dict[raw_names[index]] = ann_names[index]

def check_unique_conditions_per_epoch(dataframe):
    num_epochs = dataframe['epoch'].max()
    for no in range(num_epochs+1):
        temp_df = dataframe[dataframe['epoch'] == no]
        print(str(no) + " EPOCH")
        print(temp_df['condition'].unique())
        print('******************************************************')

def get_column_names(dataframe):
    column_names = dataframe.columns
    column_names = column_names[3:]
    return column_names.tolist()

def fft_matrix_creation(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []
    #print(final_matrix)
    """epoch = dataframe[dataframe['epoch'] == 0]
    coeficients = sp.fft.fft(epoch[column_names[0]].values)
    print(coeficients.shape)"""
    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.append(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients = sp.fft.fft(epoch[item].values)
            for co in coefficients:
                add_arr.append(co)
            #print(add_arr)
        #add_arr = add_arr.append(condition) #problem here, makes list become type None, for some fucking reason
        final_matrix.append(add_arr)
    return pd.DataFrame(final_matrix), pd.Series(labels)

def process_data(file_path_list):
    final_df = pd.DataFrame()
    final_labels = pd.Series()
    file_num = 1
    for file_name in file_path_list:
        print("************************************************************")
        print("File number: " + str(file_num))
        try:
            file_df, fft_labels, fft_columns = read_data(file_name)
            final_df = pd.concat([final_df, file_df], axis=0)
            final_labels = pd.concat([final_labels, fft_labels], axis=0)
        except ValueError:
            continue
        file_num += 1

    return final_df, final_labels

#read_data(train_file[0])
#read_data('C:\\Users\\arsms\\Documents\\tuh_eeg\\edf\\train\\aaaaaaag\\s004_2007\\03_tcp_ar_a\\aaaaaaag_s004_t000.edf')
process_df, process_labels = process_data(train_file)
print(process_df)

joblib.dump((process_df, process_labels), 'fft_processed_data.sav')