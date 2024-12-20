from glob import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path as pt

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
    #print(data.ch_names)
    annotations = read_annotation(file_path)
    annotations_obj = create_annotations(annotations)
    data.set_annotations(annotations_obj)
    #ppm = mne.find_events(data)
    #print(ppm)
    events, event_id = mne.events_from_annotations(raw=data, event_id=EVENT_DICT)
    #data.add_events(events, stim_channel='PHOTIC PH', replace=True)
    epochs2 = mne.make_fixed_length_epochs(raw=data, duration=1.8)
    epochs2.events = events
    epochs2.event_id = event_id
    epochs2.selection = np.arange(len(events))
    ppm = epochs2.to_data_frame()
    print(ppm)
    """epochs = mne.Epochs(data, events=events, event_id=event_id, event_repeated='drop', verbose=True)
    dataframe = epochs.to_data_frame(verbose=True)
    segmented_data = epochs.get_data()"""
    #print(data)
    #testing_df = mne.epochs.make_metadata(events, event_id, None, None, 250)
    #print(testing_df)
    #print(events)
    #print(data.events)
    #print(epochs.events)
    print('VALIENDO VRGA******************************************************')
    print(ppm['condition'].unique())
    #print(dataframe)
    """print(segmented_data)
    print(segmented_data.shape)
    print('DATAFRAME 2*********************PRRO')
    print(epochs2.to_data_frame(verbose=True))"""
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

read_data(train_file[0])
#read_data('C:\\Users\\arsms\\Documents\\tuh_eeg\\edf\\train\\aaaaaaag\\s004_2007\\03_tcp_ar_a\\aaaaaaag_s004_t000.edf')