from glob import glob
import mne
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import joblib
import scipy as sp
import pywt
import os
from montage import NewMontage
from feat_creation import get_stats
import ewtpy
#from fathon import DFA

#from mne.conftest import event_id
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
CED_PATH = os.path.join(os.path.expanduser('~'), 'sda1', 'Documents', 'tuh_eeg', 'edf')
HOME_PATH='E:\\Files\\tuh_eeg\\edf\\train'

#train_file = glob(CED_PATH+'\\train\\**/*.edf', recursive=True) #for windows


def read_data(file_path, preprocessing_type):
    #mne.datasets.sample.data_path(force_update=True, download=True)
    data = mne.io.read_raw_edf(file_path, preload=True)
    data.set_eeg_reference('average')
    data = data.resample(250.0)

    data = data.notch_filter(60.0)
    annotations = read_annotation(file_path)
    annotations_obj = create_annotations(annotations)
    data.set_annotations(annotations_obj)
    events, ev_id = mne.events_from_annotations(raw=data, event_id=EVENT_DICT)
    #print(events.shape)
    #print(events)
    try:
        epochs = mne.make_fixed_length_epochs(raw=data, duration=2)
        epochs.events = events
        epochs.event_id = ev_id
        epochs.selection = np.arange(len(events))
        epochs.drop_log = tuple(() if k in epochs.selection else ("IGNORED",) for k in range(6000))
        """print(len(epochs.drop_log))
        print(len(epochs.selection))"""
        #print(epochs.drop_log)
        epoch_dataframe = epochs.to_data_frame()
        new_montage = NewMontage(epoch_dataframe)
        epoch_dataframe = new_montage.change_montage()
        #print(epoch_dataframe)
    except ValueError:
        print("This file could not be used!")
        raise ValueError
    #check_unique_conditions_per_epoch(epoch_dataframe)
    #********************** THIS IS THE FAST FOURIER TRASNFORM CODE *******************************
    if preprocessing_type == 1:
        fft_df, fft_labels = fft_matrix_creation(epoch_dataframe)
        if np.shape(fft_df)[1]:
            fft_columns = [i for i in range(np.shape(fft_df)[1])]
        else:
            fft_columns = [i for i in range(np.shape(fft_df)[0])]
        return fft_df, fft_labels, fft_columns
    #********************** THIS IS THE DISCRETE WAVELET TRANSFORM CODE ****************************
    elif preprocessing_type == 2:
        dwt_df, dwt_labels = discrete_wt(epoch_dataframe)
        if np.shape(dwt_df)[1]:
            dwt_columns = [i for i in range(np.shape(dwt_df)[1])]
        else:
            dwt_columns = [i for i in range(np.shape(dwt_df)[0])]
        return dwt_df, dwt_labels, dwt_columns
    #*********************** THIS IS WAVELET PACKET DECOMPOSITION CODE ******************************
    elif preprocessing_type == 3:
        wpd_df, wpd_labels = wavelet_packet_decomp(epoch_dataframe)
        if np.shape(wpd_df)[1]:
            wpd_columns = [i for i in range(np.shape(wpd_df)[1])]
        else:
            wpd_columns = [i for i in range(np.shape(wpd_df)[0])]
        return wpd_df, wpd_labels, wpd_columns
    #********************* THIS IS EMPIRICAL WAVELET TRANSFORM ***********************
    elif preprocessing_type == 4:
        #empirical_wt(epoch_dataframe)
        ewt_df, ewt_labels = empirical_wt(epoch_dataframe)
        if np.shape(ewt_df)[1]:
            dwt_columns = [i for i in range(np.shape(ewt_df)[1])]
        else:
            dwt_columns = [i for i in range(np.shape(ewt_df)[0])]
        return ewt_df, ewt_labels, dwt_columns
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

################## GIVEN THAT CONTINUOUS WAVELET TRANSFORM IS BETTER SUITED FOR NEURAL NETWORKS, THIS METHOD WONT BE USED FOR NOW #######################
def continuous_wt(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []
    scales = np.arange(1, 11)

    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.append(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients, freq = pywt.cwt(epoch[item].values, scales, 'mexh')
            print(coefficients.shape)
            print(type(coefficients))
            for co in coefficients:
                add_arr.append(co)
                print(co.shape)
                print(type(co))
                #add_arr.extend(co)
            #print(add_arr)
        final_matrix.append(add_arr)
    #print(final_matrix)
    return pd.DataFrame(final_matrix), pd.Series(labels)
################################################################################################################

def discrete_wt(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []

    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.append(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients = pywt.wavedec(epoch[item].values, wavelet='db4', level=4)
            #print(type(coefficients))
            #print(len(coefficients))
            for co in coefficients:
                stats = get_stats(co)
                add_arr.extend(stats)
            #print(add_arr)
        final_matrix.append(add_arr)
    #print(final_matrix)
    return pd.DataFrame(final_matrix), pd.Series(labels)
################################################

def empirical_wt(dataframe):
    mode_number = 4
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []

    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.append(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients, mfb, boundaries = ewtpy.EWT1D(epoch[item].values, N=mode_number)
            """plt.plot(coefficients)
            plt.show()"""
            coefficients_dataframe = pd.DataFrame(coefficients)
            for i in range(mode_number):
                stats = get_stats(list(coefficients_dataframe[i]))
                add_arr.extend(stats)
        final_matrix.append(add_arr)
    return pd.DataFrame(final_matrix), pd.Series(labels)

def wavelet_packet_decomp(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []

    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.append(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients = []
            wavelet_dec = pywt.WaveletPacket(epoch[item].values, 'db4', 'zero', 7)
            levels = wavelet_dec.get_level(7, order='freq')
            for level in levels:
                data = level.data
                coefficients.append(data)
            for co in coefficients:
                add_arr.append(co)
            # print(add_arr)
        final_matrix.append(add_arr)
    return pd.DataFrame(final_matrix), pd.Series(labels)


def process_data(file_path_list, preprocessing_type):
    final_df = pd.DataFrame()
    final_labels = pd.Series()
    file_num = 1
    unusable_files_counter = 0
    for file_name in file_path_list:
        print("************************************************************")
        print("File number: " + str(file_num))
        try:
            file_df, fft_labels, fft_columns = read_data(file_name, preprocessing_type)
            final_df = pd.concat([final_df, file_df], axis=0)
            final_labels = pd.concat([final_labels, fft_labels], axis=0)
        except ValueError:
            unusable_files_counter += 1
            continue
        file_num += 1
    print("Unusable files: " + str(unusable_files_counter))
    return final_df, final_labels

#print(pywt.wavelist(family='db',kind='discrete'))
#df, labels, columns = read_data(train_file[0], 2)
#print(df)
#df1, labels1, columns1 = read_data(train_file[-1], 1)
#print(train_file)
#print(df)
#df, labels = process_data(train_file, 1)
def process_all_data(preprocessing_type):
    train_file = glob(CED_PATH+'/train/**/*.edf', recursive=True) #for linux
    dev_file = glob(CED_PATH+'/dev/**/*.edf', recursive=True)
    eval_file = glob(CED_PATH+'/eval/**/*.edf', recursive=True)
    df, labels = process_data(train_file, preprocessing_type)
    joblib.dump((df, labels), 'train_ewt_two.sav')
    df, labels = process_data(dev_file, preprocessing_type)
    joblib.dump((df, labels), 'dev_ewt_two.sav')
    df, labels = process_data(eval_file, preprocessing_type)
    joblib.dump((df, labels), 'eval_ewt_two.sav')

process_all_data(4)
#df, labels, columns = read_data('/home/gaelh/sda1/Documents/tuh_eeg/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf', 2)
#print(df)
#read_data('/home/gaelh/sda1/Documents/tuh_eeg/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf', 4)
#print(type(df[0]))
#print(df[0].apply(type))
#print(df)
#read_data('C:\\Users\\arsms\\Documents\\tuh_eeg\\edf\\train\\aaaaaaag\\s004_2007\\03_tcp_ar_a\\aaaaaaag_s004_t000.edf')
"""process_df, process_labels = process_data(train_file)
print(process_df)

joblib.dump((process_df, process_labels), 'fft_processed_data.sav')"""