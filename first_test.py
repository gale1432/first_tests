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
import feat_creation
import ewtpy
from PyEMD import CEEMDAN
#from fathon import DFA

#from mne.conftest import event_id
from sklearn.preprocessing import LabelEncoder

#mne.viz.set_browser_backend("qt")

EVENT_DICT = {
    'fnsz': 0,
    'gnsz': 1,
    'spsz': 2,
    'cpsz': 3,
    'absz': 4,
    'tnsz': 5,
    'cnsz': 6,
    'tcsz': 7,
    'atsz': 8,
    'mysz': 9,
    'nesz': 10,
    'bckg': 11
}
CED_PATH = os.path.join(os.path.expanduser('~'), 'sda1', 'Documents', 'tuh_eeg', 'edf')
HOME_PATH='E:\\Files\\tuh_eeg\\edf\\train'

#train_file = glob(CED_PATH+'\\train\\**/*.edf', recursive=True) #for windows

def read_mne_data(file_path):
    data = mne.io.read_raw_edf(file_path, preload=True)
    data.set_eeg_reference('average')
    data = data.resample(250.0)
    data = data.notch_filter(60.0)
    annotations = read_annotation(file_path)
    annotations_obj = create_annotations(annotations)
    data.set_annotations(annotations_obj)
    events, ev_id = mne.events_from_annotations(raw=data, event_id=EVENT_DICT)
    return data, events, ev_id

def mne_to_epochs_df(data, events, ev_id):
    try:
        epochs = mne.make_fixed_length_epochs(raw=data, duration=4.096)
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
        return epoch_dataframe
        #print(epoch_dataframe)
    except ValueError:
        print("This file could not be used!")
        raise ValueError

def return_composed_signals(file_path):
    data, events, ev_id = read_mne_data(file_path)
    try:
        epoch_dataframe = mne_to_epochs_df(data, events, ev_id)
        return numpify_signals(epoch_dataframe)
    except ValueError:
        raise ValueError

def numpify_signals(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []
    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.extend(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            #if extend is used, all channels are concatenated into a single vector
            add_arr.extend(epoch[item].values)
        final_matrix.append(add_arr)
    return final_matrix, labels

def process_signal_data(file_path_list):
    final_df = list()
    final_labels = list()
    file_num = 1
    unusable_files_counter = 0
    for file_name in file_path_list:
        print("***********************************************************")
        print("File number: " + str(file_num))
        try:
            file_matrix, labels = return_composed_signals(file_name)
            final_df.extend(file_matrix)
            final_labels.extend(labels)
        except ValueError:
            print('SOMETHING HAPPENED!')
            unusable_files_counter += 1
            continue
        file_num += 1
    print("Unusable files: " + str(unusable_files_counter))
    return np.array(final_df), np.array(final_labels)

def read_data(file_path, preprocessing_type):
    data, events, ev_id = read_mne_data(file_path)
    try:
        epoch_dataframe = mne_to_epochs_df(data, events, ev_id)
    except ValueError:
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
        return wpd_df, wpd_labels
    #********************* THIS IS EMPIRICAL WAVELET TRANSFORM ***********************
    elif preprocessing_type == 4:
        #empirical_wt(epoch_dataframe)
        ewt_df, ewt_labels = empirical_wt(epoch_dataframe)
        if np.shape(ewt_df)[1]:
            dwt_columns = [i for i in range(np.shape(ewt_df)[1])]
        else:
            dwt_columns = [i for i in range(np.shape(ewt_df)[0])]
        return ewt_df, ewt_labels, dwt_columns
    elif preprocessing_type == 5:
        ppd_matrix, ppd_labels = matrices_creation(epoch_dataframe)
        return ppd_matrix, ppd_labels
    #******************** THIS IS DWT, BUT WITH CHARACTERISTICS EXTRACTED FROM NAJAFI *******************
    elif preprocessing_type == 6:
        dwt_df, dwt_labels = najafi_dwt(epoch_dataframe)
        return dwt_df, dwt_labels
    elif preprocessing_type == 7:
        ceemdan_df, ceemdan_labels = ceemdan(epoch_dataframe)
        return ceemdan_df, ceemdan_labels, np.arange(5)
    return "GG"
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
        labels.extend(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients = pywt.wavedec(epoch[item].values, wavelet='db4', level=4)
            #print(type(coefficients))
            #print(len(coefficients))
            for co in coefficients:
                stats = feat_creation.get_stats(co)
                add_arr.extend(stats)
            #print(add_arr)
        final_matrix.append(add_arr)
    #print(final_matrix)
    return pd.DataFrame(final_matrix), pd.Series(labels)
################################################

################# NAJAFI DWT ################################
def najafi_dwt(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []
    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        ll = epoch['condition'].unique()
        for l in ll:
            labels.append(EVENT_DICT[l])
        add_arr = []
        for item in column_names:
            co_arr = list()
            coefficients = pywt.wavedec(epoch[item].values, wavelet='coif3', level=4)
            for co in coefficients:
                stats = feat_creation.get_stats_with_power(co)
                co_arr.extend(stats)
            add_arr.append(np.nan_to_num(np.asarray(co_arr).astype('float32')))
        final_matrix.append(np.array(add_arr))
    return final_matrix, labels

############### EMPIRICAL WAVELET TRANSFORM #####################
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
                stats = feat_creation.get_stats(list(coefficients_dataframe[i]))
                add_arr.extend(stats)
        final_matrix.append(add_arr)
    return pd.DataFrame(final_matrix), pd.Series(labels)

################## CEEMDAN ###################
def ceemdan(dataframe):
    mode_number = 7
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []
    c = CEEMDAN()

    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.extend(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            c_imfs = c.ceemdan(epoch[item].values)
            coefficients_dataframe = pd.DataFrame(c_imfs)
            for i in range(mode_number):
                stats = feat_creation.get_stats_with_power(list(coefficients_dataframe[i]))
                add_arr.extend(stats)
        final_matrix.append(add_arr)
    return pd.DataFrame(final_matrix), pd.Series(labels)

#################################### WAVELET PACKET DECOMPOSITION #############################
def wavelet_packet_decomp(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []

    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.extend(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            coefficients = []
            wavelet_dec = pywt.WaveletPacket(epoch[item].values, 'db4', 'zero', 4)
            levels = wavelet_dec.get_level(4, order='freq')
            #kk = [node.path for node in levels]
            #print(len(kk))
            for level in levels:
                data = level.data
                #print(data)
                coefficients.extend(feat_creation.get_full_stats(data))
            """for co in coefficients:
                add_arr.append(co)"""
            add_arr.extend(coefficients)
            # print(add_arr)
        final_matrix.append(np.array(add_arr))
    return final_matrix, labels

############## Preparing 2D matrices ###########################
def matrices_creation(dataframe):
    epoch_number = dataframe['epoch'].max()
    column_names = get_column_names(dataframe)
    labels = []
    final_matrix = []
    for i in range(epoch_number):
        epoch = dataframe[dataframe['epoch'] == i]
        labels.extend(epoch['condition'].unique())
        add_arr = []
        for item in column_names:
            co_arr = list()
            coefficients = pywt.wavedec(epoch[item].values, wavelet='db4', level=4)
            for co in coefficients:
                co_arr.append(feat_creation.get_full_stats(co))
            add_arr.append(co_arr)
        final_matrix.append(add_arr)
    return final_matrix, labels


def process_data(file_path_list, preprocessing_type):
    final_df = pd.DataFrame()
    final_labels = pd.Series()
    file_num = 1
    unusable_files_counter = 0
    for file_name in file_path_list:
        print("************************************************************")
        print("File number: " + str(file_num))
        try:
            file_df, fft_labels, fft_columns = read_data(file_name.replace('csv', 'edf'), preprocessing_type)
            print('pp')
            final_df = pd.concat([final_df, file_df], axis=0)
            final_labels = pd.concat([final_labels, fft_labels], axis=0)
        except ValueError:
            print('SOMETHING HAPPENED!')
            unusable_files_counter += 1
            continue
        file_num += 1
    print("Unusable files: " + str(unusable_files_counter))
    return final_df, final_labels

def process_lstm_data(file_path_list, preprocessing_type):
    final_df = list()
    final_labels = list()
    file_num = 1
    unusable_files_counter = 0
    for file_name in file_path_list:
        print("***********************************************************")
        print("File number: " + str(file_num))
        try:
            file_matrix, labels = read_data(file_name, preprocessing_type)
            final_df.extend(file_matrix)
            final_labels.extend(labels)
        except ValueError:
            print('SOMETHING HAPPENED!')
            unusable_files_counter += 1
            continue
        file_num += 1
    print("Unusable files: " + str(unusable_files_counter))
    return np.array(final_df), np.array(final_labels)

#print(pywt.wavelist(family='db',kind='discrete'))
#df, labels, columns = read_data(train_file[0], 2)
#print(df)
#df1, labels1, columns1 = read_data(train_file[-1], 1)
#print(train_file)
#print(df)
#df, labels = process_data(train_file, 1)
def process_all_data(preprocessing_type, train_file, dev_file, eval_file):
    df, labels = process_data(train_file, preprocessing_type)
    joblib.dump((df, labels), 'train_dwt_four.sav')
    df, labels = process_data(dev_file, preprocessing_type)
    joblib.dump((df, labels), 'dev_dwt_four.sav')
    df, labels = process_data(eval_file, preprocessing_type)
    joblib.dump((df, labels), 'eval_dwt_four.sav')

#process_all_data(6)

def process_lstm_all_data(preprocessing_type):
    train_file = pd.read_csv('train_summary.csv')['0'].tolist()
    dev_file = pd.read_csv('dev_summary.csv')['0'].tolist()
    eval_file = pd.read_csv('eval_summary.csv')['0'].tolist()
    df_train, labels_train = process_lstm_data(train_file, preprocessing_type)
    #joblib.dump((df, labels), 'train_ppd.sav')
    df_dev, labels_dev = process_lstm_data(dev_file, preprocessing_type)
    #joblib.dump((df, labels), 'dev_ppd.sav')
    #df, labels = process_lstm_data(eval_file, preprocessing_type)
    #joblib.dump((df, labels), 'eval_ppd.sav')
    return df_train, labels_train, df_dev, labels_dev

def process_bilstm_all_data(preprocessing_type):
    """train_file = glob(CED_PATH + '/train/**/*.edf', recursive=True)  # for linux
    dev_file = glob(CED_PATH + '/dev/**/*.edf', recursive=True)
    eval_file = glob(CED_PATH + '/eval/**/*.edf', recursive=True)"""
    train_file = pd.read_csv('mne_file_lists/train_not_ar.csv')['0'].tolist()
    dev_file = pd.read_csv('mne_file_lists/dev_not_ar.csv')['0'].tolist()
    eval_file = pd.read_csv('mne_file_lists/eval_not_ar.csv')['0'].tolist()
    df_train, labels_train = process_lstm_data(train_file, preprocessing_type)
    np.save('wpd/train_x.npy', df_train, allow_pickle=True)
    np.save('wpd/train_y.npy', labels_train, allow_pickle=True)
    df_dev, labels_dev = process_lstm_data(dev_file, preprocessing_type)
    np.save('wpd/dev_x.npy', df_dev, allow_pickle=True)
    np.save('wpd/dev_y.npy', labels_dev, allow_pickle=True)
    df_eval, labels_eval = process_lstm_data(eval_file, preprocessing_type)
    np.save('wpd/eval_x.npy', df_eval, allow_pickle=True)
    np.save('wpd/eval_y.npy', labels_eval, allow_pickle=True)
    #return df_train, labels_train, df_dev, labels_dev, df_eval, labels_eval

#process_lstm_all_data(5)
#process_bilstm_all_data(6)
#df, labels, columns = read_data('/home/gaelh/sda1/Documents/tuh_eeg/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf', 7)
#print(df)
#read_data('/home/gaelh/sda1/Documents/tuh_eeg/edf/train/aaaaaaac/s001_2002/02_tcp_le/aaaaaaac_s001_t000.edf', 4)
#print(type(df[0]))
#print(df[0].apply(type))
#print(df)
#read_data('C:\\Users\\arsms\\Documents\\tuh_eeg\\edf\\train\\aaaaaaag\\s004_2007\\03_tcp_ar_a\\aaaaaaag_s004_t000.edf')
"""train_file = pd.read_csv('mne_file_lists/not_ar_not_bckg.csv')['0'].tolist()
signals, final_labels = process_signal_data(train_file)
np.save('x_composed.npy', signals, allow_pickle=True)
np.save('y_composed.npy', final_labels, allow_pickle=True)"""
"""process_df, process_labels = process_data(train_file)
print(process_df)

joblib.dump((process_df, process_labels), 'fft_processed_data.sav')"""