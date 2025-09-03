import pandas as pd
import numpy as np

class NewMontage:
    def __init__(self, signal_df):
        self.signal = signal_df
        self.channel_dictionary = {
            'FP1-F7': ['FP1','F7'],
            'F7-T3':  ['F7','T3'],
            'T3-T5':  ['T3','T5'],
            'T5-O1':  ['T5','O1'],
            'FP2-F8': ['FP2','F8'],
            'F8-T4' : ['F8','T4'],
            'T4-T6':  ['T4','T6'],
            'T6-O2':  ['T6','O2'],
            'T3-C3':  ['T3','C3'],
            'C3-CZ':  ['C3','CZ'],
            'CZ-C4':  ['CZ','C4'],
            'C4-T4':  ['C4','T4'],
            'FP1-F3': ['FP1','F3'],
            'F3-C3':  ['F3','C3'],
            'C3-P3':  ['C3','P3'],
            'P3-O1':  ['P3','O1'],
            'FP2-F4': ['FP2','F4'],
            'F4-C4':  ['F4','C4'],
            'C4-P4':  ['C4','P4'],
            'P4-O2':  ['P4','O2'],
            #'T4-A2': ['T4','A2'],
            #'A1-T3': ['A1','T3']
        }

    def change_montage(self):
        new_data = {}
        new_data['time'] = self.signal['time'].to_list()
        new_data['condition'] = self.signal['condition'].to_list()
        new_data['epoch'] = self.signal['epoch'].to_list()
        columns = list(self.signal.columns)
        for key, data in self.channel_dictionary.items():
            col1 = None
            col2 = None
            for column in columns:
                if data[0] in column:
                    col1 = self.signal[column]
                elif data[1] in column:
                    col2 = self.signal[column]
            if col1 is not None and col2 is not None:
                new_col = col1-col2
                new_data[key] = new_col.to_list()
            else:
                new_data[key] = np.zeros(self.signal.shape[0])
        return pd.DataFrame.from_dict(new_data)