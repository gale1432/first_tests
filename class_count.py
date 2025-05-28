import numpy as np
import pandas as pd
from first_test import EVENT_DICT

def class_count_by_key(processed_dataframe):
    count_dictionary = {
        'fnsz': 0,
        'gnsz': 0,
        'spsz': 0,
        'cpsz': 0,
        'absz': 0,
        'tnsz': 0,
        'cnsz': 0,
        'tcsz': 0,
        'mysz': 0,
        'bckg': 0
    }

    #fused_df = processed_dataframe.assign(labels=processed_labels.values)
    keys = count_dictionary.keys()

    for k in keys: #the labels could also be in integer form
        rows = processed_dataframe[processed_dataframe['labels'] == k]
        count_dictionary[k] = rows.shape[0]

    return count_dictionary

def class_count_by_id(processed_dataframe, processed_labels):
    count_dictionary = {
        'fnsz': 0,
        'gnsz': 0,
        'spsz': 0,
        'cpsz': 0,
        'absz': 0,
        'tnsz': 0,
        'cnsz': 0,
        'tcsz': 0,
        'mysz': 0,
        'bckg': 0
    }

    fused_df = processed_dataframe.assign(labels=processed_labels.values)

    for event_name, event_id in EVENT_DICT.items(): #the labels could also be in integer form
        rows = fused_df[fused_df['labels'] == event_id]
        count_dictionary[event_name] = rows.shape[0]
    return count_dictionary

def class_count_array(processed_labels):
    count_dictionary = {
        'fnsz': 0,
        'gnsz': 0,
        'spsz': 0,
        'cpsz': 0,
        'absz': 0,
        'tnsz': 0,
        'cnsz': 0,
        'tcsz': 0,
        'mysz': 0,
        'bckg': 0
    }
    for i in range(len(processed_labels)):
        count_dictionary[processed_labels[i]] = count_dictionary[processed_labels[i]] + 1
    return count_dictionary

def class_count_array_id(processed_labels):
    count_dictionary = {
        'fnsz': 0,
        'gnsz': 0,
        'spsz': 0,
        'cpsz': 0,
        'absz': 0,
        'tnsz': 0,
        'cnsz': 0,
        'tcsz': 0,
        'mysz': 0,
        'bckg': 0
    }

    for label in processed_labels:
        for event_name, event_id in EVENT_DICT.items(): #the labels could also be in integer form
            if event_id == label:
                count_dictionary[event_name] += 1
    return count_dictionary