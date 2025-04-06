import numpy as np
import pandas as pd
import tensorflow as tf

def preprocess_truncated(df, precision=tf.float32):
    groups = df.groupby('Group_ID')
    input_I, input_J, additional_value = [], [], []
    num_sequences = len(groups)
    
    for _, group_df in groups:
        group_df = group_df.sort_values(by='Global_Time')
        input_I.append(group_df[['Subject_Space_Headway', 'Subject_Velocity', 'Subject_Delta_Velocity']].values.astype(precision.as_numpy_dtype))
        input_J.append(group_df[['Leader_Space_Headway', 'Leader_Velocity', 'Leader_Delta_Velocity']].values.astype(precision.as_numpy_dtype))
        additional_value.append(group_df[['Leader_Acceleration']].values.astype(precision.as_numpy_dtype))

    min_length = min(len(seq) for seq in input_I + input_J)
    truncated_I = np.array([seq[:min_length] for seq in input_I])
    truncated_J = np.array([seq[:min_length] for seq in input_J])
    truncated_additional = np.array([seq[:min_length] for seq in additional_value])

    return truncated_I, truncated_J, truncated_additional, min_length, num_sequences

def preprocess_masked(df, precision=tf.float32):
    groups = df.groupby('Group_ID')
    input_I, input_J, additional_value = [], [], []
    num_sequences = len(groups)

    for _, group_df in groups:
        group_df = group_df.sort_values(by='Global_Time')
        input_I.append(group_df[['Subject_Space_Headway', 'Subject_Velocity', 'Subject_Delta_Velocity']].values.astype(precision.as_numpy_dtype))
        input_J.append(group_df[['Leader_Space_Headway', 'Leader_Velocity', 'Leader_Delta_Velocity']].values.astype(precision.as_numpy_dtype))
        additional_value.append(group_df[['Leader_Acceleration']].values.astype(precision.as_numpy_dtype))

    max_length = max(len(seq) for seq in input_I + input_J)
    padded_I = tf.keras.preprocessing.sequence.pad_sequences(input_I, padding="post", dtype=precision.name)
    padded_J = tf.keras.preprocessing.sequence.pad_sequences(input_J, padding="post", dtype=precision.name)
    padded_additional = tf.keras.preprocessing.sequence.pad_sequences(additional_value, padding="post", dtype=precision.name)

    return padded_I, padded_J, padded_additional, max_length, num_sequences