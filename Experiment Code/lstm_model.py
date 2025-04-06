import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Masking
from tensorflow.keras.models import Model

def build_lstm(sequence_length, input_shape, units=10, use_mask=False, precision=tf.float32):
    inputs = tf.keras.Input(shape=(sequence_length, input_shape), dtype=precision)
    
    if use_mask:
        x = tf.keras.layers.Masking(mask_value=0.0)(inputs)
    else:
        x = inputs

    lstm_out = tf.keras.layers.LSTM(units, return_sequences=False)(x)
    output = tf.keras.layers.Dense(1, activation=None)(lstm_out)
    
    return tf.keras.Model(inputs=inputs, outputs=output)

