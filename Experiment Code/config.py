import numpy as np
import tensorflow as tf

class Config:
    def __init__(self, seed=25, float_precision=tf.float32, lstm_units=10):
        self.seed = seed
        self.float_precision = float_precision
        self.lstm_units = lstm_units

        # Set seeds for reproducibility
        np.random.seed(self.seed)
        tf.random.set_seed(self.seed)
        # Seed value will be set for scikit-learn in train_validation_test_exe.py file
