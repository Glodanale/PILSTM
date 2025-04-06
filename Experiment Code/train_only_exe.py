import os
import argparse
import pandas as pd
import numpy as np
import tensorflow as tf
from enum import Enum

from variations import Variation
from config import Config
from preprocess import preprocess_truncated, preprocess_masked
from computational_graph import ComputationalGraph
from lstm_model import build_lstm
from training import train_step


# --------- Enum for Variants ---------
class Variation(str, Enum):
    TRUNC_LINEAR = "trunc_linear"
    TRUNC_NONLINEAR = "trunc_nonlinear"
    MASK_LINEAR = "mask_linear"
    MASK_NONLINEAR = "mask_nonlinear"


# --------- Argument Parsing ---------
parser = argparse.ArgumentParser(description="Run LSTM Training Variations")
parser.add_argument("--seed", type=int, required=True, help="Seed value for reproducibility")
parser.add_argument("--float_value", type=int, choices=[32, 64], required=True, help="Floating point precision (32 or 64)")
parser.add_argument("--lstm_units", type=int, required=True, help="Number of LSTM units")
parser.add_argument("--variation", type=str, required=True, choices=[v.value for v in Variation], help="Workflow variation")

args = parser.parse_args()

# --------- Set Precision ---------
FLOAT_MAP = {32: tf.float32, 64: tf.float64}
precision = FLOAT_MAP[args.float_value]

# --------- Setup Config ---------
config = Config(seed=args.seed, float_precision=precision, lstm_units=args.lstm_units)

# --------- Determine Flags ---------
def parse_variation_flags(variation):
    return {
        "use_mask": "mask" in variation,
        "nonlinear": "nonlinear" in variation
    }

flags = parse_variation_flags(args.variation)

# --------- Prepare Folder Names ---------
base_results_dir = os.path.join(".", "Results")
os.makedirs(base_results_dir, exist_ok=True)  # Ensure Results/ exists

album = f"{args.lstm_units}Cell_F{args.float_value}_Seed{args.seed}"
result_folder = f'./Results/{album}'
output_folder = f'{result_folder}/TrainOnly_{args.variation}'

os.makedirs(result_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

# --------- Load and Preprocess Data ---------
data = pd.read_csv("../ExperimentSet.csv")
if flags["use_mask"]:
    input_I, input_J, additional_val, seq_len, num_sequences = preprocess_masked(data, precision)
else:
    input_I, input_J, additional_val, seq_len, num_sequences = preprocess_truncated(data, precision)

# --------- Build Model and Graph ---------
comp_graph = ComputationalGraph(non_linear=flags["nonlinear"], precision=precision)
lstm_model = build_lstm(seq_len, input_I.shape[-1], config.lstm_units, use_mask=flags["use_mask"], precision=precision)

IDM_optimizer = tf.optimizers.Adam(learning_rate=0.1, clipnorm=5.0)
LSTM_optimizer = tf.optimizers.Adam(learning_rate=0.001)

# --------- Training Loop ---------
def train_only_loop():
    batch_size = max(1, int(num_sequences / 40))
    dataset = tf.data.Dataset.from_tensor_slices((input_I, input_J, additional_val)).batch(batch_size)

    epochResults = pd.DataFrame(columns=["Epoch_Number", "Loss_Average", "Loss_Median"])
    num_epochs = 500

    with tf.device('/GPU:0'):
        for epoch in range(num_epochs):
            loss_list = []

            for inputs_I_batch, inputs_J_batch, additional_val_batch in dataset:
                loss = train_step(inputs_I_batch, inputs_J_batch, additional_val_batch,
                                  flags["use_mask"], comp_graph, lstm_model,
                                  IDM_optimizer, LSTM_optimizer)
                loss_list.append(loss.numpy())

            loss_avg = np.average(loss_list)
            loss_median = np.median(loss_list)

            epochResults.loc[len(epochResults)] = {"Epoch_Number": epoch + 1, "Loss_Average": loss_avg, "Loss_Median": loss_median}
            print(f"Epoch {epoch+1} | Loss avg: {loss_avg:.6f} | Loss median: {loss_median:.6f}")

            if epoch % 20 == 0:
                epoch_file_path = os.path.join(output_folder, f"{args.variation}_{epoch}.csv")
                epochResults.to_csv(epoch_file_path, index=False)

    final_file_path = os.path.join(result_folder, f"{args.variation}_TrainOnly_final.csv")
    epochResults.to_csv(final_file_path, index=False)


# --------- Run Training ---------
train_only_loop()
