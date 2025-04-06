import os
import argparse
import copy
import pandas as pd
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from enum import Enum

from config import Config
from preprocess import preprocess_truncated, preprocess_masked
from lstm_model import build_lstm
from computational_graph import ComputationalGraph
from training import train_step, validate_test_step
from variations import Variation


class Variation(str, Enum):
    TRUNC_LINEAR = "trunc_linear"
    TRUNC_NONLINEAR = "trunc_nonlinear"
    MASK_LINEAR = "mask_linear"
    MASK_NONLINEAR = "mask_nonlinear"


def parse_variation_flags(variation):
    return {
        "use_mask": "mask" in variation,
        "nonlinear": "nonlinear" in variation
    }


def create_tf_dataset(inputs_I, inputs_J, additional_value, num_sequences):
    batch_size = max(1, int(num_sequences / 40))
    dataset = tf.data.Dataset.from_tensor_slices((inputs_I, inputs_J, additional_value))
    return dataset.batch(batch_size).prefetch(tf.data.experimental.AUTOTUNE)


def split_data(input_I, input_J, additional, seed, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    train_I, rem_I, train_J, rem_J, train_add, rem_add = train_test_split(
        input_I, input_J, additional, test_size=(1 - train_ratio), random_state=seed, shuffle=True
    )
    val_I, test_I, val_J, test_J, val_add, test_add = train_test_split(
        rem_I, rem_J, rem_add, test_size=(test_ratio / (val_ratio + test_ratio)), random_state=seed, shuffle=True
    )
    return (train_I, train_J, train_add), (val_I, val_J, val_add), (test_I, test_J, test_add)


# -------- Argument Parsing --------
parser = argparse.ArgumentParser(description="Run full train/val/test workflow.")
parser.add_argument("--seed", type=int, required=True)
parser.add_argument("--float_value", type=int, choices=[32, 64], required=True)
parser.add_argument("--lstm_units", type=int, required=True)
parser.add_argument("--variation", type=str, required=True, choices=[v.value for v in Variation])
args = parser.parse_args()

FLOAT_MAP = {32: tf.float32, 64: tf.float64}
precision = FLOAT_MAP[args.float_value]
flags = parse_variation_flags(args.variation)
config = Config(seed=args.seed, float_precision=precision, lstm_units=args.lstm_units)

# -------- Load & Preprocess --------
base_results_dir = os.path.join(".", "Results")
os.makedirs(base_results_dir, exist_ok=True)  # Ensure Results/ exists

album = f"{args.lstm_units}Cell_F{args.float_value}_Seed{args.seed}"
result_folder = f'./Results/{album}'
output_folder = f'{result_folder}/TrainVal_{args.variation}'
os.makedirs(result_folder, exist_ok=True)
os.makedirs(output_folder, exist_ok=True)

df = pd.read_csv("../ExperimentSet.csv")

if flags["use_mask"]:
    input_I, input_J, additional, seq_len, num_sequences = preprocess_masked(df, precision)
else:
    input_I, input_J, additional, seq_len, num_sequences = preprocess_truncated(df, precision)

(train_I, train_J, train_add), (val_I, val_J, val_add), (test_I, test_J, test_add) = split_data(
    input_I, input_J, additional, args.seed
)

train_dataset = create_tf_dataset(train_I, train_J, train_add, num_sequences)
val_dataset = create_tf_dataset(val_I, val_J, val_add, num_sequences)
test_dataset = create_tf_dataset(test_I, test_J, test_add, num_sequences)

# -------- Build Model + Graph --------
comp_graph = ComputationalGraph(non_linear=flags["nonlinear"], precision=precision)
lstm_model = build_lstm(seq_len, input_I.shape[-1], args.lstm_units, use_mask=flags["use_mask"], precision=precision)

best_val_loss = float("inf")
best_lstm_weights = None
best_graph_weights = None

optimizer1 = tf.optimizers.Adam(learning_rate=0.1, clipnorm=5.0)
optimizer2 = tf.optimizers.Adam(learning_rate=0.001)

# -------- Training Loop --------
num_epochs = 500
epochResults = pd.DataFrame(columns=["Epoch_Number", "Train_Loss_Average", "Validation_Loss_Average",
                                     "Train_Loss_Median", "Validation_Loss_Median"])

with tf.device("/GPU:0"):
    for epoch in range(num_epochs):
        train_loss_list = []
        for batch in train_dataset:
            loss = train_step(*batch, flags["use_mask"], comp_graph, lstm_model, optimizer1, optimizer2)
            train_loss_list.append(loss.numpy())

        val_loss_list = []
        for batch in val_dataset:
            loss = validate_test_step(*batch, flags["use_mask"], comp_graph, lstm_model)
            val_loss_list.append(loss.numpy())

        avg_train = np.mean(train_loss_list)
        avg_val = np.mean(val_loss_list)
        med_train = np.median(train_loss_list)
        med_val = np.median(val_loss_list)

        if avg_val < best_val_loss:
            best_val_loss = avg_val
            best_lstm_weights = lstm_model.get_weights()
            best_graph_weights = [v.numpy() for v in comp_graph.trainable_variables]

        epochResults.loc[len(epochResults)] = {
            "Epoch_Number": epoch + 1,
            "Train_Loss_Average": avg_train,
            "Validation_Loss_Average": avg_val,
            "Train_Loss_Median": med_train,
            "Validation_Loss_Median": med_val
        }

        print(f"Epoch {epoch + 1} | Train Loss: {avg_train:.6f} | Val Loss: {avg_val:.6f}")

        if epoch % 20 == 0:
            epochResults.to_csv(os.path.join(output_folder, f"{args.variation}_{epoch}.csv"), index=False)

# -------- Save Final Training CSV --------
epochResults.to_csv(os.path.join(result_folder, f"{args.variation}_TrainVal_final.csv"), index=False)

# -------- Load Best Weights --------
lstm_model.set_weights(best_lstm_weights)
for var, val in zip(comp_graph.trainable_variables, best_graph_weights):
    var.assign(val)

# -------- Evaluate on Test Set --------
test_loss_list = []
for batch in test_dataset:
    loss = validate_test_step(*batch, flags["use_mask"], comp_graph, lstm_model)
    test_loss_list.append(loss.numpy())

avg_test_loss = np.mean(test_loss_list)
med_test_loss = np.median(test_loss_list)

test_df = pd.DataFrame([{
    "Experiment": f"{album}_{args.variation}",
    "Test Loss Average": avg_test_loss,
    "Test Loss Median": med_test_loss
}])

test_df.to_csv(os.path.join(result_folder, f"{args.variation}_TestLoss.csv"), index=False)
