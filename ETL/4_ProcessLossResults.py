import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import LSTM, Dense, Input, Masking
from tensorflow.keras.models import Model


def preprocess_data(df):
    groups = df.groupby('Group_ID')
    input_I, input_J, additional_value = [], [], []
    ids = []
    a = 0
    for group_id, group_df in groups:
        group_df = group_df.sort_values(by='Global_Time')
        seq_I = group_df[['Subject_Space_Headway', 'Subject_Velocity', 'Subject_Delta_Velocity']].values.astype(np.float32)
        seq_J = group_df[['Leader_Space_Headway', 'Leader_Velocity', 'Leader_Delta_Velocity']].values.astype(np.float32)
        additional_val = group_df[['Leader_Acceleration']].values.astype(np.float32)
        a += 1
        ids.append(group_id)
        
        input_I.append(seq_I)
        input_J.append(seq_J)
        additional_value.append(additional_val)

    max_sequence_length = max(len(seq) for seq in input_I + input_J)
    print(f"Sequence Length: {max_sequence_length}  Number of Groups = {a}")

    padded_input_I = tf.keras.preprocessing.sequence.pad_sequences(input_I, dtype="float32", padding="post")
    padded_input_J = tf.keras.preprocessing.sequence.pad_sequences(input_J, dtype="float32", padding="post")
    padded_additional_value = tf.keras.preprocessing.sequence.pad_sequences(additional_value, dtype="float32", padding="post")

    return padded_input_I, padded_input_J, padded_additional_value, max_sequence_length, ids



csvFromSQL = "processedI80.csv" #replace with csv from MySQL query



df = pd.read_csv(csvFromSQL)
input_values_I, input_values_J, additional_value_input, max_sequence_length, ids = preprocess_data(df)

def lstm_network(sequence_length, input_shape):
    inputs = Input(shape=(sequence_length, input_shape), dtype=tf.float32)
    mask = Masking(mask_value=0.0)(inputs)
    lstm_out = LSTM(10, return_sequences=False)(mask)
    output = Dense(1, activation=None)(lstm_out)
    model = Model(inputs, output)
    return model

class ComputationalGraph(tf.Module):
    def __init__(self, initial_values=None):
        super().__init__()
        if initial_values is None:
            initial_values = {
                'maxAcceleration' : 0.73,
                'deceleration' : 1.63,
                'desiredTimeHeadway' : 1.5,
                'desiredVelocity' : 30,
                'constant' : 4,
                'minSpace' : 2,
                'nonlinJam' : 3
            }
        
        self.maxAcceleration = tf.Variable(initial_values['maxAcceleration'], dtype=tf.float32, name='maxAcceleration')
        self.deceleration = tf.Variable(initial_values['deceleration'], dtype=tf.float32, name='deceleration')
        self.desiredTimeHeadway = tf.Variable(initial_values['desiredTimeHeadway'], dtype=tf.float32, name='desiredTimeHeadway')
        self.desiredVelocity = tf.Variable(initial_values['desiredVelocity'], dtype=tf.float32, name='desiredVelocity')
        self.constant = tf.Variable(initial_values['constant'], dtype=tf.float32, name='constant')
        self.minSpace = tf.Variable(initial_values['minSpace'], dtype=tf.float32, name='minSpace')
        self.nonlinJam = tf.Variable(initial_values['nonlinJam'], dtype=tf.float32, name='nonlinJam')

    def compute_AphyJ(self, space, deltaVelocity, velocity, mask):
        accel_times_decel = self.maxAcceleration * self.deceleration
        velocities = velocity * deltaVelocity
        velocity_by_time = velocity * self.desiredTimeHeadway
        velocity_over_desVel = velocity / self.desiredVelocity

        accel_times_decel_squRoot = tf.sqrt(tf.maximum(accel_times_decel, 1e-6))
        velocity_over_desVel_power = tf.pow(tf.maximum(velocity_over_desVel, 1e-6), self.constant)

        vel_over_acc_dec = velocities / tf.maximum(accel_times_decel_squRoot, 1e-6)
        half_vel_over_acc_dec = vel_over_acc_dec * 0.5
        
        nonlinJam = self.nonlinJam
        velSqrt = tf.sqrt(tf.maximum(velocity_over_desVel, 1e-6))
        nonlinvel = nonlinJam * velSqrt

        sStar = half_vel_over_acc_dec + self.minSpace + velocity_by_time + nonlinvel
        v_dv_power_neg = velocity_over_desVel_power * -1
        v_dv_power_neg_inc = v_dv_power_neg + 1.0

        sStar_over_space = sStar / tf.maximum(space, 1e-6)
        sStar_over_space_square = tf.square(sStar_over_space)
        sStar_over_space_square_neg = sStar_over_space_square * -1

        combineValue = sStar_over_space_square_neg + v_dv_power_neg_inc
        AphyJ = combineValue * self.maxAcceleration

        mask = tf.cast(mask, tf.float32)
        AphyJ = AphyJ * mask
        
        return AphyJ

input_shape = input_values_I.shape[-1]

lstm_model = lstm_network(max_sequence_length, input_shape)

comp_graph = ComputationalGraph()

optimizer1 = tf.optimizers.Adam(learning_rate=0.0001, clipnorm=1.0)
optimizer2 = tf.optimizers.Adam(learning_rate=0.0001)

mse_loss_fn = tf.keras.losses.MeanSquaredError()

def compute_loss(AphyJ, AnnJ, AnnI, additional_value):
    MSEc = mse_loss_fn(AphyJ, AnnJ)
    MSEo = mse_loss_fn(additional_value, AnnI)
    alpha = 0.7
    loss = ((1 - alpha) * MSEc) + (alpha * MSEo)
    
    return loss, MSEc, MSEo

@tf.function
def train_step(inputs_I, inputs_J, additional_value, comp_graph, lstm_model):
    inputs_I = tf.cast(inputs_I, tf.float32)
    inputs_J = tf.cast(inputs_J, tf.float32)
    additional_value = tf.cast(additional_value, tf.float32)
    
    with tf.GradientTape() as tape:
        space_J, deltaVelocity_J, velocity_J = tf.split(inputs_J, 3, axis=2)
        mask = tf.reduce_any(tf.not_equal(inputs_J, 0.0), axis=-1, keepdims=True)

        AphyJ = comp_graph.compute_AphyJ(space_J, deltaVelocity_J, velocity_J, mask)
        AnnJ = lstm_model(inputs_J, training=True)
        AnnI = lstm_model(inputs_I, training=True)

        loss, MSEc, MSEo = compute_loss(AphyJ, AnnJ, AnnI, additional_value)
    
    compGraphVariables = list(comp_graph.trainable_variables)
    lstmVariables = list(lstm_model.trainable_variables)
    
    gradients = tape.gradient(loss, compGraphVariables + lstmVariables)
    comp_graph_gradients = gradients[:len(compGraphVariables)]
    lstm_gradients = gradients[len(compGraphVariables):]

    optimizer1.apply_gradients(zip(comp_graph_gradients, compGraphVariables))
    optimizer2.apply_gradients(zip(lstm_gradients, lstmVariables))

    return loss, MSEc, MSEo

num_epochs = 1
batch_size = 1
dataset = tf.data.Dataset.from_tensor_slices((input_values_I, input_values_J, additional_value_input)).batch(batch_size)

result_storage = pd.DataFrame(columns=["Sequence_Number", "MSEc", "MSEo", "Loss"])
large_loss_list = []
for epoch in range(num_epochs):
    loss_list = []
    i = 0
    for inputs_I_batch, inputs_J_batch, additional_value_batch in dataset:
        loss, MSEc, MSEo = train_step(inputs_I_batch, inputs_J_batch, additional_value_batch, comp_graph, lstm_model)
        print(f"Epoch {epoch}.{i}, Loss: {loss.numpy()},  MSEc: {MSEc.numpy()},   MSEo: {MSEo.numpy()}")
        new_row = {"Sequence_Number": ids[i], "MSEc": MSEc.numpy(), "MSEo": MSEo.numpy(), "Loss": loss.numpy()}
        result_storage.loc[len(result_storage)] = new_row
        
        loss_list.append(loss.numpy())
        i += 1

    loss_average = np.average(loss_list)
    loss_median = np.median(loss_list)
    
    print("\n\n\n////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////")
    print(f"Epoch {epoch}       loss average: {loss_average}      loss median: {loss_median}")
    print("////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////\n\n\n")

result_storage.to_csv("ReportLoss.csv", index=False)