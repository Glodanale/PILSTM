import tensorflow as tf
import numpy as np

mse_loss_fn = tf.keras.losses.MeanSquaredError()


def compute_loss(AphyJ, AnnJ, AnnI, additional_value, alpha=0.7):
    """
    Custom loss combining computational and observational loss components.
    """
    MSEc = mse_loss_fn(AphyJ, AnnJ)
    MSEo = mse_loss_fn(additional_value, AnnI)
    return ((1 - alpha) * MSEc) + (alpha * MSEo)


@tf.function
def train_step(inputs_I, inputs_J, additional_value,
               use_mask, comp_graph, lstm_model,
               optimizer1, optimizer2):
    """
    Single training step.
    Applies gradients to both the computational graph and the LSTM model.
    """
    inputs_I = tf.cast(inputs_I, tf.float32)
    inputs_J = tf.cast(inputs_J, tf.float32)
    additional_value = tf.cast(additional_value, tf.float32)

    with tf.GradientTape() as tape:
        space_J, deltaVelocity_J, velocity_J = tf.split(inputs_J, 3, axis=2)

        if use_mask:
            mask = tf.reduce_any(tf.not_equal(inputs_J, 0.0), axis=-1, keepdims=True)
            AphyJ = comp_graph.compute_AphyJ(space_J, deltaVelocity_J, velocity_J, mask)
        else:
            AphyJ = comp_graph.compute_AphyJ(space_J, deltaVelocity_J, velocity_J)

        AnnJ = lstm_model(inputs_J, training=True)
        AnnI = lstm_model(inputs_I, training=True)

        loss = compute_loss(AphyJ, AnnJ, AnnI, additional_value)

    comp_vars = comp_graph.trainable_variables
    lstm_vars = lstm_model.trainable_variables
    gradients = tape.gradient(loss, list(comp_vars) + lstm_vars)

    comp_grads = gradients[:len(comp_vars)]
    lstm_grads = gradients[len(comp_vars):]

    optimizer1.apply_gradients(zip(comp_grads, comp_vars))
    optimizer2.apply_gradients(zip(lstm_grads, lstm_vars))

    return loss


@tf.function
def validate_test_step(inputs_I, inputs_J, additional_value, use_mask, comp_graph, lstm_model):
    inputs_I = tf.cast(inputs_I, tf.float32)
    inputs_J = tf.cast(inputs_J, tf.float32)
    additional_value = tf.cast(additional_value, tf.float32)
    
    if use_mask:
        mask = tf.reduce_any(tf.not_equal(inputs_J, 0.0), axis=-1, keepdims=True)
        AphyJ = comp_graph.compute_AphyJ(*tf.split(inputs_J, 3, axis=2), mask)
    else:
        AphyJ = comp_graph.compute_AphyJ(*tf.split(inputs_J, 3, axis=2))

    AnnJ = lstm_model(inputs_J, training=False)
    AnnI = lstm_model(inputs_I, training=False)

    loss = compute_loss(AphyJ, AnnJ, AnnI, additional_value)
    return loss

