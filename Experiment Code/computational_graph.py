import tensorflow as tf

class ComputationalGraph(tf.Module):
    def __init__(self, non_linear=False, precision=tf.float32):
        super().__init__()
        self.maxAcceleration = tf.Variable(0.73, dtype=precision)
        self.deceleration = tf.Variable(1.63, dtype=precision)
        self.desiredTimeHeadway = tf.Variable(1.5, dtype=precision)
        self.desiredVelocity = tf.Variable(30, dtype=precision)
        self.constant = tf.Variable(4, dtype=precision)
        self.minSpace = tf.Variable(2, dtype=precision)
        
        self.non_linear = non_linear
        if self.non_linear:
            self.nonlinJam = tf.Variable(3, dtype=precision)

    def compute_AphyJ(self, space, deltaVelocity, velocity, mask=None):
        # Ensure all inputs match the dtype of the internal variables
        dtype = self.maxAcceleration.dtype
        space = tf.cast(space, dtype)
        deltaVelocity = tf.cast(deltaVelocity, dtype)
        velocity = tf.cast(velocity, dtype)
        if mask is not None:
            mask = tf.cast(mask, dtype)

        accel_times_decel = self.maxAcceleration * self.deceleration
        velocity_over_desVel = velocity / self.desiredVelocity

        accel_times_decel_sqrt = tf.sqrt(tf.maximum(accel_times_decel, 1e-6))
        velocity_over_desVel_power = tf.pow(tf.maximum(velocity_over_desVel, 1e-6), self.constant)

        sStar = self.minSpace + velocity * self.desiredTimeHeadway + velocity * deltaVelocity / accel_times_decel_sqrt
        if self.non_linear:
            sStar += self.nonlinJam * tf.sqrt(tf.maximum(velocity_over_desVel, 1e-6))

        AphyJ = (self.maxAcceleration * (sStar / tf.maximum(space, 1e-6))**2) - velocity_over_desVel_power
        if mask is not None:
            AphyJ *= mask

        return AphyJ

