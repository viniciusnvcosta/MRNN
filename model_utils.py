"""Utility functions for MRNN modelling.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar,
           "Estimating Missing Data in Temporal Data Streams Using
           Multi-Directional Recurrent Neural Networks,"
           in IEEE Transactions on Biomedical Engineering,
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
---------------------------------------------------
(1) process_batch_input_for_rnn: Convert tensor for rnn training
(2) initial_point_interpolation: Initial point interpolation
(3) BiGRUCell: Bidirectional GRU Cell
"""

import tensorflow as tf
from keras import layers
from keras import initializers
import numpy as np


def initial_point_interpolation(x, m, t, imputed_x):
    """Initial point interpolation.

    If the variable at time point 0 is missing, do zero-hold interpolation.

    Args:
      - x: original features
      - m: masking matrix
      - t: time information
      - imputed_x: imputed data

    Returns:
      - imputed_x: imputed and interpolated data
    """

    no, seq_len, dim = x.shape

    for i in range(no):
        for k in range(dim):
            for j in range(seq_len):
                # If there is no previous measurements
                if t[i, j, k] > j:
                    idx = np.where(m[i, :, k] == 1)[0]
                    # Do zero-hold interpolation
                    imputed_x[i, j, k] = x[i, np.min(idx), k]

    return imputed_x


class biGRUCell(layers.Layer):
    """Bi-directional GRU cell object.

    Attributes:
      - input_size = Input Vector size
      - hidden_layer_size = Hidden layer size
      - target_size = Output vector size
    """

    def __init__(self, input_size, hidden_layer_size, target_size):
        super(biGRUCell, self).__init__()
        # Builder method for the layer

        # Initialization of given values
        self.input_size = input_size
        self.hidden_layer_size = hidden_layer_size
        self.target_size = target_size

        # Initialize weights and biases for forward and backward GRU cells
        self.init_weights()

    def init_weights(self):
        # Foward GRU weights and biases
        self.Wr = self.add_weight(
            shape=[self.input_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Wr",
            trainable=True,
        )
        self.Ur = self.add_weight(
            shape=[self.hidden_layer_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Ur",
            trainable=True,
        )
        self.br = self.add_weight(
            shape=[self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="br",
            trainable=True,
        )

        self.Wu = self.add_weight(
            shape=[self.input_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Wu",
            trainable=True,
        )
        self.Uu = self.add_weight(
            shape=[self.hidden_layer_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Uu",
            trainable=True,
        )
        self.bu = self.add_weight(
            shape=[self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="bu",
            trainable=True,
        )

        self.Wh = self.add_weight(
            shape=[self.input_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Wh",
            trainable=True,
        )
        self.Uh = self.add_weight(
            shape=[self.hidden_layer_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Uh",
            trainable=True,
        )
        self.bh = self.add_weight(
            shape=[self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="bh",
            trainable=True,
        )

        # Backward GRU weights and biases
        self.Wr1 = self.add_weight(
            shape=[self.input_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            trainable=True,
        )
        self.Ur1 = self.add_weight(
            shape=[self.hidden_layer_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            trainable=True,
        )
        self.br1 = self.add_weight(
            shape=[self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="br1",
            trainable=True,
        )

        self.Wu1 = self.add_weight(
            shape=[self.input_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Wu1",
            trainable=True,
        )
        self.Uu1 = self.add_weight(
            shape=[self.hidden_layer_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Uu1",
            trainable=True,
        )
        self.bu1 = self.add_weight(
            shape=[self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="bu1",
            trainable=True,
        )

        self.Wh1 = self.add_weight(
            shape=[self.input_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Wh1",
            trainable=True,
        )
        self.Uh1 = self.add_weight(
            shape=[self.hidden_layer_size, self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="Uh1",
            trainable=True,
        )
        self.bh1 = self.add_weight(
            shape=[self.hidden_layer_size],
            initializer=initializers.Zeros(),
            name="bh1",
            trainable=True,
        )

        # Output layer weights and biases
        self.Wo = self.add_weight(
            shape=[self.hidden_layer_size * 2, self.target_size],
            initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            name="Wo",
            trainable=True,
        )
        self.bo = self.add_weight(
            shape=[self.target_size],
            initializer=initializers.TruncatedNormal(mean=0.0, stddev=0.01),
            name="bo",
            trainable=True,
        )

    def call(self, inputs):
        forward_input, backward_input = inputs

        processed_input = self.process_batch_input_for_rnn(forward_input)
        processed_input_rev = self.process_batch_input_for_rnn(backward_input)

        # Initial hidden state
        initial_hidden = tf.matmul(
            inputs[:, 0, :], tf.zeros([self.input_size, self.hidden_layer_size])
        )

        # Forward pass
        forward_hidden_states = tf.scan(
            self.forward_gru,
            processed_input,
            initializer=initial_hidden,
            name="states_f",
        )

        # Backward pass
        backward_hidden_states = tf.scan(
            self.backward_gru,
            processed_input_rev,
            initializer=initial_hidden,
            name="states_b",
        )

        # Concatenate the forward and backward hidden states outputs
        outputs = tf.concat([forward_hidden_states, backward_hidden_states], axis=-1)

        # Output layer
        outputs = tf.nn.sigmoid(tf.matmul(outputs, self.Wo) + self.bo)
        return outputs

    def forward_gru(self, previous_hidden_state, x):
        """Function for Forward GRU cell.

        This function takes previous hidden state
        and memory tuple with input and
        outputs current hidden state.

        Args:
          - previous_hidden_state
          - x

        Returns:
          - current_hidden_state
        """
        # R Gate
        r = tf.nn.sigmoid(
            tf.matmul(x, self.Wr) + tf.matmul(previous_hidden_state, self.Ur) + self.br
        )
        # U Gate
        u = tf.nn.sigmoid(
            tf.matmul(x, self.Wu) + tf.matmul(previous_hidden_state, self.Uu) + self.bu
        )
        # Final Memory cell
        c = tf.nn.tanh(
            tf.matmul(x, self.Wh)
            + tf.matmul(r * previous_hidden_state, self.Uh)
            + self.bh
        )
        current_hidden_state = u * previous_hidden_state + (1 - u) * c
        return current_hidden_state

    def backward_gru(self, previous_hidden_state, x):
        """Function for Backward GRU cell.

        This function takes previous hidden state
        and memory tuple with input and
        outputs current hidden state.

        Args:
            - previous_hidden_state
            - x

        Returns:
            - current_hidden_state
        """

        # R Gate
        r = tf.nn.sigmoid(
            tf.matmul(x, self.Wr1)
            + tf.matmul(previous_hidden_state, self.Ur1)
            + self.br1
        )
        # U Gate
        u = tf.nn.sigmoid(
            tf.matmul(x, self.Wu1)
            + tf.matmul(previous_hidden_state, self.Uu1)
            + self.bu1
        )
        # Final Memory cell
        c = tf.nn.tanh(
            tf.matmul(x, self.Wh1)
            + tf.matmul(r * previous_hidden_state, self.Uh1)
            + self.bh1
        )
        # Current Hidden state
        current_hidden_state = u * previous_hidden_state + (1 - u) * c
        return current_hidden_state

    def process_batch_input_for_rnn(self, batch_input):
        """Convert tensor for rnn training.

        The transformed input should have the shape of [no of series, seq_len, dim]

        Args:
        - batch_input: original batch input

        Returns:
        - transformed_input: converted batch input for RNN
        """
        # this was the original code
        # batch_input_ = tf.transpose(batch_input, [2, 0, 1])
        # return tf.transpose(batch_input_)
        # note: This results in no change to the shape of the input tensor, which makes this version essentially a no-op.

        return tf.transpose(batch_input, [1, 0, 2])

    # --------------------------------------------------

    # def get_states_f(self):
    #     """Function to get the hidden and memory cells after forward pass.

    #     Iterates through time/ sequence to get all hidden state

    #     Returns:
    #       - all_hidden_states
    #     """
    #     # Getting all hidden state through time
    #     all_hidden_states = tf.scan(
    #         self.forward_gru,
    #         self.processed_input,
    #         initializer=self.initial_hidden,
    #         name="states",
    #     )
    #     return all_hidden_states

    # def get_states_b(self):
    #     """Function to get the hidden and memory cells after backward pass.

    #     Iterates through time/ sequence to get all hidden state

    #     Returns:
    #       - all_hidden_states
    #     """
    #     all_hidden_memory_states = tf.scan(
    #         self.forward_gru,
    #         self.processed_input_rev,
    #         initializer=self.initial_hidden,
    #         name="states",
    #     )
    #     # Now reversing the states to keep those in original order
    #     all_hidden_states = tf.reverse(all_hidden_memory_states, [1])
    #     return all_hidden_states

    # def get_concat_hidden(self):
    #     """Function to concat the hiddenstates for backward and forward pass.

    #     Returns:
    #       - concat_hidden
    #     """
    #     # Getting hidden and memory for the forward pass
    #     all_hidden_states_f = self.get_states_f()
    #     # Getting hidden and memory for the backward pass
    #     all_hidden_states_b = self.get_states_b()
    #     # Concating the hidden states of forward and backward pass
    #     concat_hidden = tf.concat([all_hidden_states_f, all_hidden_states_b], 2)

    #     return concat_hidden

    # def get_output(self, hidden_state):
    #     """Function to get output from a hidden layer.

    #     This function takes hidden state and returns output

    #     Returns:
    #       - output
    #     """
    #     output = tf.nn.sigmoid(tf.matmul(hidden_state, self.Wo) + self.bo)
    #     return output

    # def get_outputs(self):
    #     """Function for getting all output layers.

    #     Iterating through hidden states to get outputs for all timestamp

    #     Returns:
    #       - all_outputs
    #     """
    #     all_hidden_states = self.get_concat_hidden()
    #     all_outputs = tf.map_fn(self.get_output, all_hidden_states)

    #     return all_outputs
