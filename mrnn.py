"""MRNN core functions.

Reference: Jinsung Yoon, William R. Zame and Mihaela van der Schaar,
           "Estimating Missing Data in Temporal Data Streams Using
           Multi-Directional Recurrent Neural Networks,"
           in IEEE Transactions on Biomedical Engineering,
           vol. 66, no. 5, pp. 1477-1490, May 2019.

Paper Link: https://ieeexplore.ieee.org/document/8485748
Contact: jsyoon0823@gmail.com
---------------------------------------------------
(1) Train RNN part
(2) Test RNN part
(3) Train FC part
(4) Test FC part
"""

import logging
import tensorflow as tf
import keras
import numpy as np
from model_utils import biGRUCell, initial_point_interpolation

tf.get_logger().setLevel(logging.ERROR)


class MRNN:
    """MRNN class with core functions.

    Attributes:
      - x: incomplete data
      - model_parameters:
        - h_dim: hidden state dimensions
        - batch_size: the number of samples in mini-batch
        - iteration: the number of iteration
        - learning_rate: learning rate of model training
    """

    def __init__(self, x, model_parameters):
        self.no, self.seq_len, self.dim = x.shape
        self.h_dim = model_parameters["h_dim"]
        self.batch_size = model_parameters["batch_size"]
        self.iteration = model_parameters["iteration"]
        self.learning_rate = model_parameters["learning_rate"]

    def rnn_train(self, x, m, t, f):
        """Train RNN for each feature.

        Args:
        - x: incomplete data
        - m: mask matrix
        - t: time matrix
        - f: feature index
        """

        # Build rnn object
        rnn = biGRUCell(3, self.h_dim, 1)

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        # Training loop
        for i in range(self.iteration):
            # Batch selection
            batch_idx = np.random.permutation(x.shape[0])[: self.batch_size]

            temp_input = np.dstack((x[:, :, f], m[:, :, f], t[:, :, f]))

            forward_input = np.zeros([self.batch_size, self.seq_len, 3])
            # replace all elements from the second time step onwards
            forward_input[:, 1:, :] = temp_input[batch_idx, : (self.seq_len - 1), :]

            temp_input_reverse = np.flip(temp_input, 1)

            backward_input = np.zeros([self.batch_size, self.seq_len, 3])
            backward_input[:, 1:, :] = temp_input_reverse[
                batch_idx, : (self.seq_len - 1), :
            ]

            forward_input_tensor = tf.convert_to_tensor(forward_input, dtype=tf.float32)
            backward_input_tensor = tf.convert_to_tensor(
                backward_input, dtype=tf.float32
            )

            with tf.GradientTape() as tape:
                mask_tensor = tf.convert_to_tensor(
                    np.transpose(np.dstack(m[batch_idx, :, f]), [1, 2, 0]),
                    dtype=tf.float32,
                )
                target_tensor = tf.convert_to_tensor(
                    np.transpose(np.dstack(x[batch_idx, :, f]), [1, 2, 0]),
                    dtype=tf.float32,
                )
                # * model call
                outputs = rnn((forward_input_tensor, backward_input_tensor))
                step_loss = tf.sqrt(
                    tf.reduce_mean(
                        tf.square(mask_tensor * outputs - mask_tensor * target_tensor)
                    )
                )

            gradients = tape.gradient(step_loss, rnn.trainable_variables)
            optimizer.apply_gradients(zip(gradients, rnn.trainable_variables))

            if i % 100 == 0:  # Print loss every 100 iterations
                print(f"Iteration {i}, Loss: {step_loss.numpy()}")

        # Save model
        save_file_name = f"tmp/mrnn_imputation/rnn_feature_{f + 1}/"
        tf.saved_model.save(rnn, save_file_name)

    def rnn_predict(self, x, m, t):
        """Impute missing data using RNN block.

        Args:
          - x: incomplete data
          - m: mask matrix
          - t: time matrix

        Returns:
          - imputed_x: imputed data by rnn block
        """
        # Output Initialization
        imputed_x = np.zeros([self.no, self.seq_len, self.dim])

        # For each feature
        for f in range(self.dim):
            temp_input = np.dstack((x[:, :, f], m[:, :, f], t[:, :, f]))
            temp_input_reverse = np.flip(temp_input, 1)

            forward_input = np.zeros([self.no, self.seq_len, 3])
            forward_input[:, 1:, :] = temp_input[:, : (self.seq_len - 1), :]

            backward_input = np.zeros([self.no, self.seq_len, 3])
            backward_input[:, 1:, :] = temp_input_reverse[:, : (self.seq_len - 1), :]

            save_file_name = "tmp/mrnn_imputation/rnn_feature_" + str(f + 1) + "/"

            rnn = tf.saved_model.load(save_file_name)
            imputed_data = rnn(forward_input, backward_input)

            imputed_x[:, :, f] = (1 - m[:, :, f]) * np.transpose(
                np.squeeze(imputed_data)
            ) + m[:, :, f] * x[:, :, f]

            # Initial point interpolation for better performance
            imputed_x = initial_point_interpolation(x, m, t, imputed_x)

            return imputed_x

    def fc_train(self, x, m, t):
        """Train Fully Connected Networks after RNN block.

        Args:
            - x: incomplete data
            - m: mask matrix
            - t: time matrix
        """

        # rnn imputation results
        rnn_imputed_x = self.rnn_predict(x, m, t)

        # Reshape the data for FC train
        x = np.reshape(x, [self.no * self.seq_len, self.dim])
        rnn_imputed_x = np.reshape(rnn_imputed_x, [self.no * self.seq_len, self.dim])
        m = np.reshape(m, [self.no * self.seq_len, self.dim])

        # input place holders
        x_input = keras.Input(shape=(self.dim,), dtype=tf.float32)
        target = keras.Input(shape=(self.dim,), dtype=tf.float32)
        mask = keras.Input(shape=(self.dim,), dtype=tf.float32)

        # build a FC network
        initializer = keras.initializers.GlorotUniform()
        U = tf.Variable(initializer(shape=[self.dim, self.dim]), name="U")
        V1 = tf.Variable(initializer(shape=[self.dim, self.dim]), name="V1")
        V2 = tf.Variable(initializer(shape=[self.dim, self.dim]), name="V2")
        b = tf.Variable(tf.random.normal([self.dim]), name="b")

        L1 = tf.nn.sigmoid(
            (
                tf.matmul(
                    x_input,
                    tf.linalg.set_diag(
                        U,
                        np.zeros([self.dim]),
                    ),
                )
                + tf.matmul(
                    target,
                    tf.linalg.set_diag(
                        V1,
                        np.zeros(
                            [
                                self.dim,
                            ]
                        ),
                    ),
                )
                + tf.matmul(mask, V2)
                + b
            )
        )

        W = tf.Variable(tf.random.normal([self.dim]), name="W")
        a = tf.Variable(tf.random.normal([self.dim]), name="a")
        hypothesis = W * L1 + a

        outputs = tf.nn.sigmoid(hypothesis)

        # reshape out for sequence_loss
        loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - target)))

        # Optimizer
        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)

        # Training step
        for i in range(self.iteration * 20):
            batch_idx = np.random.permutation(x.shape[0])[: self.batch_size]
            with tf.GradientTape() as tape:
                L1 = tf.nn.sigmoid(
                    (
                        tf.matmul(
                            x_input,
                            tf.linalg.set_diag(
                                U,
                                np.zeros(
                                    [
                                        self.dim,
                                    ]
                                ),
                            ),
                        )
                        + tf.matmul(
                            target,
                            tf.linalg.set_diag(
                                V1,
                                np.zeros(
                                    [
                                        self.dim,
                                    ]
                                ),
                            ),
                        )
                        + tf.matmul(mask, V2)
                        + b
                    )
                )

                W = tf.Variable(tf.random.normal([self.dim]), name="W")
                a = tf.Variable(tf.random.normal([self.dim]), name="a")
                hypothesis = W * L1 + a

                outputs = tf.nn.sigmoid(hypothesis)

                # reshape out for sequence_loss
                loss = tf.sqrt(tf.reduce_mean(tf.square(outputs - target)))

                grads = tape.gradient(loss, [U, V1, V2, b, W, a])
                optimizer.apply_gradients(zip(grads, [U, V1, V2, b, W, a]))

        optimizer = keras.optimizers.Adam(learning_rate=self.learning_rate)
        grads = tape.gradient(loss, [U, V1, V2, b, W, a])
        optimizer.apply_gradients(zip(grads, [U, V1, V2, b, W, a]))

        # Save model
        inputs = {"x_input": x_input, "target": target, "mask": mask}
        outputs = {"imputation": outputs}

        save_file_name = "tmp/mrnn_imputation/fc_feature/"
        tf.saved_model.save(fc_network, save_file_name)

    def rnn_fc_predict(self, x, m, t):
        """Impute missing data using RNN and FC.

        Args:
          - x: incomplete data
          - m: mask matrix
          - t: time matrix

        Returns:
          - fc_imputed_x: imputed data using RNN and FC
        """

        # rnn imputation results
        rnn_imputed_x = self.rnn_predict(x, m, t)

        # Reshape the data for FC predict
        x_input = np.reshape(x, [self.no * self.seq_len, self.dim])
        rnn_imputed_x = np.reshape(rnn_imputed_x, [self.no * self.seq_len, self.dim])
        m = np.reshape(m, [self.no * self.seq_len, self.dim])

        save_file_name = "tmp/mrnn_imputation/fc_feature/"
        fc_network = tf.saved_model.load(save_file_name)

        fc_imputed_x = fc_network(x_input, rnn_imputed_x, m)

        # Reshape imputed data to 3d array
        fc_imputed_x = np.reshape(fc_imputed_x, [self.no, self.seq_len, self.dim])
        m = np.reshape(m, [self.no, self.seq_len, self.dim])
        x = np.reshape(x, [self.no, self.seq_len, self.dim])

        fc_imputed_x = fc_imputed_x * (1 - m) + x * m
        fc_imputed_x = initial_point_interpolation(x, m, t, fc_imputed_x)

        return fc_imputed_x

    def fit(self, x, m, t):
        """Train the entire MRNN.

        Args:
          - x: incomplete data
          - m: mask matrix
          - t: time matrix
        """
        # Train RNN part
        for f in range(self.dim):
            self.rnn_train(x, m, t, f)
            print(
                "Finish " + str(f + 1) + "-th feature training with RNN for imputation"
            )
        # Train FC part
        self.fc_train(x, m, t)
        print("Finish M-RNN training with both RNN and FC for imputation")

    def transform(self, x, m, t):
        """Impute missing data using the entire MRNN.

        Args:
          - x: incomplete data
          - m: mask matrix
          - t: time matrix

        Returns:
          - imputed_x: imputed data
        """
        # Impute with both RNN and FC part
        imputed_x = self.rnn_fc_predict(x, m, t)

        return imputed_x
