import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.python.keras import testing_utils
# from tensorflow.python.framework import test_util  # deprecated

from layernorm_simplernn import LayernormSimpleRNN
# from keras_layernorm_rnn import LayernormSimpleRNN


# @test_util.run_all_in_graph_and_eager_modes
class LayernormSimpleRNNTest(tf.test.TestCase):
    def test_dynamic_behavior_layernorm_rnn(self):
        num_samples = 2
        timesteps = 3
        embedding_dim = 4
        units = 2
        layer = LayernormSimpleRNN(
            units, use_layernorm=True, input_shape=(None, embedding_dim))
        model = keras.models.Sequential()
        model.add(layer)
        model.compile('rmsprop', 'mse')
        x = np.random.random((num_samples, timesteps, embedding_dim))
        y = np.random.random((num_samples, units))
        model.train_on_batch(x, y)

    def test_constraints_layernorm_rnn(self):
        embedding_dim = 4
        layer_class = LayernormSimpleRNN
        k_constraint = keras.constraints.max_norm(0.01)
        r_constraint = keras.constraints.max_norm(0.01)
        b_constraint = keras.constraints.max_norm(0.01)
        g_constraint = keras.constraints.max_norm(0.01)
        layer = layer_class(
            5,
            use_layernorm=True,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_constraint=k_constraint,
            recurrent_constraint=r_constraint,
            bias_constraint=b_constraint,
            gamma_constraint=g_constraint)
        layer.build((None, None, embedding_dim))
        self.assertEqual(layer.cell.kernel.constraint, k_constraint)
        self.assertEqual(layer.cell.recurrent_kernel.constraint, r_constraint)
        self.assertEqual(layer.cell.bias.constraint, b_constraint)
        self.assertEqual(layer.cell.gamma.constraint, g_constraint)

    def test_with_masking_layer_layernorm_rnn(self):
        layer_class = LayernormSimpleRNN
        inputs = np.random.random((2, 3, 4))
        targets = np.abs(np.random.random((2, 3, 5)))
        targets /= targets.sum(axis=-1, keepdims=True)
        model = keras.models.Sequential()
        model.add(keras.layers.Masking(input_shape=(3, 4)))
        model.add(
            layer_class(
                units=5,
                use_layernorm=True,
                return_sequences=True,
                unroll=False))
        model.compile(loss='categorical_crossentropy', optimizer='rmsprop')
        model.fit(inputs, targets, epochs=1, batch_size=2, verbose=1)

    def test_from_config_layernorm_rnn(self):
        layer_class = LayernormSimpleRNN
        for stateful in (False, True):
            l1 = layer_class(units=1, use_layernorm=True, stateful=stateful)
            l2 = layer_class.from_config(l1.get_config())
            assert l1.get_config() == l2.get_config()

    def test_regularizers_layernorm_rnn(self):
        embedding_dim = 4
        layer_class = LayernormSimpleRNN
        layer = layer_class(
            5,
            use_layernorm=True,
            return_sequences=False,
            weights=None,
            input_shape=(None, embedding_dim),
            kernel_regularizer=keras.regularizers.l1(0.01),
            recurrent_regularizer=keras.regularizers.l1(0.01),
            bias_regularizer='l2',
            gamma_regularizer='l2',
            activity_regularizer='l1')
        layer.build((None, None, 2))
        self.assertEqual(len(layer.losses), 4)

    def test_versus_simplernn(self):
        embedding_dim = 4
        timesteps = 2
        settings = {
            'units': 3,
            'bias_initializer': 'ones',
            'kernel_initializer': 'ones',
            'recurrent_initializer': 'ones'
        }
        model1 = keras.Sequential()
        model1.add(keras.layers.SimpleRNN(**settings))
        model1.build((None, None, embedding_dim))

        model2 = keras.Sequential()
        model2.add(LayernormSimpleRNN(use_layernorm=False, **settings))
        model2.build((None, None, embedding_dim))

        x = 0.5 * np.ones((1, timesteps, embedding_dim))
        y_pred1 = model1.predict(x)
        y_pred2 = model2.predict(x)
        self.assertAllEqual(y_pred1, y_pred2)


if __name__ == "__main__":
    tf.test.main()


# Excluded tests
# test_return_sequences_layernorm_rnn
# test_float64_layernorm_rnn
# test_dropout_layernorm_rnn
# test_statefulness_layernorm_rnn
