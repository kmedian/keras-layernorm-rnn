import tensorflow as tf
import tensorflow.keras as keras


class LayernormLSTM1Cell(keras.layers.LSTMCell):
    """LSTM with Layer Normalization (1: more smaller multiplications)

    Notice:
        LayernormLSTM1Cell(use_layernorm=False) is the same
        as LSTMCell(implementation=1)

    References:
    [1] Hochreiter, S., Schmidhuber, J., 1997. Long short-term memory.
          Neural computation 9, 1735–1780.
    [2] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
          "Layer Normalization." ArXiv:1607.06450 [Cs, Stat],
          July 21, 2016. http://arxiv.org/abs/1607.06450

    """
    def __init__(
            self,
            units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            use_layernorm=True,
            use_gamma=True,
            layernorm_epsilon=1e-05,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            gamma_initializer='ones',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            gamma_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            gamma_constraint=None,
            dropout=0.,
            recurrent_dropout=0.,
            **kwargs):
        # store LSTM attributes
        keras.layers.LSTMCell.__init__(
            self,
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            implementation=1,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        # store layernorm attributes
        self.use_layernorm = use_layernorm
        self.use_gamma = use_gamma
        self.layernorm_epsilon = layernorm_epsilon
        self.gamma_initializer = keras.initializers.get(gamma_initializer)
        self.gamma_regularizer = keras.regularizers.get(gamma_regularizer)
        self.gamma_constraint = keras.constraints.get(gamma_constraint)

    def build(self, input_shape):
        # build input kernel, recurrent kernel and bias in LSTMCell
        keras.layers.LSTMCell.build(self, input_shape)

        # build the layernorm objects
        if self.use_layernorm:
            ln_config = {
                'axis': -1,
                'epsilon': self.layernorm_epsilon,
                'center': False,
                'scale': self.use_gamma,
                'beta_initializer': None,
                'gamma_initializer': self.gamma_initializer,
                'beta_regularizer': None,
                'gamma_regularizer': self.gamma_regularizer,
                'beta_constraint': None,
                'gamma_constraint': self.gamma_constraint,
                'dtype': self.dtype,
                'trainable': self.trainable
            }
            self.layernorm_i = keras.layers.LayerNormalization(**ln_config)
            self.layernorm_f = keras.layers.LayerNormalization(**ln_config)
            self.layernorm_c = keras.layers.LayerNormalization(**ln_config)
            self.layernorm_o = keras.layers.LayerNormalization(**ln_config)

    def call(self, inputs, states, training=None):
        # read results from previous time step
        h_tm1 = states[0]  # previous memory state
        c_tm1 = states[1]  # previous carry state

        # IMPLEMENTATION 1
        # - Fewer small operations instead of few big operations

        # dropout, input kernel
        if (training is not None) and (0. < self.dropout < 1.):
            # generate 4x dropout masks from DropoutRNNCellMixin
            dp_mask = self.get_dropout_mask_for_cell(
                inputs, training, count=4)
            # apply dropouts
            inputs_i = inputs * dp_mask[0]
            inputs_f = inputs * dp_mask[1]
            inputs_c = inputs * dp_mask[2]
            inputs_o = inputs * dp_mask[3]
        else:
            inputs_i, inputs_f, inputs_c, inputs_o = (
                inputs, inputs, inputs, inputs)

        # multiply inputs with weight matrix
        W_i, W_f, W_c, W_o = tf.split(
            self.kernel, num_or_size_splits=4, axis=1)
        net_i = keras.backend.dot(inputs_i, W_i)
        net_f = keras.backend.dot(inputs_f, W_f)
        net_c = keras.backend.dot(inputs_c, W_c)
        net_o = keras.backend.dot(inputs_o, W_o)

        # dropout, recurrent kernel
        if (training is not None) and (0. < self.recurrent_dropout < 1.):
            # generate 4x dropout masks from DropoutRNNCellMixin
            rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
                h_tm1, training, count=4)
            # apply dropouts
            h_tm1_i = h_tm1 * rec_dp_mask[0]
            h_tm1_f = h_tm1 * rec_dp_mask[1]
            h_tm1_c = h_tm1 * rec_dp_mask[2]
            h_tm1_o = h_tm1 * rec_dp_mask[3]
        else:
            h_tm1_i, h_tm1_f, h_tm1_c, h_tm1_o = h_tm1, h_tm1, h_tm1, h_tm1

        # multiply previous hidden with recurrent weights
        R_i, R_f, R_c, R_o = tf.split(
            self.recurrent_kernel, num_or_size_splits=4, axis=1)
        net_i += keras.backend.dot(h_tm1_i, R_i)
        net_f += keras.backend.dot(h_tm1_f, R_f)
        net_c += keras.backend.dot(h_tm1_c, R_c)
        net_o += keras.backend.dot(h_tm1_o, R_o)

        # apply scaling of layer normalization to each gate
        if self.use_layernorm:
            net_i = self.layernorm_i(net_i)
            net_f = self.layernorm_f(net_f)
            net_c = self.layernorm_c(net_c)
            net_o = self.layernorm_o(net_o)

        # apply bias
        if self.use_bias:
            bias_i, bias_f, bias_c, bias_o = tf.split(
                self.bias, num_or_size_splits=4, axis=0)
            net_i = keras.backend.bias_add(net_i, bias_i)
            net_f = keras.backend.bias_add(net_f, bias_f)
            net_c = keras.backend.bias_add(net_c, bias_c)
            net_o = keras.backend.bias_add(net_o, bias_o)

        # _compute_carry_and_output_fused
        i = self.recurrent_activation(net_i)
        f = self.recurrent_activation(net_f)
        c = f * c_tm1 + i * self.activation(net_c)
        o = self.recurrent_activation(net_o)

        # merge all signals into memory state h
        h = o * self.activation(c)

        # return memory state h, and carry state c
        return h, [h, c]

    def get_config(self):
        config = {
            'use_layernorm': self.use_layernorm,
            'use_gamma': self.use_gamma,
            'layernorm_epsilon': self.layernorm_epsilon,
            'gamma_initializer': self.gamma_initializer,
            'gamma_regularizer': self.gamma_regularizer,
            'gamma_constraint': self.gamma_constraint
        }

        cell_config = keras.layers.LSTMCell.get_config(self)
        del cell_config['name']

        return {**config, **cell_config}


class LayernormLSTM1(keras.layers.LSTM):
    def __init__(
            self,
            units,
            activation='tanh',
            recurrent_activation='hard_sigmoid',
            use_bias=True,
            use_layernorm=True,
            use_gamma=True,
            layernorm_epsilon=1e-05,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            gamma_initializer='ones',
            unit_forget_bias=True,
            kernel_regularizer=None,
            recurrent_regularizer=None,
            bias_regularizer=None,
            gamma_regularizer=None,
            activity_regularizer=None,
            kernel_constraint=None,
            recurrent_constraint=None,
            bias_constraint=None,
            gamma_constraint=None,
            dropout=0.,
            recurrent_dropout=0.,
            return_sequences=False,
            return_state=False,
            go_backwards=False,
            stateful=False,
            unroll=False,
            **kwargs):
        # instantiate the RNN cell
        cell = LayernormLSTM1Cell(
            units,
            activation=activation,
            recurrent_activation=recurrent_activation,
            use_bias=use_bias,
            use_layernorm=use_layernorm,
            use_gamma=use_gamma,
            layernorm_epsilon=layernorm_epsilon,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            gamma_initializer=gamma_initializer,
            unit_forget_bias=unit_forget_bias,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            gamma_regularizer=gamma_regularizer,
            activity_regularizer=activity_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            gamma_constraint=gamma_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        # call the parent class 'RNN'
        keras.layers.RNN.__init__(
            self,
            cell,
            return_sequences=return_sequences,
            return_state=return_state,
            go_backwards=go_backwards,
            stateful=stateful,
            unroll=unroll,
            **kwargs)
        # set other parameters
        self.activity_regularizer = keras.regularizers.get(
            activity_regularizer)
        self.input_spec = [keras.layers.InputSpec(ndim=3)]
        # no cuDNN supoort
        self.could_use_cudnn = False
        # return_runtime is a flag for testing (grappler, graph mode)
        self.return_runtime = kwargs.pop('return_runtime', False)

    @property
    def use_layernorm(self):
        return self.cell.use_layernorm

    @property
    def use_gamma(self):
        return self.cell.use_gamma

    @property
    def layernorm_epsilon(self):
        return self.cell.layernorm_epsilon

    @property
    def gamma_initializer(self):
        return self.cell.gamma_initializer

    @property
    def gamma_regularizer(self):
        return self.cell.gamma_regularizer

    @property
    def gamma_constraint(self):
        return self.cell.gamma_constraint

    def get_config(self):
        base_config = keras.layers.RNN.get_config(self)
        del base_config['cell']

        cell_config = self.cell.get_config()
        return {**base_config, **cell_config}
