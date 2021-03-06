import tensorflow as tf
import tensorflow.keras as keras


class LayernormSimpleRNNCell(keras.layers.SimpleRNNCell,
                             keras.layers.LayerNormalization):
    """Cell class for LayernormSimpleRNN.

    Motivation:
    - Drop-In Replacement for keras.layers.SimpleRNNCell
    - demonstrate how to add keras.layers.LayerNormalization
       to all RNNs by introducing the `use_layernorm` argument

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
        "Layer Normalization." ArXiv:1607.06450 [Cs, Stat],
        July 21, 2016. http://arxiv.org/abs/1607.06450

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass `None`, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias
        vector.
      use_layernorm: Boolean, (default `True`), whether to apply layer
        normalization (scaling only).
      use_gamma: Boolean (default: True), whether to use gamma weights in
        layer normalization.
      layernorm_epsilon: Float, (default `1e-5`), Small float added to variance
        to avoid dividing by zero.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector (`use_bias=True`) or
         for the beta vector in layer normalization (`use_layernorm=True`).
         Default: `zeros`.
      gamma_initializer: Initializer for the gamma vector of the layer
         normalization layer (`use_layernorm=True`). Default: `ones`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_regularizer: Regularizer function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_constraint: Constraint function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_constraint: Constraint function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      dropout: Float between 0 and 1. Fraction of the units to drop for the
        linear transformation of the inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1. Fraction of the units to drop
        for the linear transformation of the recurrent state. Default: 0.

    Call arguments:
      inputs: A 2D tensor, with shape of `[batch, feature]`.
      states: A 2D tensor with shape of `[batch, units]`, which is the state
        from the previous time step. For timestep 0, the initial state provided
        by the user will be feed to cell.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. Only relevant when `dropout` or
        `recurrent_dropout` is used.

    Examples:

    ```python
    import numpy as np
    import tensorflow.keras as keras
    from keras_layernorm_rnn import LayernormSimpleRNN

    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    rnn = keras.layers.RNN(LayernormSimpleRNNCell(4))

    output = rnn(inputs)  # The output has shape `[32, 4]`.

    rnn = keras.layers.RNN(
        LayernormSimpleRNNCell(4),
        return_sequences=True,
        return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = rnn(inputs)
    ```
    """

    def __init__(
            self,
            units,
            activation='tanh',
            use_bias=True,
            use_layernorm=True,
            use_gamma=True,
            layernorm_epsilon=1e-05,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            gamma_initializer='ones',
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
        keras.layers.SimpleRNNCell.__init__(
            self,
            units,
            activation=activation,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            kernel_regularizer=kernel_regularizer,
            recurrent_regularizer=recurrent_regularizer,
            bias_regularizer=bias_regularizer,
            kernel_constraint=kernel_constraint,
            recurrent_constraint=recurrent_constraint,
            bias_constraint=bias_constraint,
            dropout=dropout,
            recurrent_dropout=recurrent_dropout,
            dtype=kwargs.get('dtype'),
            trainable=kwargs.get('trainable', True))
        if use_layernorm:
            keras.layers.LayerNormalization.__init__(
                self,
                axis=-1,
                epsilon=layernorm_epsilon,
                center=False,
                scale=use_gamma,
                beta_initializer=None,
                gamma_initializer=gamma_initializer,
                beta_regularizer=None,
                gamma_regularizer=gamma_regularizer,
                beta_constraint=None,
                gamma_constraint=gamma_constraint,
                dtype=kwargs.get('dtype'),
                trainable=kwargs.get('trainable', True))
        self.use_layernorm = use_layernorm

    def build(self, input_shape):
        keras.layers.SimpleRNNCell.build(self, input_shape)
        if self.use_layernorm:
            keras.layers.LayerNormalization.build(self, (None, self.units))

    def call(self, inputs, states, training=None):
        """Formulas.

        Notation:
            y_t : Cell output at t (`output`)
            y_{t-1} : Previous cell output at t-1 (`prev_output`)
            x_t : The new input at t (`inputs`)
            W_xh : Weight matrix for inputs x_t (`self.kernel`)
            W_hh : Weights for prev. outputs y_{t-1} (`self.recurrent_kernel`)
            b : Bias term for centering (`self.bias`)
            d1 : Dropout function for x_t (`inputs * dp_mask`)
            d2 : Dropout function for y_{t-1} (`prev_output * rec_dp_mask`)
            ln : Scaling function from layer normalization
            f : Activation function (`self.activation`)

        Case 1:
            Simple RNN, only with bias and activation
              y_t = f(x_t * W_xh + y_{t-1} * W_hh + b)
            or
              net = x_t * W_xh + y_{t-1} * W_hh
              y_t = f(net + b)

        Case 2:
            RNN with, layer normalization (only scaling), bias and activation.
              y_t = f(ln(x_t * W_xh + y_{t-1} * W_hh) + b)
            or
              net = x_t * W_xh + y_{t-1} * W_hh
              y_t = f(ln(net) + b)

            Layer normalization with scaling and centering in one go (see Ba et
            al (2016), page 3, formula 4, https://arxiv.org/abs/1607.06450)
            is the same as layer normalization only with scaling, and
            centering directly afterwards.

        Case 3:
            RNN, with dropout, bias, and activation (no scaling from LN)
              y_t = f(d1(x_t) * W_xh + d2(y_{t-1}) * W_hh + b)
            or
              net = d1(x_t) * W_xh + d2(y_{t-1}) * W_hh
              y_t = f(net + b)

        Case 4:
            Everyting is used, i.e. all dropouts, layer normalization
            (only scaling), bias, and activation
              y_t = f(ln(d1(x_t) * W_xh + d2(y_{t-1}) * W_hh) + b)
            or
              net = d1(x_t) * W_xh + d2(y_{t-1}) * W_hh
              y_t = f(ln(net) + b)
        """
        prev_output = states[0]
        dp_mask = self.get_dropout_mask_for_cell(inputs, training)
        rec_dp_mask = self.get_recurrent_dropout_mask_for_cell(
            prev_output, training)

        if dp_mask is not None:
            h = keras.backend.dot(inputs * dp_mask, self.kernel)
        else:
            h = keras.backend.dot(inputs, self.kernel)

        # don't add bias to "h" here
        # add bias after scaling with layer normalization to "output"

        if rec_dp_mask is not None:
            prev_output = prev_output * rec_dp_mask
        output = h + keras.backend.dot(prev_output,
                                       self.recurrent_kernel)  # "net"

        if self.use_layernorm:
            output = keras.layers.LayerNormalization.call(self, output)

        if self.bias is not None:
            output = keras.backend.bias_add(output, self.bias)

        if self.activation is not None:
            output = self.activation(output)

        return output, [output]

    # use SimpleRNNCell's get_initial_state method

    def get_config(self):
        config = {'use_layernorm': self.use_layernorm}
        cell_config = keras.layers.SimpleRNNCell.get_config(self)
        del cell_config['name']
        for key in ('axis', 'center', 'beta_constraint',
                    'beta_initializer', 'beta_regularizer'):
            del cell_config[key]
        cell_config['layernorm_epsilon'] = cell_config.pop("epsilon")
        cell_config['use_gamma'] = cell_config.pop("scale")
        return {**config, **cell_config}


class LayernormSimpleRNN(keras.layers.SimpleRNN):
    """Fully-connected RNN with Layer Normalization.

    Motivation:
    - Drop-In Replacement for keras.layers.SimpleRNN
    - demonstrate how to add keras.layers.LayerNormalization
       to all RNNs by introducing the `use_layernorm` argument

    References:
    [1] Ba, Jimmy Lei, Jamie Ryan Kiros, and Geoffrey E. Hinton.
        "Layer Normalization." ArXiv:1607.06450 [Cs, Stat],
        July 21, 2016. http://arxiv.org/abs/1607.06450

    Arguments:
      units: Positive integer, dimensionality of the output space.
      activation: Activation function to use.
        Default: hyperbolic tangent (`tanh`).
        If you pass None, no activation is applied
        (ie. "linear" activation: `a(x) = x`).
      use_bias: Boolean, (default `True`), whether the layer uses a bias
        vector.
      use_layernorm: Boolean, (default `True`), whether to apply layer
        normalization (scaling only).
      use_gamma: Boolean (default: True), whether to use gamma weights in
        layer normalization.
      layernorm_epsilon: Float, (default `1e-5`), Small float added to variance
        to avoid dividing by zero.
      kernel_initializer: Initializer for the `kernel` weights matrix,
        used for the linear transformation of the inputs. Default:
        `glorot_uniform`.
      recurrent_initializer: Initializer for the `recurrent_kernel`
        weights matrix, used for the linear transformation of the recurrent
        state. Default: `orthogonal`.
      bias_initializer: Initializer for the bias vector (`use_bias=True`) or
         for the beta vector in layer normalization (`use_layernorm=True`).
         Default: `zeros`.
      gamma_initializer: Initializer for the gamma vector of the layer
         normalization layer (`use_layernorm=True`). Default: `ones`.
      kernel_regularizer: Regularizer function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_regularizer: Regularizer function applied to the
        `recurrent_kernel` weights matrix. Default: `None`.
      bias_regularizer: Regularizer function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_regularizer: Regularizer function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      activity_regularizer: Regularizer function applied to the output of the
        layer (its "activation"). Default: `None`.
      kernel_constraint: Constraint function applied to the `kernel` weights
        matrix. Default: `None`.
      recurrent_constraint: Constraint function applied to the
        `recurrent_kernel` weights matrix.  Default: `None`.
      bias_constraint: Constraint function applied to the bias vector
         (`use_bias=True`) or for the beta vector of the layer normalization
         layer (`use_layernorm=True`). Default: `None`.
      gamma_constraint: Constraint function applied to the gamma vector
         of the layer normalization layer (`use_layernorm=True`).
         Default: `None`.
      dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the
        inputs. Default: 0.
      recurrent_dropout: Float between 0 and 1.
        Fraction of the units to drop for the linear transformation of the
        recurrent state. Default: 0.
      return_sequences: Boolean. Whether to return the last output
        in the output sequence, or the full sequence. Default: `False`.
      return_state: Boolean. Whether to return the last state
        in addition to the output. Default: `False`
      go_backwards: Boolean (default False).
        If True, process the input sequence backwards and return the
        reversed sequence.
      stateful: Boolean (default False). If True, the last state
        for each sample at index i in a batch will be used as initial
        state for the sample of index i in the following batch.
      unroll: Boolean (default False).
        If True, the network will be unrolled,
        else a symbolic loop will be used.
        Unrolling can speed-up a RNN,
        although it tends to be more memory-intensive.
        Unrolling is only suitable for short sequences.

    Call arguments:
      inputs: A 3D tensor, with shape `[batch, timesteps, feature]`.
      mask: Binary tensor of shape `[batch, timesteps]` indicating whether
        a given timestep should be masked.
      training: Python boolean indicating whether the layer should behave in
        training mode or in inference mode. This argument is passed to the cell
        when calling it. This is only relevant if `dropout` or
        `recurrent_dropout` is used.
      initial_state: List of initial state tensors to be passed to the first
        call of the cell.

    Examples:

    ```python
    import numpy as np
    from keras_layernorm_rnn import LayernormSimpleRNN

    inputs = np.random.random([32, 10, 8]).astype(np.float32)
    model = LayernormSimpleRNN(4)

    output = model(inputs)  # The output has shape `[32, 4]`.

    model = LayernormSimpleRNN(
        4, return_sequences=True, return_state=True)

    # whole_sequence_output has shape `[32, 10, 4]`.
    # final_state has shape `[32, 4]`.
    whole_sequence_output, final_state = model(inputs)
    ```
    """

    def __init__(
            self,
            units,
            activation='tanh',
            use_bias=True,
            use_layernorm=True,
            use_gamma=True,
            layernorm_epsilon=1e-05,
            kernel_initializer='glorot_uniform',
            recurrent_initializer='orthogonal',
            bias_initializer='zeros',
            gamma_initializer='ones',
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
        cell = LayernormSimpleRNNCell(
            units,
            activation=activation,
            use_bias=use_bias,
            use_layernorm=use_layernorm,
            use_gamma=use_gamma,
            layernorm_epsilon=layernorm_epsilon,
            kernel_initializer=kernel_initializer,
            recurrent_initializer=recurrent_initializer,
            bias_initializer=bias_initializer,
            gamma_initializer=gamma_initializer,
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

    # use SimpleRNN's call() method

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
