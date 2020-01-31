from ray.rllib.utils import try_import_tf
from ray.rllib.models.tf.tf_modelv2 import TFModelV2


tf = try_import_tf()


class CartPoleV1Convnet(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name, **kwargs):
        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name, **kwargs)
        self.inputs = tf.keras.layers.Input(
            shape=obs_space.shape, name='observations')
        layer_conv_1 = tf.keras.layers.Conv2D(
            16, (8, 8), strides=(4, 4),
            padding='same',
            activation='relu',
            # kernel_initializer=normc_initializer(1.0),
            input_shape=obs_space.shape,
            name='layer_conv_1',
        )(self.inputs)
        layer_conv_2 = tf.keras.layers.Conv2D(
            32, (4, 4), strides=(2, 2),
            padding='same',
            activation='relu',
            input_shape=obs_space.shape,
            name='layer_conv_2',
        )(layer_conv_1)
        layer_conv_3 = tf.keras.layers.Conv2D(
            512, (11, 11), strides=(1, 1),
            padding='same',
            activation='relu',
            input_shape=obs_space.shape,
            name='layer_conv_3',
        )(layer_conv_2)
        layer_flatten = tf.keras.layers.Flatten()(layer_conv_3)
        layer_out = tf.keras.layers.Dense(
            num_outputs,
            name='layer_out',
            activation='relu')(layer_flatten)
        self.base_model = tf.keras.Model(self.inputs, layer_out)
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_out = self.base_model(input_dict['obs'])
        return model_out, state
