from ray.rllib.models.tf.tf_modelv2 import TFModelV2
from ray.rllib.models.tf.tf_action_dist import (
    ActionDistribution,
    Categorical,
)
from ray.rllib.policy.policy import TupleActions
from ray.rllib.utils import try_import_tf

tf = try_import_tf()


class CartPolesActionDist(ActionDistribution):
    '''This implementation is borrowed from the example
    autoregressive_action dist.py, so it does not work right for now.
    The ActionDistribution class works only great with PG, which
    only use the sample() method.
    '''
    @staticmethod
    def required_model_output_shape(action_space, model_config):
        return len(action_space) * action_space[-1].n

    def sample(self):
        """Draw a sample from the action distribution."""
        a_dists = self._actions_distribution()
        a_samples = list(a_dist.sample() for a_dist in a_dists)
        self._action_logp = sum(
            a_dist.logp(a_sample)
            for (a_dist, a_sample) in zip(a_dists, a_samples))
        return TupleActions(a_samples)

    def sampled_action_logp(self):
        """Return the log probability of the last sampled actions."""
        # return tf.math.exp(self._action_logp)
        return self._action_logp

    def logp(self, actions):
        """The log-likelihood of the action distribution."""
        a_dists = self._actions_distribution()
        return sum(
            a_dists[i].logp(actions[:, i]) for i in range(len(a_dists)))

    # def kl(self, other):
    #     a_dists = self._actions_distribution()
    #     o_dists = other._actions_distribution()
    #     return sum(
    #         a_dist.kl(o_dist)
    #         for (a_dist, o_dist) in zip(a_dists, o_dists))

    # def entropy(self):
    #     a_dists = self._actions_distribution()
    #     return sum(
    #         a_dist.entropy()
    #         for a_dist in a_dists)

    def _actions_distribution(self):
        a_dists = list(Categorical(logit) for logit in self.model.logits)
        return a_dists


class CartPolesModel(TFModelV2):

    def __init__(self, obs_space, action_space, num_outputs,
                 model_config, name):
        super().__init__(obs_space, action_space, num_outputs,
                         model_config, name)

        env_input = tf.keras.layers.Input(
            shape=(4,), name='env_inputs')
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            name='layer_1')(env_input)
        x = tf.keras.layers.Dense(
            128,
            activation='relu',
            name='layer_2')(x)
        env_output = tf.keras.layers.Dense(
            2,
            activation=None,
            name='env_outputs')(x)
        self.env_model = tf.keras.models.Model(
            inputs=env_input,
            outputs=env_output)

        input_1 = tf.keras.layers.Input(
            shape=(4,), name='inputs_1')
        input_2 = tf.keras.layers.Input(
            shape=(4,), name='inputs_2')
        input_3 = tf.keras.layers.Input(
            shape=(4,), name='inputs_3')
        output_1 = self.env_model(input_1)
        output_2 = self.env_model(input_2)
        output_3 = self.env_model(input_3)
        outputs = tf.keras.layers.concatenate([
            output_1,
            output_2,
            output_3,
        ])

        value_out = tf.keras.layers.Dense(
            1,
            activation=None,
            name='value_out')(outputs)

        self.base_model = tf.keras.models.Model(
            inputs=[input_1, input_2, input_3],
            outputs=[outputs, value_out, output_1, output_2, output_3]
        )
        self.base_model.summary()
        self.register_variables(self.base_model.variables)

    def forward(self, input_dict, state, seq_lens):
        model_outputs, self._value_out, *self.logits = self.base_model(
            input_dict['obs'])
        return model_outputs, state

    def value_function(self):
        return tf.reshape(self._value_out, [-1])


class CartPolesStackedModel(CartPolesModel):

    def forward(self, input_dict, state, seq_lens):
        inputs = [input_dict['obs'][:, j, :] for j in range(
            input_dict['obs'].shape[1])]
        model_outputs, self._value_out, *self.logits = self.base_model(
            inputs)
        return model_outputs, state
