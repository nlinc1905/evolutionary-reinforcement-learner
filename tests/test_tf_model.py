import numpy as np
import unittest

from tensorflow.keras.layers import Dense, Input, Flatten
from tensorflow.keras.models import Model

from models.tf_model import EvolutionaryModel
from environments.flappy_bird_env import FlappyBirdEnv


def build_tf_mlp(env):
    inputs = Input(shape=len(env.reset()))
    x = Flatten()(inputs)
    x = Dense(50, activation="relu")(x)
    outputs = Dense(len(env.action_map), activation="softmax")(x)
    return Model(inputs=inputs, outputs=outputs)


class EvolutionaryModelTestCase(unittest.TestCase):

    def setUp(self):
        self.seed = 14
        self.env = FlappyBirdEnv()
        self.model = EvolutionaryModel(
            model=build_tf_mlp(env=self.env)
        )

    def test_get_params(self):
        # Assert that the params returned have the right dimensionality
        test_params = self.model.get_params()
        assert len(test_params.shape) == 1
        assert test_params.shape[0] == self.model.expected_input_shape

    def test_set_params(self):
        # Assert that the params can be set
        np.random.seed(self.seed)
        test_params = np.random.randn(self.model.expected_input_shape,)
        self.model.set_params(params=test_params)
        output = self.model.get_params()
        np.testing.assert_allclose(actual=output, desired=test_params, rtol=1e-5)

    def test_sample_action(self):
        # Assert that the sample action returns an integer index
        np.random.seed(self.seed)
        test_array = np.random.randn(len(self.env.reset()),)
        test_action = self.model.sample_action(x=test_array)
        assert isinstance(test_action, np.int64)

    def test_score_architecture(self):
        # Assert that the score is a float
        score = self.model.score_architecture(nbr_obs=20)
        assert isinstance(score, np.float64)
