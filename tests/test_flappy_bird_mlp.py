import numpy as np
import unittest

from models.flappy_bird_mlp import softmax, relu, MLP


def test_softmax():
    # Assert that the output matches what is expected for a given input
    test_array = np.array([[0.2, 0.4, 0.6], [0.4, 0.6, 0.8]])
    expected_output = np.array([
        [0.2693075, 0.32893292, 0.40175958],
        [0.2693075, 0.32893292, 0.40175958],
    ])
    np.testing.assert_allclose(
        actual=softmax(test_array), desired=expected_output, rtol=1e-5
    )


def test_relu():
    # Assert that the output matches what is expected for a given input
    test_array = np.array([[-0.2, 0.4, 0.6], [0.4, -0.6, 0.8]])
    output_array = relu(test_array)
    expected_output_array = np.array([[0., 0.4, 0.6], [0.4, 0., 0.8]])
    np.testing.assert_equal(actual=output_array, desired=expected_output_array)


class MLPTestCase(unittest.TestCase):

    def setUp(self):
        self.seed = 14
        self.mlp = MLP(
            input_dim=8,
            hidden_units=50,
            nbr_classes=2,
            seed=self.seed,
            hidden_layer_activation_func=relu
        )
        self.param_length = (
            (self.mlp.input_dim * self.mlp.hidden_units)
            + self.mlp.hidden_units
            + (self.mlp.hidden_units * self.mlp.output_dim)
            + self.mlp.output_dim
        )

    def test_init(self):
        # Assert that the parameters were initialized correctly
        assert self.mlp.input_dim == 8
        assert self.mlp.hidden_units == 50
        assert self.mlp.output_dim == 2
        np.testing.assert_equal(actual=self.mlp.b1, desired=np.zeros(50))
        np.testing.assert_equal(actual=self.mlp.b2, desired=np.zeros(2))

    def test_get_params(self):
        # Assert that the params returned have the right dimensionality
        test_params = self.mlp.get_params()
        assert len(test_params.shape) == 1
        assert test_params.shape[0] == self.param_length

    def test_set_params(self):
        # Assert that the params can be set
        np.random.seed(self.seed)
        test_params = np.random.randn(self.param_length,)
        self.mlp.set_params(params=test_params)
        output = self.mlp.get_params()
        np.testing.assert_allclose(actual=output, desired=test_params, rtol=1e-5)

    def test_sample_action(self):
        # Assert that the sample action returns an integer index
        np.random.seed(self.seed)
        test_array = np.random.randn(8, )
        test_action = self.mlp.sample_action(x=test_array)
        assert isinstance(test_action, np.int64)
