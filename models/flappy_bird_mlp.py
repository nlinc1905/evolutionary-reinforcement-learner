import numpy as np

'''
# hyperparameters
D = len(env.reset()) * HISTORY_LENGTH
M = 50
K = 2
'''

def softmax(a):
    """
    Implements a softmax activation function.

    :param a: 2D numpy array equal to
        (the dot product of the hidden input and the output layer weights) + output layer bias

    :return: 2D numpy array of the class probabilities (the classes are actions to take)
    """
    # Scale the values from (min-max:0) by subtracting the max from all of them.
    # Then exponentiate the result - this prevents the differentiation of large numbers.
    c = np.max(a, axis=1, keepdims=True)
    e = np.exp(a - c)
    # Divide by the sum of the exponentiated values to give the model's output probabilities
    return e / e.sum(axis=-1, keepdims=True)


def relu(x):
    """
    Implements a rectified linear (ReLU) activation function.

    :param x: 2D numpy array equal to (the dot product of the input and weights) + bias

    :return: 2D numpy array that is zeroed out where x <= 0 and equal to the original value if x > 0
    """
    return x * (x > 0)


class MLP:
    def __init__(self, input_dim, hidden_units, nbr_classes, hidden_layer_activation_func=relu, params=None):
        """
        Constructs a multilayer perceptron (MLP) that maps parameters of dimension input_dim (a numpy
        array of shape (8,) for Flappy Bird) to an action.  The network will learn to represent the
        environment so that the action that has the highest probability of maximal reward for a given
        state can be found.  The parameters passed as input describe the state of the environment.

        The network has 1 hidden layer with weights w1 and biases b1, and an output layer with
        weights w2 and biases b2.
        w1 shape = (input_dim, hidden_units)
        b1 shape = (input_dim,)
        w2 shape = (hidden_units, nbr_classes)
        b2 shape = (nbr_classes,)

        There is no need to implement backpropagation or optimization here, because the evolutionary
        algorithm handles that.  The evolutionary algorithm will explore the parameter space to find
        the weights and biases that maximize reward, which are used by this neural network to find
        the action.

        :param input_dim: int - the number of parameters from the environment
        :param hidden_units: int - number of units in hidden layer
        :param nbr_classes: int - number of actions that can be taken (2 for Flappy Bird)
        :param hidden_layer_activation_func: function - defines activation function to use for hidden units
        :param params: A 1D array of parameters that will be reshaped and passed through the neural network
            so that it can learn the weights for each parameter, in association with its reward from the
            given environment.  If none are provided, the weights are randomly initialized.  If they are
            provided, they will be of the shape defined by self.get_params()
        """
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = nbr_classes
        self.hidden_layer_activation_func = hidden_layer_activation_func

        # If params are provided, they are given as a flattened array, so they need to be unflattened
        #   and reshaped for the network
        if params is not None:
            self.w1 = params[:self.input_dim * self.hidden_units].reshape(self.input_dim, self.hidden_units)
            self.b1 = params[
                self.input_dim * self.hidden_units:
                self.input_dim * self.hidden_units + self.hidden_units
            ]
            self.w2 = params[
                self.input_dim * self.hidden_units + self.hidden_units:
                self.input_dim * self.hidden_units + self.hidden_units + self.hidden_units * self.output_dim
            ].reshape(self.hidden_units, self.output_dim)
            self.b2 = params[-self.output_dim:]

        # Otherwise randomly initialize the weights
        else:
            self.w1 = np.random.randn(self.input_dim, self.hidden_units) / np.sqrt(self.input_dim)
            self.b1 = np.zeros(self.hidden_units)
            self.w2 = np.random.randn(self.hidden_units, self.output_dim) / np.sqrt(self.hidden_units)
            self.b2 = np.zeros(self.output_dim)

    def _feed_forward(self, input_tensor):
        """
        Implements feed forward action through the network.

        :param input_tensor: numpy array of shape (N, input_dim), where N = 1 for Flappy Bird

        :return: Numpy array with the output probabilities for each action, from softmax activation
        """
        hidden_tensor = self.hidden_layer_activation_func(input_tensor.dot(self.w1) + self.b1)
        return softmax(hidden_tensor.dot(self.w2) + self.b2)

    def sample_action(self, x):
        """
        Samples an action from the probability space produced by feed_forward (the softmax probabilities).
        This function produces an action for a given state of the environment, such that the probability of
        that action yielding a high reward is maximized.

        :param x: input array of shape (nbr_params,) representing an environment state

        :return: np.int64 corresponding with the index of the action from the action map list
        """
        # input is a single state of size (input_dim,)
        # first make it (N, input_dim) to fit ML conventions
        X = np.atleast_2d(x)
        # get list of probabilities, which come out as an array of shape (output_dim,)
        action_probabilities = self._feed_forward(X)
        action_probabilities = action_probabilities[0]  # the first row
        # Although the evolutionary algorithm explores the parameter space, the line below could be
        #   uncommented to test a probabilistic policy.
        # return np.random.choice(len(action_probabilities), p=action_probabilities)
        return np.argmax(action_probabilities)

    def get_params(self):
        """
        Flattens all parameters of the neural network to a 1D numpy array.  The shape of this
        flattened array will be:
                 w1                +     b1    +              w2              +     b2
        (input_dim * hidden_units) + hidden_units + (hidden_units * nbr_classes) + nbr_classes

        :return: 1D array with dimension defined by the equation above
        """
        return np.concatenate([self.w1.flatten(), self.b1, self.w2.flatten(), self.b2])

    def save_model_weights(self, save_path='data/evolutionary_strategy_mlp_weights.npz'):
        weight_dict = {
            'w1': self.w1,
            'b1': self.b1,
            'w2': self.w2,
            'b2': self.b2,
        }
        np.savez(save_path, **weight_dict)
        return print(f"Model weights saved to {save_path}")

    def load_model_weights(self, weight_path='data/evolutionary_strategy_mlp_weights.npz'):
        w = np.load(weight_path)
        self.w1, self.b1, self.w2, self.b2 = w['w1'], w['b1'], w['w2'], w['b2']
        # Ensure that the dimensions for the network match the loaded weights by overwriting them
        self.input_dim, self.hidden_units = w['w1'].shape
        self.output_dim = len(w['b2'])
        return print(f"Model weights loaded with shapes: {w['w1'].shape} and {w['w2'].shape}")
