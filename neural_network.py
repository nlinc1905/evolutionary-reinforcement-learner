import numpy as np

# hyperparameters
D = len(env.reset()) * HISTORY_LENGTH
M = 50
K = 2


def softmax(a):
    """
    Implements a softmax activation function.
    :param a:
    :return:
    """
    # subtract the max from each column to avoid differentiating large numbers
    c = np.max(a, axis=1, keepdims=True)
    e = np.exp(a - c)
    # divide by the sum of the exponentiated values to give the model's output probabilities
    return e / e.sum(axis=-1, keepdims=True)


def relu(x):
    """
    Implements a rectified linear (ReLU) activation function.
    :param x:
    :return:
    """
    return x * (x > 0)


# this should match es_mnist.py pretty closely, so look there for comments
class MLP:
    def __init__(self, input_dim, hidden_units, nbr_classes, activation_func=relu):
        self.input_dim = input_dim
        self.hidden_units = hidden_units
        self.output_dim = nbr_classes
        self.activation_func = activation_func

    def init(self):
        """
        A separate init method from the constructor to allow the creation of a network with a
        given set of weights, rather than random weights.
        :return:
        """
        input_dim, hidden_units, output_dim = self.input_dim, self.hidden_units, self.output_dim
        self.W1 = np.random.randn(input_dim, hidden_units) / np.sqrt(input_dim)
        # self.W1 = np.zeros((input_dim, M))
        self.b1 = np.zeros(hidden_units)
        self.W2 = np.random.randn(hidden_units, output_dim) / np.sqrt(hidden_units)
        # self.W2 = np.zeros((M, K))
        self.b2 = np.zeros(K)

    def forward(self, X):
        """
        Implements feedforward action.
        :param X:
        :return:
        """
        Z = self.activation_func(X.dot(self.W1) + self.b1)
        return softmax(Z.dot(self.W2) + self.b2)

    def sample_action(self, x):
        # assume input is a single state of size (input_dim,)
        # first make it (N, input_dim) to fit ML conventions
        X = np.atleast_2d(x)
        # get list of probabilities of size 1*k, from which we only want the first row
        P = self.forward(X)
        p = P[0]  # the first row
        # no need to further explore parameter space - evolutionary algo does this for us
        #   however, we could use this line to explore a probabilistic policy
        # return np.random.choice(len(p), p=p)
        return np.argmax(p)

    def get_params(self):
        """
        Flattens all parameters of the neural network to a 1D numpy array.
        """
        return np.concatenate([self.W1.flatten(), self.b1, self.W2.flatten(), self.b2])

    def get_params_dict(self):
        return {
            'W1': self.W1,
            'b1': self.b1,
            'W2': self.W2,
            'b2': self.b2,
        }

    def set_params(self, params):
        """
        Takes a 1D array of parameters, and transforms them back into the shape of the neural network,
        then assigns them back to the weights.  This reverses the get_params function.
        """
        # params is a flat list
        # unflatten into individual weights
        input_dim, hidden_units, output_dim = self.input_dim, self.hidden_units, self.output_dim
        self.W1 = params[:input_dim * hidden_units].reshape(input_dim, hidden_units)
        self.b1 = params[input_dim * hidden_units:input_dim * hidden_units + hidden_units]
        self.W2 = params[input_dim * hidden_units + hidden_units:input_dim * hidden_units + hidden_units + hidden_units * output_dim].reshape(hidden_units, output_dim)
        self.b2 = params[-output_dim:]

