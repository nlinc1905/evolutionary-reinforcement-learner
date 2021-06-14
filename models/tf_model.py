import numpy as np
import tensorflow.keras.backend as K


class EvolutionaryModel:
    def __init__(self, model):
        """
        Constructs a wrapper for a Tensorflow model.

        :param model: a TF model
        """
        self.model = model
        # Save the shape of the expected input parameter array
        self.expected_input_shape = np.sum([K.count_params(w) for w in self.model.trainable_weights])

    def get_params(self):
        """
        Flattens all parameters of the neural network to a 1D numpy array.  The shape of this
        flattened array will be like:
                 w1                +     b1    +              w2              +     b2
        (input_dim * hidden_units) + hidden_units + (hidden_units * nbr_classes) + nbr_classes
        This is the same as self.expected_input_shape, as defined upon initialization.

        :return: 1D array with dimension equal to the number of trainable parameters
        """
        model_params = np.concatenate([
            np.concatenate([
                layer.get_weights()[0].flatten(),  # weights
                layer.get_weights()[1].flatten()   # biases
            ])
            for layer in self.model.layers if len(layer.get_weights()) > 0
        ])
        return model_params

    def set_params(self, params):
        """
        Updates the networks parameters with the given params array.  The network will learn to represent the
        environment so that the action that has the highest probability of maximal reward for a given
        state can be found.  The parameters describe the state of the environment.

        :param params: A 1D array of parameters that will be reshaped and passed through the neural network
            so that it can learn the weights for each parameter, in association with its reward from the
            given environment.  If none are provided, the weights are randomly initialized.  If they are
            provided, they will be of the shape defined by self.get_params()
        """
        # extract the parameters by layer and store them in list_of_arrays
        list_of_arrays = []
        # ignore the input layer, start with the flatten layer: self.layers[1:]
        list_of_tuples = [layer.output_shape for layer in self.model.layers[1:]]
        for tup_idx, tup in reversed(list(enumerate(list_of_tuples))):
            # Stop looping at tup_idx 1 (it will be approached in reverse order)
            if tup_idx == 0:
                break
            bias = params[-tup[1]:]
            # Remove the bias from the end of the array
            params = params[:-tup[1]]
            # Weights are the product of the previous layer and the current one (using original indices, not reversed)
            weights = params[-(tup[1] * list_of_tuples[tup_idx-1][1]):]
            # Reshape the weights to be n*m where n = nbr units in the previous layer and m = nbr units in this layer
            weights = weights.reshape(int(len(weights)/len(bias)), len(bias))
            # Remove the weights from the end of the array
            params = params[:-(tup[1] * list_of_tuples[tup_idx-1][1])]
            # Concatenate the weights + bias and append them to list
            list_of_arrays.append([weights, bias])
        # Reverse the list_of_arrays to put them in the correct order
        list_of_arrays.reverse()

        # Sense check: there should be a set of weights for every layer, except the input and flatten layers
        assert len(list_of_arrays) == len(self.model.layers[2:])

        # Iterate through the layers and set the weights
        # Ignore the input layer and flatten layer, as they both have 0 parameters
        #   - call self.model.model_summary() to verify
        for layer_idx, layer in enumerate(self.model.layers[2:]):
            layer.set_weights(list_of_arrays[layer_idx])

    def sample_action(self, x):
        """
        Samples an action from the probability space provided by the final softmax layer.
        This function produces an action for a given state of the environment, such that the probability of
        that action yielding a high reward is maximized.

        :param x: input array of shape (nbr_params,) representing an environment state

        :return: np.int64 corresponding with the index of the action from the action map list
        """
        # input is a single state of size (input_dim,)
        # first make it (N, input_dim) to fit ML conventions
        X = np.atleast_2d(x)
        # get list of probabilities, which come out as an array of shape (output_dim,)
        action_probabilities = self.model(X)
        action_probabilities = action_probabilities.numpy()[0]  # the first row
        # Although the evolutionary algorithm explores the parameter space, the line below could be
        #   uncommented to test a probabilistic policy.
        # return np.random.choice(len(action_probabilities), p=action_probabilities)
        return np.argmax(action_probabilities)

    def save_model_weights(self, save_path):
        self.model.save_weights(save_path)
        return print(f"Model weights saved to {save_path}")

    def load_model_weights(self, weight_path):
        self.model.load_weights(weight_path)
        # Ensure that the dimensions for the network match the loaded weights by overwriting them
        assert len(self.get_params()) == self.expected_input_shape
        return print(f"Model weights loaded.")
