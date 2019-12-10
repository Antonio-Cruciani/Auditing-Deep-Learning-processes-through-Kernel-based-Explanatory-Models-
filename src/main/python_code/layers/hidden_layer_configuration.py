class HiddenLayerConf:
    """ This class corresponds to the configuration of the hidden layer

    """

    def __init__(self, hl_size, activation_function):
        """

        :param hl_size: the size (i.e., the number of neurons) of the hidden layer
        :param activation_function: the activation function of the neurons (it one of the following strings:
            tanh, sigmoid, relu, softplus, elu
        """
        self.hl_size = hl_size
        self.activation_function = activation_function

    def get_size(self):
        return self.hl_size

    def get_activation(self):
        return self.activation_function