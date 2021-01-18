from torch import nn
from generalized_logistic_layer import GeneralizedLogisticLayer
from fully_connected_layer import FullyConnectedLayer


def create_net(input_features, hidden_units, non_linearity, output_size):
    """
    Constructs a network based on the specifications passed as input arguments

    Arguments
    --------
    input_features: (integer) the number of input features
    hidden_units: (list) of length L where L is the number of hidden layers. hidden_units[i] denotes the
                        number of units at hidden layer i + 1 for i  = 0, ..., L - 1
    non_linearity: (list)  of length L. non_linearity[i] contains a string describing the type of non-linearity to use
                           hidden layer i + 1 for i = 0, ... L-1
    output_size: (integer), the number of units in the output layer

    Returns
    -------
    net: (Sequential) the constructed model
    """

    # instantiate a sequential network
    net = nn.Sequential()

    # add the hidden layers
    for i,(h,nl) in enumerate(zip(hidden_units, non_linearity)):

        fc_name = 'fc_'+str(i); nl_name = nl+'_'+str(i)

        if i == 0:
            prev_hidden = input_features
        else:
            prev_hidden = hidden_units[i-1]

        net.add_module(fc_name,  FullyConnectedLayer(prev_hidden,h))
        net.add_module(nl_name, GeneralizedLogisticLayer(nl))

    # add output layer
    net.add_module('predictions', FullyConnectedLayer(h, output_size))

    return net