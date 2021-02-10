from torch import nn


def cnn_categorization_base(netspec_opts):
    """
    Constructs a network for the base categorization model.

    Arguments
    --------
    netspec_opts: (dictionary), the network's architecture. It has the keys
                 'kernel_size', 'num_filters', 'stride', and 'layer_type'.
                 Each key holds a list containing the values for the
                corresponding parameter for each layer.
    Returns
    ------
     net: (nn.Sequential), the base categorization model
    """
    # instantiate an instance of nn.Sequential
    net = nn.Sequential()

    # add layers as specified in netspec_opts to the network

     # for each layer in the network
    for idx,(layer,kern,filt,stride) in enumerate(zip(netspec_opts['layer_type'],
                                                    netspec_opts['kernel_size'],
                                                    netspec_opts['num_filters'],
                                                    netspec_opts['stride'])):
        # layer name (1-indexed)
        if idx <= 2:
            name = layer + '_1'
        elif idx > 2 and idx < 6:
            name = layer + '_2'
        else:
            name = layer + '_3'


        ############# ADD LAYER ################

        # convolutional layer
        if layer == 'conv':

            # last layer name
            if idx == 10:
                name = 'pred'

            # number of input filters
            elif idx == 0:
                in_channels = 3
            else:
                in_channels = netspec_opts['num_filters'][idx - 3]

            padding = (netspec_opts['kernel_size'][idx] - 1) / 2
            net.add_module(name, nn.Conv2d(in_channels, filt, kern, stride, int(padding)))
            print(name)

        # batch norm layer
        elif layer == 'bn'  :
            in_channels = netspec_opts['num_filters'][idx - 1]
            net.add_module(name, nn.BatchNorm2d(in_channels))
            print(name)

        # relu layer
        elif layer == 'relu':
            in_channels = netspec_opts['num_filters'][idx - 2]
            net.add_module(name, nn.ReLU())
            print(name)

        # pooling layer
        else:
            # padding = (netspec_opts['kernel_size'][idx] - 1) / 2
            net.add_module(name, nn.AvgPool2d(kern, stride, 0))
            # net.add_module(name, nn.AvgPool2d(kern, stride, int(padding)))
            print(name)

        ########################################

    return net

