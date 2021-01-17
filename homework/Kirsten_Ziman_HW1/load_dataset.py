import torch
from torch.utils.data import TensorDataset


def load_dataset(dataset_path, mean_subtraction, normalization):
    """
    Reads the train and validation data

    Arguments
    ---------
    dataset_path: (string) representing the file path of the dataset
    mean_subtraction: (boolean) specifies whether to do mean centering or not. Default: False
    normalization: (boolean) specifies whether to normalizes the data or not. Default: False

    Returns
    -------
    train_ds (TensorDataset): The features and their corresponding labels bundled as a dataset
    """

    ########################################

    # Load the dataset and extract the features and the labels
    data     = torch.load('xor_dataset.pt')
    features = data['features']
    labels   = data['labels']

    torch.mean(features, 0)

    # Do mean_subtraction if it is enabled
    if mean_subtraction:
        features = features - torch.mean(features, 0)

    # do normalization if it is enabled
    if normalization:
        features / torch.std(features, dim=0)

    # create tensor dataset train_ds

    # xor_dataset.pt #######################

    ########################################


    return train_ds
