import torch
from torch.utils.data import TensorDataset


def create_dataset(data_path, output_path=None, contrast_normalization=False, whiten=False):
    """
    Reads and optionally preprocesses the data.

    Arguments
    --------
    data_path: (String), the path to the file containing the data
    output_path: (String), the name of the file to save the preprocessed data to (optional)
    contrast_normalization: (boolean), flags whether or not to normalize the data (optional). Default (False)
    whiten: (boolean), flags whether or not to whiten the data (optional). Default (False)

    Returns
    ------
    train_ds: (TensorDataset, the examples (inputs and labels) in the training set
    val_ds: (TensorDataset), the examples (inputs and labels) in the validation set
    """
    # read the data and extract the various sets
    data = torch.load(data_path)

    data_tr = data['data_tr']; data_te = data['data_te']
    sets_tr = data['sets_tr']; label_tr = data['label_tr']


    # apply the necessary preprocessing as described in the assignment handout.
    # You must zero-center both the training and test data
    if data_path == "image_categorization_dataset.pt":

        # do mean centering here

        # 1: calculate pixel means
        mean_tr_pixels = data_tr.mean(axis=0)

        # 2: subtract the per-pixel mean across all images in data_tr
        data_tr = data_tr - mean_tr_pixels

        # 3: subtract per-pixel mean pre-centered data_tr images in data_te images
        data_te = data_te - mean_tr_pixels


        # %%% DO NOT EDIT BELOW %%%% #
        if contrast_normalization:
            image_std = torch.std(data_tr[sets_tr == 1], unbiased=True)
            image_std[image_std == 0] = 1
            data_tr = data_tr / image_std
            data_te = data_te / image_std

        if whiten:

            # ADDED PER PIAZZA INSTRUCTIONS #######
            data_tr = data_tr.contiguous()
            data_te = data_te.contiguous()
            #######################################

            examples, rows, cols, channels = data_tr.size()
            data_tr = data_tr.view(examples, -1)
            W = torch.matmul(data_tr[sets_tr == 1].T, data_tr[sets_tr == 1]) / examples
            E, V = torch.eig(W, eigenvectors=True)
            en = torch.sqrt(torch.mean(E[:, 0]).squeeze())
            M = torch.diag(en / torch.max(torch.sqrt(E[:, 0].squeeze()), torch.tensor([10.0])))

            data_tr = torch.matmul(data_tr.mm(V.T), M.mm(V))
            data_tr = data_tr.view(examples, rows, cols, channels)

            data_te = data_te.view(-1, rows * cols * channels)
            data_te = torch.matmul(data_te.mm(V.T), M.mm(V))
            data_te = data_te.view(-1, rows, cols, channels)

        preprocessed_data = {"data_tr": data_tr, "data_te": data_te, "sets_tr": sets_tr, "label_tr": label_tr}

        if output_path:
            torch.save(preprocessed_data, output_path)

    train_ds = TensorDataset(data_tr[sets_tr == 1], label_tr[sets_tr == 1])
    val_ds = TensorDataset(data_tr[sets_tr == 2], label_tr[sets_tr == 2])

    return train_ds, val_ds

