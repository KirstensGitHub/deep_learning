from os.path import exists
from pathlib import Path
from sys import argv

import torch
from torch import load
from torch.nn import Softmax

from cnn_categorization_base import cnn_categorization_base
from cnn_categorization_improved import cnn_categorization_improved


def create_submission(model_type):
    """
    Evaluates a trained model on the test and validation sets and creates an archive containing the evaluation results
    and the model.

    Arguments
    ---------
    model_type: (string) specifies the model {base, improved}
    """

    data_path = "{}_image_categorization_dataset.pt".format(model_type)
    model_path = "{}-model.pt".format(model_type)

    assert exists(data_path), f"The data file {data_path} does not exist"
    assert exists(model_path), f"The trained model {model_path} does not exist"

    dataset = load(data_path)
    # the relevant data sets
    data_val = dataset["data_tr"][dataset["sets_tr"] == 2]
    data_te = dataset["data_te"]

    model_state = load(model_path)
    if model_type == 'base':
        model = cnn_categorization_base(model_state['specs'])
    else:
        model = cnn_categorization_improved(model_state['specs'])

    model.load_state_dict(load(model_state['state']))
    model.eval()
    soft_max = Softmax(dim=1)

    # validation set
    prob_val = soft_max(model(data_val).squeeze())
    assert prob_val.size() == (6400, 16), f"Expected the output of the validation set to be of size (6400, 16) " \
                                          f"but was {prob_val.size()} instead"
    prob_val = torch.max(prob_val, dim=1)[1]

    with Path(f"kaggle_{model_type}_val_submission.csv").open(mode="w") as writer:
        writer.write("Id,Category\n")
        for i in range(len(prob_val)):
            writer.write(f"{i},{prob_val[i]}\n")

    # test set
    prob_test = soft_max(model(data_te).squeeze())
    assert prob_test.size() == (9600, 16), f"Expected the output of the test set to be of size (9600, 16) " \
                                           f"but was {prob_test.size()} instead"
    prob_test = torch.max(prob_test, dim=1)[1]

    with Path(f"kaggle_{model_type}_test_submission.csv").open(mode="w") as writer:
        writer.write("Id,Category\n")
        for i in range(len(prob_test)):
            writer.write(f"{i},{prob_test[i]}\n")


if __name__ == '__main__':

    # change m_type to 'improved' for the improved task
    # You can also pass in an argument from the command line

    m_type = "improved"
    try:
        m_type = argv[1]
    except IndexError:
        print("Setting model type to <base> since no type is given.")
        print("To create an archive for the improved model,"
              " execute <create_submission.py improved> at the command line")
    assert m_type in ("base", "improved"), f"Model type must be either base or improved. Found {m_type} instead"

    create_submission(m_type)
