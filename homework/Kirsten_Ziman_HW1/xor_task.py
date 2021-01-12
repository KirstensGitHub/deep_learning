from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE

# Specify the lead_data arguments
# data_path
# mean_subtraction
# normalization

xor_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# in_features
# out_size
# hidden_units
# non_linearity

# create a network base on the architecture
# net

# specify the training opts
# train_opts

# train  and save the model
train(net, xor_dataset, train_opts)
save(net, 'xor_solution.pt')
