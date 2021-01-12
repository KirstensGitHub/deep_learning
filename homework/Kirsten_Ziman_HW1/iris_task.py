from create_net import create_net
from train import train
from load_dataset import load_dataset
from torch import random, save

random.manual_seed(0)
# %%% DO NOT EDIT ABOVE %%%

# Specify the lead_data arguments
# data_path
# mean_subtraction
# normalization

iris_dataset = load_dataset(data_path, mean_subtraction, normalization)

# specify the network architecture
# in_features
# out_size
# hidden_units
# non_linearity

# create a network base on the architecture
# net

# specify the training opts
# train_opts

# Train and save the trained model
train(net, iris_dataset, train_opts)
save(net, "iris_solution.pt")
