# This code will download the MNIST dataset
# The data files in "raw" still have to be moved
from torchvision.datasets import MNIST
ds = MNIST("./data", download=True)