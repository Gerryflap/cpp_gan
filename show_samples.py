# The training script saves samples.pt every time it prints stats. Use this script to show the samples

import matplotlib.pyplot as plt
import torch
from torchvision.utils import make_grid
import numpy as np

samples = torch.load("samples.pt")
samples = list(samples.parameters())[0]
grid = make_grid(samples)
plt.imshow(np.moveaxis(grid.detach().cpu().numpy(), 0, 2))
plt.show()
