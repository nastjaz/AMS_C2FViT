import torch

use_cuda = False
#use_cuda = True
device = torch.device("cuda" if use_cuda else "cpu")