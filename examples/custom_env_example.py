import matplotlib.pyplot as plt
import numpy as np
import torch
from gfn.gflownet import TBGFlowNet  # TODO: Extend to SubTBGFlowNet
from gfn.gym.line import Line
from gfn.modules import DiscretePolicyEstimator, GFNModule
from gfn.samplers import Sampler
from gfn.states import States
from gfn.utils.common import set_seed
from gfn.utils.modules import MLP  # is a simple multi-layer perceptron (MLP)
from torch.distributions import Distribution, Normal  # TODO: extend to Beta
from torch.distributions.independent import Independent
from tqdm import tqdm, trange

from common.env import DistrictEnv

if __name__ == "__main__":
    # 1 - We define the environment.
    environment = DistrictEnv(json_file="data/IA_raw_data.json", device_str="cpu")    # Grid of size 8x8x8x8
    
    print(environment.reset(32).shape)  # (32, 64, 9)