import os
import cv2
import glob
import json
import tqdm
import pickle
import numpy as np
import shutil
from scipy.spatial.transform import Slerp, Rotation

import h5py
import torch
from torch.utils.data import DataLoader, Dataset
import yaml
from .utils import get_rays, get_event_rays
from utils.pose_utils import *
from utils.plot_utils import *
from utils.event_utils import *

import hdf5plugin

# NeRF dataset
import json
import matplotlib
matplotlib.use('Agg')



#####################
# General Helpers
#####################

def read_K(path):
    assert os.path.exists(path)
    with open(path) as f:
        calib = yaml.safe_load(f)
    return calib


#####################
# Dataset Class Definitions
#####################
from .NGPDataset import NGPDataset
from .NeRFDataset import NeRFDataset
from .EventNeRFDataset import EventNeRFDataset

