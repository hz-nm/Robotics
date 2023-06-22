# ! 
# ! START OVER!!!
"""
Author: Ameer H.
Description: Starting over.
             Everything will be custom this time.
! Example Action: np.array([0.44, 0.55)]....array([-0.8505605, -0.7363309], dtype=float32)]
! Example State: [ 0.01136427  1.4182303   0.574197    0.16522941 -0.01310821 -0.13048422 0.          0.        ]
              !  Channels: 8 and Channel Type class<'int'>
! Put the action space overe here
! Example Action -> np.array([main, lateral])
! if main < 0 - main thruster is turned off
! 0 <= main <= 1.
! lateral - Has two possibilities,
! if -1.0 < lateral < -0.5 -> Left booster will fire
! if 0.5 < lateral < 1 -> Right booster will fire

"""

import re
from tkinter import Frame
import torch
from torch import nn
from torchvision import transforms as T

from PIL import Image
import numpy as np
from pathlib import Path
from collections import deque

import random, datetime, os, copy
import time
import matplotlib.pyplot as plt
import gym
from gym.spaces import Box
from gym.wrappers import FrameStack

# * Let's initialize the LunarLander Environment
if gym.__version__ < '0.26':
    env = gym.make("LunarLander-v2", new_step_api=True, continuous=True)
    # env = gym.make("LunarLander-v2", render_mode="human", new_step_api=True, continuous=True)
else:
    env = gym.make("LunarLander-v2", render_mode="rgb", apply_api_compatibility=True, continuous=True)


# ? Let's look at an example state
action_X = env.action_space.sample()
print(f"Action Space: {action_X.shape}")

env.reset()

