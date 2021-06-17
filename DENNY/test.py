import numpy as np
from skimage import transform
from skimage.color import rgb2gray
import gym
import gym_sokoban
import torch

'''def preprocess_frame(frame):
    gray = rgb2gray(frame) # 轉灰階
    normalized_frame = gray/255.0 # normalization

    return normalized_frame

env = gym.make('Sokoban-v0')
while 1:
    action = np.random.randint(0, 9)
    next_state, reward, done, info = env.step(action, 'tiny_rgb_array')
    preprocess = preprocess_frame(next_state)
    print(preprocess.shape)
    env.render('tiny_human',scale = 20)
    input('test...')'''

from numpy import random


x = torch.rand(80,80)
xx = x.view(-1)
xxx = xx.view(80,80)
print(torch.equal(x,xxx))