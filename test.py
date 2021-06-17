'''"""
@author: Viet Nguyen <nhviet1009@gmail.com>
"""
import argparse
import torch
import cv2
from src.tetris import Tetris


def get_args():
    parser = argparse.ArgumentParser(
        """Implementation of Deep Q Network to play Tetris""")

    parser.add_argument("--width", type=int, default=10, help="The common width for all images")
    parser.add_argument("--height", type=int, default=20, help="The common height for all images")
    parser.add_argument("--block_size", type=int, default=30, help="Size of a block")
    parser.add_argument("--fps", type=int, default=300, help="frames per second")
    parser.add_argument("--saved_path", type=str, default="trained_models")
    parser.add_argument("--output", type=str, default="output.mp4")

    args = parser.parse_args()
    return args


def test(opt):
    if torch.cuda.is_available():
        torch.cuda.manual_seed(123)
    else:
        torch.manual_seed(123)
    if torch.cuda.is_available():
        model = torch.load("{}/tetris".format(opt.saved_path))
    else:
        model = torch.load("{}/tetris".format(opt.saved_path), map_location=lambda storage, loc: storage)
    model.eval()
    env = Tetris(width=opt.width, height=opt.height, block_size=opt.block_size)
    env.reset()
    if torch.cuda.is_available():
        model.cuda()
    out = cv2.VideoWriter(opt.output, cv2.VideoWriter_fourcc(*"MJPG"), opt.fps,
                          (int(1.5*opt.width*opt.block_size), opt.height*opt.block_size))
    while True:
        next_steps = env.get_next_states()
        next_actions, next_states = zip(*next_steps.items())
        next_states = torch.stack(next_states)
        if torch.cuda.is_available():
            next_states = next_states.cuda()
        predictions = model(next_states)[:, 0]
        index = torch.argmax(predictions).item()
        action = next_actions[index]
        _, done = env.step(action, render=True, video=out)

        if done:
            out.release()
            break
        


if __name__ == "__main__":
    opt = get_args()
    test(opt)'''

from numpy.core.fromnumeric import size
from numpy.lib.npyio import save
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from glob import glob
import time
from skimage import transform
from skimage.color import rgb2gray
import gym
from gym_sokoban.envs.sokoban_env import SokobanEnv
import matplotlib.pyplot as plt

def preprocess_frame(frame):
    gray = rgb2gray(frame)  # 轉灰階
    normalized_frame = gray/255.0  # normalization
    resized_img = transform.resize(normalized_frame, [84, 84])
    return resized_img

env = SokobanEnv()
while 1:
    env.reset()
    img = env.get_image('human', 1)
    state, reward, done, info = env.step(1)
    print(state, type(state), state.shape)
    plt.subplot(1,2,1)
    plt.imshow(state)
    plt.subplot(1,2,2)
    plt.imshow(preprocess_frame(state))
    plt.show()
    input('Next...')