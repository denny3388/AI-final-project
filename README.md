# AI-final-project
> 0716081 葉晨 / 0716083 羅子涵

## Introduction

Deep Q-learning 是將強化學習技巧之一的 Q-learning 與深度學習結合的技術，利用遊戲畫面當作 input，並利用 neural network (稱為 Deep Q Network ) 取代原本的 Q-table，可以大大增加 Deep Q-learning 的穩定性及實用性。

此專案的目的是使用 Deep Q Network (DQN) 來 **訓練電腦學習推箱子遊戲 (Sokoban)** 。推箱子遊戲是一款將所有積木推到特定位置的遊戲。我們這次使用的訓練環境是已經搭建好的 gym 環境。我們此次的目標為應用現有的 Deep Q-learning 技術，並且對一些超參數進行調整，希望能讓電腦學習推箱子遊戲的遊玩方式，並拿到盡量高的分數。

## Sokoban introduction

![](/docs/Sokoban.png)

## Project overview

```
├── gym_sokoban             // Gym environment of Sokoban
├── dqn.py                  // Main function
├── reward.txt              // Record the rewards of each episode
├── test.txt                // Record the rewards of each episode
├── trained_models          // Pretrained models
```

## Usage

### Train
Before the training start , you can set some parameters in command line:

| params | option | default | description |
| :-----| :---- | :---- | :---- |
| --phase | train / test (string) | train | Train or Test ? |
| --used_gpu | true / false (bool) | true | Use GPU ? |
| --load_model | true / false (bool) | false | Load pretrained model ?|
| --model_dir | any directory (string) | 'pretrained model' | Where to store your model ?|

Start **training** using the command:
```
> python dqn.py --phase train [other params]
```
> The training reward will be store in `reward.txt`

### Test

If you have pretrained model in `model_dir`, start **testing** via the command:
```
> python dqn.py --phase test
```
> "**Success**" means that **at least one box** is pushed on the target, the success ratio will be recorded in `test.txt`.

### Supervising the training result

In the directory which the `dqn.py` is, type the command below to supervise your model via [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html):
```
> tensorboard --logdir=runs
```

## Report Link
[This report](https://github.com/hedy881028/AI-final-project/blob/main/Report.pdf) includes the **methodology**, **experiments result**, some **discussion** and the **conclusion** of this project.

## References

[1] [Sokoban gym environment](https://github.com/mpSchrader/gym-sokoban)

[2] [Tetris gym environment](https://github.com/uvipen/Tetris-deep-Q-learning-pytorch)

[3] [Reinforcement Learning 進階篇：Deep Q-Learning](https://medium.com/pyladies-taiwan/reinforcement-learning-%E9%80%B2%E9%9A%8E%E7%AF%87-deep-q-learning-26b10935a745)

[4] [訓練DQN玩Atari Space Invaders](https://skywalker0803r.medium.com/%E8%A8%93%E7%B7%B4dqn%E7%8E%A9atari-space-invaders-9bc0fc264f5b)

[5] [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602)

[6] [高估問題、Target Network、Double DQN](https://www.youtube.com/watch?v=X2-56QN79zc&list=PLvOO0btloRntS5U8rQWT9mHFcUdYOUmIC&index=2)

[7] [Mnih, V., Kavukcuoglu, K., Silver, D. et al. Human-level control through deep reinforcement learning. Nature 518, 529–533 (2015).](https://doi.org/10.1038/nature14236)

[8] [van Hasselt, H., Guez, A., & Silver, D. (2016). Deep Reinforcement Learning with Double Q-Learning. Proceedings of the AAAI Conference on Artificial Intelligence, 30(1).](https://ojs.aaai.org/index.php/AAAI/article/view/10295)



