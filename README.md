# AI-final-project
> 0716081 葉晨 / 0716083 羅子涵

## Introduction

Deep Q-learning 是將強化學習技巧之一的 Q-learning 與深度學習結合的技術，利用遊戲畫面當作 input，並利用 neural network (稱為 Deep Q Network ) 取代原本的 Q-table，可以大大增加 Deep Q-learning 的穩定性及實用性。

此專案的目的是使用 Deep Q Network (DQN) 來 **訓練電腦學習推箱子遊戲 (Sokoban)** 。推箱子遊戲是一款將所有積木推到特定位置的遊戲。我們這次使用的訓練環境是已經搭建好的 gym 環境。我們此次的目標為應用現有的 Deep Q-learning 技術，並且對一些超參數進行調整，希望能讓電腦學習推箱子遊戲的遊玩方式，並拿到盡量高的分數。

## Project overview

```
├── gym_sokoban             // Gym environment of Sokoban
├── trained_models          // Pretrained models
├── dqn.py                  
├── reward.txt              // record the rewards of each episode
├── test.txt                // record the rewards of each episode
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

If you have pretrained model in `model_dir`, start **testing** via the command:
```
> python dqn.py --phase test
```
> The training reward will be store in `test.txt`

### Supervising the result

In the directory which the `dqn.py` is, type the command below to supervise your model via [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html):
```
> tensorboard --logdir=runs
```

## Report Link
[HERE!](https://docs.google.com/document/d/1xJukNOXyYxqJz3gwPA5f2EANCiOmYOJHpXHXBcki7HU/edit)


