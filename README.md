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
```

## Usage

### Train
Before the training start , you can set some parameters (in`dqn.py`):
- **used_gpu** -> Decide whether you want to use GPU for acceleration (default **True**)
- **load_model** -> Decide whether you want to load pretrained model (default **False**)
- **model_dir** -> The directory that stores your trained model (default **'trained_models'**)

After the parameters are setting properly, start training using the command:
```
> python dqn.py
```

### Supervising the result

In the directory which the `dqn.py` is, type the command below to supervise your model via [Tensorboard](https://pytorch.org/tutorials/recipes/recipes/tensorboard_with_pytorch.html):
```
> tensorboard --logdir=runs
```

## Methodology

### Training flow

### DQN arcitechture

### Action definition

### Reward setting

## Experiment

## Conclusion

## Report Link
[HERE!](https://docs.google.com/document/d/1xJukNOXyYxqJz3gwPA5f2EANCiOmYOJHpXHXBcki7HU/edit)


