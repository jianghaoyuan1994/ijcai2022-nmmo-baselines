# IJCAI2022-NMMO-BASELINES
To get started with NeuralMMO environments and reinforcement learning frameworks, we provide baseline agents.

## install
```
# install NeuralMMO
pip install xxx
```


## monobeast-baseline
A modified [monobeast](https://github.com/facebookresearch/ baseline is provided in `monobeast-baseline`. 

### train
```
python -m torchbeast.monobeast \
    --total_steps 1000000000 \
    --num_actors 4 \
    --num_learner 1 \
    --batch_size 32 \
    --unroll_length 32 \
    --savedir ./results \
    --checkpoint_interval 3600 \
    --xpid nmmo
```

### evaluate
```
python -m torchbeast.monobeast \
    --total_steps 1000000000 \
    --num_actors 4 \
    --num_learner 1 \
    --batch_size 32 \
    --unroll_length 32 \
    --savedir ./results \
    --checkpoint_interval 3600 \
    --xpid nmmo \
    --mode test
```


## Baselines based on other frameworks
neuralmmo-baselines provide RL [baselines](https://github.com/NeuralMMO/baselines) implemented using various frameworks, such as [cleanrl](https://github.com/vwxyzjn/cleanrl), [sb3](https://github.com/DLR-RM/stable-baselines3), [rllib](https://github.com/ray-project/ray/tree/master/rllib). These baselines are provided to the participants who is familar with and prefer these RL frameworks. `For the participants without specific preference, we recommend you to use the torchbeast-baseline`.
