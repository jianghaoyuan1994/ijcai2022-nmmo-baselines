# IJCAI2022-NMMO-BASELINES
To get started with NeuralMMO environments and reinforcement learning frameworks, we provide some baseline agents.

## install
```bash
pip install git+https://github.com/IJCAI2022-NMMO/ijcai2022nmmo.git
pip install -r requirements.txt
```


## monobeast-baseline
A modified [monobeast](https://github.com/facebookresearch/) baseline is provided in `monobeast/`. 

### train
```bash
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
```bash
python -m torchbeast.monobeast --mode test --savedir ./results --xpid nmmo
```


## Baselines based on other frameworks
The [NeuralMMO-baselines](https://github.com/NeuralMMO/baselines) has impletented some baseline agents based on commonly used RL frameworks, such as [cleanrl](https://github.com/vwxyzjn/cleanrl), [sb3](https://github.com/DLR-RM/stable-baselines3), [rllib](https://github.com/ray-project/ray/tree/master/rllib). These baselines are provided to the participants who is familar with and prefer these frameworks. For people who use these frameworks, we have provided an exmaple submission based on rllib baseline. Choose your favorite way to implement your own agent.
<!-- `For the participants without specific preference, we recommend you to use the torchbeast-baseline`. -->


## FAQ

##### 1. How do I handle "core dump" error when run monobeast baseline?
This error is usually encountered due to insufficient memeory. Try smaller `num_actors, batch_size, unroll_length`.

##### 2. How can I create my own submission?
For more information, please refer to [submission API](https://github.com/IJCAI2022-NMMO/ijcai2022nmmo#team).
