# IJCAI2022-NMMO-BASELINES
To get started with NeuralMMO environments and reinforcement learning frameworks, we provide some baseline agents.

## install
```bash
pip install git+http://gitlab.aicrowd.com/henryz/ijcai2022nmmo.git
pip install -r requirements.txt
```


## monobeast-baseline
A modified [monobeast](https://github.com/facebookresearch/) baseline is provided in `monobeast/`. 
- `monobeast/my-submission/`: Code for submission. 
    - For successful submission, one must copy all files under this directory and the model checkpoint to [`ijcai2022-nmmo-starter-kit/my-submission/`](https://gitlab.aicrowd.com/neural-mmo/ijcai2022-nmmo-starter-kit/-/tree/main/my-submission).
- `monobeast/training/`: Code for training.

### train
```bash
cd monobeast/training
python -m torchbeast.monobeast \
    --total_steps 1000000000 \
    --num_actors 8 \
    --num_learner 1 \
    --batch_size 32 \
    --unroll_length 64 \
    --savedir ./results \
    --checkpoint_interval 1800 \
    --xpid nmmo
```

### evaluate
```bash
cd monobeast/training
python -m torchbeast.monobeast --mode test --savedir ./results --xpid nmmo
```


## Baselines based on other frameworks
The [NeuralMMO-baselines](https://github.com/NeuralMMO/baselines) has implemented some baseline agents based on commonly used RL frameworks, such as [cleanrl](https://github.com/vwxyzjn/cleanrl), [sb3](https://github.com/DLR-RM/stable-baselines3), [rllib](https://github.com/ray-project/ray/tree/master/rllib). These baselines are provided to the participants who is familiar with and prefer these frameworks. Choose your favorite to implement your own agent.


## FAQ

##### 1. How do I handle "core dump" error when run monobeast baseline?
This error is usually encountered due to insufficient memory. Try smaller `num_actors, batch_size, unroll_length`.