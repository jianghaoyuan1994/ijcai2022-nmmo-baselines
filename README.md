# My IJCAI2022-NMMO Completion Solution
This solution was ranked 10 in the final evaluation at [here](https://www.aicrowd.com/challenges/ijcai-2022-the-neural-mmo-challenge/leaderboards).

## Install
```bash
#pip install git+http://gitlab.aicrowd.com/henryz/ijcai2022nmmo.git
pip install -r requirements.txt
```


## monobeast-baseline

### train
```bash
cd monobeast/training
python -m torchbeast.monobeast \
    --total_steps 1000000000 \
    --use_lstm \
    --num_actors 15 \
    --num_learner 1  \
    --batch_size 32  \
    --unroll_length 48 \
    --savedir ./6_23_attack \
    --learning_rate 0.001 \
    --entropy_cost 0.001 \
    --checkpoint_interval 3600 \
    --xpid nmmo
```

### evaluate
```bash
cd monobeast/training
python -m torchbeast.monobeast --mode test --savedir ./results --xpid nmmo
```


## Baselines based on other frameworks
The [NeuralMMO-baselines](https://github.com/NeuralMMO/baselines/tree/ijcai-competition) (ijcai-competition branch) has implemented some baseline agents based on commonly used RL frameworks, such as [cleanrl](https://github.com/vwxyzjn/cleanrl), [sb3](https://github.com/DLR-RM/stable-baselines3), [rllib](https://github.com/ray-project/ray/tree/master/rllib). These baselines are provided to the participants who is familiar with and prefer these frameworks. Choose your favorite to implement your own agent.


## FAQ

##### 1. How do I handle "core dump" error when run monobeast baseline?
This error is usually encountered due to insufficient memory. Try smaller `num_actors, batch_size, unroll_length`.
