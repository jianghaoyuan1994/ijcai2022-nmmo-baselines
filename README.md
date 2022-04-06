# IJCAI2022-NMMO-BASELINES
In order to get started with NeuralMMO environments, we provide a variety of baselines agent.

## install
```
# install NeuralMMO
pip install xxx
```

## xxx-baselines
xxx-baselines provides RL baseline implementations under various frameworks
More details are available [here](https://github.com/NeuralMMO/baselines)

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
