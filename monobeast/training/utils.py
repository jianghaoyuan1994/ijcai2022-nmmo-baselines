import matplotlib.pyplot as plt
import pandas as pd
import os
from os.path import join as osj
import numpy as np
import torch


def plt_maps():
    map_dir = "maps"
    maps_path = os.listdir(map_dir)[100:]

    for i in maps_path:
        map_ = np.load(osj(osj(map_dir, i), "map.npy"))
        plt.imshow(map_)
        plt.show()

def plt_logs():
    log_path = 'results/nmmo/logs.csv'
    dataframe = pd.read_csv(log_path)
    y = dataframe['step'].values
    total_loss = dataframe['total_loss'].values
    pg_loss = dataframe['pg_loss'].values
    baseline_loss = dataframe['baseline_loss'].values
    entropy_loss = dataframe['entropy_loss'].values
    mean_episode_step = dataframe['mean_episode_step'].values

    plt.plot(y, total_loss, label="total loss")
    plt.title(label="total loss")
    plt.show()

    plt.plot(y, pg_loss, label="pg_loss")
    plt.show()

    plt.plot(y, baseline_loss, label="baseline_loss")
    plt.show()

    plt.plot(y, entropy_loss, label="entropy_loss")
    plt.show()

    plt.plot(y, mean_episode_step, label="mean_episode_step")
    plt.show()


def checkpoint(flags, logging, checkpointpath, model, optimizer, step, scheduler):
    if flags.disable_checkpoint:
        return
    logging.info("Saving checkpoint to %s", checkpointpath)
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "flags": vars(flags),
        },
        checkpointpath.joinpath(f"model_{step}.pt"),
    )


def load_model(path, model, optimizer, scheduler):
    para_dict = torch.load(path)
    model.load_state_dict(para_dict['model_state_dict'])
    optimizer.load_state_dict(para_dict['optimizer_state_dict'])
    scheduler.load_state_dict(para_dict['scheduler_state_dict'])
    # flags = para_dict['flags']
    # return flags
# plt_maps()
