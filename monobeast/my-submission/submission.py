from pathlib import Path

import torch
import torch.nn as nn
import tree
from ijcai2022nmmo import Team
from nmmo import config
from numpy import ndarray
from torchbeast.monobeast import Net, batch, unbatch
from torchbeast.neural_mmo.train_wrapper import FeatureParser, TrainWrapper


class MonobeastBaselineTeam(Team):
    obs_keys = ["agents_frame", "map_frame"]
    feas_dim = [15]

    def __init__(self,
                 team_id: str,
                 env_config: config.Config,
                 checkpoint_path=None):
        super().__init__(team_id, env_config)
        self.model: nn.Module = Net()
        if checkpoint_path is not None:
            print(f"load checkpoint: {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location="cpu")
            self.model.load_state_dict(checkpoint["model_state_dict"])
        self.feature_parser = FeatureParser(self.feas_dim)

    def act(self, observations):
        def _format(x):
            if isinstance(x, ndarray):
                return torch.from_numpy(x).view(1, 1, *x.shape)
            elif isinstance(x, (int, float)):
                return torch.tensor(x).view(1, 1)
            else:
                raise RuntimeError
        observations = self.feature_parser.parse(observations)
        observations = tree.map_structure(_format, observations)
        obs_batch, ids = batch(observations, self.obs_keys)
        agent_state = self.model.initial_state(batch_size=8)
        output, _ = self.model(obs_batch, agent_state)
        output, _ = unbatch(output, _, ids)
        actions = {i: output[i]["action"].item() for i in output}
        actions = TrainWrapper.transform_action(actions)
        return actions




class Submission:
    team_klass = MonobeastBaselineTeam
    init_params = {
        "checkpoint_path": Path(__file__).parent / "checkpoints" / "model.pt"
    }
