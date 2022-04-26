import torch
import torch.nn as nn
import torch.nn.functional as F


class NMMONet(nn.Module):
    def __init__(self, observation_space, num_actions, use_lstm=False):
        super().__init__()
        # self.mlp = nn.Sequential(
        #     nn.Linear(1300, 2048),
        #     nn.ReLU(),
        #     nn.Linear(2048, 1024),
        #     nn.ReLU(),
        #     nn.Linear(1024, 512),
        #     nn.ReLU(),
        # )
        self.cnn = nn.Sequential(
            nn.Conv2d(in_channels=7,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU(),
            nn.Conv2d(in_channels=16,
                      out_channels=8,
                      kernel_size=3,
                      stride=1,
                      padding=1), nn.ReLU())
        self.core = nn.Linear(512 + 8 * 15 * 15, 512)

        self.unit_nn = nn.Linear(27, 48)
        self.unit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, batch_first=True),
            num_layers=3)

        self.policy_move = nn.Linear(512, 5)
        self.policy_attack_type = nn.Linear(512, 4)
        self.policy_attack_unit_core = nn.Linear(512,5120)
        self.baseline = nn.Linear(512, 1)

        self.attack_emb = nn.Embedding(512, 4)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, input_dict, state=()):
        # [T, B, ...]
        x1, x2, mask = input_dict["obs_emb"], input_dict["map_frame"], input_dict["mask"]
        T, B, *_ = x1.shape
        x1 = torch.flatten(x1, 0, 1)
        x1 = self.unit_nn(x1)
        mask = torch.flatten(mask, 0, 1)
        x1 = self.unit_transformer(x1, src_key_padding_mask=mask)
        print(x1.shape, mask.shape)
        x1 = torch.masked_select(x1, mask)
        print(x1.shape)
        x1 = torch.mean(x1, dim=1)
        print(x1.shape)
        x2 = torch.flatten(x2, 0, 1)
        x2 = self.cnn(x2)
        x2 = torch.flatten(x2, start_dim=1)
        x = torch.cat([x1, x2], dim=-1)
        x = F.relu(self.core(x))

        policy_logits = self.policy(x)
        baseline = self.baseline(x)
        action = torch.multinomial(F.softmax(policy_logits, dim=1),
                                   num_samples=1)
        policy_logits = policy_logits.view(T, B, -1)
        baseline = baseline.view(T, B)
        action = action.view(T, B)
        return (dict(policy_logits=policy_logits,
                     baseline=baseline,
                     action=action), tuple())