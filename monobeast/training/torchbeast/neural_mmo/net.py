import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_


def sequential_pack(layers):
    assert isinstance(layers, list)
    seq = nn.Sequential(*layers)
    for item in layers:
        if isinstance(item, nn.Conv2d) or isinstance(item, nn.ConvTranspose2d):
            seq.out_channels = item.out_channels
            break
        elif isinstance(item, nn.Conv1d):
            seq.out_channels = item.out_channels
            break
    return seq


def conv2d_block(
    in_channels,
    out_channels,
    kernel_size,
    stride=1,
    padding=0,
    dilation=1,
    groups=1,
    pad_type='zero',
    activation=None,
):
    block = []
    assert pad_type in ['zero', 'reflect', 'replication'], "invalid padding type: {}".format(pad_type)
    if pad_type == 'zero':
        pass
    elif pad_type == 'reflect':
        block.append(nn.ReflectionPad2d(padding))
        padding = 0
    elif pad_type == 'replication':
        block.append(nn.ReplicationPad2d(padding))
        padding = 0
    block.append(
        nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding=padding, dilation=dilation, groups=groups)
    )
    xavier_normal_(block[-1].weight)
    if activation is not None:
        block.append(activation)
    return sequential_pack(block)


class ResBlock(nn.Module):
    def __init__(self, in_channels, middle_channels, activation=nn.ReLU(), res_type='basic'):
        super(ResBlock, self).__init__()
        self.act = activation
        assert res_type in ['basic',
                            'bottleneck'], 'residual type only support basic and bottleneck, not:{}'.format(res_type)
        self.res_type = res_type

        self.conv1 = conv2d_block(in_channels, middle_channels, 3, 1, 1, activation=self.act)
        self.conv2 = conv2d_block(middle_channels, in_channels, 3, 1, 1, activation=None)

    def forward(self, x):
        residual = x
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.act(x + residual)
        return x

def fc_block(
    in_channels,
    out_channels,
    activation=None,
    use_dropout=False,
    dropout_probability=0.5
):
    block = []
    block.append(nn.Linear(in_channels, out_channels))
    xavier_normal_(block[-1].weight)
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim=56, project_dim=32):
        super(SpatialEncoder, self).__init__()
        self.act = nn.ReLU()
        self.project = conv2d_block(input_dim, project_dim, 1, 1, 0, activation=self.act)
        self.res = nn.ModuleList()
        self.resblock_num = 2
        for i in range(self.resblock_num):
            self.res.append(ResBlock(project_dim, 2*project_dim, self.act))

        # self.fc = nn.Linear( ,408)

    def forward(self, x, scatter_map):
        x = torch.cat([x, scatter_map], dim=1)
        x = self.project(x)
        for block in self.res:
            x = block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc(x)
        return x


class EntityEncoder(nn.Module):
    r'''
    B=batch size EN=any number of entities ID=input_dim OS=output_size=256
     (B*EN*ID)  (EN'*OS)          (EN'*OS)          (EN'*OS)           (B*EN*OS)
    x -> combine -> Transformer ->  act ->  entity_fc  -> split ->   entity_embeddings
          batch                         |      (B*EN*OS)   (B*OS)        (B*OS)
                                        \->  split ->  mean -> embed_fc -> embedded_entity
    '''

    def __init__(self, cfg):
        super(EntityEncoder, self).__init__()
        self.encode_modules = nn.ModuleDict()
        self.whole_cfg = cfg
        self.cfg = self.whole_cfg.model.encoder.obs_encoder.entity_encoder
        for k, item in self.cfg.module.items():
            if item['arc'] == 'one_hot':
                self.encode_modules[k] = nn.Embedding.from_pretrained(torch.eye(item['num_embeddings']),freeze=True,padding_idx=None)
            if item['arc'] == 'binary':
                self.encode_modules[k] = torch.nn.Embedding.from_pretrained(get_binary_embed_mat(item['num_embeddings']),freeze=True,padding_idx=None)
        self.act = build_activation(self.cfg.activation)
        self.transformer = Transformer(
            input_dim=self.cfg.input_dim,
            head_dim=self.cfg.head_dim,
            hidden_dim=self.cfg.hidden_dim,
            output_dim=self.cfg.output_dim,
            head_num=self.cfg.head_num,
            mlp_num=self.cfg.mlp_num,
            layer_num=self.cfg.layer_num,
            dropout_ratio=self.cfg.dropout_ratio,
            activation=self.act,
            ln_type=self.cfg.ln_type
        )
        self.entity_fc = fc_block(self.cfg.output_dim, self.cfg.output_dim, activation=self.act)
        self.embed_fc = fc_block(self.cfg.output_dim, self.cfg.output_dim, activation=self.act)
        if self.whole_cfg.model.entity_reduce_type == 'attention_pool':
            self.attention_pool = AttentionPool(key_dim=self.cfg.output_dim, head_num=2, output_dim=self.cfg.output_dim)
        elif self.whole_cfg.model.entity_reduce_type == 'attention_pool_add_num':
            self.attention_pool = AttentionPool(key_dim=self.cfg.output_dim, head_num=2, output_dim=self.cfg.output_dim, max_num=MAX_ENTITY_NUM + 1)

    def forward(self, x, entity_num):
        entity_embedding = []
        for k, item in self.cfg.module.items():
            assert k in x.keys(), '{} not in {}'.format(k, x.keys())
            if item['arc'] == 'one_hot':
                # check data
                over_cross_data = x[k] >= item['num_embeddings']
                # if over_cross_data.any():
                #     print(k, x[k][over_cross_data])
                lower_cross_data = x[k] < 0
                if lower_cross_data.any():
                    print(k, x[k][lower_cross_data])
                    raise RuntimeError
                clipped_data = x[k].long().clamp_(max=item['num_embeddings'] - 1)
                entity_embedding.append(self.encode_modules[k](clipped_data))
            elif item['arc'] == 'binary':
                entity_embedding.append(self.encode_modules[k](x[k].long()))
            elif item['arc'] == 'unsqueeze':
                entity_embedding.append(x[k].float().unsqueeze(dim=-1))
        x = torch.cat(entity_embedding, dim=-1)
        mask = sequence_mask(entity_num, max_len=x.shape[1])
        x = self.transformer(x, mask=mask)
        entity_embeddings = self.entity_fc(self.act(x))

        if self.whole_cfg.model.entity_reduce_type in ['entity_num', 'selected_units_num']:
            x_mask = x * mask.unsqueeze(dim=2)
            embedded_entity = x_mask.sum(dim=1) / entity_num.unsqueeze(dim=-1)
        elif self.whole_cfg.model.entity_reduce_type == 'constant':
            x_mask = x * mask.unsqueeze(dim=2)
            embedded_entity = x_mask.sum(dim=1) / 512
        elif self.whole_cfg.model.entity_reduce_type == 'attention_pool':
            embedded_entity = self.attention_pool(x, mask=mask.unsqueeze(dim=2))
        elif self.whole_cfg.model.entity_reduce_type == 'attention_pool_add_num':
            embedded_entity = self.attention_pool(x, num=entity_num, mask=mask.unsqueeze(dim=2))
        else:
            raise NotImplementedError
        embedded_entity = self.embed_fc(embedded_entity)
        return entity_embeddings, embedded_entity, mask


def scatter_connection(project_embeddings, entity_location):
    # print(project_embeddings.shape)  # T*B 100 48
    scatter_dim = project_embeddings.shape[-1]
    B, H, W = project_embeddings.shape[0], 15, 15
    # print(entity_location.shape)  # T*B 100 2
    # print(entity_location)
    device = entity_location.device
    entity_num = entity_location.shape[1]
    index = entity_location.view(-1, 2).long()  # T*B*100 2
    bias = torch.arange(B).unsqueeze(1).repeat(1, entity_num).view(-1).to(device)
    bias *= H * W
    # print(bias)
    # index[:, 0].clamp_(0, W - 1)
    # index[:, 1].clamp_(0, H - 1)
    index = index[:, 0] * W + index[:, 1]
    index += bias
    # print(index.shape)   #  T*B*100
    # print(index)
    index = index.repeat(scatter_dim, 1)
    # print(index.shape)  # scatter_dim T*B*100

    # flat scatter map and project embeddings
    scatter_map = torch.zeros(scatter_dim, B * H * W, device=device)
    project_embeddings = project_embeddings.view(-1, scatter_dim).permute(1, 0)
    # print(project_embeddings.shape)  # scatter_dim T*B*100
    # print(project_embeddings[:,:7])
    # print(index[:,:7])
    # scatter_map.scatter_(dim=1, index=index, src=project_embeddings)
    scatter_map.scatter_add_(dim=1, index=index, src=project_embeddings)

    scatter_map = scatter_map.reshape(scatter_dim, B, H, W)
    scatter_map = scatter_map.permute(1, 0, 2, 3)
    # print(scatter_map.shape)  # T*B scatter_dim H W
    return scatter_map


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
        self.spatialEncoder = SpatialEncoder()
        self.core = nn.Linear(512 + 8 * 15 * 15, 512)

        self.act = nn.ReLU()
        self.unit_nn = nn.Linear(77, 48)
        self.unit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, batch_first=True),
            num_layers=3)
        self.entity_fc = fc_block(48, 48, activation=self.act)
        self.embed_fc = fc_block(48, 64, activation=self.act)


        self.policy_move = nn.Linear(512, 5)
        self.policy_attack_type = nn.Linear(512, 4)
        self.policy_attack_unit_core = nn.Linear(512,5120)
        self.baseline = nn.Linear(512, 1)

        self.attack_emb = nn.Embedding(512, 4)

    def initial_state(self, batch_size=1):
        return tuple()

    def forward(self, input_dict, state=()):
        # [T, B, ...]
        # print(input_dict.keys())
        agent_map, entity_id, entity_in, rangeable, team_in, va_move, \
        obs_emb, meleeable, mask, magicable, local_map, entity_loc = \
            input_dict["agent_map"], input_dict["entity_id"], input_dict["entity_in"], \
            input_dict["rangeable"], input_dict["team_in"], input_dict["va_move"], \
            input_dict["obs_emb"], input_dict["meleeable"], input_dict["mask"], \
            input_dict["magicable"], input_dict["local_map"], input_dict["entity_loc"]



        local_map = F.one_hot(local_map, num_classes=6).permute(0, 1, 4, 2, 3)  # T B C H W
        local_map = torch.flatten(local_map, 0, 1)
        map_emb = torch.cat([agent_map.flatten(0, 1), local_map], dim=1)  # T*B C 15 15
        # print(map_emb.shape)

        team_in = F.one_hot(team_in, num_classes=17).flatten(0, 1)   # T*B 100 17
        entity_in = F.one_hot(entity_in, num_classes=9).flatten(0, 1)  # T*B 100 9
        obs_emb = obs_emb.flatten(0, 1)  # T*B 100 51
        entity_emb = torch.cat([obs_emb, entity_in, team_in], dim=2)   # T*B 100 77
        # print(entity_emb.shape)
        # print(entity_emb[0, -5:, :])
        mask = mask.flatten(0, 1)
        entity_emb = self.unit_nn(entity_emb)
        # print(entity_emb.shape)
        entity_emb = self.unit_transformer(entity_emb, src_key_padding_mask=mask)

        entity_embeddings = self.entity_fc(self.act(entity_emb))

        # print(torch.sum(~mask, dim=-1))
        # print(mask.shape, entity_emb.shape)
        entity_emb_mask = entity_emb * (~mask.unsqueeze(dim=2))
        scatter_map = scatter_connection(entity_emb_mask, entity_loc.flatten(0, 1))
        # print(entity_emb_mask[0, -5:, :])  # 全0
        # print(entity_emb_mask.sum(dim=1).shape, torch.sum(~mask, dim=-1).unsqueeze(dim=-1).shape)   # 8,48  8,1
        embedded_entity = entity_emb_mask.sum(dim=1) / torch.sum(~mask, dim=-1).unsqueeze(dim=-1)
        # print(embedded_entity.shape)   # 8，48
        embedded_entity = self.embed_fc(embedded_entity)
        mix_embeded_

        embedded_spatial = self.spatialEncoder(map_emb, scatter_map)




        # print(entity_emb.shape)
        # print(entity_emb[0, -5:, :])


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