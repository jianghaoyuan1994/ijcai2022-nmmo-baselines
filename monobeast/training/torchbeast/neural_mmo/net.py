import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.init import xavier_normal_
from typing import List, Tuple, Optional
from torch import Tensor
import torch.jit as jit
from torchbeast.core.mask import MaskedPolicy
import math


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


class LSTMLayer(nn.Module):
    def __init__(self, cell, *cell_args):
        super(LSTMLayer, self).__init__()
        self.cell = cell(*cell_args)

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        inputs = input.unbind(0)
        outputs = torch.jit.annotate(List[Tensor], [])
        for i in range(len(inputs)):
            out, state = self.cell(inputs[i], state)
            outputs += [out]
        return torch.stack(outputs), state


class LayerNormLSTMCell(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(LayerNormLSTMCell, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.weight_ih = nn.Parameter(torch.randn(4 * hidden_size, input_size))
        self.weight_hh = nn.Parameter(torch.randn(4 * hidden_size, hidden_size))
        # The layernorms provide learnable biases

        ln = nn.LayerNorm
        self.layernorm_i = ln(4 * hidden_size)
        self.layernorm_h = ln(4 * hidden_size)
        self.layernorm_c = ln(hidden_size)

    def forward(self, input: Tensor, state: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        hx, cx = state
        igates = self.layernorm_i(torch.mm(input, self.weight_ih.t()))
        hgates = self.layernorm_h(torch.mm(hx, self.weight_hh.t()))
        gates = igates + hgates
        ingate, forgetgate, cellgate, outgate = gates.chunk(4, 1)

        ingate = torch.sigmoid(ingate)
        forgetgate = torch.sigmoid(forgetgate)
        cellgate = torch.tanh(cellgate)
        outgate = torch.sigmoid(outgate)

        cy = self.layernorm_c((forgetgate * cx) + (ingate * cellgate))
        hy = outgate * torch.tanh(cy)

        return hy, (hy, cy)


def init_stacked_lstm(num_layers, layer, first_layer_args, other_layer_args):
    layers = [layer(*first_layer_args)] + [layer(*other_layer_args)
                                           for _ in range(num_layers - 1)]
    return nn.ModuleList(layers)


class StackedLSTM(nn.Module):
    __constants__ = ['layers']  # Necessary for iterating through self.layers

    def  __init__(self, num_layers, layer, first_layer_args, other_layer_args):
        super(StackedLSTM, self).__init__()
        self.layers = init_stacked_lstm(num_layers, layer, first_layer_args,
                                        other_layer_args)

    def forward(self, input, states):
        # List[LSTMState]: One state per layer
        output_states = jit.annotate(List[Tuple[Tensor, Tensor]], [])
        output = input
        # XXX: enumerate https://github.com/pytorch/pytorch/issues/14471
        i = 0
        for rnn_layer in self.layers:
            state = states[i]
            output, out_state = rnn_layer(output, state)
            output_states += [out_state]
            i += 1
        return output, output_states


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


class ResFCBlock(nn.Module):
    r'''
    Overview:
        Residual Block with 2 fully connected block
        x -> fc1 -> norm -> act -> fc2 -> norm -> act -> out
        \_____________________________________/+

    Interface:
        __init__, forward
    '''

    def __init__(self, in_channels, activation=nn.ReLU(), norm_type='BN'):
        r"""
        Overview:
            Init the Residual Block

        Arguments:
            - activation (:obj:`nn.Module`): the optional activation function
            - norm_type (:obj:`str`): type of the normalization, defalut set to batch normalization
        """
        super(ResFCBlock, self).__init__()
        self.act = activation
        self.fc1 = fc_block(in_channels, in_channels, activation=self.act, norm_type=norm_type)
        self.fc2 = fc_block(in_channels, in_channels, activation=None, norm_type=norm_type)

    def forward(self, x):
        r"""
        Overview:
            return  output of  the residual block with 2 fully connected block

        Arguments:
            - x (:obj:`tensor`): the input tensor

        Returns:
            - x(:obj:`tensor`): the resblock output tensor
        """
        residual = x
        x = self.fc1(x)
        x = self.fc2(x)
        x = self.act(x + residual)
        return x


def fc_block(
        in_channels,
        out_channels,
        activation=None,
        use_dropout=False,
        norm_type=None,
        dropout_probability=0.5
):
    block = [nn.Linear(in_channels, out_channels)]
    xavier_normal_(block[-1].weight)
    if norm_type is not None and norm_type != 'none':
        if norm_type == 'LN':
            block.append(nn.LayerNorm(out_channels))
        else:
            raise NotImplementedError
    if isinstance(activation, torch.nn.Module):
        block.append(activation)
    elif activation is None:
        pass
    else:
        raise NotImplementedError
    if use_dropout:
        block.append(nn.Dropout(dropout_probability))
    return sequential_pack(block)


class SpatialEncoder(nn.Module):
    def __init__(self, input_dim=55, project_dim=64):
        super(SpatialEncoder, self).__init__()
        self.act = nn.ReLU()
        self.project = conv2d_block(input_dim, project_dim, 1, 1, 0, activation=self.act)
        self.res = nn.ModuleList()
        self.resblock_num = 2
        for i in range(self.resblock_num):
            self.res.append(ResBlock(project_dim, project_dim, self.act))

        self.fc1 = fc_block(14400, 560, self.act)

    def forward(self, x, scatter_map):
        # print(x.shape, scatter_map.shape)
        x = torch.cat([x, scatter_map], dim=1)
        x = self.project(x)
        for block in self.res:
            x = block(x)
        x = x.view(x.shape[0], -1)
        x = self.fc1(x)
        return x


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
    index = index[:, 0] * W + index[:, 1]  # todo check
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


class AttackHead(nn.Module):
    # __constants__ = ['act', "res", "action_fc", 'project', "entity_fc"]

    def __init__(self):
        super(AttackHead, self).__init__()
        self.act = nn.ReLU()
        # attack type
        self.project = fc_block(512, 128, activation=self.act, norm_type=None)
        blocks = [ResFCBlock(128, self.act, 'LN') for _ in range(2)]
        self.res = nn.Sequential(*blocks)
        # self.action_fc = GLU(128, 4, 128)


        # entity fc
        self.entity_fc_meleeable = fc_block(48, 128)
        self.entity_fc_rangeable = fc_block(48, 128)
        self.entity_fc_magicable = fc_block(48, 128)
        self.max_pooling = nn.MaxPool1d(100)
        # not attack emb
        self.not_attack = torch.nn.Parameter(torch.zeros(1, 128))
        torch.nn.init.xavier_uniform_(self.not_attack)

        # select unit
        self.key_fc = fc_block(48, 32, activation=None, norm_type=None)
        self.query_fc1 = fc_block(512, 128, activation=self.act, norm_type=None)
        self.query_fc2 = fc_block(128, 32, activation=None, norm_type=None)
        self.key_dim = 32

    def forward(self, lstm_output, rangeable, meleeable, magicable,
                entity_embeddings, entity_id, is_attack, entity_mask, is_train):
        # select attack type
        x = self.project(lstm_output)
        x = self.res(x)
        T, B, N = rangeable.shape
        rangeable = rangeable.flatten(0, 1)
        meleeable = meleeable.flatten(0, 1)
        magicable = magicable.flatten(0, 1)
        is_attack = is_attack.flatten(0, 1)
        entity_id = entity_id.flatten(0, 1)

        entity_embeddings_meleeable = self.entity_fc_meleeable(entity_embeddings)
        entity_embeddings_rangeable = self.entity_fc_rangeable(entity_embeddings)
        entity_embeddings_magicable = self.entity_fc_magicable(entity_embeddings)
        rangeable_embedding = self.max_pooling(
            (entity_embeddings_rangeable * rangeable.unsqueeze(dim=2)).permute(0, 2, 1)
        ).permute(0, 2, 1)
        meleeable_embedding = self.max_pooling(
            (entity_embeddings_meleeable * meleeable.unsqueeze(dim=2)).permute(0, 2, 1)
        ).permute(0, 2, 1)
        magicable_embedding = self.max_pooling(
            (entity_embeddings_magicable * magicable.unsqueeze(dim=2)).permute(0, 2, 1)
        ).permute(0, 2, 1)

        attackable_embedding = torch.cat(
            [self.not_attack.unsqueeze(0).repeat(T*B, 1, 1),
             meleeable_embedding, rangeable_embedding, magicable_embedding],
            dim=1)
        #print(x.shape, attackable_embedding.shape)  # 8 128   8 4 128
        logits_type = (x.unsqueeze(1) * attackable_embedding).sum(dim=2)
        logits_type.masked_fill_(~is_attack.bool(), value=-1e9)
        dis_type = torch.distributions.categorical.Categorical(logits=logits_type)
        action_type = dis_type.sample()
        # print(action_type)

        # SELECT unit
        device = meleeable.device
        # attack_type_index = torch.cat([torch.ones(T*B, N).to(device=device).unsqueeze(1),
        #                                meleeable.unsqueeze(1), rangeable.unsqueeze(1),
        #                                magicable.unsqueeze(1)],
        #                                 dim=1)

        key_not_attack = self.key_fc(
            entity_embeddings * torch.ones(T*B, N).to(device=device).unsqueeze(dim=2))
        key_melee_attack = self.key_fc(
            entity_embeddings * meleeable.unsqueeze(dim=2))
        key_range_attack = self.key_fc(
            entity_embeddings * rangeable.unsqueeze(dim=2))
        key_magic_attack = self.key_fc(
            entity_embeddings * magicable.unsqueeze(dim=2))
        query = self.query_fc2(self.query_fc1(lstm_output))

        logits_not_attack_unit = query.unsqueeze(1) * key_not_attack
        logits_not_attack_unit = logits_not_attack_unit.sum(dim=2)
        logits_not_attack_unit.masked_fill_(~torch.ones(T*B, N).to(device=device).bool(), value=-1e9)
        logits_not_attack_unit.masked_fill_(entity_mask, value=-1e9)
        dis_not_attack_unit = torch.distributions.categorical.Categorical(logits=logits_not_attack_unit)
        action_not_attack_unit = dis_not_attack_unit.sample()

        logits_melee_attack_unit = query.unsqueeze(1) * key_melee_attack
        logits_melee_attack_unit = logits_melee_attack_unit.sum(dim=2)
        logits_melee_attack_unit.masked_fill_(~meleeable.bool(), value=-1e9)
        logits_melee_attack_unit.masked_fill_(entity_mask, value=-1e9)
        dis_melee_attack_unit = torch.distributions.categorical.Categorical(logits=logits_melee_attack_unit)
        action_melee_attack_unit = dis_melee_attack_unit.sample()

        logits_range_attack_unit = query.unsqueeze(1) * key_range_attack
        logits_range_attack_unit = logits_range_attack_unit.sum(dim=2)
        logits_range_attack_unit.masked_fill_(~rangeable.bool(), value=-1e9)
        logits_range_attack_unit.masked_fill_(entity_mask, value=-1e9)
        dis_range_attack_unit = torch.distributions.categorical.Categorical(logits=logits_range_attack_unit)
        action_range_attack_unit = dis_range_attack_unit.sample()

        logits_magic_attack_unit = query.unsqueeze(1) * key_magic_attack
        logits_magic_attack_unit = logits_magic_attack_unit.sum(dim=2)
        logits_magic_attack_unit.masked_fill_(~magicable.bool(), value=-1e9)
        logits_magic_attack_unit.masked_fill_(entity_mask, value=-1e9)
        dis_magic_attack_unit = torch.distributions.categorical.Categorical(logits=logits_magic_attack_unit)
        action_magic_attack_unit = dis_magic_attack_unit.sample()

        # print(entity_id)
        # print(action_unit)
        # print(logits_unit)
        dis_unit = torch.concat(
            [dis_not_attack_unit.logits.unsqueeze(1), dis_melee_attack_unit.logits.unsqueeze(1),
             dis_range_attack_unit.logits.unsqueeze(1), dis_magic_attack_unit.logits.unsqueeze(1)],
            dim=1)[range(0, T * B), action_type]
        action_unit = torch.concat(
            [action_not_attack_unit.unsqueeze(1), action_melee_attack_unit.unsqueeze(1),
             action_range_attack_unit.unsqueeze(1), action_magic_attack_unit.unsqueeze(1)],
            dim=1
        )[range(0, T * B), action_type]
        action_unit_id = entity_id[range(T*B), action_unit]
        # print(action_unit_id)
        if is_train:
            return dis_type.logits, \
                   dis_not_attack_unit.logits, dis_melee_attack_unit.logits, dis_range_attack_unit.logits, dis_magic_attack_unit.logits, \
                   action_type, action_unit_id
        else:
            return dis_type.logits, dis_unit, action_type, action_unit_id


class Policy(nn.Module):
    def __init__(self):
        super(Policy, self).__init__()
        self.move_head = fc_block(512, 5, nn.ReLU())
        self.attack_type_unit_id_head = AttackHead()

    def forward(self, lstm_output, entity_embeddings, entity_id,
                rangeable, meleeable, magicable, entity_mask, move_mask, is_attack, is_train):
        # action move
        # print(move_mask.shape)
        T, B, _ = move_mask.shape
        logit_move = self.move_head(lstm_output)
        if move_mask is not None:
            move_mask = torch.flatten(move_mask, 0, 1)
        dist_move = MaskedPolicy(logit_move, valid_actions=move_mask)
        action_move = dist_move.sample()
        action_move = action_move.view(T, B)

        # select attack type and target id
        if is_train:
            dis_type, \
            dis_not_attack_unit, dis_melee_attack_unit, dis_range_attack_unit, dis_magic_attack_unit, \
            action_type, action_unit_id = \
                self.attack_type_unit_id_head(
                    lstm_output, rangeable, meleeable, magicable,
                    entity_embeddings, entity_id, is_attack, entity_mask, is_train)
            return dist_move.logits, dis_type, \
                   dis_not_attack_unit, dis_melee_attack_unit, dis_range_attack_unit, dis_magic_attack_unit, \
                   action_move, action_type, action_unit_id

        else:

            dis_type, dis_unit, action_type, action_unit_id = \
                self.attack_type_unit_id_head(
                    lstm_output, rangeable, meleeable, magicable,
                    entity_embeddings, entity_id, is_attack, entity_mask, is_train)

            return dist_move.logits, dis_type, dis_unit, action_move, action_type, action_unit_id


def compute_denominator(x: torch.Tensor, dim: int) -> torch.Tensor:
    x = x // 2 * 2
    x = torch.div(x, dim)
    x = torch.pow(10000., x)
    x = torch.div(1., x)
    return x


class NMMONet(nn.Module):
    def __init__(self):
        super().__init__()
        self.spatialEncoder = SpatialEncoder()

        self.time_embedding_dim = 16
        self.position_array = torch.nn.Parameter(
            compute_denominator(torch.arange(0, 16, dtype=torch.float),
                                16), requires_grad=False)

        self.act = nn.ReLU()
        self.unit_nn = nn.Linear(70, 96)
        self.unit_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=96, nhead=4, dim_feedforward=128, batch_first=True),
            num_layers=3)

        self.unit_team = nn.Linear(44, 45)
        self.team_transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=48, nhead=4, dim_feedforward=96, batch_first=True),
            num_layers=3)

        self.entity_fc = fc_block(96, 48, activation=self.act)
        self.embed_fc = fc_block(96, 64, activation=self.act)

        self.mix_fc = fc_block(624, 624, activation=self.act)

        self.num_layers = 2
        self.core_lstm = StackedLSTM(num_layers=self.num_layers, layer=LSTMLayer,
                                     first_layer_args=[LayerNormLSTMCell, 1024, 512],
                                     other_layer_args=[LayerNormLSTMCell, 512, 512])
        #
        self.policy = Policy()

        # self.policy_move = nn.Linear(512, 5)
        # self.policy_attack_type = nn.Linear(512, 4)
        # self.policy_attack_unit_core = nn.Linear(512,5120)
        self.baseline = nn.Linear(512, 1)

        self.attack_emb = nn.Embedding(512, 4)

    def time_encoder(self, x: Tensor):
        v = torch.zeros(size=(x.shape[0], self.time_embedding_dim), dtype=torch.float, device=x.device)
        assert len(x.shape) == 1, "{}".format(x.shape)
        x = x.unsqueeze(dim=1)
        v[:, 0::2] = torch.sin(x * self.position_array[0::2])  # even
        v[:, 1::2] = torch.cos(x * self.position_array[1::2])  # odd
        return v

    def initial_state(self, batch_size=1):
        return torch.zeros(2, 2, batch_size, 512)

    def forward(self, input_dict, state=(), is_train=False):
        # [T, B, ...]
        # print(input_dict.keys())
        agent_map, entity_id, entity_in, rangeable, team_in, va_move, obs_emb, meleeable, \
        mask, magicable, local_map, entity_loc, is_attack, now_time, mine_loc = \
            input_dict["agent_map"], input_dict["entity_id"], input_dict["entity_in"], \
            input_dict["rangeable"], input_dict["team_in"], input_dict["va_move"], \
            input_dict["obs_emb"], input_dict["meleeable"], input_dict["mask"], \
            input_dict["magicable"], input_dict["local_map"], input_dict["entity_loc"], \
            input_dict["is_attack"], input_dict['now_time'], input_dict['mine_loc'].to(torch.float32)

        device = local_map.device
        T, B, H, W = local_map.shape
        # assert B==8, "{}".format(local_map.shape) # warn : B is batch contain
        local_map = F.one_hot(local_map, num_classes=6).permute(0, 1, 4, 2, 3)  # T B C H W
        local_map = torch.flatten(local_map, 0, 1)
        map_emb = torch.cat([agent_map.flatten(0, 1), local_map], dim=1)  # T*B C 15 15
        # print(map_emb.shape)

        time =  self.time_encoder(now_time.view(-1)).view(T, B//8, 8, -1)


        team_in = F.one_hot(team_in, num_classes=17).flatten(0, 1)   # T*B 100 17
        entity_in = F.one_hot(entity_in, num_classes=9).flatten(0, 1)  # T*B 100 9
        obs_emb = obs_emb.flatten(0, 1)  # T*B 100 44
        entity_emb = torch.cat([obs_emb, entity_in, team_in], dim=2)   # T*B 100 70
        # print(entity_emb.shape)
        # print(entity_emb[0, -5:, :])
        mask = mask.flatten(0, 1)
        entity_emb = self.unit_nn(entity_emb)
        # print(entity_emb.shape)
        entity_emb = self.unit_transformer(entity_emb, src_key_padding_mask=mask)

        entity_embeddings = self.entity_fc(self.act(entity_emb))   # 8 100 48
        # print(entity_embeddings.shape)

        # print(torch.sum(~mask, dim=-1))
        # print(mask.shape, entity_emb.shape)
        entity_emb_mask = entity_emb * (~mask.unsqueeze(dim=2))
        # print(entity_emb_mask[0, -5:, :])  # 全0
        # print(entity_emb_mask.sum(dim=1).shape, torch.sum(~mask, dim=-1).unsqueeze(dim=-1).shape)   # 8,48  8,1
        embedded_entity = entity_emb_mask.sum(dim=1) / torch.sum(~mask, dim=-1).unsqueeze(dim=-1)
        # print(embedded_entity.shape)   # 8，48
        embedded_entity = self.embed_fc(embedded_entity).view(T, B//8, 8, -1)

        entity_embeddings_mask = entity_embeddings * (~mask.unsqueeze(dim=2))

        mine_entity = obs_emb[:, 0, :]
        mine_entity = self.unit_team(mine_entity).view(T, B//8, 1, 8, -1)  # 8 46
        team_entity_ = mine_entity.repeat_interleave(8, dim=-3) # t b//8 8 8 46
        team_loc_ = mine_loc.view(T, B//8, 1, 8, -1).repeat_interleave(8, dim=-3) - mine_loc.view(T, B//8, 8, 1, -1).repeat_interleave(8, dim=-2)
        team_loc_ /= 100
        team_entity = torch.cat(
            [team_entity_, team_loc_, torch.eye(8, device=device).unsqueeze(2).repeat(T, B//8, 1, 1, 1)],
            dim=-1).view(-1, 8, 48)
        team_entity = self.team_transformer(team_entity).view(T, B//8, 8, -1)

        scatter_map = scatter_connection(entity_embeddings_mask, entity_loc.flatten(0, 1))
        embedded_spatial = self.spatialEncoder(map_emb, scatter_map).view(T, B//8, 8, -1)
        # print(embedded_entity.shape, embedded_spatial.shape)  # 8 64  8 256



        # lstm_input_before = torch.cat([time, embedded_entity, embedded_spatial, team_entity], dim=-1)  # noteblty
        lstm_input_before = torch.cat([embedded_entity, embedded_spatial], dim=-1)
        lstm_input_ = self.mix_fc(lstm_input_before)
        # print(lstm_input_.shape)
        lstm_input_p1, lstm_input_p2 = lstm_input_[:, :, :, :208], lstm_input_[:, :, :, 208:]
        lstm_input_p1 = torch.max(lstm_input_p1, 2)[0].unsqueeze(2).repeat(1, 1, 8, 1)
        lstm_input = torch.cat([lstm_input_p1, lstm_input_p2, time, team_entity], dim=-1).view(T, B, 1024)

        lstm_output, out_state = self.core_lstm(lstm_input, state)

        baseline = self.baseline(lstm_output)
        baseline = baseline.view(T, B)

        if is_train:
            dist_move, dis_type, \
            dis_not_attack_unit, dis_melee_attack_unit, dis_range_attack_unit, dis_magic_attack_unit, \
            action_move, action_type, action_unit_id = \
                self.policy(lstm_output.flatten(0, 1), entity_embeddings_mask, entity_id,
                            rangeable, meleeable, magicable, mask, va_move, is_attack, is_train)
            dist_move = dist_move.view(T, B, -1)
            dis_type = dis_type.view(T, B, -1)
            dis_not_attack_unit = dis_not_attack_unit.view(T, B, -1)
            dis_melee_attack_unit = dis_melee_attack_unit.view(T, B, -1)
            dis_range_attack_unit = dis_range_attack_unit.view(T, B, -1)
            dis_magic_attack_unit = dis_magic_attack_unit.view(T, B, -1)
            action_move = action_move.view(T, B)
            action_type = action_type.view(T, B)
            action_unit_id = action_unit_id.view(T, B)
            return (dict(dist_move=dist_move,
                         dis_type=dis_type,
                         dis_not_attack_unit=dis_not_attack_unit,
                         dis_melee_attack_unit=dis_melee_attack_unit,
                         dis_range_attack_unit=dis_range_attack_unit,
                         dis_magic_attack_unit=dis_magic_attack_unit,
                         action_move=action_move,
                         action_type=action_type,
                         action_unit_id=action_unit_id,
                         baseline=baseline),
                    out_state)
        else:
            dist_move, dis_type, dis_unit, action_move, action_type, action_unit_id = \
                self.policy(lstm_output.flatten(0, 1), entity_embeddings_mask, entity_id,
                            rangeable, meleeable, magicable, mask, va_move, is_attack, is_train)

            dist_move = dist_move.view(T, B, -1)
            dis_type = dis_type.view(T, B, -1)
            dis_unit = dis_unit.view(T, B, -1)
            action_move = action_move.view(T, B)
            action_type = action_type.view(T, B)
            action_unit_id = action_unit_id.view(T, B)
            # action = torch.multinomial(F.softmax(policy_logits, dim=1),
            #                            num_samples=1)
            # policy_logits = policy_logits.view(T, B, -1)
            # action = action.view(T, B)
            return (dict(dist_move=dist_move,
                         dis_type=dis_type,
                         dis_unit=dis_unit,
                         action_move=action_move,
                         action_type=action_type,
                         action_unit_id=action_unit_id,
                         baseline=baseline),
                    out_state)
