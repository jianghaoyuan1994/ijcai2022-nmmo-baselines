import nmmo
import numpy as np
from gym import Wrapper, spaces
from ijcai2022nmmo import TeamBasedEnv
from ijcai2022nmmo.scripted import CombatTeam, ForageTeam, RandomTeam
from ijcai2022nmmo.scripted.baselines import Scripted
from ijcai2022nmmo.scripted.scripted_team import ScriptedTeam


class FeatureParser:  # 环境obs解析
    map_size = 15
    n_move_actions = 5  # 0不动 1上 2下 3左 4右
    NEIGHBOR = [(6, 7), (8, 7), (7, 8), (7, 6)]  # north, south, east, west
    OBSTACLE = (0, 1, 5) # lava, water, stone
    n_attack_actions = 4
    feature_spec = {
        # "terrain": spaces.Box(low=0, high=6, shape=(15, 15), dtype=np.int64),
        # "camp": spaces.Box(low=0, high=4, shape=(15, 15), dtype=np.int64),
        # "entity": spaces.Box(low=0,
        #                      high=4,
        #                      shape=(7, 15, 15),
        #                      dtype=np.float32),
        # "va": spaces.Box(low=0, high=2, shape=(5, ), dtype=np.int64),
        "obs_emb": spaces.Box(low=-100, high=100, shape=(100, 14), dtype=np.float32),
        "local_map": spaces.Box(low=-100, high=100, shape=(15, 15), dtype=np.float32),
        "agent_map": spaces.Box(low=-100, high=100, shape=(1, 15, 15), dtype=np.float32),
        "mask": spaces.Box(low=0, high=1, shape=(100,), dtype=np.int64),
        "entity_loc": spaces.Box(low=-200, high=200, shape=(100, 2), dtype=np.int64),
        "entity_id": spaces.Box(low=0, high=200, shape=(100,), dtype=np.int64),
        "team_in": spaces.Box(low=0, high=17, shape=(100,), dtype=np.int64),
        "attack_id": spaces.Box(low=0, high=200, shape=(100,), dtype=np.int64),
        "entity_in": spaces.Box(low=0, high=17, shape=(100,), dtype=np.int64),
        "va_move": spaces.Box(low=0, high=200, shape=(5,), dtype=np.int64),
        "meleeable": spaces.Box(low=0, high=200, shape=(100,), dtype=np.int64),
        "rangeable": spaces.Box(low=0, high=200, shape=(100,), dtype=np.int64),
        "magicable": spaces.Box(low=0, high=200, shape=(100,), dtype=np.int64)
    }

    def __init__(self, feas_dim):

        self.map_size = feas_dim[0]
        # self.channel_num = 6  # im
        # self.onehot = np.eye(self.channel_num)  # todo move to model
        # self.onehot_team = np.eye(17)
        # self.onehot_index = np.eye(9)
        self.now_time = 0

    def parse(self, obs):

        frame_list = {}

        for entity in obs.keys():
            va_move = np.ones(self.n_move_actions, dtype=np.int64)
            # va_attack = np.zeros(self.n_attack_actions, dtype=np.int64)
            # va_attack_id = []
            obs_agents = obs[entity]["Entity"]["Continuous"]
            self.now_time = max(max(obs_agents[:, 8]), self.now_time)

            # agents_frame = obs_agents.reshape(-1) / np.linalg.norm(
            #     obs_agents.reshape(-1))   #TODO normalize

            obs_map = obs[entity]["Tile"]["Continuous"]
            local_map = np.zeros((self.map_size, self.map_size),
                                 dtype=np.float32)
            agent_map = np.zeros((1, self.map_size, self.map_size),
                                 dtype=np.float32)

            init_R = obs_map[0][2]  # 左上角loc
            init_C = obs_map[0][3]  # 左上角loc
            agent_locR = obs_map[112][2]
            agent_locC = obs_map[112][3]

            for line in obs_map:
                local_map[int(line[2] - init_R),
                          int(line[3] - init_C)] = int(line[1])
                if line[0] != 0:
                    agent_map[0][int(line[2] - init_R),
                                 int(line[3] - init_C)] = line[0]

            obs_num_part = np.zeros((100, 14), dtype="float32")
            index = 0
            entity_loc = []
            entity_id = []
            entity_in = []
            attack_id = []
            team_in = []
            magicable = []
            rangeable = []
            meleeable = []
            while index < 100:
                is_mine = 1 if index ==0 else 0
                value = obs_agents[index]
                if value[0] == 0:
                    break

                entity_id.append(value[1])

                entity_in_ = -1 if value[1] < 0 else value[1] % 16 -1  #与time
                entity_in.append(entity_in_)

                attack_id_ = -1 if value[2] < 0 else value[2] // 16 # todo 查看
                attack_id.append(attack_id_)

                level_in = value[3] / 10

                team_in_ = value[4] if value[4] > 0 else -1  # todo 查看
                team_in.append(team_in_)

                r_in = value[5]
                c_in = value[6]

                dr_in = (r_in-init_R)
                dc_in = (c_in-init_C)
                assert 0 <= dc_in < 15 and 0 <= dr_in < 15, "{}-{}-{}-{}-{}".format(r_in, init_R, c_in, init_C, obs[entity])
                entity_loc.append([dc_in, dc_in])

                # 攻击距离
                if team_in_ == 0:
                    is_attack = np.zeros(3)
                else:
                    min_d = min(abs(r_in-agent_locR), abs(c_in-agent_locC))
                    if min_d <= 4:
                        magic_attack = 1
                        magicable.append(value[1])
                    else:
                        magic_attack = 0

                    if min_d <= 3:
                        range_attack = 1
                        rangeable.append(value[1])
                    else:
                        range_attack = 0

                    if min_d <= 1:
                        melee_attack = 1
                        meleeable.append(value[1])
                    else:
                        melee_attack = 0

                    is_attack = np.array([melee_attack, range_attack, magic_attack])


                dc_in /= 10
                dr_in /= 10
                r_in = (r_in - 64) / 128
                c_in = (c_in - 64) / 128
                alive_in = 0 if value[8] < self.now_time else 1
                food_in = value[9] / 10   # 最大与等级相等
                water_in = value[10] / 10  # 最大与等级相等
                health_in = value[11] / 10  # 最大与等级相等  todo Add relative value of integral
                freezed_in = value[12]   # todo 加入目前freezed了多久
                obs_num_part[index, :] = np.hstack([level_in, r_in, c_in, dr_in, dc_in, is_mine,
                                                    alive_in, food_in, water_in, health_in, is_attack,
                                                    freezed_in])

                index += 1

            mask = np.array([0 if i < index else 1 for i in range(100)], dtype="bool")
            entity_loc = np.pad(np.array(entity_loc, dtype='int'), (0, 100-len(entity_loc)), constant_values=101)
            entity_id = np.pad(np.array(entity_id, dtype='int'), (0, 100-len(entity_id)), constant_values=101)
            team_in = np.pad(np.array(team_in, dtype='int'), (0, 100-len(team_in)), constant_values=101)
            attack_id = np.pad(np.array(attack_id, dtype='int'), (0, 100-len(attack_id)), constant_values=101)
            entity_in = np.pad(np.array(entity_in, dtype='int'), (0, 100-len(entity_in)), constant_values=101)

            meleeable = np.pad(np.array(meleeable, dtype='int'), (0, 100-len(meleeable)), constant_values=101)
            rangeable = np.pad(np.array(rangeable, dtype='int'), (0, 100-len(rangeable)), constant_values=101)
            magicable = np.pad(np.array(magicable, dtype='int'), (0, 100-len(magicable)), constant_values=101)

            # map_frame = np.concatenate([local_map, agent_map])

            # valid action
            for i, (r, c) in enumerate(self.NEIGHBOR):
                if local_map[r, c] in self.OBSTACLE:
                    va_move[i + 1] = 0

            frame_list[entity] = {
                "obs_emb": obs_num_part,
                "local_map": local_map,
                "agent_map": agent_map,
                "mask": mask,
                "entity_loc": entity_loc,
                "entity_id": entity_id,
                "team_in": team_in,
                "attack_id": attack_id,
                "entity_in": entity_in,
                "va_move": va_move,
                "meleeable": meleeable,
                "rangeable": rangeable,
                "magicable": magicable
            }

        return frame_list


class RewardParser:

    def parse(self, prev_achv, achv):
        reward = {
            i: (sum(achv[i].values()) - sum(prev_achv[i].values())) / 100.0
            for i in achv
        }
        return reward


class TrainWrapper(Wrapper):
    max_step = 1024
    TT_ID = 0  # training team index
    use_auxiliary_script = False

    def __init__(self, env: TeamBasedEnv) -> None:
        super().__init__(env)
        self.feature_parser = FeatureParser(feas_dim=[15])
        self.reward_parser = RewardParser()
        self.observation_space = spaces.Dict(self.feature_parser.feature_spec)
        self.action_space = spaces.Discrete(5)

        self._dummy_feature = {
            key: np.zeros(shape=val.shape, dtype=val.dtype)
            for key, val in self.observation_space.items()
        }

    # def reward(self, player):  # todo explore reward and https://neuralmmo.github.io/build/html/rst/tutorial.html#icon-rewards-tasks
    #     # Default survival reward
    #     reward, info = super().reward(player)
    #
    #     # Inject exploration attribute into player
    #     if not hasattr(player, 'exploration'):
    #         player.exploration = 0
    #
    #     # Historical exploration already part of player state
    #     exploration = player.history.exploration
    #
    #     # Only reward agents for distance increment
    #     # over previous farthest exploration
    #     if exploration > player.exploration:
    #         reward += 0.05 * (exploration - player.exploration)
    #
    #     return reward, info

    def reset(self):
        raw_obs = super().reset()
        obs = raw_obs[self.TT_ID]
        obs = self.feature_parser.parse(obs)

        self.reset_auxiliary_script(self.config)
        self.reset_scripted_team(self.config)
        self.agents = list(obs.keys())
        self._prev_achv = self.metrices_by_team()[self.TT_ID]
        self._prev_raw_obs = raw_obs
        self._step = 0

        return obs

    def step(self, actions):
        decisions = self.get_scripted_team_decision(self._prev_raw_obs)  # todo  联盟学习and self-play
        decisions[self.TT_ID] = self.transform_action(
            actions,
            observations=self._prev_raw_obs[self.TT_ID],
            auxiliary_script=self.auxiliary_script)

        raw_obs, _, raw_done, raw_info = super().step(decisions)
        if self.TT_ID in raw_obs:
            obs = raw_obs[self.TT_ID]
            done = raw_done[self.TT_ID]
            info = raw_info[self.TT_ID]

            obs = self.feature_parser.parse(obs)
            achv = self.metrices_by_team()[self.TT_ID]
            reward = self.reward_parser.parse(self._prev_achv, achv)
            self._prev_achv = achv
        else:
            obs, reward, done, info = {}, {}, {}, {}

        for agent_id in self.agents:
            if agent_id not in obs:
                obs[agent_id] = self._dummy_feature
                reward[agent_id] = 0
                done[agent_id] = True

        self._prev_raw_obs = raw_obs
        self._step += 1

        if self._step >= self.max_step:
            done = {key: True for key in done.keys()}
        return obs, reward, done, info

    def reset_auxiliary_script(self, config):
        if not self.use_auxiliary_script:
            self.auxiliary_script = None
            return
        if getattr(self, "auxiliary_script", None) is not None:
            self.auxiliary_script.reset()
            return
        self.auxiliary_script = AttackTeam("auxiliary", config)


    def reset_scripted_team(self, config):
        if getattr(self, "_scripted_team", None) is not None:
            for team in self._scripted_team.values():
                team.reset()
            return
        self._scripted_team = {}
        assert config.NPOP == 16
        for i in range(config.NPOP):
            if i == self.TT_ID:
                continue
            if self.TT_ID < i <= self.TT_ID + 7:
                self._scripted_team[i] = RandomTeam(f"random-{i}", config)
            elif self.TT_ID + 7 < i <= self.TT_ID + 12:
                self._scripted_team[i] = ForageTeam(f"forage-{i}", config)
            elif self.TT_ID + 12 < i <= self.TT_ID + 15:
                self._scripted_team[i] = CombatTeam(f"combat-{i}", config)

    def get_scripted_team_decision(self, observations):
        decisions = {}
        tt_id = self.TT_ID
        for team_id, obs in observations.items():
            if team_id == tt_id:
                continue
            decisions[team_id] = self._scripted_team[team_id].act(obs)
        return decisions

    @staticmethod
    def transform_action(actions, observations=None, auxiliary_script=None):
        """neural network move + scripted attack"""
        decisions = {}

        # move decisions
        for agent_id, val in actions.items():
            if observations is not None and agent_id not in observations:
                continue
            if val == 0:
                decisions[agent_id] = {}
            elif 1 <= val <= 4:
                decisions[agent_id] = {
                    nmmo.action.Move: {
                        nmmo.action.Direction: val - 1
                    }
                }
            else:
                raise ValueError(f"invalid action: {val}")

        # attack decisions
        if auxiliary_script is not None:
            assert observations is not None
            attack_decisions = auxiliary_script.act(observations)
            # merge decisions
            for agent_id, d in decisions.items():
                d.update(attack_decisions[agent_id])
                decisions[agent_id] = d
        return decisions


class Attack(Scripted):
    '''attack'''
    name = 'Attack_'

    def __call__(self, obs):
        super().__call__(obs)

        self.scan_agents()
        self.target_weak()
        self.style = nmmo.action.Range
        self.attack()
        return self.actions


class AttackTeam(ScriptedTeam):
    agent_klass = Attack


if __name__ == "__main__":
    import time

    from ijcai2022nmmo import CompetitionConfig
    env = TrainWrapper(TeamBasedEnv(config=CompetitionConfig()))
    for i in range(10):
        start, step = time.time(), 0
        env.reset()
        episode_return = {agent_id: 0 for agent_id in env.agents}
        while True:
            actions = {
                agent_id: env.action_space.sample()
                for agent_id in env.agents
            }
            obs, reward, done, info = env.step(actions)
            step += 1
            for agent_id, rew in reward.items():
                episode_return[agent_id] += rew
            if all(done.values()):
                break
        print(
            f"episode {i}, total step: {step}, episode return: {episode_return}, elapsed: {time.time() - start}"
        )
