import gym
import gym.spaces
import numpy as np
import random

from gym_multiagent.envs.plot import balls_game_dashboard


class MABallsEnv(gym.Env):
    """A very simple balls game to demo MA algo
    Attention:
    1. in multi agent env, all function should return a list of actions/states/rewards
    2. the env has a fully observable internal state, while agent only has a partially observable state
    """
    metadata = {'render.modes': ['human', 'rgb_array']}

    WIN_REWARD = 1
    LOSE_REWARD = -1

    MAX_ROUND_COUNT = 100

    MIN_CATCH_DIST = 3

    TEAMS = {
        "police": {"speed": 2},
        "thief": {"speed": 1},
    }

    def __init__(self,  agent_num=5, agent_team="police", adversary_num=2, map_size=200, adversary_static=True):
        """the init params should be passed in by code of env registering """
        self.game_dashboard = None

        self.map_size = map_size
        self.adversary_team = "thief" if agent_team == 'police' else 'police'
        self.agent_num = agent_num
        self.agent_team = agent_team
        self.adversary_num = adversary_num
        self.team_size = {
            self.agent_team: agent_num,
            self.adversary_team: adversary_num
        }
        self.adversary_static = adversary_static

        self.episode_count = 0
        self.round_count = 0
        self.last_state = None
        self.current_state = None
        self.current_action = None
        self.reward_hist = []
        self.running_ep_r = []
        self.current_is_caught = False # 用来在渲染的时候展示被抓住了

        # observation will be a list of agent_num
        # self.observation_space = gym.spaces.Box(
        #     float("-inf"), float("inf"), (self.STATE_DIM,))
        #
        # # {move_up, move_right, move_left, move_down,}
        # self.action_space = gym.spaces.Discrete(self.ACTION_DIM)

    def init_params(self, show_dashboard=True):
        """You can be in police/thief team (警察抓小偷)
        为了简化， 先用正方形即可， 因为跟长方形没有区别
        """
        self.game_dashboard = balls_game_dashboard.BallsGameDashboard(
            self.map_size, self.team_size) if show_dashboard else None


    def _trans_state(self, state):
        result = list()

        # 1v1 observation
        result.extend(np.array(state["police"][0])/self.map_size)
        result.extend(np.array(state["thief"][0])/self.map_size)

        return np.array(result)

    def _cal_reward(self, is_thief_caught, is_done):
        # if is_thief_caught:
        #     reward = self.WIN_REWARD
        # else:
        #     if is_done:
        #         reward = self.LOSE_REWARD
        #     else:
        #         reward = 0
        reward = -1  # 借鉴MountainCar的reward思路， 不抓到小偷， 一直给-1， 这样总体reward会体现出抓到的快慢

        self.reward_hist.append(reward)

        return reward

    def _cal_dist(self):
        x_dist = self.current_state.player.base.x - self.current_state.npc[0].x
        y_dist = self.current_state.player.base.y - self.current_state.npc[0].y
        return np.sqrt(np.square(x_dist) + np.square(y_dist))

    def _cal_done(self, state, is_thief_caught):
        is_exceed_max_round = self.round_count > self.MAX_ROUND_COUNT
        if is_exceed_max_round or is_thief_caught:
            return True

        return False

    def _reset(self):
        # global observation from god's view
        _map_center = int(self.map_size / 2)
        _police_radius = int(self.map_size * 0.2)
        _police_range = _map_center-_police_radius, _map_center+_police_radius
        _thief_radius = int(self.map_size * 0.4)
        _thief_range = _map_center - _thief_radius, _map_center + _thief_radius

        self.global_ob = {
            # 警察从地图中间出发， 让其需要随机向任意方向出发
            "police": [(random.randint(*_police_range), random.randint(*_police_range))
                       for _ in range(self.team_size["police"])],
            # 让小偷在地图四周
            "thief": [(random.choice(
                            [random.randint(1, _police_range[0]), random.randint(_police_range[1], self.map_size)]),
                       random.choice(
                            [random.randint(1, _thief_range[0]), random.randint(_thief_range[1], self.map_size)]))
                       for _ in range(self.team_size["thief"])],

        }
        self.current_state = self.global_ob  # todo: needs to split ob for each agent
        self.round_count = 0
        self.episode_count += 1
        self.current_is_caught = False

        last_ep_r = sum(self.reward_hist)
        self.reward_hist = []
        if len(self.running_ep_r) == 0:  # record running episode reward
            self.running_ep_r.append(last_ep_r)
        else:
            self.running_ep_r.append(0.99 * self.running_ep_r[-1] + 0.01 * last_ep_r)

        ob = self._trans_state(self.global_ob)

        return ob

    def _close(self):
        pass

    def _step(self, action):
        """action in MA env must be a list of actions for each agent"""
        # int_action = int(action)  # 上下左右 0123

        final_state = self.current_state.copy()


        # the target will run out of me as far as possible
        # 要么走x， 要么走y，先看边界，再选距离， 上下左右
        thief_list = self.current_state['thief']
        police_list = self.current_state['police']

        # 1. 原地不动的小偷
        if self.adversary_static:
            thief_new_loc = thief_list
        # 2. 会跑的小偷
        else:
            thief_new_loc = [self._take_simple_action(_thief, police_list, team="thief") for _thief in thief_list]


        # samely, for me, run to get more close to target
        # police_new_loc = [self._take_simple_action(_police, thief_list, team="police") for _police in police_list]

        directions = [[0, -1], [0, 1], [-1, 0], [1, 0]]  # 上下左右
        action_dir = np.array(directions[action])

        police_speed = self.TEAMS['police']['speed']
        police_dir = action_dir * police_speed

        p1 = police_list[0]
        p1 = (p1[0] + police_dir[0], p1[1] + police_dir[1])
        p1 = self.ensure_inside(p1)
        police_new_loc = [p1]

        final_state['thief'] = thief_new_loc
        final_state['police'] = police_new_loc

        self.last_state = self.current_state
        self.current_state = final_state
        # self.current_action = step_action
        self.round_count += 1

        is_caught = self.is_thief_caught()
        self.current_is_caught = is_caught

        ob = self._trans_state(final_state)

        is_done = self._cal_done(final_state, is_caught)
        self.current_is_caught = is_caught

        reward = self._cal_reward(is_caught, is_done)

        return ob, reward, is_done, None

    def ensure_inside(self, cord):
        x, y = cord
        x = self.map_size if x > self.map_size else 0 if x < 0 else x
        y = self.map_size if y > self.map_size else 0 if y < 0 else y
        return (x,y)


    def is_thief_caught(self):
        # police win when any thief is caught
        thief_list = self.current_state['thief']
        police_list = self.current_state['police']

        for _thief in thief_list:
            for _police in police_list:
                if self.calc_dist(_thief, _police) <= self.MIN_CATCH_DIST:
                    return True

        return False

    def _take_simple_action(self, my_pos, adversary_list, team="thief"):
        """team can be"""
        x, y = my_pos
        my_speed = self.TEAMS[team]['speed']
        available_direction = [
            y - my_speed > 0,
            y + my_speed < self.map_size,
            x - my_speed > 0,
            x + my_speed < self.map_size,
        ]
        new_location = [
            (x, y - my_speed),
            (x, y + my_speed),
            (x - my_speed, y),
            (x + my_speed, y),
        ]

        available_loc = [_l for i, _l in enumerate(new_location) if available_direction[i]]

        new_dist = [self.get_position_rating(my_new_pos, adversary_list) for my_new_pos in available_loc]

        func = max if team == "thief" else min
        best_choice = func(enumerate(new_dist), key=lambda x: x[1])[0]
        my_final_pos = available_loc[best_choice]
        return my_final_pos

    def _render(self, mode='human', close=False):
        if not self.current_state:
            return

        env_data = [self.current_state, self.running_ep_r, self.episode_count,
                    len(self.reward_hist), self.current_is_caught]
        if self.game_dashboard:
            self.game_dashboard.update_plots(env_data)
        return


    def get_position_rating(self, my_new_pos, adversary_list):
        all_dist = [self.calc_dist(my_new_pos, _ad) for _ad in adversary_list]
        return sum(all_dist)

    # TODO: actually we should use manhatton dist
    def calc_dist(self, pos1, pos2):
        _coords1 = np.array(pos1)  # location of me
        _coords2 = np.array(pos2)

        # calc manhatton dist
        return sum(abs(_coords1 - _coords2))

        # # calc Euclidean Distance
        # # alternative way: np.linalg.norm(_coords1 - _coords2)
        # eucl_dist = np.sqrt(np.sum((_coords1 - _coords2) ** 2))
        # return eucl_dist
