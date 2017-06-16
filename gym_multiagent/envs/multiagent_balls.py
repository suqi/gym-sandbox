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

    WIN_REWARD = 300
    LOSE_REWARD = -300

    MAX_DIST = 800
    MAX_ROUND_COUNT = 200

    MIN_CATCH_DIST = 20

    TEAMS = {
        "police": {"speed": 5},
        "thief": {"speed": 15},
    }

    def __init__(self,  agent_num=5, agent_team="police", adversary_num=2, map_size=200):
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

        self.episode_count = 0
        self.round_count = 0
        self.last_state = None
        self.current_state = None
        self.current_action = None
        self.reward_hist = []

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


    # def _trans_state(self, state):
    #     result = list()
    #     if state.player and state.npc:
    #         result.append(state.player.base.x - state.npc[0].x)
    #         result.append(state.player.base.y - state.npc[0].y)
    #         result.append(state.player.base.hp)
    #         result.append(state.player.base.hp_m)
    #         result.append(state.npc[0].hp)
    #         result.append(state.npc[0].hp_m)
    #     else:
    #         result.extend([0, 0, 0, 0, 0, 0])
    #     return np.array(result)

    def _cal_reward(self, is_thief_caught):
        reward = 0

        if is_thief_caught:
            reward = self.WIN_REWARD
        # else:
        #     reward = self.LOSE_REWARD

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
        self.global_ob = {_t: [(random.randint(1, self.map_size), random.randint(1, self.map_size))
                               for _ in range(self.team_size[_t])]
                            for _t in self.TEAMS.keys()}
        self.current_state = self.global_ob  # todo: needs to split ob for each agent
        self.round_count = 0
        self.episode_count += 1
        self.reward_hist = []

        return self.current_state

    def _close(self):
        pass

    def _step(self, action):
        """action in MA env must be a list of actions for each agent"""
        int_action = int(action)

        final_state = self.current_state.copy()


        # the target will run out of me as far as possible
        # 要么走x， 要么走y，先看边界，再选距离， 上下左右
        thief_list = self.current_state['thief']
        police_list = self.current_state['police']
        thief_new_loc = [self._take_simple_action(_thief, police_list, team="thief") for _thief in thief_list]

        # samely, for me, run to get more close to target
        # TODO: action should be implemented outside
        police_new_loc = [self._take_simple_action(_police, thief_list, team="police") for _police in police_list]

        final_state['thief'] = thief_new_loc
        final_state['police'] = police_new_loc

        self.last_state = self.current_state
        self.current_state = final_state
        # self.current_action = step_action
        self.round_count += 1


        is_caught = self.is_thief_caught()

        return final_state, self._cal_reward(is_caught), self._cal_done(final_state, is_caught), None

    def is_thief_caught(self):
        # police win when any thief is caught
        thief_list = self.current_state['thief']
        police_list = self.current_state['police']

        for _thief in thief_list:
            for _police in police_list:
                if self.calc_dist(_thief, _police) < self.MIN_CATCH_DIST:
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

        env_data = [self.current_state, self.reward_hist, self.episode_count]
        if self.game_dashboard:
            self.game_dashboard.update_plots(env_data)
        return


    def get_position_rating(self, my_new_pos, adversary_list):
        all_dist = [self.calc_dist(my_new_pos, _ad) for _ad in adversary_list]
        return sum(all_dist)

    # TODO: actually we should use manhatton dist
    def calc_dist(self, pos1, pos2):
        # calc Euclidean Distance
        _coords1 = np.array(pos1)  # location of me
        _coords2 = np.array(pos2)
        # alternative way: np.linalg.norm(_coords1 - _coords2)
        eucl_dist = np.sqrt(np.sum((_coords1 - _coords2) ** 2))
        return eucl_dist
