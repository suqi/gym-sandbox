# -*- coding: utf-8 -*-
from bokeh.io import output_notebook, show, push_notebook
from bokeh.plotting import figure
from bokeh.layouts import gridplot
import numpy as np
import random
import time

# ! only used to temporarily shutdown bokeh warning !
import warnings
warnings.filterwarnings('ignore')

# TODO: solve warning below
# \site-packages\bokeh\models\sources.py:89: BokehUserWarning: ColumnDataSource's columns must be of the same length
#  lambda: warnings.warn("ColumnDataSource's columns must be of the same length", BokehUserWarning))

# constans for render, change it as you like
POLICE_RADIUS = 0.05  # percent of map size
THIEF_RADIUS = 0.02


class BallsGameDashboard:
    """A game dashboard to show the game state"""

    def __init__(self, map_size, team_size):
        self.map_size = map_size
        self.team_size = team_size
        self.global_running_r = []

        output_notebook()
        _x_min, _y_min, _x_max, _y_max = 0, 0, map_size, map_size
        plt_loc = figure(
            plot_width=600,
            plot_height=600,
            # toolbar_location=None,
            # slightly show 10% more of boundary
            x_range=(_x_min-map_size/10, _x_max+map_size/10),
            y_range=(_y_max+map_size/10, _y_min-map_size/10),
            x_axis_location="above",
            title="step #0")  # use up-left corner as origin
        plt_loc.title.align = "center"
        plt_loc.title.text_color = "orange"
        plt_loc.title.text_font_size = "25px"
        plt_loc.title.background_fill_color = "blue"
        self.plt_loc = plt_loc  # 用于后续更新边界和标题中的距离显示

        # draw edge
        plt_loc.line(x=[_x_min, _x_max, _x_max, _x_min, _x_min],
                     y=[_y_min, _y_min, _y_max, _y_max, _y_min],
                     line_color="navy", line_alpha=0.3, line_dash="dotted", line_width=2)

        self.police_num, self.thief_num = team_size['police'], 1000 #team_size['thief'] # TODO: must give a biggest num
        self.total_num = self.police_num + self.thief_num

        # draw balls
        self.rd_loc = plt_loc.circle(
            [-1] * self.total_num, [-1] * self.total_num,
            radius=[POLICE_RADIUS*map_size] * self.police_num + [THIEF_RADIUS*map_size] * self.thief_num,  # radius is by percent
            # size=[50]*self.police_num + [20]*self.thief_num,  # size is px
            line_color="gold",
            line_width=[10]*self.police_num + [1]*self.thief_num,
            fill_color=["green"]*self.police_num + ["yellow"]*self.thief_num,
            fill_alpha=0.6)

        # 显示reward趋势
        plt_reward = figure(
            plot_width=400, plot_height=400, title="running ep reward: ")
        plt_reward.title.align = "center"
        plt_reward.title.text_color = "green"
        plt_reward.title.text_font_size = "20px"
        plt_reward.title.background_fill_color = "black"
        self.plt_reward = plt_reward  # 用于后续更新标题中的reward值
        self.rd_reward = plt_reward.line(
            [1], [10], line_width=2)

        # put all the plots in a gridplot
        plt_combo = gridplot(
            [[plt_loc, plt_reward]],
            # toolbar_location=None
        )

        # show the results
        self.nb_handle = show(plt_combo, notebook_handle=True)

    def update_plots(self, env_state_action):
        """update bokeh plots according to new env state and action data"""
        global_ob, rewards, ep_count, current_step, current_is_caught, current_done = env_state_action
        self.police_num = len(global_ob['police'])
        self.thief_num = len(global_ob['thief'])

        # eucl_dist = self.calc_eucl_dist((location['target_x'], location['target_y']), (location['me_x'], location['me_y']) )
        self.plt_loc.title.text = "step: #{}".format(current_step)

        # note： 如果频率过快， jupyter notebook会受不了
        all_x = [_loc[0] for _loc in global_ob['police']] + [_loc[0] for _loc in global_ob['thief']]
        all_y = [_loc[1] for _loc in global_ob['police']] + [_loc[1] for _loc in global_ob['thief']]
        self.rd_loc.data_source.data['x'] = all_x
        self.rd_loc.data_source.data['y'] = all_y

        # 游戏结束时进行闪动， 表示游戏结束
        thief_color = "red" if current_is_caught else "yellow"
        self.rd_loc.data_source.data['fill_color'] = ["green"] * self.police_num + [thief_color] * self.thief_num
        thief_lw = 3 if current_is_caught else 1
        self.rd_loc.data_source.data['line_width'] = [10] * self.police_num + [thief_lw] * self.thief_num

        if current_done:
            ep_reward = sum(rewards)
            if len(self.global_running_r) == 0:  # record running episode reward
                self.global_running_r.append(ep_reward)
            else:
                self.global_running_r.append(0.99 * self.global_running_r[-1] + 0.01 * ep_reward)

            self.rd_reward.data_source.data['x'] = range(len(self.global_running_r))
            self.rd_reward.data_source.data['y'] = self.global_running_r
            self.plt_reward.title.text = "episode #{} / last_ep_reward: {:5.1f}".format(
                ep_count, self.global_running_r[-1])

        push_notebook()  # self.nb_handle

    def calc_eucl_dist(self, pos1, pos2):
        # calc Euclidean Distance
        _coords1 = np.array(pos1)  # location of me
        _coords2 = np.array(pos2)
        # alternative way: np.linalg.norm(_coords1 - _coords2)
        eucl_dist = np.sqrt(np.sum((_coords1 - _coords2) ** 2))
        return eucl_dist