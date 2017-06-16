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


class BallsGameDashboard:
    """A game dashboard to show the game state"""

    def __init__(self, map_size, team_size):
        self.map_size = map_size
        self.team_size = team_size

        output_notebook()
        min_screen_x, min_screen_y, max_screen_x, max_screen_y = \
            0, 0, map_size, map_size
        plt_loc = figure(
            plot_width=600,
            plot_height=600,
            toolbar_location=None,
            x_range=(min_screen_x-10, max_screen_x+10),
            y_range=(max_screen_y+10, min_screen_y-10),
            x_axis_location="above",
            title="敌我距离")  # use up-left corner as origin
        plt_loc.title.align = "center"
        plt_loc.title.text_color = "orange"
        plt_loc.title.text_font_size = "25px"
        plt_loc.title.background_fill_color = "blue"

        self.police_num, self.thief_num = team_size['police'], team_size['thief']
        self.total_num = self.police_num + self.thief_num

        self.plt_loc = plt_loc  # 用于后续更新边界和标题中的距离显示
        self.rd_loc = plt_loc.circle(
            [-1] * self.total_num, [-1] * self.total_num,
            size=20,
            line_color="gold",
            fill_color=["green"]*self.police_num + ["firebrick"]*self.thief_num,
            fill_alpha=0.6)

        # 显示reward趋势
        # plt_reward = figure(
        #     plot_width=400, plot_height=400, title="last reward: ")
        # plt_reward.title.align = "center"
        # plt_reward.title.text_color = "green"
        # plt_reward.title.text_font_size = "20px"
        # plt_reward.title.background_fill_color = "black"
        # self.plt_reward = plt_reward  # 用于后续更新标题中的reward值
        # self.rd_reward = plt_reward.line(
        #     [1, 2, 3, 4, 5], [6, 7, 2, 4, 5], line_width=2)
        #
        # # put all the plots in a gridplot
        # plt_combo = gridplot(
        #     [[plt_loc, plt_hp], [plt_action, plt_reward]],
        #     toolbar_location=None)

        # show the results
        self.nb_handle = show(plt_loc, notebook_handle=True)

    def update_plots(self, env_state_action):
        """update bokeh plots according to new env state and action data"""
        global_ob, reward, episode_count = env_state_action

        # eucl_dist = self.calc_eucl_dist((location['target_x'], location['target_y']), (location['me_x'], location['me_y']) )
        # self.plt_loc.title.text = "敌我距离: {:12.2f}".format(eucl_dist)
        # self.plt_loc.x_range.start = location["min_screen_x"]
        # self.plt_loc.x_range.end = location["max_screen_x"]
        # self.plt_loc.y_range.start = location["max_screen_y"]
        # self.plt_loc.y_range.end = location["min_screen_y"]

        all_x = [_loc[0] for _loc in global_ob['police']] + [_loc[0] for _loc in global_ob['thief']]
        all_y = [_loc[1] for _loc in global_ob['police']] + [_loc[1] for _loc in global_ob['thief']]

        self.rd_loc.data_source.data['x'] = all_x
        self.rd_loc.data_source.data['y'] = all_y

        # self.rd_reward.data_source.data['x'] = range(len(reward))
        # self.rd_reward.data_source.data['y'] = reward
        # self.plt_reward.title.text = "episode #{} / step #{} / reward: {:5.1f}".format(
        #     episode_count, len(reward), reward[-1] if reward else 0)
        push_notebook()  # self.nb_handle

    def calc_eucl_dist(self, pos1, pos2):
        # calc Euclidean Distance
        _coords1 = np.array(pos1)  # location of me
        _coords2 = np.array(pos2)
        # alternative way: np.linalg.norm(_coords1 - _coords2)
        eucl_dist = np.sqrt(np.sum((_coords1 - _coords2) ** 2))
        return eucl_dist
