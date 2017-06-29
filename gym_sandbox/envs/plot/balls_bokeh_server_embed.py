# -*- coding: utf-8 -*-
from bokeh.application.handlers import FunctionHandler
from bokeh.application import Application
from bokeh.layouts import column
from bokeh.models import ColumnDataSource, Slider
from bokeh.plotting import figure
from bokeh.server.server import Server
from bokeh.themes import Theme
from bokeh.plotting import figure
from bokeh.layouts import gridplot

from tornado.ioloop import IOLoop
import yaml
import threading
from functools import partial

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


class BallsBokehServerEmbed:
    """A game dashboard to show the game state
    Using bokeh server running in a separate thread
    using a standalone embed solution: https://github.com/bokeh/bokeh/blob/0.12.6/examples/howto/server_embed/standalone_embed.py
    """

    def __init__(self, map_size, team_size):
        self.map_size = map_size
        self.team_size = team_size
        self.global_running_r = []

        # TODO: currently we are using threading, use process or other later
        t = threading.Thread(target=self.start_server)
        t.start()

    def modify_doc(self, doc):
        map_size = self.map_size
        team_size = self.team_size

        _x_min, _y_min, _x_max, _y_max = 0, 0, map_size, map_size
        plt_loc = figure(
            plot_width=600,
            plot_height=600,
            # toolbar_location=None,
            # slightly show 10% more of boundary
            x_range=(_x_min - map_size / 10, _x_max + map_size / 10),
            y_range=(_y_max + map_size / 10, _y_min - map_size / 10),
            x_axis_location="above",
            title="step #0")  # use up-left corner as origin
        plt_loc.title.align = "center"
        plt_loc.title.text_color = "orange"
        plt_loc.title.text_font_size = "25px"
        plt_loc.title.background_fill_color = "blue"
        self.plt_loc = plt_loc  # for later update of title

        # draw edge
        plt_loc.line(x=[_x_min, _x_max, _x_max, _x_min, _x_min],
                     y=[_y_min, _y_min, _y_max, _y_max, _y_min],
                     line_color="navy", line_alpha=0.3, line_dash="dotted", line_width=2)

        self.police_num, self.thief_num = team_size[
                                              'police'], 1000  # team_size['thief'] # TODO: must give a biggest num
        self.total_num = self.police_num + self.thief_num

        # draw balls
        self.rd_loc = plt_loc.circle(
            [-1] * self.total_num, [-1] * self.total_num,
            radius=[POLICE_RADIUS * map_size] * self.police_num + [
                                                                      THIEF_RADIUS * map_size] * self.thief_num,
            # radius is by percent
            # size=[50]*self.police_num + [20]*self.thief_num,  # size is px
            line_color="gold",
            line_width=[10] * self.police_num + [1] * self.thief_num,
            fill_color=["green"] * self.police_num + ["yellow"] * self.thief_num,
            fill_alpha=0.6)

        # 显示reward趋势
        plt_reward = figure(
            plot_width=400, plot_height=400, title="running ep reward: ")
        plt_reward.title.align = "center"
        plt_reward.title.text_color = "green"
        plt_reward.title.text_font_size = "20px"
        plt_reward.title.background_fill_color = "black"
        self.plt_reward = plt_reward  # for later update of title
        self.rd_reward = plt_reward.line(
            [1], [10], line_width=2)

        # put all the plots in a gridplot
        plt_combo = gridplot(
            [[plt_loc, plt_reward]],
            # toolbar_location=None
        )

        doc.add_root(column(plt_combo))
        self.doc = doc
        # doc.theme = Theme(json=yaml.load("""
        #         attrs:
        #             Figure:
        #                 background_fill_color: "#DDDDDD"
        #                 outline_line_color: white
        #                 toolbar_location: above
        #                 height: 500
        #                 width: 800
        #             Grid:
        #                 grid_line_dash: [6, 4]
        #                 grid_line_color: white
        #     """))

    def start_server(self):
        io_loop = IOLoop.current()
        bokeh_app = Application(FunctionHandler(self.modify_doc))
        server = Server({'/': bokeh_app}, io_loop=io_loop)
        server.start()

        print('Opening Bokeh application on http://localhost:5006/')
        io_loop.add_callback(server.show, "/")
        io_loop.start()

    def update_plots(self, env_state_action):
        """update bokeh plots according to new env state and action data"""
        self.doc.add_next_tick_callback(partial(self.update_bokeh_doc, env_state_action=env_state_action))

    def update_bokeh_doc(self, env_state_action):
        global_ob, rewards, ep_count, current_step, cur_action, current_is_caught, current_done = env_state_action
        self.police_num = len(global_ob['police'])
        self.thief_num = len(global_ob['thief'])

        self.plt_loc.title.text = "step: #{} action: {}".format(current_step, cur_action)

        # note： if update frequency too high， jupyter notebook will crash exausted
        all_x = [_loc[0] for _loc in global_ob['police']] + [_loc[0] for _loc in global_ob['thief']]
        all_y = [_loc[1] for _loc in global_ob['police']] + [_loc[1] for _loc in global_ob['thief']]
        self.rd_loc.data_source.data['x'] = all_x
        self.rd_loc.data_source.data['y'] = all_y

        # blink when game end
        thief_color = "red" if current_is_caught else "yellow"
        self.rd_loc.data_source.data['fill_color'] = ["green"] * self.police_num + [
                                                                                       thief_color] * self.thief_num
        thief_lw = 3 if current_is_caught else 1
        self.rd_loc.data_source.data['line_width'] = [10] * self.police_num + [
                                                                                  thief_lw] * self.thief_num

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

