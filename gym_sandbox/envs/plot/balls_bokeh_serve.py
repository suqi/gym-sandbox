# -*- coding: utf-8 -*-
import requests
from bokeh.client import push_session
from bokeh.plotting import figure, curdoc
from bokeh.layouts import gridplot

import subprocess
import time
from gym_sandbox.envs.plot.balls_game_dashboard import BallsNotebookRender


class BallsBokehServeRender(BallsNotebookRender):
    """A game dashboard to show the game state
    Firstly run bokeh serve in a separate process
    http://bokeh.pydata.org/en/latest/docs/user_guide/server.html#connecting-with-bokeh-client
    """

    def __init__(self, map_size, team_size):
        # start bokeh server if it doesn't exist
        # note: there should be only one process to update the session, otherwise chaos
        # TODO: this process will not be closed when main process quit, but it doesn't matter
        # must wait this serve process to run up then we can draw
        try:
            requests.get("http://localhost:5006")
        except Exception as e:
            subprocess.Popen(['bokeh', 'serve'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            print(">> waiting for bokeh server starting, 5 seconds...")
            time.sleep(5)
            print(">> here we go!")

        self.draw_init_plot(map_size, team_size)

        # curdoc().add_root(plt_combo)
        # open a session to keep our local document in sync with server
        session = push_session(curdoc())
        session.show(self.plt_combo)  # open the document in a browser
        # session.loop_until_closed()  # run forever

    # def update_plots(self, env_state_action):
    #     """update bokeh plots according to new env state and action data"""
    #     curdoc().add_next_tick_callback(partial(self.update_bokeh_doc, env_state_action=env_state_action))

    def update_plots(self, env_state_action):
        """update bokeh plots according to new env state and action data"""
        self.draw_new_plot(env_state_action)
