"""
Human interact with gym env
pip install pynput
"""

import time
import gym
import gym_sandbox
from pynput import keyboard
from pynput.keyboard import Key


def listen_to_mouse():
    from pynput import mouse

    def on_move(x, y):
        print('Pointer moved to {0}'.format(
            (x, y)))

    def on_click(x, y, button, pressed):
        print('{0} at {1}'.format(
            'Pressed' if pressed else 'Released',
            (x, y)))
        if not pressed:
            # Stop listener
            return False

    def on_scroll(x, y, dx, dy):
        print('Scrolled {0}'.format(
            (x, y)))

    # Collect events until released
    with mouse.Listener(
            on_move=on_move,
            on_click=on_click,
            on_scroll=on_scroll) as listener:
        listener.join()


def listen_to_keyboard():
    def on_press(key):
        try:
            print('alphanumeric key {0} pressed'.format(
                key.char))
        except AttributeError:
            print('special key {0} pressed'.format(
                key))

    def on_release(key):
        print('{0} released'.format(
            key))
        if key == keyboard.Key.esc:
            # Stop listener
            return False

    # Collect events until released
    with keyboard.Listener(
            on_press=on_press,
            on_release=on_release) as listener:
        listener.join()


def run(render=False):
    env = gym.make("police-killall-ravel-v0")
    env.env.init_params(show_dashboard=True, bokeh_output="standalone")
    env.reset()

    ACTION_KEYS = [Key.up, Key.down, Key.left, Key.right]

    def on_press(key):
        if key in ACTION_KEYS:
            s_, r, d, _ = env.step(ACTION_KEYS.index(key))
            if render: env.render()
            if d: env.reset()

    with keyboard.Listener(on_press=on_press) as listener:
        listener.join()


if __name__ == '__main__':
    # listen_to_keyboard()
    # listen_to_mouse()
    run(render=True)
