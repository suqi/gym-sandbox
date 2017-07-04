from easydict import EasyDict as edict
from gymgame.engine import Vector2, extension
from gymgame.tinyrpg import man
import numpy as np
config = man.config

Attr = config.Attr

GAME_NAME = config.GAME_NAME


# env constant
config.MAP_SIZE = Vector2(30, 30)

config.GRID_SIZE = config.MAP_SIZE

config.NUM_BULLET = 50

config.NUM_COIN = 50
config.BASE_PLAYER = edict(
    id = "player-{0}",
    position = Vector2(0, 0),
    direct = Vector2(0, 0),
    speed = 60.0,
    radius = 0.5,
    max_hp = 1,
)



@extension(man.Serializer)
class SerializerExtension():

    DIRECTS = [Vector2.up, Vector2.right, Vector2.down, Vector2.left]

    def _deserialize_action(self, data):
        direct = SerializerExtension.DIRECTS[data]
        actions = [('player-0', config.Action.move_toward, direct, None)]
        return actions

@extension(man.EnvironmentGym)
class EnvironmentExtension():
    def _reward(self):
        # todo something
        players = self.game.map.players
        hits = np.array([player.step_hits for player in players])
        coins = np.array([player.step_coins for player in players])
        if hits == 0:
            r = coins - hits + 0.1
        else:
            r = coins - hits
        return r[0] if len(r) == 1 else r


