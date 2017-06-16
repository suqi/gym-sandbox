import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MA-BALLS-v0',
    entry_point='gym_multiagent.envs:MABallsEnv',
    timestep_limit=1000,
    reward_threshold=1.0,
    nondeterministic = True,
)
