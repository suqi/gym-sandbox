import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='MA-BALLS-1v1-v0',
    entry_point='gym_multiagent.envs:MABallsEnv',
    timestep_limit=1000,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=10
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='MA-BALLS-3v2-v0',
    entry_point='gym_multiagent.envs:MABallsEnv',
    timestep_limit=1000,

    kwargs = dict(
        agent_num=3, agent_team="police", adversary_num=2, map_size=100
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)
