import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

"""
Here's the place to register env
Note: Gym will defaultly use many wrappers to limit the env, like timestep_limit by TimeLimit
"""

register(
    id='MA-BALLS-1v1-static-v0',
    entry_point='gym_sandbox.envs:MABallsEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=10, adversary_action="static",
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='MA-BALLS-1v1-dynamic-v0',
    entry_point='gym_sandbox.envs:MABallsEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=30, adversary_action="simple",
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='MA-BALLS-1v1-grid-v0',
    entry_point='gym_sandbox.envs:MABallsEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=3, map_size=20, adversary_action="simple", state_format='grid'
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='MA-BALLS-1v1-grid-ravel-v0',
    entry_point='gym_sandbox.envs:MABallsEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=3, map_size=20, adversary_action="simple",
        state_format='grid',state_ravel=True,
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='MA-BALLS-1vn-killall-ravel-v0',
    entry_point='gym_sandbox.envs:Balls1vnEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=6, map_size=20, adversary_action="simple",
        state_format='grid',state_ravel=True,
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

# register(
#     id='police-1vn-random-killall-ravel-v0',
#     entry_point='gym_sandbox.envs:RandomBallsEnv',
#     timestep_limit=100,
#
#     kwargs = dict(
#         agent_num=1, agent_team="police",
#         init_thief_num=1, step_add_thief_max=3,
#         adversary_num=100, map_size=20, adversary_action="random",
#         state_format='grid',state_ravel=True,
#     )
#     # reward_threshold=1.0,
#     # nondeterministic = True,
# )
