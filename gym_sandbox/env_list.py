"""
Here're all the env name you can use!

Just do:
>>> import gym_sandbox
>>> gym.make("xxx-v0")

Note: Gym will defaultly use many wrappers to limit the env, like timestep_limit by TimeLimit
"""

from gym.envs.registration import register

register(
    id='police-killone-static-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=10, adversary_action="static",
        state_format='cord_list', state_ravel=False
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='police-killone-dynamic-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=30, adversary_action="simple",
        state_format='cord_list',state_ravel=False
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='police-killone-grid-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=3, map_size=20, adversary_action="simple",
        state_format='grid',state_ravel=False
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='police-killone-ravel-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=3, map_size=20, adversary_action="simple",
        state_format='grid',state_ravel=True,
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='police-killall-ravel-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
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
#     entry_point='gym_sandbox.envs.police_kill_all_random:RandomBallsEnv',
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
