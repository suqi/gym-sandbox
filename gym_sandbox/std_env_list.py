"""
---- Don't change this file ----
Here're all the standard env you can use!
If you want to customize, please add your own in local_env_list.py file

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
        state_format='cord_list_unfixed'
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
        state_format='cord_list_unfixed'
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
        state_format='grid3d'
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
        state_format='grid3d_ravel',
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='police-killall-static-cords-500-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=10, map_size=10, adversary_action="static",
        state_format='cord_list_fixed_500'
    )
)

register(
    id='police-killall-static-cords-unfixed-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=10, map_size=10, adversary_action="static",
        state_format='cord_list_unfixed'
    )
)


register(
    id='police-killall-ravel-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=6, map_size=20, adversary_action="simple",
        state_format='grid3d_ravel',
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

register(
    id='police-killall-grid-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=6, map_size=10, adversary_action="simple",
        state_format='grid3d',
    )
)

register(
    id='police-killall-random-3dravel-v0',
    entry_point='gym_sandbox.envs.police_kill_all_random:RandomBallsEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police",
        init_thief_num=1, step_add_thief_max=3,
        adversary_num=50, map_size=15, adversary_action="random",
        state_format='grid3d_ravel'
    )
)

register(
    id='police-killall-trigger-3dravel-v0',
    entry_point='gym_sandbox.envs.police_trigger:PoliceTriggerEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=5, map_size=20,
        adversary_action="random",  # static/simple/random
        state_format='grid3d_ravel',
    )
)

register(
    id='police-killall-trigger-3dgrid-v0',
    entry_point='gym_sandbox.envs.police_trigger:PoliceTriggerEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=5, map_size=20,
        adversary_action="random",  # static/simple/random
        state_format='grid3d',
    )
)

register(
    id='police-generalize-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs = dict(
        agent_num=1, agent_team="police", adversary_num=5, map_size=10, adversary_action="dynamic",
        state_format='grid3d',police_speed=1, thief_speed=1, grid_scale=2, min_catch_dist=0
    )
)
