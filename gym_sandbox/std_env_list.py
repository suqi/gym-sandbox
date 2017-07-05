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


"""
Problem: A simplest linear problem, find shortest path of 2 cords.
Pros: Can be generalize to any map size.
      This is the baseline of any RL algorithm, if you can't solve, you should  
Cons: Too young too simple
"""
register(
    id='police-killone-static-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=10, adversary_action="static",
        state_format='cord_list_unfixed'
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

"""
Problem: A dynamic linear problem: find shortest path to a changing cords.
Cons: It's still just a linear problem. 
"""
register(
    id='police-killone-dynamic-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=1, map_size=30, adversary_action="simple",
        state_format='cord_list_unfixed'
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

"""
Problem: How to process dynamic input space (unknown npc number)
Pros: Proved that reshaped 3D-Grid is a nice solution for dynamic input
"""
register(
    id='police-killone-ravel-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=3, map_size=20, adversary_action="simple",
        state_format='grid3d_ravel',
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

"""
Problem: Can we use CNN on 3D-Grid?
Pros: Proved CNN works, and maybe it can work better than reshaped 3D-grid
"""
register(
    id='police-killone-grid-v0',
    entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=3, map_size=20, adversary_action="simple",
        state_format='grid3d'
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

"""
Problem: Another solution for dynamic input space. Use a list of 500 cords to fix the input. 
Cons: Works badly. A critical problem is, what cord should you fill up when a thief is killed.
      This fill-up problem will affect a lot on performance.
"""
register(
    id='police-killall-static-cords-500-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=10, map_size=10, adversary_action="static",
        state_format='cord_list_fixed_500'
    )
)

"""
Problem: This is a dynamic input version of cords-500. 
         It leave the work of fixing state size to algorithm. 
Cons: Same as cords-500
"""
register(
    id='police-killall-static-cords-unfixed-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=10, map_size=10, adversary_action="static",
        state_format='cord_list_unfixed'
    )
)

"""
Problem: Now police must kill all thieves. 
         The input will be changing more actively during an episode. 
Pros:    This is the first milestone of complex task 
"""
register(
    id='police-killall-ravel-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=6, map_size=20, adversary_action="simple",
        state_format='grid3d_ravel',
    )
    # reward_threshold=1.0,
    # nondeterministic = True,
)

"""
Problem: 3D shaped state, for CNN 
Pros:    Proved CNN works.
"""
register(
    id='police-killall-grid-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=6, map_size=10, adversary_action="simple",
        state_format='grid3d',
    )
)

"""
Problem: Now everything is random, can you hold on?
         Thief num is random, and randomly added into map, and action is random!
Pros:    Proved A3C can handle these randomness, just like it knows the game rule.
"""
register(
    id='police-killall-random-3dravel-v0',
    entry_point='gym_sandbox.envs.police_kill_all_random:RandomBallsEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police",
        init_thief_num=1, step_add_thief_max=3,
        adversary_num=50, map_size=15, adversary_action="random",
        state_format='grid3d_ravel'
    )
)

"""
Problem: Now action space has 2 type: walk or attack!
         This is a small milestone of multi-task.
Pros:    Prove that A3C can do different action in different state.
"""
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

"""
Problem: For CNN.
"""
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

"""
Problem: this is a base env for test of generalize(an idea from HRA paper), 
         which means to conquer any similar env(bigger map, more npc, action more random, etc.)
Cons:  Proved that generalize is a big problem for 3d grid state.
       1. After train with random thief, the police will stay around thief. 
          This trained model works badly when thief action is not random(even static!)
       2. When map size is bigger, trained model works badly.
       3. When catch dist is smaller, trained model fails.
"""
register(
    id='police-generalize-v0',
    entry_point='gym_sandbox.envs.police_base:PoliceKillAllEnv',
    timestep_limit=100,

    kwargs=dict(
        agent_num=1, agent_team="police", adversary_num=5, map_size=10, adversary_action="random",
        state_format='grid3d', police_speed=1, thief_speed=1, grid_scale=2, min_catch_dist=0
    )
)
