"""
Here you can copy any env from std_env_list, and customize by your own!!
"""

from gym.envs.registration import register


# ------------------------- add your own diy env below --------------------------#

# example
# register(
#     id='police-killone-static-v0',
#     entry_point='gym_sandbox.envs.police_kill_one:PoliceKillOneEnv',
#     timestep_limit=100,
#
#     kwargs = dict(
#         agent_num=1, agent_team="police", adversary_num=1, map_size=10, adversary_action="static",
#         state_format='cord_list_unfixed'
#     )
# )
